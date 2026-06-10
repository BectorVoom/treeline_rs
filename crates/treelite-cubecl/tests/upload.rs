//! Wave 2 per-column ragged-SoA upload round-trip (plan 06-03) — GREEN.
//!
//! Turns the RED Wave-0 `upload.rs` scaffold green. Asserts the host→device
//! upload contract SC3/GPU-05 demands:
//! - per-column concatenation across the forest into ONE host `Vec` per column;
//! - ONE device handle per column for the WHOLE forest (no per-tree handle
//!   explosion) via `client.create_from_slice` (the Wave-1-confirmed entry);
//! - a `tree_node_offset` prefix sum `[0, n0, n0+n1, …]` so
//!   `concat[tree_node_offset[t] + n]` addresses tree `t`'s node `n`;
//! - bool (`default_left`) materialized to a `u32` 0/1 column and enum
//!   (`node_type`) to an `i32` discriminant column (Pitfall 4);
//! - a byte-exact device round-trip: upload each column, `client.read` it back
//!   via `bytemuck::cast_slice`, assert it equals the host concatenation;
//! - host-side validation (bad `split_index`, buffer-length mismatch) returns a
//!   typed `CubeclError` BEFORE any `client.create` (T-06-06 — no OOB device
//!   write).

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;

use treelite_core::{ModelPreset, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::error::CubeclError;
use treelite_cubecl::upload::{concat_columns, upload_forest, validate_shape};

/// Single-split numerical tree on `feature` (mirrors `tests/spike.rs::split_tree`):
/// node 0 numerical `kLT` test, default-left; node 1 = leaf; node 2 = leaf.
fn split_tree<T: Copy + Default>(feature: i32, threshold: T, left_leaf: T, right_leaf: T) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, T::default(), T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default(), left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t
}

/// Read a device handle back as a typed `Vec` via `bytemuck::cast_slice`.
fn read_back<T: bytemuck::Pod>(
    client: &cubecl::client::ComputeClient<CpuRuntime>,
    handle: cubecl::server::Handle,
) -> Vec<T> {
    let bytes = client.read_one_unchecked(handle);
    bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
}

#[test]
fn upload_ragged_soa_roundtrip() {
    let client = CpuRuntime::client(&Default::default());

    // A 3-tree forest on features {0, 1, 0}. Each tree has 3 nodes, so the
    // concatenated node columns are length 9 and the prefix sum is [0,3,6,9].
    let trees = vec![
        split_tree::<f32>(0, 0.5, 1.0, -1.0),
        split_tree::<f32>(1, 1.5, 2.0, -3.0),
        split_tree::<f32>(0, -0.25, 4.0, 5.0),
    ];
    let preset = ModelPreset::new(trees);

    // ---- Host concatenation matches the expected ragged-SoA layout ----
    let cols = concat_columns(&preset);
    assert_eq!(cols.tree_node_offset, vec![0, 3, 6, 9], "prefix sum over num_nodes");
    assert_eq!(cols.cleft.len(), 9, "3 trees x 3 nodes = 9 node-column elements");
    // tree t node n is concat[tree_node_offset[t] + n]:
    //   tree 1 node 0 split_index == feature 1.
    let t1n0 = cols.tree_node_offset[1] as usize;
    assert_eq!(cols.split_index[t1n0], 1, "tree 1 node 0 addresses feature 1");
    //   tree 2 node 1 leaf_value == 4.0 (left leaf of tree 2).
    let t2 = cols.tree_node_offset[2] as usize;
    assert_eq!(cols.leaf_value[t2 + 1], 4.0, "tree 2 node 1 left leaf");
    // bool -> u32 0/1 (default_left = [true,false,false] per tree).
    assert_eq!(cols.default_left, vec![1, 0, 0, 1, 0, 0, 1, 0, 0]);
    // enum -> i32 discriminant (kNumericalTestNode=1, kLeafNode=0).
    assert_eq!(cols.node_type, vec![1, 0, 0, 1, 0, 0, 1, 0, 0]);

    // ---- Upload: ONE handle per column, validation passes (data is well-formed) ----
    // num_feature = 2, num_row = 4 rows -> data_len must be >= 8.
    let num_feature = 2i32;
    let num_row = 4usize;
    let data_len = num_row * num_feature as usize;
    let up = upload_forest::<CpuRuntime, f32>(&client, &preset, num_feature, num_row, data_len)
        .expect("well-formed forest uploads");

    assert_eq!(up.num_nodes_total, 9);
    assert_eq!(up.tree_node_offset, vec![0, 3, 6, 9]);

    // ---- Device round-trip: each column reads back byte-exact ----
    assert_eq!(read_back::<i32>(&client, up.cleft), cols.cleft);
    assert_eq!(read_back::<i32>(&client, up.cright), cols.cright);
    assert_eq!(read_back::<i32>(&client, up.split_index), cols.split_index);
    assert_eq!(read_back::<f32>(&client, up.threshold), cols.threshold);
    assert_eq!(read_back::<f32>(&client, up.leaf_value), cols.leaf_value);
    assert_eq!(read_back::<u32>(&client, up.default_left), cols.default_left);
    assert_eq!(read_back::<i32>(&client, up.node_type), cols.node_type);
}

#[test]
fn upload_rejects_bad_split_index_before_device_op() {
    // A tree whose internal node references feature index 9, but the model only
    // has num_feature = 2. Validation must reject BEFORE any client.create.
    let mut bad = split_tree::<f32>(9, 0.5, 1.0, -1.0); // split_index[0] = 9
    bad.num_nodes = 3;
    let preset = ModelPreset::new(vec![bad]);

    let cols = concat_columns(&preset);
    let err = validate_shape(2, 1, 2, &cols).expect_err("split_index 9 > num_feature 2 rejected");
    match err {
        CubeclError::FeatureIndexOutOfBounds { node, feature, num_feature } => {
            assert_eq!(node, 0);
            assert_eq!(feature, 9);
            assert_eq!(num_feature, 2);
        }
        other => panic!("expected FeatureIndexOutOfBounds, got {other:?}"),
    }

    // upload_forest surfaces the same typed error (no device op). `UploadedForest`
    // holds device `Handle`s (not `Debug`), so match the Result directly rather
    // than via `expect_err` (which needs `T: Debug`).
    let client = CpuRuntime::client(&Default::default());
    match upload_forest::<CpuRuntime, f32>(&client, &preset, 2, 1, 2) {
        Err(CubeclError::FeatureIndexOutOfBounds { .. }) => {}
        Err(other) => panic!("expected FeatureIndexOutOfBounds, got {other:?}"),
        Ok(_) => panic!("upload must reject the malformed model before any client.create"),
    }
}

#[test]
fn upload_rejects_buffer_length_mismatch_before_device_op() {
    let preset = ModelPreset::new(vec![split_tree::<f32>(0, 0.5, 1.0, -1.0)]);
    let cols = concat_columns(&preset);
    // num_feature = 2, num_row = 4 requires 8 data elements; supply only 5.
    let err = validate_shape(2, 4, 5, &cols).expect_err("short data buffer rejected");
    match err {
        CubeclError::InvalidInputShape { num_row, num_feature, required, got } => {
            assert_eq!(num_row, 4);
            assert_eq!(num_feature, 2);
            assert_eq!(required, 8);
            assert_eq!(got, 5);
        }
        other => panic!("expected InvalidInputShape, got {other:?}"),
    }

    // A negative num_feature is the impossible-shape guard (mirror predict WR-02).
    let err_neg = validate_shape(-1, 1, 0, &cols).expect_err("negative num_feature rejected");
    assert!(matches!(err_neg, CubeclError::InvalidInputShape { .. }));
}

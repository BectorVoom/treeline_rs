//! v5 serialize/deserialize round-trip + hostile-input rejection (SER-01, D-03,
//! ASVS V5).
//!
//! Two correctness gates:
//!   1. **Round-trip identity:** `serialize → deserialize → serialize` yields
//!      byte-identical output, AND deserializing the frozen `golden_v5.bin` then
//!      re-serializing equals the blob. The comparison is at the BYTE level so
//!      NaN/inf float columns round-trip bit-exact (Pitfall 4 — never float-`==`).
//!   2. **Panic-free hostile input:** a `major_ver == 3` header, a truncated
//!      blob, and an oversized array-count prefix each return a SPECIFIC typed
//!      error (never a panic, over-read, or huge allocation).

use std::path::Path;

use treelite_core::serialize::error::SerializeError;
use treelite_core::{
    Model, ModelPreset, ModelVariant, Operator, TaskType, Tree, TreeBuf, TreeNodeType, deserialize,
    serialize_to_buffer,
};

fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// Build a small but FULLY-populated `<f32>` model exercising every column type
/// (enums, bools, i32/u32/u64/f32/f64 arrays, present-flags, a NaN leaf value,
/// a non-empty `attributes` string) so the round-trip covers the whole walk.
fn sample_model() -> Model {
    let mut tree = Tree::<f32>::new();
    // 3 nodes: root (numerical test) + two leaves.
    tree.num_nodes = 3;
    tree.has_categorical_split = false;
    tree.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    tree.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    tree.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    tree.split_index = TreeBuf::from_owned(vec![0, -1, -1]);
    tree.default_left = TreeBuf::from_owned(vec![true, false, false]);
    // A NaN leaf value proves bit-exact float round-trip (Pitfall 4).
    tree.leaf_value = TreeBuf::from_owned(vec![0.0, f32::NAN, -1.5]);
    tree.threshold = TreeBuf::from_owned(vec![0.5, 0.0, 0.0]);
    tree.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    tree.category_list_right_child = TreeBuf::from_owned(vec![false, false, false]);
    tree.leaf_vector_begin = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.leaf_vector_end = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.category_list_begin = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.category_list_end = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.sum_hess = TreeBuf::from_owned(vec![6.0f64, 3.0, 3.0]);
    tree.sum_hess_present = TreeBuf::from_owned(vec![true, true, true]);
    tree.gain = TreeBuf::from_owned(vec![10.0f64, 0.0, 0.0]);
    tree.gain_present = TreeBuf::from_owned(vec![true, false, false]);

    let mut model = Model::new(ModelVariant::F32(ModelPreset::new(vec![tree])));
    model.num_feature = 2;
    model.task_type = TaskType::kBinaryClf;
    model.num_target = 1;
    model.num_class = vec![1].into();
    model.leaf_vector_shape = vec![1, 1].into();
    model.target_id = vec![0].into();
    model.class_id = vec![0].into();
    model.postprocessor = "sigmoid".into();
    model.base_scores = vec![-1.0986122886681098].into();
    model.attributes = "{}".into();
    model
}

#[test]
fn round_trip_is_byte_identical() {
    let mut model = sample_model();
    let bytes1 = serialize_to_buffer(&mut model);

    let mut decoded = deserialize(&bytes1).expect("deserialize must succeed");
    let bytes2 = serialize_to_buffer(&mut decoded);

    // BYTE compare (NaN-safe): the NaN leaf value round-trips bit-exact.
    assert_eq!(
        bytes1, bytes2,
        "serialize→deserialize→serialize must be byte-identical (SER-01)"
    );
}

#[test]
fn golden_v5_round_trips_to_itself() {
    let blob = std::fs::read(fixture_path("golden_v5.bin")).expect("read golden_v5.bin");
    let mut model = deserialize(&blob).expect("deserialize golden_v5.bin");
    let re = serialize_to_buffer(&mut model);
    assert_eq!(
        re, blob,
        "re-serializing the golden blob must equal it (D-02)"
    );
}

#[test]
fn rejects_v3_major_version_without_panicking() {
    // Start from a valid v5 blob and overwrite major_ver (first LE i32) with 3.
    let mut blob = {
        let mut m = sample_model();
        serialize_to_buffer(&mut m)
    };
    blob[0..4].copy_from_slice(&3i32.to_le_bytes());

    // `Model` is intentionally not `Debug`, so match on the `Result` directly
    // rather than `expect_err`. No panic, no over-read.
    match deserialize(&blob) {
        Err(SerializeError::UnsupportedVersion { major, .. }) => assert_eq!(major, 3),
        Err(other) => panic!("expected UnsupportedVersion, got {other:?}"),
        Ok(_) => panic!("major_ver=3 must be rejected (D-03)"),
    }
}

#[test]
fn rejects_truncated_stream_without_panicking() {
    let full = {
        let mut m = sample_model();
        serialize_to_buffer(&mut m)
    };
    // Keep only the first 30 bytes — well inside the header.
    let truncated = &full[..30];
    match deserialize(truncated) {
        Err(SerializeError::TruncatedStream { .. }) => {}
        Err(other) => panic!("expected TruncatedStream, got {other:?}"),
        Ok(_) => panic!("truncated blob must be rejected"),
    }
}

#[test]
fn rejects_oversized_array_count_without_allocating() {
    // A valid header, but corrupt the FIRST array (`num_class`) count prefix to a
    // huge u64 so the deserializer must reject it via the count-vs-remaining
    // bound BEFORE attempting any allocation.
    let mut blob = {
        let mut m = sample_model();
        serialize_to_buffer(&mut m)
    };
    // Header layout: 4+4+4 (ver) +1+1 (tags) +8 (num_tree) +4 (num_feature)
    // +1 (task) +1 (avg) +4 (num_target) = 37, then num_class's u64 count.
    let num_class_count_off = 4 + 4 + 4 + 1 + 1 + 8 + 4 + 1 + 1 + 4;
    blob[num_class_count_off..num_class_count_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());

    match deserialize(&blob) {
        Err(SerializeError::CountExceedsBuffer { .. }) => {}
        Err(other) => panic!("expected CountExceedsBuffer, got {other:?}"),
        Ok(_) => panic!("oversized count must be rejected"),
    }
}

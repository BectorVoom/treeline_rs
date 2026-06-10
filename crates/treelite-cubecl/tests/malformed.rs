//! CR-03 lock — a malformed leaf-vector span is rejected with a typed
//! [`CubeclError::MalformedLeafVector`] HOST-side, BEFORE any device op, instead
//! of an out-of-bounds leaf-vector device read (T-06-09).
//!
//! Two checks:
//! - the end-to-end `predict_cpu` path returns the typed error for a Model whose
//!   leaf node's `leaf_vector_end` exceeds its tree's leaf-vector segment length;
//! - `validate_leaf_vectors` driven directly on hand-built `HostColumns` returns
//!   the same typed error (an inverted span and an out-of-range end), and a
//!   well-formed column set passes.

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::upload::{HostColumns, validate_leaf_vectors};
use treelite_cubecl::{CubeclError, predict_cpu};
use treelite_gtil::{Config, PredictKind};

/// A single-node leaf-VECTOR tree (id 0) whose `leaf_vector` segment has length
/// `seg_len`, but whose `leaf_vector_end` is set to `bad_end` (> `seg_len`) — a
/// malformed span the kernel's broadcast loop would read past.
fn malformed_leaf_vector_tree<T: Copy + Default>(seg_len: usize, bad_end: u64) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default()]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    // The actual leaf-vector segment is `seg_len` long...
    t.leaf_vector = TreeBuf::from_owned(vec![T::default(); seg_len]);
    // ...but the node claims [0, bad_end) — a span past the segment.
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![bad_end]);
    t
}

/// A `(num_row, 1, K)` multiclass leaf-vector-broadcast model over `trees`.
fn multiclass_model<T, W>(trees: Vec<Tree<T>>, wrap: W, k: i32) -> Model
where
    T: Copy,
    W: Fn(ModelPreset<T>) -> ModelVariant,
{
    let num_tree = trees.len();
    let mut m = Model::new(wrap(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![k];
    m.leaf_vector_shape = vec![1, k];
    m.target_id = vec![-1; num_tree];
    m.class_id = vec![-1; num_tree];
    m.postprocessor = "identity".to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![0.0; k as usize];
    m
}

#[test]
fn predict_cpu_rejects_short_leaf_vector_with_typed_error() {
    // Segment length 2, but the leaf node claims leaf_vector_end = 5 → the
    // default_raw broadcast loop (reading up to num_target * max_num_class cells
    // from begin) would overrun. predict_cpu must return MalformedLeafVector
    // BEFORE any device op, never an OOB device read.
    let tree = malformed_leaf_vector_tree::<f32>(2, 5);
    let model = multiclass_model(vec![tree], ModelVariant::F32, 3);
    let data: Vec<f32> = vec![0.0]; // 1 row, 1 feature

    let err = predict_cpu::<f32>(&model, &data, 1, &Config { kind: PredictKind::Default, nthread: 0 })
        .expect_err("a leaf-vector span past the segment must error, not OOB-read");
    match err {
        CubeclError::MalformedLeafVector { tree, node, .. } => {
            assert_eq!(tree, 0, "offending tree id");
            assert_eq!(node, 0, "offending node id");
        }
        other => panic!("expected MalformedLeafVector, got {other:?}"),
    }
}

#[test]
fn validate_leaf_vectors_rejects_end_past_segment() {
    // Hand-built host columns: one tree, one leaf node, segment length 2, end = 5.
    let cols = HostColumns::<f32> {
        cleft: vec![-1],
        cright: vec![-1],
        split_index: vec![-1],
        threshold: vec![0.0],
        leaf_value: vec![0.0],
        leaf_vector: vec![0.0, 0.0], // segment length 2
        leaf_vector_begin: vec![0],
        leaf_vector_end: vec![5], // past the segment
        default_left: vec![0],
        node_type: vec![0],
        tree_node_offset: vec![0, 1],
        tree_leafvec_offset: vec![0, 2],
    };
    match validate_leaf_vectors(&cols, 1, 1) {
        Err(CubeclError::MalformedLeafVector { tree, node, begin, end, segment_len }) => {
            assert_eq!((tree, node), (0, 0));
            assert_eq!((begin, end, segment_len), (0, 5, 2));
        }
        other => panic!("expected MalformedLeafVector, got {other:?}"),
    }
}

#[test]
fn validate_leaf_vectors_rejects_inverted_span() {
    // begin > end (an inverted span) is also malformed.
    let cols = HostColumns::<f32> {
        cleft: vec![-1],
        cright: vec![-1],
        split_index: vec![-1],
        threshold: vec![0.0],
        leaf_value: vec![0.0],
        leaf_vector: vec![0.0, 0.0, 0.0],
        leaf_vector_begin: vec![2],
        leaf_vector_end: vec![1], // begin > end
        default_left: vec![0],
        node_type: vec![0],
        tree_node_offset: vec![0, 1],
        tree_leafvec_offset: vec![0, 3],
    };
    match validate_leaf_vectors(&cols, 1, 1) {
        Err(CubeclError::MalformedLeafVector { begin, end, .. }) => {
            assert_eq!((begin, end), (2, 1));
        }
        other => panic!("expected MalformedLeafVector, got {other:?}"),
    }
}

#[test]
fn validate_leaf_vectors_rejects_broadcast_overrun() {
    // begin = 0, end = 2 (within the segment), segment length 2 — but the
    // multiclass broadcast reads num_target * max_num_class = 3 cells from begin,
    // overrunning the length-2 segment. The broadcast-span check must fire.
    let cols = HostColumns::<f32> {
        cleft: vec![-1],
        cright: vec![-1],
        split_index: vec![-1],
        threshold: vec![0.0],
        leaf_value: vec![0.0],
        leaf_vector: vec![0.0, 0.0],
        leaf_vector_begin: vec![0],
        leaf_vector_end: vec![2],
        default_left: vec![0],
        node_type: vec![0],
        tree_node_offset: vec![0, 1],
        tree_leafvec_offset: vec![0, 2],
    };
    // broadcast span 1 * 3 = 3 > segment length 2 → reject.
    match validate_leaf_vectors(&cols, 1, 3) {
        Err(CubeclError::MalformedLeafVector { end, segment_len, .. }) => {
            assert_eq!((end, segment_len), (3, 2)); // broadcast_end = begin + 3
        }
        other => panic!("expected MalformedLeafVector (broadcast overrun), got {other:?}"),
    }
}

#[test]
fn validate_leaf_vectors_accepts_well_formed() {
    // A well-formed leaf-vector leaf (begin 0, end 3, segment 3) with a matching
    // broadcast span (1 * 3 = 3) passes — no false positive on real models.
    let cols = HostColumns::<f32> {
        cleft: vec![-1],
        cright: vec![-1],
        split_index: vec![-1],
        threshold: vec![0.0],
        leaf_value: vec![0.0],
        leaf_vector: vec![1.0, 2.0, 3.0],
        leaf_vector_begin: vec![0],
        leaf_vector_end: vec![3],
        default_left: vec![0],
        node_type: vec![0],
        tree_node_offset: vec![0, 1],
        tree_leafvec_offset: vec![0, 3],
    };
    assert!(validate_leaf_vectors(&cols, 1, 3).is_ok());

    // A scalar leaf (begin == end == 0) is always fine, regardless of broadcast.
    let scalar = HostColumns::<f32> {
        cleft: vec![1, -1, -1],
        cright: vec![2, -1, -1],
        split_index: vec![0, -1, -1],
        threshold: vec![0.5, 0.0, 0.0],
        leaf_value: vec![0.0, 1.0, -1.0],
        leaf_vector: vec![],
        leaf_vector_begin: vec![0, 0, 0],
        leaf_vector_end: vec![0, 0, 0],
        default_left: vec![1, 0, 0],
        node_type: vec![1, 0, 0],
        tree_node_offset: vec![0, 3],
        tree_leafvec_offset: vec![0, 0],
    };
    assert!(validate_leaf_vectors(&scalar, 1, 1).is_ok());
}

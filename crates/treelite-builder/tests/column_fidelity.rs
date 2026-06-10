//! Builder column-fidelity regression tests (CR-01 + CR-02, gap-closure 02-06).
//!
//! These assert that a `ModelBuilder`-committed `Tree<f32>` satisfies the two
//! upstream column-length invariants from `AllocNode`
//! (`treelite-mainline/include/treelite/detail/tree.h:70-101`), which the v5
//! byte image depends on for byte-for-byte fidelity (SER-01):
//!
//! - CR-01 (AllocNode per-node columns): `category_list_right_child`,
//!   `leaf_vector_begin/end`, `category_list_begin/end` are pushed on EVERY node
//!   (length num_nodes), defaulting to `false` / `0` for no-category,
//!   no-leaf-vector trees.
//! - CR-02 (empty-unless-set stats): the `data_count`/`sum_hess`/`gain` columns
//!   (and their `_present` companions) stay empty (length 0) unless at least one
//!   node set that specific stat, in which case that one column (and only it)
//!   becomes length num_nodes — per-column independence.

use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::ModelVariant;
use treelite_core::enums::{Operator, TaskType};

/// Metadata for a single-target, scalar-leaf, `expected_num_tree`-tree model
/// (mirrors `validation.rs::meta`).
fn meta(num_feature: i32, expected_num_tree: usize) -> BuilderMetadata {
    BuilderMetadata {
        num_feature,
        task_type: TaskType::kRegressor,
        average_tree_output: false,
        num_target: 1,
        num_class: vec![1],
        leaf_vector_shape: vec![1, 1],
        target_id: vec![0; expected_num_tree],
        class_id: vec![0; expected_num_tree],
        postprocessor: "identity".to_string(),
        base_scores: vec![0.0],
        attributes: None,
    }
}

/// Test A: no stat setters → all six stat columns empty (length 0), while the
/// five AllocNode per-node columns are length num_nodes with default values.
#[test]
fn builder_empty_unless_set_and_allocnode_lengths() {
    let num_nodes = 3usize;

    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    // Root: numerical test with two scalar-leaf children. No data_count/sum_hess/gain.
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5, true, Operator::kLT, 1, 2).unwrap();
    b.end_node().unwrap();
    b.start_node(1).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    b.start_node(2).unwrap();
    b.leaf_scalar(2.0).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    let model = b.commit_model().unwrap();

    let ModelVariant::F32(preset) = &model.variant else {
        panic!("expected F32 preset");
    };
    let tree = &preset.trees[0];
    assert_eq!(tree.num_nodes as usize, num_nodes);

    // CR-02: empty-unless-set — no stat was set, so all six stat columns len 0.
    assert_eq!(tree.data_count_present.len(), 0);
    assert_eq!(tree.sum_hess_present.len(), 0);
    assert_eq!(tree.gain_present.len(), 0);
    assert_eq!(tree.data_count.len(), 0);
    assert_eq!(tree.sum_hess.len(), 0);
    assert_eq!(tree.gain.len(), 0);

    // CR-01: AllocNode per-node columns are length num_nodes (== 3).
    assert_eq!(tree.category_list_right_child.len(), num_nodes);
    assert_eq!(tree.leaf_vector_begin.len(), num_nodes);
    assert_eq!(tree.leaf_vector_end.len(), num_nodes);
    assert_eq!(tree.category_list_begin.len(), num_nodes);
    assert_eq!(tree.category_list_end.len(), num_nodes);

    // AllocNode default values: category_list_right_child == false everywhere,
    // every begin/end offset == 0 (leaf_vector_/category_list_ value buffers empty).
    for nid in 0..num_nodes {
        assert!(!tree.category_list_right_child.as_slice()[nid]);
        assert_eq!(tree.leaf_vector_begin.as_slice()[nid], 0);
        assert_eq!(tree.leaf_vector_end.as_slice()[nid], 0);
        assert_eq!(tree.category_list_begin.as_slice()[nid], 0);
        assert_eq!(tree.category_list_end.as_slice()[nid], 0);
    }
}

/// Test B: setting only `sum_hess` on one node emits the `sum_hess`/`sum_hess_present`
/// columns at length num_nodes, while `data_count`/`gain` stay empty (per-column
/// independence).
#[test]
fn builder_stat_column_emitted_when_set() {
    let num_nodes = 3usize;

    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5, true, Operator::kLT, 1, 2).unwrap();
    b.sum_hess(42.0).unwrap(); // only sum_hess, on one node
    b.end_node().unwrap();
    b.start_node(1).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    b.start_node(2).unwrap();
    b.leaf_scalar(2.0).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    let model = b.commit_model().unwrap();

    let ModelVariant::F32(preset) = &model.variant else {
        panic!("expected F32 preset");
    };
    let tree = &preset.trees[0];

    // sum_hess was set → its column (and present) is length num_nodes.
    assert_eq!(tree.sum_hess.len(), num_nodes);
    assert_eq!(tree.sum_hess_present.len(), num_nodes);
    // The set node carries the value + present flag.
    assert_eq!(tree.sum_hess.as_slice()[0], 42.0);
    assert!(tree.sum_hess_present.as_slice()[0]);

    // Per-column independence: data_count and gain were never set → stay empty.
    assert_eq!(tree.data_count_present.len(), 0);
    assert_eq!(tree.gain_present.len(), 0);
    assert_eq!(tree.data_count.len(), 0);
    assert_eq!(tree.gain.len(), 0);
}

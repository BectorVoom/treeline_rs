//! Validation tests for the fluent `ModelBuilder` state machine (BLD-01, D-07/D-08).
//!
//! One `#[test]` per `<behavior>` bullet from `02-02-PLAN.md` Task 1, asserting
//! the specific `BuilderError` variant (proving the locality fields carry the
//! offending key). The forward-reference acceptance case proves child-key
//! resolution happens at `end_tree`, not `numerical_test` (RESEARCH Pitfall 6).

use treelite_builder::error::BuilderError;
use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::ModelVariant;
use treelite_core::enums::{Operator, TaskType};

/// Metadata for an `expected_num_tree`-tree, single-target, scalar-leaf model
/// with `num_feature` features.
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

#[test]
fn negative_node_key_rejected() {
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    let err = b.start_node(-1).unwrap_err();
    assert!(matches!(err, BuilderError::NegativeNodeKey { key: -1 }));
}

#[test]
fn duplicate_node_key_rejected() {
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    let err = b.start_node(0).unwrap_err();
    assert!(matches!(err, BuilderError::DuplicateNodeKey { key: 0 }));
}

#[test]
fn leaf_plus_test_conflict_rejected() {
    // After leaf_scalar the node is NodeComplete; a subsequent numerical_test is
    // an illegal transition (WrongState) — the state machine enforces the
    // leaf-vs-test mutual exclusivity.
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.leaf_scalar(1.0).unwrap();
    let err = b
        .numerical_test(0, 0.5, true, Operator::kLT, 1, 2)
        .unwrap_err();
    assert!(matches!(err, BuilderError::WrongState { .. }));
}

#[test]
fn equal_child_keys_rejected() {
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    let err = b
        .numerical_test(0, 0.5, true, Operator::kLT, 1, 1)
        .unwrap_err();
    assert!(matches!(err, BuilderError::SelfOrEqualChildKey { node: 0 }));
}

#[test]
fn self_child_key_rejected() {
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    let err = b
        .numerical_test(0, 0.5, true, Operator::kLT, 0, 2)
        .unwrap_err();
    assert!(matches!(err, BuilderError::SelfOrEqualChildKey { node: 0 }));
}

#[test]
fn dangling_child_key_rejected() {
    // Root references child key 5 which is never declared → DanglingChildKey at end_tree.
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5, true, Operator::kLT, 1, 5).unwrap();
    b.end_node().unwrap();
    b.start_node(1).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    let err = b.end_tree().unwrap_err();
    assert!(matches!(err, BuilderError::DanglingChildKey { key: 5 }));
}

#[test]
fn orphaned_node_rejected() {
    // Node 2 is declared but never referenced by any node → unreachable from root.
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5, true, Operator::kLT, 1, 3).unwrap();
    b.end_node().unwrap();
    b.start_node(1).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    b.start_node(3).unwrap();
    b.leaf_scalar(2.0).unwrap();
    b.end_node().unwrap();
    // Node 2 is orphaned: declared, never referenced.
    b.start_node(2).unwrap();
    b.leaf_scalar(3.0).unwrap();
    b.end_node().unwrap();
    let err = b.end_tree().unwrap_err();
    assert!(matches!(err, BuilderError::OrphanedNode { key: 2 }));
}

#[test]
fn forward_reference_accepted() {
    // Declare the ROOT (with children keyed 1, 2) BEFORE declaring nodes 1 and 2.
    // Resolution happens at end_tree, NOT at numerical_test (RESEARCH Pitfall 6).
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5, true, Operator::kLT, 1, 2).unwrap();
    b.end_node().unwrap();
    // Children declared AFTER the parent that references them.
    b.start_node(1).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    b.start_node(2).unwrap();
    b.leaf_scalar(2.0).unwrap();
    b.end_node().unwrap();
    // Forward reference must resolve cleanly.
    assert!(b.end_tree().is_ok());
    let model = b.commit_model().unwrap();
    match model.variant {
        ModelVariant::F32(p) => assert_eq!(p.num_trees(), 1),
        _ => panic!("expected F32 variant"),
    }
}

#[test]
fn commit_tree_count_mismatch_rejected() {
    // Metadata expects 2 trees; build only 1.
    let mut b = ModelBuilder::new(meta(4, 2)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.leaf_scalar(1.0).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    // `Model` is not `Debug`, so match instead of `unwrap_err`.
    match b.commit_model() {
        Err(BuilderError::CommitTreeCountMismatch {
            expected: 2,
            got: 1,
        }) => {}
        Err(other) => panic!("wrong error variant: {other:?}"),
        Ok(_) => panic!("expected CommitTreeCountMismatch"),
    }
}

#[test]
fn split_index_out_of_range_rejected() {
    // num_feature = 4; split_index 4 is out of range.
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    let err = b
        .numerical_test(4, 0.5, true, Operator::kLT, 1, 2)
        .unwrap_err();
    assert!(matches!(
        err,
        BuilderError::SplitIndexOutOfRange {
            split_index: 4,
            num_feature: 4
        }
    ));
}

#[test]
fn single_leaf_tree_commits() {
    // The minimal valid model: one tree, one leaf node.
    let mut b = ModelBuilder::new(meta(4, 1)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.leaf_scalar(42.0).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    let model = b.commit_model().unwrap();
    assert_eq!(model.num_feature, 4);
    match model.variant {
        ModelVariant::F32(p) => {
            assert_eq!(p.num_trees(), 1);
            assert!(p.trees[0].is_leaf(0));
            assert_eq!(p.trees[0].leaf_value(0), 42.0);
        }
        _ => panic!("expected F32 variant"),
    }
}

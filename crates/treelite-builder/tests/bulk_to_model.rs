//! Tests for `bulk_to_model` (Plan 04-01 Task 2, ported from `sklearn_bulk.cc`).
//!
//! The bulk path bypasses the per-node `ModelBuilder` (D-09) and assembles
//! pre-validated `Tree<f64>` outputs directly into a `ModelVariant::F64` Model,
//! hand-setting every header field (the metadata `sklearn_bulk.cc:244-330` sets).
//! These tests assert the variant, tree count, and all metadata fields land on
//! the assembled model — including `average_tree_output=true`, which RF needs for
//! the downstream averaging gate (Pitfall 6).

use treelite_builder::{BuilderMetadata, bulk_construct_tree, bulk_to_model};
use treelite_core::ModelVariant;
use treelite_core::enums::TaskType;

/// Build one small bulk tree (an internal split with two scalar leaves).
fn small_bulk_tree() -> treelite_core::Tree<f64> {
    bulk_construct_tree(
        3,
        &[1_i64, -1, -1],
        &[2_i64, -1, -1],
        &[1_i64, -1, -1],
        &[0.5_f64, 0.0, 0.0],
        &[0.0_f64, 10.0, 20.0],
        &[100_i64, 60, 40],
        &[100.0_f64, 60.0, 40.0],
        &[0.5_f64, 0.1, 0.2],
        100,
        1,
        1,
        false,
    )
}

/// Metadata describing a 2-tree RF regressor.
fn rf_meta() -> BuilderMetadata {
    BuilderMetadata {
        num_feature: 4,
        task_type: TaskType::kRegressor,
        average_tree_output: true, // RF averages
        num_target: 1,
        num_class: vec![1],
        leaf_vector_shape: vec![1, 1],
        target_id: vec![0, 0],
        class_id: vec![0, 0],
        postprocessor: "identity".to_string(),
        base_scores: vec![0.0],
        attributes: None,
    }
}

#[test]
fn bulk_to_model_assembles_f64_model_with_all_metadata() {
    let trees = vec![small_bulk_tree(), small_bulk_tree()];
    let meta = rf_meta();
    let model = bulk_to_model(trees, meta);

    // Variant is F64 with exactly the 2 input trees.
    match &model.variant {
        ModelVariant::F64(p) => assert_eq!(p.num_trees(), 2),
        ModelVariant::F32(_) => panic!("expected ModelVariant::F64, got F32"),
    }

    // All 10 metadata fields equal the passed values.
    assert_eq!(model.num_feature, 4);
    assert_eq!(model.task_type, TaskType::kRegressor);
    assert_eq!(model.num_class, vec![1]);
    assert_eq!(model.leaf_vector_shape, vec![1, 1]);
    assert_eq!(model.target_id, vec![0, 0]);
    assert_eq!(model.class_id, vec![0, 0]);
    assert_eq!(model.postprocessor, "identity");
    assert_eq!(model.base_scores, vec![0.0]);
    assert_eq!(model.num_target, 1);

    // sigmoid_alpha / ratio_c carry the model defaults (1.0).
    assert_eq!(model.sigmoid_alpha, 1.0);
    assert_eq!(model.ratio_c, 1.0);
}

#[test]
fn bulk_to_model_preserves_average_tree_output() {
    // RF needs average_tree_output=true (Pitfall 6 averaging gate downstream).
    let trees = vec![small_bulk_tree()];
    let mut meta = rf_meta();
    meta.target_id = vec![0];
    meta.class_id = vec![0];
    meta.average_tree_output = true;
    let model = bulk_to_model(trees, meta);
    assert!(model.average_tree_output);

    // And the false case round-trips too.
    let trees2 = vec![small_bulk_tree()];
    let mut meta2 = rf_meta();
    meta2.target_id = vec![0];
    meta2.class_id = vec![0];
    meta2.average_tree_output = false;
    let model2 = bulk_to_model(trees2, meta2);
    assert!(!model2.average_tree_output);
}

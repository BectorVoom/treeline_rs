//! Tests for the f64 construction mode of `ModelBuilder` (Plan 04-01 Task 1).
//!
//! The hard enabling gate for Phase 4 (D-05, RESEARCH Pitfall 1 / Open Q2):
//! LightGBM and sklearn both require the `<f64,f64>` preset, which means the
//! builder must produce `ModelVariant::F64` without downcasting f64
//! thresholds/leaves to f32. These tests assert:
//!   1. an f64 build commits to a `ModelVariant::F64` Model whose
//!      `threshold_type()`/`leaf_output_type()` report Float64;
//!   2. an f64 leaf carrying sub-f32 precision survives commit exactly (no
//!      downcast through f32);
//!   3. the existing f32 path is unchanged — an f32 build still produces
//!      `ModelVariant::F32`.

use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::ModelVariant;
use treelite_core::enums::{DType, Operator, TaskType};

/// Metadata for a single-tree, single-target, scalar-leaf model.
fn meta(num_feature: i32) -> BuilderMetadata {
    BuilderMetadata {
        num_feature,
        task_type: TaskType::kRegressor,
        average_tree_output: false,
        num_target: 1,
        num_class: vec![1],
        leaf_vector_shape: vec![1, 1],
        target_id: vec![0],
        class_id: vec![0],
        postprocessor: "identity".to_string(),
        base_scores: vec![0.0],
        attributes: None,
    }
}

/// Build a tiny single-tree f64 model: a numerical-test root on feature 0 at
/// `threshold`, with two scalar f64 leaves `(left_leaf, right_leaf)`.
fn build_f64_single_tree(threshold: f64, left_leaf: f64, right_leaf: f64) -> treelite_core::Model {
    let mut b = ModelBuilder::new(meta(4)).unwrap();
    b.start_tree().unwrap();
    // root
    b.start_node(0).unwrap();
    b.numerical_test_f64(0, threshold, true, Operator::kLT, 1, 2)
        .unwrap();
    b.end_node().unwrap();
    // left leaf
    b.start_node(1).unwrap();
    b.leaf_scalar_f64(left_leaf).unwrap();
    b.end_node().unwrap();
    // right leaf
    b.start_node(2).unwrap();
    b.leaf_scalar_f64(right_leaf).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    b.commit_model().unwrap()
}

#[test]
fn f64_build_commits_to_f64_variant_with_float64_type_tags() {
    let mut model = build_f64_single_tree(0.5, 10.0, 20.0);

    // Variant is F64 with exactly one tree.
    match &model.variant {
        ModelVariant::F64(p) => assert_eq!(p.num_trees(), 1),
        ModelVariant::F32(_) => panic!("expected ModelVariant::F64, got F32"),
    }

    // After staging, both type tags report Float64.
    model.stage_serialization_fields();
    assert_eq!(model.threshold_type(), DType::kFloat64);
    assert_eq!(model.leaf_output_type(), DType::kFloat64);
}

#[test]
fn f64_leaf_sub_f32_precision_survives_commit_without_downcast() {
    // A leaf value that differs from its f32 round-trip at ~1e-7. If the builder
    // downcast through f32 anywhere, this exact equality would fail.
    let precise: f64 = 0.1_f64 + 1e-7_f64;
    // Sanity: this value is NOT representable through f32 (the downcast loses it).
    assert_ne!(precise, precise as f32 as f64);

    let model = build_f64_single_tree(0.5, precise, 20.0);

    match &model.variant {
        ModelVariant::F64(p) => {
            let tree = &p.trees[0];
            // node 1 is the left leaf
            assert_eq!(tree.leaf_value(1), precise);
        }
        ModelVariant::F32(_) => panic!("expected ModelVariant::F64"),
    }
}

#[test]
fn f64_threshold_sub_f32_precision_survives_commit() {
    let precise_thresh: f64 = 0.3_f64 + 1e-7_f64;
    assert_ne!(precise_thresh, precise_thresh as f32 as f64);

    let model = build_f64_single_tree(precise_thresh, 10.0, 20.0);

    match &model.variant {
        ModelVariant::F64(p) => {
            let tree = &p.trees[0];
            assert_eq!(tree.threshold(0), precise_thresh);
        }
        ModelVariant::F32(_) => panic!("expected ModelVariant::F64"),
    }
}

#[test]
fn f32_build_path_still_produces_f32_variant() {
    // The existing f32 construction path (used by XGBoost) is unchanged.
    let mut b = ModelBuilder::new(meta(4)).unwrap();
    b.start_tree().unwrap();
    b.start_node(0).unwrap();
    b.numerical_test(0, 0.5_f32, true, Operator::kLT, 1, 2)
        .unwrap();
    b.end_node().unwrap();
    b.start_node(1).unwrap();
    b.leaf_scalar(10.0_f32).unwrap();
    b.end_node().unwrap();
    b.start_node(2).unwrap();
    b.leaf_scalar(20.0_f32).unwrap();
    b.end_node().unwrap();
    b.end_tree().unwrap();
    let mut model = b.commit_model().unwrap();

    match &model.variant {
        ModelVariant::F32(p) => assert_eq!(p.num_trees(), 1),
        ModelVariant::F64(_) => panic!("expected ModelVariant::F32, got F64"),
    }
    model.stage_serialization_fields();
    assert_eq!(model.threshold_type(), DType::kFloat32);
    assert_eq!(model.leaf_output_type(), DType::kFloat32);
}

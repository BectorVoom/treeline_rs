//! Tests for `concatenate` (BLD-02), ported from `model_concat.cc` semantics.
//!
//! Merging two built models sums their tree counts and extends `target_id` /
//! `class_id`; a different-variant input is rejected; an empty slice → `None`.

use treelite_builder::error::BuilderError;
use treelite_builder::{BuilderMetadata, ModelBuilder, concatenate};
use treelite_core::enums::TaskType;
use treelite_core::{Model, ModelPreset, ModelVariant, Tree};

/// Build a small F32 model with `n` single-leaf trees (one node each).
fn build_n_leaf_model(n: usize) -> Model {
    let meta = BuilderMetadata {
        num_feature: 4,
        task_type: TaskType::kRegressor,
        average_tree_output: false,
        num_target: 1,
        num_class: vec![1].into(),
        leaf_vector_shape: vec![1, 1].into(),
        target_id: vec![0; n].into(),
        class_id: vec![0; n].into(),
        postprocessor: "identity".into(),
        base_scores: vec![0.0].into(),
        attributes: None,
    };
    let mut b = ModelBuilder::new(meta).unwrap();
    for i in 0..n {
        b.start_tree().unwrap();
        b.start_node(0).unwrap();
        b.leaf_scalar(i as f32).unwrap();
        b.end_node().unwrap();
        b.end_tree().unwrap();
    }
    b.commit_model().unwrap()
}

#[test]
fn merge_two_models_sums_trees_and_extends_ids() {
    let a = build_n_leaf_model(2);
    let c = build_n_leaf_model(3);
    let merged = concatenate(&[&a, &c]).unwrap().expect("non-empty input");
    match &merged.variant {
        ModelVariant::F32(p) => assert_eq!(p.num_trees(), 5),
        _ => panic!("expected F32 variant"),
    }
    assert_eq!(merged.target_id.len(), 5);
    assert_eq!(merged.class_id.len(), 5);
    // Header copied from objs[0].
    assert_eq!(merged.num_feature, 4);
    assert_eq!(merged.num_target, 1);
    assert_eq!(merged.num_class.as_slice(), &[1]);
}

#[test]
fn empty_input_returns_none() {
    let merged = concatenate(&[]).unwrap();
    assert!(merged.is_none());
}

#[test]
fn variant_mismatch_rejected() {
    let a = build_n_leaf_model(1); // F32
    // A hand-built F64 model.
    let f64_model = Model::new(ModelVariant::F64(ModelPreset::new(
        vec![Tree::<f64>::new()],
    )));
    match concatenate(&[&a, &f64_model]) {
        Err(BuilderError::VariantMismatch { index: 1 }) => {}
        Err(other) => panic!("wrong error variant: {other:?}"),
        Ok(_) => panic!("expected VariantMismatch"),
    }
}

#[test]
fn num_target_mismatch_rejected() {
    let a = build_n_leaf_model(1);
    let mut b = build_n_leaf_model(1);
    b.num_target = 2; // diverge from objs[0]
    match concatenate(&[&a, &b]) {
        Err(BuilderError::HeaderMismatch {
            index: 1,
            field: "num_target",
        }) => {}
        Err(other) => panic!("wrong error variant: {other:?}"),
        Ok(_) => panic!("expected HeaderMismatch num_target"),
    }
}

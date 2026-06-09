//! Task 2 tests: load the committed XGBoost-JSON fixture into a `Model`
//! (F32 variant, sigmoid, kBinaryClf) with a correctly f64-margin-transformed
//! base_score (the live CORE-04 base_scores value), and verify malformed input
//! returns a typed error rather than panicking (ERR-01).

use std::path::PathBuf;

use treelite_core::{ModelVariant, TaskType};
use treelite_xgboost::error::XgbError;
use treelite_xgboost::{load_xgboost_json, transform_base_score_to_margin};

/// Read the committed fixture from the workspace `fixtures/` directory.
/// `CARGO_MANIFEST_DIR` is `crates/treelite-xgboost`, so go up two levels.
fn read_fixture() -> String {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../fixtures/binary_logistic.model.json");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read fixture {}: {e}", path.display()))
}

#[test]
fn fixture_loads_into_f32_binary_clf_model() {
    let model = load_xgboost_json(&read_fixture()).expect("fixture should load");

    // XGBoost-JSON always yields the F32 variant, with two trees.
    match &model.variant {
        ModelVariant::F32(preset) => assert_eq!(preset.num_trees(), 2),
        ModelVariant::F64(_) => panic!("expected F32 variant"),
    }

    assert_eq!(model.task_type, TaskType::kBinaryClf);
    assert_eq!(model.postprocessor, "sigmoid");
    assert!(!model.average_tree_output);
    assert_eq!(model.num_feature, 2);
    assert_eq!(model.num_target, 1);

    // Array-typed header fields at their Phase-1 binary:logistic values.
    assert_eq!(model.num_class, vec![1]);
    assert_eq!(model.leaf_vector_shape, vec![1, 1]);
    assert_eq!(model.target_id, vec![0, 0]); // tree_info = [0, 0]
    assert_eq!(model.class_id, vec![0, 0]); // num_tree = 2
}

#[test]
fn base_scores_is_exact_f64_margin_transform() {
    let model = load_xgboost_json(&read_fixture()).expect("fixture should load");

    // CORE-04: the live f64 margin value, NOT the raw 0.25 and NOT merely a
    // structural stub. version [4,7,0] fires the transform gate.
    let expected = transform_base_score_to_margin("sigmoid", 0.25);
    assert_eq!(model.base_scores.len(), 1);
    assert_eq!(model.base_scores[0], expected);
    // Documented exact value: -ln(3) ≈ -1.0986122886681098.
    assert!((model.base_scores[0] - (-1.0986122886681098_f64)).abs() < 1e-15);
    // And definitely NOT the untransformed probability.
    assert_ne!(model.base_scores[0], 0.25);
}

#[test]
fn tree_structure_is_built_correctly() {
    let model = load_xgboost_json(&read_fixture()).expect("fixture should load");
    let ModelVariant::F32(preset) = &model.variant else {
        panic!("expected F32 variant");
    };

    // Tree 0: root numerical test (split_index 0, threshold 0.5, default_left),
    // then two leaves with values -0.75 and 1.25.
    let t0 = &preset.trees[0];
    assert_eq!(t0.num_nodes, 3);
    assert!(!t0.is_leaf(0));
    assert_eq!(t0.split_index(0), 0);
    assert_eq!(t0.threshold(0), 0.5_f32);
    assert!(t0.default_child(0) == t0.left_child(0)); // default_left[0] == 1
    assert!(t0.is_leaf(1));
    assert_eq!(t0.leaf_value(1), -0.75_f32);
    assert!(t0.is_leaf(2));
    assert_eq!(t0.leaf_value(2), 1.25_f32);
}

#[test]
fn dimension_mismatch_returns_typed_err_not_panic() {
    // Hand-edit the fixture so left_children has the wrong length vs num_nodes.
    let json = read_fixture().replace(
        "\"left_children\": [1, -1, -1],\n            \"right_children\": [2, -1, -1],\n            \"parents\": [2147483647, 0, 0],\n            \"split_indices\": [0, 0, 0],",
        "\"left_children\": [1, -1],\n            \"right_children\": [2, -1, -1],\n            \"parents\": [2147483647, 0, 0],\n            \"split_indices\": [0, 0, 0],",
    );
    // Ensure the edit actually applied (guard against a silent no-op match).
    assert!(json.contains("\"left_children\": [1, -1],"));

    // `Model` is intentionally not `Debug` (move-only header), so match on the
    // result directly rather than using `expect_err`.
    match load_xgboost_json(&json) {
        Err(XgbError::DimensionMismatch {
            tree,
            field,
            expected,
            got,
        }) => {
            assert_eq!(tree, 0);
            assert_eq!(field, "left_children");
            assert_eq!(expected, 3);
            assert_eq!(got, 2);
        }
        Err(other) => panic!("expected DimensionMismatch, got {other:?}"),
        Ok(_) => panic!("expected DimensionMismatch error, got Ok(model)"),
    }
}

#[test]
fn malformed_json_returns_typed_err_not_panic() {
    match load_xgboost_json("{ this is not valid json") {
        Err(XgbError::Json(_)) => {}
        Err(other) => panic!("expected Json error, got {other:?}"),
        Ok(_) => panic!("expected Json error, got Ok(model)"),
    }
}

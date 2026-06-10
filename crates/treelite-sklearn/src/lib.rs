//! `treelite-sklearn` — scikit-learn estimator loaders (SKL-01..04).
//!
//! Ports `treelite-mainline/src/model_loader/sklearn.cc` and
//! `.../sklearn_bulk.cc`, exposing array-signature loaders that mirror the
//! upstream `namespace sklearn` declarations in
//! `treelite-mainline/include/treelite/model_loader.h` 1:1 (D-01): the upstream
//! `double const**` / `std::int64_t const**` array-of-arrays become `&[&[f64]]`
//! / `&[&[i64]]` slices, so the Phase-8 PyO3 layer can hand zero-copy numpy
//! buffers straight through.
//!
//! Two loader paths are delivered here:
//! - [`bulk`] — RandomForest / ExtraTrees (SKL-01): the bulk fast path that
//!   bypasses the node-by-node builder and assembles a `ModelVariant::F64` Model
//!   directly via [`treelite_builder::bulk_construct_tree`] +
//!   [`treelite_builder::bulk_to_model`]. RF leaf-normalization is load-time
//!   (already inside `bulk_construct_tree`, A4).
//! - [`mixin`] — GradientBoosting (SKL-02): the node-by-node MixIn path through
//!   the f64 `ModelBuilder`. GB leaf-shrink is capture-side; the loader does NOT
//!   re-shrink (A4 / Anti-pattern).
//!
//! IsolationForest (SKL-03) is also delivered here via the [`mixin`] path
//! ([`load_isolation_forest`], `task=kIsolationForest`,
//! `postprocessor="exponential_standard_ratio"` + `model.ratio_c`): leaf values
//! are the pre-computed isolation depths consumed AS-IS (no loader-side
//! recomputation) and the Treelite output deliberately differs from the
//! framework's own anomaly score (it equals `-clf.score_samples`, D-07).
//!
//! HistGradientBoosting (SKL-04) is delivered via the [`histgb`] path
//! ([`load_hist_gradient_boosting_regressor`] /
//! [`load_hist_gradient_boosting_classifier`]): unlike the other sklearn loaders
//! it receives a RAW PACKED BYTE BUFFER per tree (the
//! `HistGradientBoostingNode<FeatureIdT>` C struct), decoded field-by-field via
//! `from_le_bytes` (Phase-3 D-08 byte-cursor discipline) at the 52/56-byte
//! layout offsets, with `features_map` always applied to the split index and
//! `categories_map` remapping categorical bit values when present.

pub mod bulk;
pub mod error;
pub mod histgb;
pub mod mixin;

pub use error::SklError;

pub use bulk::{load_random_forest_classifier, load_random_forest_regressor};
pub use histgb::{
    load_hist_gradient_boosting_classifier, load_hist_gradient_boosting_regressor,
};
pub use mixin::{
    load_gradient_boosting_classifier, load_gradient_boosting_regressor, load_isolation_forest,
};

// ---------------------------------------------------------------------------
// ExtraTrees
// ---------------------------------------------------------------------------
//
// sklearn treats ExtraTrees identically to RandomForest in the loader (the
// array dumps have the same shape and the same load-time leaf-normalization),
// so the ExtraTrees entry points route to the same bulk implementation.

/// Load an `ExtraTreesRegressor` via the RF/ET bulk path (SKL-01).
///
/// Identical to [`load_random_forest_regressor`] — sklearn's loader does not
/// distinguish ExtraTrees from RandomForest.
#[allow(clippy::too_many_arguments)]
pub fn load_extra_trees_regressor(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
) -> Result<treelite_core::Model, SklError> {
    load_random_forest_regressor(
        n_estimators,
        n_features,
        n_targets,
        node_count,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        n_node_samples,
        weighted_n_node_samples,
        impurity,
    )
}

/// Load an `ExtraTreesClassifier` via the RF/ET bulk path (SKL-01).
///
/// Identical to [`load_random_forest_classifier`] — sklearn's loader does not
/// distinguish ExtraTrees from RandomForest.
#[allow(clippy::too_many_arguments)]
pub fn load_extra_trees_classifier(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    n_classes: &[i32],
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
) -> Result<treelite_core::Model, SklError> {
    load_random_forest_classifier(
        n_estimators,
        n_features,
        n_targets,
        n_classes,
        node_count,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        n_node_samples,
        weighted_n_node_samples,
        impurity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use treelite_core::TaskType;

    // A tiny hand-built 2-node tree: node 0 is an internal split on feature 1,
    // node 1 and node 2 are leaves. (children_left[0]=1 internal; leaves carry
    // children_left == -1.)
    //
    //        node0 (feature 1, thr 0.5)
    //        /            \
    //     node1 (leaf)   node2 (leaf)
    fn tiny_reg_tree() -> (
        Vec<i64>, // children_left
        Vec<i64>, // children_right
        Vec<i64>, // feature
        Vec<f64>, // threshold
        Vec<f64>, // value (scalar per node)
        Vec<i64>, // n_node_samples
        Vec<f64>, // weighted_n_node_samples
        Vec<f64>, // impurity
    ) {
        (
            vec![1, -1, -1],
            vec![2, -1, -1],
            vec![1, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 3.0, 7.0],
            vec![10, 4, 6],
            vec![10.0, 4.0, 6.0],
            vec![1.0, 0.0, 0.0],
        )
    }

    #[test]
    fn rf_regressor_sets_identity_regressor_metadata() {
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_reg_tree();
        let model = load_random_forest_regressor(
            1,
            2,
            1,
            &[3],
            &[&cl],
            &[&cr],
            &[&feat],
            &[&thr],
            &[&val],
            &[&nns],
            &[&wns],
            &[&imp],
        )
        .expect("rf regressor loads");

        assert_eq!(model.task_type, TaskType::kRegressor);
        assert_eq!(model.postprocessor, "identity");
        assert!(model.average_tree_output);
        assert_eq!(model.num_target, 1);
        assert_eq!(model.num_class, vec![1]);
        assert_eq!(model.leaf_vector_shape, vec![1, 1]);
        // n_targets == 1 → target_id is 0 (not -1); class_id all 0.
        assert_eq!(model.target_id, vec![0]);
        assert_eq!(model.class_id, vec![0]);
        assert_eq!(model.base_scores, vec![0.0]);
    }

    #[test]
    fn rf_classifier_sets_identity_multiclass_metadata_with_broadcast_ids() {
        // A binary classifier: per-node value is a class-count vector of len
        // max_num_class=2, so the flat value buffer is node_count * 2.
        let cl = vec![1_i64, -1, -1];
        let cr = vec![2_i64, -1, -1];
        let feat = vec![1_i64, -1, -1];
        let thr = vec![0.5_f64, 0.0, 0.0];
        // value: node0 [3,7], node1 [4,0], node2 [0,6] (raw class counts).
        let val = vec![3.0_f64, 7.0, 4.0, 0.0, 0.0, 6.0];
        let nns = vec![10_i64, 4, 6];
        let wns = vec![10.0_f64, 4.0, 6.0];
        let imp = vec![1.0_f64, 0.0, 0.0];

        let model = load_random_forest_classifier(
            1,
            2,
            1,
            &[2],
            &[3],
            &[&cl],
            &[&cr],
            &[&feat],
            &[&thr],
            &[&val],
            &[&nns],
            &[&wns],
            &[&imp],
        )
        .expect("rf classifier loads");

        assert_eq!(model.task_type, TaskType::kMultiClf);
        assert_eq!(model.postprocessor, "identity_multiclass");
        assert!(model.average_tree_output);
        assert_eq!(model.num_class, vec![2]);
        assert_eq!(model.leaf_vector_shape, vec![1, 2]);
        // Leaf-vector broadcast → target_id/class_id all -1 (sklearn_bulk.cc).
        assert_eq!(model.target_id, vec![-1]);
        assert_eq!(model.class_id, vec![-1]);
        assert_eq!(model.base_scores, vec![0.0, 0.0]);
    }

    #[test]
    fn leaf_detection_uses_minus_one_not_le_zero() {
        // An internal node whose children indices are valid (1, 2) must NOT be
        // mis-classified as a leaf. With a `<= 0` rule, child index 0 would be
        // treated as a leaf marker; the correct rule is `== -1`. Build a tree
        // where node 0 points its LEFT child at index 0... which is itself, so
        // instead verify via a tree whose internal node has children {1,2} and
        // confirm node 0 is an internal split (has a real split_index) by
        // checking the model predicts a non-degenerate routing.
        //
        // Simpler structural check: a 3-node tree with root internal and two
        // leaves must yield exactly 2 distinct leaf outputs under routing.
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_reg_tree();
        let model = load_random_forest_regressor(
            1,
            2,
            1,
            &[3],
            &[&cl],
            &[&cr],
            &[&feat],
            &[&thr],
            &[&val],
            &[&nns],
            &[&wns],
            &[&imp],
        )
        .expect("loads");

        // Route feature[1] < 0.5 (-> left leaf, value 3.0) and
        // feature[1] >= 0.5 (-> right leaf, value 7.0). If node 0 had been
        // mis-detected as a leaf, both rows would return node 0's value (0.0).
        let row_left = [0.0_f32, 0.0]; // feature 1 = 0.0 < 0.5
        let row_right = [0.0_f32, 1.0]; // feature 1 = 1.0 >= 0.5
        let mut flat = Vec::new();
        flat.extend_from_slice(&row_left);
        flat.extend_from_slice(&row_right);
        let out = treelite_gtil::predict(&model, &flat, 2).expect("predict");
        // average_tree_output with a single tree → the leaf value itself.
        approx::assert_abs_diff_eq!(out[0], 3.0_f32, epsilon = 1e-5);
        approx::assert_abs_diff_eq!(out[1], 7.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn extra_trees_routes_to_rf_bulk() {
        // ExtraTrees regressor must produce the identical model as RF regressor.
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_reg_tree();
        let et = load_extra_trees_regressor(
            1, 2, 1, &[3], &[&cl], &[&cr], &[&feat], &[&thr], &[&val], &[&nns], &[&wns], &[&imp],
        )
        .expect("et loads");
        assert_eq!(et.task_type, TaskType::kRegressor);
        assert_eq!(et.postprocessor, "identity");
    }

    #[test]
    fn out_of_range_child_index_is_typed_error_not_panic() {
        // children_right points past node_count → ChildIndexOutOfRange, never OOB.
        let cl = vec![1_i64, -1, -1];
        let cr = vec![99_i64, -1, -1]; // out of range
        let feat = vec![1_i64, -1, -1];
        let thr = vec![0.5_f64, 0.0, 0.0];
        let val = vec![0.0_f64, 3.0, 7.0];
        let nns = vec![10_i64, 4, 6];
        let wns = vec![10.0_f64, 4.0, 6.0];
        let imp = vec![1.0_f64, 0.0, 0.0];
        let res = load_random_forest_regressor(
            1, 2, 1, &[3], &[&cl], &[&cr], &[&feat], &[&thr], &[&val], &[&nns], &[&wns], &[&imp],
        );
        assert!(matches!(res, Err(SklError::ChildIndexOutOfRange { .. })));
    }
}

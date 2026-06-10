//! GradientBoosting node-by-node MixIn loader path (SKL-02).
//!
//! Ports `treelite-mainline/src/model_loader/sklearn.cc` `LoadSKLearnModel`
//! (`:200-258`) driven by the GradientBoosting MixIns (`:59-133`): per tree,
//! drive the f64 `ModelBuilder` (`start_tree`/`start_node`/`numerical_test_f64`/
//! `leaf_scalar_f64`/`gain`/`data_count`/`sum_hess`/`end_node`/`end_tree`),
//! mirroring `xgboost::build_tree` but with `Operator::kLE` (sklearn) and f64
//! entry points (D-05, no downcast).
//!
//! Metadata is set per the GradientBoosting MixIns (`sklearn.cc:59-133`):
//! - regressor  → `task=kRegressor`, `postprocessor="identity"`
//! - binary clf → `task=kBinaryClf`, `postprocessor="sigmoid"`
//! - multi clf  → `task=kMultiClf`,  `postprocessor="softmax"`,
//!   `class_id[tree] = tree % n_classes` round-robin
//!
//! CRITICAL (A4 / Anti-pattern, T-04-15): the captured array dump already
//! carries capture-side-shrunk leaf values, so the loader uses them AS-PROVIDED
//! — it does NOT multiply by `learning_rate`. Re-shrinking would scale every
//! prediction. There is intentionally NO `learning_rate` parameter and NO
//! `* learning_rate` anywhere in this module.

use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::{Model, Operator, TaskType};

use crate::error::SklError;

/// Validate that a tree's outer-slice count matches the declared tree count.
fn check_outer(field: &'static str, expected: usize, got: usize) -> Result<(), SklError> {
    if got != expected {
        return Err(SklError::TreeCountMismatch {
            field,
            expected,
            got,
        });
    }
    Ok(())
}

/// Validate that a per-tree parallel array length matches that tree's node count.
fn check_dim(
    tree: usize,
    field: &'static str,
    node_count: usize,
    got: usize,
) -> Result<(), SklError> {
    if got != node_count {
        return Err(SklError::DimensionMismatch {
            tree,
            field,
            expected: node_count,
            got,
        });
    }
    Ok(())
}

/// Reject a model scalar that must be at least 1.
fn require_positive(field: &'static str, value: i32) -> Result<i32, SklError> {
    if value < 1 {
        return Err(SklError::InvalidScalar {
            field,
            value: value as i64,
            reason: "must be at least 1",
        });
    }
    Ok(value)
}

/// Drive one tree through the f64 `ModelBuilder` (`sklearn.cc:220-256`).
///
/// Leaf detection is `children_left[node] == -1` (NOT `<= 0`). Internal nodes
/// emit `numerical_test_f64` with `Operator::kLE`, `default_left=true`, plus the
/// sklearn impurity-reduction `gain`. Every node emits `data_count` and
/// `sum_hess`. Leaf values are used AS-PROVIDED — no `learning_rate` re-shrink
/// (A4 / T-04-15).
#[allow(clippy::too_many_arguments)]
fn build_tree(
    builder: &mut ModelBuilder,
    tree: usize,
    node_count_i64: i64,
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    n_node_samples: &[i64],
    weighted_n_node_samples: &[f64],
    impurity: &[f64],
) -> Result<(), SklError> {
    // node_count <= INT_MAX overflow guard (sklearn.cc:215-217, T-04-14).
    if node_count_i64 < 0 {
        return Err(SklError::InvalidScalar {
            field: "node_count",
            value: node_count_i64,
            reason: "must be non-negative",
        });
    }
    if node_count_i64 > i32::MAX as i64 {
        return Err(SklError::InvalidScalar {
            field: "node_count",
            value: node_count_i64,
            reason: "exceeds i32::MAX",
        });
    }
    let n_nodes = node_count_i64 as usize;

    check_dim(tree, "children_left", n_nodes, children_left.len())?;
    check_dim(tree, "children_right", n_nodes, children_right.len())?;
    check_dim(tree, "feature", n_nodes, feature.len())?;
    check_dim(tree, "threshold", n_nodes, threshold.len())?;
    check_dim(tree, "value", n_nodes, value.len())?;
    check_dim(tree, "n_node_samples", n_nodes, n_node_samples.len())?;
    check_dim(
        tree,
        "weighted_n_node_samples",
        n_nodes,
        weighted_n_node_samples.len(),
    )?;
    check_dim(tree, "impurity", n_nodes, impurity.len())?;

    // Bounds-check every child index before the gain formula dereferences it
    // (T-04-13). Leaves carry -1; internal nodes must point into 0..n_nodes.
    for node in 0..n_nodes {
        for child in [children_left[node], children_right[node]] {
            if child == -1 {
                continue;
            }
            if child < 0 || child as usize >= n_nodes {
                return Err(SklError::ChildIndexOutOfRange {
                    tree,
                    node,
                    child,
                    node_count: n_nodes,
                });
            }
        }
    }

    // total_sample_cnt = n_node_samples[tree][0] (sklearn.cc:214).
    let total_sample_cnt = if n_nodes > 0 { n_node_samples[0] } else { 0 };

    builder.start_tree()?;
    for node in 0..n_nodes {
        let left_child = children_left[node] as i32;
        let sample_cnt = n_node_samples[node];
        let weighted_sample_cnt = weighted_n_node_samples[node];

        builder.start_node(node as i32)?;
        if left_child == -1 {
            // Leaf — use the value AS-PROVIDED (no learning_rate re-shrink, A4).
            builder.leaf_scalar_f64(value[node])?;
        } else {
            let right_child = children_right[node] as i32;
            let split_index = feature[node] as i32;
            let split_cond = threshold[node];

            // sklearn impurity-reduction gain (sklearn.cc:232-241), in f64.
            let lc = left_child as usize;
            let rc = right_child as usize;
            let left_sample_cnt = n_node_samples[lc] as f64;
            let right_sample_cnt = n_node_samples[rc] as f64;
            let sc = sample_cnt as f64;
            let gain = sc
                * (impurity[node]
                    - left_sample_cnt * impurity[lc] / sc
                    - right_sample_cnt * impurity[rc] / sc)
                / total_sample_cnt as f64;

            builder.numerical_test_f64(
                split_index,
                split_cond,
                true, // default_left (sklearn.cc:247)
                Operator::kLE,
                left_child,
                right_child,
            )?;
            builder.gain(gain)?;
        }
        // Every node emits data_count + sum_hess (sklearn.cc:251-252).
        builder.data_count(sample_cnt as u64)?;
        builder.sum_hess(weighted_sample_cnt)?;
        builder.end_node()?;
    }
    builder.end_tree()?;
    Ok(())
}

/// Drive every tree through the builder and commit, given a finalized
/// [`BuilderMetadata`]. Shared by the regressor and classifier entry points.
#[allow(clippy::too_many_arguments)]
fn build_model(
    n_trees: usize,
    metadata: BuilderMetadata,
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
) -> Result<Model, SklError> {
    check_outer("node_count", n_trees, node_count.len())?;
    check_outer("children_left", n_trees, children_left.len())?;
    check_outer("children_right", n_trees, children_right.len())?;
    check_outer("feature", n_trees, feature.len())?;
    check_outer("threshold", n_trees, threshold.len())?;
    check_outer("value", n_trees, value.len())?;
    check_outer("n_node_samples", n_trees, n_node_samples.len())?;
    check_outer(
        "weighted_n_node_samples",
        n_trees,
        weighted_n_node_samples.len(),
    )?;
    check_outer("impurity", n_trees, impurity.len())?;

    let mut builder = ModelBuilder::new(metadata)?;
    for t in 0..n_trees {
        build_tree(
            &mut builder,
            t,
            node_count[t],
            children_left[t],
            children_right[t],
            feature[t],
            threshold[t],
            value[t],
            n_node_samples[t],
            weighted_n_node_samples[t],
            impurity[t],
        )?;
    }
    Ok(builder.commit_model()?)
}

/// Load a `GradientBoostingRegressor` via the MixIn path (SKL-02).
///
/// Metadata per `GradientBoostingRegressorMixIn` (`sklearn.cc:59-80`):
/// `task=kRegressor`, `num_target=1`, `num_class={1}`,
/// `leaf_vector_shape={1,1}`, `target_id=class_id=vec![0; n_iter]`,
/// `postprocessor="identity"`, `base_scores={base_score}`.
#[allow(clippy::too_many_arguments)]
pub fn load_gradient_boosting_regressor(
    n_iter: i32,
    n_features: i32,
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
    base_score: f64,
) -> Result<Model, SklError> {
    let n_iter = require_positive("n_iter", n_iter)?;
    let n_features = require_positive("n_features", n_features)?;

    let metadata = BuilderMetadata {
        num_feature: n_features,
        task_type: TaskType::kRegressor,
        average_tree_output: false,
        num_target: 1,
        num_class: vec![1],
        leaf_vector_shape: vec![1, 1],
        target_id: vec![0; n_iter as usize],
        class_id: vec![0; n_iter as usize],
        postprocessor: "identity".to_string(),
        base_scores: vec![base_score],
        attributes: None,
    };
    build_model(
        n_iter as usize,
        metadata,
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

/// Load a `GradientBoostingClassifier` via the MixIn path (SKL-02).
///
/// `n_classes >= 2` (`sklearn.cc:386`). Binary
/// (`GradientBoostingBinaryClassifierMixIn`, `sklearn.cc:82-107`):
/// `task=kBinaryClf`, `postprocessor="sigmoid"`, `n_trees = n_iter`,
/// `class_id=vec![0; n_iter]`, `base_scores={base_scores[0]}`. Multiclass
/// (`GradientBoostingMulticlassClassifierMixIn`, `sklearn.cc:108-133`):
/// `task=kMultiClf`, `postprocessor="softmax"`, `n_trees = n_iter * n_classes`,
/// `class_id[tree] = tree % n_classes` round-robin,
/// `base_scores = base_scores[..n_classes]`.
///
/// `base_scores` must carry one entry per class for the multiclass case (and at
/// least one for the binary case). `value` (and the other per-tree arrays) must
/// already carry `n_iter * n_classes` outer slices for the multiclass case
/// (the capture flattens the `n_iter x n_classes` GB tree grid row-major).
#[allow(clippy::too_many_arguments)]
pub fn load_gradient_boosting_classifier(
    n_iter: i32,
    n_features: i32,
    n_classes: i32,
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
    base_scores: &[f64],
) -> Result<Model, SklError> {
    let n_iter = require_positive("n_iter", n_iter)?;
    let n_features = require_positive("n_features", n_features)?;
    if n_classes < 2 {
        return Err(SklError::InvalidScalar {
            field: "n_classes",
            value: n_classes as i64,
            reason: "must be at least 2",
        });
    }

    if n_classes > 2 {
        // Multiclass — softmax, n_iter * n_classes trees, round-robin class_id.
        let n_trees = (n_iter as i64) * (n_classes as i64);
        let n_trees = usize::try_from(n_trees).map_err(|_| SklError::InvalidScalar {
            field: "n_iter*n_classes",
            value: n_trees,
            reason: "exceeds usize",
        })?;
        if base_scores.len() < n_classes as usize {
            return Err(SklError::DimensionMismatch {
                tree: 0,
                field: "base_scores",
                expected: n_classes as usize,
                got: base_scores.len(),
            });
        }
        let class_id: Vec<i32> = (0..n_trees as i32).map(|t| t % n_classes).collect();
        let metadata = BuilderMetadata {
            num_feature: n_features,
            task_type: TaskType::kMultiClf,
            average_tree_output: false,
            num_target: 1,
            num_class: vec![n_classes],
            leaf_vector_shape: vec![1, 1],
            target_id: vec![0; n_trees],
            class_id,
            postprocessor: "softmax".to_string(),
            base_scores: base_scores[..n_classes as usize].to_vec(),
            attributes: None,
        };
        build_model(
            n_trees,
            metadata,
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
    } else {
        // Binary — sigmoid, n_iter trees, class_id all 0.
        if base_scores.is_empty() {
            return Err(SklError::DimensionMismatch {
                tree: 0,
                field: "base_scores",
                expected: 1,
                got: 0,
            });
        }
        let metadata = BuilderMetadata {
            num_feature: n_features,
            task_type: TaskType::kBinaryClf,
            average_tree_output: false,
            num_target: 1,
            num_class: vec![1],
            leaf_vector_shape: vec![1, 1],
            target_id: vec![0; n_iter as usize],
            class_id: vec![0; n_iter as usize],
            postprocessor: "sigmoid".to_string(),
            base_scores: vec![base_scores[0]],
            attributes: None,
        };
        build_model(
            n_iter as usize,
            metadata,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use treelite_core::TaskType;

    // 3-node GB tree: root split on feature 0, two leaves.
    fn tiny_gb_tree() -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<f64>)
    {
        (
            vec![1, -1, -1],
            vec![2, -1, -1],
            vec![0, -1, -1],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.1, -0.1], // capture-side-shrunk leaf values
            vec![10, 4, 6],
            vec![10.0, 4.0, 6.0],
            vec![1.0, 0.0, 0.0],
        )
    }

    #[test]
    fn gb_binary_classifier_sets_sigmoid_binaryclf() {
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_gb_tree();
        let model = load_gradient_boosting_classifier(
            1, 1, 2, &[3], &[&cl], &[&cr], &[&feat], &[&thr], &[&val], &[&nns], &[&wns], &[&imp],
            &[0.0],
        )
        .expect("gb binary loads");
        assert_eq!(model.task_type, TaskType::kBinaryClf);
        assert_eq!(model.postprocessor, "sigmoid");
        assert_eq!(model.class_id, vec![0]);
    }

    #[test]
    fn gb_multiclass_classifier_sets_softmax_roundrobin() {
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_gb_tree();
        // 2 iters * 3 classes = 6 trees.
        let trees = 6;
        let cls: Vec<&[i64]> = vec![&cl; trees];
        let crs: Vec<&[i64]> = vec![&cr; trees];
        let feats: Vec<&[i64]> = vec![&feat; trees];
        let thrs: Vec<&[f64]> = vec![&thr; trees];
        let vals: Vec<&[f64]> = vec![&val; trees];
        let nnss: Vec<&[i64]> = vec![&nns; trees];
        let wnss: Vec<&[f64]> = vec![&wns; trees];
        let imps: Vec<&[f64]> = vec![&imp; trees];
        let ncnt = vec![3_i64; trees];
        let model = load_gradient_boosting_classifier(
            2, 1, 3, &ncnt, &cls, &crs, &feats, &thrs, &vals, &nnss, &wnss, &imps,
            &[0.0, 0.0, 0.0],
        )
        .expect("gb multiclass loads");
        assert_eq!(model.task_type, TaskType::kMultiClf);
        assert_eq!(model.postprocessor, "softmax");
        // class_id round-robin: 0,1,2,0,1,2.
        assert_eq!(model.class_id, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn gb_regressor_uses_leaf_values_as_provided_no_reshrink() {
        // A single-tree regressor with base_score 0. The leaf reached by routing
        // must be exactly the provided (already-shrunk) value, proving NO
        // learning_rate re-multiplication happens in the loader (A4 / T-04-15).
        let (cl, cr, feat, thr, val, nns, wns, imp) = tiny_gb_tree();
        let model = load_gradient_boosting_regressor(
            1, 1, &[3], &[&cl], &[&cr], &[&feat], &[&thr], &[&val], &[&nns], &[&wns], &[&imp], 0.0,
        )
        .expect("gb regressor loads");
        assert_eq!(model.task_type, TaskType::kRegressor);
        assert_eq!(model.postprocessor, "identity");
        // Route feature 0 = 1.0 (>= 0.5) → right leaf, value -0.1 exactly.
        let flat = [1.0_f32];
        let out = treelite_gtil::predict(&model, &flat, 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], -0.1_f32, epsilon = 1e-5);
        // Route feature 0 = 0.0 (< 0.5) → left leaf, value 0.1 exactly.
        let flat = [0.0_f32];
        let out = treelite_gtil::predict(&model, &flat, 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 0.1_f32, epsilon = 1e-5);
    }
}

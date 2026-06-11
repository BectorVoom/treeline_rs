//! RandomForest / ExtraTrees bulk loader path (SKL-01, D-09).
//!
//! Ports `treelite-mainline/src/model_loader/sklearn_bulk.cc`
//! `LoadRandomForestClassifier` (`:230-289`) and `LoadRandomForestRegressor`
//! (`:298-340`): the RF/ET fast path that bypasses the node-by-node
//! `ModelBuilder` entirely. sklearn treats ExtraTrees identically to
//! RandomForest in the loader, so both route here.
//!
//! This module is the CALLER: per tree it bounds-checks the supplied arrays
//! (T-04-13/T-04-14), calls [`treelite_builder::bulk_construct_tree`] (which
//! already ports the per-node fill and the classifier leaf-normalization, A4),
//! collects `Vec<Tree<f64>>`, then hands them to
//! [`treelite_builder::bulk_to_model`] with the metadata set BY HAND per
//! `sklearn_bulk.cc:244-330`.

use treelite_builder::{BuilderMetadata, bulk_construct_tree, bulk_to_model};
use treelite_core::{Model, TaskType, Tree};

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

/// Per-tree bounds + length validation shared by RF/ET reg & clf
/// (T-04-13/T-04-14). Returns the validated `n_nodes` for this tree.
#[allow(clippy::too_many_arguments)]
fn validate_tree(
    tree: usize,
    node_count_i64: i64,
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    n_node_samples: &[i64],
    weighted_n_node_samples: &[f64],
    impurity: &[f64],
) -> Result<usize, SklError> {
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
    check_dim(tree, "n_node_samples", n_nodes, n_node_samples.len())?;
    check_dim(
        tree,
        "weighted_n_node_samples",
        n_nodes,
        weighted_n_node_samples.len(),
    )?;
    check_dim(tree, "impurity", n_nodes, impurity.len())?;

    // Bounds-check every child index BEFORE bulk_construct_tree dereferences it
    // for the gain formula (T-04-13, Security Domain). Leaves carry -1; internal
    // nodes must point into 0..n_nodes.
    for node in 0..n_nodes {
        for (child, _label) in [
            (children_left[node], "children_left"),
            (children_right[node], "children_right"),
        ] {
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
    Ok(n_nodes)
}

/// Shared RF/ET tree-assembly loop: validate each tree, then call the bulk
/// constructor. `value` is each tree's flat per-node leaf payload (length
/// `n_nodes * leaf_vector_size` for vector leaves, or `n_nodes` for scalar).
#[allow(clippy::too_many_arguments)]
fn build_trees(
    n_estimators: usize,
    node_count: &[i64],
    children_left: &[&[i64]],
    children_right: &[&[i64]],
    feature: &[&[i64]],
    threshold: &[&[f64]],
    value: &[&[f64]],
    n_node_samples: &[&[i64]],
    weighted_n_node_samples: &[&[f64]],
    impurity: &[&[f64]],
    n_targets: i32,
    max_num_class: i32,
    is_classifier: bool,
) -> Result<Vec<Tree<f64>>, SklError> {
    check_outer("node_count", n_estimators, node_count.len())?;
    check_outer("children_left", n_estimators, children_left.len())?;
    check_outer("children_right", n_estimators, children_right.len())?;
    check_outer("feature", n_estimators, feature.len())?;
    check_outer("threshold", n_estimators, threshold.len())?;
    check_outer("value", n_estimators, value.len())?;
    check_outer("n_node_samples", n_estimators, n_node_samples.len())?;
    check_outer(
        "weighted_n_node_samples",
        n_estimators,
        weighted_n_node_samples.len(),
    )?;
    check_outer("impurity", n_estimators, impurity.len())?;

    let leaf_vector_size = (n_targets * max_num_class) as usize;
    let has_leaf_vector = leaf_vector_size > 1;

    let mut trees = Vec::with_capacity(n_estimators);
    for t in 0..n_estimators {
        let n_nodes = validate_tree(
            t,
            node_count[t],
            children_left[t],
            children_right[t],
            feature[t],
            threshold[t],
            n_node_samples[t],
            weighted_n_node_samples[t],
            impurity[t],
        )?;

        // The flat `value` buffer must cover the per-node leaf payload before
        // bulk_construct_tree indexes it (ERR-01).
        let required = if has_leaf_vector {
            n_nodes * leaf_vector_size
        } else {
            n_nodes
        };
        if value[t].len() < required {
            return Err(SklError::ValueBufferTooShort {
                tree: t,
                expected: required,
                got: value[t].len(),
            });
        }

        // total_sample_cnt = n_node_samples[tree][0] (sklearn_bulk.cc:331-332).
        // n_nodes >= 1 is guaranteed for a fitted sklearn tree (the root); guard
        // anyway so an empty tree cannot index [0].
        let total_sample_cnt = if n_nodes > 0 { n_node_samples[t][0] } else { 0 };

        let tree = bulk_construct_tree(
            n_nodes,
            children_left[t],
            children_right[t],
            feature[t],
            threshold[t],
            value[t],
            n_node_samples[t],
            weighted_n_node_samples[t],
            impurity[t],
            total_sample_cnt,
            n_targets,
            max_num_class,
            is_classifier,
        );
        trees.push(tree);
    }
    Ok(trees)
}

/// Load a `RandomForestRegressor` / `ExtraTreesRegressor` via the bulk path
/// (`sklearn_bulk.cc:298-340`, SKL-01).
///
/// Metadata (hand-set per `:309-330`): `task=kRegressor`,
/// `average_tree_output=true`, `num_class=vec![1; n_targets]`,
/// `leaf_vector_shape={n_targets, 1}`, `target_id=vec![-1 or 0; n_estimators]`
/// (`-1` iff `n_targets > 1`), `class_id=vec![0; n_estimators]`,
/// `postprocessor="identity"`, `base_scores=vec![0.0; n_targets]`.
#[allow(clippy::too_many_arguments)]
pub fn load_random_forest_regressor(
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
) -> Result<Model, SklError> {
    let n_estimators = require_positive("n_estimators", n_estimators)?;
    let n_features = require_positive("n_features", n_features)?;
    let n_targets = require_positive("n_targets", n_targets)?;

    let trees = build_trees(
        n_estimators as usize,
        node_count,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        n_node_samples,
        weighted_n_node_samples,
        impurity,
        n_targets,
        1, // max_num_class for a regressor
        false,
    )?;

    let target_id = vec![if n_targets > 1 { -1 } else { 0 }; n_estimators as usize];
    let class_id = vec![0; n_estimators as usize];

    let metadata = BuilderMetadata {
        num_feature: n_features,
        task_type: TaskType::kRegressor,
        average_tree_output: true,
        num_target: n_targets,
        // MEM-02: BuilderMetadata fields are SmallVec/CompactString — `.into()`.
        num_class: vec![1; n_targets as usize].into(),
        leaf_vector_shape: vec![n_targets, 1].into(),
        target_id: target_id.into(),
        class_id: class_id.into(),
        postprocessor: "identity".into(),
        base_scores: vec![0.0; n_targets as usize].into(),
        attributes: None,
    };
    Ok(bulk_to_model(trees, metadata))
}

/// Load a `RandomForestClassifier` / `ExtraTreesClassifier` via the bulk path
/// (`sklearn_bulk.cc:230-289`, SKL-01).
///
/// Binary classifiers are treated as multi-class with `n_classes=2`
/// (`sklearn_bulk.cc:243-244`). Metadata (hand-set per `:248-272`):
/// `task=kMultiClf`, `average_tree_output=true`, `num_class=n_classes`,
/// `leaf_vector_shape={n_targets, max_num_class}`,
/// `target_id=vec![-1; n_estimators]`, `class_id=vec![-1; n_estimators]`
/// (leaf-vector broadcast), `postprocessor="identity_multiclass"`,
/// `base_scores=vec![0.0; n_targets * max_num_class]`.
#[allow(clippy::too_many_arguments)]
pub fn load_random_forest_classifier(
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
) -> Result<Model, SklError> {
    let n_estimators = require_positive("n_estimators", n_estimators)?;
    let n_features = require_positive("n_features", n_features)?;
    let n_targets = require_positive("n_targets", n_targets)?;

    // n_classes must be one entry per target, each >= 1.
    if n_classes.len() != n_targets as usize {
        return Err(SklError::DimensionMismatch {
            tree: 0,
            field: "n_classes",
            expected: n_targets as usize,
            got: n_classes.len(),
        });
    }
    for &nc in n_classes {
        require_positive("n_classes", nc)?;
    }
    // max_num_class = *max_element(n_classes) (sklearn_bulk.cc:258-259). The
    // n_classes.len() == n_targets >= 1 check above guarantees a non-empty slice;
    // fall back to a typed error rather than panicking if that ever changes.
    let max_num_class = n_classes
        .iter()
        .copied()
        .max()
        .ok_or(SklError::InvalidScalar {
            field: "n_classes",
            value: 0,
            reason: "must be non-empty",
        })?;

    let trees = build_trees(
        n_estimators as usize,
        node_count,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        n_node_samples,
        weighted_n_node_samples,
        impurity,
        n_targets,
        max_num_class,
        true,
    )?;

    let leaf_vec_total = (n_targets as i64) * (max_num_class as i64);
    let leaf_vec_total = usize::try_from(leaf_vec_total).map_err(|_| SklError::InvalidScalar {
        field: "n_targets*max_num_class",
        value: leaf_vec_total,
        reason: "exceeds usize",
    })?;

    let metadata = BuilderMetadata {
        num_feature: n_features,
        task_type: TaskType::kMultiClf,
        average_tree_output: true,
        num_target: n_targets,
        // MEM-02: BuilderMetadata fields are SmallVec/CompactString — `.into()`.
        num_class: n_classes.to_vec().into(),
        leaf_vector_shape: vec![n_targets, max_num_class].into(),
        target_id: vec![-1; n_estimators as usize].into(),
        class_id: vec![-1; n_estimators as usize].into(),
        postprocessor: "identity_multiclass".into(),
        base_scores: vec![0.0; leaf_vec_total].into(),
        attributes: None,
    };
    Ok(bulk_to_model(trees, metadata))
}

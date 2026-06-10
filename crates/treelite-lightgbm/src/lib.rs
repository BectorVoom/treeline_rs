//! `treelite-lightgbm` — the LightGBM text-format model loader (LGB-01/LGB-03).
//!
//! Parses a LightGBM text model into a [`treelite_core::Model`] (always the
//! `F64` variant — LightGBM thresholds/leaf values are doubles, D-02/D-05),
//! porting the objective→postprocessor map, the negative-index leaf
//! re-numbering, and the missing-type `default_left` override verbatim from
//! upstream Treelite v4.7.0.
//!
//! The line-based parser and the typed `LGBModel`/`LGBTree` structs live in
//! [`parse`]; the `CanonicalObjective` alias collapse + objective map live in
//! [`objective`]; the typed error enum lives in [`error`]. This module owns the
//! converge-then-build path ([`load_lightgbm`]) that drives the **f64**
//! [`ModelBuilder`] (Plan 04-01).
//!
//! Ports `treelite-mainline/src/model_loader/lightgbm.cc` and
//! `.../detail/lightgbm.h`. Categorical bitset decode (LGB-02) is the next slice
//! (Plan 04-05); this slice loads numerical models and rejects categorical
//! splits with a typed error rather than mis-predicting.

use std::collections::VecDeque;

pub mod error;
pub mod objective;
pub mod parse;

pub use error::LgbError;

use parse::{LGBModel, LGBTree};
use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::{Model, Operator, TaskType};

/// `kCategoricalMask` (`lightgbm.cc:182`): decision_type bit 0 ⇒ categorical.
const CATEGORICAL_MASK: i8 = 1;
/// `kDefaultLeftMask` (`lightgbm.cc:182`): decision_type bit 1 ⇒ default-left.
const DEFAULT_LEFT_MASK: i8 = 2;
/// `MissingType::kNaN` (`lightgbm.cc:184`): the (decision_type >> 2) & 3 value
/// meaning "missing maps to NaN" (vs. kNone/kZero which map missing to 0.0).
const MISSING_TYPE_NAN: i8 = 2;

/// `(decision_type & mask) > 0` (`lightgbm.cc:202-204`).
fn get_decision_type(decision_type: i8, mask: i8) -> bool {
    (decision_type & mask) > 0
}

/// `(decision_type >> 2) & 3` (`lightgbm.cc:206-208`).
fn get_missing_type(decision_type: i8) -> i8 {
    (decision_type >> 2) & 3
}

/// Reject a model scalar that must be non-negative before it is cast to `usize`
/// (mirrors the XGBoost `require_non_negative`; a negative value cast to a huge
/// size aborts the process — WR-02/ERR-01).
fn require_non_negative(field: &'static str, value: i32) -> Result<i32, LgbError> {
    if value < 0 {
        return Err(LgbError::Parse {
            line: "global".to_string(),
            detail: format!("scalar {field:?} must be non-negative, got {value}"),
        });
    }
    Ok(value)
}

/// Derive the task type from `num_class` and the canonical objective
/// (`lightgbm.cc:417-440`).
fn task_type_for(num_class: i32, canonical: &str) -> Result<TaskType, LgbError> {
    if num_class > 1 {
        // Multi-class: objective must be a multiclass family (lightgbm.cc:425-426).
        if canonical != "multiclass" && canonical != "multiclassova" {
            return Err(LgbError::Parse {
                line: "global".to_string(),
                detail: format!(
                    "objective {canonical:?} is not multiclass/multiclassova but num_class={num_class} > 1"
                ),
            });
        }
        Ok(TaskType::kMultiClf)
    } else if canonical == "binary"
        || canonical == "cross_entropy"
        || canonical == "cross_entropy_lambda"
    {
        Ok(TaskType::kBinaryClf)
    } else if canonical == "lambdarank" || canonical == "rank_xendcg" {
        Ok(TaskType::kLearningToRank)
    } else {
        Ok(TaskType::kRegressor)
    }
}

/// Emit one LightGBM tree through the f64 [`ModelBuilder`], re-numbering the
/// negative-index leaf encoding into a clean depth-wise node sequence
/// (`lightgbm.cc:526-601`).
///
/// LightGBM distinguishes leaves from internal nodes with NEGATIVE child
/// indices: `left_child[i]`/`right_child[i]` ≥ 0 is an internal node index, and
/// a value `< 0` is leaf `!(value)` (i.e. `~value`, the bitwise complement). We
/// BFS the structure from the root, assigning fresh monotonic node ids (the
/// builder requires non-negative keys with resolvable children, D-08), seeding
/// `(-1, 1)` for a single-leaf tree and `(0, 1)` otherwise; `dfs_index` starts
/// at 1 and advances by 2 per internal node (one id per child).
fn build_tree(
    builder: &mut ModelBuilder,
    tree_idx: usize,
    tree: &LGBTree,
) -> Result<(), LgbError> {
    builder.start_tree()?;

    let num_leaves = tree.num_leaves;
    if num_leaves == 0 {
        // Upstream `continue`s past an empty tree WITHOUT closing it; but the
        // builder requires a non-empty tree per StartTree/EndTree. An empty tree
        // cannot occur in a well-formed LightGBM file (num_leaves >= 1), so treat
        // it as malformed rather than emit a degenerate tree.
        return Err(LgbError::Parse {
            line: format!("Tree {tree_idx}"),
            detail: "num_leaves == 0 (empty tree)".to_string(),
        });
    }

    // BFS queue of (old_node_id, new_node_id) pairs (lightgbm.cc:533-543).
    let mut queue: VecDeque<(i32, i32)> = VecDeque::new();
    let mut dfs_index: i32 = 1;
    if num_leaves == 1 {
        // A constant-value tree: a single root that is also a leaf.
        queue.push_front((-1, dfs_index));
    } else {
        queue.push_front((0, dfs_index));
    }
    dfs_index += 1;

    let n_leaf = num_leaves as usize;

    while let Some((old_node_id, new_node_id)) = queue.pop_front() {
        builder.start_node(new_node_id)?;

        if old_node_id < 0 {
            // Leaf: value is leaf_value[!old_node_id] (bitwise complement).
            let leaf_idx = (!old_node_id) as i64;
            if leaf_idx < 0 || (leaf_idx as usize) >= n_leaf {
                return Err(LgbError::LeafIndexOutOfRange {
                    tree: tree_idx,
                    index: leaf_idx.max(0) as usize,
                    num_leaves: n_leaf,
                });
            }
            builder.leaf_scalar_f64(tree.leaf_value[leaf_idx as usize])?;
            // leaf data count (lightgbm.cc:550-554).
            if !tree.leaf_count.is_empty() {
                let dc = tree.leaf_count[leaf_idx as usize];
                if dc < 0 {
                    return Err(LgbError::Parse {
                        line: format!("Tree {tree_idx}"),
                        detail: format!("negative leaf_count {dc}"),
                    });
                }
                builder.data_count(dc as u64)?;
            }
        } else {
            // Internal node. Validate the old index against the parsed arrays.
            let oi = old_node_id as usize;
            if oi >= tree.split_feature.len()
                || oi >= tree.decision_type.len()
                || oi >= tree.threshold.len()
                || oi >= tree.left_child.len()
                || oi >= tree.right_child.len()
            {
                return Err(LgbError::NodeIndexOutOfRange {
                    tree: tree_idx,
                    index: old_node_id as i64,
                    detail: "internal node index exceeds parsed array length",
                });
            }

            let split_index = tree.split_feature[oi];
            let decision_type = tree.decision_type[oi];
            let left_child_old = tree.left_child[oi];
            let left_child_new = dfs_index;
            dfs_index += 1;
            let right_child_old = tree.right_child[oi];
            let right_child_new = dfs_index;
            dfs_index += 1;

            if get_decision_type(decision_type, CATEGORICAL_MASK) {
                // Categorical split — LGB-02, deferred to Plan 04-05. Reject with
                // a typed error rather than silently mis-predicting.
                return Err(LgbError::Parse {
                    line: format!("Tree {tree_idx}"),
                    detail: "categorical split (LGB-02) not yet supported in this loader slice"
                        .to_string(),
                });
            }

            // Numerical split (lightgbm.cc:575-586).
            let threshold = tree.threshold[oi];
            let mut default_left = get_decision_type(decision_type, DEFAULT_LEFT_MASK);
            // Pitfall 3: when missing_type != kNaN, missing values map to 0.0, so
            // the default_left flag must be overridden (lightgbm.cc:579-584).
            let missing_value_to_zero = get_missing_type(decision_type) != MISSING_TYPE_NAN;
            if missing_value_to_zero {
                default_left = 0.0 <= threshold;
            }

            builder.numerical_test_f64(
                split_index,
                threshold,
                default_left,
                Operator::kLE, // LightGBM always uses <= (lightgbm.cc:585).
                left_child_new,
                right_child_new,
            )?;

            // internal data count (lightgbm.cc:588-592).
            if !tree.internal_count.is_empty() {
                let dc = tree.internal_count[oi];
                if dc < 0 {
                    return Err(LgbError::Parse {
                        line: format!("Tree {tree_idx}"),
                        detail: format!("negative internal_count {dc}"),
                    });
                }
                builder.data_count(dc as u64)?;
            }
            // split gain (lightgbm.cc:593-595).
            if !tree.split_gain.is_empty() {
                builder.gain(tree.split_gain[oi] as f64)?;
            }

            // Enqueue children at the FRONT (lightgbm.cc:596-597 emplace_front).
            queue.push_front((left_child_old, left_child_new));
            queue.push_front((right_child_old, right_child_new));
        }
        builder.end_node()?;
    }

    builder.end_tree()?;
    Ok(())
}

/// Load a LightGBM text-format model into a [`treelite_core::Model`] (F64).
///
/// Parses the `key=value` text via [`parse::parse_lightgbm`], canonicalizes the
/// objective and resolves its postprocessor + `sigmoid_alpha`
/// ([`objective::canonical_objective`] / [`objective::map_objective`]), fills the
/// `BuilderMetadata` (routing through the f64 builder, `class_id[i] = i %
/// num_class`, `average_tree_output` from the `average_output` key presence),
/// emits every tree through [`build_tree`] (with the negative-index leaf
/// re-numbering and the missing-type `default_left` override), commits, and
/// stamps `model.sigmoid_alpha`.
///
/// Malformed input (missing key, bad number, short array, unknown objective,
/// out-of-range leaf/node index, non-positive `sigmoid_alpha`) returns a typed
/// [`LgbError`] — never a panic or an out-of-bounds index (ERR-01, ASVS V5).
pub fn load_lightgbm(model_str: &str) -> Result<Model, LgbError> {
    let parsed: LGBModel = parse::parse_lightgbm(model_str)?;

    let num_class = require_non_negative("num_class", parsed.num_class)?;
    let max_feature_idx = parsed.max_feature_idx;
    let num_feature = max_feature_idx
        .checked_add(1)
        .ok_or_else(|| LgbError::Parse {
            line: "global".to_string(),
            detail: format!("max_feature_idx {max_feature_idx} overflows num_feature"),
        })?;
    let num_feature = require_non_negative("num_feature", num_feature)?;

    // Canonicalize the objective FIRST, then resolve the postprocessor.
    let canonical = objective::canonical_objective(&parsed.objective_name)?;
    let postproc = objective::map_objective(canonical, &parsed.objective_params)?;
    let task_type = task_type_for(num_class, canonical)?;

    let num_tree = parsed.trees.len();
    // class_id[i] = i % num_class (round-robin, lightgbm.cc:427-429). num_class
    // is >= 1 here (require_non_negative + multiclass needs > 1); guard the
    // modulo against a degenerate 0.
    let modulus = num_class.max(1);
    let class_id: Vec<i32> = (0..num_tree as i32).map(|i| i % modulus).collect();
    let target_id: Vec<i32> = vec![0; num_tree];

    // base_scores: num_class zeros (lightgbm.cc:523). num_target is 1 for
    // LightGBM (lightgbm.cc:518).
    let base_scores: Vec<f64> = vec![0.0; num_class.max(1) as usize];

    let metadata = BuilderMetadata {
        num_feature,
        task_type,
        average_tree_output: parsed.average_output, // average_output key presence.
        num_target: 1,
        num_class: vec![num_class],
        leaf_vector_shape: vec![1, 1],
        target_id,
        class_id,
        postprocessor: postproc.postprocessor.to_string(),
        base_scores,
        attributes: None,
    };

    let mut builder = ModelBuilder::new(metadata)?;
    for (i, tree) in parsed.trees.iter().enumerate() {
        build_tree(&mut builder, i, tree)?;
    }
    let mut model = builder.commit_model()?;

    // sigmoid_alpha is not carried by the builder metadata API; stamp it
    // post-commit (mirrors the XGBoost loader's sigmoid_alpha/ratio_c handling).
    model.sigmoid_alpha = postproc.sigmoid_alpha as f32;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VENDORED: &str = "../../treelite-mainline/tests/examples/deep_lightgbm/model.txt";

    fn vendored_text() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(VENDORED);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
    }

    #[test]
    fn loads_vendored_numerical_model_to_f64_no_builder_errors() {
        let model = load_lightgbm(&vendored_text()).expect("load deep_lightgbm");
        // Must be the F64 variant (D-02/D-05).
        assert!(
            matches!(model.variant, treelite_core::ModelVariant::F64(_)),
            "LightGBM must load into the F64 variant"
        );
        assert_eq!(model.num_feature, 1);
        assert_eq!(model.task_type, TaskType::kRegressor);
        assert_eq!(model.postprocessor, "identity");
        // One tree → class_id[0] = 0.
        assert_eq!(model.class_id, vec![0]);
    }

    #[test]
    fn unknown_objective_is_typed_error_not_panic() {
        let bad = "num_class=1\nmax_feature_idx=0\nobjective=not_real\nTree=0\nnum_leaves=1\nnum_cat=0\nleaf_value=1\n";
        assert!(matches!(
            load_lightgbm(bad),
            Err(LgbError::UnrecognizedObjective(_))
        ));
    }

    #[test]
    fn single_leaf_constant_tree_loads() {
        // num_leaves=1 → a constant-value tree (seeded (-1, 1)).
        let m = "num_class=1\nmax_feature_idx=0\nobjective=regression\nTree=0\nnum_leaves=1\nnum_cat=0\nleaf_value=7.5\n";
        let model = load_lightgbm(m).expect("load single-leaf tree");
        assert!(matches!(model.variant, treelite_core::ModelVariant::F64(_)));
    }
}

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

pub mod bitset;
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
                // Categorical split (lightgbm.cc:563-573). The node's `threshold`
                // field holds the CATEGORICAL INDEX (cat_idx), not a numeric
                // threshold: cat_idx = static_cast<int>(threshold[old_node_id]).
                let cat_idx_raw = tree.threshold[oi];
                // The cast truncates toward zero; reject a negative / non-integral
                // / out-of-range index BEFORE slicing cat_boundaries (T-04-10).
                if !(cat_idx_raw.is_finite() && cat_idx_raw >= 0.0) {
                    return Err(LgbError::Bitset {
                        tree: tree_idx,
                        detail: format!("categorical index {cat_idx_raw} is not a non-negative integer"),
                    });
                }
                let cat_idx = cat_idx_raw as usize; // truncates toward zero (static_cast<int>).
                // cat_boundaries has num_cat + 1 entries; need cat_idx and
                // cat_idx + 1 in range (the [begin, end) slice bounds).
                if cat_idx + 1 >= tree.cat_boundaries.len() {
                    return Err(LgbError::Bitset {
                        tree: tree_idx,
                        detail: format!(
                            "cat_idx {cat_idx} out of range (cat_boundaries has {} entries)",
                            tree.cat_boundaries.len()
                        ),
                    });
                }
                let begin = tree.cat_boundaries[cat_idx] as usize;
                let end = tree.cat_boundaries[cat_idx + 1] as usize;
                // parse.rs already validated monotonicity + total length, but
                // bounds-check the concrete slice once more (defense in depth).
                if begin > end || end > tree.cat_threshold.len() {
                    return Err(LgbError::Bitset {
                        tree: tree_idx,
                        detail: format!(
                            "categorical slice [{begin}, {end}) out of cat_threshold (len {})",
                            tree.cat_threshold.len()
                        ),
                    });
                }
                let bits = &tree.cat_threshold[begin..end];
                let categories = bitset::bitset_to_list(bits);
                // Categorical splits ignore the missing_type field: NaNs always
                // map to the RIGHT child, so default_left = false and
                // category_list_right_child = false (a category MATCH routes
                // LEFT, i.e. left_categories) — lightgbm.cc:569-573.
                builder.categorical_test(
                    split_index,
                    /* default_left = */ false,
                    &categories,
                    /* category_list_right_child = */ false,
                    left_child_new,
                    right_child_new,
                )?;

                // internal data count / split gain (same as the numerical path,
                // lightgbm.cc:588-595).
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
                if !tree.split_gain.is_empty() {
                    builder.gain(tree.split_gain[oi] as f64)?;
                }

                // Enqueue children at the FRONT (lightgbm.cc:596-597).
                queue.push_front((left_child_old, left_child_new));
                queue.push_front((right_child_old, right_child_new));
                builder.end_node()?;
                continue;
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
    let postproc = objective::map_objective(canonical, &parsed.objective_params, num_class)?;
    let task_type = task_type_for(num_class, canonical)?;

    let num_tree = parsed.trees.len();
    // class_id[i] = i % num_class (round-robin, lightgbm.cc:427-429). num_class
    // is >= 1 here (require_non_negative + multiclass needs > 1); guard the
    // modulo against a degenerate 0.
    let modulus = num_class.max(1);
    let class_id: Vec<i32> = (0..num_tree as i32).map(|i| i % modulus).collect();
    let target_id: Vec<i32> = vec![0; num_tree];

    // base_scores: exactly num_class zeros (lightgbm.cc:523), with NO clamp —
    // upstream builds `std::vector<double>(num_class_, 0.0)`. Tracking num_class
    // exactly (rather than `num_class.max(1)`) keeps base_scores.len() consistent
    // with `metadata.num_class == [num_class]` even for the degenerate
    // `num_class == 0` case, so the GTIL base-score add loop cannot desync from
    // the per-target class count. num_target is 1 for LightGBM (lightgbm.cc:518).
    let base_scores: Vec<f64> = vec![0.0; num_class as usize];

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

    /// A 2-leaf categorical tree: root is a categorical split on feature 0, with
    /// one categorical index (cat_idx=0). cat_boundaries=0 1 (one bitset word);
    /// cat_threshold=6 → bits 1 and 2 set → categories {1, 2}. left_child=-1
    /// (leaf 0, value 10), right_child=-2 (leaf 1, value 20).
    fn categorical_model_text() -> String {
        // decision_type=1 → categorical mask bit set. threshold=0 → cat_idx 0.
        "num_class=1\nmax_feature_idx=0\nobjective=regression\n\
         Tree=0\nnum_leaves=2\nnum_cat=1\n\
         split_feature=0\ndecision_type=1\nthreshold=0\n\
         left_child=-1\nright_child=-2\n\
         cat_boundaries=0 1\ncat_threshold=6\n\
         leaf_value=10 20\n"
            .to_string()
    }

    #[test]
    fn categorical_node_emits_categorical_test_default_left_false() {
        let model = load_lightgbm(&categorical_model_text()).expect("load categorical tree");
        let treelite_core::ModelVariant::F64(preset) = &model.variant else {
            panic!("LightGBM must load into the F64 variant");
        };
        let tree = &preset.trees[0];
        // Root node (id 0) is the categorical split.
        assert_eq!(
            tree.node_type(0),
            treelite_core::TreeNodeType::kCategoricalTestNode
        );
        // Categorical splits set default_left = false (NaN → right, LGB).
        assert!(!tree.default_left(0));
        // category_list_right_child = false: a MATCH routes to the left child.
        assert!(!tree.category_list_right_child(0));
        // The decoded category list from cat_threshold=6 (bits 1, 2) is {1, 2}.
        assert_eq!(tree.category_list(0), &[1u32, 2u32]);
    }

    #[test]
    fn categorical_index_out_of_range_is_typed_error_not_panic() {
        // cat_boundaries=0 1 has 2 entries → only cat_idx 0 is valid. threshold=5
        // → cat_idx 5 is out of range; must be a typed Bitset error, not a panic.
        let bad = "num_class=1\nmax_feature_idx=0\nobjective=regression\n\
                   Tree=0\nnum_leaves=2\nnum_cat=1\n\
                   split_feature=0\ndecision_type=1\nthreshold=5\n\
                   left_child=-1\nright_child=-2\n\
                   cat_boundaries=0 1\ncat_threshold=6\n\
                   leaf_value=10 20\n";
        assert!(matches!(load_lightgbm(bad), Err(LgbError::Bitset { .. })));
    }
}

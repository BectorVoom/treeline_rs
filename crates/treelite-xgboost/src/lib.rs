//! `treelite-xgboost` — the XGBoost-JSON model loader (Wave 2).
//!
//! Parses an XGBoost-JSON model into a [`treelite_core::Model`] (always the
//! `F32` variant — XGBoost-JSON only ever yields `<f32, f32>`), porting the
//! objective→postprocessor map and the version-gated f64 `base_score`→margin
//! transform verbatim from upstream Treelite v4.7.0.
//!
//! Ports `treelite-mainline/src/model_loader/detail/xgboost.{h,cc}` and
//! `.../xgboost_json/delegated_handler.cc`.

pub mod error;
pub mod objective;

pub use error::XgbError;
pub use objective::{
    get_postprocessor, prob_to_margin_exponential, prob_to_margin_sigmoid,
    transform_base_score_to_margin,
};

use serde::Deserialize;
use treelite_core::{
    Model, ModelPreset, ModelVariant, Operator, TaskType, Tree, TreeBuf, TreeNodeType,
};

// ---------------------------------------------------------------------------
// serde intermediate structs — the recognized XGBoost-JSON key subset.
//
// The recognized per-tree key list mirrors `delegated_handler.cc:484-490`; the
// learner/booster nesting mirrors the `LearnerHandler` hierarchy. Numeric
// scalar params in XGBoost-JSON are JSON *strings* (e.g. `"num_feature":"2"`,
// `"base_score":"2.5E-1"`, `"num_nodes":"3"`), so they are deserialized as
// `String` and parsed via `str::parse`. Parallel node arrays are real JSON
// arrays.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct XgbModelJson {
    learner: Learner,
    /// `[major, minor, patch]`; gates the base_score→margin transform.
    #[serde(default)]
    version: Vec<i32>,
}

#[derive(Deserialize)]
struct Learner {
    learner_model_param: LearnerModelParam,
    gradient_booster: GradientBooster,
    objective: Objective,
}

#[derive(Deserialize)]
struct LearnerModelParam {
    num_feature: String,
    num_class: String,
    num_target: String,
    base_score: String,
}

#[derive(Deserialize)]
struct GradientBooster {
    model: BoosterModel,
}

#[derive(Deserialize)]
struct BoosterModel {
    trees: Vec<RegTreeJson>,
    tree_info: Vec<i32>,
}

#[derive(Deserialize)]
struct Objective {
    name: String,
}

#[derive(Deserialize)]
struct RegTreeJson {
    tree_param: TreeParam,
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<i32>,
    /// 0 = numerical, 1 = categorical (Phase 1: all 0).
    split_type: Vec<i32>,
    split_conditions: Vec<f32>,
    /// XGBoost stores `default_left` as 0/1 integers.
    default_left: Vec<i32>,
}

#[derive(Deserialize)]
struct TreeParam {
    num_nodes: String,
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Parse a numeric scalar param that XGBoost-JSON stores as a string.
fn parse_scalar<T>(field: &'static str, raw: &str) -> Result<T, XgbError>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    raw.parse::<T>().map_err(|e| XgbError::ParseScalar {
        field,
        value: raw.to_string(),
        source: Box::new(e),
    })
}

/// Validate that a per-tree parallel array's length matches `num_nodes`.
///
/// Mirrors `delegated_handler.cc:411-432`: on mismatch return a typed
/// [`XgbError::DimensionMismatch`] instead of indexing out of bounds
/// (ERR-01, ASVS V5 input validation).
fn check_dim(
    tree: usize,
    field: &'static str,
    num_nodes: usize,
    got: usize,
) -> Result<(), XgbError> {
    if got != num_nodes {
        return Err(XgbError::DimensionMismatch {
            tree,
            field,
            expected: num_nodes,
            got,
        });
    }
    Ok(())
}

/// Build a single [`Tree<f32>`] from one parsed XGBoost-JSON tree.
///
/// Ports the per-node build loop (`delegated_handler.cc:435-479`): a node with
/// `left_children[i] == -1` is a scalar leaf (`size_leaf_vector <= 1`); every
/// other node is a numerical test (`Operator::kLT` always — XGBoost never uses
/// another operator). Leaf-vector and category columns stay empty.
fn build_tree(tree_idx: usize, t: &RegTreeJson) -> Result<Tree<f32>, XgbError> {
    let num_nodes: usize = parse_scalar("tree_param.num_nodes", &t.tree_param.num_nodes)?;

    // Validate every parallel array length == num_nodes before building.
    check_dim(tree_idx, "left_children", num_nodes, t.left_children.len())?;
    check_dim(tree_idx, "right_children", num_nodes, t.right_children.len())?;
    check_dim(tree_idx, "split_indices", num_nodes, t.split_indices.len())?;
    check_dim(tree_idx, "split_type", num_nodes, t.split_type.len())?;
    check_dim(tree_idx, "split_conditions", num_nodes, t.split_conditions.len())?;
    check_dim(tree_idx, "default_left", num_nodes, t.default_left.len())?;

    let mut node_type = Vec::with_capacity(num_nodes);
    let mut cleft = Vec::with_capacity(num_nodes);
    let mut cright = Vec::with_capacity(num_nodes);
    let mut split_index = Vec::with_capacity(num_nodes);
    let mut default_left = Vec::with_capacity(num_nodes);
    let mut leaf_value = Vec::with_capacity(num_nodes);
    let mut threshold = Vec::with_capacity(num_nodes);
    let mut cmp = Vec::with_capacity(num_nodes);

    for i in 0..num_nodes {
        if t.left_children[i] == -1 {
            // Scalar leaf output (size_leaf_vector <= 1 in Phase 1).
            node_type.push(TreeNodeType::kLeafNode);
            cleft.push(-1);
            cright.push(-1);
            split_index.push(0);
            default_left.push(false);
            leaf_value.push(t.split_conditions[i]);
            threshold.push(0.0_f32);
            cmp.push(Operator::kNone);
        } else {
            // Numerical internal node (XGBoost always uses kLT).
            node_type.push(TreeNodeType::kNumericalTestNode);
            cleft.push(t.left_children[i]);
            cright.push(t.right_children[i]);
            split_index.push(t.split_indices[i]);
            default_left.push(t.default_left[i] != 0);
            leaf_value.push(0.0_f32);
            threshold.push(t.split_conditions[i]);
            cmp.push(Operator::kLT);
        }
    }

    let mut tree = Tree::<f32>::new();
    tree.node_type = TreeBuf::from_owned(node_type);
    tree.cleft = TreeBuf::from_owned(cleft);
    tree.cright = TreeBuf::from_owned(cright);
    tree.split_index = TreeBuf::from_owned(split_index);
    tree.default_left = TreeBuf::from_owned(default_left);
    tree.leaf_value = TreeBuf::from_owned(leaf_value);
    tree.threshold = TreeBuf::from_owned(threshold);
    tree.cmp = TreeBuf::from_owned(cmp);
    tree.has_categorical_split = false;
    tree.num_nodes = num_nodes as i32;
    Ok(tree)
}

/// Load one XGBoost-JSON model into a [`treelite_core::Model`] (F32 variant).
///
/// Ports the loader leg of the walking skeleton: parse the recognized
/// XGBoost-JSON key subset, build each `Tree<f32>` per the per-node loop, then
/// finalize header metadata exactly as `LearnerHandler::EndObject`
/// (`delegated_handler.cc:811-903`). Malformed input (bad JSON, array-length
/// mismatch, unrecognized objective) returns a typed [`XgbError`] — never a
/// panic or an out-of-bounds index (ERR-01).
pub fn load_xgboost_json(json: &str) -> Result<Model, XgbError> {
    let parsed: XgbModelJson = serde_json::from_str(json)?;

    let lp = &parsed.learner.learner_model_param;
    let num_feature: i32 = parse_scalar("num_feature", &lp.num_feature)?;
    let num_class_param: i32 = parse_scalar("num_class", &lp.num_class)?;
    let num_target: i32 = parse_scalar("num_target", &lp.num_target)?;
    let base_score: f64 = parse_scalar("base_score", &lp.base_score)?;

    let objective = &parsed.learner.objective.name;
    let postprocessor = get_postprocessor(objective)?.to_string();

    let booster = &parsed.learner.gradient_booster.model;
    let num_tree = booster.trees.len();

    // Build the trees (F32 — XGBoost-JSON only ever yields <f32, f32>).
    let mut trees: Vec<Tree<f32>> = Vec::with_capacity(num_tree);
    for (i, t) in booster.trees.iter().enumerate() {
        trees.push(build_tree(i, t)?);
    }
    let variant = ModelVariant::F32(ModelPreset::new(trees));

    let mut model = Model::new(variant);
    model.num_feature = num_feature;
    model.num_target = num_target;
    model.average_tree_output = false; // hardcoded upstream (delegated_handler.cc:814).
    model.postprocessor = postprocessor.clone();
    model.sigmoid_alpha = 1.0;
    model.ratio_c = 1.0;
    model.attributes = String::new();
    model.leaf_vector_shape = vec![1, 1];

    // Header metadata finalize (delegated_handler.cc:847-872 — binary/regressor
    // branch, since num_class <= 1 for binary:logistic).
    if num_class_param > 1 {
        // Multi-class — not exercised by the Phase 1 fixture, but ported for
        // completeness of the branch (delegated_handler.cc:824-846).
        model.task_type = TaskType::kMultiClf;
        model.num_class = vec![num_class_param];
        model.target_id = vec![0; num_tree];
        model.class_id = booster.tree_info.clone();
    } else {
        model.task_type = if objective.starts_with("binary:") {
            TaskType::kBinaryClf
        } else if objective.starts_with("rank:") {
            TaskType::kLearningToRank
        } else {
            TaskType::kRegressor
        };
        model.num_class = vec![1; num_target as usize];
        model.class_id = vec![0; num_tree];
        // Grove per target: target_id[i] = tree_info[i].
        model.target_id = booster.tree_info.clone();
    }

    // Base scores in f64, then version-gated margin transform
    // (delegated_handler.cc:874-897). The transform fires when the version is
    // empty or version[0] >= 1; the fixture's version [4,7,0] fires it.
    let mut base_scores = vec![base_score];
    let need_transform = parsed.version.is_empty() || parsed.version[0] >= 1;
    if need_transform {
        for e in base_scores.iter_mut() {
            *e = transform_base_score_to_margin(&postprocessor, *e);
        }
    }
    model.base_scores = base_scores;

    Ok(model)
}

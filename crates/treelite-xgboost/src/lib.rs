//! `treelite-xgboost` — the XGBoost model loaders.
//!
//! Parses an XGBoost model into a [`treelite_core::Model`] (always the `F32`
//! variant — XGBoost only ever yields `<f32, f32>`), porting the
//! objective→postprocessor map and the version-gated f64 `base_score`→margin
//! transform verbatim from upstream Treelite v4.7.0.
//!
//! The recognized key set, the NaN/Inf mechanism (D-02), and the serde struct
//! family live in [`json`]; this module owns the shared
//! [`build_model_from_parsed`] convergence path (D-01) that the JSON loader (and
//! the UBJSON/legacy loaders in 03-03/03-04) all funnel through, plus the
//! per-tree builder emission that closes DEF-02-01 for byte fidelity (D-10).
//!
//! Ports `treelite-mainline/src/model_loader/detail/xgboost.{h,cc}` and
//! `.../xgboost_json/delegated_handler.cc`.

mod detect;
pub mod error;
mod json;
pub mod objective;
mod ubjson;

pub use detect::detect_xgboost_format;

/// Test-only surface exposing the crate-internal D-02 NaN/Inf primitives so the
/// integration test `tests/nan_inf.rs` can exercise them directly. Hidden from
/// docs; not part of the stable public API.
#[doc(hidden)]
pub mod test_support {
    pub use crate::json::{de_vec_f32_value, replace_nonfinite};
    pub use crate::ubjson::decode_ubjson;
}

pub use error::XgbError;
pub use objective::{
    get_postprocessor, parse_base_score, prob_to_margin_exponential, prob_to_margin_sigmoid,
    transform_base_score_to_margin,
};

use json::{RegTreeJson, XgbModelJson};
use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::{Model, Operator, TaskType};

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

/// Reject a model scalar that must be non-negative.
///
/// `num_target`/`num_feature`/`num_class` are later cast to `usize` (e.g.
/// `vec![1; num_target as usize]`); a negative value casts to a huge size and
/// aborts the process. Surface a typed [`XgbError::InvalidScalar`] instead
/// (WR-02, ERR-01).
fn require_non_negative(field: &'static str, value: i32) -> Result<i32, XgbError> {
    if value < 0 {
        return Err(XgbError::InvalidScalar { field, value });
    }
    Ok(value)
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

/// Build a single tree by driving the [`ModelBuilder`] (D-11), emitting the
/// per-node `sum_hess`/`gain` columns that close DEF-02-01 (D-10).
///
/// Ports the per-node build loop (`delegated_handler.cc:435-479`) onto the
/// validated builder's fluent API instead of hand-assembling `Tree` columns: a
/// node with `left_children[i] == -1` is a scalar leaf (`leaf_scalar`, since
/// `size_leaf_vector <= 1` in the verify-narrow path); every other node is a
/// numerical test (`numerical_test` with `Operator::kLT` always — XGBoost never
/// uses another operator). Node `i` is keyed by `i`, so the XGBoost child
/// indices map 1:1 onto builder child keys.
///
/// D-10 byte-fidelity (RESEARCH §DEF-02-01 table): `sum_hess` is set on EVERY
/// node from `sum_hessian` (triggers builder CR-02 `any_sum_hess` → the column
/// is emitted at length `num_nodes`); `gain` is set on INTERNAL nodes only from
/// `loss_changes`; `data_count` is intentionally NOT set (upstream leaves it
/// empty). Each stat array is dimension-checked before any builder emission.
fn build_tree(
    builder: &mut ModelBuilder,
    tree_idx: usize,
    t: &RegTreeJson,
) -> Result<(), XgbError> {
    let num_nodes: usize = parse_scalar("tree_param.num_nodes", &t.tree_param.num_nodes)?;

    // Validate every parallel array length == num_nodes before building.
    check_dim(tree_idx, "left_children", num_nodes, t.left_children.len())?;
    check_dim(
        tree_idx,
        "right_children",
        num_nodes,
        t.right_children.len(),
    )?;
    check_dim(tree_idx, "split_indices", num_nodes, t.split_indices.len())?;
    check_dim(tree_idx, "split_type", num_nodes, t.split_type.len())?;
    check_dim(
        tree_idx,
        "split_conditions",
        num_nodes,
        t.split_conditions.len(),
    )?;
    check_dim(tree_idx, "default_left", num_nodes, t.default_left.len())?;
    // D-10 stat arrays: present in the recognized key set; validate before use.
    check_dim(tree_idx, "sum_hessian", num_nodes, t.sum_hessian.len())?;
    check_dim(tree_idx, "loss_changes", num_nodes, t.loss_changes.len())?;

    builder.start_tree()?;
    for i in 0..num_nodes {
        builder.start_node(i as i32)?;
        if t.left_children[i] == -1 {
            // Scalar leaf output (size_leaf_vector <= 1 in the verify-narrow path).
            builder.leaf_scalar(t.split_conditions[i])?;
        } else {
            // Numerical internal node (XGBoost always uses kLT).
            builder.numerical_test(
                t.split_indices[i],
                t.split_conditions[i],
                t.default_left[i] != 0,
                Operator::kLT,
                t.left_children[i],
                t.right_children[i],
            )?;
            // gain on internal nodes only (delegated_handler.cc gain branch).
            builder.gain(t.loss_changes[i] as f64)?;
        }
        // sum_hess on EVERY node (delegated_handler.cc:471-477) — triggers the
        // CR-02 `any_sum_hess` column emission for byte fidelity (D-10).
        builder.sum_hess(t.sum_hessian[i] as f64)?;
        builder.end_node()?;
    }
    builder.end_tree()?;
    Ok(())
}

/// The shared convergence path (D-01): finalize header metadata, drive the
/// [`ModelBuilder`], and commit, given an already-parsed [`XgbModelJson`].
///
/// `load_xgboost_json` funnels here after `replace_nonfinite` + `from_str`; the
/// UBJSON loader (03-03, via `serde_json::from_value`) and the legacy loader
/// (03-04, by filling the same logical fields) reuse this exact path so all
/// three formats produce an identical `Model` → identical v5 bytes (D-10).
///
/// Finalizes header metadata exactly as `LearnerHandler::EndObject`
/// (`delegated_handler.cc:811-903`), applies the version-gated f64 `base_score`
/// margin transform via [`parse_base_score`] (XGB-05, handling BOTH the scalar
/// and the vector form), and passes `attributes: None` so `commit_model`
/// defaults the serialized attributes to `"{}"` matching upstream (D-10).
pub(crate) fn build_model_from_parsed(parsed: XgbModelJson) -> Result<Model, XgbError> {
    let lp = &parsed.learner.learner_model_param;
    let num_feature: i32 =
        require_non_negative("num_feature", parse_scalar("num_feature", &lp.num_feature)?)?;
    let num_class_param: i32 =
        require_non_negative("num_class", parse_scalar("num_class", &lp.num_class)?)?;
    let num_target: i32 =
        require_non_negative("num_target", parse_scalar("num_target", &lp.num_target)?)?;

    let objective = &parsed.learner.objective.name;
    let postprocessor = get_postprocessor(objective)?.to_string();

    let booster = &parsed.learner.gradient_booster.model;
    let num_tree = booster.trees.len();

    // Header metadata finalize (delegated_handler.cc:824-872). `num_class` on the
    // Model is the per-class count vector; `expand_to` for base_scores is the
    // product `num_target * effective_num_class` (effective class count is
    // `max(num_class_param, 1)` — see the binary/regressor branch below).
    let (task_type, num_class, target_id, class_id, effective_num_class) = if num_class_param > 1 {
        // Multi-class — parse-wide (D-04); not exercised by the verify-narrow
        // fixture but ported for completeness (delegated_handler.cc:824-846).
        (
            TaskType::kMultiClf,
            vec![num_class_param],
            vec![0; num_tree],
            booster.tree_info.clone(),
            num_class_param,
        )
    } else {
        let task_type = if objective.starts_with("binary:") {
            TaskType::kBinaryClf
        } else if objective.starts_with("rank:") {
            TaskType::kLearningToRank
        } else {
            TaskType::kRegressor
        };
        (
            task_type,
            vec![1; num_target as usize],
            // Grove per target: target_id[i] = tree_info[i].
            booster.tree_info.clone(),
            vec![0; num_tree],
            1,
        )
    };

    // Base scores: parse the scalar OR vector form, expand to
    // `num_target * effective_num_class`, then apply the version-gated f64
    // margin transform element-wise (XGB-05). The transform fires when the
    // version is empty or version[0] >= 1; the fixture's version [3,2,0] fires
    // it (a no-op for base_score=0.5 since sigmoid(0.5)=0 margin).
    let expand_to = (num_target as i64) * (effective_num_class as i64);
    let expand_to = usize::try_from(expand_to).map_err(|_| XgbError::InvalidScalar {
        field: "num_target*num_class",
        value: -1,
    })?;
    let apply_transform = parsed.version.is_empty() || parsed.version[0] >= 1;
    let base_scores = parse_base_score(&lp.base_score, expand_to, &postprocessor, apply_transform)?;

    // Drive the validated builder (D-11). `attributes: None` makes
    // `commit_model` default the serialized attributes to `"{}"`, matching
    // upstream's serialized attributes (D-10 / DEF-02-01 close — RESEARCH line
    // 385). `sigmoid_alpha`/`ratio_c` are set after `commit_model` since the
    // builder metadata API does not cover them; they keep their `1.0` values.
    let metadata = BuilderMetadata {
        num_feature,
        task_type,
        average_tree_output: false, // hardcoded upstream (delegated_handler.cc:814).
        num_target,
        num_class,
        leaf_vector_shape: vec![1, 1],
        target_id,
        class_id,
        postprocessor,
        base_scores,
        attributes: None,
    };

    let mut builder = ModelBuilder::new(metadata)?;
    // Build the trees (F32 — XGBoost only ever yields <f32, f32>).
    for (i, t) in booster.trees.iter().enumerate() {
        build_tree(&mut builder, i, t)?;
    }
    let mut model = builder.commit_model()?;

    // Header fields the builder metadata API does not carry, preserved verbatim
    // from the upstream finalize (delegated_handler.cc:814,818).
    model.sigmoid_alpha = 1.0;
    model.ratio_c = 1.0;

    Ok(model)
}

/// Load one XGBoost-JSON model into a [`treelite_core::Model`] (F32 variant).
///
/// Applies the D-02 NaN/Inf pre-lex ([`json::replace_nonfinite`]) so bare
/// `NaN`/`Infinity`/`-Infinity` literals round-trip into f32 thresholds/leaf
/// values, parses the recognized XGBoost-JSON key set into [`XgbModelJson`],
/// then funnels through the shared [`build_model_from_parsed`] path. Malformed
/// input (bad JSON, array-length mismatch, unrecognized objective) returns a
/// typed [`XgbError`] — never a panic or an out-of-bounds index (ERR-01).
pub fn load_xgboost_json(json: &str) -> Result<Model, XgbError> {
    let prelexed = json::replace_nonfinite(json);
    let parsed: XgbModelJson = serde_json::from_str(&prelexed)?;
    build_model_from_parsed(parsed)
}

/// Load one XGBoost-UBJSON model into a [`treelite_core::Model`] (F32 variant).
///
/// Decodes the UBJSON byte stream via the hand-rolled tag decoder
/// ([`ubjson::decode_ubjson`], D-03) into a `serde_json::Value` — emitting the
/// SAME `"@NaN@"`/`"@Inf@"`/`"@-Inf@"` sentinel strings the JSON pre-lexer
/// produces for non-finite floats — then `serde_json::from_value` into the SAME
/// [`XgbModelJson`] structs and the SAME [`json::de_f32`] adapter the JSON path
/// uses, and funnels through the shared [`build_model_from_parsed`] convergence
/// path (D-01). The result is therefore the IDENTICAL [`Model`] a JSON load of
/// the same logical model produces — byte-faithful to the single upstream golden
/// blob (D-10) and predicting within 1e-5.
///
/// XGBoost stores scalar learner/tree params as UBJSON `S` strings (exactly as
/// JSON stores them as JSON strings), so they deserialize into the `String`
/// fields of [`XgbModelJson`] with no special-casing. A malformed stream
/// (unknown tag, truncation, oversized `$`/`#` count) returns a typed
/// [`XgbError::Ubjson`] rather than a panic or an OOM (ASVS V5, ERR-01).
pub fn load_xgboost_ubjson(bytes: &[u8]) -> Result<Model, XgbError> {
    let value = ubjson::decode_ubjson(bytes)?;
    let parsed: XgbModelJson = serde_json::from_value(value)?;
    build_model_from_parsed(parsed)
}

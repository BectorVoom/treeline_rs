//! Widened XGBoost-JSON serde struct family + the D-02 NaN/Inf mechanism.
//!
//! This module is the convergence point of every XGBoost format (D-01): the
//! JSON path applies [`replace_nonfinite`] then `serde_json::from_str` into the
//! [`XgbModelJson`] struct family, and the UBJSON path (03-03) emits the same
//! sentinel strings from its tag decoder so it lands on the SAME structs and the
//! SAME [`de_f32`] adapter. Legacy (03-04) fills the equivalent logical fields
//! directly.
//!
//! The recognized key set is ported verbatim from the upstream
//! `is_recognized_key` methods in
//! `treelite-mainline/src/model_loader/detail/xgboost_json/delegated_handler.cc`
//! (RegTreeHandler ~484-491, TreeParamHandler ~331-334, GBTreeModelHandler
//! ~548-551, GradientBoosterHandler ~721-723, LearnerParamHandler ~781-784,
//! LearnerHandler ~916-919, XGBoostModelHandler ~963-965). Every parse-wide
//! field (D-04) is carried so the structs are future-proof, but its *use* is
//! gated behind leaf-vector/categorical/multiclass branches so today's
//! verify-narrow numerical path is unchanged.
//!
//! ## Scalar-as-string (load-bearing)
//!
//! XGBoost stores numeric learner/tree scalars as JSON *strings*
//! (`"num_feature":"4"`, `"base_score":"[5E-1]"`, `"num_nodes":"15"`), even in
//! UBJSON (`S` tag). They are deserialized as `String` and parsed via
//! `str::parse`. Parallel node arrays (`split_conditions`, `left_children`, …)
//! are real JSON arrays.

use serde::Deserialize;
use serde::de;

// ---------------------------------------------------------------------------
// D-02: string-safe NaN/Inf pre-lex + the de_f32 / de_vec_f32 adapters.
//
// Ported verbatim from RESEARCH §Code Examples (lines 451-503), validated
// end-to-end there. The JSON loader applies `replace_nonfinite` to the input
// string before `serde_json::from_str`; the UBJSON decoder (03-03) emits the
// same sentinel strings directly. Both converge on `de_f32`.
// ---------------------------------------------------------------------------

/// Rewrite bare `NaN` / `Infinity` / `-Infinity` in **value position** to the
/// sentinel strings `"@NaN@"` / `"@Inf@"` / `"@-Inf@"`, leaving string contents
/// BYTE-UNCHANGED (RESEARCH Pitfall 2 — string-safety).
///
/// `serde_json` rejects the bare non-finite literals (strict by design) and also
/// rejects out-of-range numeric substitutes like `1e400` ("number out of range",
/// RESEARCH Pitfall 1), so the only safe recovery is a sentinel STRING paired
/// with the [`de_f32`] adapter. The scanner tracks in-string state (toggling on
/// an unescaped `"`, honoring `\` escapes) so a `"NaN_count"` attribute or a
/// `"has Infinity inside"` string is never corrupted. `-Infinity` is matched
/// BEFORE `Infinity` BEFORE `NaN`.
pub fn replace_nonfinite(input: &str) -> String {
    let b = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let (mut i, mut in_str, mut escaped) = (0usize, false, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c as char);
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                out.push('"');
                i += 1;
            }
            _ if input[i..].starts_with("-Infinity") => {
                out.push_str("\"@-Inf@\"");
                i += 9;
            }
            _ if input[i..].starts_with("Infinity") => {
                out.push_str("\"@Inf@\"");
                i += 8;
            }
            _ if input[i..].starts_with("NaN") => {
                out.push_str("\"@NaN@\"");
                i += 3;
            }
            _ => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Deserialize a single `f32`, recovering the D-02 sentinel strings.
///
/// Accepts a JSON number (cast to `f32`) OR one of the three sentinel strings
/// `"@NaN@"` / `"@Inf@"` / `"@-Inf@"` emitted by [`replace_nonfinite`] (JSON) or
/// the UBJSON decoder (03-03). Any other string is parsed as an `f32` so a
/// stray numeric-as-string still round-trips. Mirrors the harness `NanF32`
/// visitor structure (`harness/src/lib.rs:39-69`).
pub(crate) fn de_f32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = f32;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an f32 or a NaN/Inf sentinel string")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f32, E> {
            Ok(v as f32)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f32, E> {
            Ok(v as f32)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f32, E> {
            Ok(v as f32)
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f32, E> {
            match v {
                "@NaN@" => Ok(f32::NAN),
                "@Inf@" => Ok(f32::INFINITY),
                "@-Inf@" => Ok(f32::NEG_INFINITY),
                other => other.parse().map_err(de::Error::custom),
            }
        }
    }
    d.deserialize_any(V)
}

/// A `Vec<f32>` newtype whose elements are recovered through [`de_f32`], used by
/// `objective::parse_base_score` to parse the vector `base_score` string while
/// honoring the D-02 NaN/Inf sentinels (XGB-05).
pub(crate) struct BaseScoreVec(pub(crate) Vec<f32>);

impl<'de> Deserialize<'de> for BaseScoreVec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        de_vec_f32(d).map(BaseScoreVec)
    }
}

/// Deserialize a `Vec<f32>`, routing every element through [`de_f32`] so
/// embedded NaN/Inf sentinels are recovered (D-02). Attached via
/// `#[serde(deserialize_with = "de_vec_f32")]` to every `Vec<f32>` field
/// XGBoost may fill with non-finite values.
pub(crate) fn de_vec_f32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<f32>, D::Error> {
    struct Wrap(f32);
    impl<'de> Deserialize<'de> for Wrap {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            de_f32(d).map(Wrap)
        }
    }
    let v = Vec::<Wrap>::deserialize(d)?;
    Ok(v.into_iter().map(|w| w.0).collect())
}

/// Test-only helper: deserialize a JSON array string (after `replace_nonfinite`)
/// into a `Vec<f32>` through the [`de_vec_f32`] sentinel adapter. Lets the
/// integration test exercise the D-02 sentinel recovery end-to-end.
pub fn de_vec_f32_value(prelexed_json_array: &str) -> Result<Vec<f32>, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_str(prelexed_json_array);
    de_vec_f32(&mut de)
}

// ---------------------------------------------------------------------------
// The recognized XGBoost-JSON key set (parse-wide, D-04). All three formats
// converge on these structs (D-01).
// ---------------------------------------------------------------------------

/// `XGBoostModelHandler` (`delegated_handler.cc:963-965`): `version`, `learner`.
/// `Config` / `Model` wrapper keys are ignored.
#[derive(Deserialize)]
pub(crate) struct XgbModelJson {
    pub(crate) learner: Learner,
    /// `[major, minor, patch]`; gates the base_score→margin transform.
    #[serde(default)]
    pub(crate) version: Vec<i32>,
}

/// `LearnerHandler` (`:916-919`): `learner_model_param`, `gradient_booster`,
/// `objective`. `attributes` / `feature_names` / `feature_types` are ignored.
#[derive(Deserialize)]
pub(crate) struct Learner {
    pub(crate) learner_model_param: LearnerModelParam,
    pub(crate) gradient_booster: GradientBooster,
    pub(crate) objective: Objective,
}

/// `LearnerParamHandler` (`:781-784`): `num_target`, `base_score`, `num_class`,
/// `num_feature`, `boost_from_average`. All stored as JSON strings.
#[derive(Deserialize)]
pub(crate) struct LearnerModelParam {
    pub(crate) num_feature: String,
    pub(crate) num_class: String,
    pub(crate) num_target: String,
    /// Scalar (`"5E-1"`) OR vector (`"[5E-1]"`) form — handled in
    /// `objective::parse_base_score` (XGB-05).
    pub(crate) base_score: String,
    /// Parse-wide (D-04): recognized but not used by the verify-narrow path.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) boost_from_average: Option<String>,
}

/// `GradientBoosterHandler` (`:721-723`): `name`, `model`, plus the DART
/// `gbtree` nesting and `weight_drop` (parse-only this phase).
#[derive(Deserialize)]
pub(crate) struct GradientBooster {
    pub(crate) model: BoosterModel,
    /// Parse-wide (D-04): DART nests a `gbtree` here; recognized but unused.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) gbtree: Option<serde_json::Value>,
    /// Parse-wide (D-04): DART leaf-scaling weights; recognized but unused.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) weight_drop: Option<Vec<f32>>,
}

/// `GBTreeModelHandler` (`:548-551`): `trees`, `tree_info`, plus `cats`
/// (categorical, parse-only). `gbtree_model_param` / `iteration_indptr` ignored.
#[derive(Deserialize)]
pub(crate) struct BoosterModel {
    pub(crate) trees: Vec<RegTreeJson>,
    #[serde(default)]
    pub(crate) tree_info: Vec<i32>,
    /// Parse-wide (D-04): categorical encoding container; recognized but unused.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) cats: Option<serde_json::Value>,
}

/// `ObjectiveHandler` (`:753-758`): `name` + 9 ignored `*_param` keys.
#[derive(Deserialize)]
pub(crate) struct Objective {
    pub(crate) name: String,
}

/// `RegTreeHandler` (`:484-491`): the full per-tree recognized key set.
///
/// `loss_changes` (→ gain) and `sum_hessian` (→ sum_hess) are REQUIRED for the
/// D-10 byte-fidelity close (DEF-02-01). The categorical and base/leaf-weight
/// fields are parse-wide (D-04): recognized so the structs are future-proof,
/// but their use is gated behind categorical/leaf-vector branches not exercised
/// by the verify-narrow numerical path. `leaf_child_counts`, `parents`, `id`
/// are recognized-but-ignored upstream and need no field here.
#[derive(Deserialize)]
pub(crate) struct RegTreeJson {
    pub(crate) tree_param: TreeParam,
    pub(crate) left_children: Vec<i32>,
    pub(crate) right_children: Vec<i32>,
    pub(crate) split_indices: Vec<i32>,
    /// 0 = numerical, 1 = categorical (verify-narrow: all 0).
    pub(crate) split_type: Vec<i32>,
    #[serde(deserialize_with = "de_vec_f32")]
    pub(crate) split_conditions: Vec<f32>,
    /// XGBoost stores `default_left` as 0/1 integers.
    pub(crate) default_left: Vec<i32>,
    /// → builder `gain` on internal nodes (D-10). May carry NaN/Inf.
    #[serde(default, deserialize_with = "de_vec_f32")]
    pub(crate) loss_changes: Vec<f32>,
    /// → builder `sum_hess` on every node (D-10). May carry NaN/Inf.
    #[serde(default, deserialize_with = "de_vec_f32")]
    pub(crate) sum_hessian: Vec<f32>,
    /// Parse-wide (D-04): per-node base weights; recognized but unused.
    #[serde(default, deserialize_with = "de_vec_f32")]
    #[allow(dead_code)]
    pub(crate) base_weights: Vec<f32>,
    /// Parse-wide (D-04): leaf-vector path; recognized but unused.
    #[serde(default, deserialize_with = "de_vec_f32")]
    #[allow(dead_code)]
    pub(crate) leaf_weights: Vec<f32>,
    /// Parse-wide (D-04): categorical split metadata; recognized but unused.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) categories_segments: Vec<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) categories_sizes: Vec<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) categories_nodes: Vec<i32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) categories: Vec<i32>,
}

/// `TreeParamHandler` (`:331-334`): `num_feature`, `num_nodes`,
/// `size_leaf_vector`. `num_deleted` is recognized-but-ignored upstream.
#[derive(Deserialize)]
pub(crate) struct TreeParam {
    pub(crate) num_nodes: String,
    /// Parse-wide (D-04): per-tree feature count; recognized but unused here.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) num_feature: Option<String>,
    /// Parse-wide (D-04): leaf-vector size; recognized but unused here.
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) size_leaf_vector: Option<String>,
}

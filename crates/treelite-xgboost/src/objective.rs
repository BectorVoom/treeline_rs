//! Objective→postprocessor map and the f64 `base_score`→margin transform.
//!
//! Ported verbatim from upstream Treelite v4.7.0:
//! - `treelite-mainline/src/model_loader/detail/xgboost.cc:28-60`
//!   (`GetPostProcessor`, `TransformBaseScoreToMargin`)
//! - `treelite-mainline/src/model_loader/detail/xgboost.h:16-23`
//!   (`ProbToMargin::Sigmoid` / `ProbToMargin::Exponential`)
//!
//! **The transform MUST stay in f64 — never single precision.** This is the
//! #1 silent 1e-5 equivalence break: the sigmoid margin `-ln(1/p - 1)` and the
//! exponential margin `ln(p)` are both computed in `f64`, and `base_scores`
//! is an f64 column on `Model`. Doing this in single precision shifts the last
//! ULPs past the 1e-5 fidelity target (RESEARCH §Pitfall 2/3).

use crate::error::XgbError;

/// Map an XGBoost objective name to its Treelite postprocessor name.
///
/// Ports `GetPostProcessor` (`xgboost.cc:28-50`) verbatim, including the exact
/// objective groupings. An unrecognized objective is upstream a
/// `TREELITE_LOG(FATAL)`; here it returns `Err(XgbError::UnrecognizedObjective)`
/// (ERR-01) — never a panic.
pub fn get_postprocessor(objective_name: &str) -> Result<&'static str, XgbError> {
    // Exponential-postprocessor objectives (xgboost.cc:29-30).
    const EXPONENTIAL: &[&str] = &[
        "count:poisson",
        "reg:gamma",
        "reg:tweedie",
        "survival:cox",
        "survival:aft",
    ];
    match objective_name {
        "multi:softmax" | "multi:softprob" => Ok("softmax"),
        "reg:logistic" | "binary:logistic" => Ok("sigmoid"),
        _ if EXPONENTIAL.contains(&objective_name) => Ok("exponential"),
        "binary:hinge" => Ok("hinge"),
        // Identity set (xgboost.cc:41-45).
        "reg:squarederror"
        | "reg:linear"
        | "reg:squaredlogerror"
        | "reg:pseudohubererror"
        | "binary:logitraw"
        | "rank:pairwise"
        | "rank:ndcg"
        | "rank:map" => Ok("identity"),
        other => Err(XgbError::UnrecognizedObjective(other.to_string())),
    }
}

/// `ProbToMargin::Sigmoid` (`xgboost.h:17-19`): `-ln(1/p - 1)`, in f64.
pub fn prob_to_margin_sigmoid(base_score: f64) -> f64 {
    -((1.0_f64 / base_score) - 1.0).ln()
}

/// `ProbToMargin::Exponential` (`xgboost.h:20-22`): `ln(p)`, in f64.
pub fn prob_to_margin_exponential(base_score: f64) -> f64 {
    base_score.ln()
}

/// Transform a probability `base_score` into a margin score, dispatching on the
/// postprocessor name. Ports `TransformBaseScoreToMargin` (`xgboost.cc:52-60`):
/// `"sigmoid"` and `"exponential"` apply the corresponding inverse-link, every
/// other postprocessor passes the value through unchanged. f64 throughout.
pub fn transform_base_score_to_margin(postprocessor: &str, base_score: f64) -> f64 {
    match postprocessor {
        "sigmoid" => prob_to_margin_sigmoid(base_score),
        "exponential" => prob_to_margin_exponential(base_score),
        _ => base_score,
    }
}

/// Parse the XGBoost `base_score` string (scalar OR vector form) into the
/// `base_scores` column, applying the version-gated f64 margin transform
/// element-wise (XGB-05).
///
/// Ports `ParseBaseScore` (`xgboost.cc:62-79`) + the
/// `delegated_handler.cc:877-889` scalar/vector branch:
/// - **Scalar form** (XGBoost <3.1, e.g. `"5E-1"`): parse one float, fill it
///   across `expand_to` (= `num_target * num_class`) entries.
/// - **Vector form** (XGBoost 3.1+, e.g. `"[5E-1]"` or `"[0.1, 0.2]"`): the
///   string itself is a JSON array of floats; parse it (routing any embedded
///   `NaN`/`Infinity`/`-Infinity` through the same D-02 sentinel mechanism), and
///   its length MUST equal `expand_to` — a mismatch is a typed
///   [`XgbError::BaseScoreShape`], never a silent truncation (T-03-V04).
///
/// Each element is parsed as `f32` then cast to **f64 BEFORE** the transform
/// (RESEARCH Pitfall 3 — doing the sigmoid/exponential margin in f32 is the #1
/// silent 1e-5 break). When `apply_transform` is `true`, every element is run
/// through [`transform_base_score_to_margin`]; the version gate
/// (`version.is_empty() || version[0] >= 1`) is decided by the caller.
pub fn parse_base_score(
    raw: &str,
    expand_to: usize,
    postprocessor: &str,
    apply_transform: bool,
) -> Result<Vec<f64>, XgbError> {
    let trimmed = raw.trim_start();
    let mut scores: Vec<f64> = if trimmed.starts_with('[') {
        // Vector form: the string is a JSON array of floats. Route through the
        // D-02 sentinel mechanism so an embedded NaN/Inf parses (upstream uses
        // `kParseNanAndInfFlag`); recover each element via `de_vec_f32`.
        let prelexed = crate::json::replace_nonfinite(raw);
        let elems: Vec<f32> = serde_json::from_str::<crate::json::BaseScoreVec>(&prelexed)
            .map_err(XgbError::Json)?
            .0;
        if elems.len() != expand_to {
            return Err(XgbError::BaseScoreShape {
                expected: expand_to,
                got: elems.len(),
            });
        }
        // Cast f32 → f64 BEFORE the transform (Pitfall 3).
        elems.into_iter().map(|e| e as f64).collect()
    } else {
        // Scalar form: parse one f32, cast to f64, fill across expand_to.
        let scalar: f32 = raw.parse().map_err(|e: std::num::ParseFloatError| {
            XgbError::ParseScalar {
                field: "base_score",
                value: raw.to_string(),
                source: Box::new(e),
            }
        })?;
        vec![scalar as f64; expand_to]
    };

    if apply_transform {
        for e in scores.iter_mut() {
            *e = transform_base_score_to_margin(postprocessor, *e);
        }
    }
    Ok(scores)
}

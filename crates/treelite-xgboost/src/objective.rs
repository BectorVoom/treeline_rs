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

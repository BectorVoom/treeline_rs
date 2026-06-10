//! `CanonicalObjective` alias-collapse + the objective→postprocessor map.
//!
//! Ported verbatim from upstream Treelite v4.7.0:
//! - `treelite-mainline/src/model_loader/detail/lightgbm.h:26-57`
//!   (`CanonicalObjective` alias table)
//! - `treelite-mainline/src/model_loader/lightgbm.cc:442-515`
//!   (the objective→`PostProcessorFunc` switch, including the `sigmoid:<a>` parse
//!   and the `sqrt`→`signed_square` regression branch)
//!
//! The canonicalization MUST run BEFORE the map (a missing alias would route the
//! objective to the wrong postprocessor — e.g. `l2_root` must collapse to
//! `regression` so it maps to `identity`, not fall through to "unknown").

use crate::error::LgbError;

/// Canonicalize the name of an objective function (`lightgbm.h:26-57`).
///
/// Many LightGBM objectives have aliases (`l2`/`mse`/`l2_root`/… → `regression`).
/// This collapses each alias group to its canonical spelling so the downstream
/// [`map_objective`] match only ever sees canonical names. An unknown name is an
/// upstream `TREELITE_LOG(FATAL)`; here it returns
/// [`LgbError::UnrecognizedObjective`].
pub fn canonical_objective(obj_name: &str) -> Result<&'static str, LgbError> {
    match obj_name {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" | "l2_root"
        | "root_mean_squared_error" | "rmse" => Ok("regression"),
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => Ok("regression_l1"),
        "mape" | "mean_absolute_percentage_error" => Ok("mape"),
        "multiclass" | "softmax" => Ok("multiclass"),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => Ok("multiclassova"),
        "cross_entropy" | "xentropy" => Ok("cross_entropy"),
        "cross_entropy_lambda" | "xentlambda" => Ok("cross_entropy_lambda"),
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => Ok("rank_xendcg"),
        // These objectives have no aliases (lightgbm.h:48-52).
        "huber" | "fair" | "poisson" | "quantile" | "gamma" | "tweedie" | "binary"
        | "lambdarank" | "custom" => {
            // Return a `&'static str` matching the input. The set is closed, so a
            // small match keeps the borrow `'static` without leaking.
            Ok(match obj_name {
                "huber" => "huber",
                "fair" => "fair",
                "poisson" => "poisson",
                "quantile" => "quantile",
                "gamma" => "gamma",
                "tweedie" => "tweedie",
                "binary" => "binary",
                "lambdarank" => "lambdarank",
                "custom" => "custom",
                _ => unreachable!(),
            })
        }
        other => Err(LgbError::UnrecognizedObjective(other.to_string())),
    }
}

/// The postprocessor resolved for a LightGBM objective, plus the `sigmoid_alpha`
/// the GTIL layer needs for `sigmoid` / `multiclass_ova` postprocessors.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectivePostproc {
    /// The Treelite postprocessor name (e.g. `"softmax"`, `"sigmoid"`).
    pub postprocessor: &'static str,
    /// The parsed `sigmoid:<a>` alpha; `1.0` when the objective has no sigmoid
    /// parameter (the upstream `Model::sigmoid_alpha` default).
    pub sigmoid_alpha: f64,
}

/// Map a CANONICAL objective name (output of [`canonical_objective`]) to its
/// Treelite postprocessor, parsing any `sigmoid:<a>` / `sqrt` from `obj_param`.
///
/// Ports the objective switch (`lightgbm.cc:442-514`):
/// - `multiclass` → `softmax`
/// - `multiclassova` → `multiclass_ova` (requires `sigmoid:<a>` with `a > 0`)
/// - `binary` → `sigmoid` (requires `sigmoid:<a>` with `a > 0`)
/// - `cross_entropy` → `sigmoid` with `sigmoid_alpha = 1.0`
/// - `cross_entropy_lambda` → `logarithm_one_plus_exp`
/// - `poisson` / `gamma` / `tweedie` → `exponential`
/// - regression family (`regression`/`regression_l1`/`huber`/`fair`/`quantile`/
///   `mape`) → `signed_square` if `sqrt` is in `obj_param`, else `identity`
/// - `lambdarank` / `rank_xendcg` / `custom` → `identity`
///
/// `obj_param` is the whitespace-split tail of the `objective=` line (everything
/// after the objective name itself). An unrecognized canonical name returns
/// [`LgbError::UnrecognizedObjective`]; a missing/invalid `sigmoid:<a>` for an
/// objective that requires it returns [`LgbError::InvalidSigmoidAlpha`] (T-04-09).
pub fn map_objective(
    canonical: &str,
    obj_param: &[String],
) -> Result<ObjectivePostproc, LgbError> {
    match canonical {
        "multiclass" => Ok(ObjectivePostproc {
            postprocessor: "softmax",
            sigmoid_alpha: 1.0,
        }),
        "multiclassova" => {
            let alpha = parse_sigmoid_alpha(obj_param);
            require_positive_alpha(canonical, alpha)?;
            Ok(ObjectivePostproc {
                postprocessor: "multiclass_ova",
                sigmoid_alpha: alpha.unwrap_or(0.0),
            })
        }
        "binary" => {
            let alpha = parse_sigmoid_alpha(obj_param);
            require_positive_alpha(canonical, alpha)?;
            Ok(ObjectivePostproc {
                postprocessor: "sigmoid",
                sigmoid_alpha: alpha.unwrap_or(0.0),
            })
        }
        "cross_entropy" => Ok(ObjectivePostproc {
            postprocessor: "sigmoid",
            sigmoid_alpha: 1.0,
        }),
        "cross_entropy_lambda" => Ok(ObjectivePostproc {
            postprocessor: "logarithm_one_plus_exp",
            sigmoid_alpha: 1.0,
        }),
        "poisson" | "gamma" | "tweedie" => Ok(ObjectivePostproc {
            postprocessor: "exponential",
            sigmoid_alpha: 1.0,
        }),
        "regression" | "regression_l1" | "huber" | "fair" | "quantile" | "mape" => {
            // Regression family: `sqrt` toggles signed_square (lightgbm.cc:503-508).
            let sqrt = obj_param.iter().any(|p| p == "sqrt");
            Ok(ObjectivePostproc {
                postprocessor: if sqrt { "signed_square" } else { "identity" },
                sigmoid_alpha: 1.0,
            })
        }
        "lambdarank" | "rank_xendcg" | "custom" => Ok(ObjectivePostproc {
            postprocessor: "identity",
            sigmoid_alpha: 1.0,
        }),
        other => Err(LgbError::UnrecognizedObjective(other.to_string())),
    }
}

/// Parse `sigmoid:<a>` out of the objective parameter tokens, returning the alpha
/// when present and parseable. Mirrors the `Split(str, ':')` token loop in
/// `lightgbm.cc:483-490`. The `> 0` validation is left to the caller so the
/// error can name the objective.
fn parse_sigmoid_alpha(obj_param: &[String]) -> Option<f64> {
    for token in obj_param {
        let mut parts = token.splitn(2, ':');
        let key = parts.next()?;
        if key == "sigmoid"
            && let Some(val) = parts.next()
            && let Ok(a) = val.parse::<f64>()
        {
            return Some(a);
        }
    }
    None
}

/// Reject a missing or non-positive `sigmoid_alpha` (`lightgbm.cc:491-492`,
/// T-04-09).
fn require_positive_alpha(objective: &str, alpha: Option<f64>) -> Result<(), LgbError> {
    match alpha {
        Some(a) if a > 0.0 => Ok(()),
        Some(a) => Err(LgbError::InvalidSigmoidAlpha {
            objective: objective.to_string(),
            alpha: a,
        }),
        None => Err(LgbError::InvalidSigmoidAlpha {
            objective: objective.to_string(),
            alpha: -1.0,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn canonical_collapses_aliases() {
        // l2_root must collapse to regression BEFORE the map (a missing alias
        // would mis-route this to "unknown" → wrong postprocessor).
        assert_eq!(canonical_objective("l2_root").unwrap(), "regression");
        assert_eq!(canonical_objective("mse").unwrap(), "regression");
        assert_eq!(canonical_objective("softmax").unwrap(), "multiclass");
        assert_eq!(canonical_objective("xentropy").unwrap(), "cross_entropy");
        assert_eq!(canonical_objective("binary").unwrap(), "binary");
    }

    #[test]
    fn canonical_rejects_unknown() {
        assert!(matches!(
            canonical_objective("not_an_objective"),
            Err(LgbError::UnrecognizedObjective(_))
        ));
    }

    #[test]
    fn multiclass_maps_to_softmax() {
        let p = map_objective("multiclass", &[]).unwrap();
        assert_eq!(p.postprocessor, "softmax");
    }

    #[test]
    fn binary_maps_to_sigmoid_with_alpha() {
        let p = map_objective("binary", &params(&["sigmoid:1.5"])).unwrap();
        assert_eq!(p.postprocessor, "sigmoid");
        assert_eq!(p.sigmoid_alpha, 1.5);
    }

    #[test]
    fn binary_rejects_nonpositive_alpha() {
        // alpha <= 0 must be rejected (T-04-09).
        assert!(matches!(
            map_objective("binary", &params(&["sigmoid:0"])),
            Err(LgbError::InvalidSigmoidAlpha { .. })
        ));
        // Missing sigmoid:<a> also rejected for binary.
        assert!(matches!(
            map_objective("binary", &[]),
            Err(LgbError::InvalidSigmoidAlpha { .. })
        ));
    }

    #[test]
    fn cross_entropy_maps_to_sigmoid_alpha_one() {
        let p = map_objective("cross_entropy", &[]).unwrap();
        assert_eq!(p.postprocessor, "sigmoid");
        assert_eq!(p.sigmoid_alpha, 1.0);
    }

    #[test]
    fn count_family_maps_to_exponential() {
        for obj in ["poisson", "gamma", "tweedie"] {
            assert_eq!(map_objective(obj, &[]).unwrap().postprocessor, "exponential");
        }
    }

    #[test]
    fn regression_maps_to_identity_or_signed_square() {
        assert_eq!(map_objective("regression", &[]).unwrap().postprocessor, "identity");
        assert_eq!(
            map_objective("regression", &params(&["sqrt"]))
                .unwrap()
                .postprocessor,
            "signed_square"
        );
    }
}

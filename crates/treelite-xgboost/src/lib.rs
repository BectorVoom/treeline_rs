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

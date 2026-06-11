//! `frontend` submodule loaders: thin `#[pyfunction]` wrappers over the Rust
//! `treelite-xgboost` / `treelite-lightgbm` loaders (request → `Model`).
//!
//! File I/O stays Python-side (`treelite_rs/frontend.py` ports `_normalize_path`
//! and reads the bytes/text); these Rust entry points take already-read `&str` /
//! `&[u8]` and converge on `treelite_*::load_*(...).map(Model::from).map_err(Into::into)`
//! (D-06 — any loader error becomes the single `TreeliteError`).

use pyo3::prelude::*;

use crate::error::PyResult2;
use crate::model::Model;

impl From<treelite_core::Model> for Model {
    #[inline]
    fn from(inner: treelite_core::Model) -> Self {
        Model { inner }
    }
}

/// Load an XGBoost model from its JSON text (`xgb.Booster.save_model(*.json)`).
#[pyfunction]
pub fn load_xgboost_json_str(json: &str) -> PyResult2<Model> {
    Ok(treelite_xgboost::load_xgboost_json(json)?.into())
}

/// Load an XGBoost model from its UBJSON bytes (`xgb.Booster.save_model(*.ubj)`).
#[pyfunction]
pub fn load_xgboost_ubjson_bytes(bytes: &[u8]) -> PyResult2<Model> {
    Ok(treelite_xgboost::load_xgboost_ubjson(bytes)?.into())
}

/// Load an XGBoost model from its legacy binary bytes (`*.model`). Reached via the
/// explicit legacy entry point — legacy is NOT auto-detected (upstream D-09 split).
#[pyfunction]
pub fn load_xgboost_legacy_bytes(bytes: &[u8]) -> PyResult2<Model> {
    Ok(treelite_xgboost::load_xgboost_legacy(bytes)?.into())
}

/// Sniff JSON vs UBJSON from a model's leading bytes (`DetectXGBoostFormat`,
/// D-09). Returns `"json"` / `"ubjson"` / `"unknown"`; legacy binary is never
/// auto-detected (it has its own loader). The Rust `detect_xgboost_format` reads
/// only the first two bytes, so an empty/1-byte buffer is `"unknown"`.
#[pyfunction]
pub fn detect_xgboost_format_bytes(bytes: &[u8]) -> String {
    let first_two = &bytes[..bytes.len().min(2)];
    treelite_xgboost::detect_xgboost_format(first_two).to_string()
}

/// Load a LightGBM model from its text dump (`Booster.save_model` / `model.txt`).
#[pyfunction]
pub fn load_lightgbm_str(s: &str) -> PyResult2<Model> {
    Ok(treelite_lightgbm::load_lightgbm(s)?.into())
}

/// Register the frontend loaders into the `frontend` submodule.
pub fn register(frontend: &Bound<'_, PyModule>) -> PyResult<()> {
    frontend.add_function(wrap_pyfunction!(load_xgboost_json_str, frontend)?)?;
    frontend.add_function(wrap_pyfunction!(load_xgboost_ubjson_bytes, frontend)?)?;
    frontend.add_function(wrap_pyfunction!(load_xgboost_legacy_bytes, frontend)?)?;
    frontend.add_function(wrap_pyfunction!(detect_xgboost_format_bytes, frontend)?)?;
    frontend.add_function(wrap_pyfunction!(load_lightgbm_str, frontend)?)?;
    Ok(())
}

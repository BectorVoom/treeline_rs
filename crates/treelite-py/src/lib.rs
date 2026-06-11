//! PyO3 binding for treelite-rs (the compiled extension module `_treelite_rs`).
//!
//! Wave 1 (Plan 08-02) wires the walking-skeleton capability slice: the
//! `#[pyclass] Model` + its inspection getters, the single `TreeliteError`
//! exception (D-06), the `frontend` loaders (XGBoost JSON/UBJSON/legacy +
//! LightGBM), and zero-copy dense `gtil.predict_f32`/`predict_f64` + `output_shape`.
//! The `sklearn` submodule stays empty (08-04); the `backend=` kwarg + panic
//! `guard()` land in 08-05.
//!
//! Note: submodules added via `add_submodule` are NOT auto-registered in
//! `sys.modules`; the `treelite_rs` python-source package re-exports them so
//! `from treelite_rs import frontend` works regardless (D-01 layout).

use pyo3::prelude::*;

mod error;
mod frontend;
mod gtil;
mod model;

pub use error::TreeliteError;
pub use model::Model;

/// The compiled extension module. The function name MUST match the Cargo
/// `[lib] name` (`_treelite_rs`) and the maturin `module-name` leaf (D-02).
#[pymodule]
fn _treelite_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The single exception (D-06) + the Model pyclass at the top level.
    m.add("TreeliteError", m.py().get_type::<TreeliteError>())?;
    m.add_class::<Model>()?;

    // `frontend` submodule: XGBoost (JSON/UBJSON/legacy) + LightGBM loaders.
    let frontend = PyModule::new(m.py(), "frontend")?;
    frontend::register(&frontend)?;
    m.add_submodule(&frontend)?;

    // `gtil` submodule: zero-copy dense predict_f32/_f64 + output-shape helper.
    let gtil = PyModule::new(m.py(), "gtil")?;
    gtil::register(&gtil)?;
    m.add_submodule(&gtil)?;

    // Empty `sklearn` submodule (estimator loaders land in 08-04).
    let sklearn = PyModule::new(m.py(), "sklearn")?;
    m.add_submodule(&sklearn)?;

    Ok(())
}

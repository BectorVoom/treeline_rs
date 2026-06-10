//! PyO3 binding for treelite-rs (the compiled extension module `_treelite_rs`).
//!
//! Wave 0 (Plan 08-01) stands up the build/import/test plumbing only: this is a
//! minimal `#[pymodule]` that registers three EMPTY submodules — `frontend`,
//! `gtil`, and `sklearn` — so `import treelite_rs` succeeds against an abi3 wheel
//! before any capability is wired. The `#[pyclass] Model`, the loader/predict/
//! sklearn `#[pyfunction]`s, the `TreeliteError` exception, and the `backend=`
//! kwarg all land in later plans (08-02 .. 08-05).
//!
//! Note: submodules added via `add_submodule` are NOT auto-registered in
//! `sys.modules`; the `treelite_rs` python-source package re-exports them so
//! `from treelite_rs import frontend` works regardless (D-01 layout).

use pyo3::prelude::*;

/// The compiled extension module. The function name MUST match the Cargo
/// `[lib] name` (`_treelite_rs`) and the maturin `module-name` leaf (D-02).
#[pymodule]
fn _treelite_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Empty `frontend` submodule (loaders land in 08-02/08-03/08-04).
    let frontend = PyModule::new(m.py(), "frontend")?;
    m.add_submodule(&frontend)?;

    // Empty `gtil` submodule (predict* lands in 08-03; backend= in 08-05).
    let gtil = PyModule::new(m.py(), "gtil")?;
    m.add_submodule(&gtil)?;

    // Empty `sklearn` submodule (estimator loaders land in 08-04).
    let sklearn = PyModule::new(m.py(), "sklearn")?;
    m.add_submodule(&sklearn)?;

    Ok(())
}

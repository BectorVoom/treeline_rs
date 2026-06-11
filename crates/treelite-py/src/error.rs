//! Error translation: every Rust crate error → ONE Python `TreeliteError` (D-06).
//!
//! Each `treelite-*` crate defines its own rich `thiserror` enum (the canonical
//! discipline lives in `crates/treelite-gtil/src/error.rs`: every variant carries
//! a descriptive `#[error("...")]` message). At the Python boundary D-06 collapses
//! ALL of them to a single exception type, `TreeliteError`, whose message is the
//! source error's `.to_string()` — so the descriptive variant text is preserved
//! and callers branch on the message, never on a Python type hierarchy.
//!
//! ## Orphan-rule note
//! `impl From<treelite_xgboost::XgbError> for pyo3::PyErr` is NOT legal: both the
//! `From` trait's target (`PyErr`) and the source enums live in foreign crates, so
//! the coherence (orphan) rule forbids a direct impl in this crate. We therefore
//! route every crate error through the LOCAL newtype [`TreelitePyErr`] (a thin
//! wrapper over `pyo3::PyErr`). `impl From<$crate_error> for TreelitePyErr` is the
//! orphan-legal equivalent of the planned `From<…> for PyErr`, and
//! `impl From<TreelitePyErr> for pyo3::PyErr` lets a `#[pyfunction]` returning
//! `PyResult2<T>` (alias for `Result<T, TreelitePyErr>`) use `?` on any crate
//! error transparently. The `guard()` panic-remap helper (D-07) lands in 08-05.

use pyo3::create_exception;
use pyo3::exceptions::PyException;

// The single public exception, registered into `_treelite_rs` (D-06). Its module
// path string MUST match the compiled extension module name so the qualified
// name surfaces as `_treelite_rs.TreeliteError`.
create_exception!(_treelite_rs, TreeliteError, PyException);

/// Local newtype over `pyo3::PyErr` so the per-crate `From` impls are orphan-legal
/// (see module docs). It always wraps a `TreeliteError` (D-06: one exception).
pub struct TreelitePyErr(pyo3::PyErr);

impl TreelitePyErr {
    /// Wrap an already-built `pyo3::PyErr` (e.g. a `TreeliteError::new_err(...)`
    /// constructed at a call site for a non-crate-error condition such as a numpy
    /// contiguity failure).
    #[inline]
    pub fn from_pyerr(e: pyo3::PyErr) -> Self {
        TreelitePyErr(e)
    }
}

impl From<TreelitePyErr> for pyo3::PyErr {
    #[inline]
    fn from(e: TreelitePyErr) -> pyo3::PyErr {
        e.0
    }
}

/// A `#[pyfunction]`/`#[pymethods]` result whose error converts (via `?`) from any
/// crate error and then into a `pyo3::PyErr` carrying the single `TreeliteError`.
pub type PyResult2<T> = Result<T, TreelitePyErr>;

/// Generate `impl From<$crate_error> for TreelitePyErr` for each crate enum,
/// mapping the source error's descriptive `.to_string()` into a `TreeliteError`
/// (D-06). One macro arm per crate so every `?` in a body returning [`PyResult2`]
/// that bubbles a crate error converts to the single Python exception
/// automatically. This is the orphan-legal stand-in for `From<…> for PyErr`.
macro_rules! err_to_treelite {
    ($($t:ty),+ $(,)?) => {
        $(
            impl From<$t> for TreelitePyErr {
                #[inline]
                fn from(e: $t) -> TreelitePyErr {
                    TreelitePyErr(TreeliteError::new_err(e.to_string()))
                }
            }
        )+
    };
}

err_to_treelite! {
    treelite_core::CoreError,
    treelite_xgboost::XgbError,
    treelite_lightgbm::LgbError,
    treelite_sklearn::SklError,
    treelite_gtil::GtilError,
    treelite_cubecl::CubeclError,
    treelite_builder::BuilderError,
}

//! `gtil` submodule: zero-copy dense predict over numpy + per-kind output shape.
//!
//! Two MONOMORPHIZED entry points — `predict_f32` / `predict_f64` — mirror the
//! harness's four-fn-pointer dtype seam (`crates/treelite-harness/src/lib.rs:99-128`):
//! an f32 input flows through the f32 path with NO f32→f64 pre-cast, and vice
//! versa, preserving the 1e-5 contract per preset.
//!
//! Each entry point:
//! 1. takes a typed `PyReadonlyArray2<'py, O>` — pyo3/numpy rejects a wrong-dtype
//!    array BEFORE the body runs (D-03: f64 array into `predict_f32` raises, no
//!    silent cast);
//! 2. `as_slice()` borrows the buffer zero-copy (MEM-04) — a non-contiguous array
//!    yields `AsSliceError`, remapped to `TreeliteError` (D-03 strict, never coerce);
//! 3. releases the GIL with `py.detach(|| treelite_gtil::predict::<O>(...))`
//!    (Pattern 3 — the borrow guard outlives the detached region, soundness note
//!    T-08-06);
//! 4. returns the flat `Vec<O>` via `into_pyarray` (moves the buffer, zero-copy;
//!    the copying `ToPyArray` variant is deliberately NOT used) as a 1-D array —
//!    the `.py` shim reshapes to N-D via `output_shape`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use treelite_gtil::{Config, PredictKind, output_shape};

use crate::error::{PyResult2, TreelitePyErr};
use crate::model::Model;

/// `Send` shim over a `&Model` so the `py.detach` compute closure satisfies the
/// `Ungil` bound (stable pyo3 derives `Ungil` from `Send`). `treelite_core::Model`
/// is `!Send` only because of its `TreeBuf::Borrowed` raw-pointer variant; a model
/// produced by a loader owns `Vec`-backed columns, and the predict body is pure
/// CPU compute that touches NO Python objects. The borrowed numpy buffer guard
/// (`PyReadonlyArray2`) lives on the stack ACROSS the detached region, so the slice
/// stays valid — this is the documented T-08-06 GIL-release soundness mitigation.
struct SendModelRef<'a>(&'a treelite_core::Model);

// SAFETY: see the type doc — the reference is only read for pure compute inside
// the detached region; no `TreeBuf::Borrowed` pointer is mutated or sent onward,
// and the underlying model + numpy borrow both outlive the closure.
unsafe impl Send for SendModelRef<'_> {}

/// Build a GTIL [`Config`] from the Python kwargs. `pred_margin=True` selects the
/// raw-margin kind (skip post-processing); `nthread` is recorded but unused by the
/// scalar reference engine (config.rs note).
#[inline]
fn make_config(nthread: i32, pred_margin: bool) -> Config {
    Config {
        kind: if pred_margin {
            PredictKind::Raw
        } else {
            PredictKind::Default
        },
        nthread,
    }
}

/// Map a numpy contiguity failure to the single `TreeliteError` with the D-03
/// remediation hint (never silently copy/cast).
#[inline]
fn contiguity_err() -> TreelitePyErr {
    use crate::error::TreeliteError;
    // `AsSliceError` carries no detail; supply the actionable message.
    TreelitePyErr::from_pyerr(TreeliteError::new_err(
        "input array must be C-contiguous; call np.ascontiguousarray(arr) first",
    ))
}

/// Extract a typed 2-D readonly numpy view, mapping a DTYPE mismatch to the single
/// `TreeliteError` (D-03 strict + D-06 single exception). Taking `data` as an
/// untyped `PyAny` and extracting here — instead of relying on the `#[pyfunction]`
/// signature's auto-conversion — lets us control the exception TYPE: pyo3's typed
/// param would raise a bare `TypeError` on a wrong dtype, but D-06 mandates ONE
/// `TreeliteError` for every rejection. The extraction is still zero-copy (it
/// borrows the array, never casts — D-03 never coerces f64→f32).
#[inline]
fn extract_readonly<'py, O: numpy::Element>(
    data: &Bound<'py, PyAny>,
    want: &str,
) -> PyResult2<PyReadonlyArray2<'py, O>> {
    use crate::error::TreeliteError;
    data.extract::<PyReadonlyArray2<'py, O>>().map_err(|_| {
        TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "input array dtype does not match this entry point (expected {want}); \
             no implicit cast is performed — pass a {want} C-contiguous array"
        )))
    })
}

/// Zero-copy dense predict for an `<f32,f32>` model over an f32 numpy matrix.
#[pyfunction]
#[pyo3(signature = (model, data, *, nthread = -1, pred_margin = false))]
pub fn predict_f32<'py>(
    py: Python<'py>,
    model: &Model,
    data: &Bound<'py, PyAny>,
    nthread: i32,
    pred_margin: bool,
) -> PyResult2<Bound<'py, PyArray1<f32>>> {
    let data = extract_readonly::<f32>(data, "float32")?;
    let num_row = data.shape()[0];
    let slice = data.as_slice().map_err(|_| contiguity_err())?;
    let cfg = make_config(nthread, pred_margin);
    let inner = SendModelRef(&model.inner);
    // GIL released across the pure-compute predict; the `data` borrow guard lives
    // on the stack across `detach` (T-08-06 soundness). Capture the whole `inner`
    // wrapper (Send) — destructuring `.0` *inside* the closure keeps the disjoint-
    // capture analysis from grabbing the bare `&Model` (which is `!Send`).
    let out = py.detach(move || {
        // Rebind to force whole-struct capture of the `Send` wrapper (edition-2024
        // disjoint capture would otherwise grab the bare `&Model` field, `!Send`).
        let inner = inner;
        treelite_gtil::predict::<f32>(inner.0, slice, num_row, &cfg)
    })?;
    Ok(out.into_pyarray(py))
}

/// Zero-copy dense predict for a `<f64,f64>` model over an f64 numpy matrix.
#[pyfunction]
#[pyo3(signature = (model, data, *, nthread = -1, pred_margin = false))]
pub fn predict_f64<'py>(
    py: Python<'py>,
    model: &Model,
    data: &Bound<'py, PyAny>,
    nthread: i32,
    pred_margin: bool,
) -> PyResult2<Bound<'py, PyArray1<f64>>> {
    let data = extract_readonly::<f64>(data, "float64")?;
    let num_row = data.shape()[0];
    let slice = data.as_slice().map_err(|_| contiguity_err())?;
    let cfg = make_config(nthread, pred_margin);
    let inner = SendModelRef(&model.inner);
    let out = py.detach(move || {
        let inner = inner;
        treelite_gtil::predict::<f64>(inner.0, slice, num_row, &cfg)
    })?;
    Ok(out.into_pyarray(py))
}

/// Per-kind output shape for `num_row` rows. Returns the flat dimension vector so
/// the Python shim can reshape the flat predict output to upstream N-D
/// (`output_shape`/`Shape`, Pitfall 3). `pred_margin` selects Raw vs Default
/// (both produce `(num_row, num_target_or_1, max_num_class)`).
#[pyfunction]
#[pyo3(signature = (model, num_row, *, pred_margin = false))]
pub fn predict_output_shape(model: &Model, num_row: u64, pred_margin: bool) -> Vec<u64> {
    let cfg = make_config(-1, pred_margin);
    output_shape(&model.inner, num_row, &cfg).dims
}

/// Register the gtil predict entry points + output-shape helper into the `gtil`
/// submodule.
pub fn register(gtil: &Bound<'_, PyModule>) -> PyResult<()> {
    gtil.add_function(wrap_pyfunction!(predict_f32, gtil)?)?;
    gtil.add_function(wrap_pyfunction!(predict_f64, gtil)?)?;
    gtil.add_function(wrap_pyfunction!(predict_output_shape, gtil)?)?;
    Ok(())
}

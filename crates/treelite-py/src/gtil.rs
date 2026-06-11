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

use crate::error::{PyResult2, TreelitePyErr, guard_assert};
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

impl<'a> SendModelRef<'a> {
    /// Consume the `Send` wrapper and yield the wrapped `&Model` (IN-04). Taking
    /// `self` BY VALUE forces the detach closure to capture the whole
    /// `SendModelRef` struct rather than the disjoint bare `&Model` field (which
    /// is `!Send` and would fail the `Ungil` bound). This replaces the inscrutable
    /// `let inner = inner;` rebind that existed solely to defeat disjoint capture —
    /// a future edit removing the "redundant" rebind would silently reintroduce a
    /// `!Send` capture. The named consuming method makes the whole-struct move
    /// self-documenting.
    #[inline]
    fn into_ref(self) -> &'a treelite_core::Model {
        self.0
    }
}

/// Build a GTIL [`Config`] from the Python kwargs. `pred_margin=True` selects the
/// raw-margin kind (skip post-processing); `nthread` now drives the scalar
/// predict path end-to-end (Phase 10, PAR-04) — `<= 0` uses all cores, `N` bounds
/// a scoped rayon pool to `N` workers (config.rs note).
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

/// Reject a feature-count mismatch at the Python boundary (CR-01). The downstream
/// `treelite_cubecl::validate_shape` is only a LOWER bound
/// (`data_len >= num_row * num_feature`), and the kernel/scalar engine read each
/// row at the `num_feature` stride. A too-WIDE C-contiguous matrix therefore
/// passes that check, is read at the wrong stride, and returns SILENTLY WRONG
/// predictions — a direct violation of the 1e-5 core-value contract. The 2-D
/// numpy column count is known ONLY here (it is lost once the buffer is flattened
/// to a 1-D `&[F]` slice downstream), so the exact-match guard must live at this
/// boundary. The too-narrow direction is already rejected downstream; this closes
/// the too-wide gap and makes every shape rejection one typed `TreeliteError` (D-06).
#[inline]
fn check_feature_count(num_col: usize, num_feature: i32) -> PyResult2<()> {
    use crate::error::TreeliteError;
    // WR-04: `num_feature` is loader-produced/untrusted, so a NEGATIVE value is a
    // distinct corrupt-model condition — reject it FIRST with a dedicated message.
    // Conflating it with the column-equality check (the old `num_feature < 0 ||
    // num_col != num_feature as usize`) made the negative case fire only via the
    // `num_col != (negative as usize)` wrap, emitting a misleading "expects -1
    // features" message, and let a `(0,0)` input against a `num_feature == 0`
    // model slip through. Rejecting `< 0` up front keeps the equality below
    // operating on a known-non-negative value.
    if num_feature < 0 {
        return Err(TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "corrupt model: negative feature count ({num_feature})"
        ))));
    }
    if num_col != num_feature as usize {
        return Err(TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "input has {num_col} columns but the model expects {num_feature} \
             features; pass a (num_row, {num_feature}) C-contiguous array"
        ))));
    }
    Ok(())
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

/// The backends compiled into THIS wheel, assembled from the active
/// `#[cfg(feature)]` set. `cpu` is always present (the default feature); each GPU
/// backend appears only if its cargo feature was enabled at build time. Surfaced
/// in the un-built-backend error so a caller knows what the installed wheel
/// actually supports (D-05 — explicit selection, no auto-detect).
///
/// IN-03: built at runtime by pushing one guarded fragment per backend and
/// joining, so adding a backend is a single guarded line — replacing the former
/// 8-arm hand-maintained `#[cfg]` cascade that enumerated every feature
/// combination (and doubled in size with each new backend).
pub fn built_backends() -> String {
    // `mut` is only exercised when at least one GPU feature is enabled; allow the
    // unused-mut warning for the cpu-only default build.
    #[allow(unused_mut)]
    let mut backends: Vec<&str> = vec!["cpu"];
    #[cfg(feature = "rocm")]
    backends.push("rocm");
    #[cfg(feature = "cuda")]
    backends.push("cuda");
    #[cfg(feature = "wgpu")]
    backends.push("wgpu");
    backends.join(", ")
}

/// Build the typed "backend not available in this wheel" error (D-05/T-08-13):
/// an un-built backend name yields a `TreeliteError` naming [`built_backends`] —
/// NEVER a silent CPU fallback (D-08).
#[inline]
fn unbuilt_backend_err(backend: &str) -> TreelitePyErr {
    use crate::error::TreeliteError;
    let built = built_backends();
    TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
        "backend '{backend}' is not available in this wheel (built with: {built})"
    )))
}

/// Dispatch one monomorphized predict over the requested `backend` string. The
/// `"cpu"` arm runs `treelite_cubecl::predict_cpu` (which itself routes a
/// categorical / non-`kLT` model to the checked scalar fallback via
/// `model_routes_to_scalar_fallback`, D-02 — the ONLY fallback in this function,
/// and never a device-absent one). Each GPU arm is `#[cfg(feature)]`-gated and
/// dispatches to `treelite_cubecl::predict::<R, F>`; a device-absent compiled
/// backend surfaces `CubeclError::DeviceUnavailable` which the caller maps to
/// `TreeliteError` via the `From` impl — NEVER a silent CPU fallback (D-08).
/// An un-built / unknown backend name hits the `other =>` arm → typed error.
///
/// Returns the cubecl `Result` so the caller applies `?` (mapping `CubeclError`
/// → `TreeliteError`); runs inside the detached region, wrapped in `guard_assert`
/// at the call site so any trapped panic becomes a `TreeliteError` (D-07).
#[inline]
fn dispatch_backend<F: treelite_cubecl::PredictCpuElem>(
    backend: &str,
    model: &treelite_core::Model,
    slice: &[F],
    num_row: usize,
    cfg: &treelite_gtil::Config,
) -> PyResult2<Vec<F>> {
    match backend {
        "cpu" => Ok(treelite_cubecl::predict_cpu::<F>(model, slice, num_row, cfg)?),
        #[cfg(feature = "rocm")]
        "rocm" => Ok(treelite_cubecl::predict::<cubecl::hip::HipRuntime, F>(
            model, slice, num_row, cfg,
        )?),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(treelite_cubecl::predict::<cubecl::cuda::CudaRuntime, F>(
            model, slice, num_row, cfg,
        )?),
        #[cfg(feature = "wgpu")]
        "wgpu" => Ok(treelite_cubecl::predict::<cubecl::wgpu::WgpuRuntime, F>(
            model, slice, num_row, cfg,
        )?),
        other => Err(unbuilt_backend_err(other)),
    }
}

/// Zero-copy dense predict for an `<f32,f32>` model over an f32 numpy matrix.
///
/// The additive `backend="cpu"` kwarg (D-05) selects among the wheel's compiled-in
/// compute backends; omitting it (or passing `"cpu"`) keeps the call
/// upstream-identical. An un-built or device-absent backend raises `TreeliteError`,
/// never a silent CPU fallback (D-08).
#[pyfunction]
#[pyo3(signature = (model, data, *, nthread = -1, pred_margin = false, backend = "cpu"))]
pub fn predict_f32<'py>(
    py: Python<'py>,
    model: &Model,
    data: &Bound<'py, PyAny>,
    nthread: i32,
    pred_margin: bool,
    backend: &str,
) -> PyResult2<Bound<'py, PyArray1<f32>>> {
    let data = extract_readonly::<f32>(data, "float32")?;
    let num_row = data.shape()[0];
    check_feature_count(data.shape()[1], model.inner.num_feature)?;
    let slice = data.as_slice().map_err(|_| contiguity_err())?;
    let cfg = make_config(nthread, pred_margin);
    let inner = SendModelRef(&model.inner);
    let backend = backend.to_string();
    // GIL released across the pure-compute predict; the `data` borrow guard lives
    // on the stack across `detach` (T-08-06 soundness). Capture the whole `inner`
    // wrapper (Send) — destructuring `.0` *inside* the closure keeps the disjoint-
    // capture analysis from grabbing the bare `&Model` (which is `!Send`). The
    // compute is wrapped in `guard_assert` so a trapped panic becomes a catchable
    // `TreeliteError` (D-07), never an FFI abort.
    let out = py.detach(move || {
        // `into_ref` consumes the whole `SendModelRef` (IN-04: self-documenting
        // whole-struct move; no disjoint `!Send` capture of the bare `&Model`).
        let model = inner.into_ref();
        guard_assert(|| dispatch_backend::<f32>(&backend, model, slice, num_row, &cfg))
    })?;
    Ok(out.into_pyarray(py))
}

/// Zero-copy dense predict for a `<f64,f64>` model over an f64 numpy matrix.
///
/// Additive `backend="cpu"` kwarg (D-05); un-built / device-absent backend raises
/// `TreeliteError`, never a silent CPU fallback (D-08).
#[pyfunction]
#[pyo3(signature = (model, data, *, nthread = -1, pred_margin = false, backend = "cpu"))]
pub fn predict_f64<'py>(
    py: Python<'py>,
    model: &Model,
    data: &Bound<'py, PyAny>,
    nthread: i32,
    pred_margin: bool,
    backend: &str,
) -> PyResult2<Bound<'py, PyArray1<f64>>> {
    let data = extract_readonly::<f64>(data, "float64")?;
    let num_row = data.shape()[0];
    check_feature_count(data.shape()[1], model.inner.num_feature)?;
    let slice = data.as_slice().map_err(|_| contiguity_err())?;
    let cfg = make_config(nthread, pred_margin);
    let inner = SendModelRef(&model.inner);
    let backend = backend.to_string();
    let out = py.detach(move || {
        // See `predict_f32`: consume the whole `SendModelRef` (IN-04).
        let model = inner.into_ref();
        guard_assert(|| dispatch_backend::<f64>(&backend, model, slice, num_row, &cfg))
    })?;
    Ok(out.into_pyarray(py))
}

/// Per-kind output shape for `num_row` rows. Returns the flat dimension vector so
/// the Python shim can reshape the flat predict output to upstream N-D
/// (`output_shape`/`Shape`, Pitfall 3). `pred_margin` selects Raw vs Default
/// (both produce `(num_row, num_target_or_1, max_num_class)`).
#[pyfunction]
#[pyo3(signature = (model, num_row, *, pred_margin = false))]
pub fn predict_output_shape(model: &Model, num_row: u64, pred_margin: bool) -> PyResult2<Vec<u64>> {
    use crate::error::TreeliteError;
    // WR-03: `num_row` is untrusted (this fn is re-exported and callable directly,
    // e.g. `predict_output_shape(model, 2**63)`). `output_shape` multiplies
    // `num_row * num_target * max_num_class`; in a RELEASE build a wrapping
    // multiply would silently produce a wrong shape vector (no panic for
    // `guard_assert` to trap), and the downstream reshape would then mis-shape
    // (feeding CR-01). Reject an out-of-range `num_row` at the boundary BEFORE the
    // arithmetic runs. `i32::MAX` is the largest legitimate row count (numpy row
    // counts are `isize`-bounded and the engine reads rows at an `i32` feature
    // stride), so anything larger is a corrupt/pathological call.
    if num_row > i32::MAX as u64 {
        return Err(TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "num_row {num_row} exceeds the maximum supported row count ({})",
            i32::MAX
        ))));
    }
    // `predict_output_shape` is on the hot predict path (the Python `predict`
    // shim calls it on every call to compute the flat→N-D reshape target). It
    // must therefore share the predict path's panic message-parity (WR-01): a
    // panic in `output_shape` on a degenerate model would otherwise surface as a
    // bare pyo3 `PanicException`, which a caller doing `except TreeliteError`
    // (the D-06 contract) cannot catch. `guard_assert` remaps any trapped panic
    // to the single `TreeliteError` (D-07).
    let cfg = make_config(-1, pred_margin);
    guard_assert(|| Ok(output_shape(&model.inner, num_row, &cfg).dims))
}

/// Register the gtil predict entry points + output-shape helper into the `gtil`
/// submodule.
pub fn register(gtil: &Bound<'_, PyModule>) -> PyResult<()> {
    gtil.add_function(wrap_pyfunction!(predict_f32, gtil)?)?;
    gtil.add_function(wrap_pyfunction!(predict_f64, gtil)?)?;
    gtil.add_function(wrap_pyfunction!(predict_output_shape, gtil)?)?;
    Ok(())
}

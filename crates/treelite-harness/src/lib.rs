//! `treelite-harness` — the 1e-5 equivalence instrument (Wave 3).
//!
//! This crate closes the walking skeleton: it loads the frozen upstream golden
//! (`fixtures/golden.json`), runs the full Rust pipeline
//! ([`treelite_xgboost::load_xgboost_json`] → [`treelite_gtil::predict`]) over
//! the committed input matrix, and asserts every output element is within
//! `1e-5` of the golden — reporting the max observed `|delta|`.
//!
//! This is a dev/test-facing crate, so it uses `anyhow` for ALL error context
//! (ERR-02). The three library crates (`treelite-core`, `-xgboost`, `-gtil`)
//! use `thiserror` and never expose `anyhow`; the harness consumes their typed
//! errors and wraps them with `.context(...)` here.
//!
//! ## On `NaN` in the golden input
//!
//! Python's `json.dump` writes a bare `NaN` token for missing feature values
//! (the golden's row 5 routes a missing `feature[0]` via `default_left`).
//! `serde_json` (strict, by design) rejects the bare `NaN` literal, so
//! [`load_golden`] deserializes input cells via [`NanF32`], which accepts a JSON
//! `null` OR a number and maps a missing value to `f32::NAN`. The raw golden
//! text is normalized once (`NaN` → `null`) before parsing — the committed
//! `golden.json` is NEVER modified on disk.

use std::fmt;

use anyhow::Context;
use serde::Deserialize;
use serde::de::{Deserializer, Visitor};

pub mod manifest;
pub mod report;

pub use manifest::{Manifest, check_manifest};

use treelite_core::Model;
use treelite_gtil::{Config, SparseCsr};

// ----------------------------------------------------------------------------
// Backend-parameterized seam (D-11, RESEARCH Pattern 4) — DESIGN ONLY this
// phase. The minimal seam is a small tag enum + a fn-pointer registry, NOT a
// trait-object hierarchy. Phase 6/7 register a new backend by adding a variant
// and a `RunnerCase` constructor — the matrix iteration in
// `tests/gtil_matrix.rs` never changes.
// ----------------------------------------------------------------------------

/// Which compute runtime produces (and is asserted by) a matrix cell (D-09/D-11).
///
/// Phase 5 has exactly one variant — the plain-Rust scalar reference every
/// later backend is measured against to 1e-5. The future variants
/// (`CubeclCpu`, `Cuda`, `Wgpu`, `Rocm`) are NOT added yet; they land in Phase
/// 6/7 as a registration (a new variant + a [`RunnerCase`] constructor), with
/// NO change to the matrix iteration (RESEARCH Pattern 4 — "avoid
/// over-engineering into a full trait object hierarchy").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The plain-Rust scalar GTIL engine (`treelite_gtil::predict` /
    /// `predict_sparse`). The reference/fallback (manifest `backend ==
    /// "scalar-cpu"`).
    ScalarCpu,
    /// The cubecl CPU-backend GTIL engine (`treelite_cubecl::predict_cpu`) for
    /// the dense numerical path; sparse/categorical cells fall back to the
    /// scalar `treelite_gtil::predict_sparse`/`predict` (D-02). Registered
    /// additively in Phase 6 (a new variant + the [`cubecl_cpu_case`]
    /// constructor) with NO change to the matrix iteration (D-11).
    CubeclCpu,
    /// The cubecl ROCm (AMD HIP) GPU-backend GTIL engine
    /// (`treelite_cubecl::predict::<cubecl::hip::HipRuntime, _>`) for the dense
    /// numerical path; sparse/categorical cells fall back to the scalar
    /// `treelite_gtil::predict_sparse`/`predict` (D-02). Registered additively in
    /// Phase 7 (this variant + the [`rocm_case`] constructor) with NO change to
    /// the matrix iteration (D-11). **Explicit-selection only** — there is no
    /// auto-detect / "best available" resolver; the caller names the backend
    /// (D-04). A missing device surfaces as the typed `DeviceUnavailable` skip
    /// from `predict::<R, _>` (D-05), never a silent CPU fallback.
    Rocm,
    /// The cubecl CUDA GPU-backend GTIL engine
    /// (`treelite_cubecl::predict::<cubecl::cuda::CudaRuntime, _>`) for the dense
    /// numerical path; sparse/categorical cells fall back to the scalar engine
    /// (D-02). Registered additively in Phase 7 (this variant + the [`cuda_case`]
    /// constructor) with NO change to the matrix iteration (D-11).
    /// **Explicit-selection only** (no auto-detect, D-04). A missing device
    /// surfaces as the typed `DeviceUnavailable` skip (D-05), never a silent CPU
    /// fallback.
    Cuda,
    /// The cubecl wgpu GPU-backend GTIL engine
    /// (`treelite_cubecl::predict::<cubecl::wgpu::WgpuRuntime, _>`) for the dense
    /// numerical path; sparse/categorical cells fall back to the scalar engine
    /// (D-02). Registered additively in Phase 7 (this variant + the [`wgpu_case`]
    /// constructor) with NO change to the matrix iteration (D-11).
    /// **Explicit-selection only** (no auto-detect, D-04). A missing adapter
    /// surfaces as the typed `DeviceUnavailable` skip (D-05), never a silent CPU
    /// fallback.
    Wgpu,
}

/// Dense predict over an **f32-input** matrix (D-05). The output element type is
/// fixed to `f64` so EVERY golden — f32-input and f64-input alike — is compared
/// on ONE accumulator; only the *input* element type varies per fixture.
pub type DensePredictF32Fn = fn(&Model, &[f32], usize, &Config) -> anyhow::Result<Vec<f64>>;
/// Dense predict over an **f64-input** matrix (D-05).
pub type DensePredictF64Fn = fn(&Model, &[f64], usize, &Config) -> anyhow::Result<Vec<f64>>;
/// Sparse-CSR predict over an **f32-input** matrix (D-05).
pub type SparsePredictF32Fn =
    fn(&Model, SparseCsr<'_, f32>, usize, &Config) -> anyhow::Result<Vec<f64>>;
/// Sparse-CSR predict over an **f64-input** matrix (D-05).
pub type SparsePredictF64Fn =
    fn(&Model, SparseCsr<'_, f64>, usize, &Config) -> anyhow::Result<Vec<f64>>;

/// A registered backend's predict entry points — BOTH input dtypes for both
/// layouts (D-05, RESEARCH Pitfall 1).
///
/// The seam carries four fn pointers, not one: the committed matrix has both
/// f32-input and f64-input cells, and `predict`/`predict_sparse` are O-generic
/// over the input element type. An f32-input fixture MUST flow through the f32
/// entry point with NO f32→f64 pre-cast (casting before predict would erase the
/// input-dtype axis and hide the InputT-as-accumulator behavior this instrument
/// exists to verify). The output is uniformly `f64` so all goldens compare on
/// one accumulator.
///
/// Phase 6 registers `Backend::CubeclCpu` by adding a `RunnerCase` whose four
/// slots point at `cubecl_predict_f32`/`_f64` and `cubecl_predict_sparse_*` —
/// without touching the matrix iteration.
#[derive(Clone, Copy)]
pub struct RunnerCase {
    /// Which backend these fn pointers belong to.
    pub backend: Backend,
    /// Dense f32-input entry point.
    pub dense_f32: DensePredictF32Fn,
    /// Dense f64-input entry point.
    pub dense_f64: DensePredictF64Fn,
    /// Sparse f32-input entry point.
    pub sparse_f32: SparsePredictF32Fn,
    /// Sparse f64-input entry point.
    pub sparse_f64: SparsePredictF64Fn,
}

/// The scalar-cpu [`RunnerCase`] (Phase-5 reference). Wires the four
/// `treelite_gtil` O-generic entry points — `predict::<f32>`/`predict::<f64>`
/// and `predict_sparse::<f32>`/`predict_sparse::<f64>` — into the four slots,
/// bridging the typed `GtilError` into `anyhow`.
pub fn scalar_cpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::ScalarCpu,
        dense_f32: |model, data, num_row, cfg| {
            // f32 input → f32 output, widened to the common f64 accumulator for
            // comparison. The PREDICT runs in f32 (no pre-cast); only the
            // already-computed f32 results are lifted to f64 afterwards.
            let out = treelite_gtil::predict::<f32>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_gtil::predict::<f64>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
    }
}

/// The cubecl-cpu [`RunnerCase`] (Phase-6 registration, D-11). Mirrors
/// [`scalar_cpu_case`] EXACTLY except the dense slots route through the cubecl
/// CPU kernels ([`treelite_cubecl::predict_cpu`], D-01/D-05) and the sparse
/// slots keep the scalar [`treelite_gtil::predict_sparse`] fallback (D-02 — the
/// cubecl kernels are dense-numerical only this phase). Adding this constructor
/// together with the [`Backend::CubeclCpu`] variant is the WHOLE registration.
/// The `RunnerCase` struct, the slot type aliases, and the matrix iteration in
/// `tests/gtil_matrix.rs` all stay untouched (D-11 registration-not-refactor).
///
/// Provenance (D-06) is RECORDED at assertion time by the matrix sibling
/// (`tests/gtil_matrix_cubecl.rs`): a dense numerical cell that actually ran the
/// kernel is tagged `"cubecl-kernel"`; a sparse cell (or a categorical model
/// that `predict_cpu` itself routes to the scalar fallback) is tagged
/// `"scalar-fallback"`. The `"1e-5 on cubecl-cpu"` claim can therefore never
/// silently mean `"validated on scalar fallback"`.
pub fn cubecl_cpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::CubeclCpu,
        dense_f32: |model, data, num_row, cfg| {
            // f32 input → f32 output, widened to the common f64 accumulator for
            // comparison. The PREDICT runs in f32 (no pre-cast — Pitfall 6);
            // only the already-computed f32 RESULT is lifted to f64 afterwards.
            let out = treelite_cubecl::predict_cpu::<f32>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict_cpu::<f64>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
        // Sparse rides the scalar fallback (D-02): the cubecl kernels are
        // dense-numerical only this phase, so the sparse slots point at the SAME
        // `treelite_gtil::predict_sparse` entry the scalar reference uses.
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
    }
}

/// The cubecl-ROCm (AMD HIP) [`RunnerCase`] (Phase-7 registration, D-11).
/// Mirrors [`cubecl_cpu_case`] EXACTLY except the dense slots route through the
/// runtime-generic GPU launcher
/// [`treelite_cubecl::predict`]`::<cubecl::hip::HipRuntime, _>` (Plan 02). The
/// sparse slots keep the scalar [`treelite_gtil::predict_sparse`] fallback
/// (D-02 — the cubecl kernels are dense-numerical only). Adding this constructor
/// together with the [`Backend::Rocm`] variant is the WHOLE registration: the
/// [`RunnerCase`] struct, the slot type aliases, and the matrix iteration are
/// all untouched (D-11 registration-not-refactor).
///
/// **Device-absence (D-05, A3):** a missing HIP device surfaces as the typed
/// `treelite_cubecl::CubeclError::DeviceUnavailable` propagated out of
/// `predict::<R, _>` via `?` (Plan-01 proved this is a catchable error, not an
/// FFI abort — NO pre-construction probe is required). The harness branches on
/// that `Err` as a SKIP, never a silent CPU fallback (D-09). Behind the `rocm`
/// cargo feature.
#[cfg(feature = "rocm")]
pub fn rocm_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::Rocm,
        dense_f32: |model, data, num_row, cfg| {
            // f32 input → f32 output, widened to the common f64 accumulator for
            // comparison. The PREDICT runs in f32 (no pre-cast — Pitfall 6);
            // only the already-computed f32 RESULT is lifted to f64 afterwards.
            // Preserve the TYPED CubeclError as a downcastable anyhow source
            // (WR-04) so the caller can `matches!(e.downcast_ref::<CubeclError>(),
            // Some(DeviceUnavailable))` instead of brittle Display-substring
            // matching. `anyhow::Error::new` keeps the error downcastable;
            // `anyhow!("{e}")` would flatten it to an opaque string.
            let out =
                treelite_cubecl::predict::<cubecl::hip::HipRuntime, f32>(model, data, num_row, cfg)
                    .map_err(anyhow::Error::new)?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out =
                treelite_cubecl::predict::<cubecl::hip::HipRuntime, f64>(model, data, num_row, cfg)
                    .map_err(anyhow::Error::new)?;
            Ok(out)
        },
        // Sparse rides the scalar fallback (D-02): identical to cubecl_cpu_case.
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
    }
}

/// The cubecl-CUDA [`RunnerCase`] (Phase-7 registration, D-11). Identical in
/// shape to [`rocm_case`] but routes dense cells through
/// [`treelite_cubecl::predict`]`::<cubecl::cuda::CudaRuntime, _>` and tags
/// [`Backend::Cuda`]. Sparse keeps the scalar fallback (D-02). A missing CUDA
/// device propagates the typed `DeviceUnavailable` skip out of `predict::<R, _>`
/// via `?` (A3: catchable error, not an FFI abort — no pre-construction probe).
/// Behind the `cuda` cargo feature.
#[cfg(feature = "cuda")]
pub fn cuda_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::Cuda,
        dense_f32: |model, data, num_row, cfg| {
            // Preserve the TYPED CubeclError as a downcastable source (WR-04).
            let out = treelite_cubecl::predict::<cubecl::cuda::CudaRuntime, f32>(
                model, data, num_row, cfg,
            )
            .map_err(anyhow::Error::new)?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict::<cubecl::cuda::CudaRuntime, f64>(
                model, data, num_row, cfg,
            )
            .map_err(anyhow::Error::new)?;
            Ok(out)
        },
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
    }
}

/// The cubecl-wgpu [`RunnerCase`] (Phase-7 registration, D-11). Identical in
/// shape to [`rocm_case`] but routes dense cells through
/// [`treelite_cubecl::predict`]`::<cubecl::wgpu::WgpuRuntime, _>` and tags
/// [`Backend::Wgpu`]. Sparse keeps the scalar fallback (D-02). A missing adapter
/// propagates the typed `DeviceUnavailable` skip out of `predict::<R, _>` via
/// `?`. Behind the `wgpu` cargo feature.
///
/// **f64 caveat (RESEARCH Pitfall 3):** not every wgpu adapter advertises
/// 64-bit float support; on such an adapter the `dense_f64` slot's
/// `predict::<WgpuRuntime, f64>` may surface a runtime error rather than a
/// silent downcast. That error propagates through `?` (the harness reports it
/// rather than masking it), preserving the 1e-5 fidelity contract.
#[cfg(feature = "wgpu")]
pub fn wgpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::Wgpu,
        dense_f32: |model, data, num_row, cfg| {
            // Preserve the TYPED CubeclError as a downcastable source (WR-04).
            let out = treelite_cubecl::predict::<cubecl::wgpu::WgpuRuntime, f32>(
                model, data, num_row, cfg,
            )
            .map_err(anyhow::Error::new)?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict::<cubecl::wgpu::WgpuRuntime, f64>(
                model, data, num_row, cfg,
            )
            .map_err(anyhow::Error::new)?;
            Ok(out)
        },
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
    }
}

/// One input feature cell: a finite number, or a missing value (`f32::NAN`).
///
/// Accepts either a JSON number or a JSON `null` (the latter is what a bare
/// `NaN` is normalized to before parsing — see the module docs). This lets the
/// harness round-trip the golden's missing-value row without mutating the
/// committed `golden.json`.
#[derive(Debug, Clone, Copy)]
pub struct NanF32(pub f32);

impl<'de> Deserialize<'de> for NanF32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NanVisitor;
        impl<'de> Visitor<'de> for NanVisitor {
            type Value = NanF32;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a number or null (missing value)")
            }
            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
                Ok(NanF32(v as f32))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
                Ok(NanF32(v as f32))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                Ok(NanF32(v as f32))
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                // JSON `null` (a normalized `NaN`) → a missing feature value.
                Ok(NanF32(f32::NAN))
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(NanF32(f32::NAN))
            }
        }
        deserializer.deserialize_any(NanVisitor)
    }
}

/// The frozen golden artifact produced by `fixtures/capture_golden.py` (D-06/D-07).
///
/// Mirrors the JSON shape `{input, output, manifest}` exactly. `input` is the
/// committed `num_row × num_feature` feature matrix; `output` is the upstream
/// Treelite GTIL prediction vector this harness asserts against within `1e-5`.
#[derive(Debug, Deserialize)]
pub struct Golden {
    /// Row-major input matrix (`num_row` rows, each `num_feature` cells).
    pub input: Vec<Vec<NanF32>>,
    /// The frozen upstream prediction vector (one `f32` per row for
    /// `binary:logistic`).
    pub output: Vec<f32>,
    /// Provenance metadata for diagnosing environment-divergence failures.
    pub manifest: Manifest,
}

/// Load + parse the frozen golden artifact.
///
/// Reads `path`, normalizes Python's bare `NaN` tokens to JSON `null` (so
/// `serde_json` accepts the missing-value row), and deserializes into [`Golden`].
/// A read or parse failure surfaces an `anyhow` context chain rather than an
/// opaque panic (ERR-02, T-04-01).
pub fn load_golden(path: &str) -> anyhow::Result<Golden> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading golden.json at {path}"))?;
    let normalized = normalize_nan_tokens(&raw);
    let golden: Golden = serde_json::from_str(&normalized).context("parsing golden.json")?;
    Ok(golden)
}

/// Replace bare `NaN` JSON tokens with `null` so `serde_json` (which rejects the
/// non-standard `NaN` literal) can parse the golden.
///
/// Only standalone `NaN` tokens are replaced: a match must be bounded by a
/// non-identifier character (or string edge) on both sides, so substrings of
/// larger identifiers are never touched. JSON string contents in this golden
/// never contain a bare `NaN` value, so this is safe for the committed artifact.
fn normalize_nan_tokens(raw: &str) -> String {
    let bytes = raw.as_bytes();
    // Build a raw `Vec<u8>` and copy non-`NaN` bytes verbatim, so multi-byte
    // UTF-8 (any byte >= 0x80) round-trips unchanged. The previous
    // `bytes[i] as char` reinterpreted a single byte as the scalar
    // U+0080..U+00FF and re-encoded it as TWO UTF-8 bytes, corrupting any
    // non-ASCII content (WR-03). Since `raw` is valid UTF-8 and we only ever
    // splice in the ASCII literal "null", the result is always valid UTF-8.
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'N'
            && raw[i..].starts_with("NaN")
            && !preceded_by_ident(bytes, i)
            && !followed_by_ident(bytes, i + 3)
        {
            out.extend_from_slice(b"null");
            i += 3;
        } else {
            // Copy the raw byte verbatim — byte-faithful for ASCII and for the
            // continuation/lead bytes of multi-byte UTF-8 sequences alike.
            out.push(bytes[i]);
            i += 1;
        }
    }
    // SAFETY-of-correctness: `raw` is valid UTF-8 and we only removed whole
    // "NaN" ASCII runs / inserted ASCII "null", preserving UTF-8 validity, so
    // this never errors. Fall back to a lossless rebuild if it somehow does.
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn preceded_by_ident(bytes: &[u8], i: usize) -> bool {
    i > 0 && is_ident_byte(bytes[i - 1])
}

fn followed_by_ident(bytes: &[u8], i: usize) -> bool {
    i < bytes.len() && is_ident_byte(bytes[i])
}

/// Run the full equivalence pipeline against `golden`, returning the max
/// observed `|delta|`.
///
/// Reads the model JSON at `model_json_path`, loads it via
/// [`treelite_xgboost::load_xgboost_json`], flattens `golden.input` into a
/// row-major `f32` buffer, predicts via [`treelite_gtil::predict`], then asserts
/// every output element is within `1e-5` of `golden.output`
/// (`approx::assert_abs_diff_eq!`) while tracking the maximum absolute
/// deviation. Returns that maximum as an `f64`.
///
/// The `1e-5` assertion is the hard gate: if a genuine `> 1e-5` deviation
/// exists, this panics through the `assert_abs_diff_eq!` macro (surfaced by the
/// caller's test harness) — the tolerance is NEVER loosened to mask a real
/// fidelity gap.
pub fn run_equivalence(model_json_path: &str, golden: &Golden) -> anyhow::Result<f64> {
    let model_json = std::fs::read_to_string(model_json_path)
        .with_context(|| format!("reading model JSON at {model_json_path}"))?;
    let model = treelite_xgboost::load_xgboost_json(&model_json)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading fixture model")?;

    let num_row = golden.input.len();
    anyhow::ensure!(num_row > 0, "golden input has zero rows");
    let num_feature = golden.input[0].len();
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * num_feature);
    for (r, row) in golden.input.iter().enumerate() {
        anyhow::ensure!(
            row.len() == num_feature,
            "golden input row {r} has {} cells, expected {num_feature}",
            row.len()
        );
        flat.extend(row.iter().map(|c| c.0));
    }

    let rust = treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default())
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("predicting")?;

    anyhow::ensure!(
        rust.len() == golden.output.len(),
        "prediction length {} != golden output length {}",
        rust.len(),
        golden.output.len()
    );

    let mut max_dev: f64 = 0.0;
    for (i, &got) in rust.iter().enumerate() {
        let expected = golden.output[i];
        let delta = (got - expected).abs() as f64;
        if delta > max_dev {
            max_dev = delta;
        }
        // The hard 1e-5 gate. If this fires, a real fidelity gap was found —
        // do NOT loosen the epsilon to make it pass.
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }

    Ok(max_dev)
}

#[cfg(test)]
mod tests {
    use super::normalize_nan_tokens;

    #[test]
    fn non_ascii_bytes_are_preserved_byte_for_byte() {
        // A manifest-style string with non-ASCII content (>= 0x80 bytes): an
        // accented char, an em dash, and a CJK char. The previous
        // `bytes[i] as char` path mangled every such byte; the fixed version
        // must leave them byte-identical (WR-03).
        let input = r#"{"os": "Café — 日本", "x": NaN}"#;
        let out = normalize_nan_tokens(input);
        // The bare NaN token is normalized to null...
        assert_eq!(out, r#"{"os": "Café — 日本", "x": null}"#);
        // ...and every non-ASCII byte survives unchanged (no expansion/corruption).
        let expected_non_nan = r#"{"os": "Café — 日本", "x": "#;
        assert_eq!(
            out.as_bytes()[..expected_non_nan.len()],
            expected_non_nan.as_bytes()[..],
        );
    }

    #[test]
    fn standalone_nan_replaced_but_identifier_substrings_untouched() {
        // Standalone NaN → null; "NaNny" (NaN as a substring of a larger
        // identifier) is left alone.
        let out = normalize_nan_tokens(r#"[NaN, "NaNny"]"#);
        assert_eq!(out, r#"[null, "NaNny"]"#);
    }
}

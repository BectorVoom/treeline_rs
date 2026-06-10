//! Plan 07-04 — the ROCm GPU matrix sibling (GPU-04), REPORT MODE / skip-not-fail.
//!
//! This is a SIBLING of `gtil_matrix_cubecl.rs` (itself a sibling of
//! `gtil_matrix.rs`), NOT a refactor of either (D-11 smell guard): a thin new
//! file that registers [`treelite_harness::rocm_case`] and drives the SAME
//! frozen `fixtures/gtil/*.golden.json` cross-product (D-07) through the ROCm
//! backend. The developer runs it explicitly on the AMD/ROCm box (`#[ignore]`,
//! D-06) to REGENERATE the committed `docs/GPU_EQUIVALENCE_REPORT.md` (+ the
//! `docs/gpu_equivalence.json` sidecar) — the numbers are never hand-edited.
//!
//! ## Why this is observational, never a gate (D-01)
//!
//! Unlike `gtil_matrix_cubecl.rs` (which hard-gates the cubecl CPU backend at
//! `1e-5`), this sibling RECORDS each cell's max `|delta|` via
//! [`treelite_harness::report::max_abs_delta_report_mode`] and NEVER fails on a
//! GPU `|delta| > 1e-5`. Per the OpenCL spec, GPU transcendental rounding and
//! float-reduction order are implementation-defined, so a measured value above
//! the band is a *finding to record* (e.g. a `native_exp` mapping, Pitfall 2),
//! not a CI failure. The hard `1e-5` gate stays on the scalar / cubecl-cpu
//! siblings, untouched.
//!
//! ## Skip-not-fail (D-05)
//!
//! A missing HIP device surfaces as the typed `DeviceUnavailable` skip
//! propagated out of `predict::<R, _>` (Plan-01 A3: a catchable error, not an
//! FFI abort — NO pre-construction probe). The harness `rocm_case()` returns it
//! as an `anyhow` error whose message contains "no device available"; this
//! sibling detects that, marks the row "not run — no device", and the test
//! PASSES (absence is a skip). CUDA/wgpu render "not run" the same way (this is
//! the `rocm`-feature build; their `*_case()` constructors are not compiled in
//! here, so their columns are recorded as not-run without constructing a client).
//!
//! ## Why the small helpers are duplicated, not `mod`-included
//!
//! Same rationale as `gtil_matrix_cubecl.rs` lines 24-31: `#[path]`-including
//! that file would re-run its `#[test] fn gtil_matrix_cubecl()` inside this
//! binary too, and that test hard-gates at `1e-5` (wrong for the GPU path).
//! Duplicating the small decode helpers keeps both siblings byte-identical
//! (D-11) and this one self-contained.
#![cfg(feature = "rocm")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use treelite_core::Model;
use treelite_gtil::{Config, PredictKind, SparseCsr};
use treelite_harness::report::{
    ReportRow, emit, max_abs_delta_report_mode, model_routes_to_scalar_fallback, predicted_band,
};
use treelite_harness::{Manifest, RunnerCase, rocm_case};

// ---------------------------------------------------------------------------
// Fixture-load helpers — copied VERBATIM from gtil_matrix_cubecl.rs (lines
// 62-292), the "duplicate small helpers, don't mod-include" rationale. The
// matrix iteration shape is unchanged; only the comparison (report-mode RECORD
// vs hard gate) and the emission differ.
// ---------------------------------------------------------------------------

/// A frozen GTIL matrix cell — identical shape to `gtil_matrix_cubecl.rs`.
#[derive(Debug, Deserialize)]
struct MatrixGolden {
    #[allow(dead_code)]
    model_path: String,
    n_features: usize,
    input: Vec<Vec<serde_json::Value>>,
    output: serde_json::Value,
    #[allow(dead_code)]
    output_shape: Vec<usize>,
    manifest: MatrixManifest,
    #[serde(default)]
    csr: Option<FrozenCsr>,
}

/// The frozen CSR triple a sparse cell carries (WR-01).
#[derive(Debug, Deserialize)]
struct FrozenCsr {
    data: Vec<serde_json::Value>,
    indices: Vec<u64>,
    indptr: Vec<u64>,
}

/// Capture-environment provenance for a matrix cell (D-09).
#[derive(Debug, Deserialize)]
struct MatrixManifest {
    #[allow(dead_code)]
    backend: String,
    model: String,
    os: String,
    arch: String,
    kind: String,
    layout: String,
    input_dtype: String,
    #[allow(dead_code)]
    seed: u64,
    #[serde(default)]
    rustc: Option<String>,
    #[serde(default)]
    cubecl: Option<String>,
}

/// Resolve a path under the workspace-root `fixtures/` dir.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

/// Resolve a path under the workspace-root `docs/` dir.
fn docs_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs")
        .join(name)
}

/// Decode a possibly-non-finite scalar JSON cell into `f64`.
fn cell_to_f64(v: &serde_json::Value) -> anyhow::Result<f64> {
    match v {
        serde_json::Value::Null => Ok(f64::NAN),
        serde_json::Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("non-f64 number cell")),
        serde_json::Value::String(s) => match s.as_str() {
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            "nan" => Ok(f64::NAN),
            other => anyhow::bail!("unexpected string cell {other:?}"),
        },
        other => anyhow::bail!("unexpected cell type {other:?}"),
    }
}

/// Recursively flatten the nested golden output into a row-major `f64` vector.
fn flatten_output(v: &serde_json::Value, out: &mut Vec<f64>) -> anyhow::Result<()> {
    match v {
        serde_json::Value::Array(a) => {
            for x in a {
                flatten_output(x, out)?;
            }
            Ok(())
        }
        scalar => {
            out.push(cell_to_f64(scalar)?);
            Ok(())
        }
    }
}

/// Map the `manifest.kind` axis token to a typed [`PredictKind`].
fn kind_of(kind: &str) -> anyhow::Result<PredictKind> {
    Ok(match kind {
        "default" => PredictKind::Default,
        "raw" => PredictKind::Raw,
        "leaf_id" => PredictKind::LeafId,
        "score_per_tree" => PredictKind::ScorePerTree,
        other => anyhow::bail!("unknown predict kind {other:?}"),
    })
}

/// Decode the golden input matrix into a row-major `f64` buffer.
fn decode_input_f64(golden: &MatrixGolden, fname: &str) -> anyhow::Result<Vec<f64>> {
    let num_row = golden.input.len();
    anyhow::ensure!(num_row > 0, "{fname}: golden input has zero rows");
    let num_feature = golden.n_features;
    let mut flat: Vec<f64> = Vec::with_capacity(num_row * num_feature);
    for (r, row) in golden.input.iter().enumerate() {
        anyhow::ensure!(
            row.len() == num_feature,
            "{fname}: input row {r} has {} cells, expected {num_feature}",
            row.len()
        );
        for cell in row {
            flat.push(cell_to_f64(cell)?);
        }
    }
    Ok(flat)
}

/// Narrowing of a decoded `f64` CSR value into the cell's input element type.
trait FromF64: Copy {
    fn from_f64(v: f64) -> Self;
}
impl FromF64 for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl FromF64 for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Decode the FROZEN CSR triple (WR-01) into a typed `(data, col_ind, row_ptr)`.
fn frozen_csr<O: FromF64>(frozen: &FrozenCsr) -> anyhow::Result<(Vec<O>, Vec<u64>, Vec<u64>)> {
    let mut data: Vec<O> = Vec::with_capacity(frozen.data.len());
    for cell in &frozen.data {
        data.push(O::from_f64(cell_to_f64(cell)?));
    }
    Ok((data, frozen.indices.clone(), frozen.indptr.clone()))
}

/// Outcome of running one cell: either the produced output (`f64`), or the typed
/// device-absent skip (D-05).
enum CellRun {
    Output(Vec<f64>),
    DeviceAbsent,
}

/// Is this harness error the typed `DeviceUnavailable` skip (D-05)? The harness
/// `rocm_case()` preserves the typed `CubeclError` as a downcastable `anyhow`
/// source (WR-04), so we match on the VARIANT rather than a Display substring.
/// This is robust to error-message wording changes and never misclassifies an
/// unrelated error (e.g. a scalar-fallback `Unsupported` whose text happens to
/// mention "no device available") as a benign skip. ONLY a genuine
/// `CubeclError::DeviceUnavailable` marks the row "not run — no device"; a
/// `CubeclError::ClientInit` (a real init fault, WR-01) is deliberately NOT a
/// skip and propagates as a failure.
fn is_device_absent(err: &anyhow::Error) -> bool {
    matches!(
        err.downcast_ref::<treelite_cubecl::CubeclError>(),
        Some(treelite_cubecl::CubeclError::DeviceUnavailable { .. })
    )
}

/// Run one cell through the correct input-dtype + layout arm of the ROCm
/// [`RunnerCase`] (D-05 dispatch, no pre-cast). Dense cells flow through
/// `dense_*` (the ROCm kernel via `predict::<HipRuntime, _>`); sparse cells flow
/// through `sparse_*` (the scalar fallback, D-02). A device-absent error
/// becomes [`CellRun::DeviceAbsent`] (skip), every OTHER error propagates.
fn run_cell(
    case: &RunnerCase,
    model: &Model,
    golden: &MatrixGolden,
    cfg: &Config,
    fname: &str,
) -> anyhow::Result<CellRun> {
    let num_row = golden.input.len();
    let flat64 = decode_input_f64(golden, fname)?;

    let is_sparse = golden.manifest.layout == "sparse";
    if is_sparse {
        anyhow::ensure!(
            golden.csr.is_some(),
            "{fname}: sparse cell is missing the frozen CSR triple (WR-01); \
             regenerate via capture_gtil_matrix.py"
        );
    }

    let result: anyhow::Result<Vec<f64>> = match golden.manifest.input_dtype.as_str() {
        "f32" => {
            if let Some(frozen) = &golden.csr {
                let (fdata, fcol, frow) = frozen_csr::<f32>(frozen)?;
                let frozen_view = SparseCsr {
                    data: &fdata,
                    col_ind: &fcol,
                    row_ptr: &frow,
                };
                (case.sparse_f32)(model, frozen_view, num_row, cfg)
                    .with_context(|| format!("{fname}: frozen-CSR sparse f32 predict (fallback)"))
            } else {
                let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
                (case.dense_f32)(model, &flat32, num_row, cfg)
                    .with_context(|| format!("{fname}: dense f32 predict (rocm)"))
            }
        }
        "f64" => {
            if let Some(frozen) = &golden.csr {
                let (fdata, fcol, frow) = frozen_csr::<f64>(frozen)?;
                let frozen_view = SparseCsr {
                    data: &fdata,
                    col_ind: &fcol,
                    row_ptr: &frow,
                };
                (case.sparse_f64)(model, frozen_view, num_row, cfg)
                    .with_context(|| format!("{fname}: frozen-CSR sparse f64 predict (fallback)"))
            } else {
                (case.dense_f64)(model, &flat64, num_row, cfg)
                    .with_context(|| format!("{fname}: dense f64 predict (rocm)"))
            }
        }
        other => anyhow::bail!("{fname}: unknown input_dtype {other:?}"),
    };

    match result {
        Ok(out) => Ok(CellRun::Output(out)),
        Err(e) if is_device_absent(&e) => Ok(CellRun::DeviceAbsent),
        Err(e) => Err(e),
    }
}

/// Per-model-class accumulation: the report has ONE row per frozen-golden model
/// class (D-07), so we fold all of a class's cells into a single max `|delta|`.
struct ClassAcc {
    postprocessor: String,
    /// Max ROCm |delta| seen across this class's DENSE (kernel) cells, or `None`
    /// if no ROCm cell ran (device absent everywhere).
    rocm_max: Option<f64>,
    /// Whether this class routes to the scalar fallback / f64 twin (D-02, WR-04).
    f64_fallback_used: bool,
}

/// GPU-04 / D-06: drive every frozen `fixtures/gtil/*.golden.json` cell through
/// the ROCm backend in REPORT MODE (record, never panic — D-01), skip-not-fail
/// on device absence (D-05), and emit the committed
/// `docs/GPU_EQUIVALENCE_REPORT.md` (+ `gpu_equivalence.json`) with one row per
/// frozen-golden model class (D-07), the D-03 predicted band, and CUDA/wgpu
/// rendered "not run — no device". Run explicitly on the ROCm box (`#[ignore]`).
#[test]
#[ignore = "ROCm hardware only — run explicitly to regenerate docs/GPU_EQUIVALENCE_REPORT.md (D-06)"]
fn gtil_matrix_gpu() -> anyhow::Result<()> {
    let dir = fixture_path("gtil");
    let entries = std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?;
    let case = rocm_case();

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fname.ends_with(".golden.json") {
            paths.push(path);
        }
    }
    paths.sort();

    // Per-class accumulation (insertion-ordered by first sight via BTreeMap on
    // the class name → deterministic report row order).
    let mut classes: BTreeMap<String, ClassAcc> = BTreeMap::new();
    // The run manifest header is taken from the first cell we see (every cell
    // carries the same capture host; the backend column is the ROCm run).
    let mut run_manifest: Option<Manifest> = None;
    let mut cells = 0usize;
    let mut ran_cells = 0usize;
    let mut skipped_cells = 0usize;

    for path in &paths {
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let golden: MatrixGolden =
            serde_json::from_str(&raw).with_context(|| format!("parsing {fname}"))?;

        // Load the EXACT model the golden was captured from.
        let model_name = &golden.manifest.model;
        let model_path = fixture_path("gtil").join(format!("{model_name}.model.bin"));
        let model_bytes = std::fs::read(&model_path)
            .with_context(|| format!("{fname}: reading model {}", model_path.display()))?;
        let model = treelite_core::deserialize(&model_bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("{fname}: deserializing model {model_name}"))?;

        if run_manifest.is_none() {
            // Synthesize the run-provenance Manifest header from the golden's
            // capture manifest (D-06): the report records WHERE it ran.
            run_manifest = Some(Manifest {
                treelite: "4.7.0".to_string(),
                xgboost: None,
                os: golden.manifest.os.clone(),
                arch: golden.manifest.arch.clone(),
                libc: serde_json::Value::Null,
                python: None,
                backend: "rocm".to_string(),
                rustc: golden.manifest.rustc.clone(),
                cubecl: golden.manifest.cubecl.clone(),
                seed: None,
                sha256: None,
                numpy: None,
                scipy: None,
                lightgbm: None,
                scikit_learn: None,
                model: Some(model_name.clone()),
                preset: None,
                input_dtype: None,
                kind: None,
                layout: None,
            });
        }

        let cfg = Config {
            kind: kind_of(&golden.manifest.kind)?,
            nthread: 0,
        };

        let mut expected: Vec<f64> = Vec::new();
        flatten_output(&golden.output, &mut expected)?;

        // Whether this whole model routes to the scalar fallback / f64 twin
        // (D-02, WR-04) — read from the SAME predicate the engine consults.
        let routes_to_fallback = model_routes_to_scalar_fallback(&model);
        let is_sparse = golden.manifest.layout == "sparse";

        let acc = classes
            .entry(model_name.clone())
            .or_insert_with(|| ClassAcc {
                postprocessor: model.postprocessor.clone(),
                rocm_max: None,
                f64_fallback_used: false,
            });
        // The class is flagged as using the scalar/f64 fallback if ANY of its
        // cells route there (sparse layout, or a categorical/non-kLT model).
        acc.f64_fallback_used |= routes_to_fallback || is_sparse;

        // Run the cell on ROCm (report mode — RECORD, never panic on >1e-5).
        match run_cell(&case, &model, &golden, &cfg, fname)? {
            CellRun::Output(own) => {
                let max_dev = max_abs_delta_report_mode(&own, &expected);
                // Only finite recorded deviations contribute to the class max
                // (a length-mismatch sentinel NaN is surfaced, not folded in).
                if max_dev.is_finite() {
                    let cur = acc.rocm_max.unwrap_or(0.0);
                    if max_dev > cur {
                        acc.rocm_max = Some(max_dev);
                    } else if acc.rocm_max.is_none() {
                        acc.rocm_max = Some(cur);
                    }
                }
                eprintln!(
                    "{fname} [{}/{}/{}] (rocm): max |delta| = {max_dev:e} (RECORDED, observational)",
                    golden.manifest.input_dtype, golden.manifest.kind, golden.manifest.layout,
                );
                ran_cells += 1;
            }
            CellRun::DeviceAbsent => {
                eprintln!("{fname} (rocm): not run — no device (skip, D-05)");
                skipped_cells += 1;
            }
        }
        cells += 1;
    }

    anyhow::ensure!(cells > 0, "no fixtures/gtil/*.golden.json cells found");

    // Build the per-class report rows (D-07: one row per frozen model class).
    // CUDA/wgpu render "not run — no device" (None) — this is the rocm-feature
    // build; their *_case() constructors are not compiled in here, so their
    // columns are recorded as not-run WITHOUT constructing a client (T-07-09).
    let mut rows: Vec<ReportRow> = Vec::new();
    for (model_class, acc) in &classes {
        let (predicted_low, predicted_high) = predicted_band(&acc.postprocessor);
        rows.push(ReportRow {
            model_class: model_class.clone(),
            postprocessor: acc.postprocessor.clone(),
            rocm_max_abs_delta: acc.rocm_max,
            f64_fallback_used: acc.f64_fallback_used,
            cuda_max_abs_delta: None,
            wgpu_max_abs_delta: None,
            predicted_low,
            predicted_high,
        });
    }

    let manifest = run_manifest.expect("at least one cell was read");
    // Device name: best-effort from HIP_VISIBLE_DEVICES / a generic AMD label.
    // The developer's actual device string is surfaced via the env if set.
    let device_name = std::env::var("TREELITE_GPU_DEVICE")
        .unwrap_or_else(|_| "AMD ROCm device (set TREELITE_GPU_DEVICE to label)".to_string());

    let report_md = docs_path("GPU_EQUIVALENCE_REPORT.md");
    emit(&rows, &manifest, &device_name, &report_md)?;

    eprintln!(
        "gtil_matrix_gpu: {cells} cells ({ran_cells} ran on ROCm, {skipped_cells} skipped — \
         no device), {} model-class rows written to {} (+ gpu_equivalence.json). \
         OBSERVATIONAL — records |delta| without gating (D-01).",
        rows.len(),
        report_md.display(),
    );
    Ok(())
}

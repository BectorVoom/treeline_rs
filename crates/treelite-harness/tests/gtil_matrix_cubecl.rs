//! Wave 4 (plan 06-05) — the cubecl-CPU-backend matrix sibling (GPU-02), GREEN.
//!
//! This is a SIBLING of `gtil_matrix.rs`, NOT a refactor of it (D-11 smell
//! guard): a thin new file that registers [`treelite_harness::cubecl_cpu_case`]
//! and drives the SAME frozen `fixtures/gtil/*.golden.json` cross-product
//! (model × preset × input-dtype f32/f64 × predict kind × {dense,sparse} ×
//! seed) through the cubecl CPU backend, asserting every output element within
//! `1e-5` of the identical upstream goldens the scalar reference uses — never a
//! regenerated vector (T-06-14). It runs the dense numerical cells on the
//! cubecl kernels and the sparse cells on the scalar fallback (D-02).
//!
//! ## Provenance contract (D-06, T-06-12)
//!
//! This sibling asserts its OWN per-cell provenance, NEVER the
//! `golden.manifest.backend == "scalar-cpu"` literal at `gtil_matrix.rs:474`
//! (that assert belongs to the scalar reference test only). Each cell's
//! EXECUTED path is tagged at assertion time: a dense numerical cell that ran
//! the cubecl kernel is `"cubecl-kernel"`; a sparse cell, or a categorical
//! model that `predict_cpu` itself routes to the scalar fallback, is
//! `"scalar-fallback"`. So a `1e-5`-on-cubecl pass can never silently mean
//! "validated on the scalar fallback" — the gate records which engine actually
//! produced each asserted vector and requires at least one true kernel cell.
//!
//! ## Why the small helpers are duplicated, not `mod`-included
//!
//! The `run_cell`/golden-load helpers live as private items in
//! `gtil_matrix.rs`. `#[path]`-including that file would re-run its `#[test]
//! fn gtil_matrix()` inside this binary too; duplicating the small decode
//! helpers keeps `gtil_matrix.rs` byte-identical (D-11 `git diff --stat` == 0)
//! and this sibling self-contained. The duplicated logic is the dtype/layout
//! DISPATCH body only — the matrix iteration shape is unchanged.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use treelite_core::Model;
use treelite_gtil::{Config, PredictKind, SparseCsr};
use treelite_harness::{RunnerCase, cubecl_cpu_case};

/// Per-cell provenance (D-06): which engine actually produced the asserted
/// vector. Recorded at assertion time from the EXECUTED path, never from a
/// trusted manifest literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provenance {
    /// The dense numerical cubecl kernel produced this vector.
    CubeclKernel,
    /// The scalar `treelite_gtil` fallback produced this vector (sparse layout,
    /// or a categorical model `predict_cpu` routed to the fallback, D-02).
    ScalarFallback,
}

impl Provenance {
    fn as_str(self) -> &'static str {
        match self {
            Provenance::CubeclKernel => "cubecl-kernel",
            Provenance::ScalarFallback => "scalar-fallback",
        }
    }
}

/// A frozen GTIL matrix cell — identical shape to `gtil_matrix.rs::MatrixGolden`
/// (the SAME frozen fixtures are read; no schema change).
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
    /// The manifest's recorded backend (the SCALAR reference froze these as
    /// `scalar-cpu`). This sibling does NOT assert this literal — provenance is
    /// recorded from the executed path (D-06); the manifest backend is read only
    /// to drive the `check_manifest` drift warning.
    backend: String,
    model: String,
    os: String,
    arch: String,
    kind: String,
    layout: String,
    input_dtype: String,
    #[allow(dead_code)]
    seed: u64,
}

/// Resolve a path under the workspace-root `fixtures/` dir.
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

/// Warn (never fail) on capture-vs-running environment drift (D-09). Mirrors
/// `gtil_matrix.rs::check_manifest`; never a gate.
fn check_manifest(manifest: &MatrixManifest) {
    let running_os = std::env::consts::OS;
    let running_arch = std::env::consts::ARCH;
    if !manifest.os.to_lowercase().contains(running_os) {
        eprintln!(
            "WARNING: GTIL matrix golden captured on OS '{}' but running on '{}' — \
             a 1e-5 deviation here may be a libm/environment divergence (D-09).",
            manifest.os, running_os
        );
    }
    if manifest.arch.to_lowercase() != running_arch.to_lowercase() {
        eprintln!(
            "WARNING: GTIL matrix golden captured on arch '{}' but running on '{}' — \
             a 1e-5 deviation here may be an environment divergence (D-09).",
            manifest.arch, running_arch
        );
    }
    // Backend-drift note (D-09): these fixtures were frozen as `scalar-cpu`; the
    // cubecl sibling asserts the SAME goldens cross-backend. This is the intended
    // cross-backend assertion, surfaced for diagnosability — NEVER a gate.
    if manifest.backend != "scalar-cpu" {
        eprintln!(
            "WARNING: GTIL matrix golden manifest backend is '{}' (expected the frozen \
             'scalar-cpu') — cubecl sibling asserting cross-backend (D-09).",
            manifest.backend
        );
    }
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

/// Map the `manifest.kind` axis token to a typed [`PredictKind`] (D-06).
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

/// Narrowing of a decoded `f64` CSR value into the cell's input element type,
/// WITHOUT changing the value (exact-or-NaN round-trip, no pre-cast).
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

/// Run one cell through the correct input-dtype + layout arm of the cubecl
/// [`RunnerCase`] (D-05 dispatch, no pre-cast), returning the `f64` output of the
/// fixture's OWN layout. Dense cells flow through `dense_*` (the cubecl kernel);
/// sparse cells flow through `sparse_*` (the scalar fallback, D-02). The matrix
/// iteration shape is unchanged from `gtil_matrix.rs::run_cell` — this is the
/// dispatch body only, constructed against the cubecl case.
fn run_cell(
    case: &RunnerCase,
    model: &Model,
    golden: &MatrixGolden,
    cfg: &Config,
    fname: &str,
) -> anyhow::Result<Vec<f64>> {
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

    match golden.manifest.input_dtype.as_str() {
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
                // Narrow to f32 WITHOUT changing any value (exact-or-NaN). The
                // predict runs in f32 (no f32->f64 pre-cast — Pitfall 6/1); the
                // cubecl_cpu_case dense_f32 slot widens the f32 RESULT afterwards.
                let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
                (case.dense_f32)(model, &flat32, num_row, cfg)
                    .with_context(|| format!("{fname}: dense f32 predict (cubecl)"))
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
                    .with_context(|| format!("{fname}: dense f64 predict (cubecl)"))
            }
        }
        other => anyhow::bail!("{fname}: unknown input_dtype {other:?}"),
    }
}

/// Assert two f64 vectors agree element-wise, treating NaN==NaN and ±inf
/// structurally and finite cells within `eps`. Returns the max finite |delta|.
fn assert_within(got: &[f64], want: &[f64], eps: f64, ctx: &str) -> anyhow::Result<f64> {
    anyhow::ensure!(
        got.len() == want.len(),
        "{ctx}: length {} != {}",
        got.len(),
        want.len()
    );
    let mut max_dev = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g.is_nan() || w.is_nan() {
            anyhow::ensure!(
                g.is_nan() == w.is_nan(),
                "{ctx}: cell {i} NaN mismatch (got {g}, want {w})"
            );
            continue;
        }
        if !g.is_finite() || !w.is_finite() {
            anyhow::ensure!(g == w, "{ctx}: cell {i} inf mismatch (got {g}, want {w})");
            continue;
        }
        let delta = (g - w).abs();
        if delta > max_dev {
            max_dev = delta;
        }
        // HARD gate — never loosen to mask a real fidelity gap.
        approx::assert_abs_diff_eq!(g, w, epsilon = eps);
    }
    Ok(max_dev)
}

/// Does `model` route the WHOLE model to the scalar fallback inside
/// `predict_cpu`?
///
/// WR-04: this is NOT a re-derived parallel copy of the gate. It delegates to
/// [`treelite_cubecl::model_routes_to_scalar_fallback`] — the SAME predicate
/// `predict_cpu` itself consults to decide whether to defer to the scalar
/// reference (D-02: categorical split OR any internal node with a non-`kLT`
/// operator). Because both the executed routing decision and this provenance tag
/// read the one function, they cannot drift (the "green while buggy" failure D-06
/// guards against). Such a cell ran the scalar fallback, so its provenance is
/// `"scalar-fallback"` — observed from the executed path (T-06-12), never the
/// layout alone.
fn model_routes_to_fallback(model: &Model) -> bool {
    treelite_cubecl::model_routes_to_scalar_fallback(model)
}

/// GPU-02 / D-06: drive every frozen `fixtures/gtil/*.golden.json` cell through
/// the CUBECL CPU backend, assert within `1e-5` of the IDENTICAL upstream golden
/// the scalar reference uses, and record per-cell `cubecl-kernel` vs
/// `scalar-fallback` provenance (never the `scalar-cpu` manifest literal).
#[test]
fn gtil_matrix_cubecl() -> anyhow::Result<()> {
    let dir = fixture_path("gtil");
    let entries = std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?;
    let case = cubecl_cpu_case();

    let mut cells = 0usize;
    let mut kernel_cells = 0usize;
    let mut fallback_cells = 0usize;
    let mut f32_cells = 0usize;
    let mut f64_cells = 0usize;
    let mut global_max_dev: f64 = 0.0;

    // Stable iteration order so the per-cell report is deterministic — the SAME
    // iteration shape as gtil_matrix.rs (D-11: no reshaping of the cross-product).
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if fname.ends_with(".golden.json") {
            paths.push(path);
        }
    }
    paths.sort();

    for path in &paths {
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let golden: MatrixGolden =
            serde_json::from_str(&raw).with_context(|| format!("parsing {fname}"))?;

        // NOTE (T-06-12): we DELIBERATELY do NOT assert
        // `golden.manifest.backend == "scalar-cpu"` (the gtil_matrix.rs:474
        // literal). Provenance for THIS gate is the EXECUTED path, recorded
        // below — not the manifest field.
        check_manifest(&golden.manifest);

        // Load the EXACT model the golden was captured from.
        let model_name = &golden.manifest.model;
        let model_path = fixture_path("gtil").join(format!("{model_name}.model.bin"));
        let model_bytes = std::fs::read(&model_path)
            .with_context(|| format!("{fname}: reading model {}", model_path.display()))?;
        let model = treelite_core::deserialize(&model_bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("{fname}: deserializing model {model_name}"))?;

        let cfg = Config {
            kind: kind_of(&golden.manifest.kind)?,
            nthread: 0,
        };

        // Decode the golden output vector (nested, NaN/inf tolerant).
        let mut expected: Vec<f64> = Vec::new();
        flatten_output(&golden.output, &mut expected)?;

        // Run the Rust prediction through the cubecl dtype + layout arm.
        let own = run_cell(&case, &model, &golden, &cfg, fname)?;

        // --- 1e-5 golden gate (GPU-02) on the IDENTICAL frozen goldens --------
        let max_dev = assert_within(&own, &expected, 1e-5, fname)
            .with_context(|| format!("{fname}: 1e-5 golden gate (cubecl)"))?;

        // --- D-06 per-cell provenance (recorded from the EXECUTED path) -------
        // A sparse cell runs the scalar fallback (D-02). A dense cell runs the
        // cubecl kernel UNLESS `predict_cpu` itself routed the whole (categorical)
        // model to the scalar fallback — in which case it is also a fallback. The
        // tag reflects what ACTUALLY ran (T-06-12), never the layout alone.
        let is_sparse = golden.manifest.layout == "sparse";
        let provenance = if is_sparse || model_routes_to_fallback(&model) {
            Provenance::ScalarFallback
        } else {
            Provenance::CubeclKernel
        };
        match provenance {
            Provenance::CubeclKernel => kernel_cells += 1,
            Provenance::ScalarFallback => fallback_cells += 1,
        }

        eprintln!(
            "{fname} [{}/{}/{}] ({}): max |delta| = {max_dev:e} (< 1e-5)",
            golden.manifest.input_dtype,
            golden.manifest.kind,
            golden.manifest.layout,
            provenance.as_str(),
        );

        if max_dev > global_max_dev {
            global_max_dev = max_dev;
        }
        cells += 1;
        match golden.manifest.input_dtype.as_str() {
            "f32" => f32_cells += 1,
            "f64" => f64_cells += 1,
            _ => {}
        }
    }

    anyhow::ensure!(cells > 0, "no fixtures/gtil/*.golden.json cells found");
    anyhow::ensure!(f32_cells > 0, "no f32-input cells exercised (D-05)");
    anyhow::ensure!(f64_cells > 0, "no f64-input cells exercised (D-05)");
    // T-06-12: the gate must have validated at least one TRUE cubecl-kernel cell —
    // otherwise "1e-5 on cubecl-cpu" would silently mean "validated on the scalar
    // fallback only". The dense numerical cells are the kernel-provenance cells.
    anyhow::ensure!(
        kernel_cells > 0,
        "no cubecl-kernel cells exercised — the 1e-5 gate validated ONLY the scalar \
         fallback, which cannot credit the cubecl backend (D-06/T-06-12)"
    );

    eprintln!(
        "gtil_matrix_cubecl: {cells} cells ({f32_cells} f32-input, {f64_cells} f64-input), \
         provenance: {kernel_cells} cubecl-kernel + {fallback_cells} scalar-fallback, \
         global max |delta| = {global_max_dev:e} (< 1e-5) — the cubecl CPU backend matches \
         the IDENTICAL frozen goldens the scalar reference uses (GPU-02)."
    );
    Ok(())
}

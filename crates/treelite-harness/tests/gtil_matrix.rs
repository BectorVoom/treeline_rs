//! The exhaustive GTIL equivalence-matrix runner (EQV-01..04) — GREEN.
//!
//! This test drives the frozen `fixtures/gtil/*.golden.json` cross-product
//! (model × preset × input-dtype f32/f64 × predict kind × {dense,sparse} ×
//! seed, captured by Plan 05-01) against the Rust GTIL engine and asserts every
//! output element is within `1e-5` of the upstream `treelite.gtil.*` golden —
//! while tracking the max observed `|delta|` per cell (EQV-04). It ALSO asserts
//! dense == sparse parity on identical logical data (D-04).
//!
//! ## How a cell is run
//!
//! 1. Parse the fixture; load the committed treelite v5 model bytes
//!    (`fixtures/gtil/<model>.model.bin`) named by `manifest.model` via
//!    [`treelite_core::deserialize`] — the EXACT model the goldens were captured
//!    from (Plan 05-01 authored the model in-script and discarded it; the model
//!    bytes were re-frozen by `fixtures/capture_gtil_models.py`, verified to
//!    reproduce every golden to max |delta| == 0.0 in Python).
//! 2. Build a typed `Config` from `manifest.kind`.
//! 3. DISPATCH ON `manifest.input_dtype` (D-05, RESEARCH Pitfall 1): an `f32`
//!    fixture flows through the f32 entry point of the [`scalar_cpu_case`]
//!    [`RunnerCase`] with NO f32→f64 pre-cast; an `f64` fixture flows through
//!    the f64 entry point. Output is uniformly `f64` for comparison.
//! 4. DISPATCH ON `manifest.layout`: `dense` → `dense_*`; `sparse` →
//!    reconstruct the CSR from the dense-with-NaN input's PRESENT (non-NaN)
//!    cells (this is exactly the capture-time construction, predict.cc:80-85)
//!    and call `sparse_*`.
//! 5. Assert every output element within `1e-5` of the golden (`approx`,
//!    epsilon = 1e-5 — the HARD gate, never loosened to mask a real gap).
//! 6. D-04 parity: for every cell, also run the dense path on the
//!    dense-with-NaN input and the sparse path on the reconstructed CSR and
//!    assert the two Rust paths agree (independent of the golden assert).
//!
//! The `backend` is the scalar-cpu reference (manifest `backend ==
//! "scalar-cpu"`); the runner is driven through the [`RunnerCase`] seam so
//! Phase 6 registers a cubecl runtime without touching this iteration.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;
use treelite_core::Model;
use treelite_gtil::{Config, PredictKind, SparseCsr};
use treelite_harness::{RunnerCase, scalar_cpu_case};

/// A frozen GTIL matrix cell (`{model_path, n_features, input, output,
/// output_shape, manifest, sha256}`). `input`/`output` are tolerant
/// `serde_json::Value` cells because the capture encodes non-finite values as
/// JSON `null` (NaN) or the strings `"inf"`/`"-inf"`, and `output` is nested
/// (`output_shape`-dimensional).
#[derive(Debug, Deserialize)]
struct MatrixGolden {
    /// Identifier for the in-script captured model (provenance only).
    #[allow(dead_code)]
    model_path: String,
    /// Number of input features (one column per feature).
    n_features: usize,
    /// Row-major input matrix; cells may be `null`/`"inf"`/`"-inf"` (edge-seeded).
    input: Vec<Vec<serde_json::Value>>,
    /// Frozen upstream `treelite.gtil.*` output; nested per `output_shape`, cells
    /// may be non-finite tokens.
    output: serde_json::Value,
    /// Output shape per the captured kind (`GetOutputShape`).
    #[allow(dead_code)]
    output_shape: Vec<usize>,
    /// Full-provenance manifest carrying the `backend`/`model`/`kind`/`layout`/
    /// `input_dtype`/`seed` axes (D-09).
    manifest: MatrixManifest,
}

/// Capture-environment provenance for a matrix cell (D-09). Carries the axis
/// tags the runner branches on.
#[derive(Debug, Deserialize)]
struct MatrixManifest {
    /// Which `R: Runtime` produced/asserts the vector (`scalar-cpu` this phase).
    backend: String,
    /// Model axis tag (`binary`/`leaf_vec_mc`) — names the `<model>.model.bin`.
    model: String,
    /// `platform.platform()` string.
    os: String,
    /// `platform.machine()` (e.g. `x86_64`).
    arch: String,
    /// Predict kind axis: `default`/`raw`/`leaf_id`/`score_per_tree`.
    kind: String,
    /// Layout axis: `dense`/`sparse`.
    layout: String,
    /// Input-dtype axis: `f32`/`f64`.
    input_dtype: String,
}

/// Resolve a path under the workspace-root `fixtures/` dir (mirrors
/// `tests/lightgbm.rs:fixture_path`).
fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
}

/// Warn (never fail) when the running environment differs from the capture
/// environment (D-09): a `1e-5` failure on a different distro is most often a
/// libm/glibc divergence, and surfacing the drift makes it diagnosable.
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
}

/// Decode a possibly-non-finite scalar JSON cell into `f64` (NaN for `null`,
/// ±inf for the `"inf"`/`"-inf"` string tokens the capture emits).
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

/// Recursively flatten the nested (`output_shape`-dimensional) golden output
/// into a row-major `f64` vector, decoding non-finite tokens.
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

/// Decode the golden input matrix into a row-major `f64` buffer (one logical
/// value per cell, NaN for absent/`null`). This is the common decode; the
/// per-dtype dispatch narrows it to `f32`/`f64` WITHOUT changing any value
/// (every cell is an exact-or-NaN round-trip).
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

/// Reconstruct a CSR (`data`, `col_ind`, `row_ptr`) from a row-major
/// dense-with-NaN buffer: PRESENT cells are the non-NaN entries (exactly the
/// capture-time construction — absent == NaN, predict.cc:80-85). Generic over
/// the input element type so the f32 and f64 sparse arms reuse it.
fn build_csr<O: Copy + IsNan>(
    flat: &[O],
    num_row: usize,
    num_feature: usize,
) -> (Vec<O>, Vec<u64>, Vec<u64>) {
    let mut data: Vec<O> = Vec::new();
    let mut col_ind: Vec<u64> = Vec::new();
    let mut row_ptr: Vec<u64> = Vec::with_capacity(num_row + 1);
    row_ptr.push(0);
    for r in 0..num_row {
        for c in 0..num_feature {
            let v = flat[r * num_feature + c];
            if !v.is_nan_() {
                data.push(v);
                col_ind.push(c as u64);
            }
        }
        row_ptr.push(data.len() as u64);
    }
    (data, col_ind, row_ptr)
}

/// Minimal `is_nan` over the two input element types (so `build_csr` is generic
/// without pulling in `num_traits`).
trait IsNan {
    fn is_nan_(&self) -> bool;
}
impl IsNan for f32 {
    fn is_nan_(&self) -> bool {
        f32::is_nan(*self)
    }
}
impl IsNan for f64 {
    fn is_nan_(&self) -> bool {
        f64::is_nan(*self)
    }
}

/// Run one cell's Rust prediction through the correct input-dtype + layout arm
/// of the [`RunnerCase`] (D-05 dispatch, no pre-cast), returning the `f64`
/// output AND — for the D-04 parity assert — the OTHER-layout result on
/// identical logical data.
///
/// Returns `(golden_path_output, dense_output, sparse_output)`:
/// - `golden_path_output` is the result of the fixture's OWN layout (what the
///   golden is asserted against);
/// - `dense_output`/`sparse_output` are BOTH layouts on the same dtype + same
///   logical data, for the dense==sparse parity assert.
fn run_cell(
    case: &RunnerCase,
    model: &Model,
    golden: &MatrixGolden,
    cfg: &Config,
    fname: &str,
) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let num_row = golden.input.len();
    let num_feature = golden.n_features;
    let flat64 = decode_input_f64(golden, fname)?;

    match golden.manifest.input_dtype.as_str() {
        "f32" => {
            // Narrow to f32 WITHOUT changing any value (exact-or-NaN). This is
            // the input-dtype axis: the predict runs in f32 (no f32→f64 pre-cast
            // — RESEARCH Pitfall 1).
            let flat32: Vec<f32> = flat64.iter().map(|&v| v as f32).collect();
            let dense = (case.dense_f32)(model, &flat32, num_row, cfg)
                .with_context(|| format!("{fname}: dense f32 predict"))?;
            let (data, col_ind, row_ptr) = build_csr(&flat32, num_row, num_feature);
            let csr = SparseCsr {
                data: &data,
                col_ind: &col_ind,
                row_ptr: &row_ptr,
            };
            let sparse = (case.sparse_f32)(model, csr, num_row, cfg)
                .with_context(|| format!("{fname}: sparse f32 predict"))?;
            let own = if golden.manifest.layout == "sparse" {
                sparse.clone()
            } else {
                dense.clone()
            };
            Ok((own, dense, sparse))
        }
        "f64" => {
            let dense = (case.dense_f64)(model, &flat64, num_row, cfg)
                .with_context(|| format!("{fname}: dense f64 predict"))?;
            let (data, col_ind, row_ptr) = build_csr(&flat64, num_row, num_feature);
            let csr = SparseCsr {
                data: &data,
                col_ind: &col_ind,
                row_ptr: &row_ptr,
            };
            let sparse = (case.sparse_f64)(model, csr, num_row, cfg)
                .with_context(|| format!("{fname}: sparse f64 predict"))?;
            let own = if golden.manifest.layout == "sparse" {
                sparse.clone()
            } else {
                dense.clone()
            };
            Ok((own, dense, sparse))
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

/// EQV-01..04 / D-04: drive every frozen `fixtures/gtil/*.golden.json` cell
/// through the Rust GTIL engine, assert within `1e-5` of the upstream golden,
/// and assert dense == sparse parity. GREEN (the full GTIL surface is wired:
/// f32/f64 input, all 4 kinds, dense + sparse, typed `Config`).
#[test]
fn gtil_matrix() -> anyhow::Result<()> {
    let dir = fixture_path("gtil");
    let entries = std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?;
    let case = scalar_cpu_case();

    let mut cells = 0usize;
    let mut f32_cells = 0usize;
    let mut f64_cells = 0usize;
    let mut sparse_cells = 0usize;
    let mut parity_cells = 0usize;
    let mut global_max_dev: f64 = 0.0;

    // Stable iteration order so the per-cell report is deterministic.
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

        // Provenance: every cell must be the scalar-cpu reference this phase.
        anyhow::ensure!(
            golden.manifest.backend == "scalar-cpu",
            "{fname}: unexpected backend {:?}",
            golden.manifest.backend
        );
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

        // Run the Rust prediction through the correct dtype + layout arm.
        let (own, dense, sparse) = run_cell(&case, &model, &golden, &cfg, fname)?;

        // --- 1e-5 golden gate (EQV-03/EQV-04) -----------------------------
        // The fixture's OWN-layout result is asserted against its OWN golden
        // (an f32 fixture against the f32 golden, never a re-cast f64 golden).
        let max_dev = assert_within(&own, &expected, 1e-5, fname)
            .with_context(|| format!("{fname}: 1e-5 golden gate"))?;
        eprintln!(
            "{fname} [{}/{}/{}]: max |delta| = {max_dev:e} (< 1e-5)",
            golden.manifest.input_dtype, golden.manifest.kind, golden.manifest.layout
        );

        // --- D-04 dense == sparse parity (independent of the golden) -------
        // Both Rust paths on identical logical data + identical input dtype.
        // `leaf_id`/`score_per_tree` are integer-exact; default/raw within a
        // very tight tolerance (the same accumulator, same traversal).
        let parity_dev = assert_within(
            &dense,
            &sparse,
            1e-9,
            &format!("{fname}: dense==sparse parity (D-04)"),
        )?;
        anyhow::ensure!(
            parity_dev <= 1e-9,
            "{fname}: dense==sparse parity exceeded 1e-9 (got {parity_dev:e})"
        );
        parity_cells += 1;

        if max_dev > global_max_dev {
            global_max_dev = max_dev;
        }
        cells += 1;
        match golden.manifest.input_dtype.as_str() {
            "f32" => f32_cells += 1,
            "f64" => f64_cells += 1,
            _ => {}
        }
        if golden.manifest.layout == "sparse" {
            sparse_cells += 1;
        }
    }

    anyhow::ensure!(cells > 0, "no fixtures/gtil/*.golden.json cells found");
    anyhow::ensure!(f32_cells > 0, "no f32-input cells exercised (D-05)");
    anyhow::ensure!(f64_cells > 0, "no f64-input cells exercised (D-05)");
    anyhow::ensure!(sparse_cells > 0, "no sparse cells exercised (GTIL-02/D-04)");
    eprintln!(
        "gtil_matrix: {cells} cells ({f32_cells} f32-input, {f64_cells} f64-input, \
         {sparse_cells} sparse), {parity_cells} dense==sparse parity asserts, \
         global max |delta| = {global_max_dev:e} (< 1e-5)"
    );
    Ok(())
}

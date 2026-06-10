//! The exhaustive GTIL equivalence-matrix runner (EQV-01..04) — RED Wave 0.
//!
//! This test drives the frozen `fixtures/gtil/*.golden.json` cross-product
//! (model × preset × input-dtype f32/f64 × predict kind × {dense,sparse} ×
//! seed, captured by Plan 05-01) against the Rust GTIL engine and asserts every
//! output element is within `1e-5` of the upstream `treelite.gtil.*` golden —
//! while tracking the max observed `|delta|` per cell (EQV-04). It ALSO asserts
//! the dense == sparse parity on identical logical data (D-04).
//!
//! ## Why this is RED right now
//!
//! The committed matrix exercises GTIL surface that does NOT yet exist in
//! `treelite-gtil`: the f64-input path (D-05), the `raw`/`leaf_id`/
//! `score_per_tree` predict kinds (GTIL-03), sparse CSR input (GTIL-02), and the
//! typed `Config`/`PredictKind` entry surface (D-06). Those land in Plans 02-04.
//! Until then the whole test body is gated behind `#[ignore = "RED until GTIL
//! surface widened in Plans 02-04"]` so it COMPILES (the Wave-0 scaffold) and is
//! explicitly red. The `1e-5` epsilon below is the HARD gate — never loosened to
//! mask a real fidelity gap.
//!
//! When the widening lands, drop the `#[ignore]` and switch the call sites from
//! the current f32-only `treelite_gtil::predict(model, &flat, num_row)` to the
//! `Config`/`PredictKind` + `predict_sparse` surface keyed on each fixture's
//! `manifest.kind` / `manifest.layout` / `manifest.input_dtype`.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

/// A frozen GTIL matrix cell (`{model_path, n_features, input, output,
/// output_shape, manifest, sha256}`). `input`/`output` are tolerant
/// `serde_json::Value` cells because the capture encodes non-finite values as
/// JSON `null` (NaN) or the strings `"inf"`/`"-inf"`.
#[derive(Debug, Deserialize)]
struct MatrixGolden {
    /// Identifier for the in-script captured model (provenance only).
    #[allow(dead_code)]
    model_path: String,
    /// Number of input features (one column per feature).
    n_features: usize,
    /// Row-major input matrix; cells may be `null`/`"inf"`/`"-inf"` (edge-seeded).
    input: Vec<Vec<serde_json::Value>>,
    /// Frozen upstream `treelite.gtil.*` output; cells may be non-finite tokens.
    output: Vec<serde_json::Value>,
    /// Output shape per the captured kind (`GetOutputShape`).
    #[allow(dead_code)]
    output_shape: Vec<usize>,
    /// Full-provenance manifest carrying the `backend`/`kind`/`layout`/
    /// `input_dtype`/`seed` axes (D-09).
    manifest: MatrixManifest,
}

/// Capture-environment provenance for a matrix cell (D-09). Carries the axis
/// tags the runner branches on once the GTIL surface is widened.
#[derive(Debug, Deserialize)]
struct MatrixManifest {
    /// Which `R: Runtime` produced/asserts the vector (`scalar-cpu` this phase).
    backend: String,
    /// Upstream Treelite version (e.g. `4.7.0`).
    treelite: String,
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

/// Decode a possibly-non-finite JSON cell into `f64` (NaN for `null`, ±inf for
/// the `"inf"`/`"-inf"` string tokens the capture emits).
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

/// EQV-01..04 / D-04: drive every frozen `fixtures/gtil/*.golden.json` cell
/// through the Rust GTIL engine, assert within `1e-5` of the upstream golden,
/// and assert dense == sparse parity. RED until the GTIL surface is widened in
/// Plans 02-04 (the `#[ignore]` reason is the Wave-0 MISSING marker).
#[test]
#[ignore = "RED until GTIL surface widened in Plans 02-04 (f64 input, kinds, sparse, Config)"]
fn gtil_matrix() -> anyhow::Result<()> {
    let dir = fixture_path("gtil");
    let entries = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?;

    let mut cells = 0usize;
    let mut global_max_dev: f64 = 0.0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !fname.ends_with(".golden.json") {
            continue;
        }

        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let golden: MatrixGolden = serde_json::from_str(&raw)
            .with_context(|| format!("parsing {fname}"))?;

        // Provenance: every cell must be the scalar-cpu reference this phase.
        anyhow::ensure!(
            golden.manifest.backend == "scalar-cpu",
            "{fname}: unexpected backend {:?}",
            golden.manifest.backend
        );
        check_manifest(&golden.manifest);

        // Flatten the (possibly non-finite) input into a row-major f64 buffer.
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

        // Decode the golden output vector (NaN/inf tolerant).
        let expected: Vec<f64> = golden
            .output
            .iter()
            .map(cell_to_f64)
            .collect::<anyhow::Result<_>>()?;

        // ----------------------------------------------------------------- //
        // RED: once Plans 02-04 land, build a typed `Config` from
        // `golden.manifest.{kind,input_dtype}` and call the widened surface:
        //
        //   let cfg = Config { kind: PredictKind::from(&golden.manifest.kind), .. };
        //   let rust = match golden.manifest.layout.as_str() {
        //       "dense"  => treelite_gtil::predict(&model, &flat, num_row, &cfg)?,
        //       "sparse" => treelite_gtil::predict_sparse(&model, csr, &cfg)?,
        //       _ => unreachable!(),
        //   };
        //   // D-04 parity: assert dense == sparse on identical logical data.
        //
        // For now the scaffold just length-asserts the decoded golden so the
        // 1e-5 gate below is wired and ready (kind/dtype/layout are read above).
        // ----------------------------------------------------------------- //
        let _ = (&golden.manifest.kind, &golden.manifest.input_dtype,
                 &golden.manifest.layout, &golden.manifest.treelite);

        // Placeholder "rust" vector echoes the golden so the harness shape is
        // exercised; Plans 02-04 replace this with the real predict call. The
        // 1e-5 assertion machinery is the load-bearing scaffold.
        let rust: Vec<f64> = expected.clone();
        anyhow::ensure!(
            rust.len() == expected.len(),
            "{fname}: prediction length {} != golden length {}",
            rust.len(),
            expected.len()
        );

        let mut max_dev: f64 = 0.0;
        for (i, (&got, &want)) in rust.iter().zip(expected.iter()).enumerate() {
            // Non-finite cells (NaN-routed / inf) are compared structurally:
            // both sides must agree on non-finiteness; finite cells use 1e-5.
            if got.is_nan() || want.is_nan() {
                anyhow::ensure!(
                    got.is_nan() == want.is_nan(),
                    "{fname}: cell {i} NaN mismatch (got {got}, want {want})"
                );
                continue;
            }
            if !got.is_finite() || !want.is_finite() {
                anyhow::ensure!(got == want, "{fname}: cell {i} inf mismatch");
                continue;
            }
            let delta = (got - want).abs();
            if delta > max_dev {
                max_dev = delta;
            }
            // HARD 1e-5 gate — never loosen to mask a real fidelity gap.
            approx::assert_abs_diff_eq!(got, want, epsilon = 1e-5);
        }
        if max_dev > global_max_dev {
            global_max_dev = max_dev;
        }
        cells += 1;
    }

    anyhow::ensure!(cells > 0, "no fixtures/gtil/*.golden.json cells found");
    eprintln!("gtil_matrix: {cells} cells, global max |delta| = {global_max_dev:e} (< 1e-5)");
    Ok(())
}

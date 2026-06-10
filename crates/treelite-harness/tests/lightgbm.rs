//! The LightGBM numerical 1e-5 golden gate (LGB-01, EQV-04).
//!
//! Loads the vendored `deep_lightgbm/model.txt` via
//! [`treelite_lightgbm::load_lightgbm`], flattens the frozen golden's input
//! matrix, predicts via [`treelite_gtil::predict`], and asserts every output
//! element is within `1e-5` of the upstream `treelite.gtil.predict` golden
//! (`fixtures/lightgbm_numerical.golden.json`, captured by Plan 04-03) — while
//! tracking and printing the max observed `|delta|`.
//!
//! The `1e-5` epsilon is a HARD gate — never loosened to mask a real fidelity
//! gap. Uses `anyhow::Result` (ERR-02) so each step propagates with a context
//! chain.
//!
//! ## Why a local golden parse (not the shared `Golden`/`Manifest`)
//!
//! The LightGBM golden's manifest carries LightGBM-specific provenance keys
//! (`lightgbm`, `numpy`, `seed`, `variant`, `source`) that the XGBoost-shaped
//! `treelite_harness::Manifest` does not model, and its `input`/`output` are
//! finite `f64` (no `NaN`-normalization needed). So this test parses the golden
//! with a local serde struct, mirroring the `fixture_path` resolver and the
//! load→flatten→predict→assert pattern of `tests/golden_v5.rs`.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// The frozen LightGBM golden artifact (`{model_path, n_features, input,
/// output, output_shape, manifest, sha256}`).
#[derive(Debug, Deserialize)]
struct LgbGolden {
    /// Workspace-relative path to the vendored LightGBM text model.
    model_path: String,
    /// Number of input features (one column per feature).
    n_features: usize,
    /// Row-major input matrix (`num_row` rows, each `n_features` cells), f64.
    input: Vec<Vec<f64>>,
    /// The frozen upstream `treelite.gtil.predict` output vector (post-processed).
    output: Vec<f64>,
    /// Capture-environment provenance (free-form; only `treelite`/`os`/`arch`
    /// are read for the drift warning).
    manifest: LgbManifest,
}

/// Capture-environment provenance for the LightGBM golden (D-07).
#[derive(Debug, Deserialize)]
struct LgbManifest {
    /// Upstream Treelite version the golden was captured against (e.g. `4.7.0`).
    treelite: String,
    /// `platform.platform()` string.
    os: String,
    /// `platform.machine()` (e.g. `x86_64`).
    arch: String,
}

/// Resolve a path under `fixtures/` relative to the workspace root (mirrors
/// `golden_v5.rs:fixture_path`).
fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// Resolve a path RELATIVE TO THE WORKSPACE ROOT (the golden's `model_path` is
/// workspace-rooted, e.g. `treelite-mainline/tests/examples/...`).
fn workspace_path(rel: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
        .to_string_lossy()
        .into_owned()
}

/// Warn (never fail) when the running environment differs from the capture
/// environment (D-07): a `1e-5` failure on a different distro is most often a
/// libm/glibc divergence, and surfacing the drift makes it diagnosable.
fn check_manifest(manifest: &LgbManifest) {
    let running_os = std::env::consts::OS;
    let running_arch = std::env::consts::ARCH;
    if !manifest.os.to_lowercase().contains(running_os) {
        eprintln!(
            "WARNING: LightGBM golden captured on OS '{}' but running on '{}' — \
             a 1e-5 deviation here may be a libm/environment divergence (D-07).",
            manifest.os, running_os
        );
    }
    if manifest.arch.to_lowercase() != running_arch.to_lowercase() {
        eprintln!(
            "WARNING: LightGBM golden captured on arch '{}' but running on '{}' — \
             a 1e-5 deviation here may be an environment divergence (D-07).",
            manifest.arch, running_arch
        );
    }
    eprintln!(
        "LightGBM golden captured against upstream Treelite {}.",
        manifest.treelite
    );
}

/// LGB-01: a LightGBM text model loads → predicts → matches the upstream
/// treelite-GTIL golden within `1e-5`.
#[test]
fn lightgbm_numerical() -> anyhow::Result<()> {
    let golden_path = fixture_path("lightgbm_numerical.golden.json");
    let raw = std::fs::read_to_string(&golden_path)
        .with_context(|| format!("reading {golden_path}"))?;
    let golden: LgbGolden =
        serde_json::from_str(&raw).context("parsing lightgbm_numerical.golden.json")?;

    check_manifest(&golden.manifest);

    // Load the vendored LightGBM text model named by the golden.
    let model_path = workspace_path(&golden.model_path);
    let model_text =
        std::fs::read_to_string(&model_path).with_context(|| format!("reading {model_path}"))?;
    let model = treelite_lightgbm::load_lightgbm(&model_text)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading LightGBM model")?;

    // Flatten the golden input into a row-major f32 buffer (gtil::predict input).
    let num_row = golden.input.len();
    anyhow::ensure!(num_row > 0, "golden input has zero rows");
    let num_feature = golden.n_features;
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * num_feature);
    for (r, row) in golden.input.iter().enumerate() {
        anyhow::ensure!(
            row.len() == num_feature,
            "golden input row {r} has {} cells, expected {num_feature}",
            row.len()
        );
        flat.extend(row.iter().map(|&c| c as f32));
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
        let expected = golden.output[i] as f32;
        let delta = (got - expected).abs() as f64;
        if delta > max_dev {
            max_dev = delta;
        }
        // HARD 1e-5 gate — never loosen to mask a real fidelity gap.
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }

    eprintln!("lightgbm_numerical: max |delta| = {max_dev:e} (< 1e-5)");
    Ok(())
}

/// LGB-02: a LightGBM CATEGORICAL model (bitset splits decoded via
/// `BitsetToList`) loads → predicts → matches the upstream treelite-GTIL golden
/// within `1e-5`. Mirrors [`lightgbm_numerical`] but its `model_path` points at
/// the workspace-rooted `fixtures/lightgbm_categorical.txt` (a real LightGBM
/// text model fit with `max_cat_to_onehot=1` to force bitset categorical
/// splits, captured by Plan 04-03).
#[test]
fn lightgbm_categorical() -> anyhow::Result<()> {
    let golden_path = fixture_path("lightgbm_categorical.golden.json");
    let raw = std::fs::read_to_string(&golden_path)
        .with_context(|| format!("reading {golden_path}"))?;
    let golden: LgbGolden =
        serde_json::from_str(&raw).context("parsing lightgbm_categorical.golden.json")?;

    check_manifest(&golden.manifest);

    // The categorical golden's model_path is workspace-rooted
    // (`fixtures/lightgbm_categorical.txt`).
    let model_path = workspace_path(&golden.model_path);
    let model_text =
        std::fs::read_to_string(&model_path).with_context(|| format!("reading {model_path}"))?;
    let model = treelite_lightgbm::load_lightgbm(&model_text)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading LightGBM categorical model")?;

    let num_row = golden.input.len();
    anyhow::ensure!(num_row > 0, "golden input has zero rows");
    let num_feature = golden.n_features;
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * num_feature);
    for (r, row) in golden.input.iter().enumerate() {
        anyhow::ensure!(
            row.len() == num_feature,
            "golden input row {r} has {} cells, expected {num_feature}",
            row.len()
        );
        flat.extend(row.iter().map(|&c| c as f32));
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
        let expected = golden.output[i] as f32;
        let delta = (got - expected).abs() as f64;
        if delta > max_dev {
            max_dev = delta;
        }
        // HARD 1e-5 gate — never loosen to mask a real fidelity gap.
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }

    eprintln!("lightgbm_categorical: max |delta| = {max_dev:e} (< 1e-5)");
    Ok(())
}

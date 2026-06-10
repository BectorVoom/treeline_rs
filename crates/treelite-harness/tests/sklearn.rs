//! The scikit-learn 1e-5 golden gates (SKL-01 RF/ET, SKL-02 GB).
//!
//! Loads the frozen sklearn array-dump goldens (`fixtures/sklearn_rf.golden.json`
//! and `fixtures/sklearn_gb.golden.json`, captured by Plan 04-03 from upstream
//! `treelite.gtil.predict`), rebuilds each estimator family through the matching
//! `treelite_sklearn` loader, predicts via [`treelite_gtil::predict`], and
//! asserts every output element is within `1e-5` of the upstream golden — while
//! tracking and printing the max observed `|delta|`.
//!
//! The `1e-5` epsilon is a HARD gate — never loosened to mask a real fidelity
//! gap. Uses `anyhow::Result` (ERR-02). Parses the goldens with local serde
//! structs (mirroring `tests/lightgbm.rs`), since the sklearn goldens carry
//! their own per-family array-dump shape that the XGBoost-shaped
//! `treelite_harness::Golden` does not model.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// One sklearn decision tree's array dump (the `sklearn.tree_` column set).
#[derive(Debug, Deserialize)]
struct SklTree {
    node_count: i64,
    children_left: Vec<i64>,
    children_right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    /// Flat leaf payload: `node_count` scalars (scalar leaves) or
    /// `node_count * n_targets * max_num_class` (vector leaves), row-major.
    value: Vec<f64>,
    n_node_samples: Vec<i64>,
    weighted_n_node_samples: Vec<f64>,
    impurity: Vec<f64>,
}

/// One RF/ET estimator family in `sklearn_rf.golden.json`.
#[derive(Debug, Deserialize)]
struct RfFamily {
    n_estimators: i32,
    n_features_in: i32,
    n_outputs: i32,
    /// Number of classes (classifier families only; `null` for regressors).
    n_classes: Option<i32>,
    trees: Vec<SklTree>,
    output: Vec<f64>,
}

/// One GB estimator family in `sklearn_gb.golden.json`.
#[derive(Debug, Deserialize)]
struct GbFamily {
    n_estimators: i32,
    n_features_in: i32,
    /// Number of classes (`1` for the regressor; `>= 2` for the classifier).
    n_classes: i32,
    /// GB init-model base scores (one per class), derived capture-side exactly
    /// as upstream `importer.py` (regressor `init_.constant_`, classifier
    /// `_raw_predict_init`). GTIL adds these to the tree sum before the
    /// postprocessor, so they are a required loader input.
    base_scores: Vec<f64>,
    trees: Vec<SklTree>,
    output: Vec<f64>,
}

/// `sklearn_iforest.golden.json` top level (SKL-03).
///
/// IsolationForest is a single estimator (not a family map): the golden carries
/// one `trees` array plus the capture-side `ratio_c` (== `expected_depth(
/// max_samples_)`, isolation_forest.py) and the upstream `treelite.gtil.predict`
/// `output`. CRITICAL (D-07): that `output` equals `-clf.score_samples`, NOT the
/// framework's own `clf.score_samples`/anomaly score — the `cross_check` field
/// records the capture's own verification that the two agree.
#[derive(Debug, Deserialize)]
struct IForestGolden {
    input: Vec<Vec<f64>>,
    n_features_in: i32,
    n_estimators: i32,
    /// `expected_depth(max_samples_)` (isolation_forest.py); threaded into the
    /// `exponential_standard_ratio` postprocessor. Consumed AS-IS by the loader
    /// (the loader does NOT recompute it).
    ratio_c: f64,
    trees: Vec<SklTree>,
    output: Vec<f64>,
    manifest: SklManifest,
}

/// The HistGradientBoosting packed-node array dump (`histgb` object) shared by
/// both the numerical and categorical goldens (SKL-04). Mirrors the upstream
/// `importer.py` HistGB capture (`:355-478`): the per-tree packed
/// `HistGradientBoostingNode` buffers are base64-encoded (`nodes_b64`), and the
/// `features_map` / `categories_map` carry the feature/category remaps that the
/// loader must apply.
#[derive(Debug, Deserialize)]
struct HistGbDump {
    n_iter: i32,
    n_features_in: i32,
    /// 52 (i32 feature index) or 56 (i64 feature index) — selects the packed
    /// layout. On this environment the fixtures are 56.
    expected_sizeof_node_struct: usize,
    node_count: Vec<i64>,
    /// Per-tree packed node buffers, base64-encoded (exact C-struct bytes).
    nodes_b64: Vec<String>,
    /// Per-tree categorical bitsets (8 `u32` per categorical row; empty for a
    /// purely-numerical tree).
    raw_left_cat_bitsets: Vec<Vec<u32>>,
    /// Feature index remap — ALWAYS applied (`split_index = features_map[
    /// feature_idx]`, Pitfall 4).
    features_map: Vec<i32>,
    /// Per-categorical-feature category remap; empty when the model has no
    /// categorical splits (identity).
    categories_map: Vec<Vec<i64>>,
    /// HistGB baseline prediction (the GTIL base score), one entry per class.
    baseline_prediction: Vec<f64>,
}

/// `sklearn_histgb_numerical.golden.json` / `sklearn_histgb_categorical.golden.json`
/// top level (SKL-04).
#[derive(Debug, Deserialize)]
struct HistGbGolden {
    input: Vec<Vec<f64>>,
    histgb: HistGbDump,
    output: Vec<f64>,
    manifest: SklManifest,
}

/// Decode a standard (RFC 4648) base64 string into bytes.
///
/// A tiny self-contained decoder (no third-party `base64` crate, to keep the
/// dependency graph minimal): the HistGB packed-node buffers are frozen as
/// standard base64 by the capture script, so only the standard alphabet +
/// `=` padding is needed.
fn base64_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    // Reverse lookup: byte -> 6-bit value (255 = invalid).
    let mut rev = [255u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        rev[c as usize] = i as u8;
    }

    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    for &c in bytes {
        if c == b'=' || c == b'\n' || c == b'\r' {
            continue;
        }
        let v = rev[c as usize];
        anyhow::ensure!(v != 255, "invalid base64 character {:?}", c as char);
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Ok(out)
}

/// Base64-decode every per-tree packed node buffer into owned byte vectors.
fn decode_nodes(nodes_b64: &[String]) -> anyhow::Result<Vec<Vec<u8>>> {
    nodes_b64
        .iter()
        .enumerate()
        .map(|(t, s)| base64_decode(s).with_context(|| format!("base64-decoding nodes_b64[{t}]")))
        .collect()
}

/// Run a HistGB golden (numerical or categorical) end-to-end: decode the packed
/// nodes, load the regressor via `treelite_sklearn`, predict, and assert within
/// 1e-5. Returns the max observed `|delta|`.
fn run_histgb_golden(path: &str, label: &str) -> anyhow::Result<f64> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let golden: HistGbGolden =
        serde_json::from_str(&raw).with_context(|| format!("parsing {path}"))?;
    check_manifest(&golden.manifest);

    let h = &golden.histgb;
    let n_features = h.n_features_in as usize;
    let (flat, num_row) = flatten_input(&golden.input, n_features)?;

    // Decode packed nodes + borrow as slice-of-slices for the D-01 signature.
    let nodes_owned = decode_nodes(&h.nodes_b64)?;
    let nodes: Vec<&[u8]> = nodes_owned.iter().map(|v| v.as_slice()).collect();
    let bitsets: Vec<&[u32]> = h
        .raw_left_cat_bitsets
        .iter()
        .map(|v| v.as_slice())
        .collect();

    // Identity categories_map when the model has no categorical splits.
    let categories_map: Option<&[Vec<i64>]> = if h.categories_map.is_empty() {
        None
    } else {
        Some(&h.categories_map)
    };

    // The frozen HistGB fixtures (numerical + categorical) are both regressors
    // (baseline_prediction has one entry; output last-dim is 1).
    let base = h.baseline_prediction.first().copied().unwrap_or(0.0);
    let model = treelite_sklearn::load_hist_gradient_boosting_regressor(
        h.n_iter,
        h.n_features_in,
        h.expected_sizeof_node_struct,
        &h.node_count,
        &nodes,
        &bitsets,
        &h.features_map,
        categories_map,
        base,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
    .with_context(|| format!("loading {label}"))?;

    let rust = treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default())
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("predicting {label}"))?;
    let dev = assert_within_1e5(&rust, &golden.output, label)?;
    eprintln!("{label}: max |delta| = {dev:e} (< 1e-5)");
    Ok(dev)
}

/// `sklearn_rf.golden.json` top level.
#[derive(Debug, Deserialize)]
struct RfGolden {
    input: Vec<Vec<f64>>,
    families: std::collections::BTreeMap<String, RfFamily>,
    manifest: SklManifest,
}

/// `sklearn_gb.golden.json` top level.
#[derive(Debug, Deserialize)]
struct GbGolden {
    input: Vec<Vec<f64>>,
    families: std::collections::BTreeMap<String, GbFamily>,
    manifest: SklManifest,
}

/// Capture-environment provenance (D-07).
#[derive(Debug, Deserialize)]
struct SklManifest {
    treelite: String,
    os: String,
    arch: String,
}

/// Resolve a path under `fixtures/` relative to the workspace root.
fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// Warn (never fail) on capture/runtime environment drift (D-07).
fn check_manifest(manifest: &SklManifest) {
    let running_os = std::env::consts::OS;
    let running_arch = std::env::consts::ARCH;
    if !manifest.os.to_lowercase().contains(running_os) {
        eprintln!(
            "WARNING: sklearn golden captured on OS '{}' but running on '{}' — \
             a 1e-5 deviation here may be a libm/environment divergence (D-07).",
            manifest.os, running_os
        );
    }
    if manifest.arch.to_lowercase() != running_arch.to_lowercase() {
        eprintln!(
            "WARNING: sklearn golden captured on arch '{}' but running on '{}' — \
             a 1e-5 deviation here may be an environment divergence (D-07).",
            manifest.arch, running_arch
        );
    }
    eprintln!(
        "sklearn golden captured against upstream Treelite {}.",
        manifest.treelite
    );
}

/// Flatten the golden input matrix into a row-major f32 buffer.
fn flatten_input(input: &[Vec<f64>], n_features: usize) -> anyhow::Result<(Vec<f32>, usize)> {
    let num_row = input.len();
    anyhow::ensure!(num_row > 0, "golden input has zero rows");
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * n_features);
    for (r, row) in input.iter().enumerate() {
        anyhow::ensure!(
            row.len() == n_features,
            "golden input row {r} has {} cells, expected {n_features}",
            row.len()
        );
        flat.extend(row.iter().map(|&c| c as f32));
    }
    Ok((flat, num_row))
}

/// Borrow each tree column as a slice-of-slices for the D-01 array signatures.
struct TreeColumns<'a> {
    node_count: Vec<i64>,
    children_left: Vec<&'a [i64]>,
    children_right: Vec<&'a [i64]>,
    feature: Vec<&'a [i64]>,
    threshold: Vec<&'a [f64]>,
    value: Vec<&'a [f64]>,
    n_node_samples: Vec<&'a [i64]>,
    weighted_n_node_samples: Vec<&'a [f64]>,
    impurity: Vec<&'a [f64]>,
}

fn columns(trees: &[SklTree]) -> TreeColumns<'_> {
    TreeColumns {
        node_count: trees.iter().map(|t| t.node_count).collect(),
        children_left: trees.iter().map(|t| t.children_left.as_slice()).collect(),
        children_right: trees.iter().map(|t| t.children_right.as_slice()).collect(),
        feature: trees.iter().map(|t| t.feature.as_slice()).collect(),
        threshold: trees.iter().map(|t| t.threshold.as_slice()).collect(),
        value: trees.iter().map(|t| t.value.as_slice()).collect(),
        n_node_samples: trees.iter().map(|t| t.n_node_samples.as_slice()).collect(),
        weighted_n_node_samples: trees
            .iter()
            .map(|t| t.weighted_n_node_samples.as_slice())
            .collect(),
        impurity: trees.iter().map(|t| t.impurity.as_slice()).collect(),
    }
}

/// Assert every Rust output element is within `1e-5` of the golden, return max
/// `|delta|`.
fn assert_within_1e5(rust: &[f32], golden: &[f64], label: &str) -> anyhow::Result<f64> {
    anyhow::ensure!(
        rust.len() == golden.len(),
        "{label}: prediction length {} != golden output length {}",
        rust.len(),
        golden.len()
    );
    let mut max_dev: f64 = 0.0;
    for (i, &got) in rust.iter().enumerate() {
        let expected = golden[i] as f32;
        let delta = (got - expected).abs() as f64;
        if delta > max_dev {
            max_dev = delta;
        }
        // HARD 1e-5 gate — never loosen to mask a real fidelity gap.
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }
    Ok(max_dev)
}

/// SKL-01: every RandomForest / ExtraTrees family (clf + reg) loads via the
/// bulk path → predicts → matches the upstream treelite-GTIL golden within 1e-5.
#[test]
fn sklearn_rf() -> anyhow::Result<()> {
    let golden_path = fixture_path("sklearn_rf.golden.json");
    let raw =
        std::fs::read_to_string(&golden_path).with_context(|| format!("reading {golden_path}"))?;
    let golden: RfGolden = serde_json::from_str(&raw).context("parsing sklearn_rf.golden.json")?;
    check_manifest(&golden.manifest);

    let mut worst: f64 = 0.0;
    for (name, fam) in &golden.families {
        let n_features = fam.n_features_in as usize;
        let (flat, num_row) = flatten_input(&golden.input, n_features)?;
        let c = columns(&fam.trees);

        // n_targets == n_outputs for sklearn (one target per output).
        let n_targets = fam.n_outputs;

        let model = if let Some(n_classes) = fam.n_classes {
            // Classifier — one n_classes entry per target.
            let n_classes_vec = vec![n_classes; n_targets as usize];
            treelite_sklearn::load_random_forest_classifier(
                fam.n_estimators,
                fam.n_features_in,
                n_targets,
                &n_classes_vec,
                &c.node_count,
                &c.children_left,
                &c.children_right,
                &c.feature,
                &c.threshold,
                &c.value,
                &c.n_node_samples,
                &c.weighted_n_node_samples,
                &c.impurity,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {name}"))?
        } else {
            treelite_sklearn::load_random_forest_regressor(
                fam.n_estimators,
                fam.n_features_in,
                n_targets,
                &c.node_count,
                &c.children_left,
                &c.children_right,
                &c.feature,
                &c.threshold,
                &c.value,
                &c.n_node_samples,
                &c.weighted_n_node_samples,
                &c.impurity,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {name}"))?
        };

        let rust =
            treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default())
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("predicting {name}"))?;
        let dev = assert_within_1e5(&rust, &fam.output, name)?;
        eprintln!("sklearn_rf [{name}]: max |delta| = {dev:e} (< 1e-5)");
        worst = worst.max(dev);
    }
    eprintln!("sklearn_rf: worst family max |delta| = {worst:e} (< 1e-5)");
    Ok(())
}

/// SKL-02: every GradientBoosting family (clf + reg) loads via the MixIn path →
/// predicts → matches the upstream treelite-GTIL golden within 1e-5 (no leaf
/// re-shrink).
#[test]
fn sklearn_gb() -> anyhow::Result<()> {
    let golden_path = fixture_path("sklearn_gb.golden.json");
    let raw =
        std::fs::read_to_string(&golden_path).with_context(|| format!("reading {golden_path}"))?;
    let golden: GbGolden = serde_json::from_str(&raw).context("parsing sklearn_gb.golden.json")?;
    check_manifest(&golden.manifest);

    let mut worst: f64 = 0.0;
    for (name, fam) in &golden.families {
        let n_features = fam.n_features_in as usize;
        let (flat, num_row) = flatten_input(&golden.input, n_features)?;
        let c = columns(&fam.trees);

        let model = if fam.n_classes >= 2 {
            treelite_sklearn::load_gradient_boosting_classifier(
                fam.n_estimators,
                fam.n_features_in,
                fam.n_classes,
                &c.node_count,
                &c.children_left,
                &c.children_right,
                &c.feature,
                &c.threshold,
                &c.value,
                &c.n_node_samples,
                &c.weighted_n_node_samples,
                &c.impurity,
                &fam.base_scores,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {name}"))?
        } else {
            let base_score = fam.base_scores.first().copied().unwrap_or(0.0);
            treelite_sklearn::load_gradient_boosting_regressor(
                fam.n_estimators,
                fam.n_features_in,
                &c.node_count,
                &c.children_left,
                &c.children_right,
                &c.feature,
                &c.threshold,
                &c.value,
                &c.n_node_samples,
                &c.weighted_n_node_samples,
                &c.impurity,
                base_score,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {name}"))?
        };

        let rust =
            treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default())
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("predicting {name}"))?;
        let dev = assert_within_1e5(&rust, &fam.output, name)?;
        eprintln!("sklearn_gb [{name}]: max |delta| = {dev:e} (< 1e-5)");
        worst = worst.max(dev);
    }
    eprintln!("sklearn_gb: worst family max |delta| = {worst:e} (< 1e-5)");
    Ok(())
}

/// SKL-03: IsolationForest loads via the MixIn path (with the capture-side
/// `ratio_c` + `exponential_standard_ratio` postprocessor) → predicts → matches
/// the upstream treelite-GTIL golden within 1e-5.
///
/// The golden `output` is `treelite.gtil.predict` (== `-clf.score_samples`,
/// D-07) — the canonical "Treelite ≠ framework" case. We assert against THAT,
/// never the framework's own anomaly score.
#[test]
fn sklearn_iforest() -> anyhow::Result<()> {
    let golden_path = fixture_path("sklearn_iforest.golden.json");
    let raw =
        std::fs::read_to_string(&golden_path).with_context(|| format!("reading {golden_path}"))?;
    let golden: IForestGolden =
        serde_json::from_str(&raw).context("parsing sklearn_iforest.golden.json")?;
    check_manifest(&golden.manifest);

    let n_features = golden.n_features_in as usize;
    let (flat, num_row) = flatten_input(&golden.input, n_features)?;
    let c = columns(&golden.trees);

    let model = treelite_sklearn::load_isolation_forest(
        golden.n_estimators,
        golden.n_features_in,
        &c.node_count,
        &c.children_left,
        &c.children_right,
        &c.feature,
        &c.threshold,
        &c.value,
        &c.n_node_samples,
        &c.weighted_n_node_samples,
        &c.impurity,
        golden.ratio_c,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
    .context("loading isolation_forest")?;

    let rust = treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default())
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("predicting isolation_forest")?;
    let dev = assert_within_1e5(&rust, &golden.output, "isolation_forest")?;
    eprintln!(
        "sklearn_iforest: max |delta| = {dev:e} (< 1e-5) vs treelite.gtil.predict (== -score_samples)"
    );
    Ok(())
}

/// SKL-04 (numerical): HistGradientBoosting with an identity feature map and NO
/// categorical splits loads from its packed-node byte buffer (56-byte layout on
/// this env) → predicts → matches the upstream treelite-GTIL golden within 1e-5.
///
/// This isolates packed-node DECODE correctness (RESEARCH Open Q3): the
/// `features_map` is the identity arange, so any deviation here is a struct
/// offset / `from_le_bytes` field-decode bug, not a remap bug.
#[test]
fn sklearn_histgb_numerical() -> anyhow::Result<()> {
    let path = fixture_path("sklearn_histgb_numerical.golden.json");
    let dev = run_histgb_golden(&path, "sklearn_histgb_numerical")?;
    eprintln!("sklearn_histgb_numerical: max |delta| = {dev:e} (< 1e-5)");
    Ok(())
}

/// SKL-04 (categorical): HistGradientBoosting WITH categorical splits — the
/// `features_map` is a real permutation and a `categories_map` is present, so
/// this isolates the remap risk (RESEARCH Pitfall 4). The categorical node's
/// `left_cat_bitmap` is decoded via the `8*row` 256-bit stride; each set bit is
/// remapped through `categories_map[fid][cat]`. Loads → predicts → matches the
/// upstream treelite-GTIL golden within 1e-5 (not feature-transposed).
#[test]
fn sklearn_histgb_categorical() -> anyhow::Result<()> {
    let path = fixture_path("sklearn_histgb_categorical.golden.json");
    let dev = run_histgb_golden(&path, "sklearn_histgb_categorical")?;
    eprintln!("sklearn_histgb_categorical: max |delta| = {dev:e} (< 1e-5)");
    Ok(())
}

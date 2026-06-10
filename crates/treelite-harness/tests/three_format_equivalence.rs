//! RED 3-format equivalence + byte-fidelity test scaffold (Phase 3, Plan 03-01).
//!
//! ## This file is EXPECTED to fail (the RED gate is the point)
//!
//! It references the as-yet-nonexistent loader entry points
//! `treelite_xgboost::load_xgboost_ubjson` and `treelite_xgboost::load_xgboost_legacy`,
//! so the test target FAILS TO COMPILE today (only `load_xgboost_json` exists in
//! Phase 1/2). The missing entry points land in:
//!   - JSON D-10 close-out (sum_hess/gain/attributes fidelity) — Plan 03-02
//!   - UBJSON loader (`load_xgboost_ubjson`)                    — Plan 03-03
//!   - legacy-binary loader (`load_xgboost_legacy`) + cross-format close — Plan 03-04
//!
//! Do NOT stub the missing functions to make this green — the RED state proves the
//! test exercises the real downstream contracts. When 03-02..03-04 land, both tests
//! go green WITHOUT any change here.
//!
//! ## What the two tests assert (the phase acceptance bar, made executable)
//!
//! - `three_format_predicts_within_1e5`: the ONE shared logical model
//!   (`fixtures/xgb_3format.*`), loaded from JSON, UBJSON, AND legacy binary, each
//!   predicts within `1e-5` of the single shared prediction golden
//!   (`fixtures/xgb_3format.golden.json`) — D-04/D-05 verify-narrow.
//! - `three_format_serialize_byte_fidelity`: all three loaded `Model`s serialize to
//!   v5 bytes byte-identical to the SINGLE upstream golden blob
//!   (`fixtures/golden_v5_3format.bin`) — the D-10 / DEF-02-01 cross-format close.
//!
//! Uses `anyhow::Result` (ERR-02) so each step propagates with a context chain.

use std::path::Path;

use anyhow::Context;
use treelite_core::Model;
use treelite_harness::{load_golden, Golden};

/// Resolve a path under `fixtures/` relative to the workspace root
/// (copied from `golden_v5.rs:37-43`).
fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// First offset at which `got` and `want` differ (or a length-difference point).
/// Copied from `golden_v5.rs:46-57` for a precise byte-divergence report.
fn first_diff(got: &[u8], want: &[u8]) -> Option<usize> {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return Some(i);
        }
    }
    if got.len() != want.len() {
        return Some(n);
    }
    None
}

/// Read a binary fixture (e.g. the legacy `.model` / `.ubj`) as raw bytes.
fn read_fixture_bytes(name: &str) -> anyhow::Result<Vec<u8>> {
    let p = fixture_path(name);
    std::fs::read(&p).with_context(|| format!("reading {p}"))
}

/// Read a text fixture (the XGBoost-JSON model) as a string.
fn read_fixture_str(name: &str) -> anyhow::Result<String> {
    let p = fixture_path(name);
    std::fs::read_to_string(&p).with_context(|| format!("reading {p}"))
}

/// Flatten `golden.input` into a row-major `f32` buffer + return `num_row`.
/// (mirrors `run_equivalence` in the harness lib, but kept local so all three
/// Models share ONE predict path here.)
fn flatten_input(golden: &Golden) -> anyhow::Result<(Vec<f32>, usize)> {
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
    Ok((flat, num_row))
}

/// One predict path shared by all three loaded Models: GTIL predict + `1e-5`
/// assertion against the golden, tracking the max `|delta|`.
fn assert_predicts_within_1e5(model: &Model, golden: &Golden, label: &str) -> anyhow::Result<f64> {
    let (flat, num_row) = flatten_input(golden)?;
    let rust = treelite_gtil::predict(model, &flat, num_row, &treelite_gtil::Config::default())
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("predicting ({label})"))?;
    anyhow::ensure!(
        rust.len() == golden.output.len(),
        "{label}: prediction length {} != golden output length {}",
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
        // Hard 1e-5 gate — NEVER loosened to mask a real fidelity gap.
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }
    Ok(max_dev)
}

/// D-04/D-05 verify-narrow: the ONE logical model loaded from all three formats
/// predicts within `1e-5` of the single shared golden.
///
/// RED until `load_xgboost_ubjson` (03-03) and `load_xgboost_legacy` (03-04) exist.
#[test]
fn three_format_predicts_within_1e5() -> anyhow::Result<()> {
    let golden = load_golden(&fixture_path("xgb_3format.golden.json"))
        .context("loading xgb_3format.golden.json")?;

    let json_src = read_fixture_str("xgb_3format.json")?;
    let ubj_bytes = read_fixture_bytes("xgb_3format.ubj")?;
    let legacy_bytes = read_fixture_bytes("xgb_3format.model")?;

    let m_json = treelite_xgboost::load_xgboost_json(&json_src)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_json")?;
    // RED: these two entry points do NOT exist yet (03-03 / 03-04).
    let m_ubj = treelite_xgboost::load_xgboost_ubjson(&ubj_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_ubjson")?;
    let m_legacy = treelite_xgboost::load_xgboost_legacy(&legacy_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_legacy")?;

    let d_json = assert_predicts_within_1e5(&m_json, &golden, "json")?;
    let d_ubj = assert_predicts_within_1e5(&m_ubj, &golden, "ubjson")?;
    let d_legacy = assert_predicts_within_1e5(&m_legacy, &golden, "legacy")?;

    println!(
        "three-format 1e-5 max|delta|: json={d_json:e} ubjson={d_ubj:e} legacy={d_legacy:e}"
    );
    Ok(())
}

/// D-10 / DEF-02-01 single-golden cross-format close: all three loaded Models
/// serialize to v5 bytes byte-identical to the ONE upstream golden blob.
///
/// RED until the JSON D-10 fidelity close-out (03-02) AND the UBJSON/legacy entry
/// points (03-03/03-04) exist.
#[test]
fn three_format_serialize_byte_fidelity() -> anyhow::Result<()> {
    let golden_blob = read_fixture_bytes("golden_v5_3format.bin")?;

    let json_src = read_fixture_str("xgb_3format.json")?;
    let ubj_bytes = read_fixture_bytes("xgb_3format.ubj")?;
    let legacy_bytes = read_fixture_bytes("xgb_3format.model")?;

    let mut m_json = treelite_xgboost::load_xgboost_json(&json_src)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_json")?;
    // RED: these two entry points do NOT exist yet (03-03 / 03-04).
    let mut m_ubj = treelite_xgboost::load_xgboost_ubjson(&ubj_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_ubjson")?;
    let mut m_legacy = treelite_xgboost::load_xgboost_legacy(&legacy_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("load_xgboost_legacy")?;

    for (label, model) in [
        ("json", &mut m_json),
        ("ubjson", &mut m_ubj),
        ("legacy", &mut m_legacy),
    ] {
        let produced = treelite_core::serialize_to_buffer(model);
        if let Some(off) = first_diff(&produced, &golden_blob) {
            panic!(
                "{label}: v5 serialization diverges from golden_v5_3format.bin at \
                 offset {off}: produced={:?} golden={:?} (produced len {}, golden len {})",
                produced.get(off),
                golden_blob.get(off),
                produced.len(),
                golden_blob.len()
            );
        }
        assert_eq!(
            produced, golden_blob,
            "{label}: serialize must equal golden_v5_3format.bin byte-for-byte (D-10)"
        );
    }
    Ok(())
}

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
use serde::de::{Deserializer, Visitor};
use serde::Deserialize;

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

/// Capture-environment provenance (D-07).
///
/// Keys match exactly what `capture_golden.py` writes. `libc` is a
/// `serde_json::Value` because `platform.libc_ver()` is a Python tuple
/// (serialized as a JSON array, e.g. `["glibc", "2.39"]`).
#[derive(Debug, Deserialize)]
pub struct Manifest {
    /// Upstream Treelite version the golden was captured against (e.g. `4.7.0`).
    pub treelite: String,
    /// XGBoost version used to author the fixture (optional).
    #[serde(default)]
    pub xgboost: Option<String>,
    /// `platform.platform()` string (e.g. `Linux-...-x86_64-with-glibc2.39`).
    pub os: String,
    /// `platform.machine()` (e.g. `x86_64`).
    pub arch: String,
    /// `platform.libc_ver()` tuple (e.g. `["glibc", "2.39"]`).
    pub libc: serde_json::Value,
    /// Python version of the capture environment (optional).
    #[serde(default)]
    pub python: Option<String>,
}

/// Load + parse the frozen golden artifact.
///
/// Reads `path`, normalizes Python's bare `NaN` tokens to JSON `null` (so
/// `serde_json` accepts the missing-value row), and deserializes into [`Golden`].
/// A read or parse failure surfaces an `anyhow` context chain rather than an
/// opaque panic (ERR-02, T-04-01).
pub fn load_golden(path: &str) -> anyhow::Result<Golden> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading golden.json at {path}"))?;
    let normalized = normalize_nan_tokens(&raw);
    let golden: Golden =
        serde_json::from_str(&normalized).context("parsing golden.json")?;
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

/// Warn (never fail) when the running environment differs from the capture
/// environment recorded in the golden's manifest (D-07, T-04-02).
///
/// A `1e-5` failure on a different distro is most often a libm/glibc divergence
/// (RESEARCH Pitfall 4). Surfacing the drift here makes such a failure
/// immediately diagnosable, without the manifest itself ever passing/failing the
/// equivalence gate.
pub fn check_manifest(manifest: &Manifest) {
    let running_os = std::env::consts::OS;
    let running_arch = std::env::consts::ARCH;

    // `manifest.os` is `platform.platform()` (a verbose descriptor), so a
    // substring check against the coarse `std::env::consts::OS` (e.g. "linux")
    // is the right granularity here.
    if !manifest.os.to_lowercase().contains(running_os) {
        eprintln!(
            "WARNING: golden captured on OS '{}' but running on '{}' — \
             a 1e-5 deviation here may be a libm/environment divergence (D-07).",
            manifest.os, running_os
        );
    }
    if manifest.arch.to_lowercase() != running_arch.to_lowercase() {
        eprintln!(
            "WARNING: golden captured on arch '{}' but running on '{}' — \
             a 1e-5 deviation here may be an environment divergence (D-07).",
            manifest.arch, running_arch
        );
    }
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

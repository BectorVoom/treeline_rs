//! GPU equivalence **report** emission (GPU-04, D-06/D-08) — OBSERVATIONAL.
//!
//! This module turns the per-cell GPU runs of `tests/gtil_matrix_gpu.rs` into
//! the committed Phase-7 artifacts `docs/GPU_EQUIVALENCE_REPORT.md` and its
//! machine-readable sidecar `docs/gpu_equivalence.json`. It is the report half
//! of the GPU vertical: the registration (`Backend::Rocm` + `rocm_case`) landed
//! in Plan 03; here the SAME frozen `fixtures/gtil/*.golden.json` matrix is run
//! on ROCm hardware and its measured deviations are RECORDED — never asserted.
//!
//! ## Observational, NOT a CI gate (D-01)
//!
//! The CPU `1e-5` hard gate stays on the scalar / cubecl-cpu siblings
//! (`gtil_matrix.rs` / `gtil_matrix_cubecl.rs`), whose comparison helper panics
//! on a real fidelity gap. The GPU path is fundamentally different: per the
//! OpenCL spec, transcendental rounding and float-reduction order on a GPU are
//! implementation-defined, so a `|delta|` slightly above `1e-5` is a *finding to
//! record*, not a failure to gate. Accordingly [`max_abs_delta_report_mode`] is
//! that comparison helper's max-deviation accumulation loop (the cubecl matrix
//! sibling, lines 296-324) with the hard approx equality-gate line REMOVED — it
//! RECORDS the max finite `|delta|` and NEVER panics. (This module deliberately
//! contains no hard-gate equality-assertion token, D-01.)
//!
//! ## Provenance reuse, not a re-derived copy (WR-04)
//!
//! Whether a cell's f64 twin (the f64-input postprocessor / accumulator) was the
//! executed path, and whether a model routes to the scalar fallback, are read
//! from [`treelite_cubecl::model_routes_to_scalar_fallback`] — the SAME predicate
//! `predict_cpu` / `predict::<R, _>` itself consults — never a parallel
//! re-implementation. The report can therefore never drift from the executed
//! path ("green while buggy", D-06).
//!
//! ## Honest determinism note
//!
//! The header states determinism is *observed run-to-run stable on the tested
//! device, bit-identity NOT guaranteed on GPU* (per the OpenCL spec). The
//! Phase-6 SC2 bit-identical claim is a CPU-backend property and stays there.

use std::fmt::Write as _;
use std::path::Path;

use treelite_core::Model;

use crate::manifest::Manifest;

/// Report-mode max finite absolute deviation between a measured `got` vector and
/// the frozen `want` golden — **records, never panics** (D-01).
///
/// This is the cubecl matrix sibling's comparison-helper accumulation loop
/// (lines 296-324) with the hard approx equality-gate (line 321) REMOVED:
/// NaN/non-finite cells are skipped structurally and the max finite `|delta|` is
/// returned. Length mismatch returns [`f64::NAN`] (a recordable "could not
/// compare" sentinel) rather than panicking, because the GPU report must never
/// abort a run on a single anomalous cell — it surfaces the anomaly as data.
pub fn max_abs_delta_report_mode(got: &[f64], want: &[f64]) -> f64 {
    if got.len() != want.len() {
        // Recordable "could not compare" — never a panic on the observational path.
        return f64::NAN;
    }
    let mut max_dev = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        // Skip NaN / non-finite cells structurally (the cubecl sibling's NaN/inf
        // arms, but as a skip — never a structural assert on the GPU path).
        if g.is_nan() || w.is_nan() || !g.is_finite() || !w.is_finite() {
            continue;
        }
        let delta = (g - w).abs();
        if delta > max_dev {
            max_dev = delta;
        }
        // NO hard-gate equality assertion here — the GPU report RECORDS, never
        // fails (D-01). The 1e-5 hard gate lives on the CPU spine siblings.
    }
    max_dev
}

/// Does `model` route the WHOLE model to the scalar fallback (categorical split
/// OR any internal node with a non-`kLT` operator)?
///
/// WR-04: delegates to [`treelite_cubecl::model_routes_to_scalar_fallback`] — the
/// SAME predicate `predict_cpu` / `predict::<R, _>` consults — so the report's
/// "f64 fallback used?" provenance can never drift from the executed routing.
pub fn model_routes_to_scalar_fallback(model: &Model) -> bool {
    treelite_cubecl::model_routes_to_scalar_fallback(model)
}

/// The measured outcome for one model-class row of the GPU equivalence report
/// (D-06/D-08). One row per frozen-golden model class (D-07).
#[derive(Debug, Clone)]
pub struct ReportRow {
    /// Frozen-golden model class (e.g. `binary`, `leaf_vec_mc`, `lgbm_numerical`).
    pub model_class: String,
    /// The model's postprocessor (read from `Model::postprocessor`).
    pub postprocessor: String,
    /// The measured ROCm max finite `|delta|` against the f64 scalar reference,
    /// or `None` when ROCm did not run on this host (device absent — D-05).
    pub rocm_max_abs_delta: Option<f64>,
    /// Whether the f64 twin / scalar-fallback path produced the asserted vector
    /// for this class (D-02 — recorded from the executed routing, WR-04).
    pub f64_fallback_used: bool,
    /// The measured CUDA max `|delta|`, or `None` for "not run — no device" (D-05).
    pub cuda_max_abs_delta: Option<f64>,
    /// The measured wgpu max `|delta|`, or `None` for "not run — no device" (D-05).
    pub wgpu_max_abs_delta: Option<f64>,
    /// The D-03 predicted-deviation band lower bound (f32 input).
    pub predicted_low: f64,
    /// The D-03 predicted-deviation band upper bound (f32 input).
    pub predicted_high: f64,
}

/// The D-03 predicted-deviation band for a postprocessor family on **f32 input**
/// (RESEARCH lines 327-334).
///
/// The committed report carries this PREDICTED band alongside the MEASURED ROCm
/// column; a measured value materially outside its band is itself a finding
/// (e.g. a `native_exp` mapping, Pitfall 2). Returns an inclusive `(low, high)`
/// in absolute `|delta|`. Families that touch no transcendental sit effectively
/// at zero; the `exp`-family upper bounds reach into / past the `1e-5` band on
/// large margins (where D-02 f64 promotion applies).
pub fn predicted_band(postprocessor: &str) -> (f64, f64) {
    match postprocessor {
        // No transcendental — basic-op rounding only; effectively 0.
        "identity" | "identity_multiclass" | "hinge" | "signed_square" => (0.0, 1e-6),
        // Single `exp`, saturating — ~1e-6 to ~5e-6.
        "sigmoid" | "multiclass_ova" => (1e-6, 5e-6),
        // Unbounded `exp` — the class most likely to need f64 promotion (D-02);
        // may exceed 1e-5 absolute on large margins.
        "exponential" => (1e-6, 1.5e-5),
        // Bounded `exp2(-x/c)` in (0,1] — ~1e-6 to ~5e-6.
        "exponential_standard_ratio" => (1e-6, 5e-6),
        // Compounds two transcendentals (`exp` then `ln1p`) — ~2e-6 to ~1e-5.
        "logarithm_one_plus_exp" => (2e-6, 1e-5),
        // Per-class `exp` + f64 accumulate — ~1e-6 to ~5e-6, compounding.
        "softmax" => (1e-6, 5e-6),
        // Unknown / future postprocessor: no band claimed (0,0); the measured
        // column still records, but the predicted column reads "n/a" downstream.
        _ => (0.0, 0.0),
    }
}

/// Render an `Option<f64>` measured cell: the value (when run) or
/// `"not run — no device"` (D-05) when the backend was absent.
fn measured_cell(v: Option<f64>) -> String {
    match v {
        Some(d) => format!("{d:e}"),
        None => "not run — no device".to_string(),
    }
}

/// Render the predicted band cell, or `"n/a"` for an unknown postprocessor.
fn predicted_cell(low: f64, high: f64) -> String {
    if low == 0.0 && high == 0.0 {
        "n/a".to_string()
    } else {
        format!("~{low:e}..{high:e}")
    }
}

/// Build the committed markdown report body (GPU-04) from the per-class
/// [`ReportRow`]s + the run's provenance [`Manifest`] (D-06/D-08).
///
/// The header carries the device / ROCm / rustc / date provenance, the stated
/// reference (the f64 scalar GTIL spine), the "Observational — NOT a CI gate"
/// banner (D-01), and the honest determinism note. One row per frozen-golden
/// model class (D-07), columns exactly: model class | postprocessor | ROCm
/// max-|delta| | f64 fallback used? | CUDA | wgpu | predicted band (D-03).
pub fn render_markdown(rows: &[ReportRow], manifest: &Manifest, device_name: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# GPU Equivalence Report");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Regenerated on: {device} / ROCm {rocm} / {rustc} / captured-on {os} {arch} \
         (provenance from the run manifest, D-06)",
        device = device_name,
        rocm = manifest.cubecl.as_deref().unwrap_or("n/a"),
        rustc = manifest.rustc.as_deref().unwrap_or("unknown rustc"),
        os = manifest.os,
        arch = manifest.arch,
    );
    let _ = writeln!(
        s,
        "Reference: f64 scalar GTIL (the 1e-5 CPU spine). \
         **Observational — NOT a CI gate (D-01).**"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "| Model class | Postprocessor | ROCm max \\|delta\\| | f64 fallback used? | CUDA | wgpu | Predicted band (D-03) |"
    );
    let _ = writeln!(
        s,
        "|-------------|---------------|--------------------|--------------------|------|------|-----------------------|"
    );
    for r in rows {
        let _ = writeln!(
            s,
            "| {model} | {post} | {rocm} | {fb} | {cuda} | {wgpu} | {band} |",
            model = r.model_class,
            post = r.postprocessor,
            rocm = measured_cell(r.rocm_max_abs_delta),
            fb = if r.f64_fallback_used { "yes" } else { "no" },
            cuda = measured_cell(r.cuda_max_abs_delta),
            wgpu = measured_cell(r.wgpu_max_abs_delta),
            band = predicted_cell(r.predicted_low, r.predicted_high),
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Determinism: observed run-to-run stable on {device}; bit-identity NOT \
         guaranteed on GPU (per the OpenCL spec — transcendental rounding and \
         float-reduction order are implementation-defined). The Phase-6 SC2 \
         bit-identical claim is a CPU-backend property.",
        device = device_name,
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "A measured ROCm |delta| materially above its predicted band is itself a \
         finding worth recording (e.g. a `native_exp` transcendental mapping, \
         Pitfall 2) — not a CI failure (D-01)."
    );
    s
}

/// Build the machine-readable JSON sidecar (`gpu_equivalence.json`, D-08).
///
/// `[{model_class, postprocessor, backend, max_abs_delta, f64_fallback,
/// predicted_low, predicted_high}]` — observational (a future regression check
/// may assert the report didn't silently drift), NEVER asserted here as a gate.
pub fn render_json(rows: &[ReportRow]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "model_class": r.model_class,
                "postprocessor": r.postprocessor,
                "backend": "rocm",
                "max_abs_delta": r.rocm_max_abs_delta,
                "f64_fallback": r.f64_fallback_used,
                "cuda_max_abs_delta": r.cuda_max_abs_delta,
                "wgpu_max_abs_delta": r.wgpu_max_abs_delta,
                "predicted_low": r.predicted_low,
                "predicted_high": r.predicted_high,
            })
        })
        .collect();
    serde_json::Value::Array(arr)
}

/// Emit BOTH committed artifacts (D-06/D-08): the markdown report at
/// `report_md_path` and the JSON sidecar `gpu_equivalence.json` written
/// alongside it. The report is regenerated from the EXECUTED ROCm path (never
/// hand-edited, D-06); this is the only writer.
pub fn emit(
    rows: &[ReportRow],
    manifest: &Manifest,
    device_name: &str,
    report_md_path: &Path,
) -> anyhow::Result<()> {
    let md = render_markdown(rows, manifest, device_name);
    if let Some(parent) = report_md_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(report_md_path, md)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", report_md_path.display()))?;

    let json_path = report_md_path.with_file_name("gpu_equivalence.json");
    let json = render_json(rows);
    let json_text = serde_json::to_string_pretty(&json)?;
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(&json_path, json_text)
        .map_err(|e| anyhow::anyhow!("writing {}: {e}", json_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_mode_records_max_finite_delta_without_panicking() {
        // Finite deltas: max is recorded.
        let got = [1.0, 2.0, 3.0];
        let want = [1.0, 2.0 + 3e-6, 3.0 - 1e-6];
        let d = max_abs_delta_report_mode(&got, &want);
        assert!((d - 3e-6).abs() < 1e-12, "got {d:e}");
    }

    #[test]
    fn report_mode_skips_nan_and_non_finite_cells() {
        // A NaN and an inf cell are skipped structurally; the finite 5e-6 stands.
        let got = [f64::NAN, f64::INFINITY, 10.0];
        let want = [0.0, 1.0, 10.0 + 5e-6];
        let d = max_abs_delta_report_mode(&got, &want);
        assert!((d - 5e-6).abs() < 1e-12, "got {d:e}");
    }

    #[test]
    fn report_mode_never_panics_above_1e5() {
        // A delta far above 1e-5 RECORDS (does not panic) — the D-01 contract.
        let got = [0.0];
        let want = [1.0];
        let d = max_abs_delta_report_mode(&got, &want);
        assert!((d - 1.0).abs() < 1e-12, "got {d:e}");
    }

    #[test]
    fn length_mismatch_is_a_recordable_nan_not_a_panic() {
        let d = max_abs_delta_report_mode(&[1.0, 2.0], &[1.0]);
        assert!(d.is_nan());
    }

    #[test]
    fn predicted_bands_match_the_d03_table() {
        assert_eq!(predicted_band("identity"), (0.0, 1e-6));
        assert_eq!(predicted_band("sigmoid"), (1e-6, 5e-6));
        assert_eq!(predicted_band("exponential_standard_ratio"), (1e-6, 5e-6));
        assert_eq!(predicted_band("softmax"), (1e-6, 5e-6));
        // Unknown postprocessor: no band claimed.
        assert_eq!(predicted_band("some_future_postproc"), (0.0, 0.0));
    }
}

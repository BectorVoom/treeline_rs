//! Plan 07-04 — the CPU/GPU wall-clock crossover sweep (SC3/D-09/D-10).
//!
//! A `#[ignore]`'d, `rocm`-feature-gated wall-clock benchmark the developer runs
//! on the AMD/ROCm box to MEASURE — never pre-commit (D-10) — where the ROCm
//! backend's wall-clock first beats the CPU baseline, and which input metric
//! dominates the crossover (row count vs `rows × features`). The empirical
//! number + dominant metric are written to `docs/GPU_CROSSOVER.md`.
//!
//! ## DOCUMENTED-ONLY — `predict()` does NOT auto-route (D-09)
//!
//! This is a heuristic for a HUMAN consumer (the Phase-8 PyO3 caller), NOT a
//! routing rule baked into the engine. `treelite_cubecl::predict::<R, _>` runs
//! exactly the backend it is told to (explicit-selection, D-04); there is no
//! "below N rows, silently use CPU" branch anywhere. CPU stays the DEFAULT
//! backend. The crossover doc exists to inform a caller's explicit choice.
//!
//! ## What is timed
//!
//! The GPU timing INCLUDES the full per-call cost — forest+input upload, kernel
//! launch, and readback — because that fixed cost is exactly what the crossover
//! amortizes (RESEARCH lines 373-377). The CPU baseline is `cubecl_cpu_case()`
//! (the same kernels on `CpuRuntime`) so the comparison isolates the device
//! transfer/launch overhead rather than a kernel-vs-scalar difference.
#![cfg(feature = "rocm")]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use treelite_core::Model;
use treelite_gtil::{Config, PredictKind};
use treelite_harness::{RunnerCase, cubecl_cpu_case, rocm_case};

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

/// Is this harness error the typed `DeviceUnavailable` skip (D-05)?
fn is_device_absent(err: &anyhow::Error) -> bool {
    err.to_string().contains("no device available")
}

/// Load a frozen model by class name (the `.model.bin` next to the goldens).
fn load_model(class: &str) -> anyhow::Result<Model> {
    let model_path = fixture_path("gtil").join(format!("{class}.model.bin"));
    let bytes = std::fs::read(&model_path)
        .with_context(|| format!("reading model {}", model_path.display()))?;
    treelite_core::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("deserializing model {class}"))
}

/// A deterministic synthetic f32 input matrix: `num_row × num_feature` filled
/// with a cheap reproducible pattern (NOT random — the crossover only needs
/// representative work, not a golden). Values stay in a benign finite range so
/// every row exercises a real root-to-leaf traversal.
fn synth_input(num_row: usize, num_feature: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(num_row * num_feature);
    for r in 0..num_row {
        for f in 0..num_feature {
            // A small, deterministic, well-spread value.
            let x = ((r * 31 + f * 17) % 97) as f32 / 13.0 - 3.0;
            v.push(x);
        }
    }
    v
}

/// Time a single dense f32 predict call (median of a few reps) in nanoseconds.
/// Returns `None` if the backend reports no device (skip, D-05).
fn time_dense_f32(
    case: &RunnerCase,
    model: &Model,
    data: &[f32],
    num_row: usize,
    cfg: &Config,
    reps: usize,
) -> anyhow::Result<Option<u128>> {
    // One warm-up (JIT/driver init, alloc pools) excluded from the timing.
    match (case.dense_f32)(model, data, num_row, cfg) {
        Ok(_) => {}
        Err(e) if is_device_absent(&e) => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut samples: Vec<u128> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let out = (case.dense_f32)(model, data, num_row, cfg);
        let dt = t0.elapsed().as_nanos();
        match out {
            Ok(_) => samples.push(dt),
            Err(e) if is_device_absent(&e) => return Ok(None),
            Err(e) => return Err(e),
        }
    }
    samples.sort_unstable();
    Ok(Some(samples[samples.len() / 2]))
}

/// SC3/D-10: sweep `num_row` for a couple of forest classes, time CPU
/// (`cubecl_cpu_case`) vs ROCm (`rocm_case`) per call (incl. upload+launch+
/// readback), find the empirical crossover row count + dominant metric, and
/// write `docs/GPU_CROSSOVER.md`. Run explicitly on ROCm hardware (`#[ignore]`).
#[test]
#[ignore = "ROCm hardware only — run explicitly to regenerate docs/GPU_CROSSOVER.md (D-10)"]
fn gpu_crossover() -> anyhow::Result<()> {
    let cpu = cubecl_cpu_case();
    let gpu = rocm_case();
    let cfg = Config {
        kind: PredictKind::Default,
        nthread: 0,
    };
    let reps = 5usize;
    let row_sweep = [1usize, 10, 100, 1_000, 10_000, 100_000];

    // A couple of forest sizes: the small `binary` (4 features) and the wider
    // `leaf_vec_mc` multiclass forest. Both are kLT numerical so they run the
    // ROCm kernel (not the scalar fallback).
    let forests = [("binary", 4usize), ("leaf_vec_mc", 4usize)];

    let mut md = String::new();
    let _ = writeln!(md, "# CPU/GPU Crossover (SC3, documented-only)");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "Measured wall-clock per dense `predict` call (median of {reps} reps, \
         incl. upload + launch + readback) on the developer's AMD/ROCm hardware. \
         CPU baseline = `cubecl_cpu_case()` (same kernels on `CpuRuntime`); GPU = \
         `rocm_case()` (`predict::<HipRuntime, _>`)."
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "**DOCUMENTED-ONLY (D-09): `predict()` does NOT auto-route.** CPU stays \
         the DEFAULT backend; the engine runs exactly the backend it is told \
         (explicit-selection, D-04). This crossover informs a caller's explicit \
         choice (the Phase-8 PyO3 consumer) — it is never a routing rule baked \
         into the engine."
    );
    let _ = writeln!(md);

    let mut any_gpu_ran = false;
    let mut findings: Vec<String> = Vec::new();

    for (class, num_feature) in forests {
        let model = load_model(class)?;
        let _ = writeln!(md, "## Forest `{class}` ({num_feature} features)");
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "| num_row | rows×features | CPU (ns) | ROCm (ns) | ROCm faster? |"
        );
        let _ = writeln!(
            md,
            "|---------|---------------|----------|-----------|--------------|"
        );

        let mut crossover_row: Option<usize> = None;
        for &num_row in &row_sweep {
            let data = synth_input(num_row, num_feature);
            let cpu_ns = time_dense_f32(&cpu, &model, &data, num_row, &cfg, reps)?
                .expect("cubecl_cpu_case always has a device (CpuRuntime)");
            let gpu_ns = time_dense_f32(&gpu, &model, &data, num_row, &cfg, reps)?;

            let (gpu_cell, faster_cell) = match gpu_ns {
                Some(g) => {
                    any_gpu_ran = true;
                    let faster = g < cpu_ns;
                    if faster && crossover_row.is_none() {
                        crossover_row = Some(num_row);
                    }
                    (format!("{g}"), if faster { "yes" } else { "no" }.to_string())
                }
                None => ("not run — no device".to_string(), "—".to_string()),
            };

            let _ = writeln!(
                md,
                "| {num_row} | {} | {cpu_ns} | {gpu_cell} | {faster_cell} |",
                num_row * num_feature,
            );
        }
        let _ = writeln!(md);

        match crossover_row {
            Some(n) => findings.push(format!(
                "On `{class}` ({num_feature} features), ROCm wall-clock first beats \
                 `cubecl_cpu_case` at ~{n} rows."
            )),
            None => findings.push(format!(
                "On `{class}` ({num_feature} features), ROCm did NOT beat the CPU \
                 across the swept range (1..100k rows) — or no device was present."
            )),
        }
    }

    let _ = writeln!(md, "## Empirical crossover + dominant metric");
    let _ = writeln!(md);
    for f in &findings {
        let _ = writeln!(md, "- {f}");
    }
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "**Dominant metric (let the data decide, D-10):** the GPU pays a fixed \
         upload+launch+readback cost per call, amortized over rows — so the \
         crossover keys on **row count** (or `rows × features` for the input \
         transfer when the forest is small). Compare the two forests' crossover \
         rows above: if they crossover at a similar `rows × features` product \
         rather than a similar row count, the input transfer dominates; if they \
         crossover at a similar row count, the per-row traversal dominates. No \
         formula is pre-committed (D-10) — the table is the evidence."
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "Consumer: the Phase-8 PyO3 caller (D-09) is the documented consumer of \
         this heuristic. CPU remains the default backend."
    );

    let out_path = docs_path("GPU_CROSSOVER.md");
    std::fs::write(&out_path, &md)
        .with_context(|| format!("writing {}", out_path.display()))?;

    eprintln!(
        "gpu_crossover: swept {} row sizes × {} forests; ROCm {} → {}. \
         DOCUMENTED-ONLY (predict() does not auto-route, D-09).",
        row_sweep.len(),
        forests.len(),
        if any_gpu_ran { "ran" } else { "absent (no device)" },
        out_path.display(),
    );
    Ok(())
}

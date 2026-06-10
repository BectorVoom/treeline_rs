---
phase: 07-gpu-backend-equivalence-report
plan: 04
subsystem: harness
tags: [harness, gpu, rocm, equivalence-report, crossover, report-mode, observational, gpu-03, gpu-04]

# Dependency graph
requires:
  - phase: 07-gpu-backend-equivalence-report
    plan: 03
    provides: "Backend::Rocm + rocm_case() RunnerCase constructor (dense f32/f64 → predict::<HipRuntime, _>, sparse → scalar fallback, D-02); ROCm selectable end-to-end"
  - phase: 07-gpu-backend-equivalence-report
    plan: 02
    provides: "treelite_cubecl::predict::<R, F> runtime-generic launcher driving the ROCm dense cells"
provides:
  - "report.rs report-mode max-|delta| accumulator + markdown/JSON emitter (now creates docs/ parent dir before writing)"
  - "gtil_matrix_gpu.rs #[ignore]'d ROCm sibling: drives the frozen golden matrix in report mode on AMD hardware, emits GPU_EQUIVALENCE_REPORT.md + gpu_equivalence.json"
  - "gpu_crossover.rs #[ignore]'d wall-clock CPU-vs-ROCm sweep → GPU_CROSSOVER.md (documented-only, D-09/D-10)"
  - "docs/GPU_EQUIVALENCE_REPORT.md — committed, regenerated on-hardware (GPU-04): one row per frozen-golden model class, observational (D-01)"
  - "docs/gpu_equivalence.json — machine-readable sidecar of the same report"
  - "docs/GPU_CROSSOVER.md — measured ROCm-vs-CPU crossover (~100k rows both forests)"
  - "GPU-03 and GPU-04 satisfied ON-HARDWARE: 160 equivalence cells ran on AMD/ROCm, all within predicted bands « 1e-5"
affects: [07-verification, gpu-equivalence-report, gpu-03, gpu-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Report-mode (D-01): the GPU sibling RECORDS max-|delta| per model class and NEVER gates on 1e-5 — the CPU 1e-5 gate stays on the scalar/cubecl-cpu siblings, untouched. A GPU |delta| above its predicted band is a finding, not a test failure."
    - "Regenerated-never-hand-edited (D-06): both report artifacts are written by EXECUTING the ROCm-feature sibling on the developer's AMD box; the markdown/JSON are emitted from the per-cell provenance, never authored by hand."
    - "Documented-only crossover (D-09/D-10): predict() does NOT auto-route; GPU_CROSSOVER.md is a heuristic for the Phase-8 PyO3 consumer's EXPLICIT backend choice. CPU stays the default backend; no row-count routing branch exists in the engine."
    - "Artifact writers create_dir_all(parent) before std::fs::write — the docs/ dir is generated output, not a committed-empty scaffold, so the writers own its creation."

key-files:
  created:
    - docs/GPU_EQUIVALENCE_REPORT.md
    - docs/gpu_equivalence.json
    - docs/GPU_CROSSOVER.md
  modified:
    - crates/treelite-harness/src/report.rs
    - crates/treelite-harness/tests/gpu_crossover.rs

key-decisions:
  - "docs/ parent-dir creation is the writers' responsibility — both report.rs (the .md and .json writes) and gpu_crossover.rs now create_dir_all(parent) before writing. This was the root cause of the checkpoint write failure (docs/ did not exist on a clean tree)."
  - "The crossover sweep reads num_feature from each loaded Model (model.num_feature) instead of a hardcoded literal — the hardcoded (binary,4)/(leaf_vec_mc,4) under-sized the synthetic input for the 5-feature forest and panicked 'input buffer too small'. Deriving from the model makes synth_input always match the model's expected stride."

requirements-completed: [GPU-03, GPU-04]

# Metrics
duration: ~12min
completed: 2026-06-11
---

# Phase 7 Plan 04: GPU Equivalence Report + Crossover on ROCm Hardware Summary

**Ran the report-mode ROCm equivalence sibling and the CPU/GPU crossover sweep on the developer's AMD/ROCm hardware and committed the three regenerated artifacts. 160 equivalence cells (5 model classes × postprocessors × dense/sparse × f32/f64) all executed on the GPU and landed within their predicted deviation bands — worst observed `max |delta| = 2.9e-6` (large_margin f64 sigmoid), most at `0e0` — well inside the 1e-5 contract, recorded OBSERVATIONALLY (never a CI gate, D-01). CUDA and wgpu render `not run — no device` on this NVIDIA-less box. The wall-clock crossover shows ROCm first beats the CPU baseline at ~100k rows for both swept forests. GPU-03 (a GPU backend produces predictions) and GPU-04 (a committed per-model-class deviation report) are now satisfied on-hardware. Two deviation fixes were required to make the run write its artifacts; the default `cargo test --workspace` stays green.**

## Performance

- **Duration:** ~12 min (continuation past the human-action checkpoint)
- **Tasks:** continuation — docs-dir fix + crossover fix + 2 on-hardware regenerations + artifact commit
- **Files:** 3 created (artifacts), 2 modified (writers)

## Accomplishments

- **Docs-dir fix (commit `35b3896`).** `report.rs` now `create_dir_all(parent)` before both the `.md` and `.json` `std::fs::write`; `gpu_crossover.rs` does the same before `GPU_CROSSOVER.md`. This was the exact root cause of the checkpoint failure (`No such file or directory` — `docs/` did not exist). Report content/format and measurement logic unchanged.
- **Crossover num_feature fix (commit `7c832ea`).** The sweep now reads `model.num_feature` per forest instead of the hardcoded `4`; the hardcoded value under-sized the synthetic input matrix for the 5-feature forest and panicked `input buffer too small`. Rule-1 bug surfaced only on the actual hardware run.
- **ROCm equivalence regenerated on-hardware (commit `164f6f1`).** `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu -- --ignored` PASSED: **160 cells ran on ROCm, 0 skipped**, 5 model-class rows written. All `max |delta|` at `0e0` or fp-epsilon, worst `2.9e-6` (large_margin), every cell within its predicted band and « 1e-5 — recorded observationally (D-01).
- **ROCm crossover regenerated on-hardware (commit `164f6f1`).** `cargo test -p treelite-harness --features rocm --test gpu_crossover -- --ignored` PASSED: swept 6 row sizes × 2 forests; ROCm first beats `cubecl_cpu_case` at ~100k rows for both `binary` and `leaf_vec_mc`. Documented-only (predict() does not auto-route, D-09).
- **Default CI gate unaffected.** `cargo test --workspace` (no features) fully green — the GPU report path is `#[ignore]`'d + rocm-feature-gated, so the CPU 1e-5 baseline is untouched.

## Measured Equivalence (observational, D-01)

| Model class | Postprocessor | ROCm max \|delta\| | Predicted band | In band? |
|-------------|---------------|--------------------|----------------|----------|
| binary | sigmoid | 2.23e-7 | ~1e-6..5e-6 | yes |
| large_margin | sigmoid | 2.91e-6 | ~1e-6..5e-6 | yes |
| leaf_vec_mc | softmax | 2.31e-7 | ~1e-6..5e-6 | yes |
| lgbm_numerical | identity | 2.83e-7 | ~0e0..1e-6 | yes |
| mixedwidth | identity | 0e0 | ~0e0..1e-6 | yes |

All five rows show `f64 fallback used? = yes` (the sparse cells route through the scalar/f64 fallback, D-02, OR'd into the class flag). CUDA/wgpu = `not run — no device`. Crossover: ROCm wall-clock first beats CPU at ~100k rows (both forests).

## Task Commits

1. **Docs-dir parent creation before artifact writes** — `35b3896` (fix)
2. **Crossover num_feature derived from model, not hardcoded** — `7c832ea` (fix)
3. **Regenerated GPU equivalence report + crossover on ROCm hardware (GPU-04)** — `164f6f1` (docs)

**Plan metadata:** committed separately with SUMMARY/STATE/ROADMAP/REQUIREMENTS.

## Files

- `crates/treelite-harness/src/report.rs` — `create_dir_all(parent)` before the `.md` and `.json` writes.
- `crates/treelite-harness/tests/gpu_crossover.rs` — `create_dir_all(parent)` before `GPU_CROSSOVER.md`; `num_feature` read from `model.num_feature`.
- `docs/GPU_EQUIVALENCE_REPORT.md` (created) — committed per-model-class deviation report (GPU-04).
- `docs/gpu_equivalence.json` (created) — machine-readable sidecar.
- `docs/GPU_CROSSOVER.md` (created) — documented-only crossover heuristic (SC3/D-09/D-10).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `docs/` parent directory did not exist; artifact writers failed with `No such file or directory`**
- **Found during:** the orchestrator's pre-checkpoint ROCm run (the EXECUTED-with-perfect-equivalence run that failed at the final write step).
- **Issue:** `report.rs` and `gpu_crossover.rs` called `std::fs::write(...)` on a `docs/<file>` path without ensuring `docs/` exists. On a clean tree the directory is absent, so the otherwise-successful 0e0-equivalence run aborted at write time (`os error 2`).
- **Fix:** `if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }` before each write — both report writes in `report.rs` and the crossover write in `gpu_crossover.rs`. Minimal and idiomatic; report content, format, and measurement logic untouched.
- **Files modified:** `crates/treelite-harness/src/report.rs`, `crates/treelite-harness/tests/gpu_crossover.rs`.
- **Verification:** both rocm-feature siblings now PASS and write their artifacts; the 3 files exist on disk.
- **Committed in:** `35b3896`.

**2. [Rule 1 - Bug] Crossover hardcoded `num_feature = 4`, under-sizing the synthetic input for the 5-feature forest**
- **Found during:** Task 2 (the first on-hardware `gpu_crossover` run after the docs-dir fix).
- **Issue:** `gpu_crossover.rs` declared `forests = [("binary", 4), ("leaf_vec_mc", 4)]`, but one forest actually has 5 features. `synth_input(1, 4)` produced a 4-element row for a model expecting a stride of 5, panicking `input buffer too small for shape: num_row = 1, num_feature = 5 requires 5 elements, got 4`.
- **Fix:** dropped the hardcoded literal; the loop now reads `num_feature` from `model.num_feature` (via `usize::try_from`) after loading each forest, so `synth_input` always matches the model's expected stride. The crossover sweep + dominant-metric logic is otherwise unchanged.
- **Files modified:** `crates/treelite-harness/tests/gpu_crossover.rs`.
- **Verification:** `gpu_crossover` now PASSES on ROCm and writes `GPU_CROSSOVER.md` with the correct per-forest feature counts (4 and 5).
- **Committed in:** `7c832ea`.

---

**Total deviations:** 2 auto-fixed (1 blocking write-path, 1 input-sizing bug). Neither changes the report content/format or the measurement contract.

## Verification Results

- `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu -- --ignored --nocapture` — **PASS**: 160 cells ran on ROCm, 0 skipped; 5 model-class rows; worst `max |delta| = 2.9e-6` « 1e-5; report + JSON written.
- `cargo test -p treelite-harness --features rocm --test gpu_crossover -- --ignored --nocapture` — **PASS**: 6 rows × 2 forests; ROCm beats CPU at ~100k rows; `GPU_CROSSOVER.md` written.
- `cargo test --workspace` (default, no features) — fully green, zero failures; the CPU 1e-5 baseline is untouched.
- `cargo build -p treelite-harness --features rocm --tests` — green (both edited rocm-gated targets compile).
- Artifacts on disk: `docs/GPU_EQUIVALENCE_REPORT.md`, `docs/gpu_equivalence.json`, `docs/GPU_CROSSOVER.md` — all present and committed in `164f6f1`.

## Requirement Status

- **GPU-04** ("a GPU equivalence report documents observed deviation per model class within an accepted tolerance") — **COMPLETE**: `docs/GPU_EQUIVALENCE_REPORT.md` is committed, one row per frozen-golden model class, regenerated on-hardware from the executed ROCm path, all within predicted bands « 1e-5.
- **GPU-03** ("at least one GPU backend is runtime-selectable AND produces predictions") — **COMPLETE on-hardware**: the registration path landed in 07-03; this plan executes `rocm_case()` on the AMD device producing 160 real predictions matched against the frozen goldens. ROCm is the hardware-validated v1 GPU backend.

Marked complete in REQUIREMENTS.md per the orchestrator's explicit approval. **Phase completion remains the orchestrator's job** after the verifier runs — the Phase-7 ROADMAP phase checkbox is NOT checked here.

## Self-Check: PASSED

- `docs/GPU_EQUIVALENCE_REPORT.md` exists on disk (created).
- `docs/gpu_equivalence.json` exists on disk (created).
- `docs/GPU_CROSSOVER.md` exists on disk (created).
- `crates/treelite-harness/src/report.rs` exists on disk (modified).
- `crates/treelite-harness/tests/gpu_crossover.rs` exists on disk (modified).
- Commits present in git history: `35b3896` (docs-dir fix), `7c832ea` (num_feature fix), `164f6f1` (artifacts), plus prior `51c8bfc` (Task 1) and `28a3781` (Task 2).

---
*Phase: 07-gpu-backend-equivalence-report*
*Completed: 2026-06-11*

---
phase: 07-gpu-backend-equivalence-report
plan: 02
subsystem: cubecl
tags: [cubecl, runtime-generic, predict, launcher, gpu, rocm, cuda, wgpu, shim]

# Dependency graph
requires:
  - phase: 07-gpu-backend-equivalence-report
    plan: 01
    provides: "device::client::<R>() typed-skip constructor + rocm/cuda/wgpu cargo features + CubeclError::DeviceUnavailable"
provides:
  - "treelite_cubecl::predict::<R, F> — the runtime-generic GTIL launcher; selects its client via device::client::<R>() and runs the same kernels on CPU/ROCm/CUDA/wgpu"
  - "treelite_cubecl::predict_cpu::<F> reduced to a thin shim over predict::<CpuRuntime, F> (registration-not-refactor; harness surface byte-identical)"
  - "Verified: the generic predict::<R,F> compiles under --features cuda (Plan 03 may call predict::<HipRuntime/CudaRuntime/WgpuRuntime, F> directly)"
affects: [07-03, 07-04, gpu-backend-registration, harness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-file runtime-genericization: the four launcher fn pairs + three launch sites + three upload calls lifted from CpuRuntime to R: Runtime; kernels/upload were ALREADY generic and stay untouched (the smell guard)"
    - "Client construction routes through Plan-01's device::client::<R>(tag); DeviceUnavailable propagates via ? — NO silent CPU fallback (D-05/D-09)"
    - "predict_cpu kept as a one-line shim over predict::<CpuRuntime, F> so the Phase-6 cubecl-cpu harness compiles unchanged"

key-files:
  created: []
  modified:
    - crates/treelite-cubecl/src/lib.rs

key-decisions:
  - "device::client::<R>(tag) requires a &'static str backend tag (Plan-01 signature), while RESEARCH's sketch wrote device_and_client::<R>() with no arg. The generic predict::<R> cannot statically know R's backend name, so it passes std::any::type_name::<R>() as the tag — a meaningful, R-derived identity carried into any DeviceUnavailable skip on the generic path. The CpuRuntime shim always succeeds, so the tag is never surfaced on the CPU path."
  - "predict::<R> carries a `where R::Device: Default` bound — required by device::client::<R>() (Plan-01). CpuRuntime::Device: Default satisfies the shim; the cuda-feature build confirms HipRuntime/CudaRuntime device types satisfy it too."
  - "The model_routes_to_scalar_fallback gate stays BEFORE client construction (verbatim from Phase 6), so a categorical / non-kLT model on a device-less backend still returns via the scalar reference without ever constructing a GPU client (D-02 short-circuit preserved)."

patterns-established:
  - "Pattern: Phase-7's central code change is exactly ONE file (lib.rs). kernels/* and upload.rs are already R-generic; editing them would be the documented smell. git diff --stat on both shows 0 changed lines."

requirements-completed: []

# Metrics
duration: ~9min
completed: 2026-06-10
---

# Phase 7 Plan 02: Runtime-Generic cubecl Launcher (`predict::<R, F>`) Summary

**Generalized the cubecl host launcher from the hardcoded `CpuRuntime` to a generic `R: Runtime` in a single file — adding `predict::<R, F>` (wired to Plan-01's `device::client::<R>()` typed-skip constructor, no silent CPU fallback) plus a thin `predict_cpu` shim that keeps the Phase-6 cubecl-cpu 1e-5 gate byte-identical, with `kernels/*` and `upload.rs` untouched (the smell guard).**

## Performance

- **Duration:** ~9 min
- **Tasks:** 2
- **Files modified:** 1 (0 created, 1 modified)

## Accomplishments

- **Task 1 — threaded `R` through the launcher (commit `de28f8b`).** Made `launch_default_raw`/`run_default_raw`, `launch_leaf_id`/`run_leaf_id`, `launch_score_per_tree`/`run_score_per_tree` generic over `R: Runtime`: lifted all six `ComputeClient<CpuRuntime>` params to `ComputeClient<R>`, retargeted the three `launch::<F, T, CpuRuntime>` sites to `launch::<F, T, R>` and the three `upload_forest::<CpuRuntime, T>` calls to `upload_forest::<R, T>`. Everything between the type args is byte-identical — the kernels and upload were already `R`-generic.
- **Task 2 — added the generic entry + shim (commit `3f1620a`).** Added `pub fn predict<R: Runtime, F: PredictCpuElem>(...) where R::Device: Default`, keeping the `model_routes_to_scalar_fallback` D-02 gate at the top (so a fallback-routed model never needs a device), then `device::client::<R>(...)?` (the typed skip propagates — NO silent CPU fallback), then the same `match cfg.kind` arms calling `launch_*::<R, F>`. Reduced `predict_cpu::<F>` to a one-line shim: `predict::<CpuRuntime, F>(...)`.
- **Verified the generalization is genuinely runtime-generic:** `cargo build -p treelite-cubecl --features cuda` compiles the generic path, so Plan 03 can call `predict::<CudaRuntime/HipRuntime/WgpuRuntime, F>` directly with no further launcher work.

## Task Commits

1. **Task 1: thread R through the four launcher fns + launch/upload sites** — `de28f8b` (refactor)
2. **Task 2: generic predict::<R,F> + predict_cpu shim wiring device::client::<R>()** — `3f1620a` (feat)

**Plan metadata:** committed separately with SUMMARY/STATE/ROADMAP.

## Files Modified

- `crates/treelite-cubecl/src/lib.rs` — six launcher signatures lifted to `ComputeClient<R>`; three `launch::<F,T,R>` sites + three `upload_forest::<R,T>` calls; new `predict::<R,F>` entry wiring `device::client::<R>()`; `predict_cpu` reduced to a `predict::<CpuRuntime, F>` shim; doc comments updated to describe the runtime-generic launcher.

## Decisions Made

- **Backend tag for the generic path** — `device::client::<R>(tag)` (Plan-01) requires a `&'static str` tag, but the generic `predict::<R>` can't statically name `R`'s backend. Passed `std::any::type_name::<R>()` so any `DeviceUnavailable` skip on the generic path still carries an `R`-derived identity. The CpuRuntime shim always succeeds, so the tag is never surfaced on the default CPU path. (RESEARCH sketched `device_and_client::<R>()` with no arg; the implemented Plan-01 signature is the source of truth.)
- **`where R::Device: Default` bound** — mandated by `device::client::<R>()`. `CpuRuntime::Device: Default` satisfies the shim; the `--features cuda` build confirms the GPU device types satisfy it too.
- **Fallback gate stays pre-client** — `model_routes_to_scalar_fallback` is checked before `device::client::<R>()`, so a categorical / non-`kLT` model on a device-less backend returns via the scalar reference without ever constructing a GPU client (D-02 preserved, and a useful device-absence robustness property).

## Deviations from Plan

### Auto-fixed Issues

None requiring a code fix. One naming reconciliation (documented as a decision, not a deviation): the plan/RESEARCH wrote the client constructor call as `device::client::<R>()?` / `device_and_client::<R>()?` with no argument, but Plan-01's implemented signature is `client::<R>(backend: &'static str)`. Resolved by passing `std::any::type_name::<R>()` as the tag — the acceptance grep (`grep -c 'device::client::<R>'` == 1) still passes exactly.

**Total deviations:** 0 code fixes; 1 documented signature reconciliation.
**Impact on plan:** None — all acceptance criteria met verbatim; the central goal (one-file runtime-genericization) is achieved.

## Verification Results

- `cargo build -p treelite-cubecl` (default) — green.
- `cargo clippy -p treelite-cubecl -- -D warnings` — clean (both tasks).
- `cargo build -p treelite-cubecl --features cuda` — green (generic path compiles for a GPU runtime; Plan 03 readiness).
- `cargo test -p treelite-harness --test gtil_matrix_cubecl` — **ok, 1 passed** (the cubecl-cpu 1e-5 matrix gate is byte-identical via the shim).
- `cargo test --workspace` — fully green, **zero failures** across all suites.
- **Task 1 acceptance greps:** `ComputeClient<CpuRuntime>` = 0; `launch::<F, T, CpuRuntime>` = 0; `launch::<F, T, R>` = 3; `upload_forest::<CpuRuntime` = 0; `git diff --stat` on `upload.rs` + `kernels/` = **0 changed lines** (the smell guard).
- **Task 2 acceptance greps:** `pub fn predict<R` = 1; `device::client::<R>` = 1; `predict::<CpuRuntime, F>` = 1; `git diff` on `crates/treelite-harness/src/lib.rs` + `tests/gtil_matrix_cubecl.rs` = **0 changes** (harness untouched).

## Threat Model Compliance

- **T-07-03 (provenance / no silent fallback)** — mitigated: `device::client::<R>(...)?` propagates `DeviceUnavailable` via `?`; there is NO fallback-to-cpu branch in `predict::<R>`. The only scalar route is the pre-existing D-02 `model_routes_to_scalar_fallback` gate (categorical / non-kLT), which runs BEFORE client construction and is unrelated to device absence. Selected backend == backend that ran.
- **T-07-04 (OOB device read/write)** — unchanged: the host-side `validate_*` checks live in the already-generic `upload_forest`; this plan does not edit `upload.rs`, so they run before any device op on every `R`.
- **T-07-SC (cargo installs)** — no new crates added; pure source generalization.

## Requirement Status Note

The plan frontmatter lists `requirements: [GPU-03]`, but GPU-03 ("at least one GPU backend is runtime-selectable AND produces predictions") is phase-spanning and NOT satisfied by this launcher generalization alone — no GPU prediction runs here (backend registration + the harness GPU case is Plan 03). GPU-03 is left **In Progress**; the orchestrator marks it complete after end-of-phase verification. `requirements-completed` is therefore empty for this plan.

## Next Phase Readiness

- `predict::<R, F>` is the runtime-generic entry Plan 03 needs. Plan 03's `*_case()` harness constructors can call `predict::<HipRuntime, F>` / `predict::<CudaRuntime, F>` / `predict::<WgpuRuntime, F>` directly (the cuda-feature build confirms compilation), treating `Err(DeviceUnavailable)` as a skip-not-fail row per Plan-01's A3 finding.
- The CPU 1e-5 gate is preserved byte-identical via the `predict_cpu` shim, so Plan 03 adds GPU rows to the matrix without disturbing the proven CPU baseline.

## Self-Check: PASSED

- `crates/treelite-cubecl/src/lib.rs` exists on disk (modified).
- Both task commits present in git history: `de28f8b` (Task 1), `3f1620a` (Task 2).

---
*Phase: 07-gpu-backend-equivalence-report*
*Completed: 2026-06-10*

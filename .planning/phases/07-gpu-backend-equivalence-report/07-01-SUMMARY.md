---
phase: 07-gpu-backend-equivalence-report
plan: 01
subsystem: infra
tags: [cubecl, rocm, cuda, wgpu, gpu, cargo-features, thiserror, catch_unwind, device-absence]

# Dependency graph
requires:
  - phase: 06-cubecl-gtil-kernels-cpu-backend
    provides: "treelite-cubecl crate (CpuRuntime client at lib.rs:312, CubeclError enum, upload/kernels modules)"
provides:
  - "Additive rocm/cuda/wgpu cargo features on treelite-cubecl forwarding to the cubecl 0.10 umbrella (default = cpu-only)"
  - "CubeclError::DeviceUnavailable { backend: &'static str } typed-skip variant (D-05)"
  - "device.rs: generic client::<R>() + cfg-gated rocm_client/cuda_client/wgpu_client constructors that map a missing device to the typed skip"
  - "A3 RESOLVED: cubecl HIP/CUDA client() failure on a missing device is a CATCHABLE Rust panic (not an FFI abort) — Plan 03 needs no pre-construction device probe"
affects: [07-02, 07-03, 07-04, gpu-backend-registration, harness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive backend cargo features forward to the cubecl umbrella crate-locally; workspace root stays cpu-only (Pitfall 5)"
    - "Per-backend client construction via generic client::<R>() using <R::Device>::default(), wrapping R::client() in std::panic::catch_unwind → typed DeviceUnavailable (D-05)"

key-files:
  created:
    - crates/treelite-cubecl/src/device.rs
    - crates/treelite-cubecl/tests/device_absent.rs
  modified:
    - crates/treelite-cubecl/Cargo.toml
    - crates/treelite-cubecl/src/error.rs
    - crates/treelite-cubecl/src/lib.rs

key-decisions:
  - "A3 SETTLED: a missing CUDA device surfaces as a catchable Rust panic (cudarc dlopen(libcuda.so) failure), trapped by catch_unwind and mapped to DeviceUnavailable — NOT a hard FFI abort. Plan 03 may construct GPU clients directly; no pre-construction probe required."
  - "HipRuntime::Device is AmdDevice, not HipDevice (RESEARCH symbol corrected); device.rs uses the generic <R::Device>::default() form so the exact per-backend device type name is never hardcoded."
  - "rocm = [\"cubecl/rocm\"] is correct: cubecl's own feature table defines rocm = [\"hip\"], so cubecl/rocm enables cubecl::hip (HipRuntime + AmdDevice)."

patterns-established:
  - "Pattern: GPU backends are crate-local additive cargo features (rocm/cuda/wgpu) forwarding to cubecl/*; the workspace cubecl dep stays features=[\"cpu\"] so default builds need no GPU system libs (D-04)."
  - "Pattern: device-absence = typed CubeclError::DeviceUnavailable skip via catch_unwind, never a silent CPU fallback (D-05)."

requirements-completed: [GPU-03]

# Metrics
duration: ~8min
completed: 2026-06-10
---

# Phase 7 Plan 01: GPU Backend Feature Seam + A3 Device-Absence Spike Summary

**Additive `rocm`/`cuda`/`wgpu` cargo features + `CubeclError::DeviceUnavailable` typed skip + `device.rs` per-backend `ComputeClient<R>` constructors, with the Wave-0 A3 spike proving a missing CUDA device is a catchable Rust panic (not an FFI abort).**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-10T21:29Z
- **Completed:** 2026-06-10T21:37Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- Added a crate-local `[features]` table to `treelite-cubecl` (`default = []`, `rocm = ["cubecl/rocm"]`, `cuda = ["cubecl/cuda"]`, `wgpu = ["cubecl/wgpu"]`) forwarding to the cubecl 0.10 umbrella; the workspace root `Cargo.toml` stays `cubecl = { features = ["cpu"] }` so plain `cargo build` / `cargo test --workspace` need no GPU system libs (D-04 / Pitfall 5).
- Added the `CubeclError::DeviceUnavailable { backend: &'static str }` typed-skip variant (D-05); the `&'static str` tag keeps the enum's `PartialEq, Eq` derives intact.
- Authored `device.rs`: a generic `client::<R>()` plus `#[cfg(feature = …)]`-gated `rocm_client`/`cuda_client`/`wgpu_client`, each mapping a missing device to `DeviceUnavailable` via `std::panic::catch_unwind`. Declared `pub mod device;` in `lib.rs`.
- **Ran the MANDATORY A3 spike and RESOLVED the phase's single HIGH-risk unknown** (see below).

## A3 Spike Finding (RESEARCH Open Question 1 / Pitfall 1 — REQUIRED RECORD)

**Question:** Does cubecl HIP/CUDA `client()` return a catchable error or hard-FFI-abort on a missing device?

**Answer — CATCHABLE Rust panic, NOT an FFI abort.**

Built and run on this NVIDIA-less box with `cargo test -p treelite-cubecl --features cuda --test device_absent`:

- `cuda_client()` → `Err(CubeclError::DeviceUnavailable { backend: "cuda" })`.
- The test reported `ok`, process **exit 0** — no abort, no process teardown.
- `--nocapture` shows the underlying mechanism: `cudarc` fails to `dlopen` `libcuda.so` and **panics** at `cudarc-0.19.7/src/lib.rs:200` ("Unable to dynamically load the \"cuda\" shared library …"); that panic propagates through `cubecl-common-0.10.0/src/device/handle/channel.rs:327` up to the `catch_unwind` boundary inside `device::client::<R>()`, which maps it to the typed skip.

**Consequence for Plan 03:** A missing device is reachable via `catch_unwind` as written. **Plan 03 does NOT need a pre-construction device probe** — the per-backend `*_client()` helpers may construct clients directly; an absent backend yields the typed `DeviceUnavailable` skip the harness/report branches on. (Caveat: this is verified for the CUDA-driver-load path on this box; the `catch_unwind`-based skip is the standing mechanism. If a future backend on different hardware ever aborts below `catch_unwind`, that would re-open A3 — but no such behavior was observed here.)

## Task Commits

1. **Task 1: rocm/cuda/wgpu cargo features + DeviceUnavailable variant** — `44f256e` (feat)
2. **Task 2: device.rs constructors + A3 device-absence spike** — `bf4605f` (feat)

**Plan metadata:** committed separately with SUMMARY/STATE/ROADMAP.

## Files Created/Modified

- `crates/treelite-cubecl/Cargo.toml` — added the `[features]` table (default cpu-only; rocm/cuda/wgpu forward to cubecl).
- `crates/treelite-cubecl/src/error.rs` — added `CubeclError::DeviceUnavailable { backend }`.
- `crates/treelite-cubecl/src/lib.rs` — declared `pub mod device;`.
- `crates/treelite-cubecl/src/device.rs` — NEW: generic `client::<R>()` + cfg-gated `rocm_client`/`cuda_client`/`wgpu_client`, catch_unwind → `DeviceUnavailable`.
- `crates/treelite-cubecl/tests/device_absent.rs` — NEW: the A3 spike (cuda-feature) + a default-build no-op test.

## Decisions Made

- **A3 settled (catchable panic, not abort)** — recorded above; the single de-risk goal of this plan.
- **Generic device construction** — `device.rs` uses `<R::Device as Default>::default()` rather than hardcoding device type names. This was forced by a RESEARCH symbol error: `cubecl::hip::HipRuntime::Device` is `AmdDevice`, **not** `HipDevice` (no `HipDevice` type exists in cubecl 0.10). The generic form sidesteps the issue and is uniform across all three backends (all derive `Default`).
- **Feature forwarding confirmed against cubecl's own table** — `cubecl`'s `Cargo.toml` defines `rocm = ["hip"]` and `cpu/cuda/wgpu = ["cubecl-cpu/cuda/wgpu"]`, so `rocm = ["cubecl/rocm"]` correctly brings in `cubecl::hip` (HipRuntime + AmdDevice). `cubecl::{cuda,wgpu}` runtime/device symbols (`CudaRuntime`/`CudaDevice`, `WgpuRuntime`/`WgpuDevice`, the latter two `Default`-deriving) all verified in the vendored crate sources.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `ComputeClient<R>` does not implement `Debug` — spike assert wouldn't compile**
- **Found during:** Task 2 (A3 spike, under `--features cuda`)
- **Issue:** The first draft of the spike used `assert!(matches!(...), "... got {result:?} ...")`. `cubecl::client::ComputeClient<CudaRuntime>` does not implement `Debug`, so `{result:?}` failed to compile (`E0277`), blocking the spike from running.
- **Fix:** Replaced the `assert!`+`{result:?}` with an explicit `match` over `cuda_client()`: the `Ok(_client)` arm is `panic!`ed without debug-formatting the (non-`Debug`) client; only the typed-error arms render. The A3 assertion semantics are unchanged.
- **Files modified:** `crates/treelite-cubecl/tests/device_absent.rs`
- **Verification:** `cargo test -p treelite-cubecl --features cuda --test device_absent` compiles and passes (exit 0).
- **Committed in:** `bf4605f` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Necessary to run the mandatory A3 spike. No scope creep — the change is confined to the spike's assertion mechanics.

## Issues Encountered

- **Pre-existing clippy lint under `--tests` (OUT OF SCOPE, deferred).** `cargo clippy -p treelite-cubecl --tests -- -D warnings` fails on `tests/spike.rs:228` ("very complex type used"), a file committed in Phase 06 (`2fdff21`) and unmodified by this plan. The plan's exact verify command — `cargo clippy -p treelite-cubecl -- -D warnings` (no `--tests`) — is **clean**, and my own `tests/device_absent.rs` is clippy-clean under `--tests` in both default and `--features cuda` modes. Logged to `.planning/phases/07-gpu-backend-equivalence-report/deferred-items.md`; not fixed (SCOPE BOUNDARY: pre-existing, unrelated file).

## Verification Results

- `cargo build -p treelite-cubecl` (default, no features) — green.
- `cargo test --workspace` — green, no failures (58 `test result: ok` lines; the new default `device_absent` no-op test runs).
- `cargo clippy -p treelite-cubecl -- -D warnings` (plan's exact command) — clean.
- A3 spike (`--features cuda`) — `cuda_client()` returns `Err(DeviceUnavailable)`, exit 0 (catchable panic, no abort).
- All Task 1 & Task 2 acceptance greps pass; `zenforks` typosquat referenced **0** times; root `Cargo.toml` has **0** `cubecl/{rocm,cuda,wgpu}` references.

## User Setup Required

None — no external service configuration required. The GPU features are opt-in cargo flags; default builds and CI need no GPU system libs.

## Next Phase Readiness

- The feature seam (`rocm`/`cuda`/`wgpu`) and the typed `DeviceUnavailable` skip are in place — Plan 02 can generalize the launcher (`predict::<R, F>`) over these without touching the CPU path.
- **A3 is closed in favor of direct construction:** Plan 03's `*_case()` harness constructors may call `rocm_client()`/`cuda_client()`/`wgpu_client()` directly and treat `DeviceUnavailable` as a skip-not-fail row; no pre-construction device probe is needed.

## Requirement Status Note

This plan's frontmatter lists `requirements: [GPU-03]`, but GPU-03 ("at least one GPU backend is runtime-selectable and **produces predictions**") is a phase-spanning requirement that is NOT satisfied by this Wave-0 seam alone — no GPU prediction runs here (launcher generalization is Plan 02, backend registration is Plan 03). GPU-03 was therefore left **In Progress**, not marked Complete. The orchestrator marks it complete after end-of-phase verification.

## Self-Check: PASSED

All 5 plan-owned files exist on disk; both task commits (`44f256e`, `bf4605f`) are present in git history.

---
*Phase: 07-gpu-backend-equivalence-report*
*Completed: 2026-06-10*

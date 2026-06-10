---
phase: 07-gpu-backend-equivalence-report
plan: 03
subsystem: harness
tags: [harness, gpu, rocm, cuda, wgpu, backend-registration, cubecl, runner-case, features]

# Dependency graph
requires:
  - phase: 07-gpu-backend-equivalence-report
    plan: 01
    provides: "rocm/cuda/wgpu cargo features on treelite-cubecl + CubeclError::DeviceUnavailable + A3 finding (client() failure is catchable, no probe needed)"
  - phase: 07-gpu-backend-equivalence-report
    plan: 02
    provides: "treelite_cubecl::predict::<R, F> runtime-generic launcher (propagates DeviceUnavailable via ?, no silent CPU fallback)"
provides:
  - "Backend::Rocm / Backend::Cuda / Backend::Wgpu harness enum variants (explicit-selection only, D-04)"
  - "rocm_case() / cuda_case() / wgpu_case() RunnerCase constructors routing dense f32/f64 through predict::<HipRuntime|CudaRuntime|WgpuRuntime, _>, sparse through the scalar fallback (D-02), each #[cfg(feature=...)]-gated"
  - "treelite-harness rocm/cuda/wgpu cargo features forwarding to treelite-cubecl + pulling the optional cubecl umbrella crate so the runtime type paths resolve"
  - "ROCm is selectable end-to-end through the harness (GPU-03 registration path complete; hardware execution is the Plan-04 step)"
affects: [07-04, gpu-backend-registration, harness, equivalence-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Registration-not-refactor (D-11): a backend is added as a Backend enum variant + a RunnerCase constructor with ZERO change to the RunnerCase struct, the four slot type aliases, or the matrix iteration — verified by a git-diff invariance check"
    - "GPU *_case() constructors mirror cubecl_cpu_case() EXACTLY: dense slots swap predict_cpu for predict::<R, _>, sparse slots are byte-identical (scalar fallback, D-02)"
    - "Device-absence rides Plan-01's A3 finding: DeviceUnavailable propagates out of predict::<R, _> via ? as a typed skip — NO pre-construction device probe, NO silent CPU fallback (D-05/D-09)"
    - "cubecl is an OPTIONAL harness dependency gated into the rocm/cuda/wgpu features via dep:cubecl + cubecl/<backend>; the default cpu-only build never pulls it (no GPU system libs)"

key-files:
  created: []
  modified:
    - crates/treelite-harness/src/lib.rs
    - crates/treelite-harness/Cargo.toml

key-decisions:
  - "The harness names the runtime types directly as cubecl::hip::HipRuntime / cubecl::cuda::CudaRuntime / cubecl::wgpu::WgpuRuntime (matching the plan key_link pattern), so cubecl must be in the harness's own scope. Added cubecl as an OPTIONAL dependency (inheriting the workspace 0.10.0 spec) and gated it into each GPU feature via dep:cubecl + cubecl/<backend>. The treelite-cubecl/<backend> forward still enables the backend runtime symbols inside the cubecl crate; dep:cubecl brings the crate name into harness scope. A re-export from treelite-cubecl was considered and rejected — it would have changed the call path to treelite_cubecl::HipRuntime and broken the plan's predict::<cubecl::hip::HipRuntime grep contract."
  - "No pre-construction device probe was added (honoring Plan-01's settled A3 finding): predict::<R, _> already propagates CubeclError::DeviceUnavailable via ?, so each *_case() calls predict::<R, _> directly and the harness branches on the Err as a skip-not-fail. T-07-05 (device-construction DoS) is mitigated by this typed-skip propagation, not by a probe."
  - "wgpu f64 caveat (RESEARCH Pitfall 3) documented in wgpu_case's doc-comment: an adapter lacking 64-bit float support makes the dense_f64 slot surface a runtime error rather than a silent downcast; that error propagates through ?, preserving the 1e-5 fidelity contract."

patterns-established:
  - "Pattern: a GPU backend is registered into the harness by (1) a Backend variant, (2) a #[cfg(feature)] *_case() constructor mirroring cubecl_cpu_case, (3) a feature-table row forwarding to treelite-cubecl + dep:cubecl. The RunnerCase seam and matrix iteration are immutable (D-11)."

requirements-completed: []

# Metrics
duration: ~4min
completed: 2026-06-11
---

# Phase 7 Plan 03: GPU Backend Harness Registration (`Backend::Rocm/Cuda/Wgpu` + `*_case()`) Summary

**Registered the three GPU backends into the harness as a thin end-to-end vertical: added `Backend::Rocm`/`Cuda`/`Wgpu` enum variants and `rocm_case()`/`cuda_case()`/`wgpu_case()` `RunnerCase` constructors that route dense f32/f64 through Plan-02's runtime-generic `predict::<R, _>` and keep sparse on the scalar fallback (D-02) — each `#[cfg(feature=...)]`-gated, with the `RunnerCase` seam and matrix iteration untouched (D-11). ROCm is now selectable end-to-end (GPU-03 registration path complete); the default cpu-only build and `cargo test --workspace` stay green.**

## Performance

- **Duration:** ~4 min
- **Tasks:** 2
- **Files modified:** 2 (0 created, 2 modified)

## Accomplishments

- **Task 1 — feature plumbing + enum variants (commit `19196d8`).** Added a `[features]` table to `crates/treelite-harness/Cargo.toml` (`default = []`, `rocm`/`cuda`/`wgpu` forwarding to `treelite-cubecl/<backend>`) and the `Rocm`/`Cuda`/`Wgpu` `Backend` enum variants (replacing the Phase-7 reservation comment), each documented as explicit-selection-only (no auto-detect, D-04). Default build green; `cargo test --workspace` green with no GPU libs.
- **Task 2 — the three `*_case()` constructors (commit `53651ea`).** Added `#[cfg(feature = "rocm")] rocm_case()`, `#[cfg(feature = "cuda")] cuda_case()`, `#[cfg(feature = "wgpu")] wgpu_case()`, each modeled exactly on `cubecl_cpu_case()`: dense slots call `treelite_cubecl::predict::<cubecl::hip::HipRuntime|CudaRuntime|WgpuRuntime, f32/f64>` (f32 result widened to f64 AFTER predict — no pre-cast, Pitfall 6), sparse slots are byte-identical to the scalar fallback (D-02). Wired `cubecl` as an optional harness dependency gated into the GPU features so the runtime type paths resolve, while the default build stays cpu-only.
- **Verified all three GPU feature builds compile** (`--features rocm`, `--features cuda`, `--features wgpu`) on this box, and the default cpu-only path + full workspace suite stay green — the matrix-runner seam (`RunnerCase` struct, slot type aliases, iteration) is byte-unchanged (D-11 registration-not-refactor).

## Task Commits

1. **Task 1: harness rocm/cuda/wgpu features + Backend::Rocm/Cuda/Wgpu variants** — `19196d8` (feat)
2. **Task 2: rocm_case/cuda_case/wgpu_case RunnerCase constructors** — `53651ea` (feat)

**Plan metadata:** committed separately with SUMMARY/STATE/ROADMAP.

## Files Modified

- `crates/treelite-harness/Cargo.toml` — added the `[features]` table (`default = []`; `rocm`/`cuda`/`wgpu` → `dep:cubecl` + `cubecl/<backend>` + `treelite-cubecl/<backend>`); added `cubecl` as an optional workspace dependency.
- `crates/treelite-harness/src/lib.rs` — added the `Rocm`/`Cuda`/`Wgpu` `Backend` variants (with explicit-selection doc-comments) and the three `#[cfg(feature)]`-gated `*_case()` `RunnerCase` constructors routing dense → `predict::<R, _>` and sparse → scalar fallback.

## Decisions Made

- **Runtime type path = direct `cubecl::<backend>::*Runtime`** — the plan's `key_link` pattern (`predict::<cubecl::hip::HipRuntime`) requires the harness to name the cubecl runtime types directly, so `cubecl` had to be in the harness's own scope. Added `cubecl` as an OPTIONAL dependency (inheriting the workspace `0.10.0` spec) gated into each GPU feature via `dep:cubecl` + `cubecl/<backend>`. A `pub use` re-export from `treelite-cubecl` (which would have made the path `treelite_cubecl::HipRuntime`) was considered and rejected to preserve the plan's grep contract. See Deviations (Rule 3 blocking fix).
- **No pre-construction device probe** — honoring Plan-01's settled A3 finding (a missing device is a catchable error mapped to `CubeclError::DeviceUnavailable`, not an FFI abort). `predict::<R, _>` already propagates that error via `?`, so each `*_case()` calls `predict::<R, _>` directly; the harness treats the `Err` as a skip-not-fail row. T-07-05 (device-construction DoS) is mitigated by this typed-skip propagation.
- **wgpu f64 caveat documented** (RESEARCH Pitfall 3) — an adapter lacking 64-bit float support makes `dense_f64`'s `predict::<WgpuRuntime, f64>` surface a runtime error rather than a silent downcast; that error propagates through `?`, preserving the 1e-5 fidelity contract.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `cubecl` was not in the harness's crate scope — the `*_case()` runtime paths would not resolve**
- **Found during:** Task 2 (first `cargo build -p treelite-harness --features rocm`).
- **Issue:** The plan's `*_case()` bodies name `cubecl::hip::HipRuntime` (etc.) directly, but `treelite-harness` only depends on `treelite-cubecl`, which does NOT re-export the `cubecl` umbrella crate. The feature forwarding `treelite-cubecl/rocm` enables the backend runtime symbols *inside the cubecl crate* but does not add a `cubecl` dependency edge to the harness, so the build failed with `E0433: cannot find module or crate cubecl` on the `predict::<cubecl::hip::HipRuntime, _>` lines.
- **Fix:** Added `cubecl = { workspace = true, optional = true }` to the harness `[dependencies]` and gated it into each GPU feature (`rocm = ["dep:cubecl", "cubecl/rocm", "treelite-cubecl/rocm"]`, same for cuda/wgpu). This brings the `cubecl` crate name into the harness's scope ONLY under the GPU features; the default cpu-only build does not pull it in (verified). This is the lowest-deviation fix consistent with the plan's `predict::<cubecl::hip::HipRuntime` grep contract — the alternative (re-exporting the runtime types from `treelite-cubecl`) would have changed the call path and broken that contract.
- **Files modified:** `crates/treelite-harness/Cargo.toml` (the feature-table rows + the optional `cubecl` dep).
- **Verification:** `cargo build -p treelite-harness --features rocm|cuda|wgpu` all compile; default `cargo build -p treelite-harness` and `cargo test --workspace` stay green.
- **Committed in:** `53651ea` (Task 2 commit).

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** None on scope or acceptance — the fix is the minimal dependency wiring needed to make the plan's own `predict::<cubecl::hip::HipRuntime, _>` call paths resolve. All acceptance greps pass verbatim.

## Verification Results

- `cargo build -p treelite-harness` (default, cpu-only) — green; `cubecl` NOT pulled in as a direct dependency.
- `cargo test --workspace` (default cpu-only) — fully green, **zero failures** (45 `test result: … 0 failed` lines); the `Backend` enum widening breaks no exhaustive match.
- `cargo build -p treelite-harness --features rocm` — green (the `rocm_case` path compiles; `cubecl::hip::HipRuntime` resolves).
- `cargo build -p treelite-harness --features cuda` — green (the `cuda_case` path compiles).
- `cargo build -p treelite-harness --features wgpu` — green (the `wgpu_case` path compiles).
- `cargo clippy -p treelite-harness` (default) — clean.
- **Task 1 acceptance greps:** `rocm = ["treelite-cubecl/rocm"]` = 1 (plus the `dep:cubecl` forwarding), `cuda`/`wgpu` likewise; `Rocm|Cuda|Wgpu` in lib.rs = 6 (>= 3).
- **Task 2 acceptance greps:** `pub fn rocm_case|cuda_case|wgpu_case` = 3; `predict::<cubecl::hip::HipRuntime` = 3 (>= 1).
- **Registration-not-refactor (D-11):** `git diff` shows the `RunnerCase` struct, the four slot type aliases, and the matrix iteration UNCHANGED (verified — no `pub struct RunnerCase` / `pub type *PredictF*Fn` / slot-field lines in the additive diff).

## Threat Model Compliance

- **T-07-05 (device-construction DoS on `cuda_case`/`wgpu_case` on the NVIDIA-less box)** — mitigated: per Plan-01's settled A3 finding, GPU `client()` failure is a catchable error mapped to `CubeclError::DeviceUnavailable`, propagated out of `predict::<R, _>` via `?`. No `*_case()` auto-invokes an unavailable backend in a way that aborts the process; the harness branches on the typed `Err` as a skip. No pre-construction probe was required.
- **T-07-06 (provenance: selected backend == backend that ran)** — mitigated: each `*_case()` sets `backend: Backend::<X>` literally and routes dense cells ONLY to that runtime's `predict::<R, _>`; there is no fallback re-routing on device absence (the only scalar route is the pre-existing D-02 sparse/categorical fallback). D-04/D-05/D-09 preserved.
- **T-07-SC (cargo installs / supply-chain)** — mitigated: NO new crates were added. `cubecl` is the existing workspace `0.10.0` dependency, now declared optional on the harness and gated into the GPU features; the harness's GPU features forward to the existing Plan-01 `treelite-cubecl` features. The RESEARCH package-legitimacy posture (no HIP typosquat fork) holds unchanged.

## Requirement Status Note

The plan frontmatter lists `requirements: [GPU-03]`. GPU-03 ("at least one GPU backend is runtime-selectable AND **produces predictions**") is phase-spanning: this plan completes the *registration* path (ROCm is now selectable end-to-end through the harness), but no GPU prediction runs here — actual on-hardware execution + the 1e-5 equivalence assertion is the Plan-04 hardware-gated step. GPU-03 is therefore left **In Progress**; `requirements-completed` is empty. The orchestrator marks GPU-03 complete after end-of-phase verification.

## Next Phase Readiness

- `Backend::Rocm`/`Cuda`/`Wgpu` + `rocm_case()`/`cuda_case()`/`wgpu_case()` are in place and compile under their respective features. Plan 04 can add the GPU rows to the equivalence matrix on the developer's ROCm hardware, executing `rocm_case()` to produce real predictions and asserting them within 1e-5 — treating `DeviceUnavailable` as a skip-not-fail row on absent backends.
- The CPU 1e-5 gate is untouched (the matrix seam is byte-identical), so Plan 04 adds GPU rows without disturbing the proven CPU baseline.

## Self-Check: PASSED

- `crates/treelite-harness/src/lib.rs` exists on disk (modified).
- `crates/treelite-harness/Cargo.toml` exists on disk (modified).
- Both task commits present in git history: `19196d8` (Task 1), `53651ea` (Task 2).

---
*Phase: 07-gpu-backend-equivalence-report*
*Completed: 2026-06-11*

# Phase 7: GPU Backend & Equivalence Report - Research

**Researched:** 2026-06-11
**Domain:** cubecl GPU runtime selection (ROCm/CUDA/wgpu) + GPU floating-point divergence profiling + harness-emitted equivalence reporting
**Confidence:** HIGH on the host-launcher generalization and crate/feature plumbing (verified against in-repo code + crates.io + the cubecl manual); MEDIUM on the predicted-deviation magnitudes (verified against published GPU-libm ULP studies, but the committed ROCm numbers will be the ground truth this phase produces).

## Summary

Phase 7 is a **host-launcher generalization + backend-registration + reporting** phase, not a kernel phase. The cubecl kernels (`descend`, `predict_default_raw`, `predict_leaf_id`, `predict_score_per_tree`, the 10 postproc helpers) are ALREADY generic over `R: Runtime` — they are invoked as `launch::<F, T, R>` and the only `R` instantiated today is `CpuRuntime`. `UploadedForest<R>` and `upload_forest::<R, F>` are likewise already generic over `R`. The entire concrete-`CpuRuntime` surface is confined to `crates/treelite-cubecl/src/lib.rs`: every launcher fn hardcodes `client: &ComputeClient<CpuRuntime>` and constructs the client via `CpuRuntime::client(&Default::default())`, and the four `launch::<F, T, CpuRuntime>` call sites name the runtime. Generalizing those functions over `R: Runtime` and constructing the per-backend client is the central code change. The ROCm manual example confirms the construction pattern is identical across runtimes (`R::client(&device)`), so this is a mechanical lift, not a redesign.

The genuinely-unknown axis (D-03) is **GPU floating-point divergence**, and the research narrows it sharply: the accumulation kernel (`default_raw.rs`) does pure sequential `output[cell] += v` adds with NO `a*b+c` pattern, so **FMA contraction has essentially no surface on the tree-sum/base-score path** — IEEE-754 add/sub/mul/div are correctly-rounded on every cubecl backend (OpenCL/SPIR-V/HIP/CUDA all guarantee this). The divergence is therefore dominated by **GPU transcendentals** (`exp`/`exp2`/`ln_1p`) inside the postprocessors (`sigmoid`, `softmax`, `exponential`, `exponential_standard_ratio`, `logarithm_one_plus_exp`, `multiclass_ova`). Published exhaustive single-precision studies put GPU `exp`/`log` at **1-2 ULP** vs correctly-rounded on HIP/CUDA, and the OpenCL spec caps `exp`/`exp2` at **≤3-4 ULP**. The predicted per-class deviation model (below) keys on whether a model class touches a transcendental postprocessor and at what input precision.

**Primary recommendation:** Enable the `cubecl` umbrella crate's `hip`/`rocm`, `cuda`, and `wgpu` cargo features (NOT separate `cubecl-hip`/`cubecl-cuda`/`cubecl-wgpu` deps) behind additive `treelite-cubecl` features `rocm`/`cuda`/`wgpu`; make every `predict_cpu*`/`launch_*`/`run_*` fn generic over `R: Runtime` with a per-backend `device_and_client::<R>()` constructor that returns a typed `CubeclError::DeviceUnavailable` instead of panicking; register `Backend::Rocm`/`Cuda`/`Wgpu` + `rocm_case()`/`cuda_case()`/`wgpu_case()` into the harness; emit an observational markdown report (model class × per-backend max-deviation × f64-fallback-used × predicted-deviation) from the existing per-cell provenance machinery, committed and regenerated only on the developer's ROCm hardware. State honestly in the report that bit-identical determinism is NOT claimable on GPU.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: The GPU equivalence report is observational — measured, not gated.** Report the measured max deviation per model class on ROCm as DATA; no pass/fail tolerance threshold fails CI. Mirrors the Phase-6 per-cell-provenance honesty ethos. The 1e-5 CPU spine (ScalarCpu / CubeclCpu) remains the hard gate and is untouched; the GPU report sits beside it, not in place of it.
- **D-02: f64-postprocessor fallback is per-class and measured.** Run f32 postprocessors first on GPU; if a specific model class exceeds the budget, promote ONLY that class's postprocessors to f64 and record the promotion in the report. Minimal f64, documented exactly where needed. Traversal precision follows the Phase-6 in-kernel contract; this decision is about the postprocessor stage.
- **D-03: Deviation profiling is research-led, validated empirically.** Research profiles cubecl FMA contraction behavior + GPU transcendental (exp/exp2/sigmoid/softmax) divergence, producing a predicted-deviation model (expected per-class deviation range). The committed report's measured ROCm numbers validate against that predicted model — report carries both columns. DIFFERENT axis from Phase-6 D-04 (in-kernel CPU-runtime precision is settled); GPU hardware transcendental/FMA divergence is the genuinely-unknown thing. Do not conflate or re-litigate D-04.
- **D-04: Per-backend additive Cargo feature + explicit enum selection, no auto-detect.** One feature each (`rocm`, `cuda`, `wgpu`); default build = cpu only. The caller explicitly selects `Backend::Rocm`/`Cuda`/`Wgpu` — no "best available" resolver. Selection is deterministic, matches the existing `Backend`-enum registration pattern. Selected backend always == the backend that ran.
- **D-05: Device-absent is a skip, not a failure — surfaced as a typed result.** When a backend is compiled in but no device is present at runtime, selecting it returns a typed "no device" result the caller can branch on (thiserror variant, e.g. `DeviceUnavailable`); the harness/report treats that as a skip and marks the row "not run — no device." No silent fallback to CPU.
- **D-06: The report is harness-emitted and committed, regenerated on ROCm hardware.** The matrix harness emits the report (markdown + table) from the same per-cell manifest run that drives equivalence — extending the Phase-6 manifest/provenance machinery. The committed file is regenerated by running the harness on the developer's ROCm hardware, so the numbers can never drift from hand-editing.
- **D-07: Report rows = the existing equivalence-harness fixture model set.** No new taxonomy — rows mirror the XGBoost/LightGBM/sklearn model classes already in the Phase-5/6 frozen golden matrix, the same classes already validated to 1e-5 on CPU.
- **D-08: Committed markdown report with per-backend columns.** Lives as a committed markdown file (location is Claude's discretion). Columns: model class | ROCm max-deviation | f64-fallback used? | CUDA | wgpu | predicted-deviation (from D-03). CUDA and wgpu cells render "not run — no device" on this hardware (per D-05) unless hardware exposes them. A machine-readable sidecar (JSON/CSV) is a reasonable planner refinement, not mandated.
- **D-09: The crossover is documented-only — not enforced in code.** Measure and DOCUMENT the crossover point; the caller (notably the Phase-8 PyO3 layer) decides routing. `predict()` does NOT implicitly re-route a GPU-selected call to CPU below the threshold. An opt-in helper applying the documented heuristic is acceptable if it stays opt-in.
- **D-10: The crossover metric is measured, not pre-committed.** Benchmark on ROCm hardware and let the data pick the dominant metric (likely row count or rows×features); document the empirical crossover and which metric it keys on. Don't hardcode a formula up front.

### Claude's Discretion

- Concrete cubecl GPU runtime crates backing `Backend::Rocm`/`Cuda`/`Wgpu` — confirm exact crate names, versions, feature flags; pin to latest published. **(RESOLVED below — Standard Stack.)**
- Host-launcher generalization mechanics — exactly how `predict_cpu`/`predict_cpu_sparse` and friends generalize from `CpuRuntime` to `R: Runtime`, including where `ComputeClient<R>` is constructed per backend. **(RESOLVED below — Architecture Patterns.)**
- Report file location & exact markdown layout (D-08), and whether to add the machine-readable sidecar (D-06/D-08). **(RECOMMENDED below — Report Emission.)**
- Whether determinism (SC2-style bit-identical) is even claimable on GPU — the report should state it honestly. **(RESOLVED below — Determinism on GPU.)**
- wgpu's role — shares the seam, likely runs on the same AMD device via Vulkan; whether it's a validated or "not run" row depends on hardware. Determine empirically. **(GUIDANCE below — Open Questions.)**

### Deferred Ideas (OUT OF SCOPE)

- Autotuned / optimized GPU kernels — v2 (PERF-v2-02).
- Metal / Vulkan backends beyond wgpu — v2 (PERF-v2-02). wgpu rides the seam this phase.
- CUDA hardware validation — blocked on NVIDIA hardware (none locally). CUDA is build-supported; rows fill where such a device exists.
- Sparse CSR + categorical splits in GPU kernels — still ride the scalar fallback (Phase-6 D-02 deferral, unchanged).
- f16/bf16 half-precision GPU fast path — Phase 9 / v2 (PERF-v2-01).
- Enforced/auto-routing crossover in `predict()` — explicitly NOT done (D-09 documented-only).
- Machine-readable report sidecar — considered (D-08); planner discretion, not mandated.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPU-03 | At least one GPU backend (ROCm, wgpu, or CUDA) is runtime-selectable and produces predictions. ROCm is the hardware-validated backend; CUDA build-supported but validated only where NVIDIA exists. | Standard Stack confirms `cubecl` `hip`/`rocm` feature → `cubecl::hip::HipRuntime`/`HipDevice`; Architecture Patterns shows the `predict_cpu`→`predict::<R>` generalization and the `Backend::Rocm` + `rocm_case()` registration; Device-Absent section gives the typed-skip path for CUDA-no-device. |
| GPU-04 | A GPU equivalence report documents observed deviation per model class within an accepted tolerance. | Predicted-Deviation Model gives the per-class expected ranges + citations; Report Emission gives the markdown layout emitted from `manifest.rs` provenance; D-01/D-02/D-06 honesty constraints encoded in Validation Architecture. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GPU backend selection (`Backend::Rocm`/`Cuda`/`Wgpu`) | `treelite-harness` (Backend enum + RunnerCase) | `treelite-cubecl` (generic launcher) | Selection is a harness/registration concern (Phase-5 D-11 seam); the launcher just receives `R`. |
| `R: Runtime` host launcher | `treelite-cubecl/src/lib.rs` | `treelite-cubecl/src/upload.rs` (already generic) | The concrete-runtime surface is confined to lib.rs; upload + kernels are already `R`-generic. |
| `ComputeClient<R>` construction + device-availability | `treelite-cubecl` (new `device.rs` or in lib.rs) | `treelite-cubecl/src/error.rs` (typed `DeviceUnavailable`) | Per-backend client init lives at the launcher boundary; absence is a typed error, not a panic (D-05). |
| Feature gating (`rocm`/`cuda`/`wgpu`) | `treelite-cubecl/Cargo.toml` + root `Cargo.toml` | — | Additive cubecl umbrella features; default = cpu only (D-04). |
| Deviation report emission | `treelite-harness` (new report module + matrix sibling) | `treelite-harness/src/manifest.rs` (provenance source) | The report is harness-emitted from per-cell provenance (D-06); regenerated on ROCm hardware. |
| GPU predictions (the kernels) | `treelite-cubecl/src/kernels/*` (UNCHANGED) | — | Already `launch::<F, T, R>`; rewriting them is a smell. |
| Postprocessor f64 promotion (per-class) | `treelite-cubecl/src/lib.rs` `PredictCpuElem::apply_postprocessor` (host-side, already f32/f64 split) + `treelite-gtil::postprocessor` f64 twins | — | The f64 twins already exist (Phase 5 CR-01); promotion is a per-class dispatch decision, not new math. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | 0.10.0 | The umbrella crate already in `[workspace.dependencies]`; add `hip`/`rocm`, `cuda`, `wgpu` features | `[VERIFIED: crates.io]` newest 0.10.0, matches the pinned root version. The umbrella re-exports `cubecl::hip`, `cubecl::cuda`, `cubecl::wgpu` runtime modules when their features are enabled — no separate runtime-crate deps needed `[CITED: lib.rs/crates/cubecl/features]`. |

### Feature flags (verified against the cubecl features page)

`[CITED: lib.rs/crates/cubecl/features]` (cubecl 0.10.0):

| cubecl feature | Enables runtime crate | Re-exported module | System requirement |
|----------------|----------------------|--------------------|--------------------|
| `cpu` (already on) | `cubecl-cpu` 0.10.0 | `cubecl::cpu` (`CpuRuntime`, `CpuDevice`) | none |
| `hip` | `cubecl-hip` 0.10.0 | `cubecl::hip` (`HipRuntime`, `HipDevice`) | ROCm/HIP system libs (`cubecl-hip-sys` 7.2.x binds them) |
| `rocm` | implies `hip` | `cubecl::hip` | ROCm stack |
| `cuda` | `cubecl-cuda` 0.10.0 | `cubecl::cuda` (`CudaRuntime`, `CudaDevice`) | CUDA toolkit + NVIDIA driver |
| `wgpu` | `cubecl-wgpu` 0.10.0 | `cubecl::wgpu` (`WgpuRuntime`, `WgpuDevice`) | Vulkan/Metal/DX12 driver (Vulkan on the AMD box) |
| `vulkan` | implies `wgpu-spirv` | (wgpu via SPIR-V) | Vulkan driver |

**Recommended mapping** (matches the CONTEXT.md feature names `rocm`/`cuda`/`wgpu`):
- `treelite-cubecl` feature `rocm` → `cubecl/rocm` (which implies `hip`) → `cubecl::hip::HipRuntime`. `[CITED: cubecl manual Handling_Interleaved_Complex_Numbers_in_CubeCL_with_ROCm_Backend.md — uses `cubecl::hip::HipRuntime` / `cubecl::hip::HipDevice::default()` / `BackendRuntime::client(&device)`]`
- `treelite-cubecl` feature `cuda` → `cubecl/cuda` → `cubecl::cuda::CudaRuntime`. `[ASSUMED: module path inferred from the symmetric `cubecl::cpu`/`cubecl::hip` re-export pattern; verify `CudaRuntime`/`CudaDevice` symbol names against `cargo doc` once the feature is enabled.]`
- `treelite-cubecl` feature `wgpu` → `cubecl/wgpu` → `cubecl::wgpu::WgpuRuntime`. `[VERIFIED: cubecl manual cubecl_matmul_gemm_example.md uses `cubecl_wgpu::{WgpuRuntime, WgpuDevice, init_setup, init_device}` and `WgpuDevice::DefaultDevice`]`

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `cubecl` umbrella features (`cubecl/rocm` etc.) | Separate `cubecl-hip`/`cubecl-cuda`/`cubecl-wgpu` 0.10.0 deps | The umbrella already pins 0.10.0 and re-exports the runtime modules; adding separate deps risks a version-skew across the family and duplicates what the umbrella feature already does. Prefer features on the single umbrella crate. |
| `cubecl::hip::HipDevice::default()` | explicit device index enumeration | `default()` is the documented ROCm-manual pattern and sufficient for one-device dev box; enumeration is a v2 concern. |

**Installation (root `Cargo.toml`, line ~25):**
```toml
cubecl = { version = "0.10.0", features = ["cpu"] }   # default build unchanged
# GPU runtimes are NOT enabled at the workspace level; treelite-cubecl gates
# them behind its own optional features so default builds stay cpu-only (D-04).
```
**`treelite-cubecl/Cargo.toml`:**
```toml
[features]
default = []
rocm = ["cubecl/rocm"]   # implies cubecl/hip → cubecl::hip
cuda = ["cubecl/cuda"]
wgpu = ["cubecl/wgpu"]
```
(With `cubecl = { workspace = true }`, feature-forwarding `cubecl/rocm` from the crate-local feature is the idiomatic way to keep the workspace default cpu-only while letting `cargo test -p treelite-cubecl --features rocm` pull the HIP runtime.)

**Version verification performed:**
```
cargo search cubecl       → 0.10.0   [VERIFIED]
cargo search cubecl-hip   → 0.10.0   [VERIFIED]  (created 2024-10-28, 663k dl, tracel-ai monorepo)
cargo search cubecl-cuda  → 0.10.0   [VERIFIED]  (created 2024-07-19, 723k dl, tracel-ai monorepo)
cargo search cubecl-wgpu  → 0.10.0   [VERIFIED]  (created 2024-07-19, 732k dl, tracel-ai monorepo)
cargo search cubecl-cpu   → 0.10.0   [VERIFIED]  (created 2025-10-24, 230k dl — matches the 0.10 CPU-runtime addition)
```

## Package Legitimacy Audit

> All recommended packages are the official `cubecl` family from the `tracel-ai/cubecl` monorepo. slopcheck's `install` subcommand would actually install, so it was NOT run (avoiding an unwanted install); legitimacy was established via `cargo search` + crates.io metadata (registry, repo URL, download counts, age) instead.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `cubecl` 0.10.0 | crates.io | already in tree (Phase 6) | — | github.com/tracel-ai/cubecl | not run (registry+repo verified) | Approved (already a dep) |
| `cubecl-hip` 0.10.0 (via `cubecl/rocm`) | crates.io | since 2024-10 | 663k total | github.com/tracel-ai/cubecl/tree/main/crates/cubecl-hip | not run (registry+repo verified) | Approved |
| `cubecl-cuda` 0.10.0 (via `cubecl/cuda`) | crates.io | since 2024-07 | 723k total | github.com/tracel-ai/cubecl/tree/main/crates/cubecl-cuda | not run (registry+repo verified) | Approved |
| `cubecl-wgpu` 0.10.0 (via `cubecl/wgpu`) | crates.io | since 2024-07 | 732k total | github.com/tracel-ai/cubecl/tree/main/crates/cubecl-wgpu | not run (registry+repo verified) | Approved |
| `zenforks-cubecl-hip` 0.10.1 | crates.io | — | — | (third-party fork) | — | **REMOVED — do NOT use.** A `cargo search` near-name fork of `cubecl-hip`; it is NOT the official tracel-ai crate. Use `cubecl/rocm` only. |

**Packages removed due to a slop/typosquat verdict:** `zenforks-cubecl-hip` (a non-official near-name fork surfaced by `cargo search`; the planner must ensure no plan references it).
**Packages flagged as suspicious [SUS]:** none.

*slopcheck was available (0.6.1) but its only subcommands install packages; legitimacy here was established via the safer `cargo search` + crates.io-metadata path. The planner should treat the four official crates as VERIFIED and gate only the `zenforks-*` exclusion.*

## Architecture Patterns

### System Architecture Diagram

```
                    caller selects Backend::{Rocm|Cuda|Wgpu}  (explicit, no auto-detect — D-04)
                                          │
                                          ▼
        treelite-harness  ──  rocm_case()/cuda_case()/wgpu_case()  (RunnerCase: 4 fn ptrs)
                                          │
                          dense f32/f64 ──┤── sparse f32/f64 (→ scalar fallback, D-02)
                                          ▼
        treelite-cubecl::predict::<R, F>   (was predict_cpu::<F>, now R-generic)
                                          │
                    ┌─────────────────────┼──────────────────────────┐
                    ▼                     ▼                          ▼
        device_and_client::<R>()   model_routes_to_scalar_fallback   (categorical / non-kLT → scalar)
            │  (Ok(client)                │                          │
            │   or Err(DeviceUnavailable))│  (no silent CPU fallback — D-05)
            ▼                             ▼
        ComputeClient<R>          upload_forest::<R, F>  (ALREADY generic; SoA as_bytes zero-copy)
            │                             │
            └────────────┬────────────────┘
                         ▼
        kernels::*::launch::<F, T, R>   (ALREADY generic; UNCHANGED)
                         │
                         ▼  client.read_one_unchecked → bytemuck::cast_slice → Vec<F>
                         ▼
        F::apply_postprocessor (Default kind only; f32 first, per-class f64 promotion — D-02)
                         │
                         ▼
        per-cell Manifest/provenance ── max |delta| vs frozen 1e-5 golden ── report row (D-06/D-08)
```

### Recommended Project Structure (additive only)
```
crates/treelite-cubecl/src/
├── lib.rs        # GENERALIZE: predict_cpu → predict::<R>, launch_*/run_* take ComputeClient<R>
├── device.rs     # NEW: device_and_client::<R>() per-backend client init + availability probe (D-05)
├── error.rs      # ADD: CubeclError::DeviceUnavailable variant
├── upload.rs     # UNCHANGED (already generic over R)
└── kernels/*     # UNCHANGED (launch::<F, T, R> already generic)
crates/treelite-harness/
├── src/lib.rs    # ADD: Backend::Rocm/Cuda/Wgpu + rocm_case()/cuda_case()/wgpu_case()
├── src/report.rs # NEW: emit the markdown deviation report from per-cell provenance (D-06/D-08)
└── tests/gtil_matrix_gpu.rs  # NEW sibling: drive the matrix on the selected GPU backend (D-06)
docs/ or .planning/phases/07-.../
└── GPU_EQUIVALENCE_REPORT.md  # COMMITTED report, regenerated on ROCm hardware (D-08)
```

### Pattern 1: Generalize the host launcher from `CpuRuntime` to `R: Runtime`
**What:** Every fn in `lib.rs` that today names `CpuRuntime` becomes generic over `R: Runtime`. The kernels are already `launch::<F, T, R>`; only the launcher's client type and the four call-site type args change.
**When to use:** The entire `predict_cpu` family.
**Example (mechanics):**
```rust
// Source: derived from crates/treelite-cubecl/src/lib.rs (CpuRuntime sites) +
// cubecl manual Handling_Interleaved_Complex_Numbers_…_ROCm_Backend.md (R::client pattern)

// BEFORE (lib.rs:312):  let client = CpuRuntime::client(&Default::default());
// BEFORE (lib.rs:444):  kernels::default_raw::predict_default_raw::launch::<F, T, CpuRuntime>(...)

// AFTER — a per-backend client constructor that is a TYPED skip on no device (D-05):
pub fn predict<R: Runtime, F: PredictCpuElem>(
    model: &Model, data: &[F], num_row: usize, cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    if model_routes_to_scalar_fallback(model) {
        return treelite_gtil::predict::<F>(model, data, num_row, cfg)
            .map_err(|e| CubeclError::Unsupported(format!("scalar fallback: {e}")));
    }
    let client = device_and_client::<R>()?;   // Err(DeviceUnavailable) on no device — NO CPU fallback
    match cfg.kind {
        PredictKind::Default | PredictKind::Raw =>
            launch_default_raw::<R, F>(&client, model, data, num_row, cfg),
        PredictKind::LeafId       => launch_leaf_id::<R, F>(&client, model, data, num_row),
        PredictKind::ScorePerTree => launch_score_per_tree::<R, F>(&client, model, data, num_row),
    }
}

// The launch site becomes runtime-generic; everything else in run_default_raw is identical:
kernels::default_raw::predict_default_raw::launch::<F, T, R>(&client, cube_count, cube_dim, /* … */);
```
**Note:** Keep a thin `predict_cpu::<F>(…) = predict::<CpuRuntime, F>(…)` shim so `cubecl_cpu_case()` and the existing `gtil_matrix_cubecl.rs` test compile UNCHANGED (registration-not-refactor; `git diff` on the CPU path stays minimal).

### Pattern 2: Per-backend client construction with typed device-absence (D-05)
**What:** A single generic constructor that maps "no device" to `CubeclError::DeviceUnavailable` rather than a panic. cubecl client construction is `R::client(&device)`; device handles are `<Backend>Device::default()`.
**When to use:** Once, in `device.rs`; called by `predict::<R>`.
**Example:**
```rust
// Source: cubecl manual (ROCm example: HipRuntime::client(&HipDevice::default());
//         GEMM example: WgpuRuntime via init_setup/init_device then ComputeClient::load)
// The generic surface is `R::client(&R::Device::default())`. The "is a device present"
// probe is best done by attempting client construction and catching the failure — cubecl
// does not expose a stable pre-construction `is_available()` across all three runtimes in
// 0.10. [ASSUMED — verify against cargo doc for cubecl 0.10; if a Runtime::can_run / device
// enumeration API exists, prefer it over catch-on-construct.]

#[cfg(feature = "rocm")]
pub fn rocm_client() -> Result<cubecl::client::ComputeClient<cubecl::hip::HipRuntime>, CubeclError> {
    std::panic::catch_unwind(|| {
        let device = cubecl::hip::HipDevice::default();
        cubecl::hip::HipRuntime::client(&device)
    })
    .map_err(|_| CubeclError::DeviceUnavailable { backend: "rocm" })
}
```
**Caveat (verify in planning):** Whether cubecl's HIP/CUDA `client()` returns gracefully or panics/aborts on a missing device is the single most important unknown for D-05. The plan MUST include a spike task that runs `cuda_case()` on this NVIDIA-less box and confirms it yields a typed skip (not an abort). If `client()` aborts the process (a hard FFI abort, not a Rust panic `catch_unwind` can trap), the fallback is to probe device availability BEFORE construction via whatever enumeration API cubecl 0.10 exposes (e.g. a runtime `device_count`/`can_run` — confirm via `cargo doc`), or to guard the CUDA path so it is never auto-invoked on this hardware (the harness simply emits "not run — no device" for cuda/wgpu rows without attempting construction).

### Pattern 3: Additive backend registration in the harness (Phase-5 D-11 seam)
**What:** Add `Backend::Rocm`/`Cuda`/`Wgpu` variants and `rocm_case()`/`cuda_case()`/`wgpu_case()` constructors. Each `*_case()` mirrors `cubecl_cpu_case()` exactly, routing dense slots through `treelite_cubecl::predict::<HipRuntime, _>` and keeping the sparse slots on the scalar fallback (D-02). The matrix iteration is untouched.
**Example:**
```rust
// Source: crates/treelite-harness/src/lib.rs cubecl_cpu_case() (the template)
#[cfg(feature = "rocm")]
pub fn rocm_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::Rocm,
        dense_f32: |m, d, n, c| {
            let out = treelite_cubecl::predict::<cubecl::hip::HipRuntime, f32>(m, d, n, c)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |m, d, n, c|
            treelite_cubecl::predict::<cubecl::hip::HipRuntime, f64>(m, d, n, c)
                .map_err(|e| anyhow::anyhow!("{e}")),
        // sparse rides the scalar fallback (D-02), same as cubecl_cpu_case
        sparse_f32: |m, csr, n, c| { /* treelite_gtil::predict_sparse::<f32> … */ },
        sparse_f64: |m, csr, n, c| { /* treelite_gtil::predict_sparse::<f64> … */ },
    }
}
```

### Pattern 4: Zero-copy SoA upload carries to GPU unchanged
**What:** `upload_forest::<R, F>` and `UploadedForest<R>` are already generic over `R` and use `client.create_from_slice(bytemuck::cast_slice(&col))`. `create_from_slice` exists on the generic `ComputeClient<R>`, so the SoA `as_bytes` upload (GPU-05) carries to HIP/CUDA/wgpu with no code change.
**Flag for planning (alignment/size):** GPU buffers have stricter alignment and max-size limits than the CPU runtime. The forest columns are `i32`/`u32`/`f32`/`f64` slices (all naturally aligned) — bytemuck guarantees the cast alignment. The realistic risk is wgpu's **per-binding buffer size limit** and required **4-byte alignment for storage buffers**, plus wgpu's historical lack of native `f64` in WGSL. See Pitfall 3.

### Anti-Patterns to Avoid
- **Rewriting kernels.** The `launch::<F, T, R>` sites and `descend`/`predict_default_raw` are already runtime-generic. If a plan touches `kernels/*.rs` for backend support, that is the D-03/CONTEXT "smell."
- **Silent CPU fallback when a GPU backend is selected.** D-05/D-09 forbid it — a selected backend that has no device returns `DeviceUnavailable`; it does NOT quietly run on CPU. This preserves "selected backend == backend that ran."
- **Auto-detect / "best available" backend resolver.** D-04 forbids it — selection is explicit and literal.
- **Gating CI on the GPU report.** D-01 — the report is observational; the 1e-5 hard gate stays on ScalarCpu/CubeclCpu on CPU only.
- **Hand-editing the committed report.** D-06 — it is regenerated by running the harness on ROCm hardware.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-backend client construction | A custom `match Backend { … }` device factory with raw FFI | `R::client(&R::Device::default())` via the cubecl runtime trait | cubecl already abstracts device/client init uniformly across HIP/CUDA/wgpu (ROCm-manual pattern). |
| Host→device upload for GPU | A new GPU-specific uploader | The existing `upload_forest::<R, F>` (already generic) | It is already `R`-generic and zero-copy; GPU is just a different `R`. |
| f64 postprocessor math | New GPU f64 postprocessor kernels | The existing `treelite_gtil::postprocessor` f64 twins via `PredictCpuElem::apply_postprocessor` (host-side, post-readback) | Postprocessors run host-side after readback today; the f64 twins (Phase-5 CR-01) already exist. Per-class f64 promotion (D-02) is a dispatch choice, not new math. |
| GPU vs CPU deviation tolerance | A bespoke epsilon-comparison harness | The existing `assert_within` + per-cell max-`|delta|` accumulation in `gtil_matrix_cubecl.rs` | The matrix sibling already computes per-cell max deviation; the GPU sibling reuses it and emits a report row instead of (or alongside) asserting. |
| Determinism claim | A "bit-identical across runs" assert on GPU | Document honestly that it is NOT claimable on GPU | Reduction order + transcendental impl are not guaranteed bit-stable across GPU launches; the CPU backend keeps the SC2 determinism claim. |

**Key insight:** Phase 7's only genuinely new code is (a) the `R` type parameter threading through `lib.rs`, (b) one `device.rs` client constructor + one error variant, (c) three harness `*_case()` constructors + one enum extension, and (d) one report-emitter module + one matrix sibling. Everything numerical already exists.

## GPU Numerical Divergence (D-03 — the research-flagged core)

### FMA contraction: minimal surface on this workload

`[VERIFIED: crates/treelite-cubecl/src/kernels/default_raw.rs]` The accumulation kernel does **pure sequential addition** — `output[cell] += v` in `tree_id` order, then `output[cell] /= factor`, then a single `f64`-promoted `output + base_score`. There is **no `a*b + c` expression** anywhere on the tree-sum / base-score path. FMA contraction only changes results when a compiler fuses a multiply-then-add into a single rounding step; with no mul-add pattern, there is nothing to contract. `descend` (traversal) does only comparisons and index math — no float arithmetic that rounds. **Conclusion: FMA contraction contributes ~0 to the tree-sum/base-score deviation.** `[VERIFIED: code inspection]`

IEEE-754 add/sub/mul/div are **correctly rounded on every cubecl backend** — OpenCL/SPIR-V (wgpu-vulkan), HIP, and CUDA all guarantee correctly-rounded basic ops. `[CITED: Khronos OpenCL C spec — "addition, subtraction, multiplication, fused multiply-add … are IEEE 754 compliant and are therefore correctly rounded"]` So the deviation budget for the basic-arithmetic portion of any model class is **0 ULP** in principle; the only residual on the accumulate path is that the GPU runs one row per unit but the **per-row tree-sum order is preserved** (the kernel is serial in `tree_id`, GTIL-08), so even the non-associativity of float add cannot diverge from the scalar reference's order. `[VERIFIED: default_raw.rs serial tree loop]`

### Transcendental divergence: the real (and only material) axis

The divergence concentrates entirely in the postprocessors that call `exp`/`exp2`/`ln_1p`. Note that **postprocessors run host-side today** (`PredictCpuElem::apply_postprocessor` after readback, `lib.rs:394-397`) — meaning on the current architecture even the GPU path's postprocessor runs on the CPU's libm. The GPU transcendental divergence only manifests if/when postprocessors are moved into the kernel (`kernels/postproc.rs` exists but the host launcher currently applies the host-side twins). **Planner decision point:** the GPU report's transcendental deviation will be ~0 for `Default` kind IF the postprocessor stays host-side post-readback. If the plan moves postprocessing into the device kernel for GPU backends, the GPU `exp`/`exp2` divergence below applies. The predicted-deviation model covers BOTH cases.

Published exhaustive single-precision accuracy data:

| Function | GPU max ULP vs correctly-rounded | Source |
|----------|----------------------------------|--------|
| `expf` (HIP, hardware path) | 1 ULP | `[CITED: LLVM-libc GPU GSoC 2025 report — hip-math expf max ULP 1]` |
| `expf` (CUDA, cuda-math) | 2 ULP | `[CITED: LLVM-libc GPU GSoC 2025 report]` |
| `expf` (llvm-libm on AMDGPU/CUDA) | 0 ULP (correctly-rounded software path) | `[CITED: LLVM-libc GPU GSoC 2025 report]` |
| `logf` (HIP) | 2 ULP | `[CITED: LLVM-libc GPU GSoC 2025 report — hip-math logf max ULP 2]` |
| `exp`/`exp2` (OpenCL single-precision spec bound) | ≤ 3-4 ULP (embedded profile ≤4) | `[CITED: Khronos OpenCL C spec / man exp.3clc]` |
| `native_exp` (any backend) | implementation-defined (could be large) | `[CITED: Khronos OpenCL spec — native_ accuracy is implementation-defined]` — cubecl's `F::exp` should map to the standard, NOT `native_`, but this must be confirmed empirically. |

### Predicted-deviation model (the D-03 deliverable)

The deviation a model class shows on GPU is determined by **(1) whether its postprocessor calls a transcendental, (2) the input precision (f32 vs f64), and (3) whether that transcendental runs on-device or host-side.** Assuming on-device transcendentals (worst case) for the prediction:

| Model-class postprocessor family | Touches transcendental? | Predicted GPU max |delta| (f32 input) | Predicted (f64 input) | Reasoning |
|-----------------------------------|------------------------|----------------------------------------|------------------------|-----------|
| `identity` / `identity_multiclass` / `hinge` / `signed_square` | no | **≤ 1e-6** (basic-op rounding only; effectively 0 with order-preserved sum) | **≤ 1e-9** | No transcendental; correctly-rounded basic ops; serial sum order preserved. Well inside 1e-5. |
| `sigmoid` / `multiclass_ova` (single `exp`) | yes | **~1e-6 to ~5e-6** | **≤ 1e-7** | 1-2 ULP on `exp` at f32 ≈ relative ~1-2e-7; sigmoid saturates so absolute error stays small; near the slope a 2-ULP `exp` perturbation can reach the low-1e-6 absolute band. f64 input runs the f64 twin → far tighter. |
| `exponential` (raw `exp`) | yes | **~1e-6 to ~1e-5+** (input-magnitude dependent) | **≤ 1e-7** | `exp(x)` is unbounded; a 2-ULP relative error on a large `exp` output is a large absolute error. This is the class most likely to need f64 promotion (D-02) on f32 input. **Flag: may exceed 1e-5 absolute on large margins — exactly where D-02 per-class f64 promotion applies.** |
| `exponential_standard_ratio` (`exp2`) | yes | **~1e-6 to ~5e-6** | **≤ 1e-7** | `exp2(-x/c)` bounded in (0,1] for x≥0; 2-ULP `exp2` ≈ relative ~2e-7 → low-1e-6 absolute. |
| `logarithm_one_plus_exp` (`exp` then `ln_1p`) | yes (two) | **~2e-6 to ~1e-5** | **≤ 1e-7** | Compounds two transcendentals; `ln1p` on AMD ~2 ULP. Watch the large-input regime. |
| `softmax` (`exp` per class + f64 accumulate) | yes | **~1e-6 to ~5e-6**, compounding across classes | **≤ 1e-7** | Per-class `exp` errors (1-2 ULP each) partially cancel under normalization, but the max-subtraction + division can amplify near-ties. The Phase-5 `softmax` keeps `norm_const` in f64 even for f32 input, which damps the divergence. |

**Honest assumptions behind these numbers:**
1. cubecl's `F::exp`/`F::exp2` map to the **standard** (≤3-4 ULP) transcendental, not the `native_` (implementation-defined) one. **This is the #1 thing the empirical report must confirm** — if cubecl emits `native_exp` on HIP, the deviations could be an order of magnitude larger and more model classes need f64 promotion.
2. f64 input runs the f64 postprocessor twins, and AMD/ROCm supports f64 transcendentals at full accuracy. (gfx CDNA has good f64; some RDNA consumer cards throttle f64 — note the actual device in the report manifest.)
3. The ranges are **per-element max absolute deviation** against the f64 scalar reference, on the existing frozen fixture margins (not adversarial large inputs).

The committed report carries this **predicted** column alongside the **measured ROCm** column (D-03/D-08); a measured value materially outside its predicted band is itself a finding worth recording (e.g. it would reveal a `native_` mapping).

### Determinism on GPU (SC2-style bit-identical) — NOT claimable, state honestly

`[CITED: Khronos OpenCL spec — "whether or when the underlying work-item and global floating-point state is reused is implementation-defined"]` Combined with non-associative float reduction and implementation-defined transcendental rounding, **bit-identical determinism across GPU launches is NOT guaranteed and must NOT be claimed in the report.** The Phase-6 SC2 determinism claim is a **CPU-backend** property and stays there. For THIS workload the per-row serial tree-sum IS order-stable (each unit owns disjoint output cells, no cross-unit reduction — `default_raw.rs` doc), so in practice GPU runs may well be run-to-run stable; but the report should state determinism is **observed-stable on the tested device, not guaranteed** rather than asserting bit-identity. This is already consistent with the project's "Out of Scope: Bit-exact GPU reproducibility" entry in REQUIREMENTS.md.

## Report Emission (D-06/D-08)

**Recommended location:** `docs/GPU_EQUIVALENCE_REPORT.md` (a committed, top-level docs artifact is more discoverable than burying it in `.planning/`; the planner may choose the phase dir if it prefers co-location with the phase). Either satisfies D-08.

**Recommended markdown layout** (rows = the frozen `fixtures/gtil/*.golden.json` model classes, D-07):

```markdown
# GPU Equivalence Report

Regenerated on: <device name / ROCm version / rustc / date>   (from manifest, D-06)
Reference: f64 scalar GTIL (the 1e-5 CPU spine).  Observational — NOT a CI gate (D-01).

| Model class | Postprocessor | ROCm max |delta| | f64 fallback used? | CUDA | wgpu | Predicted band (D-03) |
|-------------|--------------|------------------|--------------------|------|------|-----------------------|
| binary (xgb) | sigmoid | 3.1e-6 | no | not run — no device | <measured or not run> | ~1e-6..5e-6 |
| leaf_vec_mc | softmax | 2.4e-6 | no | not run — no device | … | ~1e-6..5e-6 |
| iforest | exponential_standard_ratio | 1.8e-6 | no | not run — no device | … | ~1e-6..5e-6 |
| … (one row per frozen fixture model class) | … | … | … | … | … | … |

Determinism: observed run-to-run stable on <device>; bit-identity NOT guaranteed on GPU (per OpenCL spec).
```

**Emission mechanics:** Add a `report.rs` to `treelite-harness` that the new `gtil_matrix_gpu.rs` sibling calls. The sibling drives the SAME frozen fixtures through `rocm_case()` (the registered backend), reuses `assert_within`'s max-`|delta|` computation but in **report mode** (record, do not panic — D-01 observational), records per-cell `Backend` + max-`|delta|` + whether the f64 twin was used, and writes the markdown. CUDA/wgpu columns render "not run — no device" via the `DeviceUnavailable` typed skip (D-05). The manifest header reuses `manifest.rs` provenance fields (`backend`, `rustc`, `os`, `arch`, device name).

**Machine-readable sidecar (D-08, planner discretion — RECOMMENDED):** Emit a `gpu_equivalence.json` alongside the markdown with `[{model_class, postprocessor, backend, max_abs_delta, f64_fallback, predicted_low, predicted_high}]`. Low cost (the data already exists in the report-mode accumulation) and lets a future regression test assert the report didn't silently regress. Keep it observational (not a gate) to honor D-01.

## CPU/GPU Crossover (D-09/D-10)

**How to benchmark (D-10, measured-not-pre-committed):** Add a `criterion`-style or simple wall-clock micro-benchmark (the planner may use a plain `#[test]`-gated timing loop behind `--ignored` to avoid CI cost) that runs the SAME model on `ScalarCpu`/`CubeclCpu` vs `Rocm` across a sweep of `num_row ∈ {1, 10, 100, 1k, 10k, 100k}` and a couple of forest sizes, on the developer's ROCm hardware. Record where the GPU wall-clock (including upload + launch + readback) first beats the CPU.

**Which metric likely dominates:** The GPU pays a fixed upload+launch+readback cost per call; that cost is amortized over rows. So the crossover almost certainly keys on **row count** (or `rows × num_feature` for the input transfer, plus a forest-upload constant). For a small forest the input transfer dominates → `rows × num_feature`; for a large forest the per-row traversal work dominates → row count. **Do NOT pre-commit a formula** (D-10) — document the empirical crossover number and which metric it keyed on, e.g. "on <device>, ROCm beats CubeclCpu above ~N rows for the <forest> fixture." The Phase-8 PyO3 caller is the documented consumer (D-09).

## Common Pitfalls

### Pitfall 1: Assuming `cuda_case()`/`wgpu_case()` panic-aborts on a missing device
**What goes wrong:** On the NVIDIA-less dev box, constructing a CUDA client may hard-abort the FFI (not a catchable Rust panic), taking down the test process — turning a D-05 "skip" into a crash.
**Why it happens:** CUDA/HIP driver init failures sometimes `abort()` in the C layer below `catch_unwind`'s reach.
**How to avoid:** A spike task MUST run `cuda_case()` (and `wgpu_case()`) on this hardware FIRST and confirm graceful behavior. If construction aborts, the harness must NOT attempt construction for an unavailable backend — instead emit "not run — no device" without calling `client()`, using whatever device-enumeration/probe cubecl 0.10 exposes (confirm via `cargo doc`).
**Warning signs:** A test binary that aborts with a CUDA/HIP driver message instead of returning a typed error.

### Pitfall 2: cubecl emitting `native_exp` instead of standard `exp` on HIP
**What goes wrong:** The predicted-deviation model assumes ≤3-4 ULP transcendentals. `native_exp` is implementation-defined and can be far worse, pushing `exponential`/`logarithm_one_plus_exp` classes past 1e-5 and forcing more f64 promotion than predicted.
**Why it happens:** Some frameworks map `.exp()` to the fast native intrinsic for performance.
**How to avoid:** The empirical report IS the check — a measured deviation far above the predicted band signals a `native_` mapping. If so, document it and apply D-02 f64 promotion to the affected classes.
**Warning signs:** Measured ROCm `|delta|` an order of magnitude above the predicted band for an `exp`-family class.

### Pitfall 3: wgpu f64 + buffer-size/alignment limits
**What goes wrong:** WGSL historically lacks native `f64`; an f64-preset model on the wgpu backend may fail to compile the kernel or silently downcast. Also wgpu enforces per-binding storage-buffer size limits and 4-byte alignment.
**Why it happens:** WebGPU's portability baseline targets f32; f64 is an extension that not all backends expose.
**How to avoid:** Treat wgpu as **f32-only and best-effort** this phase (D-08 lets wgpu be a "not run" or partial row); validate empirically on the AMD-via-Vulkan path. The forest columns are naturally aligned (i32/u32/f32/f64 via bytemuck), so alignment is fine for f32; the size limit matters only for very large forests (the fixtures are small).
**Warning signs:** A wgpu kernel-compile error mentioning f64/`double`, or a buffer-binding-size validation error.

### Pitfall 4: Postprocessor location ambiguity (host vs device) skews the report
**What goes wrong:** If the postprocessor runs host-side (current architecture), the GPU report shows ~0 transcendental divergence and the D-03 model looks "wrong" (predicted bands unused). If it runs on-device, the bands apply.
**Why it happens:** `lib.rs` applies `F::apply_postprocessor` AFTER readback on the CPU libm today.
**How to avoid:** The planner must DECIDE and DOCUMENT whether GPU postprocessing stays host-side (simplest, report measures only traversal/accumulate divergence ≈ 0, GPU-04 still satisfied with the predicted column documenting the on-device risk) or moves on-device (report measures the real transcendental divergence). Recommended: keep it host-side this phase (less code, the f64 twins already work), and have the report's predicted column document what on-device would cost. Either way the report must state which path produced the numbers.
**Warning signs:** All measured `|delta|` values are ~1e-7 regardless of postprocessor — that means postprocessing ran host-side.

### Pitfall 5: Workspace default build accidentally pulling a GPU runtime
**What goes wrong:** Enabling `cubecl/rocm` at the workspace level (not behind a crate feature) makes `cargo build` require ROCm system libs, breaking the default/CI build (D-04 violation).
**How to avoid:** Keep root `Cargo.toml` `cubecl` features = `["cpu"]` only; forward GPU runtimes via `treelite-cubecl` crate features (`rocm = ["cubecl/rocm"]`). Confirm `cargo build` (no features) and `cargo test --workspace` stay green with NO ROCm libs present.
**Warning signs:** A clean `cargo build` failing with a HIP/`cubecl-hip-sys` link error on a machine without ROCm.

## Runtime State Inventory

> This is an additive feature phase (new features, new harness registration, new report file). It is NOT a rename/refactor/migration. The categories below are checked for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified: no datastore keys, IDs, or collection names change. The frozen `fixtures/gtil/*.golden.json` are READ unchanged (the GPU sibling asserts against the SAME goldens). | none |
| Live service config | None — verified: no external service config. The only new committed artifact is `docs/GPU_EQUIVALENCE_REPORT.md` (regenerated on ROCm hardware, D-06), which is a repo file, not service state. | regenerate report on ROCm hardware (a developer action, D-06) |
| OS-registered state | None — verified: no OS task/service registration. GPU device access is via the ROCm/HIP driver at runtime, not a registered task. | none |
| Secrets/env vars | None — verified: no secrets or env-var names. (cubecl may read HIP/CUDA driver env vars like `HIP_VISIBLE_DEVICES`, but those are pre-existing system env, not project-introduced.) | none |
| Build artifacts | New cargo feature surface means `cargo test -p treelite-cubecl --features rocm` produces a DIFFERENT build than the default. The default `target/` artifacts are unaffected. No stale-artifact risk from a rename. | none (feature builds are separate by design) |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `approx` (`assert_abs_diff_eq!`) — the established harness pattern |
| Config file | none — cargo-native; fixtures under workspace-root `fixtures/gtil/` |
| Quick run command | `cargo test -p treelite-cubecl` (CPU path, no GPU) |
| Full suite command | `cargo test --workspace` (default, cpu-only) + `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu -- --ignored` (ROCm hardware only) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-03 | ROCm backend runtime-selectable, produces predictions for the harness model set | integration (hardware-gated) | `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu` | ❌ Wave 0 — new sibling `tests/gtil_matrix_gpu.rs` |
| GPU-03 | `predict::<R>` generic launcher still passes on CpuRuntime (registration-not-refactor regression) | integration | `cargo test -p treelite-harness --test gtil_matrix_cubecl` | ✅ (existing; must stay green via the `predict_cpu` shim) |
| GPU-03 (D-05) | Selecting a backend with no device returns `CubeclError::DeviceUnavailable` (skip, not crash) | unit + spike | `cargo test -p treelite-cubecl --features cuda device_absent_is_typed_skip` | ❌ Wave 0 — new test + spike |
| GPU-04 | Report documents per-class max deviation on ROCm, predicted vs measured columns | integration (hardware-gated, observational) | `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu` (emits `docs/GPU_EQUIVALENCE_REPORT.md`) | ❌ Wave 0 — new `report.rs` + sibling |
| GPU-04 (D-02) | Per-class f64 postprocessor promotion recorded when a class would exceed budget | integration | same sibling; assert the report's "f64 fallback used?" column reflects the executed path | ❌ Wave 0 |
| GPU-04 (D-01) | The GPU report is observational — a GPU `|delta|` above 1e-5 RECORDS, does not fail CI | integration (report-mode, no panic) | report-mode `assert_within` variant that returns max-`|delta|` without panicking | ❌ Wave 0 |
| (regression) | Default workspace build stays cpu-only, green, with NO ROCm libs (D-04) | build | `cargo build` and `cargo test --workspace` (no features) | ✅ (existing CI; must stay green) |

**Skip-not-fail semantics (encode in the sibling):** GPU rows are hardware-gated. The `gtil_matrix_gpu.rs` sibling MUST:
- Attempt the selected backend; on `Err(DeviceUnavailable)` mark the row "not run — no device" and CONTINUE (the test passes — absence is a skip, D-05). Use `#[ignore]` for the ROCm hardware test so default CI does not run it (the developer runs it explicitly on ROCm hardware to regenerate the committed report, D-06).
- NEVER fail on a GPU `|delta| > 1e-5` (D-01 observational); only RECORD it. The hard 1e-5 gate remains on the scalar/cubecl-cpu siblings.

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-cubecl` + `cargo clippy -p treelite-cubecl` (CPU path; fast)
- **Per wave merge:** `cargo test --workspace` (default cpu-only; the hard 1e-5 gate)
- **Phase gate (on ROCm hardware):** `cargo test -p treelite-harness --features rocm --test gtil_matrix_gpu -- --ignored` regenerates and commits `docs/GPU_EQUIVALENCE_REPORT.md`; full default suite green before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/treelite-cubecl/src/device.rs` — `device_and_client::<R>()` + per-backend constructors (D-05)
- [ ] `crates/treelite-cubecl/src/error.rs` — add `CubeclError::DeviceUnavailable { backend }` variant
- [ ] `crates/treelite-cubecl/tests/device_absent.rs` — spike + test that no-device is a typed skip, not a crash (Pitfall 1)
- [ ] `crates/treelite-harness/src/report.rs` — markdown (+ optional JSON sidecar) emitter from per-cell provenance
- [ ] `crates/treelite-harness/tests/gtil_matrix_gpu.rs` — `#[ignore]`d ROCm sibling driving the frozen matrix, report-mode (record not panic)
- [ ] Feature plumbing: `rocm = ["cubecl/rocm"]`, `cuda = ["cubecl/cuda"]`, `wgpu = ["cubecl/wgpu"]` in `treelite-cubecl/Cargo.toml`
- [ ] `predict_cpu::<F>` retained as a shim over `predict::<CpuRuntime, F>` so the existing cubecl-cpu test/case compile unchanged

## Security Domain

> `security_enforcement` was not found set to `false` in config; included for completeness. This phase has a narrow surface (GPU compute over already-loaded, already-validated models from trusted fixtures), so most ASVS categories are N/A.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (local compute library) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access-control surface |
| V5 Input Validation | yes | The existing `validate_shape`/`validate_leaf_vectors` host-side checks (run BEFORE any device op, T-06-06/09) carry to GPU unchanged — a malformed model returns a typed error, never an OOB device write. Confirm these still run on the `R`-generic path (they live in `upload_forest`, which is already generic). |
| V6 Cryptography | no | No cryptography |

### Known Threat Patterns for {Rust + cubecl GPU}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read/write from a malformed model | Tampering / DoS | Host-side `validate_shape` + `validate_leaf_vectors` before any `create_from_slice` (already implemented, carries to GPU). |
| FFI driver abort on missing device crashing the host process | Denial of Service | D-05 typed `DeviceUnavailable` skip + Pitfall-1 spike confirming graceful behavior; never auto-invoke an unavailable backend. |
| `unsafe { ArrayArg::from_raw_parts(...) }` length mismatch | Tampering | Lengths are derived from the validated host columns (`num_nodes_total` etc.); validation precedes the unsafe block. Unchanged on GPU. |

## Code Examples

### Generalized launcher call site (the one-line change repeated at 3 sites)
```rust
// Source: crates/treelite-cubecl/src/lib.rs:444/512/580 (the launch sites)
// BEFORE:  kernels::leaf_id::predict_leaf_id::launch::<F, T, CpuRuntime>(client, …)
// AFTER:   kernels::leaf_id::predict_leaf_id::launch::<F, T, R>(client, …)
// (client is now &ComputeClient<R>; everything between is byte-identical)
```

### ROCm client construction (the verified pattern)
```rust
// Source: cubecl manual Handling_Interleaved_Complex_Numbers_in_CubeCL_with_ROCm_Backend.md
#[cfg(feature = "rocm")]
type BackendRuntime = cubecl::hip::HipRuntime;
#[cfg(feature = "rocm")]
let device = cubecl::hip::HipDevice::default();
let client = BackendRuntime::client(&device);   // identical to CpuRuntime::client(&device)
```

### wgpu client construction (alternate init path — note it differs from cpu/hip)
```rust
// Source: cubecl manual cubecl_matmul_gemm_example.md
let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
// wgpu may use init_setup/init_device then ComputeClient::load(&device), OR
// WgpuRuntime::client(&device) — confirm the 0.10 surface via cargo doc; the
// GEMM example shows the init_setup/init_device/ComputeClient::load path.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate per-backend cubecl runtime crates as direct deps | `cubecl` umbrella crate features (`hip`/`cuda`/`wgpu`) | cubecl ≥ 0.x (current 0.10) | One pinned dep, feature-gated runtimes; avoids version skew. |
| `rocm` as a distinct runtime | `rocm` feature implies `hip`; the runtime is `cubecl::hip::HipRuntime` | cubecl 0.10 | The "ROCm backend" is the HIP runtime; the feature name is `rocm` (or `hip`), the type is `HipRuntime`. |

**Deprecated/outdated:**
- Do NOT use `zenforks-cubecl-hip` (a non-official `cargo search` near-name fork). Use `cubecl/rocm`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cubecl::cuda::CudaRuntime`/`CudaDevice` are the exact symbol names (inferred from the symmetric cpu/hip re-export). | Standard Stack | Low — verify via `cargo doc` when enabling the `cuda` feature; symbol rename is a trivial fix. |
| A2 | cubecl's `F::exp`/`F::exp2` map to the STANDARD (≤3-4 ULP) transcendental, not `native_`. | Predicted-Deviation Model | Medium — if `native_`, predicted bands understate divergence and more classes need f64 promotion. The empirical report catches this. |
| A3 | HIP/CUDA `client()` returns gracefully (or panics catchably) on a missing device. | Pattern 2 / Pitfall 1 | High — if it FFI-aborts, `catch_unwind` cannot trap it; a pre-construction probe is required. MUST be confirmed by a Wave-0 spike. |
| A4 | Predicted per-class deviation magnitudes (1e-6..1e-5 bands). | Predicted-Deviation Model | Medium — these are predictions; the committed ROCm report is the ground truth and validates against them (D-03 by design). |
| A5 | Postprocessors continue to run host-side post-readback (current architecture), so traversal/accumulate GPU divergence ≈ 0. | Pitfall 4 | Medium — drives whether the measured report shows transcendental divergence at all; the planner must decide host-vs-device and document it. |
| A6 | wgpu on the AMD box runs via Vulkan and is f32-only this phase. | Pitfall 3 / Open Questions | Low — determined empirically; wgpu is allowed to be a "not run"/partial row (D-08). |

## Open Questions

1. **Does cubecl CUDA/HIP `client()` FFI-abort on a missing device?**
   - What we know: D-05 requires a typed skip; `catch_unwind` traps Rust panics only.
   - What's unclear: whether the C driver init `abort()`s below `catch_unwind`.
   - Recommendation: Wave-0 spike on this NVIDIA-less box running `cuda_case()`; if it aborts, probe device availability before construction (cubecl device-enumeration API — confirm via `cargo doc`) and never call `client()` for an absent backend.

2. **Will wgpu validate as a real row on the AMD/Vulkan device, or "not run"?**
   - What we know: wgpu shares the seam; the AMD box has Vulkan; WGSL f64 is shaky.
   - What's unclear: whether the f32 fixtures run cleanly and whether f64-preset fixtures compile.
   - Recommendation: Treat wgpu as best-effort f32 (D-08 permits partial/"not run"); determine empirically and record the actual device capability in the report manifest.

3. **Host-side vs on-device postprocessing for the GPU path?**
   - What we know: postprocessors run host-side today (`lib.rs:394-397`); `kernels/postproc.rs` exists.
   - What's unclear: which path the plan chooses; it determines whether the report shows transcendental divergence.
   - Recommendation: keep host-side this phase (less code, f64 twins already work, GPU-04 still satisfied with the predicted column documenting on-device risk); state the choice explicitly in the report.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable, edition 2024 | all | ✓ | (project toolchain) | — |
| `cubecl` 0.10.0 (cpu) | existing CPU path | ✓ | 0.10.0 (in Cargo.lock) | — |
| ROCm/HIP system libs + AMD GPU | `Backend::Rocm` validation (GPU-03/GPU-04) | ✓ (per project memory `gpu-hardware-rocm-only`) | ROCm (device-specific) | none — this is THE validated backend; report regenerated here (D-06) |
| `cubecl-hip-sys` 7.2.x (ROCm binding) | `cubecl/rocm` feature build | pulled transitively when `rocm` feature on | 7.2.5321100 | none |
| CUDA toolkit + NVIDIA driver | `Backend::Cuda` validation | ✗ (no NVIDIA locally) | — | **D-05 skip**: build-supported, rows render "not run — no device"; validated only where NVIDIA exists |
| Vulkan driver (for wgpu on AMD) | `Backend::Wgpu` | ✓ (likely, via the AMD GPU) | device-specific | wgpu may be partial/"not run" (Pitfall 3, D-08) |

**Missing dependencies with no fallback:** none that block the phase — ROCm (the GPU-03-satisfying backend) is present.
**Missing dependencies with fallback:** CUDA (no NVIDIA) → D-05 typed skip, "not run — no device" rows. This is the explicitly-designed-for case, not a blocker.

## Sources

### Primary (HIGH confidence)
- In-repo code (read in full this session): `crates/treelite-cubecl/src/lib.rs`, `upload.rs`, `error.rs`, `kernels/mod.rs`, `kernels/traversal.rs`, `kernels/default_raw.rs`, `Cargo.toml`; root `Cargo.toml`; `crates/treelite-harness/src/lib.rs`, `src/manifest.rs`, `tests/gtil_matrix_cubecl.rs`; `crates/treelite-gtil/src/postprocessor.rs`; `treelite-mainline/src/gtil/predict.cc` (postprocessor/base-score region).
- cubecl manual: `Handling_Interleaved_Complex_Numbers_in_CubeCL_with_ROCm_Backend.md` (ROCm `HipRuntime`/`HipDevice`/`R::client` pattern), `cubecl_matmul_gemm_example.md` (wgpu init), `Cubecl_algebra.md` (`Float` transcendental methods, "f64 support varies by GPU; CpuRuntime supports both").
- `cargo search` (this session): `cubecl`/`cubecl-hip`/`cubecl-cuda`/`cubecl-wgpu`/`cubecl-cpu` all 0.10.0.
- crates.io metadata (this session): repo URLs (tracel-ai monorepo), download counts, creation dates for the four runtime crates.

### Secondary (MEDIUM confidence)
- `[CITED: lib.rs/crates/cubecl/features]` — cubecl 0.10.0 feature flags (`cpu`/`cuda`/`hip`/`rocm`/`wgpu`/`vulkan`).
- `[CITED: LLVM-libc GPU GSoC 2025 report — blog.llvm.org]` — exhaustive single-precision ULP: hip `expf` 1 ULP, cuda `expf` 2 ULP, hip `logf` 2 ULP.
- `[CITED: Khronos OpenCL C spec / man exp.3clc]` — basic ops correctly-rounded; `exp`/`exp2` ≤3-4 ULP; `native_` implementation-defined; reuse of FP state implementation-defined (determinism caveat).

### Tertiary (LOW confidence — flagged for empirical validation)
- Predicted per-class deviation magnitudes (1e-6..1e-5 bands) — derived from the ULP studies above; the committed ROCm report validates them (D-03 by design).
- `CudaRuntime`/`CudaDevice` exact symbol names (A1) — inferred from the cpu/hip pattern; confirm via `cargo doc`.

## Metadata

**Confidence breakdown:**
- Standard stack (crates/features): HIGH — verified versions via `cargo search` + crates.io; feature flags via the cubecl features page; ROCm types via the cubecl manual.
- Architecture (host-launcher generalization): HIGH — read every `CpuRuntime` site; the change is a mechanical `R` lift confirmed by the already-generic `upload`/kernels and the ROCm-manual `R::client` pattern.
- Pitfalls / device-absence (D-05): MEDIUM — the FFI-abort-on-no-device behavior (A3) is the one genuine risk and is gated behind a mandatory Wave-0 spike.
- Predicted deviation (D-03): MEDIUM — magnitudes cited from published GPU-libm ULP studies; the committed ROCm report is the ground truth by design.

**Research date:** 2026-06-11
**Valid until:** ~2026-07-11 (30 days; cubecl is moving but 0.10.0 is the current pin and the workload is stable). Re-verify cubecl version/feature names if cubecl publishes 0.11 before planning.

# Phase 7: GPU Backend & Equivalence Report - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 9 (3 new, 6 modified)
**Analogs found:** 9 / 9 (every new symbol has an in-repo template)

This is a **host-launcher generalization + backend-registration + reporting** phase, NOT a kernel phase. The kernels (`kernels/*`), `UploadedForest<R>`, and `upload_forest::<R, F>` are ALREADY generic over `R: Runtime` (verified live below) — the planner must NOT touch them. The concrete-`CpuRuntime` surface is confined entirely to `crates/treelite-cubecl/src/lib.rs`.

## File Classification

| New/Modified File | New/Mod | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|---------|------|-----------|----------------|---------------|
| `crates/treelite-cubecl/src/lib.rs` | modified | service (host launcher) | transform / request-response | self (the `predict_cpu` family — lift `CpuRuntime` → `R`) | exact (in-file) |
| `crates/treelite-cubecl/src/device.rs` | **new** | utility (per-backend client init) | transform | `lib.rs:312` `CpuRuntime::client(&Default::default())` + RESEARCH Pattern 2 | role-match |
| `crates/treelite-cubecl/src/error.rs` | modified | model (typed error enum) | n/a | self (`CubeclError::Unsupported` variant, lines 82-88) | exact (in-file) |
| `crates/treelite-cubecl/Cargo.toml` | modified | config | n/a | self (`[dependencies]` cubecl workspace) | exact (in-file) |
| `Cargo.toml` (root) | modified | config | n/a | self (`[workspace.dependencies]` line 25) | exact (in-file) |
| `crates/treelite-harness/src/lib.rs` | modified | service (backend registry) | request-response | `cubecl_cpu_case()` (lines 157-187) + `Backend` enum (53-66) | exact (in-file) |
| `crates/treelite-harness/src/report.rs` | **new** | utility (markdown/JSON emitter) | transform / file-I/O | `manifest.rs` (Manifest fields) + matrix-sibling `assert_within`/max-dev loop | role-match |
| `crates/treelite-harness/tests/gtil_matrix_gpu.rs` | **new** | test | request-response / file-I/O | `tests/gtil_matrix_cubecl.rs` (whole file — 463 lines) | exact (sibling-of) |
| `docs/GPU_EQUIVALENCE_REPORT.md` | **new** | doc artifact | n/a | RESEARCH "Report Emission" layout (07-RESEARCH.md lines 351-367) | template-only |
| `crates/treelite-gtil/src/postprocessor.rs` | **read-only reuse** | utility (f64 twins) | transform | self (`*_f64` twins already exist) | exact (reuse, no edit) |

---

## Pattern Assignments

### `crates/treelite-cubecl/src/lib.rs` (service / host launcher) — MODIFIED, the central change

**Analog:** itself. The change is a mechanical `CpuRuntime` → `R: Runtime` lift. CONFIRMED live: every concrete-runtime mention is in THIS file; `upload_forest::<R, F>` (upload.rs:409) and `kernels::*::launch::<F, T, R>` are already `R`-generic.

**Current shape that MUST keep compiling/passing (the cubecl-cpu path is the green 1e-5 gate, D-01):**

Imports (lines 31-36) — note `CpuRuntime` is imported concretely today:
```rust
use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime};
use treelite_core::{Model, ModelVariant};
use treelite_gtil::{Config, PredictKind, postprocessor};
```

The public entry today (lines 294-321) — hardcodes the CPU client at line 312:
```rust
pub fn predict_cpu<F: PredictCpuElem>(
    model: &Model, data: &[F], num_row: usize, cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    if model_routes_to_scalar_fallback(model) {                       // D-02 gate — UNCHANGED
        return treelite_gtil::predict::<F>(model, data, num_row, cfg)
            .map_err(|e| CubeclError::Unsupported(format!("scalar fallback: {e}")));
    }
    let client = CpuRuntime::client(&Default::default());             // <-- the ONE concrete-runtime line
    match cfg.kind {
        PredictKind::Default | PredictKind::Raw => launch_default_raw::<F>(&client, model, data, num_row, cfg),
        PredictKind::LeafId => launch_leaf_id::<F>(&client, model, data, num_row),
        PredictKind::ScorePerTree => launch_score_per_tree::<F>(&client, model, data, num_row),
    }
}
```

**Generalization to copy (RESEARCH Pattern 1, lines 196-227):** make `predict::<R, F>` generic; the client comes from `device.rs`; keep a thin `predict_cpu` SHIM so `cubecl_cpu_case()` and `gtil_matrix_cubecl.rs` compile UNCHANGED (`git diff` on the CPU path stays minimal — registration-not-refactor):
```rust
pub fn predict<R: Runtime, F: PredictCpuElem>(
    model: &Model, data: &[F], num_row: usize, cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    if model_routes_to_scalar_fallback(model) { /* …same D-02 fallback… */ }
    let client = crate::device::client::<R>()?;     // Err(DeviceUnavailable) — NO silent CPU fallback (D-05)
    match cfg.kind { /* …same arms, but launch_*::<R, F> … */ }
}

// Shim keeps the entire Phase-6 surface byte-identical:
pub fn predict_cpu<F: PredictCpuElem>(m: &Model, d: &[F], n: usize, c: &Config) -> Result<Vec<F>, CubeclError> {
    predict::<CpuRuntime, F>(m, d, n, c)
}
```

**The four launcher fns to thread `R` through** — each currently names `ComputeClient<CpuRuntime>` in its signature AND `CpuRuntime` at the launch call site. These are the ONLY edits beyond `predict`:

| fn | signature client type (today) | launch call site (today) |
|----|-------------------------------|--------------------------|
| `launch_default_raw` (342) / `run_default_raw` (403) | `&cubecl::client::ComputeClient<CpuRuntime>` (343, 404) | `predict_default_raw::launch::<F, T, CpuRuntime>` (line 444) |
| `launch_leaf_id` (478) / `run_leaf_id` (490) | `…ComputeClient<CpuRuntime>` (479, 491) | `predict_leaf_id::launch::<F, T, CpuRuntime>` (line 512) |
| `launch_score_per_tree` (534) / `run_score_per_tree` (555) | `…ComputeClient<CpuRuntime>` (535, 556) | `predict_score_per_tree::launch::<F, T, CpuRuntime>` (line 580) |
| `predict_cpu_sparse` (327) | (no client — pure scalar fallback) | — UNCHANGED (D-02) |

The one-line launch-site change (repeated at 444/512/580), per RESEARCH lines 484-488:
```rust
// BEFORE: kernels::leaf_id::predict_leaf_id::launch::<F, T, CpuRuntime>(client, …)
// AFTER:  kernels::leaf_id::predict_leaf_id::launch::<F, T, R>(client, …)   // client now &ComputeClient<R>
```
Everything between the call-site type args is byte-identical (the `upload_forest::<CpuRuntime, T>` calls at 422/499/566 become `upload_forest::<R, T>`).

**Postprocessor (D-02) stays as-is — it is already an f32/f64 split that runs HOST-side after readback** (lines 394-397, `F::apply_postprocessor`). The f32 impl is lines 84-161, the f64 twin impl 163-232. Per RESEARCH Pitfall 4 (lines 399-403), the planner DECIDES host-vs-device and DOCUMENTS it; recommended host-side (the f64 twins already work). Per-class f64 promotion (D-02) is a dispatch choice over these existing arms, NOT new math.

---

### `crates/treelite-cubecl/src/device.rs` (utility / per-backend client init) — NEW

**Analog:** the single line `lib.rs:312` `CpuRuntime::client(&Default::default())`, generalized per RESEARCH Pattern 2 (lines 229-251) + the verified ROCm pattern (RESEARCH lines 491-508).

**Pattern to model on** — a generic `client::<R>()` returning a typed skip, with `#[cfg(feature = …)]`-gated per-backend constructors:
```rust
// CpuRuntime path mirrors lib.rs:312 exactly; GPU paths add the typed skip (D-05).
#[cfg(feature = "rocm")]
pub fn rocm_client() -> Result<ComputeClient<cubecl::hip::HipRuntime>, CubeclError> {
    // R::client(&R::Device::default()) is the uniform cubecl pattern (ROCm manual).
    // Catch-on-construct is the FIRST attempt; the Wave-0 spike (Pitfall 1) confirms
    // whether HIP/CUDA client() returns gracefully or FFI-aborts on a missing device.
    std::panic::catch_unwind(|| {
        cubecl::hip::HipRuntime::client(&cubecl::hip::HipDevice::default())
    }).map_err(|_| CubeclError::DeviceUnavailable { backend: "rocm" })
}
```
**Caveat for the planner (RESEARCH Pitfall 1 / Open Q1, A3 = HIGH risk):** a missing CUDA/HIP device may hard-`abort()` below `catch_unwind`. The plan MUST include a Wave-0 spike (`tests/device_absent.rs`) confirming a typed skip, not a crash; if it aborts, probe device availability BEFORE construction (cubecl enumeration API — confirm via `cargo doc`) and never call `client()` for an absent backend. wgpu uses a DIFFERENT init path (`WgpuDevice::DefaultDevice` + possibly `init_setup`/`init_device`/`ComputeClient::load`, RESEARCH lines 501-508) — confirm the 0.10 surface.

---

### `crates/treelite-cubecl/src/error.rs` (model / typed error enum) — MODIFIED (add one variant)

**Analog:** the existing `CubeclError` enum, specifically the simplest variant `Unsupported(String)` (lines 82-88) and the documentation/`thiserror` discipline established for `InvalidInputShape` etc.

**Current enum derives + a representative variant to copy the style from:**
```rust
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CubeclError {
    // …InvalidInputShape / FeatureIndexOutOfBounds / NodeIndexOutOfBounds / MalformedLeafVector…
    #[error("unsupported on the cubecl backend: {0}")]
    Unsupported(String),
}
```

**New variant to add (D-05) — model on the above `#[error(...)]` + doc-comment discipline:**
```rust
/// A backend was compiled in (its cargo feature is enabled) but no device is
/// present at runtime. A typed SKIP the caller branches on (D-05) — the
/// harness/report marks the row "not run — no device"; NEVER a silent CPU
/// fallback (that would hide which backend ran).
#[error("no device available for the {backend} backend (skip, not a failure)")]
DeviceUnavailable {
    /// The selected backend's feature name (`"rocm"` / `"cuda"` / `"wgpu"`).
    backend: &'static str,
},
```
Note: the enum derives `PartialEq, Eq` — `&'static str` keeps that intact (matches the existing variants' comparable fields).

---

### `crates/treelite-cubecl/Cargo.toml` (config) — MODIFIED (add `[features]`)

**Analog:** the existing `[dependencies]` block (lines 7-12, `cubecl = { workspace = true }`). No `[features]` table exists today.

**Add (RESEARCH Standard Stack lines 110-118) — additive features that forward to the cubecl umbrella, default = cpu-only (D-04, Pitfall 5):**
```toml
[features]
default = []
rocm = ["cubecl/rocm"]   # implies cubecl/hip → cubecl::hip::{HipRuntime, HipDevice}
cuda = ["cubecl/cuda"]    # cubecl::cuda::{CudaRuntime, CudaDevice}  (A1 — confirm symbols via cargo doc)
wgpu = ["cubecl/wgpu"]    # cubecl::wgpu::{WgpuRuntime, WgpuDevice}
```

### `Cargo.toml` (root, config) — MODIFIED (leave cpu-only, DO NOT add GPU features here)

**Analog/current:** line 25 `cubecl = { version = "0.10.0", features = ["cpu"] }`.

**Critical (RESEARCH Pitfall 5, lines 405-408):** the workspace `cubecl` features stay `["cpu"]` ONLY. GPU runtimes are forwarded via the `treelite-cubecl` crate feature (`rocm = ["cubecl/rocm"]`), NOT enabled at the workspace level — otherwise a plain `cargo build` would require ROCm system libs and break CI/default builds (D-04). The only possible root edit is documenting this; the line itself should remain cpu-only.

---

### `crates/treelite-harness/src/lib.rs` (service / backend registry) — MODIFIED (registration-not-refactor)

**Analog:** `cubecl_cpu_case()` (lines 157-187) for the `*_case()` constructors, and the `Backend` enum (lines 53-66) for the new variants. The comment at lines 49/65 ALREADY reserves `Cuda`/`Wgpu`/`Rocm` for "Phase 7."

**Current `Backend` enum (the seam to extend, lines 53-66):**
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    ScalarCpu,
    CubeclCpu,
    // Phase 7: Cuda, Wgpu, Rocm — added as further registrations.   <-- this line
}
```
Add `Rocm`, `Cuda`, `Wgpu` variants here (the doc on `CubeclCpu` is the template for their doc-comments).

**Current `cubecl_cpu_case()` — the EXACT template for `rocm_case()`/`cuda_case()`/`wgpu_case()` (lines 157-187):**
```rust
pub fn cubecl_cpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::CubeclCpu,
        dense_f32: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict_cpu::<f32>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())   // f32 RESULT widened — NO pre-cast (Pitfall 6)
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict_cpu::<f64>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
        // Sparse rides the scalar fallback (D-02) — points at treelite_gtil::predict_sparse
        sparse_f32: |model, csr, num_row, cfg| { /* treelite_gtil::predict_sparse::<f32> … */ },
        sparse_f64: |model, csr, num_row, cfg| { /* treelite_gtil::predict_sparse::<f64> … */ },
    }
}
```

**New `rocm_case()` to model on it (RESEARCH Pattern 3, lines 253-275)** — identical except `backend: Backend::Rocm`, dense slots call `treelite_cubecl::predict::<cubecl::hip::HipRuntime, _>`, `#[cfg(feature = "rocm")]`-gated; sparse slots IDENTICAL (scalar fallback, D-02):
```rust
#[cfg(feature = "rocm")]
pub fn rocm_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::Rocm,
        dense_f32: |m, d, n, c| {
            let out = treelite_cubecl::predict::<cubecl::hip::HipRuntime, f32>(m, d, n, c)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |m, d, n, c| treelite_cubecl::predict::<cubecl::hip::HipRuntime, f64>(m, d, n, c)
            .map_err(|e| anyhow::anyhow!("{e}")),
        sparse_f32: /* same as cubecl_cpu_case — scalar fallback */ ,
        sparse_f64: /* same as cubecl_cpu_case — scalar fallback */ ,
    }
}
```
`cuda_case()`/`wgpu_case()` are the same shape with `Backend::Cuda`/`Wgpu` + `CudaRuntime`/`WgpuRuntime`, each `#[cfg(feature = …)]`-gated. The `RunnerCase` struct (95-107) and the four slot type aliases (71-79) are UNCHANGED — this is purely additive (D-11). The harness `Backend` enum widening will force a non-exhaustive `match` somewhere only if a downstream `match` was exhaustive; verify no exhaustive `match Backend` exists that breaks (none in lib.rs).

---

### `crates/treelite-harness/src/report.rs` (utility / markdown+JSON emitter) — NEW

**Analog:** two in-repo sources combined —
1. **The max-deviation accumulation loop** from `tests/gtil_matrix_cubecl.rs::assert_within` (lines 296-324) — BUT in **report mode**: RECORD `max_dev`, do NOT `approx::assert_abs_diff_eq!` panic (D-01 observational). Copy the NaN/inf-structural + finite-`|delta|` logic, drop the hard-gate line 321.
2. **The provenance/manifest fields** from `manifest.rs::Manifest` (lines 36-101) — reuse `backend`, `rustc`, `os`, `arch`, `model`, device name for the report header (D-06).

**Report-mode delta (the one change from `assert_within`):**
```rust
// Source: gtil_matrix_cubecl.rs:296-324 assert_within, with the hard gate removed (D-01)
fn max_abs_delta_report_mode(got: &[f64], want: &[f64]) -> f64 {
    let mut max_dev = 0.0f64;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if g.is_nan() || w.is_nan() || !g.is_finite() || !w.is_finite() { continue; }
        let delta = (g - w).abs();
        if delta > max_dev { max_dev = delta; }
        // NO approx::assert_abs_diff_eq! — the GPU report RECORDS, never fails (D-01).
    }
    max_dev
}
```

**Markdown layout to emit (RESEARCH lines 351-367, D-08)** — columns: model class | postprocessor | ROCm max-|delta| | f64 fallback used? | CUDA | wgpu | predicted band (D-03). CUDA/wgpu render `"not run — no device"` via the `DeviceUnavailable` typed skip. RECOMMENDED machine-readable JSON sidecar `gpu_equivalence.json` (RESEARCH line 371) — low cost, observational, planner discretion.

---

### `crates/treelite-harness/tests/gtil_matrix_gpu.rs` (test) — NEW (sibling-of)

**Analog:** `tests/gtil_matrix_cubecl.rs` (the WHOLE 463-line file). This is a SIBLING (the same pattern that file is itself a sibling of `gtil_matrix.rs`, D-11). Copy the entire structure; the deltas are:

| Element | `gtil_matrix_cubecl.rs` (copy from) | `gtil_matrix_gpu.rs` (new) |
|---------|-------------------------------------|----------------------------|
| Case constructor | `cubecl_cpu_case()` (line 350) | `rocm_case()` (and cuda/wgpu) |
| Attribute | `#[test]` (line 346) | `#[test] #[ignore]` (hardware-gated; developer runs explicitly, D-06) |
| Feature gate | none | `#![cfg(feature = "rocm")]` |
| 1e-5 gate | `assert_within(…, 1e-5)` panics (line 406) | report-mode RECORD; NEVER fail on `>1e-5` (D-01) |
| Device absent | n/a | `Err(DeviceUnavailable)` → mark row "not run — no device", CONTINUE (test passes — skip, D-05) |
| Output | `eprintln!` summary | call `report.rs` to emit `docs/GPU_EQUIVALENCE_REPORT.md` (+ optional JSON) |
| Provenance | `CubeclKernel`/`ScalarFallback` via `model_routes_to_scalar_fallback` (lines 44-51, 338-340) | SAME — reuse `treelite_cubecl::model_routes_to_scalar_fallback` (WR-04) |

The fixture-load helpers (`fixture_path`, `cell_to_f64`, `flatten_output`, `kind_of`, `decode_input_f64`, `frozen_csr`, `run_cell`, `MatrixGolden`/`MatrixManifest` structs — lines 62-292) are copied verbatim (the same "duplicate small helpers, don't `mod`-include" rationale, lines 24-31). Reads the SAME frozen `fixtures/gtil/*.golden.json` (D-07) — never a regenerated vector.

---

### `crates/treelite-gtil/src/postprocessor.rs` (utility / f64 twins) — READ-ONLY REUSE (no edit)

**Analog:** itself. The f64 twins for per-class f64 promotion (D-02) ALREADY exist and are confirmed live:
- `sigmoid` (line 74) / `sigmoid_f64` (line 91)
- `exponential` (103) / `exponential_f64` (109)
- `exponential_standard_ratio` (124) / `_f64` (132)
- `logarithm_one_plus_exp_f64` (152)
- `softmax` (175) / `softmax_f64` (227)
- `signed_square_f64` (273), `multiclass_ova_f64` (312)

These are already dispatched arm-for-arm by `treelite_cubecl::lib.rs` `PredictCpuElem for f32` (84-161) and `for f64` (163-232). D-02 per-class promotion is a DISPATCH choice over these, NOT new math — `RESEARCH Don't-Hand-Roll` (lines 290-294). DO NOT add GPU f64 postprocessor kernels.

---

## Shared Patterns

### thiserror typed errors (D-05, no panic)
**Source:** `crates/treelite-cubecl/src/error.rs` (whole file — `#[derive(Debug, Error, PartialEq, Eq)]` + per-variant `#[error("…")]`)
**Apply to:** the new `DeviceUnavailable` variant. Mirrors the C++/CLAUDE.md "translate `throw` to `Result`" contract; library crates use `thiserror` only, never `anyhow`.

### Registration-not-refactor (D-11, Phase 5)
**Source:** `crates/treelite-harness/src/lib.rs::cubecl_cpu_case` (157-187) + `Backend` enum (53-66) + the matrix sibling pattern (`gtil_matrix_cubecl.rs` being a sibling of `gtil_matrix.rs`)
**Apply to:** `rocm_case`/`cuda_case`/`wgpu_case`, the three `Backend` variants, and `gtil_matrix_gpu.rs`. A new backend = enum variant + `*_case()` constructor + a sibling test. The `RunnerCase` struct, slot type aliases, and the matrix iteration shape are UNCHANGED. Verify `git diff --stat` on `gtil_matrix.rs` / `gtil_matrix_cubecl.rs` stays 0.

### Provenance honesty over green checkmark (D-01 / D-06)
**Source:** `gtil_matrix_cubecl.rs` provenance machinery (lines 44-60 `Provenance` enum, 338-340 the shared-predicate delegation, 405-431 the recorded-from-executed-path tagging) + `manifest.rs::check_manifest` (warns, never fails, lines 110-153)
**Apply to:** `gtil_matrix_gpu.rs` + `report.rs`. The GPU report MEASURES and RECORDS; it does NOT assert a passing tolerance. Reuse `treelite_cubecl::model_routes_to_scalar_fallback` for provenance (WR-04 — never a re-derived parallel copy). No silent CPU fallback (D-05/D-09).

### Additive Cargo features, default-minimal (D-04)
**Source:** root `Cargo.toml` line 25 (`features = ["cpu"]`) + `treelite-cubecl/Cargo.toml` `[dependencies]`
**Apply to:** the new `[features]` table in `treelite-cubecl/Cargo.toml`. GPU runtimes forward via crate features (`rocm = ["cubecl/rocm"]`); workspace stays cpu-only so `cargo build` / `cargo test --workspace` need NO ROCm libs (Pitfall 5).

### Zero-copy SoA upload carries to GPU unchanged
**Source:** `crates/treelite-cubecl/src/upload.rs` — `UploadedForest<R: Runtime>` (line 53) and `pub fn upload_forest<R: Runtime, F: Copy + bytemuck::Pod>(client: &ComputeClient<R>, …)` (line 409). CONFIRMED already generic over `R`; `client.create_from_slice(bytemuck::cast_slice(&col))` works on any `ComputeClient<R>` (lines 432-441).
**Apply to:** nothing — this is the proof the generic path already works. DO NOT edit `upload.rs` or `kernels/*` (the "smell" guard, RESEARCH Anti-Patterns line 282).

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `docs/GPU_EQUIVALENCE_REPORT.md` | doc artifact | n/a | No prior committed report exists. Template is the RESEARCH "Report Emission" markdown layout (07-RESEARCH.md lines 351-367); it is GENERATED by the `gtil_matrix_gpu.rs` run on ROCm hardware, never hand-edited (D-06). Location is Claude's discretion (`docs/` or phase dir). |

All code symbols have an in-repo analog; only the committed report file is purely template-driven.

## Metadata

**Analog search scope:** `crates/treelite-cubecl/{src/lib.rs, src/error.rs, src/upload.rs, Cargo.toml}`, `crates/treelite-harness/{src/lib.rs, src/manifest.rs, tests/gtil_matrix_cubecl.rs}`, `crates/treelite-gtil/src/postprocessor.rs`, root `Cargo.toml`.
**Files scanned:** 9 (read in full or targeted) + grep across upload.rs / postprocessor.rs for signatures.
**Verification status:** All five CONTEXT hints CONFIRMED against live code — (1) `CpuRuntime` is the only concrete-runtime surface, confined to lib.rs (312/444/512/580, plus the 6 `ComputeClient<CpuRuntime>` fn signatures); (2) `UploadedForest<R>`/`upload_forest::<R,F>` already `R`-generic (upload.rs:53/409); (3) `CubeclError` enum present, `Unsupported` is the simplest analog for `DeviceUnavailable`; (4) `Backend`/`RunnerCase`/`cubecl_cpu_case()` present with the Phase-7 reservation comment; (5) f64 postprocessor twins present in postprocessor.rs; (6) manifest provenance + the matrix-sibling `assert_within`/max-dev loop present as the report analog.
**Pattern extraction date:** 2026-06-11

## PATTERN MAPPING COMPLETE

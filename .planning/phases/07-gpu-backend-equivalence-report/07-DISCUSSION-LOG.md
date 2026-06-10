# Phase 7: GPU Backend & Equivalence Report - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 7-gpu-backend-equivalence-report
**Areas discussed:** GPU tolerance budget, Backend selection & device-absent, Equivalence report shape, CPU/GPU crossover heuristic

---

## GPU tolerance budget

### Q1: Report tolerance contract for GPU deviation

| Option | Description | Selected |
|--------|-------------|----------|
| Observed, no gate | Report measured max deviation per model class as DATA; no pass/fail threshold fails CI | ✓ |
| Relaxed GPU tolerance gate | Define an accepted GPU tolerance (1e-4/1e-3) that DOES gate ROCm | |
| Keep 1e-5, document misses | Hold 1e-5; rows that exceed are documented but don't block | |

**User's choice:** Observed, no gate → D-01.

### Q2: When does f64-postprocessor fallback engage

| Option | Description | Selected |
|--------|-------------|----------|
| Per-class, measured | f32 first; promote only the exceeding class's postprocessors to f64 and record it | ✓ |
| Always f64 postproc on GPU | GPU always runs postprocessors in f64 | |
| Never — report f32 as-is | No f64 promotion; report raw f32 divergence | |

**User's choice:** Per-class, measured → D-02.

### Q3: Deviation profiling — research deliverable or in-phase empirical

| Option | Description | Selected |
|--------|-------------|----------|
| Empirical in-phase | Run harness on ROCm; measured numbers ARE the profiling | |
| Research-led analysis | Research profiles cubecl FMA contraction + GPU transcendental divergence before planning, producing a predicted-deviation model the report validates against | ✓ |

**User's choice:** Research-led analysis → D-03.

**Notes:** GPU hardware transcendental/FMA divergence is a different axis from Phase-6 D-04 (in-kernel CPU-runtime precision, already settled). Research predicts; the report measures and validates.

---

## Backend selection & device-absent

### Q1: How to gate and select GPU backends (CPU stays default)

| Option | Description | Selected |
|--------|-------------|----------|
| Per-backend feature + explicit enum | Additive feature each (rocm/cuda/wgpu), default cpu only; caller explicitly picks Backend::Rocm; no auto-detect | ✓ |
| Per-backend feature + auto-detect | Same gating + 'best available' resolver | |
| Single 'gpu' feature | One umbrella feature compiles all GPU runtimes | |

**User's choice:** Per-backend feature + explicit enum → D-04.

### Q2: Behavior when backend compiled but no device present

| Option | Description | Selected |
|--------|-------------|----------|
| Skip, not fail | Row marked 'not run — no device'; caller gets a typed 'no device' result | ✓ |
| Typed error to caller | thiserror DeviceUnavailable; no implicit fallback | |
| Silent fallback to CPU | Auto-route to CpuRuntime when device absent | |

**User's choice:** Skip, not fail (typed result, harness treats as skip) → D-05.

---

## Equivalence report shape

### Q1: How the report is produced / kept honest

| Option | Description | Selected |
|--------|-------------|----------|
| Harness-emitted, committed | Harness emits the report from the per-cell manifest; regenerated on ROCm hardware; numbers can't drift from hand-editing | ✓ |
| Hand-authored from harness output | Developer writes prose manually from deviations | |
| Hybrid: auto table + prose | Harness auto-emits table; developer wraps with hand-written analysis | |

**User's choice:** Harness-emitted, committed → D-06.

### Q2: What defines a 'model class' row

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse harness model set | Existing XGBoost/LightGBM/sklearn fixture classes already validated to 1e-5 on CPU | ✓ |
| By postprocessor/output kind | Rows grouped by postprocessor × predict kind | |
| Both axes (loader × postproc) | Cross-tabulate loader family against postprocessor | |

**User's choice:** Reuse harness model set → D-07.

### Q3: Report location and columns

| Option | Description | Selected |
|--------|-------------|----------|
| docs/ markdown, per-backend cols | Committed markdown; cols: model class \| ROCm max-dev \| f64-fallback used? \| CUDA \| wgpu \| predicted-dev | ✓ |
| Report + machine-readable sidecar | Same markdown + committed JSON/CSV the harness asserts against | |

**User's choice:** docs/ markdown, per-backend cols → D-08. (Sidecar left as optional planner refinement.)

---

## CPU/GPU crossover heuristic

### Q1: Documented-only or enforced in code

| Option | Description | Selected |
|--------|-------------|----------|
| Documented-only | Measure and document; caller (Phase-8 PyO3) decides; no implicit re-routing | ✓ |
| Enforced auto-route | predict() auto-routes inputs below crossover to CPU | |
| Documented + opt-in helper | Document + ship an opt-in helper that applies it | |

**User's choice:** Documented-only → D-09.

### Q2: What input metric defines the threshold

| Option | Description | Selected |
|--------|-------------|----------|
| Measured, report the metric | Benchmark on ROCm; let data pick the dominant metric; document the empirical crossover | ✓ |
| Row count | Fixed: below N rows, prefer CPU | |
| Total work (rows×features×forest) | Crossover on an estimated FLOP/work proxy | |

**User's choice:** Measured, report the metric → D-10.

---

## Claude's Discretion

- Concrete cubecl GPU runtime crates + versions/feature flags for ROCm/CUDA/wgpu.
- Host-launcher generalization mechanics (`CpuRuntime` → `R: Runtime` in `treelite-cubecl/src/lib.rs`).
- Report file location + exact markdown layout; whether to add the machine-readable sidecar.
- Whether GPU determinism (bit-identical) is claimable — state honestly in the report.
- wgpu's role — validated row vs "not run" depends on what the AMD device exposes; determine empirically.

## Deferred Ideas

- Autotuned/optimized GPU kernels — v2 (PERF-v2-02).
- Metal/Vulkan backends beyond wgpu — v2.
- CUDA hardware validation — blocked on NVIDIA hardware.
- Sparse CSR + categorical splits in GPU kernels — scalar fallback persists (Phase-6 D-02).
- f16/bf16 half-precision GPU fast path — Phase 9 / v2.
- Enforced/auto-routing crossover in predict() — explicitly not done (D-09 documented-only).
- Machine-readable report sidecar — optional planner refinement.

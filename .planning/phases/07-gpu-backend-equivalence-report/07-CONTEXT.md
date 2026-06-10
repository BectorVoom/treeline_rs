# Phase 7: GPU Backend & Equivalence Report - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Layer **runtime-selectable GPU backends** onto the green CubeclCpu seam and ship a **committed per-model-class GPU equivalence report** — proving GPU acceleration in v1 *without* making it a 1e-5 correctness prerequisite. The kernels already launch generic over the runtime (`kernels::*::launch::<F, T, R>` in `treelite-cubecl`); the work is to generalize the **host launcher** (currently hardcoded to `CpuRuntime` in `crates/treelite-cubecl/src/lib.rs`) over `R: Runtime`, gate concrete GPU runtimes behind additive Cargo features, register them into the `Backend` enum + `RunnerCase` seam (`treelite-harness`), and emit the deviation report from the existing per-cell manifest machinery.

**ROCm is the hardware-validated backend** (developer has an AMD/ROCm device, no NVIDIA — see memory `gpu-hardware-rocm-only`). CUDA and wgpu share the same generic `R: Runtime` seam and are runtime-selectable, but **CUDA is build-supported only** — its predictions are exercised only where an NVIDIA device exists (CI/elsewhere); a missing CUDA device is a **skip, not a failure**.

**Requirements covered:** GPU-03 (≥1 GPU backend runtime-selectable, produces predictions — satisfied by ROCm), GPU-04 (GPU equivalence report documents deviation per model class within an accepted tolerance).

In scope (HOW, not WHETHER):
- Generalizing the cubecl **host launcher** over `R: Runtime` so the same kernels run on ROCm/CUDA/wgpu.
- Registering `Backend::Rocm` / `Backend::Cuda` / `Backend::Wgpu` into the harness `Backend` enum + `RunnerCase` (the Phase-5/6 registration-not-refactor seam).
- Per-backend additive Cargo features (`rocm`/`cuda`/`wgpu`), default = cpu only.
- Device-absent skip semantics (typed "no device" result).
- A committed, harness-emitted GPU deviation report + a research-led predicted-deviation model.
- A documented (not enforced) CPU/GPU crossover heuristic, measured on ROCm hardware.

Out of scope (this phase):
- **Sparse CSR and categorical splits in cubecl kernels** — still ride the scalar fallback (Phase-6 D-02 deferral carries forward; on GPU they ride the same fallback path).
- **PyO3 binding / numpy marshalling** — Phase 8 (the Phase-8 caller is the documented consumer of the crossover heuristic).
- **Memory-efficiency hardening** (input-buffer bytemuck recast beyond SoA upload, f16/bf16 fast path, custom allocator) — Phase 9 / v2.
- **Autotuned/optimized GPU kernels, Metal/Vulkan backends** — v2 (PERF-v2-02).
- Changing CPU default: **CPU stays the default backend** (SC3, locked).

</domain>

<decisions>
## Implementation Decisions

### GPU tolerance budget & deviation profiling (GPU-04, the research-flagged core)
- **D-01: The GPU equivalence report is observational — measured, not gated.** Report the measured **max deviation per model class** on ROCm as DATA; no pass/fail tolerance threshold fails CI. This mirrors the Phase-6 per-cell-provenance honesty ethos (D-06): the report tells the truth about GPU divergence rather than asserting a green checkmark. The 1e-5 CPU spine (ScalarCpu / CubeclCpu) remains the hard gate and is untouched; the GPU report sits *beside* it, not in place of it.
- **D-02: f64-postprocessor fallback is per-class and measured.** Run f32 postprocessors first on GPU; if a specific model class exceeds the budget, promote **only that class's postprocessors** to f64 and record the promotion in the report. Minimal f64, documented exactly where it was needed (this is the "noting where f64 postprocessor fallback is needed to stay in budget" of SC2). Traversal precision policy follows the Phase-6 in-kernel contract; this decision is about the postprocessor stage.
- **D-03: Deviation profiling is research-led, validated empirically.** The research phase profiles **cubecl FMA contraction behavior + GPU transcendental (exp/exp2/sigmoid/softmax) divergence** in depth before planning, producing a **predicted-deviation model** (an expected per-class deviation range). The committed report's measured ROCm numbers then *validate against* that predicted model. So: research predicts, the report measures, and the report carries both columns. NOTE: this is a DIFFERENT axis from Phase-6 D-04 — in-kernel arithmetic precision on the CPU runtime is settled (memory `cubecl-precision-validated`); GPU *hardware* transcendental/FMA divergence is the genuinely-unknown thing this research quantifies. Do not conflate them or re-litigate D-04.

### Backend selection & device-absent semantics (GPU-03)
- **D-04: Per-backend additive Cargo feature + explicit enum selection, no auto-detect.** One feature each (`rocm`, `cuda`, `wgpu`); default build = cpu only. The caller explicitly selects `Backend::Rocm` / `Backend::Cuda` / `Backend::Wgpu` — there is no "best available" resolver. Selection is deterministic and matches the existing `Backend`-enum registration pattern (Phase 5 D-11). This keeps the harness matrix reproducible and the selected backend always == the backend that ran.
- **D-05: Device-absent is a skip, not a failure — surfaced as a typed result.** When a backend is compiled in but no device is present at runtime (the CUDA-on-this-dev-box case), selecting it returns a **typed "no device" result the caller can branch on** (thiserror variant, e.g. `DeviceUnavailable`); the harness/report treats that as a skip and marks the row **"not run — no device."** No silent fallback to CPU — that would hide which backend actually ran and conflicts with the provenance-honesty ethos (D-01). SC1's "missing CUDA device is a skip, not a failure" is honored literally.

### Equivalence report shape (GPU-04)
- **D-06: The report is harness-emitted and committed, regenerated on ROCm hardware.** The matrix harness emits the report (markdown + table) from the same **per-cell manifest** run that drives equivalence — extending the Phase-6 manifest/provenance machinery. The committed file is *regenerated by running the harness on the developer's ROCm hardware*, so the numbers can never drift from hand-editing. (Pairs with D-08's machine-readable consideration during planning.)
- **D-07: Report rows = the existing equivalence-harness fixture model set.** No new taxonomy — rows mirror the XGBoost / LightGBM / sklearn model classes already in the Phase-5/6 frozen golden matrix, the same classes already validated to 1e-5 on CPU. The report shows how those *same* models deviate on GPU.
- **D-08: Committed markdown report with per-backend columns.** Lives as a committed markdown file in the repo (location is Claude's discretion — `docs/` or the phase dir `.planning/phases/07-.../`). Columns: **model class | ROCm max-deviation | f64-fallback used? | CUDA | wgpu | predicted-deviation (from D-03's research model)**. CUDA and wgpu cells render **"not run — no device"** on this hardware (per D-05). A machine-readable sidecar (JSON/CSV the harness emits and asserts against) is a reasonable planner refinement of D-06 but not mandated here.

### CPU/GPU crossover heuristic (SC3)
- **D-09: The crossover is documented-only — not enforced in code.** Measure and DOCUMENT the crossover point; the caller (notably the Phase-8 PyO3 layer) decides whether to route small inputs to CPU. `predict()` does NOT implicitly re-route a GPU-selected call to CPU below the threshold — explicit backend selection stays literal and honest, consistent with D-05's no-silent-fallback rule. (An opt-in helper that applies the documented heuristic is acceptable if it stays opt-in, but the default path is selection-is-literal.)
- **D-10: The crossover metric is measured, not pre-committed.** Benchmark on ROCm hardware and let the data pick the dominant metric (likely row count or rows×features); document the empirical crossover number and which metric it keys on. Same empirical spirit as the deviation report — don't hardcode a formula up front and then justify it.

### Claude's Discretion (for research/planner)
- **Concrete cubecl GPU runtime crates** backing `Backend::Rocm` / `Backend::Cuda` / `Backend::Wgpu` (cubecl's ROCm/HIP, CUDA, and wgpu runtimes) — confirm exact crate names, versions, and feature flags against the cubecl manual; pin to latest published per project constraint.
- **Host-launcher generalization mechanics** — exactly how `predict_cpu` / `predict_cpu_sparse` and friends in `treelite-cubecl/src/lib.rs` get generalized from `CpuRuntime` to `R: Runtime` (the kernels' `launch::<F,T,R>` already are generic), including where the `ComputeClient<R>` is constructed per backend.
- **Report file location & exact markdown layout** (D-08), and whether to add the machine-readable sidecar (D-06/D-08).
- **Whether determinism (SC2-style bit-identical) is even claimable on GPU** — the report should state it honestly; GPU reductions/FMA may not be bit-reproducible. Surface this in research, don't assume.
- **wgpu's role** — it shares the seam (D-04) and will likely run on the same AMD device via Vulkan/the wgpu backend; whether it's a validated row or a "not run" row depends on what the hardware actually exposes. Determine empirically.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### cubecl authoring + GPU runtime selection (primary references for this phase)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — cubecl kernel authoring + **backend/runtime selection** (generic `R: Runtime`, the ROCm/CUDA/wgpu runtimes, `ComputeClient<R>` construction, FMA/transcendental knobs). The primary reference for generalizing the host launcher and for D-03's FMA-contraction profiling.
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_ARROW_CUBECL.md`, `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_TRANSMUTATION_CUBECL.md` — zero-copy host→device upload patterns; the SoA `as_bytes()` upload (already built in Phase 6) must keep working across the GPU runtimes.
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` — half-precision context (f16/bf16 is Phase-9/v2, OFF this phase's path, but informs how cubecl handles dtypes across runtimes).

### Upstream GTIL (the behavior the GPU path must reproduce / measure deviation against)
- `treelite-mainline/src/gtil/postprocessor.cc` + `treelite-mainline/src/gtil/postprocessor.h` — the postprocessors whose **transcendentals (exp/exp2/sigmoid/softmax) are the GPU divergence hot-spots** (D-03) and the target of the per-class f64 promotion (D-02).
- `treelite-mainline/src/gtil/predict.cc` — traversal + serial tree-sum + f64 base-score add; the numerical sequence the GPU kernels reproduce.

### In-repo assets to extend (the seam Phase 7 plugs into)
- `crates/treelite-cubecl/src/lib.rs` — **the host launcher to generalize.** `predict_cpu` (~294, hardcodes `CpuRuntime::client` ~312), `predict_cpu_sparse` (~327), and the per-kind launchers (`launch::<F,T,CpuRuntime>` at ~444/512/...). The kernels are already runtime-generic; this file is where `CpuRuntime` becomes `R: Runtime`.
- `crates/treelite-cubecl/src/upload.rs` — `UploadedForest<R: Runtime>` + `upload_forest::<R,F>` are ALREADY generic over `R` (lines ~53/409); confirm they work unchanged on GPU runtimes.
- `crates/treelite-cubecl/src/error.rs` — the typed-error surface; add the `DeviceUnavailable`-style variant for D-05.
- `crates/treelite-cubecl/Cargo.toml` — currently `cubecl = { workspace = true }` with cpu only; add per-backend optional deps + the `rocm`/`cuda`/`wgpu` features (D-04).
- Root `Cargo.toml` (line ~25) — `cubecl = { version = "0.10.0", features = ["cpu"] }`; extend with the GPU runtime features/crates (pinned, latest published).
- `crates/treelite-harness/src/lib.rs` — the `Backend` enum (~54, comment at ~49 already names `Cuda`/`Wgpu`/`Rocm` as Phase-7 work), `RunnerCase` (~90), `scalar_cpu_case()` (~113), `cubecl_cpu_case()` (~157). Phase 7 adds `Backend::Rocm`/`Cuda`/`Wgpu` + parallel `*_case()` constructors here (registration-not-refactor).
- `crates/treelite-harness/src/manifest.rs` — the per-cell `Manifest` + provenance (D-06 source-of-truth); the report is emitted from this (D-06/D-08).
- `crates/treelite-harness/tests/gtil_matrix.rs` — the exhaustive matrix runner the new backends register into unchanged.
- `crates/treelite-gtil/src/postprocessor.rs` — the 10 verbatim postprocessors incl. f64 sigmoid/hinge twins; the per-class f64 promotion (D-02) reuses these f64 twins on GPU.

### Prior context (precedent inherited — read before planning)
- `.planning/phases/06-cubecl-gtil-kernels-cpu-backend/06-CONTEXT.md` — D-02 (sparse/categorical scalar-fallback deferral, carries forward), D-04 (in-kernel precision settled — DON'T re-litigate; GPU hardware divergence is a different axis), D-06 (per-cell provenance — the report's machinery), D-11/registration-not-refactor seam.
- `.planning/phases/05-full-scalar-gtil-equivalence-harness/05-CONTEXT.md` — D-10/D-11: generic `R: Runtime` runtime selection + backend-parameterized harness seam this phase extends; scalar-reference-as-measuring-stick.
- `.planning/ROADMAP.md` §"Phase 7" — SC1/SC2/SC3 (the locked WHAT); the research flag (GPU transcendental/FMA divergence profiling; cubecl FMA contraction behavior) → D-03.
- `.planning/REQUIREMENTS.md` — GPU-03, GPU-04 (this phase); GPU-01/02/05 (Phase-6 complete, the green base).
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — SoA/variant pattern, thiserror translation, frozen-golden harness layout.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Runtime-generic kernels** (`treelite-cubecl/src/kernels/*`) — already invoked as `launch::<F, T, R>`; the GPU work is at the *host launcher* layer, not the kernel layer. Major de-risk: the per-row-serial-tree kernel shape (SC1, Phase 6) is unchanged across runtimes.
- **`UploadedForest<R: Runtime>` + `upload_forest::<R, F>`** (`upload.rs`) — already generic over the runtime; the zero-copy SoA upload (GPU-05) should carry to GPU unchanged.
- **`Backend` enum + `RunnerCase` seam** (`treelite-harness/src/lib.rs`) — the comment at line ~49 already reserves `Cuda`/`Wgpu`/`Rocm` for "Phase 7." Adding them is registration, not refactor (Phase 5 D-11).
- **Per-cell manifest/provenance** (`manifest.rs`, Phase-6 D-06) — the honest source-of-truth the deviation report is emitted from (D-06).
- **f64 postprocessor twins** (`treelite-gtil/src/postprocessor.rs`) — reused for the per-class f64 promotion on GPU (D-02).
- **Frozen golden matrix + scalar reference** (Phase 5) — the 1e-5 measuring stick the GPU max-deviation is computed against; same fixtures, same model classes (D-07).

### Established Patterns
- **Registration-not-refactor** (Phase 5 D-11) — new backend = enum variant + `*_case()` constructor; the matrix runner is untouched.
- **Provenance honesty over green checkmark** (Phase 6 D-06) — the GPU report measures and records; it does not assert a passing tolerance (D-01). No silent CPU fallback (D-05/D-09).
- **Additive Cargo features, default-minimal** — cpu is the default build; GPU runtimes are opt-in features (D-04).
- **thiserror typed errors** — device-absence is a typed result, not a panic (D-05).
- **Frozen goldens, CI never regenerates** — the GPU report is regenerated only on real ROCm hardware by the developer (D-06), separate from the CI 1e-5 gate which stays on CPU.

### Integration Points
- **Host launcher** `treelite-cubecl/src/lib.rs`: `CpuRuntime` → `R: Runtime` generalization is the central code change; `ComputeClient<R>` is constructed per selected backend.
- **`Backend::Rocm`/`Cuda`/`Wgpu` + `rocm_case()`/`cuda_case()`/`wgpu_case()`** in `treelite-harness` — the single registration point; routes dense f32/f64 through GPU kernels, sparse/categorical through scalar fallback (carries Phase-6 D-02).
- **Cargo feature plumbing** — root `Cargo.toml` + `treelite-cubecl/Cargo.toml`: per-backend optional cubecl GPU runtime deps behind `rocm`/`cuda`/`wgpu`.
- **Report emission** — from `manifest.rs` provenance → committed markdown (D-06/D-08); regenerated on ROCm hardware.
- **CI stays green on CPU** — the 1e-5 ScalarCpu/CubeclCpu gates are unchanged; the GPU report is an additive, hardware-gated artifact, not a CI gate.

</code_context>

<specifics>
## Specific Ideas

- **Honest measurement beats a green check.** The recurring theme across all four areas: the report MEASURES GPU divergence (D-01), records f64 promotion exactly where used (D-02), marks absent devices "not run" rather than hiding them (D-05), regenerates numbers from the harness rather than hand-editing (D-06), and never silently re-routes a selected backend (D-09). The 1e-5 hard gate lives on CPU; GPU is proven-and-documented, not gated.
- **Research predicts, the report validates.** D-03 wants the FMA/transcendental divergence understood *before* planning (predicted-deviation model), then the committed ROCm numbers validate against that prediction — both appear as columns in the report (D-08). This is the one genuinely-unknown axis; treat it with real research depth.
- **The kernels are already generic — this is a host-launcher + registration + reporting phase.** If planning finds itself rewriting kernels, that's a smell. The `launch::<F,T,R>` sites and `UploadedForest<R>` already prove the runtime-generic path; Phase 7 turns `CpuRuntime` into `R` at the launcher and registers the concrete runtimes.
- **Don't re-litigate Phase-6 D-04.** In-kernel arithmetic precision on the cubecl CPU runtime is settled and validated. GPU *hardware* transcendental/FMA divergence (D-03) is a separate, real unknown — that's what this phase quantifies. Keep the two distinct in research framing.
- **ROCm carries GPU-03 alone.** The developer's AMD/ROCm device is the only validatable GPU here (memory `gpu-hardware-rocm-only`). One green GPU backend (ROCm) satisfies GPU-03; CUDA/wgpu ride the same seam and are documented as "not run — no device" until such hardware exists.

</specifics>

<deferred>
## Deferred Ideas

- **Autotuned / optimized GPU kernels** — v2 (PERF-v2-02). This phase proves correctness-equivalence and a crossover heuristic, not peak throughput.
- **Metal / Vulkan backends beyond wgpu** — v2 (PERF-v2-02). wgpu rides the seam this phase; dedicated Metal/Vulkan runtimes are later.
- **CUDA hardware validation** — blocked on NVIDIA hardware (none locally). CUDA is build-supported this phase; its report rows fill wherever such a device exists (CI/future).
- **Sparse CSR + categorical splits in GPU kernels** — still ride the scalar fallback (Phase-6 D-02 deferral, unchanged here). A later cubecl-coverage pass ports them.
- **f16/bf16 half-precision GPU fast path** — Phase 9 / v2 (PERF-v2-01); off the equivalence path. `HALF_PRECISION_CUBECL.md` noted for when it lands.
- **Enforced/auto-routing crossover in `predict()`** — explicitly NOT done (D-09 is documented-only). If a future phase wants automatic CPU/GPU routing, it builds on the documented heuristic this phase measures.
- **Machine-readable report sidecar (JSON/CSV asserted by the harness)** — considered (D-08); left to planner discretion as a refinement of the harness-emitted markdown (D-06), not mandated.

None of the above is scope creep out of Phase 7 — all recorded so they aren't lost.

</deferred>

---

*Phase: 7-gpu-backend-equivalence-report*
*Context gathered: 2026-06-11*

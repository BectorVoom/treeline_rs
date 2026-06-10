# Phase 6: cubecl GTIL Kernels (CPU Backend) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 6-cubecl-gtil-kernels-cpu-backend
**Areas discussed:** MVP kernel slice scope, Postprocessor execution location, f64 path under cubecl, Fallback honesty / loudness

---

## MVP kernel slice scope

| Option | Description | Selected |
|--------|-------------|----------|
| Numerical-dense core, all 4 kinds | Kernelize numerical-split dense traversal across all 4 predict kinds + leaf-vector broadcast; sparse CSR + categorical on scalar fallback | ✓ |
| Thinnest vertical slice | Kernelize only dense numerical `default`/`raw`; everything else on scalar fallback | |
| Full surface in cubecl | Kernelize everything incl. sparse + categorical; highest risk vs cubecl constraints | |

**User's choice:** Numerical-dense core, all 4 kinds.
**Notes:** Proves cubecl on the dominant inference path; defers the two raggedest data-dependent shapes (sparse, categorical) to a later cubecl-coverage pass rather than abandoning them.

---

## Postprocessor execution location

| Option | Description | Selected |
|--------|-------------|----------|
| Traversal kernel, postproc plain-Rust host | Kernelize traversal only; keep postprocessors on the proven host path; zero fidelity risk but not literal SC1 | |
| Both in cubecl, verbatim cast order | Port traversal AND postprocessors as `#[cube]` kernels reproducing the exact mixed-precision cast order; honors SC1 literally | ✓ |
| Simple postproc in-kernel, mixed-precision on host | Split: elementwise postprocessors in-kernel, delicate mixed-precision ones (softmax, f64 twins) on host | |

**User's choice:** Both in cubecl, verbatim cast order.
**Notes:** A planned follow-up question about a spike-failure contingency was withdrawn after the user clarified that **cubecl in-kernel precision is not a risk** — they have tested it many times, including f64 and mixed-precision. The spike is therefore a confirmation step, not a go/no-go gate. (Saved to memory: `cubecl-precision-validated`.)

---

## f64 path under cubecl

| Option | Description | Selected |
|--------|-------------|----------|
| f64 in-kernel, no fallback | f64 input + `<f64,f64>` preset run in-kernel on CubeclCpu; all 4 input×preset combos kernelized | ✓ (confirmed, not objected) |
| f64 routes to scalar fallback | Only f32 is the proven cubecl path this phase; anything f64 falls back to scalar | |

**User's choice:** f64 in-kernel (resolved implicitly via the precision clarification — the user has tested f64 in cubecl directly, so no f64 fallback is needed). Confirmed by the user proceeding without objection after I stated I'd lock it that way.
**Notes:** Keeps the Phase-8 PyO3 zero-copy numpy `float64` path on real kernels.

---

## Fallback honesty / loudness

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit per-cell provenance in manifest | Each harness cell records `cubecl-kernel` vs `scalar-fallback` in the manifest backend field (per-cell granularity); auditable | ✓ |
| Loud at runtime, summarized at end | Fallback logs a WARNING + end-of-run summary; honest in CI logs but not committed as data | |
| Hard split: separate backend tags | CubeclCpu only claims truly-kernelized cells; fallback cells run under ScalarCpu explicitly | |

**User's choice:** Explicit per-cell provenance in manifest.
**Notes:** Extends the Phase-5 manifest `backend` field (D-09) to per-cell granularity so SC2's "passes on cubecl-cpu" is auditable cell-by-cell — sparse/categorical fallbacks are recorded as data, never hidden.

---

## Claude's Discretion

- Kernel granularity (fused vs per-kind kernels; leaf-vector broadcast mapping).
- Exact ragged-SoA concatenation layout (per-column offset/length bookkeeping), provided SC3's one-handle-per-column / zero-copy holds.
- Which concrete cubecl CPU runtime backs `Backend::CubeclCpu` (subject to SC2 bit-identical determinism).
- Spike scope — minimal kernel confirming control-flow shape (no `continue`), f64 in-kernel, one postprocessor's cast order.

## Deferred Ideas

- Sparse CSR input in cubecl kernels — later cubecl-coverage pass.
- Categorical splits in cubecl kernels — later cubecl-coverage pass.
- GPU backends (CUDA/wgpu/ROCm) + per-model-class equivalence report — Phase 7 (GPU-03/04); ROCm hardware-validated.
- f16/bf16 half-precision in-kernel fast path — v2 PERF-v2-01 / Phase 9.
- Input-buffer bytemuck zero-copy recast beyond the SoA model upload — Phase 9 (MEM-01).

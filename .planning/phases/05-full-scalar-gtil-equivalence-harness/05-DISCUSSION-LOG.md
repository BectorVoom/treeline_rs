# Phase 5: Full Scalar GTIL & Equivalence Harness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 5-full-scalar-gtil-equivalence-harness
**Areas discussed:** Harness coverage matrix, Input dtype scope, Predict API surface, Reproducibility rigor (+ forward backend-selection directive)

---

## Harness Coverage Matrix

| Option | Description | Selected |
|--------|-------------|----------|
| Curated representative | One fixture per capability axis, each run through the kinds/presets it meaningfully exercises | |
| Exhaustive cross-product | Every model type × both presets × all 4 kinds × dense+sparse, all seeded | ✓ |
| Reuse Phase 1-4 goldens + gaps | Existing per-loader goldens as model coverage; add only new-axis fixtures | |

**User's choice:** Exhaustive cross-product
**Notes:** Goldens captured once and frozen, so cost is one-time capture + fixture size + CI assert time — accepted for maximum confidence in the instrument every later phase trusts.

### Sub-question — input scale per cell

| Option | Description | Selected |
|--------|-------------|----------|
| Few seeds, wide rows | ~2-3 seeds/cell, 100-500 rows each, edge values (NaN/±inf/boundary) in one shot | ✓ |
| Many seeds, narrow rows | ~10+ seeds/cell, smaller matrices | |
| You decide | Defer seed/row choice to research/planner | |

**User's choice:** Few seeds, wide rows

### Sub-question — invariance pruning

| Option | Description | Selected |
|--------|-------------|----------|
| Run every cell literally | True exhaustive, no invariance reasoning, every cell captured | ✓ |
| Prune provably-invariant | Skip mathematically-redundant cells with documented logging | |

**User's choice:** Run every cell literally

---

## Input Dtype Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both f32 + f64 input | Mirror upstream generic `Predict<InputT>`, independent of preset | ✓ |
| f32 input only this phase | Keep current f32 path, defer f64 input to Phase 8 | |
| Input must match preset | Constrain input dtype = preset dtype (diverges from upstream) | |

**User's choice:** Both f32 + f64 input
**Notes:** Faithful to upstream's orthogonal generic; adds the input-dtype axis to the exhaustive matrix; lets Phase-8 PyO3 hand numpy float32 and float64 zero-copy.

---

## Predict API Surface

| Option | Description | Selected |
|--------|-------------|----------|
| Idiomatic Rust config struct | `Config { kind: PredictKind, … }` + predict/predict_sparse; JSON-config at PyO3 edge | ✓ |
| Faithful JSON Configuration | Port upstream JSON-config parsing verbatim into the compute crate | |
| Separate fn per kind | predict_default/raw/leaf_id/score_per_tree, 8 entry points | |

**User's choice:** Idiomatic Rust config struct

### Sub-question — output type

| Option | Description | Selected |
|--------|-------------|----------|
| Flat buffer + shape | Flat Vec + Shape descriptor (GetOutputShape per kind); extends current | ✓ |
| Typed shaped output | Richer typed enum/struct per kind | |
| You decide | Defer to research/planner | |

**User's choice:** Flat buffer + shape

---

## Reproducibility Rigor

| Option | Description | Selected |
|--------|-------------|----------|
| Matrices are the contract | Commit actual input matrices + goldens as source of truth; seeds = regen docs; CI never re-draws | ✓ |
| Seeds + script regenerate | Commit only seeds + script; re-derive matrices (fragile across RNG/libm drift) | |

**User's choice:** Matrices are the contract

### Sub-question — manifest fields

| Option | Description | Selected |
|--------|-------------|----------|
| Full provenance + backend | OS/arch, libm/libc, rustc+cubecl, all framework versions, seed, sha256/fixture, `backend` field | ✓ (adopted) |
| Full provenance, no backend field | Same minus backend field | |
| Versions + seed only | Lighter; drops platform-libm provenance + backend axis | |

**User's choice:** Full provenance + backend field (adopted as recommended default after the backend directive; AskUserQuestion was set aside in favor of capturing the directive directly).

---

## Forward Directive — User-Selectable Runtime Backend

**User directive (verbatim intent):** "User can switch cpu/gpu (rocm, wgpu, cuda) backend. Use generics runtime." / "User select feature and switch backend."

**Interpretation captured (CONTEXT.md D-10/D-11):** The end user of treelite-rs selects and switches the compute backend at runtime across `{ scalar-cpu, cubecl-cpu, cuda, wgpu, rocm }`, implemented via cubecl generics over `R: Runtime` (uniform runtime selection, not a recompile). Implementation is Phase 6 (generic seam + cubecl-CPU default) and Phase 7 (runtime-selectable GPU). Phase-5 effect: the equivalence harness is built backend-parameterized with a `backend` field in the manifest; the scalar GTIL is the `scalar-cpu` reference/fallback behind that seam.

**ROCm scope note:** ROCm currently sits in v2 (PERF-v2-02) while wgpu+CUDA are v1 (GPU-03). Flagged as a roadmap-level reconciliation (Deferred Ideas), not a Phase-5 decision. Generic `R: Runtime` keeps adding `cubecl-rocm` a registration rather than a refactor.

---

## Claude's Discretion

- Sparse CSR Rust representation (provided absent = NaN + dense↔sparse parity holds).
- `Config`/`PredictKind` exact fields + module placement (core crate stays JSON-free).
- Harness fixture layout / capture-script structure under `fixtures/`.
- Which concrete models populate each capability axis of the exhaustive matrix.

## Deferred Ideas

- cubecl GTIL kernels generic over `R: Runtime`, CPU default — Phase 6.
- Runtime-selectable GPU backends + per-model-class equivalence report — Phase 7.
- ROCm as v1 user-selectable backend — roadmap-level reconciliation.
- PyO3 JSON-config compatibility shim — Phase 8 edge.
- Memory-efficiency (bytemuck/smallvec/compact_str) for input buffers — Phase 9.

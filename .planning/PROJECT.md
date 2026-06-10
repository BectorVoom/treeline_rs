# treelite-rs

## What This Is

A from-scratch Rust rewrite of [Treelite](https://github.com/dmlc/treelite) — the tree-ensemble model library that imports trained gradient-boosted/forest models (XGBoost, LightGBM, scikit-learn), holds them in a compact in-memory representation, runs reference inference (GTIL), and serializes them. The upstream C++ source (v4.7.0) is vendored read-only at `treelite-mainline/` and is the porting source of truth. The Rust version is a Cargo workspace with strict separation of concerns, a PyO3 Python binding, `cubecl`-accelerated inference, and aggressive memory efficiency — validated to match upstream predictions within 1e-5.

## Core Value

**Predictions match upstream Treelite within 1e-5.** A model loaded and predicted through treelite-rs must produce numerically equivalent output to the C++ original. Everything else (speed, memory, GPU) is secondary to that fidelity.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. The Rust port is greenfield; nothing validated yet.
     The upstream C++ in treelite-mainline/ is the reference SPEC, not validated Rust capability. -->

- [x] Programmatic `ModelBuilder` API with topology/orphan validation — *Validated in Phase 2 (BLD-01/02/03): fluent strict state machine, `ConcatenateModelObjects`, `BulkConstructTree`; XGBoost-JSON loader rewired through it, still within 1e-5.*
- [x] Serialization: binary + JSON round-trip for the current (v5) format generation — *Validated in Phase 2 (SER-01..04): byte-for-byte golden v5 match, bounds-checked v5-only deserialize, zero-copy PyBuffer frames, `DumpAsJSON`, typed field accessors. (XGBoost loader→serialize byte-fidelity deferred to Phase 3 as DEF-02-01.)*

### Active

<!-- v1 scope. All hypotheses until shipped and validated against the 1e-5 equivalence harness. -->

- [ ] Cargo workspace with modular crates, one responsibility each (core model, enums, loaders, builder, GTIL, serialization, Python binding)
- [ ] Import XGBoost models: JSON, UBJSON, and legacy binary formats
- [ ] Import LightGBM text-format models
- [ ] Import scikit-learn estimators — RandomForest, ExtraTrees, GradientBoosting, IsolationForest, AND HistGradientBoosting (incl. the bulk tree-construction path)
- [ ] In-memory struct-of-arrays `Model` / `Tree` representation parameterized over float32/float64
- [ ] GTIL inference: dense + sparse CSR input, the 4 predict kinds, full postprocessor set (sigmoid, softmax, etc.)
- [ ] GTIL inference hot path (tree traversal + postprocessors) implemented as `cubecl` kernels
- [ ] cubecl CPU backend by default; at least one GPU backend (CUDA or wgpu) working and runtime-selectable in v1, with a GPU equivalence report
- [ ] PyO3 Python binding exposing load → predict → serialize directly over the Rust core
- [ ] Equivalence harness: random seeded input matrices → golden output vectors captured from C++ Treelite → assert Rust within 1e-5
- [ ] `thiserror`-based typed errors in library crates; `anyhow` in binaries/tests
- [ ] Memory-efficiency techniques applied (see Context): zero-copy buffers, small-vector/compact-string types, custom allocator

### Out of Scope

<!-- Explicit boundaries with reasoning to prevent re-adding. -->

- **C-API / extern "C" FFI** — explicitly excluded by request; the only language binding is PyO3 over the Rust core.
- **Legacy serialization formats v3.9 and v4.0** — v1 reads/writes the current v5 generation only; multi-version migration deferred.
- **Full cubecl coverage beyond the inference hot path** — loaders, builder, and serialization stay plain idiomatic Rust; only GTIL traversal + postprocessors become cubecl kernels (keeps 1e-5 equivalence low-risk).
- **Live C++ build in CI for equivalence** — golden vectors are generated once from upstream and frozen as fixtures; CI does not compile C++ Treelite.
- **Bit-exact GPU reproducibility** — GPU float reduction ordering may differ; the 1e-5 tolerance absorbs it. Deterministic guarantees apply to the default CPU backend.

## Context

- **Upstream reference:** `treelite-mainline/` (C++17, Treelite 4.7.0). Key subsystems mapped in `.planning/codebase/`: core model (`tree.h`, `contiguous_array.h`), enums, `ModelBuilder`, loaders (`src/model_loader/`), GTIL (`src/gtil/`), serializer (`src/serializer.cc`, `src/json_serializer.cc`). Parallelism upstream is **OpenMP CPU** — there is *no* CUDA in the source; "replace CUDA with cubecl" is interpreted as reimplementing the parallel inference compute as cubecl kernels.
- **Architecture pattern to preserve:** Struct-of-Arrays tree storage; type-erased `Model` over float32/float64 presets (the C++ `std::variant` → Rust `enum`); fluent builder; mixin serializer → trait-based serializer in Rust.
- **CubeCL manual:** `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — primary reference for kernel authoring and backend selection.
- **Optimiser manual:** `/home/user/Documents/workspace/optimisor/manual/` — memory-efficiency playbook. Relevant entries: `ZERO_COPY_ARROW_CUBECL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `HALF_PRECISION_CUBECL.md`, `SMALLVEC_MANUAL.md`, `COMPACT_STR_OPTIMIZATION_EN.md`, `JEMALLOC_MANUAL.md`, `MIMALLOC_MANUAL.md`, `ARROW_DICTIONARY_CASTING.md`, `ARROW_NUMERIC_BRANCHING.md`.
- **Test corpus:** `treelite-mainline/tests/examples/` holds example model files (mushroom, lightgbm, sparse_categorical, etc.) usable as equivalence fixtures; `treelite-mainline/tests/` (C++ and Python) document expected behavior.
- **Naming:** repo dir is `treeline_rs`; the crates/library are named `treelite-rs` to signal a faithful port (workspace members `treelite-core`, `treelite-gtil`, `treelite-py`, …).

## Constraints

- **Tech stack**: Rust (edition 2024) Cargo workspace — modular crates, clear separation of responsibilities.
- **Equivalence**: predictions must match upstream Treelite within **1e-5** (the highest precision upstream targets).
- **Python**: PyO3 module — the sole external binding. No C-API.
- **Error handling**: `thiserror` in library crates, `anyhow` in binaries/tests.
- **Compute**: GTIL inference hot path via `cubecl`; CPU backend default, GPU opt-in.
- **Dependencies**: all crates pinned to their latest published versions.
- **Performance/Memory**: high focus on memory efficiency — zero-copy where possible, compact data structures, custom allocator (jemalloc/mimalloc), optional f16 half-precision via cubecl.
- **Serialization**: current (v5) format generation only for v1.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| cubecl applied to GTIL inference hot path only (not full compute layer) | Lowest risk to 1e-5 equivalence; loaders/serialization gain little from GPU | — Pending |
| cubecl CPU backend default, GPU opt-in | Deterministic, runs in CI without a GPU; GPU is an acceleration opt-in | — Pending |
| Equivalence via golden vectors from C++ over random seeded inputs | Reproducible, no C++ toolchain in CI, broad input coverage beyond canned examples | — Pending |
| Serialize current v5 format only; defer v3.9/v4.0 | Cuts large legacy-migration surface from v1 without hurting core value | — Pending |
| No C-API; PyO3 is the only binding | Explicit user constraint | — Pending |
| Name crates `treelite-rs` (repo stays `treeline_rs`) | Signal a faithful port of the real upstream project (treelite) | — Pending |
| `thiserror` (libs) + `anyhow` (bins/tests) | Typed recoverable errors at API boundaries; ergonomic context at the top level | — Pending |
| HistGradientBoosting included in v1 sklearn scope | User chose full sklearn parity over deferring the most complex loader | — Pending |
| At least one GPU backend validated in v1 (not deferred to v1.x) | User wants GPU acceleration proven in v1, not just CPU cubecl | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-10 — Phase 2 (Builder & Serialization) complete; ModelBuilder + v5 serialization validated.*

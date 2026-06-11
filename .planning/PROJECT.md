# treelite-rs

## What This Is

A from-scratch Rust rewrite of [Treelite](https://github.com/dmlc/treelite) — the tree-ensemble model library that imports trained gradient-boosted/forest models (XGBoost, LightGBM, scikit-learn), holds them in a compact in-memory representation, runs reference inference (GTIL), and serializes them. The upstream C++ source (v4.7.0) is vendored read-only at `treelite-mainline/` and is the porting source of truth. The Rust version is a Cargo workspace with strict separation of concerns, a PyO3 Python binding, `cubecl`-accelerated inference, and aggressive memory efficiency — validated to match upstream predictions within 1e-5.

## Core Value

**Predictions match upstream Treelite within 1e-5.** A model loaded and predicted through treelite-rs must produce numerically equivalent output to the C++ original. Everything else (speed, memory, GPU) is secondary to that fidelity.

## Current Milestone: v1.1 Parallel Scalar Inference

**Goal:** Parallelize the single-threaded scalar GTIL fallback engine across CPU cores — without regressing the 1e-5 equivalence contract — so LightGBM, categorical, and sparse models stop running on one core.

**Target features:**
- Row-parallel `treelite_gtil::predict` (dense) and `predict_sparse` (sparse-CSR) using all available cores.
- A sound, documented `unsafe impl Sync`/`Send` for `Model` justified by read-only-during-predict (mirrors upstream OpenMP); the existing `_assert_not_send` invariant is replaced by the new shareability contract.
- `Config.nthread` honored end-to-end (≤0 = all cores; N = bounded), wiring the existing Python `nthread=` kwarg that is currently recorded-but-unused on the scalar path.

**Why this scope (measured this milestone):** The cubecl CPU kernel path (XGBoost numerical `kLT`) already parallelizes — ~783% CPU (~8/16 cores). The scalar fallback runs at 99% CPU (1 core) and is the whole-model path for *all* LightGBM numerical (`kLE`) models, every categorical / non-`kLT` model, and **all** sparse input. A row-parallel prototype measured 3.0–4.6× there. The cubecl grid-tune (≈8→16 cores) was deliberately deferred as an uncertain incremental win.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. The Rust port is greenfield; nothing validated yet.
     The upstream C++ in treelite-mainline/ is the reference SPEC, not validated Rust capability. -->

- [x] Programmatic `ModelBuilder` API with topology/orphan validation — *Validated in Phase 2 (BLD-01/02/03): fluent strict state machine, `ConcatenateModelObjects`, `BulkConstructTree`; XGBoost-JSON loader rewired through it, still within 1e-5.*
- [x] Serialization: binary + JSON round-trip for the current (v5) format generation — *Validated in Phase 2 (SER-01..04): byte-for-byte golden v5 match, bounds-checked v5-only deserialize, zero-copy PyBuffer frames, `DumpAsJSON`, typed field accessors. (XGBoost loader→serialize byte-fidelity, DEF-02-01, now closed in Phase 3.)*
- [x] Import XGBoost models: JSON, UBJSON, and legacy binary formats — *Validated in Phase 3 (XGB-01..05): all three formats load one logical `binary:logistic` model, predict within 1e-5 of a shared golden, and serialize byte-identically to a single upstream v5 golden blob (closes DEF-02-01 across all three formats). Auto-detect (JSON/UBJSON), NaN/Inf-tolerant numeric path, explicit little-endian legacy decoder (no native-endian transmute), version-gated scalar+vector base_score margin transform. Parse-wide/verify-narrow: categorical/multiclass parsed but their prediction parity deferred to Phase 5. 4 critical untrusted-input hardening findings (panic/overflow/recursion) fixed with regression tests.*
- [x] Import LightGBM text-format models — *Validated in Phase 4 (LGB-01/02/03): numerical + categorical text models load through the f64 `ModelBuilder` and predict within 1e-5 of frozen upstream-GTIL goldens (numerical max |delta| = 0e0, categorical 9.54e-7). Verbatim `BitsetToList` categorical decode, per-field precision (leaf_value/threshold f64, split_gain f32), objective→postprocessor map with sigmoid_alpha + class_id round-robin + average_output.*
- [x] Import scikit-learn estimators — RandomForest, ExtraTrees, GradientBoosting, IsolationForest, AND HistGradientBoosting (incl. the bulk tree-construction path) — *Validated in Phase 4 (SKL-01..04): new `treelite-sklearn` crate mirroring upstream `namespace sklearn` 1:1; RF/ET via bulk path, GB + IsolationForest via f64 MixIn, HistGB via packed-node `from_le_bytes` decode (52/56 itemsize) + features_map/categories_map remap. All within 1e-5 (worst HistGB-categorical 1.19e-7). IsolationForest golden is `treelite.gtil.predict == -score_samples`, not the framework anomaly score.*

- [x] PyO3 Python binding exposing load → predict → serialize directly over the Rust core — *Validated in Phase 8 (PY-01..06, MEM-04): `treelite-py` abi3 maturin wheel; `frontend.load_*` (XGB/LGB) + `sklearn.import_model` (estimator→arrays Python-side) + `Model.serialize/deserialize/dump_as_json/concatenate` + `gtil.predict` (zero-copy numpy borrow, GIL released) all within 1e-5 of upstream `treelite`. Single `TreeliteError` (D-06), panics remapped not aborting (D-07), strict dtype/contiguity + exact feature-count gating (CR-01 fix), additive `backend=` kwarg with `rocm` hardware-validated on-device (bitwise-exact vs cpu). 37 Python tests green.*

- [x] **PAR-01 / PAR-02**: Scalar dense + sparse-CSR predict (`treelite_gtil::predict` / `predict_sparse`) run row-parallel across all cores within 1e-5 — *Validated in Phase 10: the four serial row-loop families converted to rayon `par_chunks_mut`/`map_init` (per-worker scratch), inner per-row tree loop kept serial (GTIL-08). Measured 3.68× speedup on a categorical LightGBM model (4M rows, 16 cores; 0.71s→0.19s), output bit-identical (max |serial−parallel| = 0e0). `gtil_matrix` 1e-5 golden + determinism (dense+sparse) green.*
- [x] **PAR-03**: `Model` soundly shareable across threads for read-only prediction — *Validated in Phase 10: `unsafe impl Sync for Model` + `unsafe impl<T: Copy + Sync> Sync for TreeBuf<T>` with documented read-only-predict SAFETY argument; `requires_sync::<Model>()` contract test supersedes the Phase-9 `_assert_not_send` invariant. Security-verified (T-10-03 closed).*
- [x] **PAR-04**: `Config.nthread` honored end-to-end — *Validated in Phase 10: `run_with_nthread` scoped pool (≤0 = all cores, N bounded, never `build_global`, build failure → typed `GtilError::ThreadPool`); Python `nthread=` kwarg now drives the scalar path (test_nthread.py green).*

### Active

<!-- v1 scope. All hypotheses until shipped and validated against the 1e-5 equivalence harness. -->

- [ ] Cargo workspace with modular crates, one responsibility each (core model, enums, loaders, builder, GTIL, serialization, Python binding)
- [ ] In-memory struct-of-arrays `Model` / `Tree` representation parameterized over float32/float64
- [ ] GTIL inference: dense + sparse CSR input, the 4 predict kinds, full postprocessor set (sigmoid, softmax, etc.)
- [ ] GTIL inference hot path (tree traversal + postprocessors) implemented as `cubecl` kernels
- [ ] cubecl CPU backend by default; at least one GPU backend (ROCm hardware-validated; CUDA/wgpu build-supported) working and runtime-selectable in v1, with a GPU equivalence report
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
- **cubecl CPU grid tuning (v1.1)** — the numerical `kLT` cubecl path already uses ~8/16 cores; pushing its `CubeCount`/`CubeDim` toward full saturation is an uncertain incremental win and is deferred. v1.1 parallelism targets the 1-core scalar fallback only.

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
| ROCm promoted to v1 GPU-03 (with CUDA + wgpu) | User wants end-user runtime backend switching across {scalar-cpu, cubecl-cpu, cuda, wgpu, rocm} via generic `R: Runtime`; ROCm moved out of v2 PERF-v2-02 (2026-06-10) | — Pending |
| ROCm is the v1 hardware-validated GPU backend; CUDA build-supported only | Developer has an AMD/ROCm device, no NVIDIA — ROCm proves GPU-03/GPU-04, CUDA path stays runtime-selectable but is a skip-not-fail where no NVIDIA device exists (2026-06-10) | — Pending |

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
*Last updated: 2026-06-11 — Phase 10 (Parallel Scalar Inference) complete, closing milestone v1.1. PAR-01..04 validated: rayon row-parallel scalar GTIL (dense + sparse CSR) honoring `Config.nthread`, `Model`/`TreeBuf` soundly `Sync`, output bit-identical within 1e-5; measured 3.68× on a categorical LightGBM model (4M rows / 16 cores). Verified 6/6 must-haves + human throughput UAT; code review 5/5 fixed; security 5/5 threats closed (ASVS high). Milestone v1.1 is 100% (1/1 phase). Next: `/gsd-complete-milestone v1.1` to archive. (Note: v1.0 Active requirements still pending migration to Validated — reconcile during milestone completion.)*

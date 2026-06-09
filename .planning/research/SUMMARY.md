# Project Research Summary

**Project:** treelite-rs (from-scratch Rust port of Treelite 4.7.0)
**Domain:** Tree-ensemble model library — load → build → in-memory SoA model → GTIL inference → serialize → PyO3 binding
**Researched:** 2026-06-09
**Confidence:** HIGH

---

## Executive Summary

treelite-rs is a strict numerical port, not a reimagining. The C++ source at `treelite-mainline/` is the specification, and every decision flows from a single non-negotiable invariant: **predictions must match upstream Treelite within 1e-5**. All four research dimensions independently converged on the same implementation order — enums → core model → builder/serialize → loaders (XGBoost first) → scalar GTIL reference + equivalence harness → cubecl GTIL kernels (CPU backend first, validated to 1e-5 before any GPU) → PyO3 binding — because the dependency graph of the upstream C++ leaves no room to reorder. This is not a stylistic preference; building a loader before the builder compiles is impossible, and building cubecl kernels before the scalar reference is validated has historically produced silent 1e-5 regressions that are extremely expensive to debug post-facto.

The principal technical risks all centre on the 1e-5 contract. Three are existential: (1) tree-summation accumulation order — the reference sums trees serially in `tree_id` order per row; any parallel reduction over the tree axis changes float rounding and breaks equivalence on real 500-tree ensembles; (2) postprocessors are intentionally mixed-precision and must be copied verbatim from C++ (the `softmax` `max_margin` is a hardcoded `float` even on the `double` preset path — "fixing" it breaks equivalence); (3) f16 inference carries ~1e-3 error and is flatly incompatible with the 1e-5 contract — it must be an explicit opt-in fast path, never on the equivalence-validated code path. A fourth risk — the HistGradientBoosting scope question — is unresolved and must be answered before the sklearn loader phase is planned.

The cubecl GPU acceleration is the project's performance differentiator, but the architecture correctly treats it as a late-phase concern. The equivalence harness must be green on the CPU cubecl backend before any GPU backend is attempted. GPU floating-point (FMA contraction, transcendental ULP differences) is a separate budget from the CPU equivalence guarantee, and the PROJECT.md explicitly accepts that GPU bit-exactness is out of scope. The development strategy is therefore: build correctness first (CPU path, 1e-5 validated), then layer GPU acceleration on a green test bed.

---

## Key Findings

### Recommended Stack

All versions verified live against the crates.io sparse index on 2026-06-09; PyO3/maturin confirmed via Context7. The stack is deliberately conservative — pre-release crates are explicitly rejected wherever they appear on the critical path.

**Core technologies:**

| Technology | Version | Purpose |
|------------|---------|---------|
| Rust edition | 2024 | Language edition (PROJECT.md constraint) |
| Cargo resolver | "3" | Edition-2024 default, MSRV-aware |
| `cubecl` | **0.10.0** (no default backend — features: `cpu`, `cuda`, `wgpu`, `rocm`) | GTIL inference hot path; CPU enabled by default via feature, GPU opt-in |
| `serde_json` | **1.0.150** | XGBoost JSON loader — preferred over `simd-json` (mutation, unsafe, float-faithfulness concerns) |
| Hand-rolled UBJSON | — | XGBoost UBJSON — the only `ubjson` crate on crates.io (`0.1.0`) is unmaintained and disqualified |
| Hand-rolled LightGBM text | — | `getline`/`strtod`-style line parser; no grammar crate needed |
| `pyo3` | **0.28.3** (`extension-module`, `abi3-py38`) | Python binding |
| `maturin` | **1.13.3** | Wheel build/package tool |
| `thiserror` | **2.0.18** | Typed errors in library crates |
| `anyhow` | **1.0.102** | Ergonomic context in binaries/tests |
| `bytemuck` | **1.25.0** | Zero-copy `&[T] ↔ &[u8]` bridge; Pod/Zeroable for SoA columns and cubecl upload |
| `smallvec` | **1.15.1** (NOT 2.0-alpha) | Inline-small node children/scratch vecs |
| `compact_str` | **0.9.1** | SSO strings for feature names/metadata |
| `half` | **2.7.1** (`bytemuck`, `num-traits`) | Host-side f16/bf16 (opt-in only; never on equivalence path) |
| `arrow` (arrow-rs) | **58.3.0** | Zero-copy input matrices and Python buffer interop (NOT for Tree SoA storage) |
| `tikv-jemallocator` | **0.7.0** | Recommended global allocator (maintained fork; mimalloc as portability fallback) |
| `proptest` | **1.11.0** | Seeded input-matrix generation for equivalence harness |
| `approx` | **0.5.1** (NOT 0.6.0-rc2) | `assert_abs_diff_eq!` with `epsilon = 1e-5` |
| `criterion` | **0.8.2** | Inference/loader benchmarks |
| `insta` | **1.47.2** | Snapshot tests for serializer structural round-trips (not float vectors) |

**Critical stack notes:**
- cubecl 0.10.0 ships with **no default backend**. The `treelite-gtil` crate declares `default = ["cpu"]` in its own `[features]` so the CPU backend compiles by default without any global feature leak.
- `simd-json` is rejected: it mutates input, its float parse is less predictably bit-faithful for a 1e-5 port, and loaders are not the hot path.
- Hand-rolling UBJSON over `byteorder` is required (not optional) because sharing the JSON/UBJSON state machine guarantees numeric parity by construction — mirroring upstream's `DelegatedHandler` design.
- `smallvec` 2.0.0-alpha and `approx` 0.6.0-rc2 are pre-release and must not be used.
- f16 on the CPU backend is a storage/bandwidth optimization only; f16 accumulation carries ~1e-3 error and violates the 1e-5 contract.

---

### Expected Features

All features verified against `treelite-mainline/` headers and source files on 2026-06-09.

**Must have — table stakes (required for 1e-5 equivalence or basic load→predict→serialize usability):**

- `Model` enum over two and only two presets: `<f32,f32>` and `<f64,f64>` — upstream `static_assert` forbids mixed types
- Struct-of-arrays `Tree<T>` with ~20 parallel `ContiguousArray` columns (`cleft_`, `cright_`, `split_index_`, `threshold_`, `leaf_value_`, `default_left_`, `cmp_`, `category_list_right_child_`, etc.)
- 3 node types (`kLeafNode`, `kNumericalTestNode`, `kCategoricalTestNode`) and 6 comparison operators (`kNone`, `kEQ`, `kLT`, `kLE`, `kGT`, `kGE`) — verified from `enum/`
- Missing-value routing: NaN-only trigger via `default_left_`/`DefaultChild` — not zero, not absent-from-sparse
- Categorical splits with exact float-representability guard and `category_list_right_child_` polarity
- Full header metadata: `num_feature`, `task_type`, `average_tree_output`, `num_target`, `num_class[]`, `leaf_vector_shape[2]`, `target_id[]`/`class_id[]`, `postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores[]`, `attributes`
- `ModelBuilder` fluent API with orphan/topology validation and `ConcatenateModelObjects`
- XGBoost loaders: JSON (SAX/streaming), UBJSON (hand-rolled, shared state machine), legacy binary (LE struct layout, 136-byte `LearnerModelParam`)
- XGBoost objective → postprocessor mapping + base_score margin-transform (version-gated: `major_version >= 1`)
- LightGBM text-format loader + categorical bitset decode + per-field precision (leaf_value/threshold = f64, split_gain = f32)
- LightGBM `class_id[i] = i % num_class` round-robin + `average_output` flag
- scikit-learn array loaders: `LoadRandomForest{Regressor,Classifier}`, `LoadGradientBoosting{Regressor,Classifier}`, `LoadIsolationForest` + `BulkConstructTree` fast path
- GTIL: 4 predict kinds (`kPredictDefault`, `kPredictRaw`, `kPredictLeafID`, `kPredictPerTree`) — verified from `gtil.h`
- GTIL: dense row-major + sparse CSR inputs (CSR materializes dense row prefilled with NaN)
- GTIL: all 10 postprocessors: `identity`, `signed_square`, `hinge`, `sigmoid`, `exponential`, `exponential_standard_ratio` (exp2, ratio_c), `logarithm_one_plus_exp` (log1p), `identity_multiclass`, `softmax` (mixed-precision — see critical risk), `multiclass_ova`
- GTIL accumulation order: **serial tree sum in `tree_id` order per row** — row-parallel only; no tree-axis reduction
- GTIL output shaping: `GetOutputShape` per predict kind; leaf-vector broadcast (4 cases); tree averaging; f64 base-score addition
- Serialization: v5 binary round-trip, v5 PyBuffer zero-copy, `DumpAsJSON`, field accessors
- PyO3 binding mirroring `frontend.py`, `gtil/gtil.py`, `model.py`, `sklearn/importer.py`
- Seeded golden-vector equivalence harness (frozen C++ fixtures, `approx` at `epsilon = 1e-5`)

**Should have — differentiators:**

- cubecl-accelerated GTIL hot path (tree traversal + postprocessors as `#[cube(launch)]` kernels generic over `R: Runtime`)
- CPU cubecl backend default; GPU (CUDA/wgpu/ROCm) opt-in via Cargo features at runtime
- Memory efficiency: `TreeBuf<T>` zero-copy borrowed column mode, `bytemuck` recast, `smallvec`/`compact_str`, custom allocator
- Typed `thiserror` errors at crate boundaries

**Defer to v1.x:**

- HistGradientBoosting loaders — scope confirmation required (see Gaps)
- f16 half-precision inference opt-in fast path
- Memory-efficiency hardening sweep

**Defer to v2+:**

- Legacy serialization formats v3.9/v4.0 + cross-version migration
- Additional GPU backend tuning / autotuned kernels

---

### Architecture Approach

The workspace is a strict DAG of 8–9 crates, each with a single responsibility, all edges pointing downward toward `treelite-enum` (the dependency-free vocabulary root). No crate outside `treelite-gtil` ever touches cubecl; no crate outside `treelite-py` ever links libpython. This containment is what keeps the 1e-5 equivalence risk surface bounded.

**Major components:**

| Crate | Responsibility | Key pattern |
|-------|----------------|-------------|
| `treelite-enum` | `TaskType`, `TreeNodeType`, `Operator`, `DType` + conversions | Zero deps; everything imports this |
| `treelite-core` | `Model`/`ModelPreset` enum/`Tree<T>` SoA/`TreeBuf<T>`/typed errors | Single `T: TreeFloat` param (not `<T,L>`) |
| `treelite-builder` | Fluent runtime-validated `ModelBuilder` + `ConcatenateModelObjects` | Runtime-validated (not typestate) |
| `treelite-loader-xgboost` | JSON (SAX/streaming), UBJSON (hand-rolled shared state machine), legacy binary | Shared delegated-handler for JSON/UBJSON parity |
| `treelite-loader-lightgbm` | Line-oriented text parser; categorical bitset decode; objective mapping | Hand-rolled; per-field precision must match C++ |
| `treelite-loader-sklearn` | Array-dump loaders; `BulkConstructTree` bulk path | Feature-gated bypass (replaces C++ `friend`) |
| `treelite-serialize` | Trait-based `Sink`/`Source`; v5 binary + PyBuffer + JSON dump | Core never imports serializer |
| `treelite-gtil` | Tree traversal + postprocessors as cubecl kernels; `Backend` enum; output shaping | Only cubecl-dependent crate; `default = ["cpu"]` |
| `treelite-py` | PyO3 `cdylib`; numpy/buffer-protocol I/O; sklearn marshalling | Sole C-ABI crate; `abi3-py38` |
| `treelite-rs` (opt) | Re-export umbrella | Pure re-exports |

**Three architectural invariants that must not be violated:**
1. Enum dispatch for `Model` (not generics, not trait objects) — `match` once, monomorphized inner loop
2. Struct-of-arrays `TreeBuf<T>` columns (not `Vec<Node>` AoS) — required for cubecl upload and serializer zero-copy
3. Row-parallel, tree-serial accumulation in all GTIL compute paths — this is a correctness requirement, not a performance choice

---

### Critical Pitfalls

All pitfalls verified directly against `treelite-mainline/src/gtil/*.cc` and `model_loader/*`.

1. **Tree-summation accumulation order (CRITICAL)** — The reference sums trees serially in `tree_id` order per row. Any cross-tree parallel reduction changes float rounding; on real 500-tree f32 ensembles divergence exceeds 1e-5. In the cubecl kernel: one unit = one row, loop over trees inside. Never use `atomicAdd` or `cubecl-reduce` over the tree axis.

2. **Mixed-precision postprocessors must be copied verbatim (CRITICAL)** — `softmax` uses hardcoded `float max_margin` and `double norm_const` even on the `<f64,f64>` preset, then narrows with `static_cast<float>`. "Cleaning" this breaks multiclass equivalence. `exponential_standard_ratio` uses `exp2` (not `exp`); `logarithm_one_plus_exp` uses `log1p`. Port and unit-test each postprocessor against scalar C++ output before wiring into the predict path.

3. **NaN-only missing-value routing (CRITICAL)** — Only `std::isnan(fvalue)` triggers the default-direction path. For sparse CSR, absent columns must materialize as NaN (not 0). Dense and sparse paths for identical logical data must produce identical output.

4. **f16 is NOT 1e-5-compliant (CRITICAL)** — f16 carries ~1e-3 error. It must be an explicit opt-in fast path gated behind feature detection; the equivalence harness must never run in f16 mode.

5. **XGBoost base_score margin transform (version-gated, CRITICAL)** — For legacy binary with `major_version >= 1`, stored `base_score` is in probability space and must be inverted. Missing this produces a constant offset on every prediction. Base scores stored as `f64` regardless of preset.

6. **Categorical-split polarity + float-representability guard** — `NextNodeCategorical` has a non-obvious guard (`fvalue >= 0` AND `fvalue <= min(uint32::MAX, 2^mantissa_digits)`) and a `category_list_right_child_` polarity flag easy to invert. Port literally.

7. **cubecl control-flow constraints** — `continue` unsupported in `#[cube]`; normal Rust functions cannot be called inside a `#[cube]` without themselves being `#[cube]`. Use `while` + `break`, annotate all helpers `#[cube]`. Have a plain-Rust fallback so the project is not blocked by a cubecl gap.

8. **Reproducible golden-vector generation** — Store actual input matrices (not just seeds) as fixtures; record the C++ toolchain, libm, and framework versions in a committed manifest. Capture goldens per preset and per predict kind. Build the harness skeleton early.

---

## Implications for Roadmap

The dependency DAG of the upstream C++ mandates this phase order. All four research files converged on it independently.

### Phase 1: Workspace Bootstrap + Enums
**Rationale:** `treelite-enum` is the dependency root. Workspace version pinning and edition 2024 validation must be verified before any functional code.
**Delivers:** Compilable workspace; `treelite-enum` with `TaskType`/`TreeNodeType`/`Operator`/`DType`; shared `[workspace.dependencies]` with all pinned versions.
**Avoids:** Pitfall 19 (version pinning), Pitfall 20 (edition 2024 — validate cubecl/pyo3 compile under 2024 now, not mid-port)
**Research flag:** Standard patterns — skip.

### Phase 2: Core Model (`treelite-core`)
**Rationale:** Every later crate depends on this. The `ModelPreset` variant representation and SoA column set must be frozen here — the serializer's wire format and cubecl upload path both depend on this layout.
**Delivers:** `ModelPreset::{F32,F64}` enum; `Tree<T: TreeFloat>` with all ~20 SoA `TreeBuf<T>` columns; full header metadata struct; `thiserror` error enum; `TreeBuf<T>` in Owned/Borrowed modes.
**Avoids:** Pitfall 8 (threshold typing — single `T` param, never promote), Pitfall 15 (foreign-buffer lifetime)
**Research flag:** Standard patterns — skip.

### Phase 3: Builder + Serializer (parallel)
**Rationale:** Both depend only on `treelite-core` and can be built in parallel. Builder must land before any loader. Serializer provides early round-trip fixtures.
**Delivers:** Runtime-validated `ModelBuilder` + `ConcatenateModelObjects` + `BulkConstructTree` bulk path; trait-based `Sink`/`Source`; v5 binary + PyBuffer + `DumpAsJSON` + field accessors.
**Avoids:** Anti-Pattern 3 (core must not import serializer), Pitfall 12 (bulk path decision documented)
**Research flag:** Standard patterns — skip.

### Phase 4: XGBoost Loaders
**Rationale:** Richest fixture set; establishes the SAX/streaming JSON pattern and the hand-rolled UBJSON shared state machine. Three format variants exercise the most loader code paths.
**Delivers:** `treelite-loader-xgboost`; JSON/UBJSON/legacy binary; `DetectXGBoostFormat`; objective→postprocessor mapping; base_score margin-transform (version-gated).
**Avoids:** Pitfall 5 (base_score transform), Pitfall 9 (legacy binary endianness — `from_le_bytes`, not struct transmute), Pitfall 10 (NaN/Inf in JSON/UBJSON)
**Research flag:** Needs research-phase — SAX/streaming serde_json; UBJSON type-tag decoding; NaN/Inf extension; base_score version gate.

### Phase 5: LightGBM Loader
**Rationale:** Independent of XGBoost; exercises the categorical bitset decode path.
**Delivers:** `treelite-loader-lightgbm`; categorical bitset decode; per-field precision; objective→postprocessor+alpha mapping; `class_id = i % num_class` round-robin; `average_output` flag.
**Avoids:** Pitfall 4 (categorical normalization), Pitfall 8 (parse precision), Pitfall 11 (text quirks, alpha, empty threshold)
**Research flag:** Standard patterns — skip (upstream `lightgbm.cc` is the complete spec).

### Phase 6: scikit-learn Loaders
**Rationale:** Depends on `BulkConstructTree` (Phase 3). HistGradientBoosting scope must be confirmed before planning this phase.
**Delivers:** `treelite-loader-sklearn`; RF/ET/GBM/IsolationForest array loaders; PyO3 `sklearn.import_model` marshalling.
**Avoids:** Pitfall 12 (f64 preset for sklearn, bulk path decision)
**Unresolved:** HistGradientBoosting scope — confirm v1 vs v1.x before planning.
**Research flag:** Needs research-phase if HistGB is in scope; standard patterns otherwise.

### Phase 7: Scalar GTIL Reference + Equivalence Harness
**Rationale:** The scalar GTIL is the 1e-5 baseline that validates everything after. The harness must be built simultaneously — it is the measurement instrument. Postprocessors unit-tested first.
**Delivers:** Scalar `EvaluateTree` + leaf accumulation + averaging + base-score + all 10 postprocessors; `GetOutputShape`; dense + sparse CSR; all 4 predict kinds; frozen golden-vector fixtures (per preset, per predict kind); `proptest` + `approx` harness at 1e-5.
**Implements:** Row-parallel, tree-serial accumulation; NaN-only missing-value routing; `NextNodeCategorical` with float-representability guard.
**Avoids:** Pitfalls 1, 2, 3, 4, 17, 18, 21 (all the 1e-5-critical risks)
**Research flag:** Needs research-phase — leaf-vector broadcast (4 cases); mixed-precision softmax details; cubecl control-flow constraints spike before kernel authoring.

### Phase 8: cubecl GTIL Kernels (CPU Backend First)
**Rationale:** Built on the green scalar reference. CPU backend validated to 1e-5 before any GPU backend is attempted. "Row-parallel, tree-serial" contract from Phase 7 maps directly to kernel topology.
**Delivers:** `#[cube(launch)]` traverse + postprocessor kernels generic over `R: Runtime`; `Backend` enum dispatch; ragged-SoA column concatenation; zero-copy host→device via `TreeBuf::as_bytes()` + `client.create_from_slice`; CPU backend validated to 1e-5 in CI.
**Avoids:** Pitfall 1 (no `atomicAdd`/`cubecl-reduce` over tree axis), Pitfall 14 (no `continue` in `#[cube]`), Pitfall 6 (CPU is the validated baseline; GPU is follow-on)
**Research flag:** Needs research-phase — data-dependent branching kernel shape is unusual; ragged-SoA concatenation design; spike needed.

### Phase 9: PyO3 Binding
**Rationale:** Depends on all crates above. Must be last functional crate.
**Delivers:** `treelite` Python module; load/predict/serialize/dump API; numpy/buffer-protocol zero-copy I/O; `sklearn.import_model`; `abi3-py38` wheel.
**Avoids:** Pitfall 16 (GIL release, `PyReadonlyArray`, `From<Error> for PyErr`, `catch_unwind`), Pitfall 22 (allocator in cdylib)
**Research flag:** Needs research-phase — PyO3 0.28 buffer-protocol implementation, numpy zero-copy return, GIL/threading pattern.

### Phase 10: GPU Backends (opt-in, post-equivalence)
**Rationale:** Layered onto green CPU equivalence. Not a v1 blocker.
**Delivers:** CUDA/wgpu opt-in backends; GPU equivalence report per model class; crossover threshold heuristic.
**Avoids:** Pitfall 6 (FMA/transcendental budget), Pitfall 13 (launch overhead), Pitfall 23 (denormals)
**Research flag:** Needs research-phase — GPU transcendental divergence; cubecl FMA contraction behaviour.

---

### Phase Ordering Rationale

- Enums → core is non-negotiable: vocabulary root frozen first, keystone second.
- Builder and serializer before any loader: builder is a loader prerequisite; serializer provides early fixtures.
- XGBoost before LightGBM before sklearn: richest fixture set, most format variants, establishes patterns.
- Scalar GTIL + harness before cubecl kernels: the scalar reference is the correctness definition; kernels are verified against it.
- CPU cubecl before GPU cubecl: deterministic default before non-deterministic optimization.
- PyO3 last: depends on everything; Python concerns added only after correctness is established.

---

### Research Flags

Needs `gsd-plan-phase --research-phase`:
- **Phase 4 (XGBoost loaders):** SAX/streaming serde_json; UBJSON type-tag decoding; NaN/Inf extension; base_score version gate.
- **Phase 7 (Scalar GTIL + harness):** leaf-vector broadcast; mixed-precision softmax; cubecl control-flow spike.
- **Phase 8 (cubecl kernels):** data-dependent branching kernel shape; ragged-SoA concatenation.
- **Phase 9 (PyO3):** buffer-protocol; numpy zero-copy return; GIL/threading.
- **Phase 10 (GPU):** transcendental divergence profiling; FMA contraction.
- **Phase 6 (sklearn):** only if HistGradientBoosting confirmed in v1 scope.

Standard patterns (skip research-phase): Phases 1, 2, 3, 5.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified live against crates.io sparse index 2026-06-09; PyO3/maturin via Context7. |
| Features | HIGH | Enums, predict kinds, postprocessors, model fields, loader signatures all verified against vendored C++ headers and sources. |
| Architecture | HIGH (crate split, variant, SoA, PyO3, build order) / MEDIUM (cubecl kernel ergonomics for tree traversal) | Grounded in upstream headers and verified cubecl/PyO3 docs; tree-traversal kernel is unusual shape vs manual examples. |
| Pitfalls | HIGH (float/traversal — verified against source) / MEDIUM (cubecl-specific — pre-1.0, evolving) | 1e-5 risks verified against actual reference implementation. |

**Overall confidence: HIGH** for the non-cubecl path; **MEDIUM-HIGH** overall given cubecl's pre-1.0 status.

### Gaps to Address

- **HistGradientBoosting scope (UNRESOLVED — answer before Phase 6):** PROJECT.md lists "scikit-learn estimators" generally. Research recommends deferring HistGB to v1.x (most complex sklearn path: packed node structs, `_bin_mapper`, version-gated `_preprocessor`). Must be confirmed with milestone owner. If in v1 scope, Phase 6 needs a research-phase and significantly more time.
- **cubecl tree-traversal kernel ergonomics:** API shape is verified but tree traversal (data-dependent branching) is the opposite of the documented matmul/axpy examples. A kernel spike before Phase 8 planning is recommended.
- **libm platform consistency for golden vectors:** C++ goldens embed platform `exp`/`log1p`/`exp2` results. The golden-generation manifest must record the exact toolchain; the harness should track actual max deviation (not just pass/fail) to detect libm drift.
- **serde_json NaN/Inf extension:** XGBoost JSON uses rapidjson's `kParseNanAndInfFlag`. serde_json rejects NaN/Inf by default. The XGBoost JSON loader phase must resolve this (custom deserializer or parser configuration).

---

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/src/gtil/predict.cc` — traversal, missing-value, accumulation order, sparse NaN-fill, averaging, base_score
- `treelite-mainline/src/gtil/postprocessor.cc` — all 10 postprocessors, mixed-precision softmax
- `treelite-mainline/include/treelite/tree.h` — Model/ModelPreset/Tree fields, SoA arrays, version triple
- `treelite-mainline/include/treelite/gtil.h` — PredictKind enum, Predict/PredictSparse/GetOutputShape
- `treelite-mainline/include/treelite/model_builder.h` — builder API, Metadata, TreeAnnotation, PostProcessorFunc
- `treelite-mainline/include/treelite/model_loader.h` — loader signatures
- `treelite-mainline/include/treelite/enum/{task_type,tree_node_type,operator}.h` — enum values
- `treelite-mainline/src/model_loader/xgboost_legacy.cc`, `detail/xgboost.cc` — struct layout, base_score transform
- `treelite-mainline/src/model_loader/lightgbm.cc` — text parse, per-field precision, objective/alpha, round-robin
- `treelite-mainline/python/treelite/{model,frontend}.py`, `gtil/gtil.py`, `sklearn/importer.py` — Python API
- Context7 `/pyo3/pyo3` — PyO3 0.28.3 API
- Context7 `/pyo3/maturin` — maturin 1.13.3
- Context7 `/tracel-ai/cubecl` — cubecl runtime feature flags, kernel authoring, `continue` unsupported
- crates.io sparse index (2026-06-09) — all pinned versions verified live

### Secondary (MEDIUM confidence)
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` — f16 ~1e-3 tolerance
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_TRANSMUTATION_CUBECL.md` — bytemuck alignment, Bytes copy semantics
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_ARROW_CUBECL.md` — Arrow zero-copy (rejected for Tree SoA)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — control-flow constraints, CPU backend state

### Tertiary (LOW confidence — needs validation)
- cubecl pre-1.0 stability and CPU-backend completeness — some ops unimplemented; pin exactly and spike early

---

*Research completed: 2026-06-09*
*Ready for roadmap: yes*

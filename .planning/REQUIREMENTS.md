# Requirements: treelite-rs

**Defined:** 2026-06-09
**Core Value:** Predictions match upstream Treelite within 1e-5.

## v1 Requirements

Requirements for the initial release. Each maps to a roadmap phase. All are hypotheses until shipped and validated against the equivalence harness.

### Foundations

- [x] **FND-01**: Cargo workspace (edition 2024, resolver "3") builds all member crates from a single pinned `[workspace.dependencies]` table
- [x] **FND-02**: All third-party crates pinned to current latest-stable versions, with no pre-release crate on the critical path

### Vocabulary / Enums

- [x] **ENUM-01**: `TaskType`, `TreeNodeType`, `Operator`, and `DType` enums exist with string conversions matching upstream values

### Core Model

- [x] **CORE-01**: A `Model` is represented as a two-variant enum over `<f32,f32>` and `<f64,f64>` presets (no mixed types)
- [x] **CORE-02**: A `Tree<T>` stores all upstream node fields as parallel struct-of-arrays columns
- [x] **CORE-03**: Tree column storage (`TreeBuf<T>`) supports both owned and zero-copy borrowed (foreign-buffer) modes
- [x] **CORE-04**: A model carries full header metadata (`num_feature`, `task_type`, `num_class`, `leaf_vector_shape`, `target_id`/`class_id`, `postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores`, `average_tree_output`, attributes)

### Model Builder

- [x] **BLD-01**: A fluent `ModelBuilder` constructs a model node-by-node with orphan/topology validation
- [x] **BLD-02**: `ConcatenateModelObjects` merges multiple models into one
- [x] **BLD-03**: A `BulkConstructTree` fast path builds trees from pre-validated bulk input

### XGBoost Loader

- [x] **XGB-01**: User can load an XGBoost JSON model
- [x] **XGB-02**: User can load an XGBoost UBJSON model (parser shares the JSON state machine for numeric parity)
- [x] **XGB-03**: User can load an XGBoost legacy binary model (little-endian layout)
- [x] **XGB-04**: The loader auto-detects which XGBoost format a file uses
- [x] **XGB-05**: XGBoost objective maps to the correct postprocessor, with the version-gated `base_score` margin transform applied

### LightGBM Loader

- [x] **LGB-01**: User can load a LightGBM text-format model
- [x] **LGB-02**: Categorical splits decode correctly (bitset) with upstream-matching per-field precision
- [x] **LGB-03**: LightGBM objective maps to the correct postprocessor (+ `sigmoid_alpha`), with `class_id` round-robin and `average_output` honored

### scikit-learn Loader

- [x] **SKL-01**: User can import `RandomForest` and `ExtraTrees` (classifier + regressor)
- [x] **SKL-02**: User can import `GradientBoosting` (classifier + regressor)
- [x] **SKL-03**: User can import `IsolationForest`
- [x] **SKL-04**: User can import `HistGradientBoosting` (classifier + regressor)

### GTIL Inference

- [x] **GTIL-01**: User can predict over a dense row-major input matrix
- [x] **GTIL-02**: User can predict over a sparse CSR input matrix (absent entries treated as NaN)
- [x] **GTIL-03**: All four predict kinds are supported (`default`, `raw`, `leaf_id`, `score_per_tree`)
- [x] **GTIL-04**: All ten postprocessors are ported verbatim, preserving upstream mixed-precision arithmetic
- [x] **GTIL-05**: Missing-value routing fires on NaN only, via the node default direction
- [x] **GTIL-06**: Categorical-split evaluation applies the float-representability guard and correct child polarity
- [x] **GTIL-07**: Output shaping is correct — `GetOutputShape`, leaf-vector broadcast, tree averaging, and `f64` base-score addition
- [x] **GTIL-08**: Per-row tree summation is serial in `tree_id` order (parallelism only across rows)

### cubecl Compute & GPU

- [x] **GPU-01**: The GTIL inference hot path (traversal + postprocessors) is implemented as cubecl kernels
- [x] **GPU-02**: The cubecl CPU backend is the default and is validated to 1e-5
- [x] **GPU-03**: At least one GPU backend (ROCm, wgpu, or CUDA) is runtime-selectable and produces predictions. ROCm is the hardware-validated backend (developer's AMD device); CUDA is build-supported but validated only where an NVIDIA device exists
- [x] **GPU-04**: A GPU equivalence report documents observed deviation per model class within an accepted tolerance
- [x] **GPU-05**: SoA model buffers upload host→device zero-copy

### Serialization

- [x] **SER-01**: A model round-trips through the v5 binary format (serialize + deserialize)
- [x] **SER-02**: A model serializes to the v5 PyBuffer (zero-copy) representation
- [x] **SER-03**: A model dumps to JSON (`DumpAsJSON`)
- [x] **SER-04**: Field accessors expose model/tree fields for inspection

### Python Binding (PyO3)

- [x] **PY-01**: From Python, a user can load XGBoost / LightGBM / scikit-learn models
- [x] **PY-02**: From Python, a user can predict over numpy arrays with zero-copy buffer I/O
- [x] **PY-03**: From Python, a user can serialize/deserialize and JSON-dump a model
- [x] **PY-04**: A Python `sklearn.import_model` entry point marshals fitted estimators
- [x] **PY-05**: Library `thiserror` errors translate into Python exceptions
- [x] **PY-06**: The binding builds as an abi3 wheel via maturin

### Equivalence & Testing

- [x] **EQV-01**: The harness generates random seeded input matrices (dense + sparse CSR)
- [x] **EQV-02**: Golden output vectors are captured from C++ Treelite and committed as fixtures with a toolchain/libm manifest
- [x] **EQV-03**: Rust predictions assert within 1e-5 of goldens across model types, both presets, and all predict kinds
- [x] **EQV-04**: The harness reports max observed deviation, not just pass/fail

### Memory Efficiency

- [x] **MEM-01**: SoA columns use `bytemuck` `Pod` zero-copy recasting where layout allows
- [x] **MEM-02**: `smallvec` and `compact_str` are used for small collections and metadata strings
- [x] **MEM-03**: A custom global allocator (jemalloc) is wired into benchmarks/binaries
- [x] **MEM-04**: Borrowed buffers from the Python buffer protocol are consumed zero-copy

### Error Handling

- [x] **ERR-01**: Library crates expose typed `thiserror` errors at their API boundaries
- [x] **ERR-02**: Binaries and tests use `anyhow` for error context

## v1.1 Requirements

Milestone v1.1 — Parallel Scalar Inference. Row-parallelize the single-threaded scalar GTIL fallback (LightGBM `kLE`, categorical, non-`kLT`, and all sparse models) without regressing the 1e-5 contract. The cubecl numerical `kLT` path already parallelizes (~8/16 cores) and is out of scope.

### Parallel Inference

- [ ] **PAR-01**: Scalar dense predict (`treelite_gtil::predict`) runs row-parallel across all available cores, with output identical to the current serial path within 1e-5 and serial per-row `tree_id` summation preserved (GTIL-08)
- [ ] **PAR-02**: Scalar sparse predict (`predict_sparse`, and the `predict_cpu_sparse` fallback) runs row-parallel under the same equivalence guarantee
- [x] **PAR-03**: `Model` is soundly shareable across threads for read-only prediction — documented `unsafe impl Sync`/`Send` justified by predict being read-only over the model (mirrors upstream OpenMP); the `_assert_not_send` invariant is superseded by the new shareability contract
- [ ] **PAR-04**: `Config.nthread` is honored end-to-end (`≤0` = all cores; `N` = bounded pool), wiring the Python `nthread=` kwarg that is currently recorded-but-unused on the scalar path

## v2 Requirements

Deferred to a future release. Tracked but not in the current roadmap.

### Serialization

- **SER-v2-01**: Read/write legacy wire formats v3.9 and v4.0 with cross-version migration

### Performance

- **PERF-v2-01**: f16/bf16 half-precision inference opt-in fast path (off the 1e-5 equivalence path)
- **PERF-v2-02**: Autotuned / optimized GPU kernels and additional GPU backends (Metal, Vulkan) — ROCm promoted to v1 GPU-03
- **PERF-v2-03**: Dedicated memory-efficiency hardening sweep with regression budgets

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| C-API / `extern "C"` FFI | Explicit user constraint; PyO3 over the Rust core is the only binding |
| Bit-exact GPU reproducibility | GPU float reduction ordering differs; 1e-5 tolerance absorbs it. Determinism guaranteed only on the CPU backend |
| Live C++ Treelite build in CI | Golden vectors are generated once and frozen as fixtures; CI does not compile C++ |
| Full cubecl coverage beyond the inference hot path | Loaders/builder/serialization stay plain Rust to keep 1e-5 risk bounded |
| cubecl CPU grid tuning of the numerical `kLT` path (v1.1) | That path already uses ~8/16 cores; pushing `CubeCount`/`CubeDim` to full saturation is an uncertain incremental win. v1.1 parallelism targets the 1-core scalar fallback only |

## Traceability

Which phases cover which requirements.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FND-01 | Phase 1 | Complete |
| FND-02 | Phase 1 | Complete |
| ENUM-01 | Phase 1 | Complete |
| CORE-01 | Phase 1 | Complete |
| CORE-02 | Phase 1 | Complete |
| CORE-03 | Phase 1 | Complete |
| CORE-04 | Phase 1 | Complete |
| ERR-01 | Phase 1 | Complete |
| ERR-02 | Phase 1 | Complete |
| BLD-01 | Phase 2 | Complete (02-02; exercised end-to-end via the rewired XGBoost-JSON loader in 02-05) |
| BLD-02 | Phase 2 | Complete (02-02) |
| BLD-03 | Phase 2 | Complete (02-02) |
| SER-01 | Phase 2 | Complete |
| SER-02 | Phase 2 | Complete |
| SER-03 | Phase 2 | Complete |
| SER-04 | Phase 2 | Complete |
| XGB-01 | Phase 3 | Complete |
| XGB-02 | Phase 3 | Complete |
| XGB-03 | Phase 3 | Complete |
| XGB-04 | Phase 3 | Complete |
| XGB-05 | Phase 3 | Complete |
| LGB-01 | Phase 4 | Complete |
| LGB-02 | Phase 4 | Complete |
| LGB-03 | Phase 4 | Complete |
| SKL-01 | Phase 4 | Complete |
| SKL-02 | Phase 4 | Complete |
| SKL-03 | Phase 4 | Complete |
| SKL-04 | Phase 4 | Complete |
| GTIL-01 | Phase 5 | Complete |
| GTIL-02 | Phase 5 | Complete |
| GTIL-03 | Phase 5 | Complete |
| GTIL-04 | Phase 5 | Complete |
| GTIL-05 | Phase 5 | Complete |
| GTIL-06 | Phase 5 | Complete |
| GTIL-07 | Phase 5 | Complete |
| GTIL-08 | Phase 5 | Complete |
| EQV-01 | Phase 5 | Complete |
| EQV-02 | Phase 5 | Complete |
| EQV-03 | Phase 5 | Complete |
| EQV-04 | Phase 5 | Complete |
| GPU-01 | Phase 6 | Complete |
| GPU-02 | Phase 6 | Complete |
| GPU-05 | Phase 6 | Complete |
| GPU-03 | Phase 7 | Complete |
| GPU-04 | Phase 7 | Complete |
| PY-01 | Phase 8 | Complete |
| PY-02 | Phase 8 | Complete |
| PY-03 | Phase 8 | Complete |
| PY-04 | Phase 8 | Complete |
| PY-05 | Phase 8 | Complete |
| PY-06 | Phase 8 | Complete |
| MEM-04 | Phase 8 | Complete |
| MEM-01 | Phase 9 | Complete |
| MEM-02 | Phase 9 | Complete |
| MEM-03 | Phase 9 | Complete |
| PAR-01 | Phase 10 | Planned |
| PAR-02 | Phase 10 | Planned |
| PAR-03 | Phase 10 | Planned |
| PAR-04 | Phase 10 | Planned |

**Coverage:**

- v1 requirements: 45 total — Mapped: 45 ✓
- v1.1 requirements: 4 total (PAR-01..04) — Mapped to Phase 10: 4 ✓
- Unmapped: 0

**Note:** Phase 1 (the end-to-end MVP spine) additionally exercises a *minimal subset* of XGB-01 (single XGBoost-JSON model), GTIL-01 (scalar dense predict), and EQV-01/EQV-02 (harness skeleton + one golden) to prove the load→predict→verify pipeline. The full requirements are owned and completed in their dedicated phases (XGB in Phase 3, GTIL/EQV in Phase 5) to keep each requirement mapped to exactly one owning phase.

---
*Requirements defined: 2026-06-09*
*Last updated: 2026-06-11 — v1.1 (Parallel Scalar Inference) roadmapped: PAR-01..04 mapped to Phase 10 (Planned).*

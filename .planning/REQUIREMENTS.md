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

- [ ] **BLD-01**: A fluent `ModelBuilder` constructs a model node-by-node with orphan/topology validation
- [ ] **BLD-02**: `ConcatenateModelObjects` merges multiple models into one
- [ ] **BLD-03**: A `BulkConstructTree` fast path builds trees from pre-validated bulk input

### XGBoost Loader

- [ ] **XGB-01**: User can load an XGBoost JSON model
- [ ] **XGB-02**: User can load an XGBoost UBJSON model (parser shares the JSON state machine for numeric parity)
- [ ] **XGB-03**: User can load an XGBoost legacy binary model (little-endian layout)
- [ ] **XGB-04**: The loader auto-detects which XGBoost format a file uses
- [ ] **XGB-05**: XGBoost objective maps to the correct postprocessor, with the version-gated `base_score` margin transform applied

### LightGBM Loader

- [ ] **LGB-01**: User can load a LightGBM text-format model
- [ ] **LGB-02**: Categorical splits decode correctly (bitset) with upstream-matching per-field precision
- [ ] **LGB-03**: LightGBM objective maps to the correct postprocessor (+ `sigmoid_alpha`), with `class_id` round-robin and `average_output` honored

### scikit-learn Loader

- [ ] **SKL-01**: User can import `RandomForest` and `ExtraTrees` (classifier + regressor)
- [ ] **SKL-02**: User can import `GradientBoosting` (classifier + regressor)
- [ ] **SKL-03**: User can import `IsolationForest`
- [ ] **SKL-04**: User can import `HistGradientBoosting` (classifier + regressor)

### GTIL Inference

- [ ] **GTIL-01**: User can predict over a dense row-major input matrix
- [ ] **GTIL-02**: User can predict over a sparse CSR input matrix (absent entries treated as NaN)
- [ ] **GTIL-03**: All four predict kinds are supported (`default`, `raw`, `leaf_id`, `score_per_tree`)
- [ ] **GTIL-04**: All ten postprocessors are ported verbatim, preserving upstream mixed-precision arithmetic
- [ ] **GTIL-05**: Missing-value routing fires on NaN only, via the node default direction
- [ ] **GTIL-06**: Categorical-split evaluation applies the float-representability guard and correct child polarity
- [ ] **GTIL-07**: Output shaping is correct — `GetOutputShape`, leaf-vector broadcast, tree averaging, and `f64` base-score addition
- [ ] **GTIL-08**: Per-row tree summation is serial in `tree_id` order (parallelism only across rows)

### cubecl Compute & GPU

- [ ] **GPU-01**: The GTIL inference hot path (traversal + postprocessors) is implemented as cubecl kernels
- [ ] **GPU-02**: The cubecl CPU backend is the default and is validated to 1e-5
- [ ] **GPU-03**: At least one GPU backend (CUDA or wgpu) is runtime-selectable and produces predictions
- [ ] **GPU-04**: A GPU equivalence report documents observed deviation per model class within an accepted tolerance
- [ ] **GPU-05**: SoA model buffers upload host→device zero-copy

### Serialization

- [ ] **SER-01**: A model round-trips through the v5 binary format (serialize + deserialize)
- [ ] **SER-02**: A model serializes to the v5 PyBuffer (zero-copy) representation
- [ ] **SER-03**: A model dumps to JSON (`DumpAsJSON`)
- [ ] **SER-04**: Field accessors expose model/tree fields for inspection

### Python Binding (PyO3)

- [ ] **PY-01**: From Python, a user can load XGBoost / LightGBM / scikit-learn models
- [ ] **PY-02**: From Python, a user can predict over numpy arrays with zero-copy buffer I/O
- [ ] **PY-03**: From Python, a user can serialize/deserialize and JSON-dump a model
- [ ] **PY-04**: A Python `sklearn.import_model` entry point marshals fitted estimators
- [ ] **PY-05**: Library `thiserror` errors translate into Python exceptions
- [ ] **PY-06**: The binding builds as an abi3 wheel via maturin

### Equivalence & Testing

- [ ] **EQV-01**: The harness generates random seeded input matrices (dense + sparse CSR)
- [ ] **EQV-02**: Golden output vectors are captured from C++ Treelite and committed as fixtures with a toolchain/libm manifest
- [ ] **EQV-03**: Rust predictions assert within 1e-5 of goldens across model types, both presets, and all predict kinds
- [ ] **EQV-04**: The harness reports max observed deviation, not just pass/fail

### Memory Efficiency

- [ ] **MEM-01**: SoA columns use `bytemuck` `Pod` zero-copy recasting where layout allows
- [ ] **MEM-02**: `smallvec` and `compact_str` are used for small collections and metadata strings
- [ ] **MEM-03**: A custom global allocator (jemalloc) is wired into benchmarks/binaries
- [ ] **MEM-04**: Borrowed buffers from the Python buffer protocol are consumed zero-copy

### Error Handling

- [x] **ERR-01**: Library crates expose typed `thiserror` errors at their API boundaries
- [ ] **ERR-02**: Binaries and tests use `anyhow` for error context

## v2 Requirements

Deferred to a future release. Tracked but not in the current roadmap.

### Serialization

- **SER-v2-01**: Read/write legacy wire formats v3.9 and v4.0 with cross-version migration

### Performance

- **PERF-v2-01**: f16/bf16 half-precision inference opt-in fast path (off the 1e-5 equivalence path)
- **PERF-v2-02**: Autotuned / optimized GPU kernels and additional GPU backends (ROCm, Metal, Vulkan)
- **PERF-v2-03**: Dedicated memory-efficiency hardening sweep with regression budgets

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| C-API / `extern "C"` FFI | Explicit user constraint; PyO3 over the Rust core is the only binding |
| Bit-exact GPU reproducibility | GPU float reduction ordering differs; 1e-5 tolerance absorbs it. Determinism guaranteed only on the CPU backend |
| Live C++ Treelite build in CI | Golden vectors are generated once and frozen as fixtures; CI does not compile C++ |
| Full cubecl coverage beyond the inference hot path | Loaders/builder/serialization stay plain Rust to keep 1e-5 risk bounded |

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
| ERR-02 | Phase 1 | Pending |
| BLD-01 | Phase 2 | Pending |
| BLD-02 | Phase 2 | Pending |
| BLD-03 | Phase 2 | Pending |
| SER-01 | Phase 2 | Pending |
| SER-02 | Phase 2 | Pending |
| SER-03 | Phase 2 | Pending |
| SER-04 | Phase 2 | Pending |
| XGB-01 | Phase 3 | Pending |
| XGB-02 | Phase 3 | Pending |
| XGB-03 | Phase 3 | Pending |
| XGB-04 | Phase 3 | Pending |
| XGB-05 | Phase 3 | Pending |
| LGB-01 | Phase 4 | Pending |
| LGB-02 | Phase 4 | Pending |
| LGB-03 | Phase 4 | Pending |
| SKL-01 | Phase 4 | Pending |
| SKL-02 | Phase 4 | Pending |
| SKL-03 | Phase 4 | Pending |
| SKL-04 | Phase 4 | Pending |
| GTIL-01 | Phase 5 | Pending |
| GTIL-02 | Phase 5 | Pending |
| GTIL-03 | Phase 5 | Pending |
| GTIL-04 | Phase 5 | Pending |
| GTIL-05 | Phase 5 | Pending |
| GTIL-06 | Phase 5 | Pending |
| GTIL-07 | Phase 5 | Pending |
| GTIL-08 | Phase 5 | Pending |
| EQV-01 | Phase 5 | Pending |
| EQV-02 | Phase 5 | Pending |
| EQV-03 | Phase 5 | Pending |
| EQV-04 | Phase 5 | Pending |
| GPU-01 | Phase 6 | Pending |
| GPU-02 | Phase 6 | Pending |
| GPU-05 | Phase 6 | Pending |
| GPU-03 | Phase 7 | Pending |
| GPU-04 | Phase 7 | Pending |
| PY-01 | Phase 8 | Pending |
| PY-02 | Phase 8 | Pending |
| PY-03 | Phase 8 | Pending |
| PY-04 | Phase 8 | Pending |
| PY-05 | Phase 8 | Pending |
| PY-06 | Phase 8 | Pending |
| MEM-04 | Phase 8 | Pending |
| MEM-01 | Phase 9 | Pending |
| MEM-02 | Phase 9 | Pending |
| MEM-03 | Phase 9 | Pending |

**Coverage:**
- v1 requirements: 45 total
- Mapped to phases: 45 ✓
- Unmapped: 0

**Note:** Phase 1 (the end-to-end MVP spine) additionally exercises a *minimal subset* of XGB-01 (single XGBoost-JSON model), GTIL-01 (scalar dense predict), and EQV-01/EQV-02 (harness skeleton + one golden) to prove the load→predict→verify pipeline. The full requirements are owned and completed in their dedicated phases (XGB in Phase 3, GTIL/EQV in Phase 5) to keep each requirement mapped to exactly one owning phase.

---
*Requirements defined: 2026-06-09*
*Last updated: 2026-06-09 after roadmap creation (traceability populated, 45/45 mapped)*

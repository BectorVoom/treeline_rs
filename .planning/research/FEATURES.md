# Feature Research

**Domain:** Tree-ensemble model library (Rust port of Treelite 4.7.0) — loaders, core model, GTIL inference, serialization, PyO3 binding
**Researched:** 2026-06-09
**Confidence:** HIGH (enums, predict kinds, postprocessor list, and model fields verified directly against the vendored C++ headers/sources in `treelite-mainline/`; no training-data guessing)

> Categorization convention for this greenfield port:
> - **Table stakes** = required for 1e-5 prediction equivalence or basic load→predict→serialize usability. Missing one breaks the Core Value.
> - **Differentiator** = the Rust port's reason to exist (cubecl GPU acceleration, memory efficiency). Not in upstream.
> - **Anti-feature** = deliberately NOT in v1 (already decided in PROJECT.md Out of Scope, plus newly-surfaced ones).
>
> Complexity is relative porting effort (LOW/MEDIUM/HIGH). Dependencies are called out per subsystem and consolidated in the Dependencies section.

---

## Feature Landscape

### Table Stakes (Required for 1e-5 Equivalence / Basic Usability)

#### Subsystem 1 — Core Model (`tree.h`) — everything else depends on this

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| `Model` type-erased over two presets: `<f32,f32>` and `<f64,f64>` | Threshold/leaf numeric type drives all arithmetic; wrong type breaks equivalence | MEDIUM | C++ `std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` → Rust `enum`. **Only these two**; mixed types unsupported (`static_assert`). `GetThresholdType`/`GetLeafOutputType` return `TypeInfo`. |
| Struct-of-Arrays `Tree<T,L>` node storage | Layout must match for serialization + cache-friendly traversal | MEDIUM | Parallel `ContiguousArray` columns: `node_type_`, `cleft_`, `cright_`, `split_index_`, `default_left_`, `leaf_value_`, `threshold_`, `cmp_`, `category_list_right_child_`. |
| 3 node types: `kLeafNode`, `kNumericalTestNode`, `kCategoricalTestNode` | Traversal dispatch; enum verified in `tree_node_type.h` (int8: 0,1,2) | LOW | No other node types exist upstream. |
| 6 comparison operators: `kNone, kEQ, kLT, kLE, kGT, kGE` | `NextNode` switches on these; verified `operator.h` | LOW | XGBoost uses `kLT`; LightGBM uses `kLE`. Must preserve exact operator per loader to match. |
| Numerical split evaluation (`fvalue [op] threshold`) | Core inference; `predict.cc::NextNode` | LOW | Branch goes to left child if condition true, else right. |
| Missing-value handling via `default_left_` / `DefaultChild` | NaN input routes to default child; verified `predict.cc::EvaluateTree` (`std::isnan`) | LOW | `DefaultChild = default_left ? cleft : cright`. Critical for equivalence on sparse/missing data. |
| Categorical splits: `category_list_`, begin/end offsets, `category_list_right_child_`, `has_categorical_split_` | `NextNodeCategorical`; mushroom/sparse_categorical/toy_categorical examples exercise this | MEDIUM | Category match test has exacting validity rules (see Pitfalls): value must be `>=0`, representable as the input float type, and `<= uint32 max`; otherwise treated as no-match. |
| Scalar leaf output (`leaf_value_`) | Most XGBoost/GBM trees | LOW | `LeafValue(nid)`. |
| Vector leaf output (`leaf_vector_`, begin/end offsets, `HasLeafVector`) | Multiclass RandomForest, multi-target trees | MEDIUM | Per-node variable-length slice. `OutputLeafVector` handles 4 target/class broadcast cases. |
| Header metadata fields | Drive output shape + post-processing | MEDIUM | `num_feature`, `task_type`, `average_tree_output`, `num_target`, `num_class[]`, `leaf_vector_shape[2]`, `target_id[]`, `class_id[]`, `postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores[]`, `attributes`. |
| Per-tree annotation `target_id[i]` / `class_id[i]` (with `-1` = "all targets/classes") | Determines which output cell each tree contributes to | MEDIUM | The `-1` broadcast semantics are load-bearing for multiclass/multitarget. |
| Optional node stats: `data_count_`, `sum_hess_`, `gain_` + `_present_` flags | Round-trip fidelity (not used in default predict) | LOW | Present-bitmaps must serialize. |
| `ContiguousArray<T>` owned + foreign-buffer modes | Storage primitive; foreign mode enables zero-copy PyBuffer deser | MEDIUM | `UseForeignBuffer`. Owned mode is table stakes; foreign mode overlaps the memory differentiator. |

#### Subsystem 2 — Enums (`enum/`) — no dependencies, everyone depends on it

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| `TaskType`: `kBinaryClf, kRegressor, kMultiClf, kLearningToRank, kIsolationForest` (uint8 0–4) | Verified `task_type.h` | LOW | Drives output shaping + loader behavior. |
| `TreeNodeType`, `Operator`, `TypeInfo` (float32/float64) | Shared vocabulary | LOW | String<->enum conversions needed for JSON dump + config parse. |

#### Subsystem 3 — Model Builder (`model_builder.h`) — loaders depend on it

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| Fluent builder: `StartTree/EndTree`, `StartNode(key)/EndNode`, `NumericalTest`, `CategoricalTest`, `LeafScalar`, `LeafVector` (f32 + f64 overloads), `CommitModel` | Every loader builds models through this | HIGH | Node addressed by integer `node_key`; children referenced by key, resolved at `EndTree`. |
| `Metadata` struct (num_feature, task_type, average_tree_output, num_target, num_class[], leaf_vector_shape[2]) | Builder initialization | MEDIUM | `InitializeMetadata` + `GetModelBuilder` overloads (typed args, empty, JSON-string). |
| `TreeAnnotation` (num_tree, target_id[], class_id[]) | Sets per-tree target/class mapping | LOW | |
| `PostProcessorFunc` (name + config map of `int64\|double\|string`) | Names the postprocessor + carries params (e.g. sigmoid_alpha) | LOW | |
| `base_scores` + `attributes` (arbitrary JSON string) | Intercept + opaque passthrough | LOW | |
| Per-node stats setters: `Gain`, `DataCount`, `SumHess` | Round-trip fidelity | LOW | |
| Validation flag `check_orphaned_nodes` (default true) + topology validation at `EndTree` | Catches malformed loader output | MEDIUM | |
| `ConcatenateModelObjects` (model_concat.cc) | Merges models; used by `Model.concatenate` Python API + parallel build pattern | MEDIUM | Builder is single-threaded; concat is the parallelism story. |

#### Subsystem 4 — Model Loaders (`model_loader/`) — depend on Builder + Core

**XGBoost (3 input formats + variants):**

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| XGBoost JSON loader (SAX-based) | `LoadXGBoostModelJSON` / `...FromJSONString`; primary modern format | HIGH | Uses delegated SAX handler → ModelBuilder adapters. `allow_unknown_field` config flag. |
| XGBoost UBJSON loader | `LoadXGBoostModelUBJSON` / `...FromUBJSONString`; default for XGBoost >=2.1 `save_raw` | HIGH | Binary universal-binary-JSON parser. Same model graph as JSON path. |
| XGBoost legacy binary loader | `LoadXGBoostModelLegacyBinary` (file + memory buffer) | HIGH | Old `.model` format; raw struct layout parsing. |
| `DetectXGBoostFormat` (JSON vs UBJSON by first bytes) | Powers Python `format_choice="inspect"` | LOW | |
| XGBoost objective → postprocessor + base_score mapping | Equivalence: e.g. `binary:logistic`→sigmoid, `reg:squarederror`→identity, `multi:softmax/softprob`→softmax, `count:poisson`→exponential, `reg:logistic`, `rank:*` | MEDIUM | Shared logic in `model_loader/detail/xgboost.cc`. Getting these mappings right is essential for 1e-5. |
| XGBoost base_score / global_bias handling (incl. the logit-inversion quirk) | XGBoost stores base_score already transformed for some objectives | MEDIUM | Known XGBoost gotcha; see Pitfalls. |

**LightGBM (text format):**

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| LightGBM text-format loader | `LoadLightGBMModel` / `...FromString` | HIGH | Parses `model_to_string()` text. Key fields: `max_feature_idx`, `num_class`, `num_tree_per_iteration`, `objective`, per-tree `split_feature`, `threshold`, `decision_type` (bitflags: default-left + categorical + comparison), `left_child`, `right_child`, `leaf_value`, `cat_threshold`, `cat_boundaries`, `internal_*`. |
| LightGBM categorical splits via bitset (`cat_threshold`/`cat_boundaries`) | LightGBM encodes categories as packed bitsets | MEDIUM | Must decode to `category_list_`. |
| LightGBM objective → postprocessor mapping | `binary`→sigmoid, `multiclass`→softmax, `multiclassova`→multiclass_ova, `regression/regression_l1/huber`→identity, `poisson`→exponential, `lambdarank`→identity, etc. | MEDIUM | `average_output` flag (random forest mode) → `average_tree_output`. |
| LightGBM `decision_type` missing-value semantics | LightGBM has its own default-direction + NaN-vs-zero rules | MEDIUM | Must reproduce exactly; differs subtly from XGBoost. |

**scikit-learn (array bulk-construction path):**

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| `LoadRandomForestRegressor` (also serves ExtraTreesRegressor) | Multi-target supported; `average_tree_output=true` | MEDIUM | Array-of-arrays interface: node_count, children_left/right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity. |
| `LoadRandomForestClassifier` (also ExtraTreesClassifier) | `n_classes[]` per target; leaf vector = class probs | MEDIUM | Multi-output classification → leaf_vector_shape. |
| `LoadGradientBoostingRegressor` | Single-target; `baseline_prediction`; leaf shrunk by learning_rate (done in Python importer) | MEDIUM | base_scores shape (1,). |
| `LoadGradientBoostingClassifier` | base_scores shape (n_classes,); softmax/sigmoid postprocess | MEDIUM | Python computes base_scores from `init_` (zero or DummyClassifier prior). |
| `LoadIsolationForest` + `ratio_c` standardizing constant + isolation-depth leaf values | Anomaly score; `exponential_standard_ratio` postprocessor | MEDIUM-HIGH | Leaf = expected isolation depth; equivalence target is `-clf.score_samples(X)`. Depth computation (`calculate_depths`) lives in Python importer — port logic too. |
| `BulkConstructTree` fast path (`sklearn_bulk.cc`) | The "bulk-construction path" — bypasses ModelBuilder, writes SoA arrays directly | HIGH | Friend function on `Tree<T,L>`; assumes validated input. Required to match sklearn import behavior + perf. |

> **Note:** The scikit-learn *integration* (introspecting live Python estimator objects) lives in the upstream Python package (`sklearn/importer.py`), which marshals estimator arrays and calls the C-API loaders. In the Rust port the marshalling moves into the **PyO3 layer**; the array-based loader functions themselves are core Rust. HistGradientBoosting loaders are listed under Anti-Features (see below) for v1 scoping.

#### Subsystem 5 — GTIL Inference (`gtil.h`, `gtil/`) — depends on Core

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| 4 predict kinds — **verified exact names from `gtil.h`** | The core inference API | HIGH | `kPredictDefault=0` (sum + postprocess), `kPredictRaw=1` (sum, no postprocess), `kPredictLeafID=2` (one int leaf ID/tree), `kPredictPerTree=3` (margin score(s)/tree). Python names: `default`, `raw`, `leaf_id`, `score_per_tree`. |
| Dense row-major input (`Predict<f32>`/`<f64>`) | Primary path; `DenseMatrixAccessor` | MEDIUM | mdspan-style 2D view, row-major. |
| Sparse CSR input (`PredictSparse`) | `data/col_ind/row_ptr`; `SparseMatrixAccessor` materializes dense row per thread, filling absent cols with NaN | MEDIUM | Per-thread scratch row; NaN-fill is what makes missing-value handling kick in. |
| `GetOutputShape` | Caller allocates output; shapes differ per predict kind | MEDIUM | default/raw → `(num_row, num_target, max_num_class)`; leaf_id → `(num_row, num_tree)`; per_tree → `(num_row, num_tree, leaf_vector_shape[0]*leaf_vector_shape[1])`. |
| Tree traversal `EvaluateTree` (numerical + categorical + NaN→default) | Hot path | LOW | This is the cubecl kernel target (differentiator). |
| Leaf accumulation with target/class broadcast (`OutputLeafVector` 4 cases, `OutputLeafValue`) | Multi-target/multiclass output correctness | HIGH | The `-1` target_id/class_id broadcast + leaf_vector_shape assertions are subtle; high equivalence risk. |
| Tree-output averaging (`average_tree_output`) | RandomForest/sklearn divides by per-cell tree count | MEDIUM | `average_factor` computed with same broadcast rules; applied before base_scores. |
| Base-score (intercept) addition | Added per (target,class) after averaging, before postprocess | LOW | `base_scores` is f64 regardless of preset. |
| **Full postprocessor set — verified from `postprocessor.cc`** | Equivalence | MEDIUM | 10 functions: `identity`, `signed_square`, `hinge`, `sigmoid` (uses `sigmoid_alpha`), `exponential`, `exponential_standard_ratio` (uses `ratio_c`, base-2 exp), `logarithm_one_plus_exp` (`log1p(exp(x))`), `identity_multiclass`, `softmax` (max-shifted, per-row over num_class), `multiclass_ova` (per-class sigmoid). Unknown name = FATAL. |
| `Configuration` from JSON (`nthread`, `pred_kind`) | Config parsing (`config.cc`) | LOW | `nthread<=0` = all threads. |
| Multi-threaded predict (OpenMP `ParallelFor`, static schedule, over rows) | Perf; deterministic per-row | MEDIUM | **This is the cubecl parallelism target** (see Threading). Row-parallel, so reductions are per-row independent → safe for 1e-5. |

#### Subsystem 6 — Serialization (`serializer.cc`, `json_serializer.cc`) — depends on Core

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| v5 binary serialize/deserialize (stream + in-memory buffer) | Persistence; round-trip | HIGH | `SerializeToStream`/`DeserializeFromStream`, `SerializeToBuffer`. Encodes version triple (`major/minor/patch_ver`), all header metadata fields, and every tree's SoA arrays + optional-field extension slots (`num_opt_field_per_model/tree/node`). |
| v5 PyBuffer (zero-copy) serialize/deserialize | Python binding path; PEP-3118 frames | HIGH | `SerializeToPyBuffer` → `Vec<PyBufferFrame>` (buf, format, itemsize, nitem); `DeserializeFromPyBuffer`. Foreign-buffer aliasing enables zero-copy. |
| `DumpAsJSON(pretty_print)` | Model inspection (`json_serializer.cc`) | MEDIUM | Human-readable dump including per-node IDs (BFS order). |
| Header/Tree field accessors (`GetHeaderField`/`SetHeaderField`/`GetTreeField`/`SetTreeField`) | Python `HeaderAccessor`/`TreeAccessor` editing by field name | MEDIUM | Field-name → PyBufferFrame. Needed to mirror Python API. |
| Round-trip of optional fields + present-bitmaps + has_categorical_split + attributes | Lossless persistence | MEDIUM | Extension-slot counts recomputed at serialize time. |

#### Subsystem 7 — PyO3 Binding (mirror upstream Python API) — depends on all above

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| `Model` class: `num_tree`, `num_feature`, `input_type`, `output_type`, `__del__` lifecycle | Basic usability | LOW | |
| Loaders: `load_xgboost_model(format_choice, allow_unknown_field)`, `load_xgboost_model_legacy_binary`, `load_lightgbm_model`, `from_xgboost_json`, `from_xgboost_ubjson` | Mirror `frontend.py` | MEDIUM | `from_xgboost`/`from_lightgbm` (live Booster objects) call into these — keep as Python-side helpers. |
| `treelite.sklearn.import_model(estimator)` | Introspects live sklearn estimators → array loaders | HIGH | Port the array-marshalling + isolation-depth logic from `importer.py` into the PyO3 layer. |
| GTIL: `predict(pred_margin)`, `predict_leaf`, `predict_per_tree`; numpy + scipy CSR input | Core predict surface | MEDIUM | Dense path pads missing trailing features with NaN; output dtype follows `model.output_type`. |
| `serialize`/`serialize_bytes`/`deserialize`/`deserialize_bytes`, `dump_as_json`, `get_tree_depth`, `Model.concatenate` | Persistence + introspection | MEDIUM | |
| `get_header_accessor`/`get_tree_accessor` (field get/set via numpy buffer protocol) | Field-level editing | MEDIUM | |
| numpy/buffer-protocol zero-copy I/O (`_TreelitePyBufferFrame` equivalent) | Memory efficiency + correctness of dtype mapping | MEDIUM | dtype format strings: `=l`(i32), `=Q`(u64), `=L`(u32), `=B`(u8), `=f`(f32), `=d`(f64). |

---

### Differentiators (The Rust Port's Reason to Exist — Not in Upstream)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| cubecl-accelerated GTIL hot path (tree traversal + postprocessors as kernels) | GPU/parallel acceleration of the inference loop | HIGH | Replaces upstream's OpenMP `ParallelFor`. Row-parallel structure maps cleanly to cubecl launch dims. Scope is the hot path only. |
| cubecl CPU backend default, GPU (CUDA/ROCm/wgpu) opt-in at runtime | Runs in CI without a GPU; deterministic default | MEDIUM | GPU backends are runtime-selected. |
| Memory efficiency: zero-copy buffers, smallvec/compact-string, custom allocator (jemalloc/mimalloc) | Lower footprint than C++ for large ensembles | MEDIUM-HIGH | Per optimiser manual. `ContiguousArray` foreign-buffer mode + Arrow zero-copy patterns. |
| Optional f16 half-precision via cubecl | Further memory/throughput win for inference | MEDIUM | Within 1e-5 tolerance only where safe; opt-in. |
| Typed `thiserror` errors at crate boundaries | Better DX than upstream's `TREELITE_CHECK`/thread-local error string | LOW | `anyhow` in bins/tests. |
| Seeded golden-vector equivalence harness | Confidence that the port matches C++ across random inputs, not just canned examples | MEDIUM | Frozen fixtures generated once from C++. |

---

### Anti-Features (Deliberately NOT in v1)

| Feature | Why It Seems Needed | Why Excluded for v1 | Alternative |
|---------|---------------------|---------------------|-------------|
| C-API / `extern "C"` FFI | Upstream's entire public surface is the C-API | Explicit PROJECT.md constraint; PyO3 binds the Rust core directly | PyO3 over native Rust types. |
| Legacy serialization formats v3.9 and v4.0 (+ migration) | Upstream maintains a 3-generation compatibility matrix | Large migration surface; v1 reads/writes **v5 only** (`From: >=5.0 → To: >=5.0` only) | Defer multi-version to v1.x. |
| HistGradientBoosting loaders (Regressor + Classifier) | Part of sklearn estimator family | Most complex sklearn path: packed node structs, categorical bitsets, feature/category remapping, `_bin_mapper`, version-gated `_preprocessor` introspection | Cover RF/ExtraTrees/GBM/IsolationForest in v1; add Hist in v1.x. *(Confidence MEDIUM — confirm with milestone owner; PROJECT.md lists "scikit-learn estimators" generally.)* |
| cubecl for loaders / builder / serialization | "Accelerate everything" | No throughput benefit; raises 1e-5 risk | Keep them plain idiomatic Rust (PROJECT.md decision). |
| Bit-exact GPU reproducibility | Determinism | GPU float reduction ordering differs | 1e-5 tolerance absorbs it; determinism guaranteed only on CPU backend. |
| Live C++ build in CI for equivalence | "Always compare against source of truth" | Toolchain burden | Frozen golden vectors. |
| Compiled-model code generation (Treelite's historical "compiler"/TL2cgen) | Treelite was originally a model→C compiler | Not in vendored 4.7.0 core; out of scope entirely | GTIL interpretation only. |
| `Model.load()` / `from_xgboost()` legacy Python entrypoints | Backwards compat | Already removed/deprecated upstream (raise `RuntimeError`) | Use `frontend.*` functions. |
| Multi-threaded `ModelBuilder` | Parallel construction | Upstream forbids it (single-thread only) | Build N models, `ConcatenateModelObjects`. |
| PyPy support for buffer-protocol field accessors | Cross-interpreter | Upstream itself raises `NotImplementedError` on PyPy | CPython only. |

---

## Feature Dependencies

```
Enums (TaskType, TreeNodeType, Operator, TypeInfo)
    └──used by──> EVERYTHING

ContiguousArray<T>
    └──required by──> Tree<T,L> (SoA storage)
                          └──required by──> Model (variant over presets)
                                                └──required by──> ModelBuilder
                                                                      └──required by──> ALL Loaders
                                                Model
                                                  ├──required by──> GTIL (Predict / PredictSparse)
                                                  ├──required by──> Serializer (v5 binary/PyBuffer/JSON)
                                                  └──required by──> PyO3 binding

ModelBuilder ──used by──> XGBoost(JSON/UBJSON/legacy), LightGBM loaders
BulkConstructTree (bypasses Builder) ──used by──> sklearn array loaders

GTIL postprocessors ──require──> Model.{postprocessor, sigmoid_alpha, ratio_c, num_class}
GTIL output shaping ──require──> Model.{num_target, num_class, leaf_vector_shape, target_id, class_id}
Loaders ──set──> {postprocessor name, base_scores, average_tree_output}  (objective→postprocessor mapping)

cubecl GTIL kernels ──replace──> OpenMP ParallelFor in predict.cc  (differentiator over the same Model)
PyO3.sklearn.import_model ──marshals──> sklearn array loaders
Serializer.PyBuffer ──enables──> PyO3 zero-copy I/O ──overlaps──> memory differentiator
```

### Dependency Notes

- **All loaders require the Builder + Core Model.** No loader can be written until `ModelBuilder` (or `BulkConstructTree` for sklearn) and the `Tree`/`Model` types exist. This forces phase ordering: Enums → Core Model → Builder → Loaders.
- **GTIL requires the full Core Model metadata**, not just trees: postprocessor name, sigmoid_alpha, ratio_c, base_scores, target_id/class_id broadcast rules, leaf_vector_shape. A loader that sets these wrong produces correct trees but wrong predictions.
- **Objective→postprocessor mapping lives in the loaders**, but is *verified* by GTIL. The equivalence harness can't pass until both the loader mapping and the matching postprocessor are correct — they must land together per format.
- **Serializer v5 requires the final Core Model field layout to be frozen first.** SoA column set + optional-field extension slots must be stable before wire format is written.
- **cubecl kernels depend on a working scalar GTIL reference.** Implement scalar (CPU, single-thread) GTIL first to establish the 1e-5 baseline, then port the hot path to cubecl and validate against it.
- **PyO3 binding depends on everything** — it's necessarily the last subsystem.
- **Categorical splits cut across loaders, core, GTIL, and serialization** (LightGBM bitsets, XGBoost category lists, `NextNodeCategorical`, `category_list_*` arrays). Treat categorical as a vertical slice that touches every layer.

---

## MVP Definition

### Launch With (v1) — per PROJECT.md Active scope

- [ ] Enums + Core Model (`<f32,f32>`/`<f64,f64>` presets, SoA `Tree`, all node/leaf/categorical fields, header metadata) — foundation for everything
- [ ] `ModelBuilder` + validation + `ConcatenateModelObjects` — required by loaders
- [ ] XGBoost loaders: JSON, UBJSON, legacy binary + objective→postprocessor mapping + base_score quirk
- [ ] LightGBM text loader + categorical bitsets + objective mapping
- [ ] scikit-learn array loaders: RandomForest(R/C), ExtraTrees (same path), GradientBoosting(R/C), IsolationForest + `BulkConstructTree`
- [ ] GTIL: dense + sparse CSR, 4 predict kinds, all 10 postprocessors, averaging + base_scores, output shaping — scalar reference first
- [ ] GTIL hot path as cubecl kernels (CPU default, GPU opt-in)
- [ ] Serialization: v5 binary + PyBuffer + JSON dump + field accessors, full round-trip
- [ ] PyO3 binding mirroring `frontend.py` + `gtil.py` + `model.py` + `sklearn.import_model`
- [ ] Seeded golden-vector equivalence harness (1e-5)

### Add After Validation (v1.x)

- [ ] HistGradientBoosting loaders (Regressor + Classifier) — once core sklearn path proven; high complexity (node structs, bitsets, remapping)
- [ ] f16 half-precision inference path
- [ ] Memory-efficiency hardening (custom allocator, smallvec/compact-string sweep)

### Future Consideration (v2+)

- [ ] Legacy serialization v3.9 / v4.0 read + cross-version migration — only if downstream consumers need to load old checkpoints
- [ ] Additional GPU backend tuning / autotuned kernels

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Core Model (SoA, presets, metadata) | HIGH | MEDIUM | P1 |
| Enums | HIGH | LOW | P1 |
| ModelBuilder + validation | HIGH | HIGH | P1 |
| XGBoost JSON/UBJSON loaders | HIGH | HIGH | P1 |
| XGBoost legacy binary loader | MEDIUM | HIGH | P1 |
| LightGBM loader | HIGH | HIGH | P1 |
| sklearn array loaders (RF/ET/GBM/IsoForest) | HIGH | MEDIUM-HIGH | P1 |
| GTIL scalar reference (4 kinds, postprocessors, shaping) | HIGH | HIGH | P1 |
| Serialization v5 (binary/PyBuffer/JSON/accessors) | HIGH | HIGH | P1 |
| PyO3 binding | HIGH | HIGH | P1 |
| Equivalence harness | HIGH | MEDIUM | P1 |
| cubecl GTIL kernels | HIGH (project raison d'être) | HIGH | P1 |
| Memory efficiency (allocator/compact types) | MEDIUM | MEDIUM-HIGH | P2 |
| HistGradientBoosting loaders | MEDIUM | HIGH | P2 |
| f16 half-precision | MEDIUM | MEDIUM | P2 |
| Legacy v3.9/v4.0 serialization | LOW | HIGH | P3 |

**Priority key:** P1 = required for v1 launch / Core Value · P2 = add when core proven · P3 = future.

---

## Competitor / Reference Feature Analysis

| Feature | Upstream Treelite 4.7.0 (C++) | This Port (treelite-rs) |
|---------|-------------------------------|--------------------------|
| Numeric presets | `variant<float,float / double,double>` | Rust `enum` over the same two presets |
| Parallelism | OpenMP `ParallelFor` over rows | cubecl kernels, row-parallel; CPU default, GPU opt-in |
| Language binding | C-API → ctypes Python | PyO3 directly over Rust (no C-API) |
| Serialization versions | v3.9 / v4.0 / v5 | v5 only (v1) |
| sklearn coverage | RF, ExtraTrees, GBM, IsolationForest, HistGB | RF, ExtraTrees, GBM, IsolationForest (HistGB deferred to v1.x) |
| Error handling | `TREELITE_CHECK` + thread-local string | `thiserror` typed errors |
| Memory | owned/foreign `ContiguousArray` | + smallvec/compact-string + custom allocator + optional f16 |

---

## Verified-Against-Source Checklist (quality gate)

- [x] **Predict kinds** verified from `include/treelite/gtil.h`: `kPredictDefault(0)`, `kPredictRaw(1)`, `kPredictLeafID(2)`, `kPredictPerTree(3)` — and the Python names `default/raw/leaf_id/score_per_tree` from `gtil/gtil.py`.
- [x] **Postprocessor list** verified from `src/gtil/postprocessor.cc` (10): identity, signed_square, hinge, sigmoid, exponential, exponential_standard_ratio, logarithm_one_plus_exp, identity_multiclass, softmax, multiclass_ova.
- [x] **Enums** verified: TaskType (5), TreeNodeType (3), Operator (6 incl. kNone).
- [x] **Core model fields** verified from `include/treelite/tree.h` (SoA arrays + header metadata + version triple + compatibility matrix lines 491–500).
- [x] **Loader surface** verified from `include/treelite/model_loader.h` + `python/treelite/frontend.py` + `sklearn/importer.py`.
- [x] **PyO3 surface** verified from `python/treelite/{model.py, frontend.py, gtil/gtil.py, sklearn/importer.py}`.
- [x] Complexity + dependencies noted per feature.

## Sources

- `treelite-mainline/include/treelite/gtil.h` (predict kinds, Predict/PredictSparse/GetOutputShape) — HIGH
- `treelite-mainline/src/gtil/predict.cc` (traversal, NaN→default, categorical match rules, leaf broadcast, averaging, base_scores) — HIGH
- `treelite-mainline/src/gtil/postprocessor.cc` (10 postprocessors) — HIGH
- `treelite-mainline/include/treelite/enum/{task_type,tree_node_type,operator}.h` — HIGH
- `treelite-mainline/include/treelite/tree.h` (Model/Tree fields, SoA arrays, version, compat matrix) — HIGH
- `treelite-mainline/include/treelite/model_builder.h` (builder API, Metadata, TreeAnnotation, PostProcessorFunc) — HIGH
- `treelite-mainline/include/treelite/model_loader.h` (XGBoost/LightGBM/sklearn loader signatures) — HIGH
- `treelite-mainline/python/treelite/{model,frontend}.py`, `gtil/gtil.py`, `sklearn/importer.py` (Python API to mirror) — HIGH
- `.planning/PROJECT.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` (scope + subsystem map) — HIGH

---
*Feature research for: tree-ensemble model library (Treelite Rust port)*
*Researched: 2026-06-09*

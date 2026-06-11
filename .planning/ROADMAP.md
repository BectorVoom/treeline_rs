# Roadmap: treelite-rs

## Overview

treelite-rs is a strict numerical port of Treelite 4.7.0 (C++) to a Rust Cargo workspace, where the single non-negotiable invariant is that predictions match upstream within **1e-5**. The journey is structured as vertical MVP slices laid along the dependency spine the research mandated: enums → core model → builder/serialize → loaders → scalar GTIL + equivalence harness → cubecl CPU kernels → GPU backend → PyO3 → memory hardening. Phase 1 stands up the *thinnest possible end-to-end pipeline* (workspace + enums + minimal core + a minimal XGBoost-JSON load + a scalar identity/sigmoid predict + an equivalence harness asserting 1e-5 against a committed golden) so the core value is proven and de-risked on day one. Every subsequent phase **widens one layer of that proven spine** — more formats, the full GTIL surface, full serialization, then GPU acceleration, then the Python binding — and each widening phase ends in a runnable, equivalence-tested state. The dependency DAG of the upstream C++ leaves no room to reorder: a slice cannot use a crate that does not yet exist.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: End-to-End Spine** - Workspace + enums + minimal core + minimal XGBoost-JSON load → scalar predict → 1e-5 golden verify (completed 2026-06-09)
- [x] **Phase 2: Builder & Serialization** - Validated `ModelBuilder` + bulk path + v5 binary/PyBuffer/JSON round-trip and field accessors (completed 2026-06-10)
- [x] **Phase 3: Full XGBoost Loaders** - JSON + UBJSON + legacy binary with auto-detect and version-gated base_score margin transform (completed 2026-06-10)
- [x] **Phase 4: LightGBM & scikit-learn Loaders** - LightGBM text + RF/ET/GBM/IsolationForest + HistGradientBoosting (completed 2026-06-10)
- [x] **Phase 5: Full Scalar GTIL & Equivalence Harness** - All 4 predict kinds, 10 postprocessors, sparse CSR, categoricals, output shaping, seeded golden harness at 1e-5 (plans 5/5 built + 2 gap-closure plans 05-06/05-07 for CR-01 + WR-01…WR-06) (completed 2026-06-10)
- [x] **Phase 6: cubecl GTIL Kernels (CPU Backend)** - Traversal + postprocessor kernels; CPU backend default, validated to 1e-5 with zero-copy SoA upload (completed 2026-06-10)
- [x] **Phase 7: GPU Backend & Equivalence Report** - Runtime-selectable GPU backend (CUDA/wgpu/ROCm) with a documented per-model-class deviation report (completed 2026-06-10)
- [ ] **Phase 8: PyO3 Python Binding** - load/predict/serialize/dump from Python with zero-copy numpy I/O and abi3 wheel
- [ ] **Phase 9: Memory-Efficiency Hardening** - bytemuck zero-copy recast, smallvec/compact_str, custom global allocator

## Phase Details

### Phase 1: End-to-End Spine

**Goal**: Prove the core value early — a model can be loaded, predicted, and verified within 1e-5 against a committed C++ golden — by standing up the thinnest end-to-end slice through the whole pipeline.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: FND-01, FND-02, ENUM-01, CORE-01, CORE-02, CORE-03, CORE-04, ERR-01, ERR-02
**Success Criteria** (what must be TRUE):

  1. `cargo build` and `cargo test` succeed across all workspace member crates under edition 2024 / resolver "3", with every third-party crate pinned to a current stable version in a single `[workspace.dependencies]` table (no pre-release on the critical path).
  2. `TaskType`, `TreeNodeType`, `Operator`, and `DType` round-trip to/from their upstream string values, asserted against values read from `treelite-mainline`.
  3. A `Model` exists as a two-variant enum over `<f32,f32>`/`<f64,f64>` presets, holding a `Tree<T>` whose ~20 node fields are stored as parallel SoA `TreeBuf<T>` columns in both Owned and Borrowed modes, carrying full header metadata (num_feature, task_type, num_class, leaf_vector_shape, target/class ids, postprocessor, sigmoid_alpha, ratio_c, base_scores, average_tree_output, attributes).
  4. A minimal walking skeleton loads one simple XGBoost-JSON model into that `Model`, runs a scalar single-threaded predict with identity/sigmoid postprocessing only, and the equivalence-harness skeleton asserts the output is within 1e-5 of a committed golden vector (with a frozen toolchain/libm manifest).
  5. Library crates surface typed `thiserror` errors at their boundaries; the harness/binaries use `anyhow` for context.

**Plans**: 4 plansPlans:
**Wave 1**

- [x] 01-01-PLAN.md — Workspace scaffold + four enums + Tree/TreeBuf SoA core + Model header + committed fixture & frozen golden (Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — XGBoost-JSON loader: parse fixture into a Model, objective→postprocessor map, f64 base_score margin transform (Wave 2)
- [x] 01-03-PLAN.md — Scalar GTIL predict: EvaluateTree + NextNode + serial tree-sum + f64 base_score add + identity/sigmoid (Wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-04-PLAN.md — Equivalence harness: load→predict→assert within 1e-5 of golden, report max deviation, manifest check (Wave 3)

### Phase 2: Builder & Serialization

**Goal**: Widen the construction and persistence layers along the spine — a fluent validated `ModelBuilder` (plus concatenate and a bulk fast path) and full v5 serialization — so loaders have a construction target and models round-trip.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: BLD-01, BLD-02, BLD-03, SER-01, SER-02, SER-03, SER-04
**Success Criteria** (what must be TRUE):

  1. A fluent `ModelBuilder` constructs a model node-by-node and rejects orphaned/ill-formed topologies with a typed error; the skeleton XGBoost-JSON loader from Phase 1 is rewired to build through it and still verifies within 1e-5.
  2. `ConcatenateModelObjects` merges multiple built models into one, and a `BulkConstructTree` fast path builds trees from pre-validated bulk input (bypass behavior documented to match upstream).
  3. A model round-trips through the v5 binary format (serialize → deserialize → identical model) and serializes to the v5 zero-copy PyBuffer representation.
  4. `DumpAsJSON` emits a model as JSON and field accessors expose model/tree fields for inspection.

**Plans**: 7 plans (5 shipped + 2 gap closure)
Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Core Model/Tree private bookkeeping fields + frozen D-02 golden v5 blob capture (Wave 1) ✅

**Wave 2** *(blocked on Wave 1)*

- [x] 02-02-PLAN.md — treelite-builder crate: fluent ModelBuilder (BLD-01) + ConcatenateModelObjects (BLD-02) + BulkConstructTree (BLD-03) (Wave 2) ✅
- [x] 02-03-PLAN.md — Core serialize module: v5 binary round-trip + golden byte-compare (SER-01) + bounds-checked deserialize (D-03) + zero-copy PyBuffer frames (SER-02) (Wave 2)

**Wave 3** *(blocked on Wave 2)*

- [x] 02-04-PLAN.md — DumpAsJSON (SER-03) + Model/Tree field accessors (SER-04) (Wave 3) ✅
- [x] 02-05-PLAN.md — Rewire XGBoost loader through ModelBuilder (D-11) + 1e-5 regression gate (Wave 3) ✅

**Gap closure** *(post-verification)*

- [x] 02-06-PLAN.md — Builder byte-fidelity: end_tree emits AllocNode per-node columns (CR-01) + empty-unless-set stats (CR-02), closing 02-VERIFICATION criterion 3 PARTIAL ✅

### Phase 3: Full XGBoost Loaders

**Goal**: Widen the loader layer to the full XGBoost surface — all three formats with auto-detection and the version-gated base_score margin transform — proven across formats against the richest fixture set.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: XGB-01, XGB-02, XGB-03, XGB-04, XGB-05
**Success Criteria** (what must be TRUE):

  1. Loading `treelite-mainline/tests/examples/` XGBoost models in JSON, UBJSON, and legacy-binary form each produces a `Model`; the same logical model loaded from all three formats predicts within 1e-5 of the shared golden.
  2. The loader auto-detects which XGBoost format a file is, with the UBJSON path sharing the JSON numeric state machine for parity (NaN/Inf literals accepted as upstream does) and legacy binary read via explicit little-endian decoders (no native-endian struct transmute).
  3. XGBoost objective maps to the correct postprocessor, and the version-gated (`major_version >= 1`) base_score probability→margin transform is applied (scalar and vector base_score forms handled), with no constant offset on predictions.

**Plans**: 4 plans
Plans:
**Wave 0**

- [x] 03-01-PLAN.md — Three-format fixtures + shared prediction golden + single v5 byte-fidelity golden + frozen manifest (A1/A2 generation spike) + RED 3-format test scaffold (Wave 0) — DONE (3/3 tasks, A1=xgboost 1.7.6)

**Wave 1** *(blocked on Wave 0)*

- [x] 03-02-PLAN.md — JSON slice: widen recognized key set, NaN/Inf (D-02), shared build_model_from_parsed, sum_hess/gain/attributes (D-10 JSON leg), scalar/vector base_score version gate (XGB-01/XGB-05) (Wave 1) — DONE (2/2 tasks; DEF-02-01 JSON byte-fidelity closed; json_/nan_inf_ tests green, no workspace regression)

**Wave 2** *(blocked on Wave 1)*

- [x] 03-03-PLAN.md — UBJSON slice: hand-rolled tag decoder → serde_json::Value + DetectXGBoostFormat, converging at the shared structs (XGB-02/XGB-04) (Wave 2) — DONE (2/2 tasks; UBJSON byte-fidelity closed: load_xgboost_ubjson==load_xgboost_json==golden_v5_3format.bin + 1e-5; detect_/ubjson_ tests green; legacy leg of three_format_equivalence stays RED until 03-04)

**Wave 3** *(blocked on Wave 2)*

- [x] 03-04-PLAN.md — Legacy-binary slice: LE byte cursor + PeekableReader (no transmute) + close DEF-02-01/D-10 across all three formats (XGB-03/XGB-05) (Wave 3) — DONE (2/2 tasks; load_xgboost_legacy via from_le_bytes cursor; mushroom smoke 1501B; DEF-02-01 fully closed: serialize(load_json)==serialize(load_ubjson)==serialize(load_legacy)==golden_v5_3format.bin; three_format_equivalence + cargo test --workspace fully green)

### Phase 4: LightGBM & scikit-learn Loaders

**Goal**: Widen loaders to LightGBM text format and the full scikit-learn estimator set (including the most complex path, HistGradientBoosting), so every supported source framework loads into the proven spine.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: LGB-01, LGB-02, LGB-03, SKL-01, SKL-02, SKL-03, SKL-04
**Success Criteria** (what must be TRUE):

  1. A LightGBM text-format model loads and predicts within 1e-5 of its golden, with categorical splits decoded from their bitset and per-field precision (leaf_value/threshold = f64, split_gain = f32) matching upstream.
  2. LightGBM objective maps to the correct postprocessor with parsed `sigmoid_alpha`, `class_id[i] = i % num_class` round-robin, and `average_output` honored.
  3. `RandomForest`/`ExtraTrees`, `GradientBoosting`, and `IsolationForest` (classifier + regressor where applicable) import from sklearn array dumps via the bulk path and predict within 1e-5 of their goldens.
  4. `HistGradientBoosting` (classifier + regressor) imports — including the bulk tree-construction path — and predicts within 1e-5 of its golden.

**Plans**: 8 plans
Plans:
**Wave 1** *(enablers + frozen golden capture — gate everything)*

- [x] 04-01-PLAN.md — f64 ModelBuilder mode + bulk→Model assembly (D-05 enabler) (Wave 1) — DONE (2/2 tasks; leaf_scalar_f64/leaf_vector_f64/numerical_test_f64 → ModelVariant::F64 no-downcast + bulk_to_model; f32 XGBoost path unchanged; treelite-lightgbm/treelite-sklearn registered; cargo test --workspace green)
- [x] 04-02-PLAN.md — GTIL output-shaping/averaging/base-score add + softmax/exp-std-ratio/exponential/log1pexp postprocessors (D-03 enabler) (Wave 1) — DONE (2/2 tasks; predict widened to flat (num_row,num_target,max_num_class) routed by target_id/class_id + RF averaging + f64 2D base-score add; 4 postprocessors verbatim cast order incl. exp2; bounds-checked routing → typed GtilError; binary (num_row,1,1) byte-identical, cargo test --workspace green)
- [x] 04-03-PLAN.md — Frozen per-estimator goldens from treelite.gtil.predict + version-pinned manifests (D-06/D-07) (Wave 1)

**Wave 2** *(blocked on Wave 1)*

- [x] 04-04-PLAN.md — LightGBM numerical slice: parser + objective→postprocessor map + node-id reassignment, 1e-5 golden (LGB-01/LGB-03) (Wave 2) — DONE (2/2 tasks; treelite-lightgbm crate: load_lightgbm → ModelVariant::F64 via f64 builder, negative-index leaf BFS re-numbering + missing-type default_left override + CanonicalObjective map + sigmoid_alpha + class_id round-robin; lightgbm_numerical golden max |delta| = 0e0 < 1e-5; cargo test --workspace green)
- [x] 04-06-PLAN.md — sklearn crate (D-01 array signatures) + RF/ExtraTrees bulk path + GradientBoosting MixIn, 1e-5 goldens (SKL-01/SKL-02) (Wave 2)

**Wave 3** *(blocked on Wave 2)*

- [x] 04-05-PLAN.md — LightGBM categorical bitset (BitsetToList) + minimal NextNodeCategorical GTIL, 1e-5 golden (LGB-02) (Wave 3) — DONE (2/2 tasks; bitset.rs ports BitsetToList verbatim (word=bits[i/32] bit=i%32, T-04-11 in-bounds); cat_threshold u32 / cat_boundaries u64 exact precision with length+monotone validation → LgbError::Bitset (T-04-10); builder categorical_test extended to carry the category list + polarity, CSR columns filled at end_tree; load_lightgbm emits categorical_test (cat_idx=threshold[node], default_left=false, NaN→right); minimal NextNodeCategorical GTIL branch (membership + polarity, GTIL-06 deferred) + bounds-safe category_list_safe (T-04-12); lightgbm_categorical golden max |delta| = 9.54e-7 < 1e-5; cargo test --workspace green)
- [x] 04-07-PLAN.md — IsolationForest MixIn (exponential_standard_ratio + ratio_c), golden == -score_samples 1e-5 (SKL-03) (Wave 3)

**Wave 4** *(blocked on Wave 3 — the HistGB tentpole)*

- [x] 04-08-PLAN.md — HistGradientBoosting packed-node decode (52/56) + features_map + categories_map; numerical then categorical, 1e-5 goldens (SKL-04) (Wave 4)

### Phase 5: Full Scalar GTIL & Equivalence Harness

**Goal**: Widen the inference spine to the complete scalar GTIL reference — all predict kinds, all postprocessors, sparse input, categoricals, output shaping — and the full seeded equivalence harness that is the 1e-5 measurement instrument for everything after.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: GTIL-01, GTIL-02, GTIL-03, GTIL-04, GTIL-05, GTIL-06, GTIL-07, GTIL-08, EQV-01, EQV-02, EQV-03, EQV-04
**Success Criteria** (what must be TRUE):

  1. Prediction works over a dense row-major matrix and over a sparse CSR matrix (absent entries materialized as NaN, not 0), with dense↔sparse parity asserted on identical logical data.
  2. All four predict kinds (`default`, `raw`, `leaf_id`, `score_per_tree`) and all ten postprocessors are ported verbatim (mixed-precision softmax/exp2/log1p preserved), with NaN-only missing-value routing and the categorical float-representability guard + child polarity matching upstream.
  3. Output shaping is correct — `GetOutputShape` per kind, leaf-vector broadcast, tree averaging, f64 base-score addition — and per-row tree summation is serial in `tree_id` order (parallelism only across rows).
  4. The harness generates random seeded dense + sparse CSR inputs, compares against C++-captured golden vectors (committed with a toolchain/libm manifest) across model types, both presets, and all predict kinds, asserting within 1e-5 and reporting the max observed deviation.

**Plans**: 7 plans (5 shipped + 2 gap closure)
Plans:
**Wave 1** *(Wave 0 — frozen contract; gates everything)*

- [x] 05-01-PLAN.md — Seeded exhaustive-matrix capture (dense + CSR, both input dtypes, all 4 kinds, edge-seeded) + frozen `fixtures/gtil/` goldens with `backend: scalar-cpu` manifest + RED matrix test & unit scaffolds (EQV-01/EQV-02)

**Wave 2** *(blocked on Wave 1)*

- [x] 05-02-PLAN.md — GTIL foundation: typed `Config`/`PredictKind` (D-06) + public `Shape`/`output_shape` (D-07) + `O`-generic input/output element (f64 input over both presets, D-05); internal `Shape`→`OutputLayout` rename (GTIL-01/GTIL-03/GTIL-07/GTIL-08)

**Wave 3** *(blocked on Wave 2 — same lib.rs)*

- [x] 05-03-PLAN.md — 3 remaining postprocessors (`signed_square`/`hinge`/`multiclass_ova`) verbatim + full categorical float-representability guard (2^24 f32 / 2^32-1 f64) + child polarity; NaN→default routing confirmed (GTIL-04/GTIL-05/GTIL-06)

**Wave 4** *(blocked on Wave 3 — same lib.rs)*

- [x] 05-04-PLAN.md — Sparse CSR accessor (absent=NaN, dense==sparse parity D-04) + `LeafId`/`ScorePerTree` predict kinds; bounds-checked `col_ind`/`row_ptr` → typed errors (GTIL-02/GTIL-03)

**Wave 5** *(blocked on Wave 4 — full GTIL surface)*

- [x] 05-05-PLAN.md — Exhaustive equivalence-matrix runner (1e-5 across model×preset×dtype×kind×{dense,sparse}×seed) + dense==sparse parity + max-dev report + `Manifest` `backend` field & backend-parameterized seam (D-09/D-11) (EQV-03/EQV-04)

**Wave 6** *(gap closure — engine fixes; blocked on the shipped GTIL surface)*

- [x] 05-06-PLAN.md — CR-01 f64-input postprocessors run in f64 (softmax stays f32) + WR-02 shape/predict third-dim clamp agreement + WR-03 0-node-tree guard + WR-04 typed errors for malformed category-list/leaf-vector offsets + WR-05 `UnrecognizedOperator` on `kNone` (GTIL-04/05/06/07, ERR-01)

**Wave 7** *(gap closure — fixtures + harness; blocked on 05-06 engine fixes)*

- [x] 05-07-PLAN.md — CR-01 large-margin f64 sigmoid/exponential fixture asserted to 1e-5 + WR-01 frozen CSR triple consumed verbatim by sparse cells + WR-06 f32/f64 paired divergence assertion (GTIL-02/04, EQV-01/02/03/04)

### Phase 6: cubecl GTIL Kernels (CPU Backend)

**Goal**: Reimplement the GTIL hot path (traversal + postprocessors) as cubecl kernels with the CPU backend as the deterministic default, validated to 1e-5 against the green scalar reference — the project's compute spine widened onto cubecl.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: GPU-01, GPU-02, GPU-05
**Success Criteria** (what must be TRUE):

  1. Tree traversal and the postprocessor set run as `#[cube(launch)]` kernels generic over `R: Runtime`, with one unit per row looping over trees serially (no `atomicAdd`/reduce over the tree axis, no `continue`).
  2. The cubecl CPU backend is the default and the full equivalence harness passes within 1e-5 on it in CI, with output bit-identical across two runs of the same input (determinism check).
  3. SoA model buffers upload host→device via `TreeBuf::as_bytes()` + `client.create_from_slice` with per-column ragged-SoA concatenation across the forest (no per-tree handle explosion), and a plain-Rust fallback exists for any unimplemented cubecl op.

**Plans**: 7 plans (5 + 2 gap-closure)
Plans:
**Wave 1** *(Wave 0 — crate scaffold + as_bytes + RED scaffolds; gates everything)*

- [x] 06-01-PLAN.md — treelite-cubecl crate scaffold + cubecl 0.10.0/bytemuck pin + TreeBuf::as_bytes() + RED test scaffolds (spike/upload/determinism/matrix sibling) (Wave 1)

**Wave 2** *(blocked on Wave 1 — the mandatory spike)*

- [x] 06-02-PLAN.md — D-04 spike: break-free #[cube] descent + 2-tree default kernel + softmax_f64/exp2 micro-kernels vs scalar twins to 1e-5 (retires A1–A4) (Wave 2) — DONE (2/2 tasks; kernels/traversal.rs descend<F: Float> break-free helper Wave 3 reuses; spike 3/3 green: 2-tree default f32+f64 vs predict, exp_standard_ratio f32+f64, softmax_f64 all within 1e-5 on CpuRuntime; A1 resolved via exp(x*ln2) identity NOT direct exp2 — cube frontend has Exp, no Exp2; NaN via fv!=fv NOT F::is_nan; ABSOLUTE_POS as u32; Float scalars ride as 1-element Array<F>; cargo test --workspace green)

**Wave 3** *(blocked on Wave 2 — building blocks)*

- [x] 06-03-PLAN.md — Per-column ragged-SoA upload (one handle/column, SC3/GPU-05) + all 10 #[cube] postprocessors verbatim cast order to 1e-5 (D-03) (Wave 3)

**Wave 4** *(blocked on Wave 3 — the dense kernel slice)*

- [x] 06-04-PLAN.md — #[cube(launch)] kernels for all 4 predict kinds + leaf-vector broadcast (D-01) + predict_cpu host launcher with categorical/sparse scalar fallback (D-02), f32+f64 to 1e-5 (Wave 4)

**Wave 5** *(blocked on Wave 4 — registration capstone)*

- [x] 06-05-PLAN.md — Backend::CubeclCpu + cubecl_cpu_case() registration + gtil_matrix_cubecl sibling gate (frozen goldens to 1e-5 + per-cell provenance D-06) + SC2 determinism (Wave 5)

**Wave 6** *(gap closure — fixes the 3 BLOCKERs from 06-VERIFICATION.md)*

- [x] 06-06-PLAN.md — Gap closure: CR-01 non-kLT operator fallback gate + CR-02 f64-promoted descend() comparison + CR-03 leaf-vector span validation (MalformedLeafVector) + matrix provenance honesty + malformed-model test (Wave 6)

**Wave 7** *(blocked on Wave 6 — real-upstream-golden lock)*

- [x] 06-07-PLAN.md — Gap closure: capture kLE LightGBM-numerical + f32-unrepresentable-threshold mixed-width fixtures from upstream Treelite, re-run cubecl matrix gate within 1e-5 on the previously-uncovered classes, full workspace green (Wave 7)

### Phase 7: GPU Backend & Equivalence Report

**Goal**: Layer runtime-selectable GPU backends onto the green CPU equivalence and document numerical behavior — proving GPU acceleration in v1 without making it a correctness prerequisite. **ROCm is the hardware-validated backend** (the developer has an AMD/ROCm device, no NVIDIA); CUDA is implemented and runtime-selectable but validated only where an NVIDIA device is available.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: GPU-03, GPU-04
**Success Criteria** (what must be TRUE):

  1. **ROCm** is selectable at runtime via Cargo feature + `Backend` enum and produces predictions for the harness model set on the developer's AMD/ROCm hardware (this is the backend that satisfies GPU-03's "at least one ... produces predictions"). CUDA and wgpu share the same generic `R: Runtime` seam and are runtime-selectable from the same enum; **CUDA is build-supported but not locally hardware-validated** (no NVIDIA device) — its predictions are exercised only where such hardware exists (e.g. CI), and a missing CUDA device is a skip, not a failure.
  2. A committed GPU equivalence report documents the observed max deviation per model class **on ROCm** against an accepted tolerance, noting where f64 postprocessor fallback is needed to stay in budget. CUDA/wgpu rows are filled wherever their hardware is available and explicitly marked "not run — no device" otherwise.
  3. CPU remains the default backend and small inputs do not pay GPU transfer/launch overhead (documented crossover heuristic).

**Plans**: 4 plans
Plans:
**Wave 0** *(infrastructure spike — gates backend registration)*

- [x] 07-01-PLAN.md — rocm/cuda/wgpu cargo features + CubeclError::DeviceUnavailable + device.rs client constructor + A3 device-absence spike (D-04/D-05) (Wave 0)

**Wave 1** *(blocked on Wave 0 — same treelite-cubecl lib.rs/device.rs)*

- [x] 07-02-PLAN.md — Generalize host launcher predict_cpu → predict::<R> + predict_cpu shim; CPU 1e-5 gate byte-identical (GPU-03) (Wave 1)

**Wave 2** *(blocked on Wave 1 — needs predict::<R>)*

- [x] 07-03-PLAN.md — Register Backend::Rocm/Cuda/Wgpu + rocm_case()/cuda_case()/wgpu_case() in harness; ROCm selectable end-to-end (GPU-03/SC1) (Wave 2)

**Wave 3** *(blocked on Wave 2 — same harness lib.rs; hardware-gated checkpoint)*

- [x] 07-04-PLAN.md — report.rs + gtil_matrix_gpu.rs sibling → committed GPU_EQUIVALENCE_REPORT.md (GPU-04/SC2) + gpu_crossover.rs → GPU_CROSSOVER.md (SC3/D-09/D-10); regenerated on ROCm hardware (Wave 3)

**Research flag (resolved):** GPU transcendental/FMA divergence profiling + cubecl FMA contraction completed in 07-RESEARCH.md (FMA has ~0 surface on the pure-add tree-sum; divergence is transcendental-only, predicted-deviation model in the report's predicted column).

### Phase 8: PyO3 Python Binding

**Goal**: Expose the proven Rust pipeline to Python as the sole external binding — load, predict, serialize, dump, and sklearn marshalling — with zero-copy numpy I/O and an abi3 wheel.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: PY-01, PY-02, PY-03, PY-04, PY-05, PY-06, MEM-04
**Success Criteria** (what must be TRUE):

  1. From Python, a user can load XGBoost / LightGBM / scikit-learn models, predict over numpy arrays with zero-copy buffer I/O, and serialize/deserialize/JSON-dump a model — with results matching the Rust path within 1e-5.
  2. A `sklearn.import_model` entry point marshals fitted estimators, and borrowed buffers from the Python buffer protocol are consumed zero-copy.
  3. Library `thiserror` errors translate into Python exceptions (no panic crosses the FFI boundary), and the binding builds and imports as an abi3 wheel via maturin.

**Plans**: 5 plans
Plans:
**Wave 0**

- [x] 08-01-PLAN.md — treelite-py crate scaffold + maturin abi3-py310 wheel (imports as `treelite_rs`) + pytest conftest & 7 RED test stubs (PY-06) (Wave 0)

**Wave 1** *(walking skeleton — blocked on Wave 0)*

- [x] 08-02-PLAN.md — Model pyclass + frontend loaders + zero-copy dense predict (predict_f32/_f64, GIL release) + single TreeliteError → live A/B 1e-5 (PY-01/PY-02/MEM-04) (Wave 1)

**Wave 2** *(blocked on Wave 1)*

- [ ] 08-03-PLAN.md — Model serialize_bytes/deserialize_bytes + dump_as_json + concatenate, 1e-5 round-trip (PY-03) (Wave 2)

**Wave 3** *(blocked on Wave 2)*

- [ ] 08-04-PLAN.md — sklearn.import_model: array-loader pyfunctions + estimator→arrays shim (port importer.py) → live A/B 1e-5 (PY-04) (Wave 3)

**Wave 4** *(blocked on Wave 3 — completes the phase)*

- [ ] 08-05-PLAN.md — Panic guard (no abort crosses FFI) + additive backend= kwarg + un-built/device-absent → TreeliteError + hardware-gated rocm wheel checkpoint (PY-05/D-05/D-08) (Wave 4)

**Research flag (resolved):** PyO3 0.28 buffer-protocol / numpy zero-copy return / GIL release (`Python::detach`) resolved in 08-RESEARCH.md.

### Phase 9: Memory-Efficiency Hardening

**Goal**: Apply the memory-efficiency techniques across the proven, equivalence-tested workspace without regressing the 1e-5 contract — closing out the last v1 requirements.
**Mode:** mvp
**Depends on**: Phase 8
**Requirements**: MEM-01, MEM-02, MEM-03
**Success Criteria** (what must be TRUE):

  1. SoA columns use `bytemuck` `Pod` zero-copy recasting where layout allows, with the full equivalence harness still green within 1e-5 after the change.
  2. `smallvec` and `compact_str` back small collections and metadata strings, verified by existing tests still passing.
  3. A custom global allocator (jemalloc) is wired into benchmarks/binaries and validated to import/run on Linux (and not enabled in a way that breaks the abi3 wheel).

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. End-to-End Spine | 4/4 | Complete    | 2026-06-09 |
| 2. Builder & Serialization | 6/6 | Complete    | 2026-06-10 |
| 3. Full XGBoost Loaders | 4/4 | Complete    | 2026-06-10 |
| 4. LightGBM & scikit-learn Loaders | 8/8 | Complete    | 2026-06-10 |
| 5. Full Scalar GTIL & Equivalence Harness | 7/7 | Complete    | 2026-06-10 |
| 6. cubecl GTIL Kernels (CPU Backend) | 7/7 | Complete    | 2026-06-10 |
| 7. GPU Backend & Equivalence Report | 4/4 | Complete    | 2026-06-10 |
| 8. PyO3 Python Binding | 1/5 | In Progress|  |
| 9. Memory-Efficiency Hardening | 0/TBD | Not started | - |

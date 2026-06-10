# Phase 4: LightGBM & scikit-learn Loaders - Research

**Researched:** 2026-06-10
**Domain:** Tree-ensemble model loaders (LightGBM text format, scikit-learn array dumps incl. HistGradientBoosting), f64 ModelPreset, minimal pulled-forward GTIL
**Confidence:** HIGH (porting source-of-truth is vendored read-only at `treelite-mainline/`; every claim below is line-cited from the C++ v4.7.0 source or existing Rust crates)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 — sklearn array signatures mirrored 1:1.** Rust sklearn loaders mirror the upstream C-API `namespace sklearn` signatures verbatim (array-of-arrays as `&[&[T]]`), so Phase-8 PyO3 can hand them borrowed numpy buffers zero-copy. No intermediate file format. Faithful port of `treelite-mainline/include/treelite/model_loader.h` lines ~103–345.
- **D-02 — emit through `ModelBuilder` (converge-then-build).** Reuse the Phase-2/3 builder + the `BulkConstructTree` fast path where upstream uses `sklearn_bulk.cc`.
- **D-03 — pull forward MINIMAL GTIL per loader.** Port just enough GTIL surface to assert real 1e-5 parity per estimator family this phase (IsolationForest `exponential_standard_ratio`, multiclass output shaping + `softmax`, LightGBM `sigmoid`/`softmax`). Phase 5 widens to the complete surface and does NOT backfill Phase-4 basic parity.
- **D-04 — HistGB verified in full this phase** (see D-08); its required GTIL (bin-threshold eval path, postprocessor) is pulled forward under D-03.
- **D-05 — LightGBM and sklearn both map to `<f64,f64>` ModelPreset.** LightGBM `leaf_value`/`threshold` = f64 (`split_gain` = f32 metadata); sklearn `threshold`/`value` = f64. First end-to-end exercise of the f64 variant. Confirm exact upstream ThresholdType/LeafOutputType; do not silently downcast.
- **D-06 — one-time frozen `uv run python` capture.** A single Python session fits each estimator family / loads each LightGBM model, dumps node arrays (sklearn) + input matrix + frozen manifest (sklearn/lightgbm/treelite versions + seed), committed read-only. CI never regenerates.
- **D-07 — golden = upstream Treelite GTIL** (`treelite.gtil.predict`), NOT the framework's `predict()`. IsolationForest is the canonical "Treelite ≠ framework" case (golden == `-clf.score_samples(X)`). Framework predict recorded only as secondary sanity cross-check.
- **D-08 — full HistGB import + 1e-5 verify this phase.** Port `_bin_mapper` bin→threshold reconstruction, version-gated `_preprocessor`/`features_map` (embedded OrdinalEncoder) feature remapping, packed node-struct decode. Largest single chunk; the research-flagged risk.

### Claude's Discretion
- **Crate organization** — `treelite-lightgbm` + `treelite-sklearn` as parallels to `treelite-xgboost`, vs a combined loader crate. Planner's call (follow the established per-format-crate pattern).
- **LightGBM text-parse mechanics** — streaming/line-based parser shape, categorical-bitset decode, `string_utils` analog. Mirror `lightgbm.cc` + `detail/lightgbm.h`.
- **HistGB packed-node decode mechanics** — exact struct unpacking strategy is an implementation detail for research/planner to derive from upstream.

### Deferred Ideas (OUT OF SCOPE)
- **Complete GTIL surface** (all 4 predict kinds, all 10 postprocessors, sparse CSR, full categorical/output-shaping matrix) — Phase 5.
- **PyO3 marshalling of live fitted estimators** — Phase 8. Phase 4 tests loaders with frozen array-dump fixtures only.
- **Multi-target / multi-output sklearn beyond captured fixtures** — verify-narrow; broader coverage rides Phase 5.
- **LightGBM categorical-split PREDICTION parity beyond the captured fixture** — bitset decode is implemented (LGB-02), but exhaustive categorical evaluation parity aligns with Phase 5's categorical GTIL.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LGB-01 | Load a LightGBM text-format model | §LightGBM Parser — `lightgbm.cc:234-604` line-based key=value parse, tree node-id reassignment (negative-index leaf decode), `ParseStream` → `ModelBuilder`. Vendored fixture `deep_lightgbm/model.txt`. |
| LGB-02 | Categorical splits decode (bitset) + upstream per-field precision | §Categorical Bitset Decode — `BitsetToList` (`lightgbm.cc:210-221`), `cat_boundaries`/`cat_threshold` slicing (`:563-573`); per-field precision: `leaf_value`/`threshold`=f64, `split_gain`=f32, `decision_type`=i8, `cat_threshold`=u32, `cat_boundaries`=u64. |
| LGB-03 | Objective→postprocessor map (+`sigmoid_alpha`), `class_id` round-robin, `average_output` | §Objective→Postprocessor Map — `CanonicalObjective` (`detail/lightgbm.h:26-57`) + the postprocessor switch (`lightgbm.cc:442-515`); `class_id[i]=i%num_class` (`:427-429`); `average_output` from key presence (`:289-291`). |
| SKL-01 | Import RandomForest + ExtraTrees (clf + reg) | §sklearn Array Loaders + §Bulk Path — `sklearn_bulk.cc:232-350` (RF clf/reg direct Model assembly via `BulkConstructTree`). |
| SKL-02 | Import GradientBoosting (clf + reg) | §sklearn Array Loaders — `sklearn.cc:59-133, 385-415` (GB MixIns; binary→sigmoid, multiclass→softmax, leaf-shrink-by-learning-rate done capture-side). |
| SKL-03 | Import IsolationForest | §IsolationForest — `sklearn.cc:33-57, 373-383`; `ratio_c`/`expected_depth`/`calculate_depths` (`isolation_forest.py`); postprocessor `exponential_standard_ratio`. |
| SKL-04 | Import HistGradientBoosting (clf + reg) | §HistGB Decode (the tentpole) — `sklearn.cc:260-446` packed `HistGradientBoostingNode` struct, `features_map`, `categories_map`; capture side `importer.py:355-478`. |
</phase_requirements>

## Summary

Phase 4 widens the loader layer to two new source frameworks. **LightGBM** is a line-based `key=value` text parser whose only real subtleties are (1) LightGBM's negative-index leaf encoding, which upstream re-numbers into a clean depth-wise node sequence, (2) the categorical bitset decode, and (3) a ~15-branch objective→postprocessor map. It emits through the existing `ModelBuilder` exactly like XGBoost — but into the **`<f64,f64>` preset**, which the builder cannot currently produce (it hardcodes `Tree<f32>` / `ModelVariant::F32`). That is the single largest piece of shared enabling work.

**scikit-learn** splits into two paths matching upstream exactly: (a) the node-by-node `ModelBuilder` MixIn path (`sklearn.cc::LoadSKLearnModel`) used by IsolationForest and GradientBoosting, and (b) the **bulk** path (`sklearn_bulk.cc::LoadRandomForest{Classifier,Regressor}`) that assembles a `Model` directly from `BulkConstructTree` (already ported as `treelite-builder::bulk_construct_tree`, returning `Tree<f64>`, but never yet assembled into a `Model`). The hard part is **HistGradientBoosting** (D-08): a packed C node struct (`#pragma pack(1)`, 52 or 56 bytes depending on a 32-/64-bit feature-index field) decoded field-by-field, plus `features_map` feature re-ordering and an embedded `OrdinalEncoder` (`categories_map`) for categoricals.

The verify-narrow boundary (D-03) means Phase 4 must pull forward a small, exact GTIL slice: **multiclass / leaf-vector output shaping** (the current `predict` only handles scalar `(num_row,1,1)`), **tree averaging** (RF uses `average_tree_output=true`), the **f64 base-score-per-(target,class)** add, and four new postprocessors (`softmax`, `exponential_standard_ratio`, `exponential`, `logarithm_one_plus_exp`, plus `multiclass_ova` if a `multiclassova` fixture is captured). The golden is captured from `treelite.gtil.predict` in a frozen Python session — and `treelite==4.7.0` + `numpy` are already installed locally, but **`scikit-learn` and `lightgbm` are NOT** and must be added to the capture environment.

**Primary recommendation:** Sequence the phase as (Wave A) enable the f64 builder/bulk→Model path + pull forward GTIL output-shaping/averaging/postprocessors; then thin vertical slices per family in ascending risk order — LightGBM-numerical → GradientBoosting → RandomForest/ExtraTrees → IsolationForest → **HistGB last** (the tentpole). Each slice ends load→predict→verify-1e-5 against its own frozen upstream-GTIL golden.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LightGBM text parse → typed structs | `treelite-lightgbm` (new loader crate) | — | Per-format crate pattern (mirrors `treelite-xgboost`); converge-then-build at the struct layer (D-02). |
| sklearn array → Model (MixIn path) | `treelite-sklearn` (new loader crate) | `treelite-builder` (ModelBuilder) | IsolationForest/GradientBoosting go node-by-node through the builder (`sklearn.cc::LoadSKLearnModel`). |
| sklearn RF bulk array → Model | `treelite-sklearn` | `treelite-builder::bulk_construct_tree` | Upstream `sklearn_bulk.cc` bypasses the builder and assembles a Model directly from bulk trees (D-02). |
| HistGB packed-node decode + remap | `treelite-sklearn` | `treelite-builder` (ModelBuilder, categorical path) | Struct unpack + feature/category remap is loader logic; emission is through the builder's `categorical_test`/`numerical_test`. |
| f64 `<f64,f64>` Model construction | `treelite-builder` + `treelite-core` | — | Builder must gain an f64 mode (currently f32-only); core `ModelVariant::F64`/`Tree<f64>` already exist. |
| Output shaping / averaging / base-score add | `treelite-gtil` | `treelite-core` (metadata) | Reference inference assembly order is GTIL's responsibility (`predict.cc:174-305`). |
| Postprocessors (softmax, exp-std-ratio, …) | `treelite-gtil::postprocessor` | — | Verbatim port of `postprocessor.cc:19-82` (cast-ordering is the 1e-5 contract). |
| Golden capture (frozen) | `fixtures/` capture scripts (`uv run python`) | `treelite-harness` (assert) | One-time capture from upstream `treelite.gtil.predict` (D-06/D-07); harness asserts 1e-5. |

## Standard Stack

This is a **porting phase**, not a library-selection phase. The "stack" is the vendored C++ source-of-truth plus the existing Rust workspace crates. No new third-party Rust crates are required — the parser is hand-rolled line splitting and byte-cursor struct decode (same discipline as Phase 3's legacy decoder, D-07 there).

### Core (existing workspace crates — extend, do not replace)
| Crate | Role in Phase 4 | Why Standard |
|-------|-----------------|--------------|
| `treelite-core` | `ModelVariant::F64`, `Tree<f64>`, leaf_vector/category_list accessors already exist | The SoA spine all loaders target [VERIFIED: codebase grep `tree.rs:40-198`, `model.rs:44`] |
| `treelite-builder` | `ModelBuilder` (currently f32-only) + `bulk_construct_tree` (already `Tree<f64>`) + `concatenate` | The converge-then-build emission path (D-02) [VERIFIED: codebase grep `lib.rs:135,558`, `bulk.rs:48`] |
| `treelite-gtil` | Pull-forward target for output shaping + new postprocessors | Where verify-narrow GTIL lands (D-03) [VERIFIED: codebase `lib.rs:178-239`, `postprocessor.rs`] |
| `treelite-harness` | Per-estimator golden + manifest assertion pattern | Golden discipline (D-06/D-07) [VERIFIED: codebase `tests/golden_v5.rs`, `fixtures/`] |

### Supporting (new loader crates — planner's call per Claude's Discretion)
| Crate | Purpose | When to Use |
|-------|---------|-------------|
| `treelite-lightgbm` (proposed) | LightGBM text parser + objective map | Mirrors `treelite-xgboost` per-format pattern |
| `treelite-sklearn` (proposed) | sklearn array loaders (MixIn + bulk + HistGB) | Mirrors per-format pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Two separate crates (`treelite-lightgbm`, `treelite-sklearn`) | One combined `treelite-loaders` crate | Combined is fewer Cargo members but breaks the established 1-crate-per-format symmetry with `treelite-xgboost`; planner decides (Claude's Discretion). |
| Hand-rolled byte-cursor HistGB struct decode | A `bytemuck`/`zerocopy` `Pod` cast of the packed struct | `bytemuck` is deferred to Phase 9 (MEM-01); hand-rolled `from_le_bytes` field reads match the Phase-3 legacy precedent (D-07/D-08 there: "no native-endian transmute"). [CITED: 03-CONTEXT.md D-08] |
| New text-split crate | `std` `str::split`/`lines` | LightGBM lines are simple `key=value` and space-delimited arrays — `std` suffices (upstream uses `std::getline`/`istringstream`). [CITED: lightgbm.cc:157-180] |

**Installation (Rust):** No new third-party crates required for the loaders/GTIL. New workspace member crates are internal (`path` deps), pinned through the existing `[workspace.dependencies]`. `thiserror` (library errors) and `anyhow` (tests) are already in the workspace.

**Python capture environment (one-time, `uv run python`):** `scikit-learn` and `lightgbm` must be ADDED; `treelite==4.7.0` and `numpy` are already present.

```bash
# Capture-side only — never a Rust runtime or CI dependency (D-06).
uv pip install scikit-learn lightgbm   # treelite==4.7.0, numpy already installed
```

**Version verification:** `treelite 4.7.0` and `numpy 2.4.6` confirmed installed via `uv run python` [VERIFIED: `uv run python -c "import treelite"` → 4.7.0]. `scikit-learn` and `lightgbm` are NOT installed [VERIFIED: `ModuleNotFoundError: No module named 'sklearn'`] — the capture plan must pin and record their versions in the manifest (sklearn ≥1.4.0 changes the HistGB `_preprocessor` contract; ≥1.7.0 adds a 3rd "remainder" transformer — see Pitfall 4).

## Package Legitimacy Audit

> No new third-party **Rust** packages are introduced in this phase (loaders + GTIL are hand-rolled over `std`, emitting through existing workspace crates). The only external packages are **Python capture-side, dev-only** (`scikit-learn`, `lightgbm`), which never enter the Rust build graph, the shipped artifact, or CI runtime.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| scikit-learn | PyPI | 18+ yrs | ~80M/mo | github.com/scikit-learn/scikit-learn | not run (sandbox) | Approved — capture-only, ubiquitous [ASSUMED] |
| lightgbm | PyPI | 9+ yrs | ~8M/mo | github.com/microsoft/LightGBM | not run (sandbox) | Approved — capture-only, Microsoft-maintained [ASSUMED] |
| treelite | PyPI | (installed 4.7.0) | — | github.com/dmlc/treelite | n/a (already pinned) | Approved — the golden source-of-truth |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*slopcheck could not be installed/run in this sandbox; both Python packages are tagged `[ASSUMED]`. They are universally-known, first-party-maintained scientific packages used ONLY in the frozen one-time capture step (D-06), so the slopsquat risk is negligible — but the capture plan should still pin exact versions and record SHAs in the manifest. No Rust dependency gate is needed.*

## Architecture Patterns

### System Architecture Diagram

```
LightGBM .txt ──► [treelite-lightgbm]                       scikit-learn array dumps (frozen fixtures)
                   line-based key=value parse                 │
                   → LGBTree structs                          ├─ RF/ExtraTrees ─► [treelite-sklearn::bulk]
                   → objective→postprocessor map              │                    BulkConstructTree → Model<f64>  (sklearn_bulk.cc)
                   → node-id reassignment (neg-idx leaves)    │
                   → ModelBuilder<f64>                        ├─ GradientBoosting ─► [treelite-sklearn::mixin]
                        │                                     │      LoadSKLearnModel → ModelBuilder<f64>  (sklearn.cc)
                        │                                     │
                        │                                     ├─ IsolationForest ─► [treelite-sklearn::mixin]
                        │                                     │      ratio_c, isolation-depth leaves → ModelBuilder<f64>
                        │                                     │
                        │                                     └─ HistGB ─► [treelite-sklearn::histgb]
                        │                                            packed-node struct decode (52/56B)
                        │                                            + features_map + categories_map
                        ▼                                            → ModelBuilder<f64> (numerical/categorical tests)
              ┌──────────────────────────────────────────────────────────────┐
              │  treelite-core: Model { ModelVariant::F64(ModelPreset<f64>) }  │
              │  metadata: task_type, num_class[], target_id[], class_id[],     │
              │            leaf_vector_shape, postprocessor, base_scores[],      │
              │            average_tree_output, sigmoid_alpha, ratio_c           │
              └──────────────────────────────────────────────────────────────┘
                        │
                        ▼
              [treelite-gtil] PredictRaw (predict.cc:231-305)
                 EvaluateTree (+ NextNodeCategorical for HistGB/LGB cat)
                 → OutputLeafValue / OutputLeafVector (output shaping by target_id/class_id)
                 → tree averaging (RF: average_tree_output)
                 → += base_scores[target,class]   (f64)
                 → ApplyPostProcessor (softmax / sigmoid / exponential_standard_ratio / ...)
                        │
                        ▼
              [treelite-harness] assert |rust − golden| < 1e-5
                 golden = treelite.gtil.predict(frozen fixtures)   (D-07)
```

File-to-implementation mapping lives in the Component Responsibilities of the Responsibility Map above; the diagram shows data flow only.

### Recommended Project Structure
```
crates/
├── treelite-lightgbm/      # NEW (proposed) — mirrors treelite-xgboost
│   └── src/
│       ├── lib.rs          # load_lightgbm(&str) -> Result<Model, LgbError>
│       ├── parse.rs        # line tokenizer + LGBTree structs (lightgbm.cc:234-414)
│       ├── objective.rs    # CanonicalObjective + objective→postprocessor map
│       ├── bitset.rs       # BitsetToList categorical decode (lightgbm.cc:210-221)
│       └── error.rs        # LgbError (thiserror)
├── treelite-sklearn/       # NEW (proposed)
│   └── src/
│       ├── lib.rs          # the array-signature entry points (D-01)
│       ├── mixin.rs        # IsolationForest / GradientBoosting MixIn path (sklearn.cc)
│       ├── bulk.rs         # RF/ExtraTrees direct-Model assembly (sklearn_bulk.cc)
│       ├── histgb.rs       # packed-node decode + features_map + categories_map (D-08)
│       └── error.rs        # SklError (thiserror)
├── treelite-builder/       # EXTEND: add f64 ModelBuilder mode + bulk→Model assembly
└── treelite-gtil/          # EXTEND: output shaping, averaging, new postprocessors
fixtures/
├── capture_sklearn.py      # NEW — one-time frozen capture (uv run python)
├── capture_lightgbm.py     # NEW
├── sklearn_*.golden.json   # per-estimator goldens (input + treelite-GTIL output + manifest)
└── lightgbm_*.{txt,golden.json}
```

### Pattern 1: Converge-then-build (inherited from XGBoost)
**What:** parse → typed intermediate structs → validators → `ModelBuilder` emission → `CommitModel`.
**When to use:** LightGBM and the sklearn MixIn path (IsolationForest, GradientBoosting, HistGB).
**Example (LightGBM node emission, the negative-index leaf reassignment):**
```rust
// Source: treelite-mainline/src/model_loader/lightgbm.cc:533-601
// LightGBM uses negative indices to mark leaves; upstream re-numbers nodes into
// a depth-wise 1,2,3,... sequence via a BFS deque of (old_id, new_id) pairs.
// old_node_id < 0  =>  leaf, value = leaf_value[~old_node_id]  (bitwise NOT)
// new_node_id starts at 1 (NOT 0) — root is new id 1, dfs_index increments by 2 per internal node.
```

### Pattern 2: Bulk direct-Model assembly (RandomForest / ExtraTrees)
**What:** bypass the per-node `ModelBuilder`; build `Tree<f64>` columns in bulk, then assemble the `Model` and set metadata fields directly.
**When to use:** RF/ExtraTrees clf+reg (`sklearn_bulk.cc`), where upstream sets `model->postprocessor`, `num_class`, `leaf_vector_shape`, `target_id`/`class_id` by hand.
**Example (the metadata RF sets directly):**
```rust
// Source: sklearn_bulk.cc:244-270 (classifier)
// task_type = kMultiClf;  average_tree_output = true;
// num_class = n_classes_vec;  leaf_vector_shape = {n_targets, max_num_class};
// target_id = vec![-1; n_estimators];  class_id = vec![-1; n_estimators];   // BOTH -1 => leaf-vector broadcast
// postprocessor = "identity_multiclass";  base_scores = vec![0.0; n_targets * max_num_class];
// Regressor (sklearn_bulk.cc:309-330): task=kRegressor, num_class=vec![1;n_targets],
//   leaf_vector_shape={n_targets,1}, target_id=vec![n_targets>1?-1:0], class_id=vec![0], postprocessor="identity".
```
**Note (RF classifier leaf normalization):** the bulk path normalizes each leaf's class counts to probabilities (`norm_factor`) at construction time (`sklearn_bulk.cc:145-159`) — this is a load-time transform, not a GTIL transform. `treelite-builder::bulk_construct_tree` already ports this [VERIFIED: codebase `bulk.rs`].

### Pattern 3: Packed-struct field-by-field decode (HistGB, D-08)
**What:** the HistGB `nodes` array is a raw byte buffer of `#pragma pack(1)` C structs. Decode each field at its byte offset with `from_le_bytes` (no transmute — same discipline as Phase-3 legacy, D-08 there).
**The struct (`sklearn.cc:260-282`), packed, 52 bytes (FeatureIdT=i32) or 56 bytes (FeatureIdT=i64):**
```
offset(i32 variant)  field
 0   double  value             (8)
 8   uint32  count             (4)
12   int32/64 feature_idx      (4 or 8)   ← FeatureIdT; itemsize selects 52 vs 56
16/20 double num_threshold     (8)
24/28 uint8  missing_go_to_left(1)
25/29 uint32 left              (4)
29/33 uint32 right             (4)
33/37 double gain              (8)
41/45 uint32 depth             (4)
45/49 uint8  is_leaf           (1)
46/50 uint8  bin_threshold     (1)
47/51 uint8  is_categorical    (1)
48/52 uint32 bitset_idx        (4)
                                = 52 (i32) / 56 (i64)
```
The `expected_sizeof_node_struct` argument (Python passes `sub_estimator.nodes.itemsize`, `importer.py:427`) selects which `FeatureIdT` to decode — `static_assert(sizeof == 52)` / `== 56` upstream (`sklearn.cc:281-282`).

**Decode rules (`sklearn.cc:314-346`):**
- Leaf iff `left <= 0` (note: `<= 0`, not `== -1`, because `left` is `uint32` here).
- `split_index = features_map[node.feature_idx]` — feature re-ordering ALWAYS applied.
- `default_left = (missing_go_to_left == 1)`.
- If `is_categorical == 1`: walk `i in 0..256`, test bit via `check(left_cat_bitmap, i, bitset_idx)` where `check(bitmap,val,row) = (bitmap[8*row + val/32] >> (val%32)) & 1` (`sklearn.cc:296-298`); for each set bit push `cat_transform(feature_idx, i)`; `cat_transform = categories_map[fid][cat]` if `categories_map` present, else identity (`:300-305`). Emit `CategoricalTest(split_index, default_left, left_categories, /*right_child=*/false, left, right)`.
- Else `NumericalTest(split_index, num_threshold, default_left, kLE, left, right)`.
- Note: upstream decodes the **already-reconstructed `num_threshold` (a double)** straight from the struct — the `_bin_mapper` bin→threshold reconstruction happens **capture-side in numpy** (`importer.py`), NOT in the C++ loader. The Rust loader consumes `num_threshold` directly; `known_cat_bitsets`/`known_cat_bitsets_offset_map` are passed but the v4.7.0 loader does NOT use them in `LoadHistGradientBoostingImpl` (they're captured for forward-compat). [VERIFIED: `sklearn.cc:284-350` reads only `node.num_threshold`; `known_cat_bitsets` params unused in impl]

### Anti-Patterns to Avoid
- **Downcasting LightGBM/sklearn to f32.** D-05 is explicit: both are `<f64,f64>`. The builder currently produces only `Tree<f32>`/`ModelVariant::F32` — DO NOT route the new loaders through that path unchanged; add an f64 mode. Downcasting will blow the 1e-5 bound on f64-precision thresholds/leaves.
- **Using `== -1` for HistGB leaf detection.** HistGB `left` is `uint32`; upstream uses `left <= 0` (`sklearn.cc:320`). sklearn array loaders (non-HistGB) DO use `== -1` (`children_left` is `int64`, `sklearn.cc:229`). Don't mix them.
- **Transmuting the packed node buffer onto a native struct.** Carries the same alignment/endianness hazard Phase 3 banned (03-CONTEXT D-08). Decode field-by-field.
- **Applying `learning_rate` leaf-shrink in the Rust loader.** For GradientBoosting it's done capture-side (`importer.py:220-223`: `value = tree.value * learning_rate`); the array dump already carries shrunk leaves. The Rust loader must NOT re-shrink.
- **Parallelizing the tree sum.** GTIL-08: per-row tree summation is serial in tree_id order (float add is non-associative). The current `predict_preset` is already serial — preserve that when widening to multiclass.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LightGBM categorical bitset → category list | A custom bit-iteration with different word order | Port `BitsetToList` verbatim (`lightgbm.cc:210-221`): word = `bits[i/32]`, bit = `i%32`, slot count = `cat_boundaries[k+1]-cat_boundaries[k]` | Word/bit order and the boundary slicing are exact; any deviation silently mis-decodes categories. |
| HistGB categorical bit test | A reinvented bitmap layout | Port `check(bitmap,val,row) = (bitmap[8*row + val/32] >> (val%32)) & 1` (`sklearn.cc:296`) | The `8*row` stride (8 uint32 = 256-bit row) is load-bearing; 256-category fixed scan. |
| sklearn gain computation | Your own impurity formula | Port the exact weighted-impurity gain (`sklearn.cc:236-243` / `sklearn_bulk.cc:197-203`) | The `sample_cnt * (impurity - left/right weighted) / total_sample_cnt` form must match byte-for-byte if goldens compare serialized models. |
| IsolationForest anomaly score | `clf.decision_function` semantics | `exponential_standard_ratio` postprocessor with `ratio_c = expected_depth(max_samples_)` (`isolation_forest.py:11-17`) | Treelite ≠ framework (D-07); golden is `-clf.score_samples`, computed via `exp2(-margin/ratio_c)` (`postprocessor.cc:45-47`). |
| Objective name aliasing | A partial alias table | Port `CanonicalObjective` verbatim (`detail/lightgbm.h:26-57`) — ~40 aliases collapse to ~12 canonical names | Missing an alias (e.g. `l2_root`→`regression`) routes to the wrong postprocessor. |
| Softmax / sigmoid numerics | Naive `exp` loop | Port `softmax` with the max-subtraction + `double norm_const` + `float` cast (`postprocessor.cc:57-75`) | The mixed f32/f64 reduction order IS the 1e-5 contract (mirrors the Phase-1 sigmoid cast-ordering note). |

**Key insight:** Every numeric transform in this phase has an exact upstream form whose float/double cast ordering determines whether the 1e-5 bound holds. This is a transcription exercise with verification, not a design exercise — the same posture that made Phase 3 land byte-identical.

## Runtime State Inventory

> Not a rename/refactor/migration phase — this is greenfield loader addition. No stored data, live-service config, OS-registered state, secrets, or pre-existing build artifacts carry an old string forward. **None — verified by: Phase 4 adds new loader crates and new fixtures; it renames/migrates nothing. The only "state" is the frozen fixtures it creates (write-once, D-06).**

## Common Pitfalls

### Pitfall 1: The ModelBuilder only builds `Tree<f32>` / `ModelVariant::F32`
**What goes wrong:** D-05 requires `<f64,f64>`, but `ModelBuilder::commit_model` hardcodes `Model::new(ModelVariant::F32(...))` and stages `Vec<Tree<f32>>` (`lib.rs:135,558`). Routing LightGBM or the sklearn MixIn path through the builder as-is yields an f32 model and silently downcasts thresholds/leaves.
**Why it happens:** Phase 2/3 only needed f32 (XGBoost). The f64 variant exists in `treelite-core` (`ModelVariant::F64`, `Tree<f64>`) and `bulk_construct_tree` already returns `Tree<f64>`, but nothing assembles an F64 `Model` yet [VERIFIED: grep shows no `ModelVariant::F64(ModelPreset::new` outside concat].
**How to avoid:** First enabling task — give `ModelBuilder` an f64 mode (generic over `T`, or a parallel f64 builder), and add a bulk→`Model` assembly that wraps `bulk_construct_tree` output in `ModelVariant::F64`. Mirror upstream's `GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64)` (`sklearn.cc:210-211`, `lightgbm.cc:521-523`).
**Warning signs:** a sklearn/LightGBM predict test passes structurally but max|delta| sits around 1e-6–1e-7 from f32 rounding, or `model.threshold_type()` returns `Float32`.

### Pitfall 2: LightGBM node-id reassignment (negative-index leaves)
**What goes wrong:** LightGBM stores `left_child`/`right_child` with negative values denoting leaves (`-1` → leaf index 0, via `~old_id`), and arrays indexed by *internal* node id (`num_leaves - 1` entries), separate from leaf arrays (`num_leaves` entries). Treating the raw indices as node ids produces a malformed tree.
**Why it happens:** The on-disk LightGBM layout is split internal/leaf arrays; Treelite re-numbers into one depth-wise `1,2,3,...` sequence.
**How to avoid:** Port the BFS deque exactly (`lightgbm.cc:533-601`): seed with `(0, 1)` for normal trees or `(-1, 1)` for the single-leaf constant tree; `dfs_index` starts at 1 and increments by 2 per internal node; `old_node_id < 0` ⇒ leaf with value `leaf_value[~old_node_id]`. Note new ids start at **1**, not 0.
**Warning signs:** orphan/dangling-child errors from the builder, or off-by-one leaf values.

### Pitfall 3: LightGBM missing-value → default-direction override
**What goes wrong:** For numerical splits, when `missing_type != kNaN`, LightGBM maps missing values to 0.0, which means the `default_left` flag must be *recomputed* as `default_left = (0.0 <= threshold)`, overriding the parsed `kDefaultLeftMask` bit (`lightgbm.cc:579-584`). Skipping this routes missing values the wrong way.
**Why it happens:** LightGBM encodes "missing→zero" semantics that Treelite must translate into an explicit default direction since GTIL routes NaN via `default_child`.
**How to avoid:** Port `GetMissingType` (`decision_type >> 2 & 3`, `:206-208`) and the override branch verbatim. For categorical splits, missing always goes right and `default_left=false` (`:569-572`).
**Warning signs:** predictions diverge only on rows containing NaN/zero features.

### Pitfall 4: HistGB `_preprocessor` is sklearn-version-gated (capture-side)
**What goes wrong:** The `features_map`/`categories_map` the Rust loader consumes are derived capture-side from `sklearn_model._preprocessor`, whose structure changed across sklearn versions: ≥1.4.0 introduced `_preprocessor` with `transformers_[0]=="encoder"`, `[1]=="numerical"`; ≥1.7.0 adds a 3rd `transformers_[2]=="remainder"` asserted to be `"drop"` (`importer.py:371-411`). Capturing with a mismatched version produces wrong maps, and the golden misses by a feature-permuted offset.
**Why it happens:** The embedded OrdinalEncoder + ColumnTransformer reorders categorical-before-numerical features; the map encodes that permutation.
**How to avoid:** Pin the exact sklearn version in the manifest (D-06); record it. For Phase-4 verify-narrow, prefer a HistGB fixture **without categorical features** first (so `feat_remapper = arange`, the identity, `importer.py:407-410`), then add a categorical HistGB fixture as a second slice. The Rust loader itself is version-agnostic (it consumes the already-computed maps).
**Warning signs:** HistGB regressor golden off by a constant or with features transposed; classifier class order scrambled.

### Pitfall 5: HistGB `num_threshold` vs bin reconstruction
**What goes wrong:** Assuming the Rust loader must reconstruct thresholds from `_bin_mapper` bins. It must NOT — the v4.7.0 C++ loader reads `node.num_threshold` (a `double`) directly from the packed struct (`sklearn.cc:338`); the bin→threshold reconstruction is already materialized in the numpy `nodes` array capture-side.
**Why it happens:** The CONTEXT.md framing ("`_bin_mapper` bin→threshold reconstruction") describes the *capture-side* contract, which the frozen fixture bakes in.
**How to avoid:** Decode `num_threshold` straight from the struct. The `_bin_mapper.make_known_categories_bitsets()` output (`known_cat_bitsets`, `f_idx_map`) is passed to the loader but **unused** by `LoadHistGradientBoostingImpl` in v4.7.0 — capture it for the fixture, but the Rust decode ignores it.
**Warning signs:** Over-engineering a bin-mapper port that upstream doesn't have at the C++ layer.

### Pitfall 6: GTIL output shaping — current `predict` is scalar-only
**What goes wrong:** The existing `treelite-gtil::predict` produces a flat `Vec<f32>` of length `num_row` and only handles `(num_row,1,1)` (`lib.rs:150-239`). Multiclass classifiers (GB/HistGB/RF) need `(num_row, num_target, max_num_class)` output with per-`(target,class)` accumulation routed by `target_id[tree]`/`class_id[tree]`, plus tree averaging (RF) and per-`(target,class)` base-score add.
**Why it happens:** Phase 1 only needed binary scalar output.
**How to avoid:** Pull forward `OutputLeafValue`/`OutputLeafVector` (`predict.cc:174-229`) — the four-way branch on `(target_id==-1, class_id==-1)` — plus the averaging block (`:259-293`) gated on `average_tree_output`, plus the f64 `base_scores` 2D add (`:294-304`). Keep the serial-tree-sum invariant (GTIL-08). The leaf-vector broadcast (RF, both ids `-1`) and the round-robin class routing (GB/HistGB multiclass, `class_id[tree]=tree%n_class`) are the two shapes Phase 4 must support.
**Warning signs:** classifier predictions collapse all classes into column 0, or RF outputs are summed-not-averaged (off by a factor of `n_estimators`).

## Code Examples

Verified patterns from the vendored upstream source.

### LightGBM objective → postprocessor map (LGB-03)
```rust
// Source: lightgbm.cc:442-515 + detail/lightgbm.h:26-57 (CanonicalObjective first)
// After CanonicalObjective collapses aliases:
//   "multiclass"            -> softmax           (validate num_class param == num_class_)
//   "multiclassova"         -> multiclass_ova    {sigmoid_alpha = parsed "sigmoid:<a>"}
//   "binary"                -> sigmoid           {sigmoid_alpha = parsed "sigmoid:<a>", must be > 0}
//   "cross_entropy"         -> sigmoid           {sigmoid_alpha = 1.0}
//   "cross_entropy_lambda"  -> logarithm_one_plus_exp
//   "poisson"/"gamma"/"tweedie" -> exponential
//   "regression"/"regression_l1"/"huber"/"fair"/"quantile"/"mape"
//        -> "sqrt" in params ? signed_square : identity
//   "lambdarank"/"rank_xendcg"/"custom" -> identity
// num_class > 1 forces task=kMultiClf and requires obj in {multiclass, multiclassova}.
// class_id[i] = i % num_class_   (lightgbm.cc:427-429)
// average_output = presence of the "average_output" key in the global dict (:289-291).
```

### sklearn MixIn metadata (IsolationForest, the canonical Treelite≠framework case)
```rust
// Source: sklearn.cc:33-57, 373-383
// Metadata{n_features, kIsolationForest, average_tree_output=true, num_target=1,
//          num_class={1}, leaf_vector_shape={1,1}}
// postprocessor = "exponential_standard_ratio" with param {"ratio_c": ratio_c}
// base_scores = {0.0}
// leaf = LeafScalar(value[tree][node])   // value = pre-computed isolation depth (importer.py:191-205)
// ratio_c = expected_depth(max_samples_):  (isolation_forest.py:11-17)
//   n<=1 -> 0; n==2 -> 1; else 2*(ln(n-1)+euler_gamma) - 2*(n-1)/n
// Golden assert: treelite.gtil.predict(model, X) == -clf.score_samples(X)   (D-07)
```

### exponential_standard_ratio postprocessor (pull-forward, SKL-03)
```rust
// Source: postprocessor.cc:45-47  — note exp2 (base-2), not exp
// *elem = exp2(-*elem / model.ratio_c);
// Rust: v = (-v / ratio_c).exp2()   // keep cast ordering consistent with model.ratio_c type
```

### softmax postprocessor (pull-forward, multiclass GB/HistGB/LGB)
```rust
// Source: postprocessor.cc:57-75 — mixed f32/f64 reduction is the 1e-5 contract
// max_margin = max over row (f32); norm_const accumulates in f64;
// row[i] = exp(row[i] - max_margin) (f32 exp); then row[i] /= (f32)norm_const.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| LightGBM legacy v3 model strings | `version=v4` text format (vendored fixture is `v4`) | LightGBM 4.x | Phase-4 fixtures should be captured from current LightGBM; the parser keys (`max_feature_idx`, `num_class`, `objective`, `Tree=`) are stable across v3/v4. [VERIFIED: `deep_lightgbm/model.txt` line 2 = `version=v4`] |
| sklearn HistGB without `_preprocessor` | sklearn ≥1.4.0 `_preprocessor` (encoder/numerical); ≥1.7.0 adds `remainder=drop` | sklearn 1.4 / 1.7 | Capture-side version pin is load-bearing (Pitfall 4). [CITED: importer.py:371-411] |
| Treelite C-API ctypes marshalling | (Phase 8 PyO3) — Phase 4 tests with frozen array dumps | — | Phase 4 deliberately does NOT build the Python extraction layer (deferred D / Phase 8). |

**Deprecated/outdated:**
- Nothing in the v4.7.0 loader path is deprecated for v1 scope; the upstream source IS the pinned target.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `scikit-learn`/`lightgbm` versions are unpinned (not yet installed); capture plan will pin them | Standard Stack | Wrong HistGB `_preprocessor` shape → feature-permuted golden (Pitfall 4). Mitigated by pinning + manifest. |
| A2 | The vendored `deep_lightgbm/model.txt` (regression, 32 leaves, no categorical) is adequate as the LGB-01 numerical smoke fixture; a fresh categorical fixture is captured for LGB-02 | Validation Architecture | If the vendored fixture's golden can't be captured (needs LightGBM to re-load it), a fresh fit is needed. Low risk — `LoadLightGBMModelFromString` + `treelite.gtil.predict` works on any text model. |
| A3 | v4.7.0 HistGB loader ignores `known_cat_bitsets`/`known_cat_bitsets_offset_map` (passed but unused in `LoadHistGradientBoostingImpl`) | HistGB Decode | If a later codepath uses them, a categorical-HistGB fixture could miss. Verified unused in `sklearn.cc:284-350`; deferring exhaustive categorical-HistGB parity to a 2nd slice de-risks it. |
| A4 | The RF classifier leaf-normalization (`norm_factor`) is the only load-time leaf transform; GB leaf-shrink is capture-side | Pattern 2 / Anti-Patterns | Double-applying learning-rate shrink would scale GB predictions. Verified: `importer.py:220-223` shrinks capture-side; loader must not. |
| A5 | slopcheck verdicts for `scikit-learn`/`lightgbm` would be `[OK]` (couldn't run in sandbox) | Package Legitimacy Audit | Negligible — both are first-party-maintained, capture-only, never in the Rust/CI graph. |

## Open Questions (RESOLVED)

1. **Exact `treelite.gtil.predict` keyword for predict-kind in 4.7.0**
   - What we know: Phase-1's `capture_golden.py` already prints `help(treelite.gtil.predict)` to confirm the 4.7.0 signature; default kind applies the postprocessor.
   - What's unclear: whether IsolationForest needs a non-default predict kind (it does not — default applies `exponential_standard_ratio`).
   - Recommendation: the capture script should print `help()` once (as Phase 1 did) and assert the golden against the DEFAULT kind; cross-check IsolationForest golden equals `-clf.score_samples(X)`.
   - RESOLVED in Plan 04-03 Task 1: the capture script prints `help(treelite.gtil.predict)` once, asserts the golden against the DEFAULT predict kind, and cross-checks the IsolationForest golden against `-clf.score_samples(X)` — confirming no non-default kind is needed.

2. **f64 ModelBuilder: generic-over-T vs parallel builder**
   - What we know: the builder is f32-only; `bulk_construct_tree` is already `Tree<f64>`; core has `ModelVariant::F64`.
   - What's unclear: whether to make `ModelBuilder` generic `<T>` or add a thin f64 variant — a design call.
   - Recommendation: planner's call (architecture). Generic-over-T keeps one state machine; a parallel builder avoids touching the working f32 path. Either way it's the FIRST enabling task.
   - RESOLVED in Plan 04-01: the plan makes this design call and implements the f64 builder mode + `bulk_to_model` assembly as the first enabling task (planner's discretion exercised per Claude's Discretion in CONTEXT.md).

3. **Single combined HistGB slice vs numerical-then-categorical**
   - What we know: D-08 requires full HistGB verify; categorical adds `categories_map`/`features_map` complexity (Pitfall 4).
   - Recommendation: two slices — numerical-only HistGB first (identity feature map), then categorical HistGB — to isolate the remap risk. Both still land in Phase 4.
   - RESOLVED in Plan 04-08: a two-task split — numerical-only HistGB first (identity feature map), categorical HistGB second — isolates the `features_map`/`categories_map` remap risk, both landing in Phase 4 per D-08.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `uv run python` | Golden capture (D-06) | ✓ | (venv on main tree) | — (capture runs on main tree, not worktrees — see Tooling note) |
| `treelite` (Python wheel) | Golden = `treelite.gtil.predict` (D-07) | ✓ | 4.7.0 | — |
| `numpy` | Array dumps / input matrix | ✓ | 2.4.6 | — |
| `scikit-learn` | Fit RF/ET/GB/IsolationForest/HistGB + extract arrays | ✗ | — | Install in capture env: `uv pip install scikit-learn` (capture-only, never CI runtime) |
| `lightgbm` | Fit/load LightGBM text models | ✗ | — | Install in capture env: `uv pip install lightgbm` (capture-only) |
| Vendored `treelite-mainline/` | Porting source-of-truth | ✓ | C++ v4.7.0 | — |
| Vendored `deep_lightgbm/model.txt` | LightGBM parse smoke fixture | ✓ | v4 format, 1351 B | — |

**Missing dependencies with no fallback:** none (all blockers have a one-line install in the capture-only environment).
**Missing dependencies with fallback:** `scikit-learn`, `lightgbm` — install in the frozen `uv run python` capture environment only; record exact versions in each golden's manifest (D-06). They never enter the Rust build, the shipped artifact, or CI.

## Validation Architecture

> nyquist_validation is enabled (config `nyquist_validation: true`). Tests are Rust (`cargo test --workspace`), goldens are JSON fixtures asserted by `treelite-harness`.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `anyhow` (ERR-02) in tests |
| Config file | none (Cargo convention); fixtures under `fixtures/` resolved via `CARGO_MANIFEST_DIR/../../fixtures` (harness pattern) |
| Quick run command | `cargo test -p treelite-lightgbm -p treelite-sklearn -p treelite-gtil` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LGB-01 | LightGBM text model loads + predicts within 1e-5 (numerical, vendored fixture) | integration (golden) | `cargo test -p treelite-harness lightgbm_numerical` | ❌ Wave 0 |
| LGB-02 | Categorical bitset decode correct; per-field precision (leaf/threshold f64, gain f32) | unit + golden | `cargo test -p treelite-lightgbm bitset` ; `cargo test -p treelite-harness lightgbm_categorical` | ❌ Wave 0 |
| LGB-03 | objective→postprocessor map (+sigmoid_alpha), class_id round-robin, average_output | unit | `cargo test -p treelite-lightgbm objective` | ❌ Wave 0 |
| SKL-01 | RF + ExtraTrees (clf+reg) import + predict 1e-5 via bulk path | integration (golden) | `cargo test -p treelite-harness sklearn_rf` | ❌ Wave 0 |
| SKL-02 | GradientBoosting (clf+reg) import + predict 1e-5 | integration (golden) | `cargo test -p treelite-harness sklearn_gb` | ❌ Wave 0 |
| SKL-03 | IsolationForest import; golden == -score_samples (exponential_standard_ratio) | integration (golden) | `cargo test -p treelite-harness sklearn_iforest` | ❌ Wave 0 |
| SKL-04 | HistGB (clf+reg) import incl. bulk/packed path + predict 1e-5 | unit (struct decode) + golden | `cargo test -p treelite-sklearn histgb_decode` ; `cargo test -p treelite-harness sklearn_histgb` | ❌ Wave 0 |
| (enabler) | f64 ModelBuilder + bulk→Model assembly produces `ModelVariant::F64` | unit | `cargo test -p treelite-builder f64_mode` | ❌ Wave 0 |
| (enabler) | GTIL multiclass/leaf-vector output shaping + averaging + f64 base-score | unit | `cargo test -p treelite-gtil output_shaping` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-under-edit>` (the quick run for the touched crate)
- **Per wave merge:** `cargo test --workspace` (full suite — must stay green; preserves the Phase-3 XGBoost 1e-5 regression gate)
- **Phase gate:** `cargo test --workspace` green before `/gsd-verify-work`; max|delta| < 1e-5 on every per-estimator golden, recorded (EQV-04 spirit).

### Wave 0 Gaps
- [ ] `crates/treelite-builder` f64 builder mode + bulk→`Model` assembly (Pitfall 1) — gates everything
- [ ] `crates/treelite-gtil` output-shaping/averaging/base-score + new postprocessors (Pitfall 6) — gates all golden asserts
- [ ] `crates/treelite-lightgbm/` crate + tests (LGB-01/02/03)
- [ ] `crates/treelite-sklearn/` crate + tests (SKL-01/02/03/04)
- [ ] `fixtures/capture_sklearn.py`, `fixtures/capture_lightgbm.py` — frozen captures (D-06), needs `scikit-learn`+`lightgbm` installed in the capture env
- [ ] per-estimator golden JSONs + manifests (input matrix + treelite-GTIL output + version pins)
- [ ] `treelite-harness` per-estimator assert tests (extend the `golden_v5.rs`/`equivalence.rs` pattern)

## Security Domain

> `security_enforcement: true`, ASVS level 1. This phase parses **untrusted file input** (LightGBM text files, sklearn array dumps, HistGB packed byte buffers). The relevant surface is input-validation / memory-safety on malformed input — there is no auth/session/network/crypto surface (loaders are pure functions; no I/O beyond reading the supplied bytes; no Python binding yet).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (library loader) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | yes | Typed `thiserror` errors on malformed input; bounds-checked indices; never panic/OOB on hostile input (ERR-01, the Phase-3 T-03-01 discipline) |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for {LightGBM text parser, sklearn array loaders, HistGB byte decode}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed LightGBM `num_leaves`/`cat_boundaries` causing OOB array access | Denial of Service | Validate counts before slicing; `cat_threshold` length = `cat_boundaries.back()` (`lightgbm.cc:332`); return typed `LgbError`, not panic |
| Negative / out-of-range child indices in sklearn arrays → OOB | Denial of Service | The MixIn path validates via `ModelBuilder` (dangling/orphan checks already ported); the bulk path trusts input (D-09) — gate bulk behind the frozen-fixture contract and bounds-check `children_left/right` before indexing `n_node_samples`/`impurity` for gain |
| HistGB `expected_sizeof_node_struct` ≠ 52/56, or `nodes` buffer shorter than `node_count × itemsize` → OOB read | Tampering / DoS | Reject any itemsize not in {52,56} (upstream `TREELITE_LOG(FATAL)`, `sklearn.cc:366`); verify buffer length ≥ `node_count × itemsize` before field decode; `feature_idx` range-check before `features_map[feature_idx]` |
| HistGB `bitset_idx` / `feature_idx` indexing `raw_left_cat_bitsets` / `features_map` out of range | DoS | Bounds-check both indices against the supplied map/bitset lengths before access |
| Integer overflow in `node_count` → `int` cast | DoS | Port the `TREELITE_CHECK_LE(node_count, INT_MAX)` guard (`sklearn.cc:216-219`) as a typed error |

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/include/treelite/model_loader.h` lines 103–345 — exact sklearn `namespace sklearn` signatures (D-01 contract)
- `treelite-mainline/src/model_loader/sklearn.cc` — MixIns (IsolationForest/GB/HistGB), `LoadSKLearnModel`, the packed `HistGradientBoostingNode` struct (52/56 B) + `LoadHistGradientBoostingImpl` decode
- `treelite-mainline/src/model_loader/sklearn_bulk.cc` — `BulkConstructTree` + RF clf/reg direct-Model assembly + leaf-normalization
- `treelite-mainline/src/model_loader/lightgbm.cc` — line parser, `LGBTree`, `BitsetToList`, node-id reassignment, objective→postprocessor switch
- `treelite-mainline/src/model_loader/detail/lightgbm.h` — `CanonicalObjective` alias table
- `treelite-mainline/src/gtil/predict.cc` — `EvaluateTree`, `NextNodeCategorical`, `OutputLeafValue`/`OutputLeafVector`, averaging + base-score add (output-shaping pull-forward)
- `treelite-mainline/src/gtil/postprocessor.cc` — softmax/exponential_standard_ratio/exponential/logarithm_one_plus_exp/multiclass_ova (cast-ordering)
- `treelite-mainline/python/treelite/sklearn/importer.py` — ArrayOfArrays extraction contract, GB leaf-shrink, HistGB `_preprocessor`/`features_map`/`categories_map` capture
- `treelite-mainline/python/treelite/sklearn/isolation_forest.py` — `expected_depth`/`calculate_depths`/`ratio_c`
- `crates/treelite-builder/src/lib.rs`, `bulk.rs`, `concat.rs` — current builder (f32-only) + `bulk_construct_tree` (Tree<f64>)
- `crates/treelite-gtil/src/lib.rs`, `postprocessor.rs` — current scalar-only predict + identity/sigmoid
- `crates/treelite-core/src/tree.rs`, `model.rs` — `Tree<f64>`, `ModelVariant::F64`, leaf_vector/category_list accessors (already present)
- `fixtures/capture_golden.py`, `golden_v5.manifest.json`, `crates/treelite-harness/tests/golden_v5.rs` — capture + manifest + assert pattern
- `.planning/config.json` — `nyquist_validation: true`, `security_enforcement: true`

### Secondary (MEDIUM confidence)
- Environment probes: `uv run python -c "import treelite"` → 4.7.0; `import numpy` → 2.4.6; `import sklearn` → ModuleNotFoundError (capture deps absent)
- `treelite-mainline/tests/examples/deep_lightgbm/model.txt` — vendored v4 LightGBM text fixture (regression, 32 leaves, no categorical)

### Tertiary (LOW confidence)
- PyPI download/age figures for scikit-learn/lightgbm in the Package Legitimacy Audit (`[ASSUMED]`; slopcheck unavailable in sandbox)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — porting source is vendored read-only; no library-selection risk; existing crates inspected directly.
- Architecture: HIGH — every data-flow stage line-cited from `treelite-mainline/`; the two gaps (f64 builder, GTIL output-shaping) confirmed by codebase grep.
- Pitfalls: HIGH — each derived from a specific upstream line range and cross-checked against existing Rust crate state.
- Environment: MEDIUM — capture-dep versions (sklearn/lightgbm) not yet pinned (must be pinned at capture time); treelite/numpy confirmed.

**Research date:** 2026-06-10
**Valid until:** ~2026-07-10 (stable — pinned to vendored C++ v4.7.0 which does not move; the only volatile element is capture-side sklearn/lightgbm versions, fixed at capture time)

# Phase 4: LightGBM & scikit-learn Loaders - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 19 new + 3 modified
**Analogs found:** 19 / 22 (3 modified files extend existing code; 3 new files have no in-repo analog)

This phase replicates the proven `treelite-xgboost` converge-then-build pipeline for two new
frameworks. **Almost everything has a precise in-repo analog** — the planner's job is mostly
"copy this file's structure, swap XGBoost specifics for LightGBM/sklearn specifics, and route to
`<f64,f64>` instead of `<f32,f32>`." The three genuinely novel pieces (f64 builder mode, GTIL
output-shaping, HistGB packed-struct decode) are explicitly mapped to their closest partial
analogs plus the upstream porting source.

## File Classification

### New crate: `treelite-lightgbm` (mirrors `treelite-xgboost`)

| New File | Role | Data Flow | Closest Analog | Match |
|----------|------|-----------|----------------|-------|
| `crates/treelite-lightgbm/Cargo.toml` | config | — | `crates/treelite-xgboost/Cargo.toml` | exact |
| `crates/treelite-lightgbm/src/lib.rs` | loader | transform (parse→build) | `crates/treelite-xgboost/src/lib.rs` | exact |
| `crates/treelite-lightgbm/src/parse.rs` | loader | file-I/O (text line parse) | `crates/treelite-xgboost/src/json.rs` (struct family) + `legacy.rs` (cursor) | role-match |
| `crates/treelite-lightgbm/src/objective.rs` | utility | transform | `crates/treelite-xgboost/src/objective.rs` | exact |
| `crates/treelite-lightgbm/src/bitset.rs` | utility | transform | *(no analog — see No Analog Found)* | none |
| `crates/treelite-lightgbm/src/error.rs` | model (error enum) | — | `crates/treelite-xgboost/src/error.rs` | exact |

### New crate: `treelite-sklearn` (mirrors `treelite-xgboost`)

| New File | Role | Data Flow | Closest Analog | Match |
|----------|------|-----------|----------------|-------|
| `crates/treelite-sklearn/Cargo.toml` | config | — | `crates/treelite-xgboost/Cargo.toml` | exact |
| `crates/treelite-sklearn/src/lib.rs` | loader | request-response (array signatures) | `crates/treelite-xgboost/src/lib.rs` | role-match |
| `crates/treelite-sklearn/src/mixin.rs` | loader | transform (array→builder) | `crates/treelite-xgboost/src/lib.rs::build_tree` | role-match |
| `crates/treelite-sklearn/src/bulk.rs` | loader | transform (array→Model) | `crates/treelite-builder/src/bulk.rs` (consumer side) | role-match |
| `crates/treelite-sklearn/src/histgb.rs` | loader | file-I/O (packed-byte decode) | `crates/treelite-xgboost/src/legacy.rs` (byte cursor) | role-match |
| `crates/treelite-sklearn/src/error.rs` | model (error enum) | — | `crates/treelite-xgboost/src/error.rs` | exact |

### Modified: enabling work (the two Wave-0 gates)

| Modified File | Role | Data Flow | Analog (in-file precedent to mirror) | Match |
|---------------|------|-----------|--------------------------------------|-------|
| `crates/treelite-builder/src/lib.rs` | service | transform | the existing f32 `commit_model`/`end_tree` (mirror for f64) | role-match |
| `crates/treelite-builder/src/bulk.rs` (+ new assembly) | service | transform | `bulk_construct_tree` (already `Tree<f64>`; needs bulk→`Model` wrap) | role-match |
| `crates/treelite-gtil/src/lib.rs` | service | request-response | scalar `predict_preset`/`predict` (widen to multiclass shaping) | role-match |
| `crates/treelite-gtil/src/postprocessor.rs` | utility | transform | `identity`/`sigmoid` (add softmax/exp-std-ratio/exponential/log1pexp) | role-match |

### New fixtures + harness (mirrors `golden_v5`/`golden.json` discipline)

| New File | Role | Data Flow | Closest Analog | Match |
|----------|------|-----------|----------------|-------|
| `fixtures/capture_lightgbm.py` | test (capture) | file-I/O | `fixtures/capture_golden_v5.py` | role-match |
| `fixtures/capture_sklearn.py` | test (capture) | file-I/O | `fixtures/capture_golden_v5.py` | role-match |
| `fixtures/{lightgbm,sklearn}_*.golden.json` + manifests | test (fixture data) | — | `fixtures/golden.json` + `golden_v5.manifest.json` | exact |
| `crates/treelite-harness/tests/lightgbm.rs` | test | request-response | `crates/treelite-harness/src/lib.rs::run_equivalence` + `tests/golden_v5.rs` | role-match |
| `crates/treelite-harness/tests/sklearn.rs` | test | request-response | same | role-match |
| `crates/treelite-harness/src/lib.rs` (extend) | test util | — | existing `Golden`/`run_equivalence`/`Manifest` | role-match |

---

## Pattern Assignments

### `crates/treelite-lightgbm/src/lib.rs` (loader, transform)

**Analog:** `crates/treelite-xgboost/src/lib.rs` — copy the whole shape: validators
(`require_non_negative`, `check_dim`), a per-tree `build_tree(builder, idx, t)`, a shared
`build_model_from_parsed(parsed) -> Result<Model, Err>` convergence path, and a thin public
`load_lightgbm(&str)` entry.

**Validators to copy verbatim** (`treelite-xgboost/src/lib.rs:63-96`):
```rust
fn require_non_negative(field: &'static str, value: i32) -> Result<i32, XgbError> { ... }
fn check_dim(tree, field, num_nodes, got) -> Result<(), XgbError> { ... }
```

**Builder-emission core** (`treelite-xgboost/src/lib.rs:142-167`) — the per-node loop is the
template; LightGBM differs only in (a) the negative-index leaf reassignment (Pitfall 2 — port
`lightgbm.cc:533-601`) producing the `start_node` key sequence, and (b) `numerical_test` uses the
`default_left` override from Pitfall 3:
```rust
builder.start_tree()?;
for i in 0..num_nodes {
    builder.start_node(i as i32)?;
    if /* leaf */ { builder.leaf_scalar(value)?; }
    else { builder.numerical_test(split_index, threshold, default_left, op, l, r)?; builder.gain(g)?; }
    builder.end_node()?;
}
builder.end_tree()?;
```

**Metadata + commit** (`treelite-xgboost/src/lib.rs:248-274`) — copy the `BuilderMetadata` struct
fill + `commit_model()` + post-commit `model.sigmoid_alpha`/`model.ratio_c` assignment.
DIFFERENCE: route through the NEW f64 builder mode (D-05), and set `leaf_vector_shape`,
`num_class`, `class_id[i]=i%num_class` (round-robin), `average_tree_output` from LightGBM's
`average_output` key (`lightgbm.cc:289-291,427-429`).

**Porting source:** `treelite-mainline/src/model_loader/lightgbm.cc` +
`.../detail/lightgbm.h`.

---

### `crates/treelite-lightgbm/src/objective.rs` (utility, transform)

**Analog:** `crates/treelite-xgboost/src/objective.rs` — near-exact structural twin.

**Map pattern to copy** (`treelite-xgboost/src/objective.rs:23-48`): a `match` over canonicalized
objective names returning `Ok(&'static str)` postprocessor name, `Err(...UnrecognizedObjective)`
on miss. LightGBM differs by (a) running `CanonicalObjective` alias-collapse FIRST
(`detail/lightgbm.h:26-57` — ~40 aliases → ~12), and (b) parsing `sigmoid:<a>` for `sigmoid_alpha`.
The `parse_base_score`/`transform_base_score_to_margin` helpers are XGBoost-specific and NOT
needed here.

**Porting source:** `lightgbm.cc:442-515` (switch) + `detail/lightgbm.h:26-57` (`CanonicalObjective`).

---

### `crates/treelite-lightgbm/src/parse.rs` (loader, file-I/O)

**Analog:** `crates/treelite-xgboost/src/json.rs` for the **typed-struct family** discipline
(serde structs with one field per recognized key, doc-cited against the upstream recognized-key
set), and `crates/treelite-xgboost/src/legacy.rs` for the **fallible byte/string cursor** pattern
(every read bounds-checked, returns a typed error never a panic).

LightGBM is line-based `key=value`, so `std::str::lines()`/`split('=')` replace serde — but the
output is the same: typed `LGBTree`/`LGBModel` structs with per-field precision (leaf_value/threshold
= **f64**, split_gain = f32, decision_type = i8, cat_threshold = u32, cat_boundaries = u64 per LGB-02).

**Error discipline** (mirror `treelite-xgboost/src/json.rs` doc header + `lib.rs` validators): a
malformed `num_leaves`/`cat_boundaries` returns `LgbError`, never an OOB slice (ASVS V5 / Security
Domain threat table in RESEARCH).

**Porting source:** `lightgbm.cc:157-414` (line parse, `LGBTree`).

---

### `crates/treelite-lightgbm/src/error.rs` (model, error enum)

**Analog:** `crates/treelite-xgboost/src/error.rs` — copy the `#[derive(Debug, Error)]` enum
shape verbatim.

**Variants to copy directly** (`treelite-xgboost/src/error.rs`): `InvalidScalar { field, value }`,
`DimensionMismatch { tree, field, expected, got }`, `UnrecognizedObjective(String)`, plus the two
`#[error(transparent)]` bridges:
```rust
#[error(transparent)]
Core(#[from] treelite_core::CoreError),
#[error(transparent)]
Builder(#[from] treelite_builder::BuilderError),
```
ADD a LightGBM-specific parse-error variant (analogous to `Legacy { pos, detail }` /
`Ubjson { pos, detail }` at lines 93-121) for malformed text / OOB bitset / bad node count.

---

### `crates/treelite-sklearn/src/lib.rs` (loader, request-response)

**Analog:** `crates/treelite-xgboost/src/lib.rs` for the convergence-path discipline, BUT the
public surface is the array-signature set (D-01) — a near-line-for-line Rust translation of
`treelite-mainline/include/treelite/model_loader.h` `namespace sklearn` (lines ~103-345). The
C++ `double const**` array-of-arrays becomes `&[&[f64]]`:
```rust
// Mirrors LoadRandomForestRegressor (model_loader.h:135-139)
pub fn load_random_forest_regressor(
    n_estimators: i32, n_features: i32, n_targets: i32,
    node_count: &[i64],
    children_left: &[&[i64]], children_right: &[&[i64]], feature: &[&[i64]],
    threshold: &[&[f64]], value: &[&[f64]],
    n_node_samples: &[&[i64]], weighted_n_node_samples: &[&[f64]], impurity: &[&[f64]],
) -> Result<Model, SklError> { ... }
```
Plus `LoadIsolationForest` (`+ ratio_c: f64`), classifier variants (`+ n_classes: &[i32]`), and
`LoadHistGradientBoosting{Regressor,Classifier}`.

**Porting source:** `model_loader.h` lines 103-345 (signatures), `sklearn.cc` /
`sklearn_bulk.cc` (dispatch).

---

### `crates/treelite-sklearn/src/mixin.rs` (loader, transform) — IsolationForest + GradientBoosting

**Analog:** `crates/treelite-xgboost/src/lib.rs:114-167` (`build_tree`) — the per-node
`start_node` → `numerical_test`/`leaf_scalar` → `end_node` loop driving `ModelBuilder`. sklearn's
leaf-detection sentinel is `children_left[i] == -1` (NOT `<= 0` — that's HistGB; see Anti-Patterns
in RESEARCH).

**Metadata pattern** — set directly via `BuilderMetadata` (`treelite-xgboost/src/lib.rs:248-260`),
routed to the f64 builder. IsolationForest specifics (RESEARCH Code Examples):
`task_type=kIsolationForest`, `average_tree_output=true`, `postprocessor="exponential_standard_ratio"`,
`model.ratio_c = ratio_c`, `base_scores=[0.0]`.

**Porting source:** `sklearn.cc:33-57` (IsolationForest MixIn), `:59-133` (GradientBoosting MixIn),
`:373-415` (`LoadSKLearnModel`).

---

### `crates/treelite-sklearn/src/bulk.rs` (loader, transform) — RandomForest / ExtraTrees

**Analog:** `crates/treelite-builder/src/bulk.rs` — `bulk_construct_tree(...) -> Tree<f64>` already
exists and ports `sklearn_bulk.cc:36-211` (incl. the classifier leaf-normalization). This new file
is the **caller**: it builds the `&[&[...]]` arrays per tree, calls `bulk_construct_tree` for each,
then assembles the trees into a `Model` and sets metadata BY HAND (the bulk path bypasses the
builder).

**The metadata to set directly** (RESEARCH Pattern 2, `sklearn_bulk.cc:244-330`):
```text
// classifier: task=kMultiClf; average_tree_output=true;
//   num_class=n_classes; leaf_vector_shape={n_targets, max_num_class};
//   target_id=vec![-1; n_estimators]; class_id=vec![-1; n_estimators];
//   postprocessor="identity_multiclass"; base_scores=vec![0.0; n_targets*max_num_class].
// regressor: task=kRegressor; num_class=vec![1; n_targets];
//   leaf_vector_shape={n_targets,1}; postprocessor="identity".
```

**Model assembly** — mirror `commit_model`'s tail (`treelite-builder/src/lib.rs:557-570`):
`Model::new(ModelVariant::F64(ModelPreset::new(trees)))` then assign each metadata field. This is
the bulk→`Model` wrap that the builder gate (below) must expose.

**Porting source:** `sklearn_bulk.cc:232-350`.

---

### `crates/treelite-sklearn/src/histgb.rs` (loader, file-I/O — packed-byte decode) — THE TENTPOLE

**Analog:** `crates/treelite-xgboost/src/legacy.rs` — the fallible little-endian byte-cursor
discipline (D-07/D-08 of Phase 3): read each field with `from_le_bytes` at its byte offset, never
`transmute`/`bytemuck` onto a `#[repr(C)]` struct, bounds-check before every read. The `Legacy`
error variant doc (`treelite-xgboost/src/error.rs:101-121`) describes EXACTLY this posture — copy
it for a `HistGbDecode { offset, detail }` error variant.

**Struct layout to decode** (RESEARCH Pattern 3, `sklearn.cc:260-282`): packed 52-byte (FeatureIdT=i32)
or 56-byte (i64) record; `expected_sizeof_node_struct` ∈ {52,56} selects the width — reject anything
else (Security Domain). Decode rules at `sklearn.cc:314-346`:
- leaf iff `left <= 0` (NOTE: `<= 0`, NOT `== -1`),
- `split_index = features_map[node.feature_idx]` (always remap),
- categorical bit test `check(bitmap,val,row) = (bitmap[8*row + val/32] >> (val%32)) & 1`,
- else `NumericalTest(split_index, num_threshold, default_left, kLE, left, right)`.

Emits through `ModelBuilder` (f64 mode) `numerical_test`/`categorical_test`.

**Porting source:** `sklearn.cc:260-446`, capture-side `python/treelite/sklearn/importer.py:355-478`.

---

### `crates/treelite-builder/src/lib.rs` (service, transform) — ENABLER: f64 builder mode

**Analog:** the existing f32 path IN THIS FILE is the precedent to mirror. Two hardcoded f32 sites
must gain an f64 path (RESEARCH Pitfall 1):
- `lib.rs:135` — `trees: Vec<Tree<f32>>` staging,
- `lib.rs:478-525` — `end_tree` builds `Tree::<f32>::new()` and fills 25 columns,
- `lib.rs:558` — `commit_model` wraps `ModelVariant::F32(ModelPreset::new(trees))`.

**Decision (RESEARCH Open Q2, planner's call):** make `ModelBuilder` generic `<T>` OR add a parallel
f64 builder. Either way mirror `end_tree`'s column-fill (`lib.rs:436-523`) and `commit_model`'s
metadata assignment (`lib.rs:557-570`) for `Tree<f64>` / `ModelVariant::F64`. `leaf_scalar`/
`leaf_vector` currently take `f32` (`lib.rs:282,309`) — these need an f64 surface.

**Reference for the f64 column shape:** `crates/treelite-builder/src/bulk.rs` already builds a
complete `Tree<f64>` with all 25 columns (`bulk.rs:167-191`) — copy that column set.

**Porting source:** `model_builder.cc` (already cited in-file); upstream uses
`GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64)`.

---

### `crates/treelite-gtil/src/lib.rs` (service, request-response) — ENABLER: output shaping

**Analog:** the existing scalar `predict_preset`/`predict` IN THIS FILE
(`gtil/src/lib.rs:143-239`). The current path produces a flat `Vec<f32>` of length `num_row`
shape `(num_row,1,1)` and only handles `identity`/`sigmoid`. Phase 4 widens it (RESEARCH Pitfall 6)
to:
- `(num_row, num_target, max_num_class)` output, routed by `target_id[tree]`/`class_id[tree]`
  (the four-way `OutputLeafValue`/`OutputLeafVector` branch, `predict.cc:174-229`),
- tree averaging gated on `average_tree_output` (`predict.cc:259-293`),
- f64 `base_scores` 2D per-`(target,class)` add (`predict.cc:294-304`).

**Invariants to preserve** (already correct in `predict_preset`, `lib.rs:151-162`): serial tree-sum
in tree_id order (GTIL-08 — do NOT parallelize), `T::to_f32()` leaf cast then f32 accumulate, f64
base-score promotion `(acc as f64 + base) as f32`. The `PredictScalar` trait + `evaluate_tree`
(`lib.rs:28-127`) are reused unchanged; for HistGB/LGB categorical, add the `NextNodeCategorical`
branch (deferred-but-minimal per D-03/D-04).

**Porting source:** `treelite-mainline/src/gtil/predict.cc:174-305`.

---

### `crates/treelite-gtil/src/postprocessor.rs` (utility, transform) — ENABLER: new postprocessors

**Analog:** the existing `identity`/`sigmoid` IN THIS FILE
(`gtil/src/postprocessor.rs:20-35`) — the same `fn(alpha, v) -> f32` signature + the cast-ordering
doc-comment discipline. ADD: `softmax` (max-subtract + f64 `norm_const` + f32 cast,
`postprocessor.cc:57-75`), `exponential_standard_ratio` (`v = (-v/ratio_c).exp2()`,
`postprocessor.cc:45-47` — note `exp2` base-2), `exponential`, `logarithm_one_plus_exp`, and
optionally `multiclass_ova`.

**Critical:** the mixed f32/f64 reduction order in `softmax` IS the 1e-5 contract — port verbatim,
do not "simplify" (RESEARCH Don't Hand-Roll).

**Porting source:** `treelite-mainline/src/gtil/postprocessor.cc:19-82`.

---

### `crates/treelite-harness/tests/{lightgbm,sklearn}.rs` (test, request-response)

**Analog:** `crates/treelite-harness/src/lib.rs::run_equivalence` (`src/lib.rs:189-233`) and
`tests/golden_v5.rs`. The pattern: `fixture_path(name)` resolver
(`golden_v5.rs:35-41`), load model via the new loader, flatten input matrix, `predict`, assert
`approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5)` while tracking `max_dev`.

**Per-estimator extension:** the existing `Golden`/`Manifest` structs (`src/lib.rs:76-108`) +
`load_golden`/`check_manifest` are reused; add one golden file + one test per estimator family.
The `1e-5` epsilon is a HARD gate — never loosened (`src/lib.rs:227-229`).

---

### `fixtures/capture_{lightgbm,sklearn}.py` (test, capture)

**Analog:** `fixtures/capture_golden_v5.py` — the "run once, commit, CI never regenerates"
discipline (D-06). Copy: paths-relative-to-script resolution (`capture_golden_v5.py:44-48`), the
manifest key set + sha256/version pins (`:66-76`), and the empirical-assertion-in-capture pattern
(`:82-88`).

**DIFFERENCES (D-07):** the golden output is `treelite.gtil.predict(model, X)` — NOT the
framework's own `predict()`. For IsolationForest, cross-check golden == `-clf.score_samples(X)`.
The capture env must `uv pip install scikit-learn lightgbm` (capture-only). Dump sklearn node
arrays per `importer.py` contract; freeze the input matrix + manifest (sklearn/lightgbm/treelite
versions + seed).

---

## Shared Patterns

### Converge-then-build pipeline (applies to BOTH new loader crates)
**Source:** `crates/treelite-xgboost/src/lib.rs:114-289`
**Apply to:** `treelite-lightgbm/src/lib.rs`, `treelite-sklearn/src/{mixin,histgb}.rs`
Parse → typed structs → `require_non_negative`/`check_dim` validators → `ModelBuilder` per-node
emission → `BuilderMetadata` fill → `commit_model()` → post-commit scalar assignment.

### Typed `thiserror` error enum with transparent bridges
**Source:** `crates/treelite-xgboost/src/error.rs:14-135`
**Apply to:** `treelite-lightgbm/src/error.rs`, `treelite-sklearn/src/error.rs`
```rust
#[derive(Debug, Error)]
pub enum SklError {
    #[error("...")] DimensionMismatch { tree: usize, field: &'static str, expected: usize, got: usize },
    #[error(transparent)] Core(#[from] treelite_core::CoreError),
    #[error(transparent)] Builder(#[from] treelite_builder::BuilderError),
}
```
No `anyhow` in library crates; no panic on malformed input (ERR-01, ASVS V5).

### Fallible little-endian byte cursor (HistGB packed decode)
**Source:** `crates/treelite-xgboost/src/legacy.rs` + its error doc `error.rs:101-121`
**Apply to:** `treelite-sklearn/src/histgb.rs`
`from_le_bytes` per field at explicit offsets; bounds-check before each read; reject
`itemsize ∉ {52,56}` and buffer shorter than `node_count × itemsize`; NO native-endian transmute.

### Crate manifest shape
**Source:** `crates/treelite-xgboost/Cargo.toml`
**Apply to:** `treelite-lightgbm/Cargo.toml`, `treelite-sklearn/Cargo.toml`
```toml
[dependencies]
treelite-core = { path = "../treelite-core" }
treelite-builder = { path = "../treelite-builder" }
thiserror = { workspace = true }
[dev-dependencies]
treelite-gtil = { path = "../treelite-gtil" }   # avoid harness dep cycle
approx = { workspace = true }
```
Also add both crates to the workspace `members` list in the root `Cargo.toml`
(`Cargo.toml:3-9`). `serde`/`serde_json` are needed only if a fixture JSON loader lives in-crate.

### Golden + manifest harness (1e-5 assertion)
**Source:** `crates/treelite-harness/src/lib.rs:76-263` + `tests/golden_v5.rs`
**Apply to:** all new per-estimator goldens and harness tests
`fixture_path` resolver, `Golden`/`Manifest` structs, `run_equivalence` flatten+predict+assert,
`check_manifest` environment-drift warning, hard `1e-5` epsilon.

### Frozen Python capture (D-06/D-07)
**Source:** `fixtures/capture_golden_v5.py`
**Apply to:** `fixtures/capture_{lightgbm,sklearn}.py`
Run-once-commit-never-regenerate; manifest with version pins + sha256; golden output from
`treelite.gtil.predict`, NOT the framework.

---

## No Analog Found

| File | Role | Data Flow | Reason / Fallback |
|------|------|-----------|-------------------|
| `crates/treelite-lightgbm/src/bitset.rs` | utility | transform | No categorical-bitset decode exists yet in-repo (GTIL `next_node` only does scalar; builder stores categorical-test nodes but never decodes a bitset). Port `BitsetToList` verbatim from `lightgbm.cc:210-221` per RESEARCH Don't-Hand-Roll. The HistGB `check(bitmap,...)` test (`sklearn.cc:296`) is a sibling but a DIFFERENT layout — do not share code. |
| HistGB f64 categorical emission through builder | loader | transform | `ModelBuilder::categorical_test` exists (`builder/src/lib.rs:254-278`) but Phase 2 note says category LISTS are "not yet exercised in GTIL" and `leaf_vector`/category columns are stubbed. The f64 builder mode (enabler) must actually wire `category_list` columns for HistGB — no working in-repo precedent; derive from `sklearn.cc:340-345` + upstream `Tree` category accessors. |
| Multiclass `(num_row, num_target, max_num_class)` GTIL output buffer | service | request-response | The current `predict` returns a flat scalar `Vec<f32>` only; the 3D shaping has no in-repo analog. Port from `predict.cc:174-305` (RESEARCH Pitfall 6). Closest partial precedent is the existing serial-sum loop (`gtil/src/lib.rs:151-162`) — reuse its invariants, widen its output. |

---

## Metadata

**Analog search scope:** `crates/treelite-xgboost/src/`, `crates/treelite-builder/src/`,
`crates/treelite-gtil/src/`, `crates/treelite-harness/{src,tests}/`, `fixtures/`,
`treelite-mainline/{src/model_loader,include/treelite}/` (porting source-of-truth).
**Files scanned:** 14 in-repo Rust files + 3 upstream C++ references + 1 capture script.
**Pattern extraction date:** 2026-06-10

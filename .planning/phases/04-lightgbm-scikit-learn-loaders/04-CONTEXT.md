# Phase 4: LightGBM & scikit-learn Loaders - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Widen the loader layer to **LightGBM text-format models** and the **full scikit-learn estimator set** — `RandomForest`/`ExtraTrees`, `GradientBoosting`, `IsolationForest`, and `HistGradientBoosting` (classifier + regressor where applicable) — so every source framework in v1 scope loads into the proven SoA `Model` spine and predicts within 1e-5 of upstream Treelite.

**Requirements covered:** LGB-01, LGB-02, LGB-03, SKL-01, SKL-02, SKL-03, SKL-04.

In scope (HOW, not WHETHER): LightGBM text parse incl. categorical bitset decode + per-field precision; LightGBM objective→postprocessor mapping with `sigmoid_alpha`, `class_id` round-robin, `average_output`; sklearn array-based import for all five estimator families incl. the HistGB bulk/packed path; freezing upstream-Treelite goldens for each; and pulling forward the minimal GTIL pieces each loader needs to verify 1e-5 this phase.

Out of scope: the complete GTIL surface (all 4 predict kinds, all 10 postprocessors, sparse CSR, full categorical/output-shaping matrix) — that is Phase 5. The PyO3 Python binding and live estimator marshalling — that is Phase 8.

</domain>

<decisions>
## Implementation Decisions

### sklearn input contract
- **D-01:** The Rust sklearn loader **mirrors the upstream C-API array signature 1:1** — functions like `load_random_forest_regressor(n_estimators, n_features, n_targets, node_count: &[i64], children_left: &[&[i64]], children_right: &[&[i64]], feature: &[&[i64]], threshold: &[&[f64]], value: &[&[f64]], n_node_samples, weighted_n_node_samples, impurity)` and the analogous `LoadIsolationForest` (with `ratio_c`), classifier variants (with `n_classes`), and `LoadHistGradientBoosting{Regressor,Classifier}` (with `features_map` / bin data). This is a faithful port of `treelite-mainline/include/treelite/model_loader.h` (`namespace sklearn`). Phase 8 PyO3 will later call these directly with zero-copy numpy buffers — no separate intermediate file format is introduced.
- **D-02:** Loaders emit through the existing `ModelBuilder` (converge-then-build), reusing the Phase-2/3 bulk path (`BulkConstructTree`) where upstream uses `sklearn_bulk.cc`.

### Verify-narrow boundary (Phase 4 vs Phase 5)
- **D-03:** **Pull forward the minimal GTIL each loader needs to verify 1e-5 this phase.** For every estimator family, port just enough GTIL surface (e.g. `IsolationForest`'s `exponential_standard_ratio` postprocessor, the multiclass output shaping a classifier needs, LightGBM's `sigmoid`/`softmax`) to assert real prediction parity in Phase 4. Phase 5 then widens to the *complete* GTIL surface — it does not have to backfill Phase-4 loaders' basic parity. Phase 4 proves loading is real, not just structural.
- **D-04:** HistGradientBoosting is verified in full this phase (see D-08), so whatever GTIL HistGB requires (bin-threshold evaluation path, its postprocessor) is also pulled forward under D-03.

### Numeric preset mapping
- **D-05:** **LightGBM and sklearn both map to the `<f64,f64>` `ModelPreset`** — matching upstream per-field precision: LightGBM `leaf_value`/`threshold` = f64 (`split_gain` = f32 metadata), sklearn `threshold`/`value` = f64. This is the first end-to-end exercise of the f64 variant (XGBoost stays `<f32,f32>` from Phase 1). Planner/research must confirm the exact upstream `ThresholdType`/`LeafOutputType` each loader sets and not silently downcast.

### Fixtures & goldens
- **D-06:** **One-time `uv run python` capture, frozen.** A single Python session fits each estimator family / loads each LightGBM model, dumps the node arrays (sklearn) + input matrix + a frozen manifest (sklearn, lightgbm, treelite versions + seed), and commits them as read-only fixtures. CI never regenerates — mirrors the Phase-1/3 golden discipline (D-05/D-06 lineage). Pin sklearn + LightGBM + upstream-Treelite versions in the manifest.
- **D-07:** **The golden prediction vector is captured from upstream Treelite's GTIL** (`treelite.gtil.predict` in the frozen Python session), NOT from the framework's own `predict()`. The 1e-5 contract is against upstream Treelite — for `IsolationForest` especially, Treelite's output is `-clf.score_samples(X)` and deliberately differs from the framework. The framework's own predict may be recorded only as a secondary sanity cross-check, never as the asserted target.

### HistGradientBoosting scope
- **D-08:** **Full import + 1e-5 verify this phase.** Port the complete HistGB path — `_bin_mapper` bin→threshold reconstruction, version-gated `_preprocessor` / `features_map` (embedded OrdinalEncoder) feature remapping, packed node-struct decode — and verify against an upstream-Treelite golden in Phase 4. Honors SKL-04 fully now rather than deferring; this is the largest single chunk of phase work and is the research-flagged risk.

### Claude's Discretion
- **Crate organization** (e.g. `treelite-lightgbm` + `treelite-sklearn` as parallels to `treelite-xgboost`, vs a combined loader crate) — planner's call, following the established per-format-crate pattern.
- **LightGBM text-parse mechanics** — the streaming/line-based parser shape, categorical-bitset decode implementation, and `string_utils` analog — research/planner decide, mirroring `treelite-mainline/src/model_loader/lightgbm.cc` + `detail/lightgbm.h`.
- **HistGB packed-node decode mechanics** — exact struct unpacking strategy is an implementation detail for research to derive from upstream.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Upstream loaders (porting source of truth)
- `treelite-mainline/src/model_loader/lightgbm.cc` — LightGBM text-format parser (objective map, sigmoid_alpha, class_id round-robin, average_output).
- `treelite-mainline/src/model_loader/detail/lightgbm.h` — LightGBM parse detail incl. categorical bitset decode + per-field precision.
- `treelite-mainline/src/model_loader/sklearn.cc` — sklearn array→Model loader; per-estimator MixIns (IsolationForest `ratio_c`/`exponential_standard_ratio`, GradientBoosting, RF/ET, HistGB).
- `treelite-mainline/src/model_loader/sklearn_bulk.cc` — bulk tree-construction path used by the sklearn/HistGB loaders.
- `treelite-mainline/include/treelite/model_loader.h` §`namespace sklearn` (lines ~103–345) — the exact array signatures the Rust loaders mirror (D-01), incl. `LoadRandomForestRegressor`, `LoadIsolationForest`, classifier variants, `LoadHistGradientBoosting{Regressor,Classifier}` + `features_map`.

### Python reference (fixture capture + array semantics)
- `treelite-mainline/python/treelite/sklearn/importer.py` — `ArrayOfArrays` marshalling: exactly which estimator attributes (`children_left`, `children_right`, `feature`, `threshold`, `value`, `n_node_samples`, `weighted_n_node_samples`, `impurity`, `node_count`) are extracted and in what dtype — the contract the frozen fixtures (D-06) must reproduce.
- `treelite-mainline/python/treelite/sklearn/isolation_forest.py` — `calculate_depths` / `expected_depth` and the `ratio_c` derivation for IsolationForest.
- `treelite-mainline/python/treelite/sklearn/__init__.py`, `exporter.py` — package surface.

### Test corpus
- `treelite-mainline/tests/examples/deep_lightgbm/model.txt` — vendored LightGBM text fixture (1351 bytes) — real LightGBM parse smoke test.
- `treelite-mainline/tests/examples/{toy_categorical,sparse_categorical}/` — categorical-split material (LightGBM bitset relevance).
- `treelite-mainline/tests/examples/mushroom/mushroom.model` — XGBoost legacy (independent; not a Phase-4 source).

### In-repo assets to extend
- `crates/treelite-xgboost/src/{json.rs,legacy.rs,ubjson.rs,objective.rs,error.rs}` — the converge-then-build template (parse → typed structs → validators → `ModelBuilder` → `CommitModel`) the new loaders replicate.
- `crates/treelite-builder/` — `ModelBuilder` / `BulkConstructTree` / `ConcatenateModelObjects` (BLD-01/02/03) the loaders emit through.
- `crates/treelite-harness/` + `fixtures/golden_v5.*`, `fixtures/golden.json` — the golden + manifest harness pattern extended for the new per-estimator goldens (D-06/D-07).
- `crates/treelite-gtil/` — where the minimal pulled-forward GTIL pieces (D-03/D-04) land.
- `.planning/codebase/{ARCHITECTURE,CONVENTIONS,TESTING}.md` — SoA/variant pattern, error-translation, test layout.
- `.planning/phases/03-full-xgboost-loaders/03-CONTEXT.md` — parse-wide/verify-narrow precedent + golden discipline this phase inherits.

### Tooling note
- Python is run via `uv run python` (not bare `python`); the venv/pyproject are untracked and absent from worktrees, so golden-capture & checkpoint plans run on the main tree.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`treelite-xgboost` converge-then-build pipeline** — the structural template for both new loaders; parse into typed intermediate structs, validate, emit through `ModelBuilder`.
- **`ModelBuilder` + `BulkConstructTree`** (Phase 2, incl. 02-06 AllocNode per-node column emission) — sklearn/HistGB bulk import emits through this; the upstream `sklearn_bulk.cc` analog.
- **`objective.rs` pattern** — objective→postprocessor mapping + parameter parsing (sigmoid_alpha) mirrors how LightGBM's objective line maps to a postprocessor.
- **golden + manifest harness** (`treelite-harness`, `fixtures/golden_v5.*`, `fixtures/golden.json`) — extended for per-estimator prediction goldens captured from upstream Treelite GTIL.

### Established Patterns
- **Converge-then-build:** parse → typed structs → validators (`require_non_negative`/`check_dim`) → `ModelBuilder` emission → `CommitModel`. New frameworks plug in at the struct layer.
- **Parse-wide / verify-narrow:** load the full surface; assert 1e-5 only where GTIL supports it. Phase 4 extends this by *pulling GTIL forward* (D-03) so the verify-narrow set is larger than Phase 3's.
- **Golden frozen from upstream; CI never regenerates** (D-06/D-07) — but the asserted target is upstream **Treelite** GTIL, not the framework.
- **thiserror transparent propagation:** loader errors surface as a typed `…Error` enum; no panic, no anyhow in library crates.

### Integration Points
- New loader crate(s) join `treelite-xgboost` as workspace members, all emitting through `treelite-builder`.
- New f64-preset code paths exercise the `<f64,f64>` `ModelPreset` variant end-to-end for the first time (D-05) — core `Model`/`Tree`/serialize must handle it without regression.
- Pulled-forward GTIL pieces (D-03/D-04/D-08) land in `treelite-gtil` and must coexist with the Phase-1 scalar path without breaking the existing 1e-5 XGBoost regression gate.

</code_context>

<specifics>
## Specific Ideas

- sklearn loader API should read as a near-line-for-line Rust translation of `model_loader.h`'s `namespace sklearn` signatures (D-01) — array-of-arrays as `&[&[T]]` slices, so the Phase-8 PyO3 layer can hand it borrowed numpy buffers with zero copy.
- IsolationForest is the canonical "Treelite ≠ framework" case: golden = `treelite.gtil.predict` (== `-clf.score_samples`), and the postprocessor is `exponential_standard_ratio` with `ratio_c` (D-07).
- HistGB is the phase's tentpole — `_bin_mapper` threshold reconstruction + `features_map` remapping must be correct or its golden will miss by a constant/feature-permuted offset (D-08).
- LightGBM `class_id[i] = i % num_class` round-robin and `average_output` are easy to drop — call them out explicitly in the LightGBM plan (LGB-03).

</specifics>

<deferred>
## Deferred Ideas

- **Complete GTIL surface** (all 4 predict kinds, all 10 postprocessors, sparse CSR, full categorical/output-shaping matrix) — Phase 5. Phase 4 only pulls forward the minimal slices its loaders need (D-03).
- **PyO3 marshalling of live fitted estimators** — Phase 8. Phase 4 builds the array-signature loaders (D-01) and tests them with frozen array-dump fixtures; the Python extraction layer that calls them comes later.
- **Multi-target / multi-output sklearn estimators beyond what the captured fixtures exercise** — verify-narrow keeps fixtures to representative cases; broader output-shape coverage rides Phase 5's harness.
- **LightGBM categorical-split PREDICTION parity beyond the captured fixture** — bitset decode is implemented (LGB-02), but exhaustive categorical evaluation parity aligns with Phase 5's categorical GTIL (continuation of Phase-3's deferred categorical item).

None of the above is scope creep out of Phase 4 — all recorded so they aren't lost.

</deferred>

---

*Phase: 4-lightgbm-scikit-learn-loaders*
*Context gathered: 2026-06-10*

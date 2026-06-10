---
phase: 04-lightgbm-scikit-learn-loaders
plan: 03
subsystem: testing
tags: [golden-vectors, treelite-gtil, sklearn, lightgbm, fixtures, capture]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "golden capture + manifest discipline (capture_golden.py / capture_golden_v5.py)"
  - phase: 04-01
    provides: "f64 builder/bulk->Model path the sklearn/lightgbm loaders target"
  - phase: 04-02
    provides: "GTIL output-shaping + postprocessors (the predict path the goldens assert against)"
provides:
  - "Frozen per-estimator goldens for every Phase-4 estimator slice (Plans 04-04..04-08)"
  - "sklearn goldens: RF/ET, GB, IsolationForest, HistGB numerical, HistGB categorical"
  - "LightGBM goldens: numerical (vendored) + categorical (fresh, bitset)"
  - "Two one-time frozen capture scripts (capture_sklearn.py, capture_lightgbm.py)"
affects: [04-04, 04-05, 04-06, 04-07, 04-08, treelite-sklearn, treelite-lightgbm, treelite-harness]

# Tech tracking
tech-stack:
  added: [scikit-learn 1.9.0 (capture-only), lightgbm 4.6.0 (capture-only)]
  patterns:
    - "Golden source is treelite.gtil.predict (D-07), never framework predict()"
    - "IsolationForest golden cross-checked == -score_samples at capture time (D-07)"
    - "GB leaf-shrink applied capture-side; loader must NOT re-shrink"
    - "HistGB packed nodes buffer frozen as base64 + itemsize + features_map/categories_map"

key-files:
  created:
    - fixtures/capture_sklearn.py
    - fixtures/capture_lightgbm.py
    - fixtures/sklearn_rf.golden.json
    - fixtures/sklearn_gb.golden.json
    - fixtures/sklearn_iforest.golden.json
    - fixtures/sklearn_histgb_numerical.golden.json
    - fixtures/sklearn_histgb_categorical.golden.json
    - fixtures/lightgbm_numerical.golden.json
    - fixtures/lightgbm_categorical.txt
    - fixtures/lightgbm_categorical.golden.json
  modified: []

key-decisions:
  - "Golden = treelite.gtil.predict output (default post-processed kind), NOT framework predict (D-07)"
  - "IsolationForest golden equals -clf.score_samples(X) within 1e-5, asserted at capture time (max delta 6.9e-9)"
  - "HistGB node itemsize is 56 (the 64-bit feature-index packed-struct variant) on this env"
  - "LightGBM categorical uses max_cat_to_onehot=1 to force bitset (non-one-hot) splits (num_cat=3, 10 cat_threshold lines)"

patterns-established:
  - "Per-estimator frozen golden JSON: input matrix + treelite-GTIL output + version-pinned manifest + sha256"
  - "Capture-side empirical assertion (IForest cross-check, HistGB identity-map / categories-map split, LGB bitset presence)"

requirements-completed: [LGB-01, LGB-02, SKL-01, SKL-02, SKL-03, SKL-04]

# Metrics
duration: 6min
completed: 2026-06-10
---

# Phase 4 Plan 3: Frozen Estimator Goldens Summary

**Seven frozen per-estimator golden JSONs (sklearn RF/ET, GB, IsolationForest, HistGB numerical + categorical; LightGBM numerical + categorical) captured from upstream `treelite.gtil.predict` with version-pinned manifests — the fixtures gate for all Phase-4 estimator slices.**

## Performance

- **Duration:** ~6 min
- **Completed:** 2026-06-10
- **Tasks:** 2
- **Files modified:** 10 created

## Accomplishments
- All five `sklearn_*.golden.json` goldens captured from `treelite.gtil.predict` (D-07), each carrying the `importer.py` node-array dtype contract.
- IsolationForest golden cross-checked at capture time against `-clf.score_samples(X)` — passed with max |delta| = 6.9e-9 (« 1e-5), settling the canonical "Treelite ≠ framework" case (D-07).
- HistGB split into a numerical fixture (features_map = identity arange, no categories_map — Pitfall 4) and a categorical fixture (categories_map present, exercises the embedded OrdinalEncoder remap); packed `nodes` buffers frozen as base64 with `expected_sizeof_node_struct = 56`.
- LightGBM numerical golden references the vendored `deep_lightgbm/model.txt` (no byte duplication); a fresh categorical model with real bitset splits (num_cat=3, 10 `cat_threshold` lines) freezes the LGB-02 target.
- Both capture scripts are one-time frozen (`uv run python`, main worktree); each golden manifest pins sklearn/lightgbm/treelite/numpy versions + seed (D-06).

## Task Commits

Each task was committed atomically:

1. **Task 1: Frozen sklearn capture (RF/ET, GB, IForest, HistGB numerical + categorical)** - `3e9cb69` (test)
2. **Task 2: Frozen LightGBM capture (numerical + categorical)** - `541cb68` (test)

## Files Created/Modified
- `fixtures/capture_sklearn.py` - one-time frozen sklearn capture (node arrays + treelite-GTIL golden + manifest; IForest cross-check)
- `fixtures/capture_lightgbm.py` - one-time frozen LightGBM capture (vendored numerical + fresh categorical)
- `fixtures/sklearn_rf.golden.json` - RF/ET clf+reg node arrays + GTIL output
- `fixtures/sklearn_gb.golden.json` - GB clf+reg, leaf-shrink applied capture-side
- `fixtures/sklearn_iforest.golden.json` - IsolationForest, golden == -score_samples
- `fixtures/sklearn_histgb_numerical.golden.json` - HistGB identity feature map (Pitfall 4)
- `fixtures/sklearn_histgb_categorical.golden.json` - HistGB with categories_map
- `fixtures/lightgbm_numerical.golden.json` - references vendored deep_lightgbm/model.txt
- `fixtures/lightgbm_categorical.txt` - fresh categorical LightGBM text model (bitset splits)
- `fixtures/lightgbm_categorical.golden.json` - categorical GTIL golden

## Decisions Made
- Goldens use the GTIL **default** predict kind (post-processed; `pred_margin=False`), asserted in each script so a future API default change is caught.
- Captured matrices and seed (1234) are frozen in each golden; HistGB packed nodes stored as base64 to preserve exact bytes for the Rust byte-cursor decoder (Phase 3 precedent).
- LightGBM categorical model forced to bitset splits via `max_cat_to_onehot=1` so LGB-02 has a genuine bitset-decode target.

## Deviations from Plan

None - plan executed exactly as written. Both capture tasks ran successfully on the first attempt; all in-script empirical assertions (GTIL default kind, IForest cross-check, HistGB identity/categories-map split, LightGBM bitset presence) passed.

## Issues Encountered
None. scikit-learn 1.9.0 and lightgbm 4.6.0 were already present in the capture venv (no install gate tripped); treelite 4.7.0 + numpy 2.4.6 confirmed.

## User Setup Required
None - the capture scripts are one-time/frozen and were executed during this plan. scikit-learn and lightgbm are capture-only and never enter the Rust build graph or CI runtime (D-06). The committed goldens are read-only; CI never regenerates them.

## Next Phase Readiness
- All seven frozen goldens are committed and available for the per-estimator slices (Plans 04-04 LightGBM, 04-05, 04-06 sklearn, 04-07 IsolationForest, 04-08 HistGB) to assert load→predict→1e-5 against.
- HistGB itemsize is 56 on this environment — the Rust byte-cursor decoder (Plan 04-08) must handle the 56-byte (64-bit feature index) packed-node variant; the 52-byte variant is documented but not present in these fixtures.
- No blockers.

## Self-Check: PASSED

---
*Phase: 04-lightgbm-scikit-learn-loaders*
*Completed: 2026-06-10*

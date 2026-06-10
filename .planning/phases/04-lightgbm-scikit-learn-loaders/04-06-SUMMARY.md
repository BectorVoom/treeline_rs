---
phase: 04-lightgbm-scikit-learn-loaders
plan: 06
subsystem: model-loader
tags: [sklearn, random-forest, extra-trees, gradient-boosting, bulk-path, mixin, 1e-5]
requires:
  - "treelite-builder::bulk_construct_tree + bulk_to_model (04-01)"
  - "treelite-builder f64 ModelBuilder mode (04-01)"
  - "treelite-gtil::predict RF averaging + f64 base-score + postprocessors (04-02)"
  - "fixtures/sklearn_rf.golden.json + sklearn_gb.golden.json (04-03)"
provides:
  - "treelite-sklearn crate (RF/ET bulk + GB MixIn loaders, D-01 array signatures)"
  - "load_random_forest_{regressor,classifier} + load_extra_trees_* (SKL-01)"
  - "load_gradient_boosting_{regressor,classifier} (SKL-02)"
  - "treelite-gtil identity_multiclass postprocessor (no-op)"
affects:
  - "Phase-8 PyO3 will call the D-01 array surface with zero-copy numpy buffers"
tech-stack:
  added: []
  patterns:
    - "Loader-crate converge-then-build (mirrors treelite-xgboost / treelite-lightgbm)"
    - "Bulk path (RF/ET) bypasses the node-by-node builder; MixIn path (GB) drives the f64 builder"
key-files:
  created:
    - crates/treelite-sklearn/Cargo.toml
    - crates/treelite-sklearn/src/lib.rs
    - crates/treelite-sklearn/src/bulk.rs
    - crates/treelite-sklearn/src/mixin.rs
    - crates/treelite-sklearn/src/error.rs
    - crates/treelite-harness/tests/sklearn.rs
  modified:
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/src/postprocessor.rs
    - crates/treelite-harness/Cargo.toml
    - fixtures/capture_sklearn.py
    - fixtures/sklearn_gb.golden.json
decisions:
  - "04-06: RF/ET classifier & regressor route through the bulk path; ExtraTrees is an alias to the RF impl (sklearn loader does not distinguish them)"
  - "04-06: GB base_scores are NOT in the array dump; derived capture-side per importer.py (reg init_.constant_, clf _raw_predict_init) and added to the golden additively — frozen (input,output) sha256 unchanged"
  - "04-06: identity_multiclass added to gtil as a verbatim no-op (postprocessor.cc:55) so RF/ET classifier averaged-leaf-vector outputs (already normalized at load time) pass the 1e-5 gate"
metrics:
  duration: ~13min
  completed: 2026-06-10
---

# Phase 4 Plan 06: scikit-learn RF/ET + GradientBoosting Loaders Summary

The `treelite-sklearn` crate, flesh-out from the Plan-01 placeholder: array-signature loaders mirroring upstream `namespace sklearn` 1:1 (D-01), delivering two vertical slices — RandomForest/ExtraTrees via the bulk path (SKL-01) and GradientBoosting via the f64-`ModelBuilder` MixIn path (SKL-02) — each verified within 1e-5 (worst observed |delta| = 5.96e-8) of its frozen treelite-GTIL golden.

## What Was Built

- **`treelite-sklearn` crate** — D-01 array signatures (`&[&[i64]]` / `&[&[f64]]`) translating the upstream `std::int64_t const**` / `double const**` array-of-arrays; deps `treelite-core` + `treelite-builder` + `thiserror`, dev-deps `treelite-gtil` + `approx`.
- **`bulk.rs` (SKL-01)** — the RF/ET caller: per tree it bounds-checks the arrays, calls `treelite_builder::bulk_construct_tree` (which already ports per-node fill + classifier leaf-normalization, A4), collects `Vec<Tree<f64>>`, then `bulk_to_model` with metadata hand-set per `sklearn_bulk.cc:244-330` (clf → `kMultiClf`/`identity_multiclass`/target_id&class_id all -1; reg → `kRegressor`/`identity`).
- **`mixin.rs` (SKL-02)** — the GB node-by-node MixIn: drives the f64 builder (`numerical_test_f64`/`leaf_scalar_f64`, `Operator::kLE`, `gain`/`data_count`/`sum_hess`) per `sklearn.cc:200-258`. Binary → `sigmoid`/`kBinaryClf`; multiclass → `softmax`/`kMultiClf` with `class_id = tree % n_classes`; regressor → `identity`. Leaf values used AS-PROVIDED (no `learning_rate` re-shrink — A4/T-04-15).
- **`error.rs`** — `SklError` enum (`DimensionMismatch`, `InvalidScalar`, `ChildIndexOutOfRange`, `ValueBufferTooShort`, `TreeCountMismatch`, transparent `Core`/`Builder` bridges). No `panic!`/`anyhow` in the library.
- **`treelite-gtil`** — added `identity_multiclass` (verbatim no-op, `postprocessor.cc:55`) to the postprocessor dispatch.
- **`crates/treelite-harness/tests/sklearn.rs`** — `sklearn_rf` (RF/ET clf+reg, 4 families) and `sklearn_gb` (GB clf+reg) goldens, each asserting 1e-5 with max-|delta| reporting.

## Verification

- `cargo test -p treelite-sklearn` — 8 unit tests green (RF reg/clf metadata, leaf detection `== -1` not `<= 0`, ExtraTrees aliasing, OOB child → typed error; GB sigmoid/softmax/round-robin metadata, no-reshrink leaf-value assert).
- `cargo test -p treelite-harness --test sklearn` — `sklearn_rf` + `sklearn_gb` green; worst family max |delta| = **5.96e-8 < 1e-5**.
- `cargo test --workspace` — fully green (no XGBoost/LightGBM/serializer regression).
- `cargo clippy -p treelite-sklearn -p treelite-gtil` — clean.

Per-family deltas: rf_classifier 0e0, et_classifier 5.96e-8, rf_regressor 5.96e-8, et_regressor 5.96e-8, gb_classifier 0e0, gb_regressor 5.96e-8.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] gtil missing `identity_multiclass` postprocessor**
- **Found during:** Task 2 (RF/ET classifier golden could not predict)
- **Issue:** The RF/ET classifier metadata sets `postprocessor="identity_multiclass"`, but `treelite-gtil` only supported `identity`/`sigmoid`/`softmax`/`exponential*` — the classifier golden returned `UnsupportedPostprocessor`.
- **Fix:** Added `identity_multiclass` as a verbatim no-op (upstream `postprocessor.cc:55` body is empty) plus a dispatch arm in `apply_postprocessor`. The averaged leaf-vector outputs are already normalized class probabilities at load time (A4), so the postprocessor is a pass-through.
- **Files modified:** `crates/treelite-gtil/src/postprocessor.rs`, `crates/treelite-gtil/src/lib.rs`
- **Commit:** 7dc6403

**2. [Rule 3 - Blocking] GB base_scores absent from the frozen golden**
- **Found during:** Task 2 (GB golden could not be reproduced — the array dump carries no base scores)
- **Issue:** GTIL adds the GB init-model base scores to the tree sum BEFORE the postprocessor, but `sklearn_gb.golden.json` (Plan 03) stored only the array dump, not the base scores (which live in sklearn's `init_` estimator, computed inside `treelite.sklearn.import_model`). Without them the Rust loader cannot match the golden's `output`.
- **Fix:** Derived `base_scores` capture-side exactly as upstream `importer.py` (regressor `init_.constant_`; classifier `_raw_predict_init`) and added a `base_scores`/`n_classes` field per GB family **additively** to the golden. The frozen `(input, output)` sha256 contract is verified unchanged (the predictions were NOT regenerated — only the missing loader input was supplied). The capture script was updated in lockstep so the field is reproducible.
- **Files modified:** `fixtures/capture_sklearn.py`, `fixtures/sklearn_gb.golden.json`
- **Commit:** 7dc6403

## Threat Model Coverage

- **T-04-13** (OOB child index → OOB into per-node arrays): both paths bounds-check every `children_left`/`children_right` against `node_count` before the gain formula dereferences them → `SklError::ChildIndexOutOfRange` (unit-tested).
- **T-04-14** (`node_count` overflow on `int` cast): `node_count <= i32::MAX` guard → typed `SklError::InvalidScalar`.
- **T-04-15** (GB leaf re-shrink double-applying learning_rate): loader consumes capture-side-shrunk leaves AS-IS; no `* learning_rate` in the leaf path (grep-clean, unit-asserted via a leaf-value test).
- **T-04-SC** (package installs): N/A — no new third-party packages; `treelite-sklearn` is an internal path crate.

## Known Stubs

None. IsolationForest (SKL-03) and HistGradientBoosting (SKL-04) are out of scope for this plan (declared as the next slices in the objective); the ExtraTrees entry points are real (routed to the RF bulk impl), not stubs.

## Self-Check: PASSED

All created files exist on disk; both task commits (8bc10d5, 7dc6403) present in git history.

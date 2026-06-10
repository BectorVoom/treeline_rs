---
phase: 04-lightgbm-scikit-learn-loaders
plan: 07
subsystem: model-loader
tags: [sklearn, isolation-forest, mixin, exponential-standard-ratio, ratio_c, 1e-5]
requires:
  - "treelite-sklearn MixIn machinery: build_tree/build_model + BuilderMetadata (04-06)"
  - "treelite-gtil exponential_standard_ratio postprocessor (exp2 base-2, f32 ratio_c) (04-02)"
  - "treelite-gtil ratio_c dispatch in apply_postprocessor (04-02)"
  - "fixtures/sklearn_iforest.golden.json (04-03, frozen == -score_samples)"
provides:
  - "load_isolation_forest(... ratio_c: f64) -> Result<Model, SklError> (SKL-03, D-01)"
  - "IsolationForest MixIn metadata (kIsolationForest + exponential_standard_ratio + model.ratio_c)"
affects:
  - "Phase-8 PyO3 will call load_isolation_forest with zero-copy numpy buffers + capture-side ratio_c"
tech-stack:
  added: []
  patterns:
    - "MixIn path reuse: IsolationForest routes through the same build_tree/build_model as GradientBoosting (leaf == -1, leaf_scalar_f64 as-is)"
    - "Model-field post-commit assignment: ratio_c is a Model field (not BuilderMetadata), set after commit_model() exactly as upstream PostProcessorFunc config"
key-files:
  created: []
  modified:
    - crates/treelite-sklearn/src/mixin.rs
    - crates/treelite-sklearn/src/lib.rs
    - crates/treelite-harness/tests/sklearn.rs
decisions:
  - "04-07: ratio_c assigned post-commit (model.ratio_c = ratio_c as f32) because it is a Model field, not part of BuilderMetadata — mirrors upstream sklearn.cc:46 PostProcessorFunc {{ratio_c, ratio_c_}}"
  - "04-07: leaf isolation depths consumed AS-IS via the shared GB build_tree leaf path (leaf_scalar_f64(value[node])); no loader-side depth recomputation (D-07)"
  - "04-07: zero/non-finite ratio_c rejected with typed SklError::InvalidScalar rather than producing inf/NaN through (-v/ratio_c).exp2() (T-04-17)"
  - "04-07: new test code matches the prevailing compact single-line call style of plan-06's existing sklearn tests (repo does not gate cargo fmt --check; HEAD~1 already had 4 fmt diffs)"
metrics:
  duration: ~2min
  completed: 2026-06-10
requirements: [SKL-03]
---

# Phase 4 Plan 07: scikit-learn IsolationForest Loader Summary

IsolationForest imports (SKL-03) via the Plan-06 MixIn path with the `exponential_standard_ratio` postprocessor and a capture-side `ratio_c`, verified within 1e-5 (max |delta| = 5.96e-8) of its frozen treelite-GTIL golden. This is the canonical "Treelite ≠ framework" slice: the golden is `treelite.gtil.predict` (== `-clf.score_samples`, D-07), deliberately NOT the framework's own anomaly score.

## What Was Built

- **`load_isolation_forest(n_estimators, n_features, node_count, children_left, children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity, ratio_c: f64)`** in `mixin.rs` — D-01 array signature mirroring upstream `LoadIsolationForest` (`sklearn.cc:373-383`) driven by the `IsolationForestMixIn` (`sklearn.cc:33-57`). Routes through the existing `build_model` / `build_tree` (the same per-node f64 builder path as GradientBoosting): internal nodes emit `numerical_test_f64` + the sklearn impurity-reduction `gain`; leaves (`children_left[node] == -1`) emit `leaf_scalar_f64(value[node])` — the pre-computed isolation depth consumed AS-IS, no recomputation.
- **Metadata** (per `IsolationForestMixIn::HandleMetadata`): `task=kIsolationForest`, `average_tree_output=true`, `num_target=1`, `num_class=[1]`, `leaf_vector_shape={1,1}`, `target_id=class_id=vec![0; n_estimators]`, `postprocessor="exponential_standard_ratio"`, `base_scores=[0.0]`.
- **`model.ratio_c = ratio_c as f32`** assigned post-commit — `ratio_c` is a `Model` field (f32), not a `BuilderMetadata` field, so it is set after `commit_model()`, mirroring upstream's `PostProcessorFunc{"exponential_standard_ratio", {{"ratio_c", ratio_c_}}}` (`sklearn.cc:45-46`). The value itself (`expected_depth(max_samples_)`, isolation_forest.py) is computed capture-side and passed in; the loader does NOT recompute it.
- **`lib.rs`** re-exports `load_isolation_forest`; module doc updated (SKL-03 delivered).
- **`crates/treelite-harness/tests/sklearn.rs`** — new `IForestGolden` struct (flat single-estimator layout: top-level `trees`, `ratio_c`, `output`) + `sklearn_iforest` test that loads via `load_isolation_forest` (passing the captured `ratio_c`), predicts, and asserts the golden within 1e-5 with max-|delta| reporting + `check_manifest`.

## Verification

- `cargo test -p treelite-harness sklearn_iforest` — green; **max |delta| = 5.96e-8 < 1e-5** vs `treelite.gtil.predict` (== `-score_samples`).
- `cargo test -p treelite-sklearn iforest` — 3 unit tests green: metadata fields (kIsolationForest + exponential_standard_ratio + base_scores=[0.0] + ratio_c post-commit), zero-ratio_c rejection, and the leaf-depth-as-is routing assert (`exp2(0.1/2.0)` reached exactly, proving no recomputation).
- `cargo test --workspace` — fully green, zero failures (no XGBoost/LightGBM/sklearn-RF/GB/serializer regression).
- `cargo clippy -p treelite-sklearn` — clean (no warnings).

## TDD Gate Compliance

- RED: `test(04-07)` commit `ff882ee` — `sklearn_iforest` added; failed to compile (`load_isolation_forest` did not exist).
- GREEN: `feat(04-07)` commit `aecbc7c` — loader implemented; golden + unit tests pass.
- REFACTOR: none needed (the loader reuses the existing `build_tree`/`build_model` machinery verbatim; no cleanup pass produced changes).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] ratio_c divide-by-zero guard (T-04-17)**
- **Found during:** Task 1 (implementing the loader)
- **Issue:** The `exponential_standard_ratio` postprocessor computes `(-v/ratio_c).exp2()`. A `ratio_c == 0` (or non-finite) would silently produce inf/NaN predictions. The threat register (T-04-17) assigns this a `mitigate` disposition.
- **Fix:** Reject `ratio_c == 0.0 || !ratio_c.is_finite()` with a typed `SklError::InvalidScalar { field: "ratio_c", .. }` before building. Upstream `expected_depth` only returns 0 for the degenerate `max_samples <= 1` case (an unreachable fixture); rejecting it is safer than emitting inf/NaN.
- **Files modified:** `crates/treelite-sklearn/src/mixin.rs`
- **Commit:** aecbc7c

## Threat Model Coverage

- **T-04-16** (OOB child indices in IsolationForest arrays): reuses the shared `build_tree` bounds-check — every `children_left`/`children_right` is validated against `node_count` before the gain formula dereferences it → `SklError::ChildIndexOutOfRange`; per-tree dimension checks (`check_dim`) + outer-count checks (`check_outer`) cover all parallel arrays. Never OOB.
- **T-04-17** (`ratio_c == 0` div-by-zero → inf/NaN): rejected with a typed error before any prediction (unit-tested via `iforest_rejects_zero_ratio_c`).
- **T-04-SC** (package installs): N/A — no new packages; `treelite-sklearn` is an internal path crate.

## Known Stubs

None. SKL-03 is fully delivered (real loader, golden-verified). HistGradientBoosting (SKL-04) remains the next slice (out of scope for this plan).

## Self-Check: PASSED

- `crates/treelite-sklearn/src/mixin.rs`, `crates/treelite-sklearn/src/lib.rs`, `crates/treelite-harness/tests/sklearn.rs` all present on disk.
- Commits `ff882ee` (RED test), `aecbc7c` (GREEN impl) present in git history.
- `sklearn_iforest` golden gate passes within 1e-5 (5.96e-8).

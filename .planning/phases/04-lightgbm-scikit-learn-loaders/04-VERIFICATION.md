---
phase: 04-lightgbm-scikit-learn-loaders
verified: 2026-06-10T05:27:49Z
status: passed
score: 4/4
overrides_applied: 0
---

# Phase 04: LightGBM + scikit-learn Loaders — Verification Report

**Phase Goal:** Widen loaders to LightGBM text format and the full scikit-learn estimator set (including HistGradientBoosting), so every supported source framework loads into the proven spine.
**Verified:** 2026-06-10T05:27:49Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | LightGBM text model loads and predicts within 1e-5 of golden, with categorical bitset splits decoded and per-field precision (leaf_value/threshold = f64, split_gain = f32) matching upstream | VERIFIED | `tests/lightgbm.rs::lightgbm_numerical` PASS + `lightgbm_categorical` PASS. `parse.rs` explicitly declares `leaf_value: Vec<f64>`, `threshold: Vec<f64>`, `split_gain: Vec<f32>`. `bitset.rs` ports `BitsetToList` verbatim from upstream `lightgbm.cc:210-221`. |
| 2 | LightGBM objective maps to correct postprocessor with parsed sigmoid_alpha, class_id[i] = i % num_class round-robin, and average_output honored | VERIFIED | `objective.rs` parses `sigmoid_alpha` per objective family; `lib.rs:342` computes `class_id: Vec<i32> = (0..num_tree).map(|i| i % modulus).collect()`; `lib.rs:352` assigns `average_tree_output: parsed.average_output`. Unit tests confirm: `binary_maps_to_sigmoid_with_alpha`, `multiclass_maps_to_softmax`, `regression_maps_to_identity_or_signed_square` all PASS. |
| 3 | RandomForest/ExtraTrees, GradientBoosting, and IsolationForest import from sklearn array dumps via bulk path and predict within 1e-5 of their goldens | VERIFIED | `tests/sklearn.rs::sklearn_rf` PASS, `sklearn_gb` PASS, `sklearn_iforest` PASS. ExtraTrees routes to the same RF bulk implementation (`lib.rs:56-100`). IsolationForest uses `exponential_standard_ratio` postprocessor with `ratio_c`. Golden `output` field captured from `treelite.gtil.predict` (confirmed in `capture_sklearn.py`). |
| 4 | HistGradientBoosting (classifier + regressor) imports via bulk tree-construction path and predicts within 1e-5 of its golden | VERIFIED | `tests/sklearn.rs::sklearn_histgb_numerical` PASS + `sklearn_histgb_categorical` PASS. `histgb.rs` uses `from_le_bytes` exclusively (no transmute). `features_map` always applied; `categories_map` applied when present. `load_hist_gradient_boosting_regressor` / `load_hist_gradient_boosting_classifier` both exported. |

**Score:** 4/4 truths verified

---

## `cargo test --workspace` Result

```
TOTAL passed: 205   failed: 0
```

All 205 tests across the full workspace pass. Phase-critical harness tests:

| Test | Crate | Requirement | Result |
|------|-------|-------------|--------|
| `lightgbm_numerical` | treelite-harness | LGB-01 | PASS |
| `lightgbm_categorical` | treelite-harness | LGB-02 | PASS |
| `sklearn_rf` | treelite-harness | SKL-01 | PASS |
| `sklearn_gb` | treelite-harness | SKL-02 | PASS |
| `sklearn_iforest` | treelite-harness | SKL-03 | PASS |
| `sklearn_histgb_numerical` | treelite-harness | SKL-04 | PASS |
| `sklearn_histgb_categorical` | treelite-harness | SKL-04 | PASS |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-lightgbm/src/lib.rs` | LightGBM text-format loader entry point | VERIFIED | `load_lightgbm` public API; uses f64 builder (`leaf_scalar_f64`); `ModelVariant::F64` confirmed in tests |
| `crates/treelite-lightgbm/src/parse.rs` | Per-field precision parser | VERIFIED | `leaf_value: Vec<f64>`, `threshold: Vec<f64>`, `split_gain: Vec<f32>` — explicitly documented in module header |
| `crates/treelite-lightgbm/src/bitset.rs` | `BitsetToList` categorical decoder | VERIFIED | Verbatim port of `lightgbm.cc:210-221`; 6 unit tests pass |
| `crates/treelite-lightgbm/src/objective.rs` | Objective → postprocessor mapping | VERIFIED | `sigmoid_alpha` parsed; round-robin `class_id`; 9 unit tests pass |
| `crates/treelite-sklearn/src/lib.rs` | sklearn loader public API | VERIFIED | Exports RF/ET (regressor + classifier), GB (reg + clf), IsolationForest, HistGB (reg + clf) |
| `crates/treelite-sklearn/src/mixin.rs` | GB/IsolationForest metadata | VERIFIED | `exponential_standard_ratio`, `ratio_c`, `average_tree_output`, `sigmoid`, `softmax` — 9 unit tests pass |
| `crates/treelite-sklearn/src/histgb.rs` | HistGB packed-node decode | VERIFIED | `from_le_bytes` at every field (no transmute); 52/56-byte layouts; `features_map` + `categories_map` — 13 unit tests pass |
| `crates/treelite-builder/src/lib.rs` | f64 ModelBuilder mode | VERIFIED | `leaf_scalar_f64`, `leaf_vector_f64`, `commit_model` f64 branch → `ModelVariant::F64`; 4 f64-mode tests pass |
| `fixtures/lightgbm_numerical.golden.json` | Upstream-treelite golden | VERIFIED | `manifest.treelite = "4.7.0"`; output captured via `treelite.gtil.predict` |
| `fixtures/lightgbm_categorical.golden.json` | Upstream-treelite golden (categorical) | VERIFIED | `manifest.treelite = "4.7.0"`; `model_path` points to `fixtures/lightgbm_categorical.txt` |
| `fixtures/sklearn_rf.golden.json` | RF/ET upstream-treelite golden | VERIFIED | `manifest.treelite = "4.7.0"`; captured via `treelite.gtil.predict` |
| `fixtures/sklearn_gb.golden.json` | GB upstream-treelite golden | VERIFIED | `manifest.treelite = "4.7.0"` |
| `fixtures/sklearn_iforest.golden.json` | IsolationForest upstream-treelite golden | VERIFIED | `manifest.treelite = "4.7.0"`; `output = treelite.gtil.predict = -clf.score_samples` confirmed |
| `fixtures/sklearn_histgb_numerical.golden.json` | HistGB numerical golden | VERIFIED | `manifest.treelite = "4.7.0"` |
| `fixtures/sklearn_histgb_categorical.golden.json` | HistGB categorical golden | VERIFIED | `manifest.treelite = "4.7.0"` |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `treelite-lightgbm/src/lib.rs` | `treelite-builder` f64 path | `leaf_scalar_f64` calls | VERIFIED | `lib.rs:146` calls `builder.leaf_scalar_f64(tree.leaf_value[...])` |
| `treelite-lightgbm/src/lib.rs` | `ModelVariant::F64` | `commit_model` f64 branch | VERIFIED | Test asserts `matches!(model.variant, ModelVariant::F64(...))` |
| `treelite-lightgbm/src/bitset.rs` | categorical split nodes | `bitset_to_list` → `categorical_test` | VERIFIED | `lib.rs:379` invokes `bitset_to_list`; result fed to categorical node builder |
| `treelite-sklearn/src/mixin.rs` | `treelite_gtil` `exponential_standard_ratio` | `postprocessor` string + `ratio_c` | VERIFIED | `mixin.rs` sets `postprocessor = "exponential_standard_ratio"` and `model.ratio_c = ratio_c`; GTIL postprocessor test confirms it resolves |
| `treelite-sklearn/src/histgb.rs` | `treelite_builder::ModelBuilder` | decoded nodes → `numerical_test` / `categorical_test` | VERIFIED | `histgb.rs:59` imports `ModelBuilder`; `histgb.rs:480` constructs it; both test types emitted |
| `crates/treelite-harness/tests/lightgbm.rs` | `fixtures/lightgbm_*.golden.json` | 1e-5 `assert_abs_diff_eq!` | VERIFIED | Hard gate: `approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5)` |
| `crates/treelite-harness/tests/sklearn.rs` | `fixtures/sklearn_*.golden.json` | 1e-5 `assert_abs_diff_eq!` | VERIFIED | Same hard gate pattern; all 5 golden tests pass |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| LGB-01 | 04-04 | User can load a LightGBM text-format model | SATISFIED | `lightgbm_numerical` harness test PASS |
| LGB-02 | 04-04 | Categorical splits decode correctly (bitset) with upstream-matching per-field precision | SATISFIED | `lightgbm_categorical` harness test PASS; `bitset.rs` verbatim port verified |
| LGB-03 | 04-05 | LightGBM objective maps to correct postprocessor (+sigmoid_alpha), class_id round-robin, average_output honored | SATISFIED | `objective.rs` + `lib.rs` wiring verified; 9 objective unit tests PASS |
| SKL-01 | 04-06 | RandomForest + ExtraTrees (clf + reg) | SATISFIED | `sklearn_rf` harness test PASS; ExtraTrees routes to RF bulk path |
| SKL-02 | 04-06 | GradientBoosting (clf + reg) | SATISFIED | `sklearn_gb` harness test PASS |
| SKL-03 | 04-07 | IsolationForest | SATISFIED | `sklearn_iforest` harness test PASS; golden = `treelite.gtil.predict = -score_samples` |
| SKL-04 | 04-08 | HistGradientBoosting (clf + reg) | SATISFIED | `sklearn_histgb_numerical` + `sklearn_histgb_categorical` harness tests PASS |

All 7 phase requirements confirmed SATISFIED. REQUIREMENTS.md traceability table matches.

---

## Anti-Patterns Found

No `TBD`, `FIXME`, or `XXX` markers found in `crates/treelite-lightgbm/src/` or `crates/treelite-sklearn/src/`. No unresolved placeholders detected. No empty implementations or stubs found in the loader or harness paths.

---

## Behavioral Spot-Checks

| Behavior | Verification | Result |
|----------|-------------|--------|
| LightGBM golden from `treelite.gtil.predict` (not LightGBM framework) | `capture_lightgbm.py:109` calls `treelite.gtil.predict` for `output` field | PASS |
| sklearn goldens from `treelite.gtil.predict` | `capture_sklearn.py:191,236,288,414,457` all call `treelite.gtil.predict` | PASS |
| IsolationForest golden = `-clf.score_samples` | `sklearn_iforest.golden.json` contains `neg_score_samples_cross_check` + `cross_check_max_delta` fields confirming identity | PASS |
| No `transmute` in HistGB node decode | `grep transmute histgb.rs` → zero results; only `from_le_bytes` at every field | PASS |
| f64 precision preserved end-to-end (no f32 downcast) | `parse.rs:leaf_value Vec<f64>` fed directly to `leaf_scalar_f64`; `f64_leaf_sub_f32_precision_survives_commit_without_downcast` test PASS | PASS |

---

## Human Verification Required

None. All success criteria are numerically verifiable via the golden harness and confirmed by `cargo test --workspace`.

---

_Verified: 2026-06-10T05:27:49Z_
_Verifier: Claude (gsd-verifier)_

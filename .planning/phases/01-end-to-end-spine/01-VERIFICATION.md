---
phase: 01-end-to-end-spine
verified: 2026-06-10T08:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 1: End-to-End Spine Verification Report

**Phase Goal:** Prove the core value early — a model can be loaded, predicted, and verified within 1e-5 against a committed C++ golden — by standing up the thinnest end-to-end slice through the whole pipeline.
**Verified:** 2026-06-10T08:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `cargo build --workspace` and `cargo test --workspace` succeed under edition 2024 / resolver "3", all third-party crates pinned to stable versions in a single `[workspace.dependencies]` table | ✓ VERIFIED | `cargo build --workspace` exits 0 (Finished). `cargo test --workspace` exits 0, 52 tests pass across 4 crates + doc-tests. Root `Cargo.toml` has `resolver = "3"`, `[workspace.package] edition = "2024"`, single `[workspace.dependencies]` table with `thiserror="2.0.18"`, `anyhow="1.0.102"`, `serde="1.0.228"`, `serde_json="1.0.150"`, `approx="0.5.1"`. No pre-release strings (`rc`, `alpha`, `beta`) present. |
| 2 | `TaskType`, `TreeNodeType`, `Operator`, `DType` round-trip to/from their upstream string values | ✓ VERIFIED | `enums.rs` strings verified verbatim against `treelite-mainline/src/enum/*.cc`: TaskType uses `kXxx` strings, TreeNodeType uses `leaf_node`/`numerical_test_node`/`categorical_test_node`, Operator uses `""/</<=/>/>=` (kNone→""), DType uses `invalid/uint32/float32/float64`. `DType::from_str("invalid")` returns `Err` (mirrors `TypeInfoFromString` upstream). 7 enum tests pass including `dtype_round_trip_excludes_invalid` and `unknown_strings_are_typed_errors_not_panics`. |
| 3 | A `Model` exists as a two-variant enum over `<f32,f32>`/`<f64,f64>` presets, `Tree<T>` stores ~20 node fields as parallel SoA `TreeBuf<T>` columns in Owned and Borrowed modes, carrying full header metadata | ✓ VERIFIED | `ModelVariant { F32(ModelPreset<f32>), F64(ModelPreset<f64>) }` confirmed in `model.rs`. `Tree<T>` has 20 SoA `TreeBuf` columns (node_type, cleft, cright, split_index, default_left, leaf_value, threshold, cmp, category_list_right_child, leaf_vector/begin/end, category_list/begin/end, data_count/sum_hess/gain + present flags). `grep -c 'struct Node'` returns 0. `num_class`, `leaf_vector_shape`, `target_id`, `class_id` are `Vec<i32>`. `TreeBuf` has `Owned(Vec<T>)` and `Borrowed{ptr,len}` modes; `from_borrowed` is `unsafe`. All 7 tree_model tests and 4 tree_buf tests pass including the borrowed-mode round-trip. |
| 4 | A minimal walking skeleton loads one XGBoost-JSON model, runs a scalar single-threaded predict with identity/sigmoid postprocessing, and the equivalence harness asserts output within 1e-5 of the committed golden | ✓ VERIFIED | `equivalence_within_1e5` test passes. `--nocapture` output: `max observed |delta| = 0e0` (bitwise-exact match against upstream Treelite 4.7.0). `golden.json` confirmed: 5 rows, all output values in (0,1), manifest records treelite=4.7.0, xgboost=3.2.0, OS=Linux, arch=x86_64, glibc=2.39. `run_equivalence_catches_perturbation_beyond_1e5` confirms the harness actually detects a >1e-5 deviation. |
| 5 | Library crates surface typed `thiserror` errors at their boundaries; harness/binaries use `anyhow` | ✓ VERIFIED | `CoreError` (thiserror, `UnknownEnumString`), `XgbError` (thiserror, `Json`/`ParseScalar`/`InvalidScalar`/`DimensionMismatch`/`UnrecognizedObjective`/`Core`), `GtilError` (thiserror, `FeatureIndexOutOfBounds`/`NodeIndexOutOfBounds`/`InvalidInputShape`/`UnsupportedPostprocessor`/`Core`). `treelite-harness` uses `anyhow` throughout; `run_equivalence` returns `anyhow::Result`. Never panics on: unknown enum strings, bad JSON, array-length mismatch, unrecognized objective, OOB feature index, OOB child node id (CR-01 fix verified), negative num_feature/num_target (WR-01/WR-02 fixes verified). |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | Virtual workspace root, resolver 3, edition 2024, `[workspace.dependencies]` | ✓ VERIFIED | Confirmed present; resolver="3"; `[workspace.package]` edition="2024"; `[workspace.dependencies]` with 5 pinned deps |
| `crates/treelite-core/src/enums.rs` | Four enums + to_string/from_string (≥80 lines) | ✓ VERIFIED | 198 lines; all four enums with `as_str()` and `from_str()` |
| `crates/treelite-core/src/tree_buf.rs` | `TreeBuf<T>` owned + borrowed (≥30 lines) | ✓ VERIFIED | 99 lines; `Owned(Vec<T>)` and `Borrowed{ptr,len}` modes |
| `crates/treelite-core/src/tree.rs` | `Tree<T>` SoA columns + traversal getters (≥60 lines) | ✓ VERIFIED | 166 lines; 20 SoA columns; `is_leaf`, `default_child`, `left_child`, `right_child`, `split_index`, `leaf_value`, `threshold`, `comparison_op`, `has_leaf_vector` |
| `crates/treelite-core/src/model.rs` | `Model` two-variant enum + header metadata (≥60 lines) | ✓ VERIFIED | 112 lines; `ModelPreset<T>`, `ModelVariant { F32, F64 }`, `Model` with all header fields |
| `crates/treelite-core/src/error.rs` | `thiserror CoreError` | ✓ VERIFIED | `CoreError::UnknownEnumString { kind, value }` with `#[derive(Debug, Error)]` |
| `crates/treelite-xgboost/src/lib.rs` | `load_xgboost_json(&str) -> Result<Model, XgbError>` (≥60 lines) | ✓ VERIFIED | 289 lines; `pub fn load_xgboost_json(json: &str) -> Result<Model, XgbError>` |
| `crates/treelite-xgboost/src/objective.rs` | objective→postprocessor map + f64 transforms (≥30 lines) | ✓ VERIFIED | 71 lines; `get_postprocessor`, `prob_to_margin_sigmoid` (`-ln(1/p-1)`), `prob_to_margin_exponential` (`ln(p)`), `transform_base_score_to_margin` — zero `f32` occurrences |
| `crates/treelite-xgboost/src/error.rs` | `thiserror XgbError` | ✓ VERIFIED | All 6 variants present including `InvalidScalar` (WR-02 fix) |
| `crates/treelite-gtil/src/lib.rs` | `predict(&Model, &[f32], num_row) -> Result<Vec<f32>, GtilError>` (≥60 lines) | ✓ VERIFIED | 240 lines; `pub fn predict(...)`, `evaluate_tree`, `next_node`, `predict_preset`, `PredictScalar` trait |
| `crates/treelite-gtil/src/postprocessor.rs` | identity + sigmoid (f32, exp) (≥15 lines) | ✓ VERIFIED | 36 lines; zero `f64` occurrences; sigmoid: `1.0_f32 / (1.0_f32 + (-sigmoid_alpha * v).exp())` |
| `crates/treelite-gtil/src/error.rs` | `thiserror GtilError` | ✓ VERIFIED | 5 variants including `NodeIndexOutOfBounds` (CR-01 fix) and `InvalidInputShape` (WR-01 fix) |
| `crates/treelite-harness/src/lib.rs` | load golden, run pipeline, assert 1e-5, manifest check, anyhow (≥50 lines) | ✓ VERIFIED | 295 lines; `load_golden`, `run_equivalence`, `check_manifest`, `normalize_nan_tokens` (WR-03 fix), `NanF32` deserializer |
| `crates/treelite-harness/tests/run_equivalence.rs` | unit test against hand-computed scalar, no golden.json dependency (≥20 lines) | ✓ VERIFIED | 181 lines; tests correct match, perturbation detection, missing-path error |
| `crates/treelite-harness/tests/equivalence.rs` | end-to-end 1e-5 spine test against golden (≥20 lines) | ✓ VERIFIED | 51 lines; `equivalence_within_1e5` loads committed fixture, predicts, asserts `max_dev < 1e-5` |
| `fixtures/binary_logistic.model.json` | XGBoost-JSON fixture, `binary:logistic`, `base_score=0.25`, version `[4,7,0]` | ✓ VERIFIED | Confirmed: `"name":"binary:logistic"`, `"base_score":"2.5E-1"`, `"version":[4,7,0]`, 2 trees |
| `fixtures/golden.json` | Frozen golden `{input, output, manifest}`, output in (0,1) | ✓ VERIFIED | Confirmed: 5 rows, output = [0.206, 0.475, 0.109, 0.657, 0.206] — all in (0,1); manifest: treelite=4.7.0, xgboost=3.2.0 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tree.rs` | `tree_buf.rs` | `TreeBuf<T>` column fields | ✓ WIRED | All 20 columns are `TreeBuf<_>` types; confirmed in source |
| `model.rs` | `enums.rs` | `TaskType` header field | ✓ WIRED | `use crate::enums::TaskType; pub task_type: TaskType` confirmed |
| `treelite-xgboost/src/lib.rs` | `treelite_core::Model` | builds and returns `Model` (F32 variant) | ✓ WIRED | `ModelVariant::F32` construction confirmed; `grep -c 'ModelVariant::F32'` = 1 |
| `treelite-xgboost/src/lib.rs` | `objective.rs` | `transform_base_score_to_margin` + `get_postprocessor` | ✓ WIRED | Both functions called in `load_xgboost_json` |
| `treelite-gtil/src/lib.rs` | `treelite_core::Tree` | SoA getters `is_leaf`, `default_child` | ✓ WIRED | `tree.is_leaf(nid)` and `tree.default_child(nid)` in `evaluate_tree` |
| `treelite-gtil/src/lib.rs` | `postprocessor.rs` | `sigmoid`/`identity` by postprocessor name | ✓ WIRED | `postprocessor::sigmoid(...)` and `postprocessor::identity(...)` called in `predict` |
| `treelite-harness/tests/equivalence.rs` | `load_xgboost_json` + `predict` via `run_equivalence` | full pipeline | ✓ WIRED | `treelite_harness::run_equivalence` calls both; spine test confirmed passing |
| `treelite-harness/src/lib.rs` | `fixtures/golden.json` | reads via serde_json in `load_golden` | ✓ WIRED | `std::fs::read_to_string(path)` + `serde_json::from_str(&normalized)` confirmed |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `treelite-harness/tests/equivalence.rs` | `golden.output` | `fixtures/golden.json` (committed frozen file with real upstream predictions) | Yes | ✓ FLOWING |
| `treelite-gtil/src/lib.rs::predict` | `output: Vec<f32>` | `Tree<T>` SoA columns from loaded Model via `evaluate_tree` | Yes | ✓ FLOWING |
| `treelite-xgboost/src/lib.rs::load_xgboost_json` | `model` | `fixtures/binary_logistic.model.json` via serde_json | Yes | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `cargo build --workspace` succeeds | `cargo build --workspace` | `Finished dev profile` (exit 0) | ✓ PASS |
| `cargo test --workspace` passes (52 tests) | `cargo test --workspace` | 52 tests pass, 0 failures | ✓ PASS |
| Equivalence spine test passes within 1e-5 | `cargo test -p treelite-harness --test equivalence -- --nocapture` | `max observed |delta| = 0e0`, 1 passed | ✓ PASS |
| ERR-01 regression: CR-01 (child OOB) | `out_of_bounds_child_node_id_is_typed_error` in predict.rs | ok | ✓ PASS |
| ERR-01 regression: WR-01 (input buffer too small) | `input_buffer_too_small_is_typed_error` in predict.rs | ok | ✓ PASS |
| ERR-01 regression: WR-02 (negative num_feature) | `negative_num_feature_is_typed_error` in predict.rs | ok | ✓ PASS |
| WR-03 fix (non-ASCII bytes preserved) | `non_ascii_bytes_are_preserved_byte_for_byte` in lib.rs tests | ok | ✓ PASS |

### Probe Execution

Step 7c: SKIPPED — no `scripts/*/tests/probe-*.sh` files and phase plan declares no probes. The Behavioral Spot-Checks above serve the same function for this Rust workspace.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| FND-01 | 01-01 | Cargo workspace (edition 2024, resolver "3") builds all member crates from a single pinned table | ✓ SATISFIED | `cargo build --workspace` passes; resolver="3"; edition="2024" in `[workspace.package]` |
| FND-02 | 01-01 | All third-party crates pinned to current latest-stable, no pre-release on critical path | ✓ SATISFIED | `approx="0.5.1"` (not `0.6.0-rc2`); no `rc`/`alpha`/`beta` in `[workspace.dependencies]` |
| ENUM-01 | 01-01 | `TaskType`, `TreeNodeType`, `Operator`, `DType` with upstream-matching string conversions | ✓ SATISFIED | Strings verified against `treelite-mainline/src/enum/*.cc`; 7 enum tests pass |
| CORE-01 | 01-01 | `Model` as two-variant enum over `<f32,f32>`/`<f64,f64>` presets | ✓ SATISFIED | `ModelVariant::F32(ModelPreset<f32>)` and `F64(ModelPreset<f64>)` in model.rs |
| CORE-02 | 01-01 | `Tree<T>` stores all upstream node fields as parallel SoA columns | ✓ SATISFIED | 20 parallel `TreeBuf` columns; no `Node` struct (`grep` count = 0) |
| CORE-03 | 01-01 | `TreeBuf<T>` supports owned and zero-copy borrowed modes | ✓ SATISFIED | `Owned(Vec<T>)` and `Borrowed{ptr,len}`; `from_borrowed` is `unsafe`; borrowed round-trip test passes |
| CORE-04 | 01-01/01-02 | `Model` carries full header metadata with array-typed fields | ✓ SATISFIED | All header fields present; `num_class`/`leaf_vector_shape`/`target_id`/`class_id` are `Vec<i32>`; `base_scores[0]` = exact f64 margin transform `-ln(3) ≈ -1.0986...` verified in `base_scores_is_exact_f64_margin_transform` test |
| ERR-01 | 01-01/01-02/01-03 | Library crates expose typed `thiserror` errors | ✓ SATISFIED | `CoreError`, `XgbError`, `GtilError` all use `thiserror`; CR-01/WR-01/WR-02 regression tests confirm no panics on malformed input |
| ERR-02 | 01-04 | Binaries and tests use `anyhow` for error context | ✓ SATISFIED | `treelite-harness` uses `anyhow` throughout; `load_golden` and `run_equivalence` return `anyhow::Result`; `load_golden_on_missing_path_returns_err_with_context` test passes |

**All 9 declared requirement IDs for Phase 1 are satisfied. No orphaned requirements.**

Note on requirement traceability scope: REQUIREMENTS.md lists FND-01, FND-02, ENUM-01, CORE-01..04, ERR-01, ERR-02 as Phase 1 Complete. The REQUIREMENTS.md footnote acknowledges Phase 1 also exercises a minimal subset of XGB-01 (one JSON model), GTIL-01 (scalar dense predict), and EQV-01/EQV-02 (one golden). Those full requirements are owned by Phases 3 and 5 respectively — correctly deferred.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | No TBD/FIXME/XXX markers found in any phase file | — | None |
| — | — | No empty `return null` / `return {}` stubs in user-facing paths | — | None |
| — | — | No rayon/par_iter in gtil lib.rs | — | None |

**WR-04 (vector base_scores) note:** `crates/treelite-xgboost/src/lib.rs:278` hardcodes `vec![base_score]` (single-element). The code review flagged this as a known Phase 1 narrowing, deliberately deferred to a later multi-output phase. It is not a stub in the current binary:logistic path — it produces the correct result for the fixture. No `TODO`/`TBD` marker is present, but the review document (01-REVIEW.md WR-04) serves as the formal deferred-work record. This is informational (not a blocker) for Phase 1.

**IN-01 (num_class clamp):** upstream clamps to `max(num_class, 1)` at parse; Rust port branches on the raw value but the behavior is identical for all current inputs. Informational only.

### Human Verification Required

None. All success criteria are observable programmatically and all tests pass.

### Gaps Summary

No gaps. All 5 success criteria are fully achieved:

1. `cargo build --workspace` and `cargo test --workspace` pass (52/52 tests). Workspace uses edition 2024, resolver "3", single `[workspace.dependencies]` table, no pre-release crates.
2. All four enums round-trip against their exact upstream C++ string values (verified against `treelite-mainline/src/enum/*.cc`).
3. `Model` is a two-variant enum; `Tree<T>` has 20 SoA columns; `TreeBuf<T>` supports owned and borrowed; header metadata is array-typed and complete.
4. Walking skeleton closes: fixture loads via `load_xgboost_json` → `predict` → `equivalence_within_1e5` passes with `max |delta| = 0e0` (bitwise-exact against upstream Treelite 4.7.0). All code-review findings (CR-01, WR-01, WR-02, WR-03) are fixed and covered by regression tests.
5. `CoreError`/`XgbError`/`GtilError` are typed `thiserror` errors; harness uses `anyhow` throughout.

---

_Verified: 2026-06-10T08:00:00Z_
_Verifier: Claude (gsd-verifier)_

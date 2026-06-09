---
phase: 01-end-to-end-spine
plan: 02
subsystem: xgboost-loader
tags: [rust, xgboost-json, serde, thiserror, objective-map, base-score-margin, f64-transform]

# Dependency graph
requires:
  - phase: 01-01
    provides: treelite-core (Model, ModelVariant::F32, ModelPreset, Tree<f32>, TreeBuf, four enums, CoreError) + committed binary_logistic.model.json fixture
provides:
  - treelite-xgboost::load_xgboost_json(&str) -> Result<Model, XgbError> (F32 variant only)
  - objective->postprocessor map + f64 ProbToMargin/TransformBaseScoreToMargin (ported verbatim)
  - XgbError (thiserror) typed-error surface for the loader leg
affects: [treelite-gtil predict (consumes the loaded Model), treelite-harness equivalence]

# Tech tracking
tech-stack:
  added: []
  patterns: [serde intermediate structs over recognized-key subset, string-scalar params parsed via str::parse, per-tree array-length validation before build, f64 base_score margin transform, always-F32 variant for XGBoost-JSON]

key-files:
  created:
    - crates/treelite-xgboost/src/objective.rs
    - crates/treelite-xgboost/src/error.rs
    - crates/treelite-xgboost/tests/error.rs
    - crates/treelite-xgboost/tests/load_fixture.rs
  modified:
    - crates/treelite-xgboost/src/lib.rs

key-decisions:
  - "load_xgboost_json builds the F32 variant unconditionally — XGBoost-JSON only ever yields <f32,f32> (matches upstream)"
  - "base_score margin transform stays in f64 throughout (sigmoid -ln(1/p-1)); no f32 anywhere in objective.rs (grep == 0)"
  - "Per-tree parallel arrays validated against tree_param.num_nodes before building -> DimensionMismatch, never an OOB index (ERR-01, T-02-01)"
  - "XgbError intentionally does not derive PartialEq/Eq (carries serde_json::Error + Box<dyn Error>); tests match on variants, not =="
  - "Tests match on Result directly instead of expect_err because treelite_core::Model is intentionally not Debug (move-only header)"

requirements-completed: [ERR-01, CORE-04]

# Metrics
duration: 4min
completed: 2026-06-10
---

# Phase 1 Plan 02: XGBoost-JSON Loader Summary

**Implemented `treelite-xgboost::load_xgboost_json` — the minimal XGBoost-JSON loader that parses the committed `binary_logistic.model.json` fixture into a `treelite_core::Model` (F32 variant), porting the objective→postprocessor map and the version-gated f64 `base_score`→margin transform verbatim, and producing the live CORE-04 `base_scores[0]` value `-ln(3) ≈ -1.0986122886681098` rather than the raw probability.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-06-09T21:33:17Z
- **Tasks:** 2 (both `tdd="true"`, plan-level RED/GREEN per task)
- **Files modified:** 5 (1 modified, 4 created)

## Accomplishments

- `objective.rs`: `get_postprocessor` ported verbatim from `xgboost.cc:28-50` (softmax / sigmoid / exponential / hinge / identity groupings, unrecognized → typed `Err`); `prob_to_margin_sigmoid` (`-ln(1/p-1)`), `prob_to_margin_exponential` (`ln(p)`), and `transform_base_score_to_margin` — all in f64 (`grep -c 'f32'` == 0).
- `error.rs`: `XgbError` (thiserror) — `Json(#[from])`, `ParseScalar`, `DimensionMismatch`, `UnrecognizedObjective`, `Core(#[from] CoreError)`.
- `lib.rs`: `load_xgboost_json(&str) -> Result<Model, XgbError>` — serde intermediate structs over the recognized XGBoost-JSON key subset (`delegated_handler.cc:484-490`), string-scalar params parsed via `str::parse`, per-tree array-length validation, the per-node build loop (`delegated_handler.cc:435-479`, `Operator::kLT` always), and the header-metadata finalize (`delegated_handler.cc:847-897`).
- The committed fixture loads into a Model: F32 variant (2 trees), `task_type == kBinaryClf`, `postprocessor == "sigmoid"`, `num_class == [1]`, `leaf_vector_shape == [1,1]`, `target_id == [0,0]`, `class_id == [0,0]`, `average_tree_output == false`, and `base_scores[0] == transform_base_score_to_margin("sigmoid", 0.25)` exactly (version `[4,7,0]` fired the gate).
- 11 tests pass (6 in `tests/error.rs`, 5 in `tests/load_fixture.rs`); `cargo build --workspace` and `cargo clippy --workspace --all-targets` are clean.

## Task Commits

1. **Task 1: Objective→postprocessor map + f64 base_score→margin transform** — `9bea203` (feat) — `objective.rs`, `error.rs`, `lib.rs` wiring, `tests/error.rs` (6 tests).
2. **Task 2: Parse fixture into F32 Model with margin-transformed base_scores** — `71b97ec` (feat) — `lib.rs` loader, `tests/load_fixture.rs` (5 tests).

_Note: both tasks are `tdd="true"`; global `tdd_mode` is `false`, so RED/GREEN were not split into separate commits (tests + impl committed together per task)._

## Files Created/Modified

- `crates/treelite-xgboost/src/objective.rs` — objective map + f64 ProbToMargin/TransformBaseScoreToMargin (ported verbatim from `xgboost.{h,cc}`).
- `crates/treelite-xgboost/src/error.rs` — `XgbError` (thiserror) typed-error surface.
- `crates/treelite-xgboost/src/lib.rs` — `load_xgboost_json` + serde intermediate structs + module wiring/re-exports.
- `crates/treelite-xgboost/tests/error.rs` — objective map, exact f64 sigmoid margin, masking case at 0.5, identity passthrough, unrecognized-objective `Err`.
- `crates/treelite-xgboost/tests/load_fixture.rs` — fixture load → F32/kBinaryClf/sigmoid, exact f64 base_score, tree structure, dimension-mismatch + malformed-JSON typed errors.

_(`Cargo.toml` already declared `treelite-core`/`thiserror`/`serde`/`serde_json` from Wave 1; no manifest change needed.)_

## Decisions Made

- **Always F32:** `load_xgboost_json` builds `ModelVariant::F32` unconditionally — XGBoost-JSON only ever yields `<f32,f32>` (upstream behavior).
- **f64 transform discipline:** the entire `base_score`→margin path is f64; `objective.rs` contains zero `f32` tokens (the #1 silent 1e-5 break). Doc prose reworded to keep the acceptance `grep -c 'f32'` at exactly 0.
- **Validate-before-build:** every per-tree parallel array length is checked against `tree_param.num_nodes` before the node loop, returning `XgbError::DimensionMismatch` (T-02-01, ERR-01) — no out-of-bounds indexing on adversarial input.
- **No `PartialEq` on `XgbError`:** it carries `serde_json::Error` + `Box<dyn Error>`, which aren't `Eq`; tests match on variants instead of `==`.
- **Match-on-Result in tests:** `treelite_core::Model` is intentionally not `Debug` (move-only header), so the error tests pattern-match the `Result` rather than calling `expect_err` (which requires `T: Debug`).

## Deviations from Plan

None — plan executed as written. Two test-mechanics adjustments (Rule 3 — blocking, no behavior change): (1) error tests pattern-match the `Result` instead of `expect_err`, because `Model` is intentionally not `Debug`; (2) the `f32`-token doc comment in `objective.rs` was reworded to "single precision" so the acceptance `grep -c 'f32' == 0` holds without weakening the documentation. Neither changes any runtime behavior.

## Issues Encountered

The fixture's `learner_model_param.num_class` is the string `"0"` (XGBoost's raw param), which is `<= 1`, so the binary/regressor branch fires correctly (`num_class = vec![1; num_target]` → `[1]`). The upstream multi-class branch (`num_class > 1`) is ported for completeness but is not exercised by the Phase-1 fixture.

## Known Stubs

None. `load_xgboost_json` returns a fully-populated `Model` from real fixture data — no hardcoded/placeholder values flow into the returned model. Leaf-vector and category-list columns are legitimately empty for `binary:logistic` (scalar leaves, no categorical splits), matching upstream `HasLeafVector == false`. The multi-class header branch is reachable, ported code (not a stub) but unexercised this phase.

## Threat Flags

None. The loader introduces no new network/auth/file-access surface beyond the in-memory JSON string it is handed; all three mitigations from the plan's threat register are implemented (T-02-01 array-length validation → `DimensionMismatch`; T-02-02 unrecognized objective → `UnrecognizedObjective`; T-02-03 malformed JSON → `Json` via `#[from]`, no `.unwrap()` on parse).

## Verification Evidence

- `cargo test -p treelite-xgboost` — 11/11 pass (6 error + 5 load_fixture).
- `cargo build --workspace` — clean.
- `cargo clippy --workspace --all-targets` — clean.
- `grep -c 'f32' crates/treelite-xgboost/src/objective.rs` → `0` (transform is f64-only).
- `grep -c 'ModelVariant::F32' crates/treelite-xgboost/src/lib.rs` → `1` (F32 variant built).
- `model.base_scores[0] == transform_base_score_to_margin("sigmoid", 0.25)` (≈ `-1.0986122886681098`), asserted exactly and asserted `!= 0.25` (CORE-04 live value, version gate fired).
- Dimension-mismatch and malformed-JSON inputs return typed `XgbError`, never a panic or OOB index (ERR-01).

## Self-Check: PASSED

All four declared created files plus the modified `lib.rs` exist on disk; both task commits (`9bea203`, `71b97ec`) are present in git history.

---
phase: 04-lightgbm-scikit-learn-loaders
plan: 04
subsystem: model-loader
tags: [lightgbm, loader, objective, postprocessor, f64-builder, 1e-5-golden, thiserror]

# Dependency graph
requires:
  - phase: 04-01
    provides: "f64 ModelBuilder mode (leaf_scalar_f64 / numerical_test_f64 → ModelVariant::F64) + treelite-lightgbm placeholder crate"
  - phase: 04-02
    provides: "treelite-gtil::predict widened to (num_row,num_target,max_num_class) with RF averaging + postprocessors"
  - phase: 04-03
    provides: "frozen fixtures/lightgbm_numerical.golden.json (upstream treelite.gtil.predict)"
provides:
  - "treelite-lightgbm crate: load_lightgbm(&str) -> Result<Model, LgbError>"
  - "LightGBM line-based key=value parser with exact per-field precision (LGBModel/LGBTree)"
  - "CanonicalObjective alias collapse + objective→postprocessor map + sigmoid_alpha parse"
  - "negative-index leaf BFS re-numbering + missing-type default_left override"
  - "lightgbm_numerical 1e-5 golden harness test (max |delta| = 0e0)"
affects: [04-05, lightgbm-categorical, sklearn-loaders]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Loader crate mirrors treelite-xgboost: parse.rs / objective.rs / error.rs / lib.rs converge-then-build through the validated ModelBuilder"
    - "LightGBM loads into the F64 variant unconditionally (D-02/D-05); leaf_value/threshold are f64, no downcast"
    - "Categorical splits rejected with a typed error in this slice (LGB-02 deferred to Plan 04-05)"

key-files:
  created:
    - crates/treelite-lightgbm/src/parse.rs
    - crates/treelite-lightgbm/src/objective.rs
    - crates/treelite-lightgbm/src/error.rs
    - crates/treelite-harness/tests/lightgbm.rs
  modified:
    - crates/treelite-lightgbm/Cargo.toml
    - crates/treelite-lightgbm/src/lib.rs
    - crates/treelite-harness/Cargo.toml

key-decisions:
  - "LightGBM loads into ModelVariant::F64 unconditionally — thresholds/leaf values are doubles, emitted through the f64 ModelBuilder (numerical_test_f64/leaf_scalar_f64) with no f32 downcast (D-02/D-05)."
  - "Negative-index leaf re-numbering ported verbatim from lightgbm.cc:533-601: BFS deque seeded (-1,1) for single-leaf / (0,1) otherwise; dfs_index starts at 1, +2 per internal node; leaf value = leaf_value[!old_node_id]; children pushed at the FRONT."
  - "Missing-type default_left override (Pitfall 3): default_left = (0.0 <= threshold) when missing_type != kNaN (lightgbm.cc:579-584); operator is always kLE."
  - "CanonicalObjective alias-collapse runs BEFORE the objective→postprocessor map; sigmoid:<a> parsed with a strict >0 check (T-04-09); class_id[i]=i%num_class round-robin; average_tree_output from average_output key presence; base_scores = num_class zeros; sigmoid_alpha stamped post-commit."
  - "Categorical splits are rejected with a typed LgbError in this slice rather than silently mis-predicting — LGB-02 (cat bitset decode) is Plan 04-05."

patterns-established:
  - "Per-field precision parser: each LightGBM array typed exactly (leaf_value/threshold f64, split_gain f32, decision_type i8, cat_boundaries u64, cat_threshold u32); short/empty arrays return LgbError, never an OOB slice."
  - "LgbError enum mirrors XgbError shape with a Parse{line,detail} positional variant + bounds-checked LeafIndexOutOfRange/NodeIndexOutOfRange + InvalidSigmoidAlpha; transparent Core/Builder bridges; no panic!/anyhow in lib code."

requirements-completed: [LGB-01, LGB-03]

# Metrics
duration: 6min
completed: 2026-06-10
---

# Phase 4 Plan 4: LightGBM Numerical Loader Summary

**A LightGBM text-format model loads → predicts → matches the upstream treelite-GTIL golden bitwise (max |delta| = 0e0 < 1e-5), via a new `treelite-lightgbm` crate with the line parser, the objective→postprocessor map, and the negative-index leaf re-numbering emitted through the f64 ModelBuilder.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-10T04:31:16Z
- **Completed:** 2026-06-10T04:37Z
- **Tasks:** 2 completed
- **Files modified:** 7 (4 created, 3 modified)

## Accomplishments
- New `treelite-lightgbm` crate fleshed out from the Plan 04-01 placeholder: `load_lightgbm(&str) -> Result<Model, LgbError>` loads the vendored `deep_lightgbm/model.txt` numerical model into a `ModelVariant::F64` Model with the correct objective map, `class_id` round-robin, `average_output`, and node-id re-numbering.
- The `lightgbm_numerical` golden harness test passes within the hard `1e-5` gate at **max |delta| = 0e0** (bitwise-exact vs upstream `treelite.gtil.predict`).
- Full `cargo test --workspace` green — no XGBoost/serializer regression (110 tests across the workspace).

## Task Commits

Each task was committed atomically:

1. **Task 1: Create treelite-lightgbm crate — parser, objective map, error enum, converge-then-build** - `3694e8a` (feat) — TDD: tests written alongside implementation in the same commit (objective/parse/node-id unit tests all green at commit time).
2. **Task 2: LightGBM numerical 1e-5 golden harness test** - `b623ab4` (test)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP) committed in the final docs commit.

## Files Created/Modified
- `crates/treelite-lightgbm/src/parse.rs` (created) - Line-based `key=value` tokenizer → typed `LGBModel`/`LGBTree` with exact per-field precision; malformed counts → `LgbError`, never OOB.
- `crates/treelite-lightgbm/src/objective.rs` (created) - `canonical_objective` alias-collapse + `map_objective` postprocessor map + `sigmoid:<a>` parse (>0 check).
- `crates/treelite-lightgbm/src/error.rs` (created) - `LgbError` typed enum (Parse / DimensionMismatch / LeafIndexOutOfRange / NodeIndexOutOfRange / InvalidSigmoidAlpha / UnrecognizedObjective + Core/Builder transparent bridges).
- `crates/treelite-lightgbm/src/lib.rs` (modified) - Placeholder → `load_lightgbm` converge-then-build path: parse → objective → metadata → per-tree BFS re-numbering through the f64 builder → commit → stamp `sigmoid_alpha`.
- `crates/treelite-lightgbm/Cargo.toml` (modified) - Real deps (treelite-core, treelite-builder, thiserror; dev: treelite-gtil + approx, no harness cycle).
- `crates/treelite-harness/tests/lightgbm.rs` (created) - `lightgbm_numerical` test: load → flatten → predict → `assert_abs_diff_eq!(epsilon = 1e-5)` with max_dev tracking + manifest drift warning.
- `crates/treelite-harness/Cargo.toml` (modified) - `treelite-lightgbm` added as a dev-dependency.

## Decisions Made
See `key-decisions` in the frontmatter. Headline: LightGBM is F64-only through the f64 builder; the negative-index leaf re-numbering and missing-type `default_left` override are ported verbatim from `lightgbm.cc`; categorical splits are rejected with a typed error pending Plan 04-05.

## Deviations from Plan

None - plan executed exactly as written.

The plan's `<action>` anticipated categorical-test plumbing reads; this slice scopes LightGBM to numerical splits (LGB-01/LGB-03) and rejects categorical splits with a typed `LgbError` rather than emitting them, exactly as the plan's objective states ("Categorical bitset decode (LGB-02) is the next slice (Plan 05)"). This is the planned scope boundary, not a deviation.

## Issues Encountered
None. The implementation compiled and passed all 14 crate unit tests + the golden test on the first run; the golden matched bitwise (0e0).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **Ready for Plan 04-05 (LightGBM categorical, LGB-02):** the parser already decodes `num_cat`/`cat_boundaries` (u64) / `cat_threshold` (u32) per-field; the `build_tree` BFS has the categorical branch stubbed with a typed rejection and the exact upstream code path (`BitsetToList`, `cat_boundaries[cat_idx]` slicing) to fill in. The builder's `categorical_test` (f32) exists; an f64 categorical path may be needed.
- **No blockers.** The 1e-5 LightGBM-numerical gate is green and protects against regression in the next slice.

## Known Stubs
- `crates/treelite-lightgbm/src/lib.rs` — categorical-split branch in `build_tree` returns `LgbError::Parse` ("categorical split (LGB-02) not yet supported"). Intentional: the plan scopes this slice to numerical models (LGB-01); categorical decode is Plan 04-05. The numerical golden does not exercise this path.
- `crates/treelite-lightgbm/src/parse.rs` — `cat_boundaries`/`cat_threshold` are parsed and stored on `LGBTree` but not yet consumed (no categorical emission this slice). Parsed now so Plan 04-05 only adds the emission, not the parse.

## Self-Check: PASSED

All created files exist on disk (parse.rs, objective.rs, error.rs, lib.rs, tests/lightgbm.rs, 04-04-SUMMARY.md) and both task commits (`3694e8a`, `b623ab4`) are present in git history.

---
*Phase: 04-lightgbm-scikit-learn-loaders*
*Completed: 2026-06-10*

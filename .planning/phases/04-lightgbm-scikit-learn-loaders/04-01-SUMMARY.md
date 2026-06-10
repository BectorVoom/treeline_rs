---
phase: 04-lightgbm-scikit-learn-loaders
plan: 01
subsystem: api
tags: [model-builder, f64, struct-of-arrays, sklearn, lightgbm, fidelity]

# Dependency graph
requires:
  - phase: 02-core-model-builder-serializer
    provides: ModelBuilder f32 state machine, bulk_construct_tree (Tree<f64>), concat assembly precedent
  - phase: 02-core-model-builder-serializer
    provides: ModelVariant::F64 / ModelPreset / Tree<f64> core types
provides:
  - f64 ModelBuilder construction mode (leaf_scalar_f64, leaf_vector_f64, numerical_test_f64) producing ModelVariant::F64 with no downcast
  - bulk_to_model assembling Vec<Tree<f64>> + hand-set metadata into a ModelVariant::F64 Model
  - treelite-lightgbm + treelite-sklearn placeholder crates registered in the workspace
affects: [04-04-lightgbm-loader, 04-06-sklearn-loaders, sklearn-bulk-path, lightgbm-text-loader]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Parallel-staging f64 builder mode (not generic-over-T): f32 path stays byte-identical, f64 values stored without downcast"
    - "fill_common! macro centralizes the 25-column Tree shape so f64 cannot drift from the f32 CR-01/CR-02 discipline"
    - "Numeric-mode latch + mutual-exclusion (MixedNumericMode) to protect the 1e-5 fidelity gate"

key-files:
  created:
    - crates/treelite-builder/tests/f64_mode.rs
    - crates/treelite-builder/tests/bulk_to_model.rs
    - crates/treelite-lightgbm/Cargo.toml
    - crates/treelite-lightgbm/src/lib.rs
    - crates/treelite-sklearn/Cargo.toml
    - crates/treelite-sklearn/src/lib.rs
  modified:
    - crates/treelite-builder/src/lib.rs
    - crates/treelite-builder/src/bulk.rs
    - crates/treelite-builder/src/error.rs
    - Cargo.toml

key-decisions:
  - "Chose the parallel-staging f64 mode (RESEARCH Open Q2 option B) over a generic-over-T state machine: lowest regression risk to the working f32 XGBoost path; NodeStaging carries both f32 and f64 value fields, end_tree branches on a latched is_f64 flag."
  - "Added MixedNumericMode error (Rule 2): mixing f32/f64 entry points in one builder is rejected to prevent silent downcast/discard that would break 1e-5 fidelity."
  - "Created treelite-lightgbm/treelite-sklearn as placeholder crates (Rule 3): registering non-existent members breaks the whole workspace, so minimal lib stubs keep cargo valid while satisfying the plan's single-Wave-1-edit goal."
  - "bulk_to_model sets sigmoid_alpha/ratio_c via Model::new defaults (1.0) and attributes defaults to {} — mirrors the commit_model tail; no topology re-validation (D-09)."

patterns-established:
  - "f64 builder mode: latch on first f64 entry point, guard_f32 on f32 entry points, branch end_tree/commit_model/num_tree on is_f64"
  - "fill_common! macro for the type-agnostic 25-column Tree fill shared by both numeric modes"

requirements-completed: [SKL-01, SKL-02, SKL-03, SKL-04, LGB-01]

# Metrics
duration: 12min
completed: 2026-06-10
---

# Phase 4 Plan 01: f64 ModelBuilder Mode + Bulk Assembly Summary

**f64 construction path for ModelBuilder (leaf/threshold f64 entry points + F64 commit) and a bulk_to_model assembly, exercising ModelVariant::F64 / Tree<f64> end-to-end with zero regression to the f32 XGBoost path**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 (both TDD)
- **Files modified:** 4 modified, 6 created

## Accomplishments
- ModelBuilder now produces a `ModelVariant::F64` model whose `threshold_type()`/`leaf_output_type()` report Float64, with f64 leaf/threshold values stored WITHOUT downcast (verified by a ~1e-7-precision round-trip test).
- The existing f32 XGBoost construction path is byte-identical — still produces `ModelVariant::F32`, all pre-existing tests green.
- `bulk_to_model` assembles `Vec<Tree<f64>>` + hand-set metadata into a `ModelVariant::F64` Model (the sklearn bulk path, `sklearn_bulk.cc:244-330`), preserving `average_tree_output=true` for the downstream RF averaging gate.
- The two Phase-4 loader crates (`treelite-lightgbm`, `treelite-sklearn`) are registered in the root workspace as placeholder crates, so the root `Cargo.toml` is touched exactly once in Wave 1.

## Task Commits

1. **Task 1: Add f64 mode to ModelBuilder (+ register loader crates)** - `f2f2be1` (feat)
2. **Task 2: Add bulk_to_model** - `4f79444` (feat)

_Both tasks marked tdd="true"; RED tests were written first (compile-fail confirmed) then GREEN; each task is a single feat commit (TDD test+impl committed together since the executor was not run in split-commit mode)._

## Files Created/Modified
- `crates/treelite-builder/src/lib.rs` - f64 builder mode: NodeStaging f64 value fields, is_f64 latch, trees_f64 accumulator, leaf_scalar_f64/leaf_vector_f64/numerical_test_f64, guard_f32 on f32 detail methods, fill_common! macro + per-mode branch in end_tree, f64 branch in commit_model/num_tree, bulk_to_model re-export
- `crates/treelite-builder/src/bulk.rs` - bulk_to_model: wraps Tree<f64> + BuilderMetadata into ModelVariant::F64 Model with all 10 header fields hand-set
- `crates/treelite-builder/src/error.rs` - MixedNumericMode error variant
- `crates/treelite-builder/tests/f64_mode.rs` - 4 tests: F64 variant + Float64 type tags, leaf/threshold sub-f32 precision survival, f32 path unchanged
- `crates/treelite-builder/tests/bulk_to_model.rs` - 2 tests: full metadata assembly, average_tree_output preservation
- `crates/treelite-lightgbm/` , `crates/treelite-sklearn/` - placeholder loader crates
- `Cargo.toml` - register the two new workspace members

## Decisions Made
- Parallel-staging f64 mode (RESEARCH Open Q2 option B) over generic-over-T — lowest regression risk; f32 staging untouched.
- `MixedNumericMode` guard added to protect 1e-5 fidelity (Rule 2).
- Placeholder crates created so the early member registration does not break the workspace (Rule 3).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created placeholder crates for the two registered workspace members**
- **Found during:** Task 1 (registering treelite-lightgbm/treelite-sklearn in root Cargo.toml)
- **Issue:** The plan instructs registering the two loader crates in `members` "up front" in Wave 1, but their directories are created by Plans 04-04/04-06. Cargo fails to load the whole workspace when a `members` entry has no `Cargo.toml` ("failed to load manifest for workspace member ... No such file or directory") — this blocked every `cargo test`/`cargo build`.
- **Fix:** Created minimal placeholder crates (Cargo.toml + a doc-only `src/lib.rs` noting the real impl arrives in the later plan) for both `treelite-lightgbm` and `treelite-sklearn`. This satisfies the plan's single-Wave-1-edit intent while keeping the workspace valid.
- **Files modified:** crates/treelite-lightgbm/Cargo.toml, crates/treelite-lightgbm/src/lib.rs, crates/treelite-sklearn/Cargo.toml, crates/treelite-sklearn/src/lib.rs
- **Verification:** `cargo test --workspace` green (40 test binaries) after the stubs were added.
- **Committed in:** f2f2be1 (Task 1 commit)

**2. [Rule 2 - Missing Critical] Added MixedNumericMode error to reject mixing f32/f64 entry points**
- **Found during:** Task 1 (designing the f64 mode)
- **Issue:** Without a guard, calling an f32 entry point and an f64 entry point on the same builder would silently write to one staging type and read from the other, downcasting or discarding values — a direct threat to the 1e-5 fidelity core value.
- **Fix:** Added `BuilderError::MixedNumericMode { existing }`; f64 entry points latch via `latch_f64()`, f32 entry points assert via `guard_f32()`. A builder produces exactly one variant.
- **Files modified:** crates/treelite-builder/src/error.rs, crates/treelite-builder/src/lib.rs
- **Verification:** f32-path-unchanged test + all builder tests green.
- **Committed in:** f2f2be1 (Task 1 commit)

**3. [Rule 3 - Blocking] Staged Cargo.lock with Task 1**
- **Found during:** Task 1 commit
- **Issue:** Adding two workspace members updated `Cargo.lock`; leaving it unstaged would diverge the lockfile from the manifest.
- **Fix:** Staged `Cargo.lock` alongside the manifest edit.
- **Committed in:** f2f2be1 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing-critical)
**Impact on plan:** All auto-fixes necessary for correctness/fidelity and to keep the workspace buildable. No scope creep — placeholder crates export nothing and explicitly defer to the Wave-2 plans.

## Issues Encountered
None beyond the deviations above. The f64 mode and bulk_to_model implementations matched the plan's design intent directly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The `<f64,f64>` construction gate (D-05) is open: both the per-node builder f64 mode and the bulk assembly produce `ModelVariant::F64` without downcasting, ready for the LightGBM (Plan 04-04) and sklearn (Plan 04-06) loaders to route through.
- The two loader crates exist as placeholders; the Wave-2 crate-creation plans no longer need to touch the root `Cargo.toml`.
- No blockers. The 1e-5 XGBoost regression gate remains intact (full workspace tests green).

---
*Phase: 04-lightgbm-scikit-learn-loaders*
*Completed: 2026-06-10*

## Self-Check: PASSED

All created files exist on disk and both task commits (`f2f2be1`, `4f79444`) are in git history.

---
phase: 02-builder-serialization
plan: 05
subsystem: api
tags: [xgboost, model-builder, loader, equivalence, thiserror]

# Dependency graph
requires:
  - phase: 02-builder-serialization
    provides: treelite-builder ModelBuilder (BLD-01) — fluent node-by-node construction with orphan/topology validation
  - phase: 01-end-to-end-spine
    provides: load_xgboost_json skeleton loader + 1e-5 equivalence harness + frozen golden
provides:
  - load_xgboost_json now constructs its Model by driving treelite_builder::ModelBuilder (D-11) instead of hand-assembling Tree columns
  - XgbError::Builder typed propagation of builder failures (no panic crosses the loader boundary)
  - Proof that rewiring the loader through the validated builder leaves predictions bit-identical (max |delta| = 0e0 < 1e-5)
affects: [Phase 3 Full XGBoost Loaders, Phase 4 LightGBM & scikit-learn Loaders]

# Tech tracking
tech-stack:
  added: [treelite-builder (in-repo path dep added to treelite-xgboost)]
  patterns:
    - "Loaders emit builder calls (start_tree / start_node / numerical_test / leaf_scalar / end_node / end_tree / commit_model) rather than hand-filling SoA columns"
    - "Loader validators (require_non_negative / check_dim) run BEFORE builder emission; builder strict validation is defense-in-depth"

key-files:
  created:
    - .planning/phases/02-builder-serialization/02-05-SUMMARY.md
  modified:
    - crates/treelite-xgboost/Cargo.toml
    - crates/treelite-xgboost/src/lib.rs
    - crates/treelite-xgboost/src/error.rs

key-decisions:
  - "load_xgboost_json drives ModelBuilder for tree/node construction (D-11); TreeBuf::from_owned hand-assembly removed from the build path — builder owns column assembly"
  - "Builder failures propagate as #[error(transparent)] XgbError::Builder(#[from] treelite_builder::BuilderError) — thiserror only, no anyhow, no panic across the loader boundary (T-02-X02)"
  - "The objective→postprocessor map, the f64 base_score margin transform, and the F32-only variant choice are unchanged — only the construction mechanism changed"
  - "1e-5 regression gate is the proof of no value drift: max observed |delta| = 0e0 after rewiring (Phase 2 success criterion 1, second half)"

patterns-established:
  - "Pattern: a real loader exercises the builder end-to-end, not just builder unit tests — the equivalence harness is the regression gate that proves no prediction drift"

requirements-completed: [BLD-01]

# Metrics
duration: ~10min
completed: 2026-06-10
---

# Phase 2 Plan 5: Rewire XGBoost Loader through ModelBuilder (D-11) Summary

**`load_xgboost_json` rewired to construct its Model by driving the validated `treelite_builder::ModelBuilder`, with the 1e-5 equivalence harness proving predictions stay bit-identical (max |delta| = 0e0) after the rewiring.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2 (1 auto + 1 human-verify gate)
- **Files modified:** 3

## Accomplishments
- Rewired `load_xgboost_json` to emit `ModelBuilder` calls (`start_tree` / `start_node` / `numerical_test` / `leaf_scalar` / `end_node` / `end_tree` / `commit_model`) — 11 ModelBuilder calls in the build path, 0 `TreeBuf::from_owned` remaining.
- Added the `treelite-builder` in-repo path dependency to `treelite-xgboost` and a typed `XgbError::Builder` arm so builder failures propagate as typed loader errors (no panic, no `anyhow`).
- Preserved the loader's existing validators (`require_non_negative` / `check_dim`) ahead of builder emission as defense-in-depth (T-02-X01); the builder's strict orphan/topology checks are now layered behind them.
- Proved via the 1e-5 regression gate that the rewiring perturbed nothing: equivalence harness `max observed |delta| = 0e0` (< 1e-5), closing Phase 2 success criterion 1's second half.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the builder dependency and rewire load_xgboost_json through ModelBuilder (D-11)** - `5cfa84e` (feat)
2. **Task 2: Verify 1e-5 regression gate after the builder rewiring** - human-verify checkpoint (no commit; independently confirmed GREEN and APPROVED by human)

**Plan metadata:** docs commit (this SUMMARY + STATE + ROADMAP + REQUIREMENTS)

## Files Created/Modified
- `crates/treelite-xgboost/Cargo.toml` - Added `treelite-builder = { path = "../treelite-builder" }` path dependency (D-11)
- `crates/treelite-xgboost/src/lib.rs` - `build_tree` and the model-assembly leg of `load_xgboost_json` now drive `ModelBuilder` instead of hand-assembling `Tree` columns; validators preserved ahead of builder emission
- `crates/treelite-xgboost/src/error.rs` - Added `#[error(transparent)] Builder(#[from] treelite_builder::BuilderError)` arm to `XgbError`

## Decisions Made
- The construction mechanism is the only thing that changed: the objective→postprocessor map, the f64 base_score margin transform, and the unconditional F32 variant choice all stay exactly as in Phase 1.
- Leaf-vs-internal branching maps 1:1 to `leaf_scalar` (when `left_children[i] == -1`) vs `numerical_test` (`split_index`, `threshold`, `op=kLT`, `default_left`, left/right keys) per node.
- Builder errors surface as `XgbError::Builder` (thiserror transparent), never as a panic crossing the loader boundary.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## Verification

- `cargo test -p treelite-harness --test equivalence` → PASS, `max observed |delta| = 0e0` (well under 1e-5) — the D-11 regression gate.
- `cargo test --workspace` → all crates green (builder unit tests, serialize round-trip + golden byte-compare, dump/fields, equivalence harness); 27 test binaries, no failures.
- `grep` confirms 11 ModelBuilder calls and 0 `TreeBuf::from_owned` in the rewired build path.
- Task 2 human-verify gate independently confirmed GREEN and APPROVED.

## Deferred Items

- **DEF-02-01** (XGBoost loader byte-fidelity gap) remains deferred and out of scope for this plan. Golden byte-fidelity is proven via `serialize(deserialize(golden_v5.bin)) == blob` (Plan 02-03), making the serializer gate loader-independent; full loader→serialize byte-fidelity is owned by the Phase 3 XGBoost loader work.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 2 (Builder & Serialization) is now 5/5 complete: validated `ModelBuilder` (BLD-01/02/03), full v5 serialization (SER-01..04), and a real loader rewired through the builder and proven at 1e-5.
- Ready for phase verification, then Phase 3 (Full XGBoost Loaders) — which inherits the builder-driven construction path established here and owns the remaining XGBoost loader byte-fidelity (DEF-02-01).

## Self-Check: PASSED

- FOUND: `.planning/phases/02-builder-serialization/02-05-SUMMARY.md`
- FOUND: Task 1 commit `5cfa84e`

---
*Phase: 02-builder-serialization*
*Completed: 2026-06-10*

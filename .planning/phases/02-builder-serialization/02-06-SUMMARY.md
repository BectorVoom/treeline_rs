---
phase: 02-builder-serialization
plan: 06
subsystem: builder
tags: [model-builder, serialization, byte-fidelity, allocnode, gap-closure]
gap_closure: true

# Dependency graph
requires:
  - phase: 02-builder-serialization
    provides: treelite-builder ModelBuilder (BLD-01) end_tree column finalization
  - phase: 02-builder-serialization
    provides: v5 serializer + golden byte-compare + 1e-5 equivalence harness (SER-01)
provides:
  - end_tree emits the five AllocNode per-node columns (category_list_right_child, leaf_vector_begin/end, category_list_begin/end) at length num_nodes (CR-01 closed)
  - end_tree gates the three stat pairs (data_count/sum_hess/gain + _present) empty-unless-set, per-column independent (CR-02 closed)
  - tests/column_fidelity.rs regression guard for both invariants
affects: [Phase 2 re-verification, Phase 3 Full XGBoost Loaders]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Builder end_tree ports upstream AllocNode column-length invariants verbatim (detail/tree.h:70-101): per-node columns always length num_nodes; stat columns empty-unless-set"

key-files:
  created:
    - crates/treelite-builder/tests/column_fidelity.rs
    - .planning/phases/02-builder-serialization/02-06-SUMMARY.md
  modified:
    - crates/treelite-builder/src/lib.rs

key-decisions:
  - "The five AllocNode per-node columns are populated at length num_nodes with upstream defaults (category_list_right_child=false; begin/end offsets=0 because the builder never fills the leaf_vector_/category_list_ value buffers, so their .Size() is 0 for every node) — porting detail/tree.h:79-84"
  - "Stat columns are emitted only when at least one node set that specific stat (any_data_count / any_sum_hess / any_gain gate TreeBuf::empty() vs from_owned), porting upstream's if(!_present_.Empty()) guards (tree.h:87-98); each stat pair is independent"
  - "The deserializer reads each column by its serialized length, so changing the builder's column lengths keeps serialize→deserialize self-consistent — confirmed by the unchanged golden round-trip and 1e-5 equivalence tests staying green"

patterns-established:
  - "Pattern: a builder-output regression test drills into the committed Model via ModelVariant::F32(preset) and asserts on public column .len() to lock the upstream column-length contract"

requirements-completed: [SER-01]

# Metrics
duration: 6min
tasks: 2
files-changed: 2
completed: 2026-06-10
---

# Phase 2 Plan 06: Builder Column Byte-Fidelity (CR-01 + CR-02) Summary

Closed the two builder-side byte-fidelity defects that phase verification flagged (02-VERIFICATION.md criterion 3 PARTIAL): `end_tree` now ports upstream `AllocNode`'s column-length invariants verbatim, so a treelite-rs-built `Tree<f32>` serializes to a column layout consistent with upstream Treelite's v5 byte image.

## What Was Built

- **CR-01 (AllocNode per-node columns):** `end_tree` now populates `category_list_right_child` (all `false`), `leaf_vector_begin`/`leaf_vector_end`, and `category_list_begin`/`category_list_end` (all `0u64`) at length `num_nodes`, where they were previously left empty (length 0). The begin/end offsets default to 0 because the builder leaves the `leaf_vector`/`category_list` value buffers empty, exactly mirroring upstream `leaf_vector_.Size() == 0` / `category_list_.Size() == 0` for no-leaf-vector, no-category trees (`detail/tree.h:79-84`).
- **CR-02 (empty-unless-set stats):** the per-node fill loop no longer unconditionally pushes `data_count`/`sum_hess`/`gain` and their `_present` companions. Instead three `any_*` flags (computed from the `NodeStaging` present flags set by the `data_count()`/`sum_hess()`/`gain()` setters) gate each stat pair independently: empty (length 0) when no node set that stat, length `num_nodes` once any node did — porting upstream's `if (!*_present_.Empty())` guards (`detail/tree.h:87-98`).
- **Regression guard:** new `crates/treelite-builder/tests/column_fidelity.rs` with two tests — `builder_empty_unless_set_and_allocnode_lengths` (no-stat 3-node tree: stat columns len 0, AllocNode columns len 3 with default false/0 values) and `builder_stat_column_emitted_when_set` (only `sum_hess` set: `sum_hess`/`sum_hess_present` len 3, `data_count`/`gain` len 0 — per-column independence).

## Task Commits

| Task | Name | Commit |
| ---- | ---- | ------ |
| 1 | Fix end_tree column emission (CR-01 + CR-02) | 08f1402 |
| 2 | Add column-fidelity regression test (TDD) | df7ac1c |

## TDD Gate Compliance

Task 2 carried `tdd="true"`. The production fix (Task 1) is the subject under test; the `test(02-06): ...` commit (df7ac1c) lands after the `fix(02-06): ...` GREEN commit (08f1402). Both `column_fidelity` tests genuinely encode the upstream invariant — against the pre-fix code Test A would fail on both directions (stat columns were length num_nodes where the test expects 0; CR-01 columns were length 0 where the test expects num_nodes), so the test is a real regression guard, not a tautology.

## Verification

- `cargo build -p treelite-builder` — exits 0.
- `cargo test -p treelite-builder --test column_fidelity` — 2 passed, 0 failed.
- `cargo test --workspace` — exits 0, 0 failures. The NO-REGRESSION gate held: `serializer_reproduces_golden_v5_byte_for_byte` (golden round-trip), `round_trip_is_byte_identical` / `golden_v5_round_trips_to_itself` (self-consistent round-trip), and the 1e-5 equivalence test all stayed green. The deserializer reads each column by its serialized length, so the new column lengths remain serialize→deserialize self-consistent.
- `cargo clippy -p treelite-builder --tests` — clean.

## Deviations from Plan

None - plan executed exactly as written.

## Scope Respected

- No loader crate touched. DEF-02-01 (XGBoost loader value gap — `sum_hess`/`gain`/CSR columns empty, leaf `split_index=-1`, `attributes={}`) remains owned by Phase 3.
- Plans 02-01..02-05 and their SUMMARYs unmodified.
- The `end_tree` state machine, orphan resolution, and child-key resolution above the column-fill section were left unchanged.

## Self-Check: PASSED

- FOUND: crates/treelite-builder/src/lib.rs (modified, builds + lints clean)
- FOUND: crates/treelite-builder/tests/column_fidelity.rs (2 tests passing)
- FOUND commit 08f1402 (Task 1 fix)
- FOUND commit df7ac1c (Task 2 test)

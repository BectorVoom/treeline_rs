---
phase: quick-260620-dpy
plan: 01
subsystem: treelite-cubecl (GTIL inference kernels)
tags: [perf, cubecl, comptime, cse, fidelity]
requires:
  - "crates/treelite-cubecl/src/kernels/traversal.rs (descend helper)"
  - "crates/treelite-cubecl/src/kernels/default_raw.rs (predict_default_raw kernel)"
  - "crates/treelite-cubecl/src/lib.rs (run_default_raw launcher)"
provides:
  - "descend with hoisted node-index CSE (single (base+nid) compute per iteration)"
  - "predict_default_raw with #[comptime] single_cell straight-line specialization"
  - "host launcher passing single_cell = (cells_per_row == 1) as a comptime bool"
affects:
  - "All three traversal kernels (default_raw, leaf_id, score_per_tree) via shared descend"
  - "Default/Raw predict-kind hot path for single-cell XGBoost binary/regression models"
tech-stack:
  added: []
  patterns:
    - "CubeCL #[comptime] bool specialization with if/else dead-arm pruning (manual: comptime_specialization.md)"
    - "Common-subexpression hoist of per-iteration array index"
key-files:
  created: []
  modified:
    - "crates/treelite-cubecl/src/kernels/traversal.rs"
    - "crates/treelite-cubecl/src/kernels/default_raw.rs"
    - "crates/treelite-cubecl/src/lib.rs"
decisions:
  - "single_cell passed as plain comptime bool (not ScalarArg), per manual #[comptime] use_plane convention"
  - "single-cell arm reuses identical F::new(0.0)/F::cast_from/f64-promote arithmetic so output is byte-identical to one iteration of the general loop"
metrics:
  duration: "~9 min"
  completed: "2026-06-20"
  tasks: 3
  files: 3
---

# Phase quick-260620-dpy Plan 01: Optimise cubecl GTIL kernels (CSE + comptime single-cell) Summary

CSE-hoisted the per-iteration node index in the shared `descend` helper and added a
`#[comptime] single_cell` straight-line specialization to `predict_default_raw` for the
common `cells_per_row == 1` shape — both numerically byte-identical, validated by the
golden matrix at max |delta| = 2.9e-6 (< 1e-5).

## What Was Built

### Task 1 — descend node-index CSE (traversal.rs)
Introduced `let idx = (base + nid) as usize;` as the first statement in the `descend`
`while`-loop body and replaced all five in-body `(base + nid) as usize` indexing sites
(`split_index`, `cright`, `default_left`, `threshold`, `cleft`) with `idx`. The loop guard
(`cleft[(base + nid) as usize] != -1`) is intentionally left recomputing the index for the
next iteration's `!is_leaf` test. Pure read-site substitution: no branch, operand, or
ordering change. All three traversal kernels reuse `descend`, so all benefit from one edit.
Commit: `804184f`.

### Task 2 — comptime single-cell specialization (default_raw.rs + lib.rs)
Added `#[comptime] single_cell: bool` as the final parameter of `predict_default_raw`. The
three per-cell regions (zeroing, RF-average divide, f64 base-score add) now branch
`if single_cell { straight-line n==1 } else { existing while loops }`. The single-cell arm
uses the same `F::new(0.0)`, `F::cast_from(factor)` divide, and `f64::cast_from(...) +
base_scores[0]` promote/narrow arithmetic the general path performs for exactly one cell at
`li = 0`, `cell = row_base`. The serial `for tree_id in 0..num_tree` accumulation block and
the OutputLeafVector/OutputLeafValue branches were NOT touched (GTIL-08 order contract).
`run_default_raw` computes `let single_cell = cells_per_row == 1;` and passes it as a plain
comptime bool to `predict_default_raw::launch`. Commit: `c42b8d9`.

### Task 3 — fidelity gate (no code edit)
Ran the harness golden matrix and the full cubecl suite to prove byte/1e-5 equivalence on
real model fixtures (single-cell XGBoost binary/regression now exercise `single_cell=true`;
multiclass/multi-target fixtures exercise the `single_cell=false` arm).

## Test Results

- `cargo test -p treelite-cubecl` — 34 tests across 9 binaries, all pass (0 failed). Suites:
  determinism (9), predict_kinds (9), postproc (2), upload (3), spike (3), device_absent (7),
  malformed (1), plus integration binaries.
- `cargo test -p treelite-harness --test gtil_matrix_cubecl` — 1 test, PASS.
  160 cells (80 f32-input, 80 f64-input), 48 cubecl-kernel + 112 scalar-fallback,
  **global max |delta| = 2.907300050480899e-6 (< 1e-5)**.
- `cargo clippy -p treelite-cubecl` — clean, no new lints; scoped `#[allow(clippy::eq_op)]`
  on the `fv != fv` NaN test retained.

## Deviations from Plan

None affecting code. The only operational note: the repository's git index was already
carrying ~1600 pre-staged files (a prior uncommitted repo restructuring) before this session
began. To keep the GSD per-task commits atomic, each task commit was scoped with explicit
pathspecs (`git commit -- <file>`) so only the files I edited landed in each commit; the
pre-existing staged blob was left untouched in the index. Task commits `804184f` (1 file) and
`c42b8d9` (2 files) each contain only their task's files.

## Self-Check: PASSED

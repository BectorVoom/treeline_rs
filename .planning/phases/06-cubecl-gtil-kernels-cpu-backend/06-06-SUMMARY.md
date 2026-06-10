---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 06
subsystem: infra
tags: [cubecl, gtil, kernels, cpu-backend, lightgbm, kLE, leaf-vector, fidelity, security]

# Dependency graph
requires:
  - phase: 06-cubecl-gtil-kernels-cpu-backend (plans 06-01..06-05)
    provides: predict_cpu host launcher, descend<F,T> kernel, ragged-SoA upload, gtil_matrix_cubecl provenance gate
provides:
  - "CR-01 operator-coverage fallback gate: any non-kLT model (e.g. LightGBM kLE) defers WHOLE to the scalar reference"
  - "CR-02 f64-promoted in-kernel comparison: descend() compares both operands in f64 (no f64->f32 narrowing)"
  - "CR-03 host-side leaf-vector span validation: out-of-range spans return CubeclError::MalformedLeafVector before any device op"
  - "Matrix provenance widened (model_routes_to_fallback) so non-kLT cells are tagged scalar-fallback (D-06 honesty)"
affects: [06-07-PLAN (real upstream kLE/mixed-width golden capture), phase-07-gpu-backends]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Whole-model operator-coverage fallback gate (mirrors the categorical gate; the cubecl kernel implements only kLT)"
    - "f64-promoted comparison in #[cube] descend (matches the scalar next_node f64 promotion contract from 05-02)"
    - "Host-side leaf-vector span validation before any device op (T-06-09 no-OOB, mirrors GtilError::MalformedLeafVector)"

key-files:
  created:
    - crates/treelite-cubecl/tests/malformed.rs
  modified:
    - crates/treelite-cubecl/src/lib.rs
    - crates/treelite-cubecl/src/kernels/traversal.rs
    - crates/treelite-cubecl/tests/predict_kinds.rs
    - crates/treelite-cubecl/src/upload.rs
    - crates/treelite-cubecl/src/error.rs
    - crates/treelite-harness/tests/gtil_matrix_cubecl.rs

key-decisions:
  - "CR-01 fixed via approach A (whole-model fallback gate), NOT approach B (upload+dispatch cmp in-kernel) — mirrors the proven categorical gate, defers to the exact scalar reference, never touches the hot kernel (D-02)."
  - "validate_leaf_vectors threads num_target/max_num_class so the default_raw broadcast span is bounded; leaf_id/score_per_tree pass (0,0) (no broadcast); absent/short CSR offset columns are treated as scalar leaves (no false positive)."
  - "upload_forest signature gained two params (num_target, max_num_class) at all 3 call sites + the 2 upload.rs test call sites — a localized signature change, not a refactor."

patterns-established:
  - "Operator-coverage gate: the cubecl path covers exactly kLT; every other operator rides the scalar fallback whole-model this phase."
  - "Self-contained regression locks: each gap-closure fix ships a test that would have caught the original bug, independent of downstream golden fixtures."

requirements-completed: [GPU-01, GPU-02, GPU-05]

# Metrics
duration: 9min
completed: 2026-06-10
---

# Phase 6 Plan 06: cubecl Gap-Closure (CR-01/02/03) Summary

**Closes the three 06-VERIFICATION BLOCKERs as pure-code fixes in treelite-cubecl: a non-kLT operator-coverage fallback gate (LightGBM kLE no longer mis-routed), an f64-promoted in-kernel comparison (no f64->f32 narrowing on mixed-width presets), and host-side leaf-vector span validation returning a typed MalformedLeafVector before any device read — each paired with a self-contained regression test.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-06-10T14:14:46Z
- **Completed:** 2026-06-10T14:24:00Z
- **Tasks:** 3
- **Files modified:** 6 (1 created, 5 modified) + this SUMMARY

## Accomplishments

- **CR-01 — operator-coverage fallback gate:** `predict_cpu` now computes `has_non_klt` by scanning every tree's per-node `cmp` column across both `ModelVariant` arms, restricted to internal nodes (`cleft != -1`). Any model carrying a non-`kLT` operator — e.g. every LightGBM numerical model (always `kLE`) — defers WHOLE to `treelite_gtil::predict`, mirroring the categorical gate (approach A, D-02). Previously such models silently reached the kLT-hardcoded kernel and mis-routed every `fv == threshold` tie.
- **CR-02 — f64-promoted comparison:** `descend()` changed the numerical comparison from `fv < F::cast_from(threshold)` (a lossy f64→f32 narrowing) to `f64::cast_from(fv) < f64::cast_from(threshold)`, reproducing the scalar `next_node`'s f64 promotion for every (F, T). An f64 threshold (e.g. `0.1`) between two adjacent f32 values now routes a straddling f32 input to the SAME child as the scalar reference. The misleading "usual-arithmetic-conversion promotion" doc comment was corrected to describe the f64 widening.
- **CR-03 — leaf-vector span validation:** Added `CubeclError::MalformedLeafVector` and a host-side `validate_leaf_vectors` (called from `upload_forest` after `validate_shape`, BEFORE any `client.create_from_slice`). It asserts `begin <= end`, `end <= segment_len`, and `begin + num_target*max_num_class <= segment_len` (the broadcast span the `default_raw` kernel reads) for every leaf node. The T-06-09 no-OOB contract now holds for leaf-vector paths.
- **Matrix provenance honesty (D-06):** `model_has_categorical` widened to `model_routes_to_fallback` (categorical OR non-kLT); non-kLT cells are now tagged `ScalarFallback`, never miscounted as a `CubeclKernel` cell. `gtil_matrix.rs` byte-unchanged (D-11).

## Task Commits

1. **Task 1: CR-01 operator-coverage fallback gate** - `1e9faaf` (fix)
2. **Task 2: CR-02 promote both operands to f64 in descend()** - `9a659b7` (fix; TDD task, single commit — TDD_MODE off, fix+test shipped together per plan)
3. **Task 3: CR-03 leaf-vector span validation + typed error** - `1590a36` (fix; TDD task, single commit)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP) — committed in the final docs commit.

## Files Created/Modified

- `crates/treelite-cubecl/src/lib.rs` - Added the `has_non_klt` whole-model fallback gate in `predict_cpu` (after the categorical gate); threaded `num_target`/`max_num_class` into the 3 `upload_forest` call sites.
- `crates/treelite-cubecl/src/kernels/traversal.rs` - Changed `descend()`'s comparison to f64-promote both operands; corrected the doc comment.
- `crates/treelite-cubecl/tests/predict_kinds.rs` - Added `f64_threshold_f32_input_routes_like_scalar` (F64 preset split at `0.1_f64`, f32 input straddling the f32-unrepresentable boundary, cubecl == scalar routing).
- `crates/treelite-cubecl/src/error.rs` - Added the `MalformedLeafVector` typed variant.
- `crates/treelite-cubecl/src/upload.rs` - Added `validate_leaf_vectors`; called from `upload_forest`; `upload_forest` gained `num_target`/`max_num_class` params.
- `crates/treelite-cubecl/tests/malformed.rs` - New: `predict_cpu` + `validate_leaf_vectors` reject end-past-segment, inverted span, and broadcast overrun with `MalformedLeafVector`; well-formed/scalar columns pass.
- `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` - `model_routes_to_fallback` predicate (categorical OR non-kLT) used in the provenance branch.
- `crates/treelite-cubecl/tests/upload.rs` - Updated the 2 `upload_forest` call sites for the new signature (no behavior change).

## Decisions Made

- **CR-01 approach A over B:** A whole-model fallback gate (not uploading the `cmp` column and branching in-kernel) — it mirrors the proven categorical gate, defers to the exact scalar reference for 1e-5 fidelity, and leaves the hot kernel untouched (consistent with D-02).
- **validate_leaf_vectors broadcast bound:** The `default_raw` multiclass broadcast reads up to `num_target * max_num_class` cells from `begin`, so that span is the safety bound; `leaf_id`/`score_per_tree` read no broadcast and pass `(0, 0)`.
- **Absent CSR columns are scalar leaves:** A binary scalar model may leave `leaf_vector_begin/end` empty (length 0); `validate_leaf_vectors` bounds-checks the column access and skips such leaves, avoiding a false positive (and an indexing panic — see Issues).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bounds-guard the leaf-vector CSR column access in validate_leaf_vectors**
- **Found during:** Task 3 (CR-03 validation)
- **Issue:** The first `validate_leaf_vectors` draft indexed `leaf_vector_begin[n]`/`leaf_vector_end[n]` for every leaf node, but a well-formed binary scalar model (the `upload_ragged_soa_roundtrip` fixture) leaves those CSR offset columns EMPTY (length 0). This panicked (`index out of bounds: the len is 0 but the index is 1`), breaking the pre-existing upload round-trip test.
- **Fix:** Skip leaf nodes whose index is beyond the (absent/short) `leaf_vector_begin`/`leaf_vector_end` columns — such a leaf carries no leaf-vector span and the kernel never reads `leaf_vector` for it. This matches the scalar reference's graceful handling of absent CSR offsets.
- **Files modified:** crates/treelite-cubecl/src/upload.rs (in the new `validate_leaf_vectors` fn, committed with Task 3)
- **Verification:** `upload_ragged_soa_roundtrip` and the new `validate_leaf_vectors_accepts_well_formed` both pass; full `cargo test -p treelite-cubecl` green.
- **Committed in:** `1590a36` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The bounds-guard is a correctness requirement for the new validation fn (it must not panic on a legitimately-empty CSR column). No scope creep — confined to the function the plan asked for.

## Issues Encountered

- Initial `validate_leaf_vectors` panicked on empty CSR offset columns (well-formed binary models). Resolved by bounds-guarding the column access (see Deviation 1). Caught immediately by the pre-existing `upload_ragged_soa_roundtrip` test on the first Task-3 run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three 06-VERIFICATION BLOCKERs are closed CODE-side, each regression-locked by a self-contained test (`f64_threshold_f32_input_routes_like_scalar`, `malformed.rs`, the widened matrix provenance). `cargo test -p treelite-cubecl` (8 test binaries) and `cargo test --workspace` are fully green; `gtil_matrix.rs` is byte-unchanged (D-11).
- **06-07 (depends on this plan)** still owes the *real-upstream-golden* half: capturing a LightGBM kLE fixture and a mixed-width-boundary fixture from upstream Treelite and re-running the matrix gate. The CR-01/CR-02 fixes here are locked by synthetic tests; 06-07 proves them against frozen upstream goldens.
- Phase verification (06-VERIFICATION re-run) is the orchestrator's job — this plan delivers the code-correctness half only.

## Self-Check: PASSED

- Created files exist: `06-06-SUMMARY.md`, `crates/treelite-cubecl/tests/malformed.rs` — both FOUND.
- Task commits exist: `1e9faaf`, `9a659b7`, `1590a36` — all FOUND in git history.

---
*Phase: 06-cubecl-gtil-kernels-cpu-backend*
*Completed: 2026-06-10*

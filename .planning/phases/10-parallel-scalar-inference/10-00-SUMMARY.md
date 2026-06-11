---
phase: 10-parallel-scalar-inference
plan: 00
subsystem: testing
tags: [rayon, gtil, sync, thread-safety, determinism, parallelism]

# Dependency graph
requires:
  - phase: 09-memory-efficiency
    provides: "SmallVec/CompactString Model fields + the Wave-0 model_invariants size-budget test that this plan rewrites"
  - phase: 06-cubecl-cpu-backend
    provides: "treelite-cubecl/tests/determinism.rs split_tree/scalar_model fixtures and the .to_bits() assertion pattern duplicated here"
provides:
  - "rayon 1.12.0 pinned in [workspace.dependencies] and wired into treelite-gtil (first parallelism dependency)"
  - "unsafe impl Sync for Model with the read-only-predict soundness argument (Sync only, not Send)"
  - "GtilError::ThreadPool(String) typed variant for Wave 1's scoped-pool build failure"
  - "model_is_sync_for_readonly_predict invariant test superseding the Phase-9 !Send check (PAR-03)"
  - "tests/determinism.rs and tests/parallel_nthread.rs RED scaffolds (ignored, MISSING markers) defining the Wave-1 contract"
affects: [10-01, parallel-scalar-inference, gtil-predict, wave-1]

# Tech tracking
tech-stack:
  added: [rayon 1.12.0]
  patterns:
    - "unsafe impl <marker> with a multi-point SAFETY doc block (generalized from SendModelRef to the Model type)"
    - "RED #[ignore] integration-test scaffolds with MISSING reason strings that a later wave un-ignores"

key-files:
  created:
    - crates/treelite-gtil/tests/determinism.rs
    - crates/treelite-gtil/tests/parallel_nthread.rs
  modified:
    - Cargo.toml
    - crates/treelite-gtil/Cargo.toml
    - crates/treelite-gtil/src/error.rs
    - crates/treelite-core/src/model.rs
    - crates/treelite-core/tests/model_invariants.rs

key-decisions:
  - "Asserted Sync only on Model, NOT Send (A4) — rayon shares &Model, never moves it; no unsafe impl Send for TreeBuf"
  - "Sparse determinism left as an explicit TODO Wave-1 marker rather than a SparseCsr fixture, keeping the Wave-0 scaffold minimal"
  - "Determinism/nthread fixtures use 64+ rows (not the cubecl 4-row fixture) so the Wave-1 par_chunks_mut split is real, not vacuous"

patterns-established:
  - "Type-level unsafe impl Sync with a documented predict-is-read-only soundness argument mirroring upstream OpenMP Model const& sharing"
  - "MISSING-marked #[ignore] RED scaffolds compile + list under the harness so a later wave un-ignores them to green"

requirements-completed: [PAR-03]

# Metrics
duration: ~12min
completed: 2026-06-11
---

# Phase 10 Plan 00: Parallel Scalar Inference Scaffolding Summary

**Pinned rayon 1.12.0, made `Model` soundly `Sync` for read-only predict, added `GtilError::ThreadPool`, and stood up two RED test scaffolds defining the byte-identical-determinism and nthread-equivalence contract Wave 1 implements against.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-11T04:27Z (approx)
- **Completed:** 2026-06-11T04:39Z
- **Tasks:** 2
- **Files modified:** 5 modified, 2 created

## Accomplishments
- rayon 1.12.0 pinned in the root workspace and wired into `treelite-gtil` — the first parallelism dependency, legitimacy-audited (10-RESEARCH, Approved).
- `unsafe impl Sync for Model` with the multi-point SAFETY argument (predict is `&Model`/read-only; borrowed const-pointer slice outlives the borrow; no interior mutability on the predict path; `stage_serialization_fields` is `&mut self`). Sync only, not Send.
- `GtilError::ThreadPool(String)` added before the `#[error(transparent)]` catch-all so Wave 1's scoped-pool build failure surfaces as a typed error, never a panic (T-10-01).
- `model_invariants.rs` rewritten: `model_is_sync_for_readonly_predict` (`requires_sync::<Model>()` + `requires_send::<&Model>()`) supersedes the Phase-9 `_assert_not_send` check; the `MODEL_SIZE_BUDGET` test is retained unchanged (PAR-03).
- Two net-new gtil integration tests compile as ignored RED scaffolds carrying the intended assertions for Wave 1 to un-ignore.

## Task Commits

Each task was committed atomically:

1. **Task 1: Pin rayon, add GtilError::ThreadPool, make Model: Sync, rewrite the invariant test** - `48642f3` (feat)
2. **Task 2: Stand up the parallel_nthread.rs and determinism.rs RED test scaffolds** - `ff2ac55` (test)

**Plan metadata:** (this docs commit)

## Files Created/Modified
- `Cargo.toml` - Added `rayon = "1.12.0"` to `[workspace.dependencies]` with the legitimacy-audit pin comment.
- `crates/treelite-gtil/Cargo.toml` - Added `rayon = { workspace = true }`.
- `crates/treelite-gtil/src/error.rs` - Added `GtilError::ThreadPool(String)` before the transparent catch-all.
- `crates/treelite-core/src/model.rs` - Added `unsafe impl Sync for Model {}` with the SAFETY doc block after the struct.
- `crates/treelite-core/tests/model_invariants.rs` - Replaced `_assert_not_send` with `model_is_sync_for_readonly_predict`; updated the module doc to the Sync contract; kept the size-budget test.
- `crates/treelite-gtil/tests/determinism.rs` (new) - `determinism_byte_identical_n_runs` ignored RED scaffold: N-run byte-identical predict over 64 rows, all four PredictKinds, via `.to_bits()`.
- `crates/treelite-gtil/tests/parallel_nthread.rs` (new) - `nthread_equivalence` (nthread 0/1/2 byte-identical) + `parallel_uses_more_than_one_core` (>1 rayon worker on a multi-core runner), both ignored RED scaffolds.

## Decisions Made
- **Sync only, not Send** on `Model` (A4): rayon shares `&Model` across workers and never moves the model; a blanket `Send`/`unsafe impl Send for TreeBuf` would be over-broad and is explicitly avoided.
- **Sparse determinism deferred** to an explicit `// TODO Wave 1` marker rather than building a `SparseCsr` fixture now — the Wave-0 scaffold stays minimal while marking the gap.
- **64+ row fixtures** (vs. the cubecl 4-row fixture) so Wave 1's `par_chunks_mut` split is real; a determinism test on a single row proves nothing about parallel row reordering.

## Deviations from Plan

None - plan executed exactly as written. The single literal-grep nuance (the `_assert_not_send` acceptance grep expects 0) was satisfied by also removing the two historical mentions of that name from the module/test doc prose, which is consistent with the plan's "supersede, not linger" intent.

## Issues Encountered
- The Task 1 acceptance criterion `grep -c '_assert_not_send' ... returns 0` initially returned 2 because the rewritten doc comments referenced the old check name in prose. Resolved by rewording the two doc mentions to "not-Send" so the superseded symbol name no longer appears; both invariant tests stay green.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Every symbol Wave 1 (plan 10-01) consumes now exists: `rayon` (pinned + wired), `Model: Sync`, `GtilError::ThreadPool`, and the two ignored RED tests with MISSING markers.
- Wave 1 does pure mechanical rewiring: convert the four serial row loops in `treelite-gtil/src/lib.rs` to `par_chunks_mut`/`map_init`, thread `Config.nthread` into a scoped `ThreadPoolBuilder` (mapping `build()` Err to `GtilError::ThreadPool`), then un-ignore the three scaffold tests.
- No blockers. `cargo test --workspace` green (3 ignored = the new scaffolds); golden serialization fixtures untouched.

## Self-Check: PASSED

- FOUND: crates/treelite-gtil/tests/determinism.rs
- FOUND: crates/treelite-gtil/tests/parallel_nthread.rs
- FOUND: .planning/phases/10-parallel-scalar-inference/10-00-SUMMARY.md
- FOUND: commit 48642f3 (Task 1)
- FOUND: commit ff2ac55 (Task 2)

---
*Phase: 10-parallel-scalar-inference*
*Completed: 2026-06-11*

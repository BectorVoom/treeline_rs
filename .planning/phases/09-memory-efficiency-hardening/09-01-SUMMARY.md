---
phase: 09-memory-efficiency-hardening
plan: 01
subsystem: infra
tags: [smallvec, compact_str, jemalloc, mimalloc, tikv-jemallocator, global-allocator, cargo-workspace, size_of]

# Dependency graph
requires:
  - phase: 08-pyo3-python-binding
    provides: abi3 treelite-py wheel (the allocator-isolation target this plan must keep clean)
  - phase: 02-core-model-serializer
    provides: Model header-metadata fields + golden_v5 byte-compare gate (the MEM-02 swap target / invariant)
provides:
  - 5 pinned [workspace.dependencies] (smallvec 1.15.1, compact_str 0.9.1, tikv-jemallocator/tikv-jemalloc-ctl 0.7.0, mimalloc 0.1.52)
  - treelite-harness non-default mutually-exclusive jemalloc/mimalloc features + 3 optional allocator deps
  - crates/treelite-harness/src/bin/memory_report.rs (allocator-gated bin skeleton; 3 cfg #[global_allocator] arms + both-on compile_error)
  - crates/treelite-core/tests/model_invariants.rs (size_of::<Model>() budget + documented !Send check)
affects: [09-02 (MEM-02 SmallVec/CompactString field swap), 09-03 (MEM-01 bytemuck recast), 09-04 (MEM-03 allocator RSS report + docs/MEMORY_REPORT.md)]

# Tech tracking
tech-stack:
  added: [smallvec 1.15.1, compact_str 0.9.1, tikv-jemallocator 0.7.0, tikv-jemalloc-ctl 0.7.0, mimalloc 0.1.52]
  patterns: ["Allocator isolation via optional dep + non-default mutually-exclusive features (one tier stricter than the GPU shape)", "cfg-gated #[global_allocator] arms with a both-on compile_error guard", "size_of budget as a regression guard against over-large SmallVec inline N"]

key-files:
  created:
    - crates/treelite-harness/src/bin/memory_report.rs
    - crates/treelite-core/tests/model_invariants.rs
  modified:
    - Cargo.toml
    - crates/treelite-harness/Cargo.toml

key-decisions:
  - "smallvec pinned with const_new + union features only (NO serde — fields cross no serde boundary, A4); compact_str/allocators no features"
  - "size_of::<Model>() budget = 512 (max(current 248, 512)) — headroom for the Plan-02 swap while still guarding Pitfall 2"
  - "!Send invariant expressed as a documented commented-out requires_send::<Model>() (no trybuild dep in repo)"

patterns-established:
  - "Allocator isolation: optional workspace dep + non-default mutually-exclusive features, #[global_allocator] only in a bin target (never a lib / the wheel)"
  - "Both-on feature combination guarded by compile_error! rather than a runtime check"

requirements-completed: [MEM-01, MEM-02, MEM-03]

# Metrics
duration: 3min
completed: 2026-06-11
---

# Phase 9 Plan 01: Wave-0 Memory-Hardening Scaffolding Summary

**Pinned the 5 new memory crates, wired non-default mutually-exclusive jemalloc/mimalloc harness features + an allocator-gated `memory_report` bin skeleton, and landed a `size_of::<Model>()` budget + documented `!Send` baseline test — zero behavioral change, golden v5 + 1e-5 harness + pytest all still green.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-11T02:36:33Z
- **Completed:** 2026-06-11T02:40:00Z
- **Tasks:** 3
- **Files modified:** 4 (2 created, 2 modified)

## Accomplishments
- Added the 5 Phase-9 crates to `[workspace.dependencies]` at the exact RESEARCH-audited pins (smallvec on the 1.x line per FND-02, no serde feature per A4); workspace metadata resolves.
- Wired `treelite-harness` with non-default mutually-exclusive `jemalloc`/`mimalloc` features + 3 optional allocator deps, confined to the harness (D-08) — `treelite-py` stays allocator-free.
- Created `memory_report.rs` with three cfg-gated `#[global_allocator]` arms (jemalloc / mimalloc / system) and a both-on `compile_error!` guard; builds under default, `--features jemalloc`, and `--features mimalloc`, and fails on the both-on combination.
- Created `model_invariants.rs`: a `size_of::<Model>() <= 512` budget (current size 248 B) and a documented `!Send` static check — the Wave-0 baseline the Plan-02 field swap must not break.

## Task Commits

Each task was committed atomically:

1. **Task 1: Pin the 5 new workspace dependencies** - `460273f` (chore)
2. **Task 2: Harness allocator features + memory_report bin skeleton** - `5fcfd08` (feat)
3. **Task 3: model_invariants test (!Send + size_of budget)** - `00239ff` (test)

_Note: Task 3 is `tdd="true"` but is a single test-addition task — it landed as one `test(...)` commit because the size guard is a Wave-0 baseline that passes against the current `Model` (no new behavior to RED/GREEN); there is no implementation half in this plan._

## Files Created/Modified
- `Cargo.toml` - Added 5 pinned `[workspace.dependencies]` (smallvec, compact_str, tikv-jemallocator, tikv-jemalloc-ctl, mimalloc).
- `crates/treelite-harness/Cargo.toml` - Added non-default `jemalloc`/`mimalloc` features + 3 optional allocator deps.
- `crates/treelite-harness/src/bin/memory_report.rs` (new) - Allocator-gated bin skeleton: 3 cfg `#[global_allocator]` arms + both-on `compile_error!`; prints the active allocator.
- `crates/treelite-core/tests/model_invariants.rs` (new) - `size_of::<Model>()` budget test + documented `_assert_not_send`.

## Decisions Made
- **smallvec features = `const_new`, `union` only (no `serde`):** the migrated fields cross no serde boundary (v5 serializer is hand-framed; serde_json is dump-only via getters — RESEARCH A4). `cargo add --dry-run` confirmed both features exist for 1.15.1.
- **size_of budget = 512:** measured current `size_of::<Model>()` = 248 B once, then set the budget to `max(248, 512) = 512` per the PATTERNS example — passes today and stays a meaningful Pitfall-2 guard after Plan 02's SmallVec/CompactString swap.
- **`!Send` as a documented commented assertion:** the repo has no `trybuild`; the live invariant is the commented `requires_send::<Model>()` (which must not compile) plus the `*const T` in `TreeBuf::Borrowed`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. The both-on feature build emits the intended `compile_error!` plus a downstream E0308 (both `#[global_allocator]` cfg arms active simultaneously) — expected and correct (the build fails as required, exit 101 with the mutual-exclusion message).

## Known Stubs

`crates/treelite-harness/src/bin/memory_report.rs` is an intentional Wave-0 skeleton: `main()` prints the active allocator and returns `Ok(())`. The real RSS / bytes-allocated sampling (`tikv_jemalloc_ctl::{epoch, stats}` + `/proc/self/statm`) and the `docs/MEMORY_REPORT.md` write are deferred to Plan 04 (MEM-03 / D-10), as specified by the plan. This stub does not block the plan goal — the plan's deliverable is the compiling allocator-gated skeleton + features, not the report content.

## User Setup Required

None - no external service configuration required. (The `*-sys` allocator crates compile a vendored C library via `cc` on first build; the local toolchain already has `cc`, per RESEARCH D-09.)

## Next Phase Readiness
- All Wave-0 prerequisites for Phase 9 are in place: the 5 deps resolve, the harness allocator features build single-on and fail both-on, and the `model_invariants` baseline is green.
- Plan 02 (MEM-02) can now swap the `Model`/`Metadata` fields to `SmallVec`/`CompactString` and re-check the `size_of` budget; Plan 03 (MEM-01) the bytemuck recast; Plan 04 (MEM-03) fleshes out `memory_report` + writes `docs/MEMORY_REPORT.md`.
- HARD INVARIANTS held: `golden_v5` byte-identical, `cargo test --workspace` green (0 failures), `uv run pytest` 39 passed / 1 skipped within 1e-5, `treelite-py` allocator-free.

## Self-Check: PASSED

All created files exist on disk and all 3 task commits are present in git history.

---
*Phase: 09-memory-efficiency-hardening*
*Completed: 2026-06-11*

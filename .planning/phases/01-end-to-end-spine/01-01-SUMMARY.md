---
phase: 01-end-to-end-spine
plan: 01
subsystem: core
tags: [rust, cargo-workspace, soa, enums, thiserror, xgboost-fixture, golden]

# Dependency graph
requires:
  - phase: 01-research
    provides: pinned [workspace.dependencies], exact enum string tables, SoA field set, golden capture script
provides:
  - Virtual Cargo workspace (resolver 3, edition 2024) with four member crates
  - treelite-core foundation layer (enums, TreeBuf<T>, Tree<T>, two-variant Model, CoreError)
  - Frozen golden.json + hand-crafted XGBoost-JSON fixture defining the 1e-5 target
affects: [treelite-xgboost loader, treelite-gtil predict, treelite-harness equivalence]

# Tech tracking
tech-stack:
  added: [thiserror 2.0.18, serde 1.0.228, anyhow 1.0.102, serde_json 1.0.150, approx 0.5.1]
  patterns: [struct-of-arrays node columns, owned/borrowed storage primitive, typed thiserror errors, two-variant Model enum, move-only no-casual-Clone]

key-files:
  created:
    - Cargo.toml
    - crates/treelite-core/src/enums.rs
    - crates/treelite-core/src/error.rs
    - crates/treelite-core/src/tree_buf.rs
    - crates/treelite-core/src/tree.rs
    - crates/treelite-core/src/model.rs
    - crates/treelite-core/src/lib.rs
    - fixtures/golden.json
  modified:
    - .gitignore

key-decisions:
  - "Enum variant names mirror upstream kXxx verbatim; non_camel_case_types lint suppressed at module level for porting fidelity"
  - "Inherent from_str (not std::str::FromStr) mirrors upstream FromString fallible-parse API; should_implement_trait lint suppressed"
  - "TreeBuf<T> is a two-mode enum (Owned(Vec<T>) / Borrowed{ptr,len}); T: Copy mirrors upstream POD bound, bytemuck deferred to Phase 9"
  - "num_class/leaf_vector_shape/target_id/class_id are Vec<i32> (array-typed) per tree.h:543-547, NOT scalars as ROADMAP wording implied"

patterns-established:
  - "Struct-of-Arrays: every Tree node field is a separate parallel TreeBuf column; no Node struct (verified grep count 0)"
  - "Typed errors: every upstream TREELITE_LOG(FATAL) path becomes a returned CoreError, never a panic (ERR-01)"
  - "Move-only: TreeBuf/Tree/Model expose explicit deep_copy instead of casual #[derive(Clone)]"

requirements-completed: [FND-01, FND-02, ENUM-01, CORE-01, CORE-02, CORE-03, CORE-04, ERR-01]

# Metrics
duration: 5min
completed: 2026-06-10
---

# Phase 1 Plan 01: End-to-End Spine Foundation Summary

**Stood up the virtual Cargo workspace and a fully-implemented `treelite-core` (four upstream-exact enums, `TreeBuf<T>` SoA primitive, `Tree<T>` node columns, two-variant `Model` with array-typed header metadata) plus the frozen golden artifact that fixes the 1e-5 target before any Rust code computes a prediction.**

## Performance

- **Duration:** ~5 min (continuation session)
- **Started:** 2026-06-10T06:23:47+09:00 (golden freeze commit)
- **Completed:** 2026-06-10T06:28:28+09:00
- **Tasks:** 3 (Task 1 completion + Tasks 2-3)
- **Files modified:** 20 (across the plan)

## Accomplishments
- Virtual workspace root (resolver 3, edition 2024, single `[workspace.dependencies]` table) with four member crates; the three downstream crates are compile-only stubs wired for Waves 2-3.
- `treelite-core`: four enums (`TaskType`/`TreeNodeType`/`Operator`/`DType`) round-tripping their EXACT non-uniform upstream strings, including `Operator::kNone → ""` and `DType::from_str` rejecting `"invalid"` while `kInvalid.as_str() == "invalid"`.
- `TreeBuf<T>` (Owned + zero-copy Borrowed), `Tree<T>` with all ~20 parallel SoA columns and upstream-semantics getters, `ModelPreset<T>` + two-variant `ModelVariant::F32/F64`, and `Model` with the full array-typed header metadata field set.
- 18 `treelite-core` tests pass; `cargo build --workspace` and `cargo clippy --workspace` are clean.
- Frozen `golden.json` (treelite 4.7.0, output values in (0,1)) committed unmodified.

## Task Commits

1. **Task 1: Freeze captured golden artifact** - `f65acf1` (feat) — committed the human-captured `fixtures/golden.json` unmodified (the Claude-authorable fixture + capture script were committed earlier in `768215d`).
2. **Task 2: Workspace + four enums** - `1f1f365` (feat) — workspace manifest, member crates, enums, `CoreError`, 7 enum tests.
3. **Task 3: TreeBuf / Tree / Model** - `83afa44` (feat) — SoA storage + tree + two-variant model, 11 tests.

_Note: Plan-level TDD structure (tests + impl per task) was followed; global `tdd_mode` is `false` so RED/GREEN were not split into separate commits._

## Files Created/Modified
- `Cargo.toml` — virtual workspace root; resolver 3, edition 2024, pinned `[workspace.dependencies]`.
- `crates/treelite-core/src/enums.rs` — four enums with exact upstream string tables + integer reprs.
- `crates/treelite-core/src/error.rs` — `CoreError` (thiserror) with `UnknownEnumString`.
- `crates/treelite-core/src/tree_buf.rs` — `TreeBuf<T>` owned/borrowed storage primitive.
- `crates/treelite-core/src/tree.rs` — `Tree<T>` SoA columns + traversal getters.
- `crates/treelite-core/src/model.rs` — `ModelPreset<T>`, `ModelVariant`, `Model` + header metadata.
- `crates/treelite-core/src/lib.rs` — module wiring + re-exports.
- `crates/treelite-core/tests/{enums,tree_buf,tree_model}.rs` — 18 tests.
- `crates/treelite-{xgboost,gtil,harness}/` — compile-only stubs for Waves 2-3.
- `fixtures/golden.json` — frozen golden {input, output, manifest}.
- `.gitignore` — `/target` entry (added by cargo).

## Decisions Made
- Suppressed `non_camel_case_types` at the `enums` module level so variant names mirror upstream `kXxx` enumerators verbatim — porting fidelity outweighs Rust naming convention here.
- Suppressed `clippy::should_implement_trait` for the inherent `from_str` methods, which intentionally mirror upstream `FromString` (a fallible parser returning the crate's typed `CoreError`).
- `TreeBuf<T>` is an `enum { Owned(Vec<T>), Borrowed { ptr, len } }` with `T: Copy` (the upstream POD bound); `bytemuck` is deliberately NOT pulled in (Phase 9 seam).
- Confirmed `num_class`/`leaf_vector_shape`/`target_id`/`class_id` are `Vec<i32>` per `tree.h:543-547`, the critical deviation from ROADMAP scalar wording.

## Deviations from Plan

None - plan executed exactly as written. The only out-of-plan additions were two module-level lint allows (`non_camel_case_types`, `clippy::should_implement_trait`), both required to keep the build/clippy clean while preserving the plan-mandated upstream-verbatim API surface (Rule 3 - blocking, no behavior change).

## Issues Encountered
None. `src/main.rs` was untracked (created by prior cargo activity) so it was removed from the filesystem rather than via `git rm`; the workspace root became virtual as planned.

## Known Stubs
The three downstream crates (`treelite-xgboost`, `treelite-gtil`, `treelite-harness`) ship a single `crate_name()` placeholder function each. These are intentional Wave 1 compile-only stubs — they exist solely so `cargo build --workspace` passes and are implemented in Waves 2-3 (loader, predict, equivalence harness) per the phase plan. No stub flows into a user-facing prediction path this wave.

## Verification Evidence
- `cargo build --workspace` — clean (FND-01, FND-02; no pre-release crate in `[workspace.dependencies]`).
- `cargo test -p treelite-core` — 18/18 pass (ENUM-01, CORE-01..04, ERR-01).
- `cargo clippy --workspace --all-targets` — clean.
- `grep -v '^[[:space:]]*//' crates/treelite-core/src/tree.rs | grep -c 'struct Node'` → `0` (no Node struct).
- `fixtures/golden.json` present with `{input, output, manifest}`; all outputs in (0,1) (D-06/D-07).

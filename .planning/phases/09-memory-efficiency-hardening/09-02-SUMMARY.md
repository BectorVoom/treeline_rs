---
phase: 09-memory-efficiency-hardening
plan: 02
subsystem: core-model
tags: [smallvec, compact_str, mem-02, deref-transparency, size_of, golden-byte-compat, struct-of-arrays]

# Dependency graph
requires:
  - phase: 09-memory-efficiency-hardening
    plan: 01
    provides: 5 pinned [workspace.dependencies] (smallvec 1.15.1 / compact_str 0.9.1) + size_of::<Model>() budget guard
  - phase: 02-core-model-serializer
    provides: Model header-metadata fields + golden_v5 byte-compare gate (the MEM-02 swap target / invariant)
provides:
  - "Model: 7 metadata fields are SmallVec<[i32;N]>/SmallVec<[f64;1]>/CompactString (D-04)"
  - "builder::BuilderMetadata fields migrated in lockstep (no per-field assign churn)"
  - "serializer read-back assigns carry .into(); EMIT path UNCHANGED (deref-transparent)"
  - "all loaders (xgboost/lightgbm/sklearn bulk+histgb+mixin) write the migrated fields"
affects: [09-03 (MEM-01 bytemuck recast — shares the serialize emit seam), 09-04 (MEM-03 allocator RSS report + size_of before/after row)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deref-transparency migration: SmallVec<[T;N]> derefs to &[T] and CompactString to &str, so the read-only consumers (serializer EMIT, gtil shape.rs, json dump via slice) need zero or .into()-only edits"
    - "Inline N chosen for the DOMINANT shape (1/2), not the max — size_of::<Model>() stays byte-identical to the Vec/String baseline (248 B), no Pitfall-2 bloat"

key-files:
  created: []
  modified:
    - crates/treelite-core/Cargo.toml
    - crates/treelite-builder/Cargo.toml
    - crates/treelite-core/src/model.rs
    - crates/treelite-core/src/serialize/mod.rs
    - crates/treelite-core/src/serialize/json.rs
    - crates/treelite-builder/src/lib.rs
    - crates/treelite-builder/src/bulk.rs
    - crates/treelite-builder/src/concat.rs
    - crates/treelite-xgboost/src/lib.rs
    - crates/treelite-lightgbm/src/lib.rs
    - crates/treelite-sklearn/src/bulk.rs
    - crates/treelite-sklearn/src/histgb.rs
    - crates/treelite-sklearn/src/mixin.rs
    - crates/treelite-sklearn/src/lib.rs

key-decisions:
  - "Inline N = [1] for num_class/target_id/class_id/base_scores and [2] for leaf_vector_shape (the always-2-tuple): size_of::<Model>() = 248 B, BYTE-IDENTICAL to the pre-migration Vec/String layout — no budget pressure, no Pitfall-2 reduction needed."
  - "json.rs DumpAsJSON path (NOT in the plan/PATTERNS file list) required derefing the migrated fields to &[T]/&str — smallvec has NO serde feature (A4), so json!()/Value::from operate on the slice/str. Same JSON output, no serde feature added (Rule 3 blocking fix)."
  - "BuilderMetadata.attributes kept Option<CompactString>; the '{}' fallback became CompactString::from(\"{}\") at both assign sites (lib.rs + bulk.rs)."
  - "concat.rs locals changed to SmallVec<[i32;1]> so extend_from_slice + field assigns stay verbatim (no .into() at the merge)."

requirements-completed: [MEM-02]

# Metrics
duration: 13min
completed: 2026-06-11
---

# Phase 9 Plan 02: Model + Metadata SmallVec/CompactString Migration (MEM-02) Summary

**Migrated the 7 public `Model` header-metadata fields (and `builder::BuilderMetadata` in lockstep) from `Vec<i32>`/`Vec<f64>`/`String` to `SmallVec<[i32; N]>`/`SmallVec<[f64; 1]>`/`CompactString` — a pure storage-type swap with ZERO behavior change, proven by the golden v5 byte-compare (both fixtures byte-identical) and the 1e-5 equivalence harness; `size_of::<Model>()` stays at 248 B (no inline-N bloat) and `Model` stays `!Send`.**

## Performance

- **Duration:** ~13 min
- **Started:** 2026-06-11T02:43:41Z
- **Completed:** 2026-06-11
- **Tasks:** 3
- **Files modified:** 14 production/config + 17 test files (.into()/.as_slice() ripple)

## Accomplishments

- **Task 1 — field migration (the ripple source):** `model.rs` swaps the 7 fields to `SmallVec<[i32; 1]>` (num_class/target_id/class_id), `SmallVec<[i32; 2]>` (leaf_vector_shape), `SmallVec<[f64; 1]>` (base_scores), `CompactString` (postprocessor/attributes); `Model::new()` initializers updated. `builder::BuilderMetadata` migrated to the SAME types so the `commit_model`/`bulk_to_model` assign blocks stay verbatim (only the `"{}"` fallback became `CompactString::from`). `concat.rs` locals → `SmallVec<[i32; 1]>`. The serializer read-back (in-crate) wrapped `.into()` at the 7 assign sites.
- **Task 2 — loaders + serializer:** xgboost/lightgbm/sklearn (bulk + histgb + mixin) `BuilderMetadata` literals carry `.into()` at the `vec![]`/`.to_string()` sites. The serializer **EMIT path is unchanged** (deref-transparent), and `gtil/src/shape.rs` + `treelite-py/` have **zero diff** (verified — the slice/iter API derefs through SmallVec).
- **Task 3 — both hard invariants green:** golden_v5.bin AND golden_v5_3format.bin byte-identical; full harness equivalence/matrix within 1e-5; `cargo test --workspace` 0 failures; `uv run pytest crates/treelite-py` 39 passed / 1 skipped; `size_of::<Model>()` = 248 B (budget 512, held without reduction); `Model` `!Send`; `treelite-py` allocator-free.

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate Model + builder::Metadata field types** — `73e51e7` (feat)
2. **Task 2: Update loaders + serializer read-back to migrated types** — `92899fe` (feat)
3. **Task 3: Prove both invariants green (golden + 1e-5 + pytest)** — `465da0a` (test)

_Task 1 is `tdd="true"`; the RED/GREEN guard is the pre-existing `model_invariants` size/`!Send` test + the serialize-roundtrip + golden suite (the Wave-0 baseline from Plan 01). This is a storage-type migration with zero new behavior, so — like Plan 01 Task 3 — the existing green tests are the gate; there is no separate failing-then-passing behavior to author. The migration kept every gate green._

## Files Created/Modified

- `crates/treelite-core/src/model.rs` — 7 metadata fields → SmallVec/CompactString; `new()` initializers; `use` imports.
- `crates/treelite-core/src/serialize/mod.rs` — read-back assigns wrapped `.into()` (7 sites); EMIT block untouched.
- `crates/treelite-core/src/serialize/json.rs` — DumpAsJSON derefs migrated fields to `&[T]`/`&str` (no smallvec serde feature; identical JSON).
- `crates/treelite-builder/src/{lib.rs,bulk.rs,concat.rs}` — BuilderMetadata fields migrated; `"{}"` fallback via `CompactString::from`; concat locals → SmallVec.
- `crates/treelite-{xgboost,lightgbm}/src/lib.rs`, `crates/treelite-sklearn/src/{bulk.rs,histgb.rs,mixin.rs}` — loader `BuilderMetadata` literals carry `.into()`.
- `crates/treelite-{core,builder}/Cargo.toml` — `smallvec`/`compact_str` as `{ workspace = true }`.
- Test ripple (`.into()` / `.as_slice()`): core (3), builder (5), gtil (6), xgboost (1), cubecl (4) test files + inline `#[cfg(test)]` modules in lightgbm/sklearn lib.rs + sklearn mixin.rs.

## Decisions Made

- **Inline N for the dominant shape:** `[1]`/`[2]` keep `size_of::<Model>()` byte-identical to the old `Vec`/`String` (24 B each → SmallVec/CompactString same footprint), so the migration is free on struct size — no Pitfall-2 reduction was needed and the 512-byte budget held with 264 B to spare.
- **json.rs deref (out-of-plan Rule-3 fix):** the `DumpAsJSON` path was not in PATTERNS' file list but breaks without the smallvec `serde` feature (deliberately absent, A4). Fixed by derefing to slice/`&str` at the 7 `json!`/`Value::from` call sites — same emitted JSON, no new dependency feature.
- **BuilderMetadata migrated in lockstep (RESEARCH Open-Q1):** changing the builder field types too means the `commit_model`/`bulk_to_model` assign blocks need no per-field `.into()` — the only change is the `"{}"` fallback constructor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `serialize/json.rs` DumpAsJSON path required deref edits (not in PATTERNS file list)**
- **Found during:** Task 1 (compiling treelite-core after the field swap)
- **Issue:** `json.rs` builds the dump JSON via `json!(m.num_class)` / `Value::from(m.attributes.clone())`. These rely on `Vec`/`String` implementing `Serialize`/`From<Value>`; `SmallVec`/`CompactString` do not (no serde feature, A4), so the lib failed to compile.
- **Fix:** deref each migrated field to its slice/str at the call site (`json!(&m.num_class[..])`, `Value::from(m.postprocessor.as_str())`). Identical emitted JSON (same values), no serde feature added.
- **Files modified:** `crates/treelite-core/src/serialize/json.rs`
- **Commit:** `73e51e7`

**2. [Rule 3 - Blocking] Extra loader files (`sklearn/histgb.rs`, `sklearn/mixin.rs`) carry BuilderMetadata literals**
- **Found during:** Task 2 (workspace build)
- **Issue:** the plan listed `sklearn/bulk.rs` but `histgb.rs` (4 literals) and `mixin.rs` (4 literals) also construct `BuilderMetadata` — they must write the migrated types to compile.
- **Fix:** `.into()` at every `vec![]`/`.to_string()` field in those literals (same mechanical change as the listed loaders).
- **Files modified:** `crates/treelite-sklearn/src/{histgb.rs,mixin.rs}`
- **Commit:** `92899fe`

**3. [Rule 3 - Blocking] Test-suite ripple across gtil/xgboost/cubecl + inline src test modules**
- **Found during:** Tasks 1–3 (test compilation)
- **Issue:** numerous `#[test]` helper builders assign `m.field = vec![...]` and assert `assert_eq!(model.field, vec![...])` against migrated fields; these don't compile against SmallVec/CompactString.
- **Fix:** append `.into()` to assignment exprs; switch field assertions to `.as_slice()` vs array literals. Test intent and asserted values unchanged.
- **Files modified:** core (3), builder (5), gtil (6), xgboost (1), cubecl (4) test files + inline `#[cfg(test)]` modules in lightgbm/sklearn `lib.rs` + sklearn `mixin.rs`.
- **Commits:** `73e51e7`, `92899fe`, `465da0a`

## Issues Encountered

None blocking. One tooling note: `uv run maturin develop` warns it cannot set rpath (`patchelf` not installed) but still builds + installs the abi3 wheel successfully and pytest passes — pre-existing environment state, not introduced by this plan.

## Known Stubs

None. This plan is a complete storage-type migration; every consumer is wired to the migrated types.

## Threat Flags

None. No new network/auth/file-access surface; the v5 read path is unchanged (still the element-wise bounds-checked `Reader::array`/`Reader::string` + `.into()` — no `cast_slice` over untrusted input, T-09-D mitigated). SmallVec/CompactString deref to the identical wire payload (T-09-05 gated by the byte-identical golden compare on both fixtures).

## Next Phase Readiness

- MEM-02 closed: `Model`/`BuilderMetadata` are SmallVec/CompactString-backed; both hard invariants green.
- Plan 03 (MEM-01) can now tighten the serializer `le_bytes_of` to `bytemuck::cast_slice` over the same EMIT seam this plan left untouched (the migrated fields deref to `&[i32]`/`&[f64]` Pod slices, ready for the recast).
- Plan 04 (MEM-03) can record the `size_of::<Model>()` = 248 B before/after row in `docs/MEMORY_REPORT.md` (no change pre→post MEM-02, the headline "compact data structures with zero struct-size cost" result).

## Self-Check: PASSED

- `crates/treelite-core/src/model.rs` exists with SmallVec/CompactString fields (grep: 16 SmallVec, 7 CompactString).
- Commits `73e51e7`, `92899fe`, `465da0a` present in git history.
- golden_v5 (both fixtures), full workspace (0 failures), pytest (39/1), model_invariants (size 248 ≤ 512, !Send) all green; gtil shape.rs + treelite-py zero diff.

---
*Phase: 09-memory-efficiency-hardening*
*Completed: 2026-06-11*

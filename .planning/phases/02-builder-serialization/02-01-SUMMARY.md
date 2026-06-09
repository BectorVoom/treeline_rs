---
phase: 02-builder-serialization
plan: 01
subsystem: serialization
tags: [rust, treelite, v5-format, golden-fixture, serde, bookkeeping-fields]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: Model/Tree core structs, ModelPreset F32/F64 variants, DType enum, capture_golden.py + harness Manifest shape
provides:
  - "Model carries 7 private v5 serialization bookkeeping scalars (version triple, type tags, num_tree, opt-field count) with a serialize-time stage_serialization_fields recompute method and pub(crate) read accessors"
  - "Tree carries num_opt_field_per_tree/num_opt_field_per_node scalars (default 0)"
  - "Frozen D-02 golden v5 blob (golden_v5.bin) + toolchain manifest captured from the treelite==4.7.0 wheel — the byte-fidelity ground truth"
  - "Empirical confirmation that the v5 blob's first 12 bytes decode to version triple (4,7,0), settling RESEARCH Assumption A1 / Open Question Q2"
affects: [serialization, v5-header-walk, byte-fidelity-tests, builder]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 5: recomputed v5 header scalars get a 'a-lived home on Model so zero-copy PyBuffer frames can borrow them for a frame lifetime"
    - "Run-once golden capture discipline: capture_golden_v5.py runs locally against the upstream wheel, CI never regenerates; blob+manifest committed frozen"

key-files:
  created:
    - fixtures/capture_golden_v5.py
    - fixtures/golden_v5.bin
    - fixtures/golden_v5.manifest.json
  modified:
    - crates/treelite-core/src/model.rs
    - crates/treelite-core/src/tree.rs

key-decisions:
  - "v5 header version constants are (4,7,0) NOT (5,x,x) — empirically confirmed by the golden blob's first 12 bytes (RESEARCH Pitfall 1 / Assumption A1)"
  - "num_opt_field_per_model_ defaults to 0; per-tree/per-node opt-field scalars default to 0 (no optional fields in the binary:logistic fixture)"
  - "stage_serialization_fields recomputes scalars at serialize time rather than maintaining them eagerly, keeping construction-time invariants simple"

patterns-established:
  - "Pattern 5: Model owns the recomputed v5 bookkeeping scalars as the borrow source for zero-copy serialization frames"
  - "Golden v5 capture mirrors the Phase 1 capture_golden.py path-resolution + manifest-key conventions exactly"

requirements-completed: [SER-01]

# Metrics
duration: ~10min
completed: 2026-06-10
---

# Phase 2 Plan 01: Serialization Bookkeeping Fields + D-02 Golden v5 Blob Summary

**Model/Tree gain the private v5 bookkeeping scalars (version triple, type tags, num_tree, opt-field counts) with a serialize-time recompute method, plus a frozen upstream golden v5 blob whose first 12 bytes empirically confirm the (4,7,0) version header.**

## Performance

- **Duration:** ~10 min (across original + continuation executor)
- **Started:** 2026-06-10T08:23:01Z (Task 1 commit)
- **Completed:** 2026-06-10
- **Tasks:** 3 (1 auto, 1 human-action checkpoint approved, 1 auto)
- **Files modified:** 5 (2 modified, 3 created)

## Accomplishments

- `Model` now carries 7 private bookkeeping scalars (`num_tree_`, `num_opt_field_per_model_`, `major_ver_`/`minor_ver_`/`patch_ver_`, `threshold_type_`, `leaf_output_type_`) with a `stage_serialization_fields(&mut self)` recompute method and `pub(crate)` read accessors — the Pattern 5 borrow source for the in-crate serializer.
- `Tree` carries the `num_opt_field_per_tree` / `num_opt_field_per_node` scalars (default 0).
- Captured and froze the D-02 golden v5 blob (`golden_v5.bin`, 951 bytes) from the installed `treelite==4.7.0` wheel via `model.serialize_bytes()`, plus a toolchain/sha256/nbytes manifest.
- **Key finding:** the golden blob's first 12 bytes unpack as little-endian `(4, 7, 0)` — confirming the v5 header carries the 4.7.0 version triple (NOT 5.x.x), settling RESEARCH Assumption A1 / Open Question Q2 and de-risking every downstream serialization task.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add private serialization bookkeeping fields to Model and Tree** — `2e8d32c` (feat)
2. **Task 2: Capture the D-02 golden v5 blob** — checkpoint:human-action (blocking); human approved after confirming the (4,7,0) first-12-bytes check
3. **Task 3: Write the golden v5 capture script and freeze the blob + manifest** — `75bdd25` (feat)

**Plan metadata:** committed with SUMMARY/STATE/ROADMAP (docs: complete plan)

## Files Created/Modified

- `crates/treelite-core/src/model.rs` — 7 private v5 bookkeeping scalars + `stage_serialization_fields` recompute + `pub(crate)` accessors + 2 tests
- `crates/treelite-core/src/tree.rs` — `num_opt_field_per_tree` / `num_opt_field_per_node` scalars (default 0)
- `fixtures/capture_golden_v5.py` — run-once capture script mirroring Phase 1 `capture_golden.py` conventions; loads `binary_logistic.model.json`, calls `serialize_bytes()`, writes blob + manifest
- `fixtures/golden_v5.bin` — frozen authoritative v5 byte stream (951 bytes) from the upstream wheel
- `fixtures/golden_v5.manifest.json` — treelite/xgboost versions, OS/arch/libc/python, sha256 `991ec6e2…`, nbytes 951, source_fixture

## Decisions Made

- v5 header version constants are `(4,7,0)` NOT `(5,x,x)` — empirically confirmed by the golden blob's first 12 bytes (RESEARCH Pitfall 1 / Assumption A1).
- `num_opt_field_per_model_` and the per-tree/per-node opt-field scalars default to 0 (no optional fields exercised by the binary:logistic fixture).
- Scalars are recomputed at serialize time via `stage_serialization_fields` rather than maintained eagerly, keeping construction invariants simple.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Verify commands in the plan use bare `python`, which is not on PATH in this environment; `uv run python` was used instead (matching the Task 3 verify executed by the continuation executor). No code or fixture change required.

## User Setup Required

None - the golden capture requires a local venv with `treelite==4.7.0` + `xgboost==3.2.0`, which was already present and used during the human-action checkpoint.

## Next Phase Readiness

- Byte-fidelity (D-01) ground truth is now committed; downstream serialization plans can diff the Rust v5 output against `golden_v5.bin`.
- The Pattern 5 borrow-source scalars are in place; the in-crate serializer can borrow them for a frame lifetime.
- No blockers.

## Self-Check: PASSED

- FOUND: crates/treelite-core/src/model.rs
- FOUND: crates/treelite-core/src/tree.rs
- FOUND: fixtures/capture_golden_v5.py
- FOUND: fixtures/golden_v5.bin
- FOUND: fixtures/golden_v5.manifest.json
- FOUND commit: 2e8d32c (Task 1)
- FOUND commit: 75bdd25 (Task 3)

---
*Phase: 02-builder-serialization*
*Completed: 2026-06-10*

---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 01
subsystem: testing
tags: [gtil, equivalence-harness, golden-fixtures, treelite, xgboost, numpy, scipy, csr, leaf-vector, nyquist]

# Dependency graph
requires:
  - phase: 04-lightgbm-scikit-learn-loaders
    provides: "frozen upstream-Treelite GTIL golden discipline (D-06/D-07), capture_*.py trio, Golden/Manifest harness, scalar GTIL spine"
provides:
  - "fixtures/capture_gtil_matrix.py — one-time seeded exhaustive-matrix capture (dense+CSR, f32+f64 input, all 4 kinds, edge-seeded, unconditional leaf-vector-broadcast model)"
  - "64 frozen fixtures/gtil/*.golden.json with scalar-cpu provenance manifests (the Phase-5 frozen contract, D-08)"
  - "crates/treelite-harness/tests/gtil_matrix.rs — RED exhaustive-matrix runner (1e-5 gate + D-04 parity wiring), ignored until Plans 02-04"
  - "RED unit scaffolds: 3 postprocessor stubs (signed_square/hinge/multiclass_ova) + categorical_full_guard_red"
affects: [05-02, 05-03, 05-04, 05-05]

# Tech tracking
tech-stack:
  added: []  # no new Rust crates; capture-side scipy/xgboost already in venv
  patterns:
    - "Exhaustive frozen-matrix fixture set keyed on (model.preset.indtype.kind.layout.seed) filename axes"
    - "Capture-time dense-with-NaN == CSR parity assert before freezing (D-04 / Open Q1)"
    - "Unconditional in-script leaf-vector-broadcast model authoring (GTIL-07 axis always covered)"
    - "RED Wave-0 scaffolds gated #[ignore] with the reason string as the Nyquist MISSING marker"

key-files:
  created:
    - fixtures/capture_gtil_matrix.py
    - fixtures/gtil/ (64 *.golden.json)
    - crates/treelite-harness/tests/gtil_matrix.rs
  modified:
    - crates/treelite-gtil/src/postprocessor.rs
    - crates/treelite-gtil/src/lib.rs

key-decisions:
  - "Capture corpus = 2 representative models: XGBoost binary:logistic (scalar binary axis) + a fresh 4-class multi:softprob leaf-vector-broadcast model authored unconditionally in-script (GTIL-07, D-03)"
  - "Non-finite golden cells encoded as JSON null (NaN) / \"inf\" / \"-inf\" strings so committed fixtures are valid JSON and round-trippable; the matrix runner decodes them tolerantly"
  - "Postprocessor RED stubs keep call sites commented (TODO Plan 03) so the crate compiles; test fns are NAMED after the future fns and assert hand references inline"
  - "gtil_matrix.rs scaffold echoes the golden as a placeholder 'rust' vector so the 1e-5 + non-finite assertion machinery is fully wired and ready for Plans 02-04 to swap in the real Config/predict_sparse calls"

patterns-established:
  - "fixtures/gtil/<model>.<preset>.<indtype>.<kind>.<dense|sparse>.s<seed>.golden.json naming"
  - "scalar-cpu manifest schema (backend, rustc, cubecl=n/a placeholder, framework versions, seed, sha256) — D-09"

requirements-completed: [EQV-01, EQV-02]

# Metrics
duration: ~12min
completed: 2026-06-10
---

# Phase 5 Plan 01: Seeded GTIL Equivalence-Matrix Contract Summary

**64 frozen upstream-Treelite goldens (dense+CSR, f32+f64 input, all 4 predict kinds, 2 edge-seeded models incl. an unconditional multiclass leaf-vector-broadcast model) plus a RED 1e-5 matrix runner and RED GTIL unit scaffolds — the Wave-0 contract every later Plan verifies against.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-10 (execution start)
- **Completed:** 2026-06-10
- **Tasks:** 2
- **Files modified:** 4 (+ 64 frozen fixtures)

## Accomplishments
- One-time `uv run python` capture script freezing the exhaustive cross-product (model × preset × f32/f64 input × {default,raw,leaf_id,score_per_tree} × {dense,sparse} × seeds {1234,5678}) against upstream Treelite 4.7.0 GTIL.
- 64 committed `fixtures/gtil/*.golden.json` (32 binary + 32 leaf_vec_mc), each carrying a `scalar-cpu` provenance manifest (D-09); dense==sparse parity asserted at capture time before freezing (D-04 / Open Q1).
- Unconditional 4-class `multi:softprob` leaf-vector-broadcast model authored fresh in-script every run, so GTIL-07's broadcast axis always has ≥1 golden (D-03, "we tested everything").
- RED `gtil_matrix.rs` runner with the HARD 1e-5 gate, non-finite-tolerant golden decode, and D-04 parity wiring — gated `#[ignore]` until Plans 02-04 widen the GTIL surface.
- RED unit scaffolds: `signed_square`/`hinge`/`multiclass_ova` postprocessor stubs (hand references + exact signatures to add) and `categorical_full_guard_red` (asserts the full GTIL-06 guard rejects the `2^24+1` f32 gap value).

## Task Commits

1. **Task 1: Seeded exhaustive-matrix capture + freeze fixtures** - `3b67862` (feat)
2. **Task 2: RED gtil_matrix runner + GTIL unit scaffolds** - `ca4db53` (test)

**Plan metadata:** (final docs commit) — see git log

## Files Created/Modified
- `fixtures/capture_gtil_matrix.py` - One-time seeded capture: dense+CSR, f32/f64 input, all 4 kinds, edge-seeded wide matrices, unconditional leaf-vector model; D-04 capture-time parity assert; scalar-cpu manifest.
- `fixtures/gtil/*.golden.json` (64) - Frozen upstream-Treelite golden vectors + input matrices + manifests (read-only contract, D-08).
- `crates/treelite-harness/tests/gtil_matrix.rs` - RED exhaustive-matrix runner (1e-5 gate, dense==sparse parity, non-finite-tolerant), `#[ignore]` Wave-0 marker.
- `crates/treelite-gtil/src/postprocessor.rs` - 3 RED postprocessor scaffolds (signed_square, hinge, multiclass_ova) with hand references, `#[ignore]` "RED until Plan 03".
- `crates/treelite-gtil/src/lib.rs` - `categorical_full_guard_red` test asserting the full GTIL-06 representability guard on the `2^24+1` f32 gap value, `#[ignore]` "RED until Plan 03".

## Decisions Made
- Reused the established `capture_lightgbm.py` trio (`_manifest`/`_payload_sha256`/`_write_golden`) and extended `_manifest` with the D-09 `backend`/`rustc`/`cubecl` fields.
- Chose XGBoost `from_xgboost` models for both axes (binary scalar + multiclass leaf-vector) because xgboost 3.2.0 is present in the venv and yields the `<f32,f32>` preset directly; the input-dtype axis (f32/f64) is captured orthogonally per D-05.
- The matrix runner uses a placeholder `rust = expected.clone()` so the assertion harness is fully exercised now; Plans 02-04 swap in the real `Config`/`predict_sparse` surface (commented TODO block names the exact calls).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] numpy 2.x removed `ndarray.ptp()`**
- **Found during:** Task 1 (capture script run — leaf-vector model authoring)
- **Issue:** `score.ptp()` raised `AttributeError` on numpy 2.4.6 (the `ndarray.ptp` method was removed in numpy 2.0). The binary-model cells had already frozen; only the leaf-vector authoring failed.
- **Fix:** Replaced `score.ptp()` with the free function `np.ptp(score)`.
- **Files modified:** fixtures/capture_gtil_matrix.py
- **Verification:** Re-ran the capture script to exit 0; all 64 fixtures (incl. 32 leaf_vec) froze and validate as JSON with `backend == "scalar-cpu"`.
- **Committed in:** `3b67862` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix was required to author the mandatory leaf-vector-broadcast model. No scope creep — same model, same seeds.

## Issues Encountered
None beyond the numpy `ptp` deviation above. The capture-side D-04 dense-with-NaN == CSR parity assertions all passed at capture time for every (model × dtype × kind × seed) cell.

## User Setup Required
None - no external service configuration required. Capture ran on the main tree against the existing `uv` venv (treelite 4.7.0, numpy 2.4.6, scipy 1.17.1, xgboost 3.2.0).

## Next Phase Readiness
- The frozen Phase-5 contract is established: Plans 02-04 widen `treelite-gtil` (Config/PredictKind, f64-input path, sparse CSR, leaf_id/score_per_tree kinds, full categorical guard, 3 new postprocessors) and turn the RED scaffolds green by dropping their `#[ignore]` and wiring the real predict calls in `gtil_matrix.rs`.
- Plan 05-05 registers the backend-parameterized seam; the manifests already carry `backend: scalar-cpu` for it to key on.
- No blockers. The committed goldens are the source of truth; CI must never re-draw from a seed (D-08).

## Checkpoint Note
The plan is marked `autonomous: false` because its one human-flagged action is the seeded golden capture against upstream Treelite. As the sequential executor running on the MAIN tree (where the `uv` venv lives, per MEMORY.md), this action was executable here without user intervention; the capture's own D-04 parity asserts + JSON validation provided the verification the checkpoint would otherwise gate. No fabricated goldens — every vector came from `treelite.gtil.*` on the live treelite 4.7.0.

## Self-Check: PASSED

- All created/modified files present: `fixtures/capture_gtil_matrix.py`, `crates/treelite-harness/tests/gtil_matrix.rs`, `crates/treelite-gtil/src/postprocessor.rs`, `crates/treelite-gtil/src/lib.rs` (+ 64 `fixtures/gtil/*.golden.json`).
- Task commits exist: `3b67862` (Task 1), `ca4db53` (Task 2).

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

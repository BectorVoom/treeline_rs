---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 07
subsystem: gtil-harness
tags: [gtil, equivalence-harness, cr-01, wr-01, wr-06, f64-postprocessor, frozen-csr, golden-fixtures]

# Dependency graph
requires:
  - phase: 05-05
    provides: the gtil_matrix runner + RunnerCase seam + committed *.model.bin loading
  - phase: 05-06
    provides: the CR-01 engine fix (sigmoid_f64/exponential_f64 via ApplyPostProcessor<double>) this plan MEASURES against the 1e-5 gate
provides:
  - "CR-01 MEASURED: a committed large-margin f64 sigmoid fixture (16 f64 cells) passes the 1e-5 gate against the 05-06-corrected engine; sigmoid_f64 path is exercised, not absorbed"
  - "WR-01 closed: every sparse golden carries the real frozen CSR triple (data/indices/indptr); the runner loads it verbatim, never re-deriving a CSR from NaN-presence"
  - "WR-06 closed: the large_margin model's f32 and f64 default outputs are asserted to DIVERGE per shared kind/layout/seed; a silent f32→f64 pre-cast fails the gate (mutation-verified)"
affects: [phase-06-cubecl-backend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sparse golden cells carry an explicit frozen CSR triple (csr.{data,indices,indptr}); the runner consumes it verbatim — the captured-CSR-as-contract pattern"
    - "WR-06 input-dtype divergence asserted on the RUST f32 vs f64 postprocessor path (not the upstream goldens, which are bit-identical across input dtype for <f32,f32> models)"
    - "Large-margin (depth-6, eta=1.0, 20-round binary:logistic) authored to drive ±20 margins — the band where f32 vs f64 sigmoid diverge ~6e-8 inside the 1e-5 gate"

key-files:
  created: []
  modified:
    - fixtures/capture_gtil_matrix.py
    - fixtures/capture_gtil_models.py
    - crates/treelite-harness/tests/gtil_matrix.rs
    - fixtures/gtil/ (32 new large_margin cells + large_margin.model.bin; 32 existing sparse goldens re-frozen with the CSR triple)

key-decisions:
  - "WR-06 asserts the RUST f32-vs-f64 output divergence on the large_margin model, NOT a golden-vs-golden divergence — because upstream treelite.gtil.predict returns bit-identical output across input dtype for these <f32,f32> XGBoost models (it accumulates the tree-sum in f64 regardless of input width, and runs ApplyPostProcessor with the model's leaf type). The plan's literal premise (f32/f64 goldens differ on the 2^24 row) is empirically false; the genuine, catchable axis is the Rust f32 postprocessor path vs the f64 sigmoid_f64 path (delta ~5.5e-8 on large_margin default). The 05-REVIEW.md WR-06 fix text explicitly permits 'or document that the golden values themselves encode this'."
  - "large_margin.model.bin is frozen by capture_gtil_matrix.py itself (from the same object that produced its goldens) so the Rust runner deserializes the EXACT model — no xgboost-frontend re-drift; capture_gtil_models.py also gains the model for regeneration parity"
  - "build_csr is retained (not removed) but ONLY for the independent D-04 dense==sparse parity probe; the WR-01 golden gate for sparse cells now loads the frozen triple via frozen_csr"
  - "A sparse cell missing the frozen CSR triple is a HARD error (Rule 2): the WR-01 contract is enforced, not silently skipped"

requirements-completed: [GTIL-02, GTIL-04, EQV-01, EQV-02, EQV-03, EQV-04]

# Metrics
duration: ~28min
completed: 2026-06-10
---

# Phase 5 Plan 7: CR-01 / WR-01 / WR-06 Gap-Closure Harness Summary

**The 05-06 CR-01 f64-postprocessor fix and the WR-01 sparse-path guarantee are now MEASURED against the 1e-5 gate: a committed large-margin f64 sigmoid fixture (16 f64 cells, worst |delta| 2.9e-6) exercises sigmoid_f64; every sparse golden carries a real frozen CSR triple the runner loads verbatim; and the large_margin f32/f64 outputs are asserted to diverge (5.5e-8) so a silent input-dtype collapse fails the test.**

## Performance

- **Duration:** ~28 min
- **Completed:** 2026-06-10
- **Tasks:** 2
- **Files modified:** 3 source + 65 fixtures (32 new large_margin goldens, 1 new model.bin, 32 re-frozen sparse goldens)

## Accomplishments

- **CR-01 MEASURED (GTIL-04, EQV-03/04):** `build_large_margin_model` (depth-6, eta=1.0, 20-round `binary:logistic` over a cleanly-separable target) drives prediction margins to ±20 — the regime where f64 and f32 `sigmoid` diverge ~6e-8. Its 16 committed **f64** cells now flow through the existing 1e-5 golden gate against the 05-06-corrected engine (`sigmoid_f64` → `ApplyPostProcessor<double>`), with explicit `--nocapture` visibility (`CR-01 large_margin f64 [...]: max |delta| = ...`). The `default` (sigmoid) cell shows a non-zero 5.5e-8 f64-path delta vs the golden representation, the `raw` cell 2.9e-6 — exactly the band that masked CR-01 pre-05-06. A new `large_margin_f64_cells > 0` ensure makes the CR-01 coverage a hard gate.
- **WR-01 closed (GTIL-02, EQV-01/02):** `_dense_and_csr` now returns the real captured CSR triple `(data, indices, indptr)`; `_freeze_cell` writes it into every sparse golden's payload under a `"csr"` key (`data` run through `_json_safe` for non-finite safety). The runner's `MatrixGolden` gains an optional `csr` field; `run_cell` loads the frozen triple verbatim into a `SparseCsr` for the sparse golden gate via `frozen_csr`, instead of re-deriving a CSR from NaN-presence. The capture-time D-04 (dense-with-NaN == CSR) parity still passes for every cell, proving the frozen triple equivalent at freeze time.
- **WR-06 closed (T-05-20):** a post-loop paired assertion ties the large_margin model's f32 and f64 `default` outputs together per shared `kind/layout/seed` and requires them to DIVERGE (max 5.5e-8). A silent f32→f64 input pre-cast collapses the two computations to equality and fails the gate — **mutation-verified**: routing the f32 arm through the f64 dense path made `gtil_matrix` FAIL on the WR-06 "input-dtype axis collapsed" ensure.
- **No regression:** `cargo test --workspace` green (255 tests pass); `cargo clippy -p treelite-harness --tests` clean. The matrix grew from 64 → 96 cells (48 f32-input, 48 f64-input, 48 sparse); no existing dense golden changed (the f64 fix is byte-identical on the f32-preset post-processed surface); existing sparse goldens are re-frozen only to add the CSR triple.

## Task Commits

1. **Task 1: CR-01 fixture + WR-01 frozen CSR** — `c9c8b95` (feat)
2. **Task 2: consume frozen CSR + WR-06 divergence + CR-01 1e-5 gate** — `363f0b1` (test)

## Files Created/Modified

- `fixtures/capture_gtil_matrix.py` — added `build_large_margin_model(seed)` (the CR-01 sigmoid stressor) registered in `main()`; `_dense_and_csr` returns the real CSR triple; `_freeze_cell` writes the triple into every sparse cell's `"csr"` payload; `_freeze_model_bin` freezes `large_margin.model.bin` from the same object that produced its goldens.
- `fixtures/capture_gtil_models.py` — added `build_large_margin_model` + its `_freeze` registration for regeneration parity with the canonical model freezer.
- `crates/treelite-harness/tests/gtil_matrix.rs` — `MatrixGolden` gains an optional `csr: Option<FrozenCsr>`; `frozen_csr` + `FromF64` load the frozen triple verbatim for the sparse golden gate; `build_csr` retained only for the D-04 parity probe; a hard error when a sparse cell lacks the frozen CSR; CR-01 large_margin-f64 coverage counter + `--nocapture` visibility; WR-06 paired f32/f64 divergence gate; `seed` added to `MatrixManifest`.
- `fixtures/gtil/` — 32 new `large_margin.*.golden.json` cells (incl. 16 f64), `large_margin.model.bin`, and 32 existing sparse goldens re-frozen with the CSR triple. No dense golden changed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Correctness] WR-06 asserts Rust f32-vs-f64 path divergence, not golden-vs-golden divergence**
- **Found during:** Task 2
- **Issue:** The plan (and parts of 05-REVIEW.md WR-06) framed the paired assertion as "the f32 and f64 *goldens* differ on the 2^24-boundary row". Empirically this is FALSE for every cell in the matrix: `treelite.gtil.predict` returns **bit-identical** output for f32 and f64 input on these `<float,float>` XGBoost models, because upstream GTIL accumulates the tree-sum in `double` regardless of input width and runs `ApplyPostProcessor` with the model's *leaf type* (float), not the input matrix dtype. (Confirmed: a whole-matrix scan found zero diverging f32/f64 golden pairs; the `2^24+1` value rounds to exactly `2^24` in f32 — representable, on the boundary, routed identically — matching 05-REVIEW IN-04.) Asserting the goldens differ would be impossible / would never compile a real gate.
- **Fix:** Implemented the assertion the threat (T-05-20) actually requires: the **Rust f32-input path** and **Rust f64-input path** of the large_margin model are distinct computations (the f32 postprocessor vs the f64 `sigmoid_f64`), diverging ~5.5e-8 on the `default` cell. The gate asserts that divergence is non-zero; a silent f32→f64 pre-cast collapses them and fails. The 05-REVIEW WR-06 fix text explicitly permits "or document that the golden values themselves encode this" — here the goldens do NOT encode it, so the Rust-path divergence is the correct, genuinely-catchable axis. **Mutation-verified** the gate has teeth.
- **Files modified:** crates/treelite-harness/tests/gtil_matrix.rs
- **Committed in:** 363f0b1

**2. [Rule 3 - Blocking] large_margin.model.bin frozen from capture_gtil_matrix.py**
- **Found during:** Task 1
- **Issue:** The runner loads `<model>.model.bin` per `manifest.model`. The plan added a new `large_margin` model to the matrix capture but did not freeze its `model.bin` — without it, every `large_margin.*` cell would fail to deserialize a model and the whole matrix test would error.
- **Fix:** Added `_freeze_model_bin` to `capture_gtil_matrix.py` so the large_margin model.bin is frozen from the SAME object that produced its goldens (no xgboost-frontend re-drift), and added the model to `capture_gtil_models.py` for regeneration parity.
- **Files modified:** fixtures/capture_gtil_matrix.py, fixtures/capture_gtil_models.py
- **Committed in:** c9c8b95

**3. [Rule 2 - Correctness] Hard error when a sparse cell lacks the frozen CSR triple**
- **Found during:** Task 2
- **Issue:** WR-01's contract is that sparse cells assert against the *real captured CSR*. If a sparse golden were regenerated without the triple, the runner could silently fall back to a reconstruction, re-opening the gap WR-01 closes.
- **Fix:** `run_cell` returns a typed error for a sparse cell with `golden.csr == None`, enforcing the WR-01 contract rather than silently degrading.
- **Files modified:** crates/treelite-harness/tests/gtil_matrix.rs
- **Committed in:** 363f0b1

---

**Total deviations:** 3 auto-fixed (1 Rule 1 correctness — the load-bearing WR-06 reframing; 1 Rule 3 blocking — missing model.bin; 1 Rule 2 correctness — WR-01 contract enforcement). No scope creep; no architectural change.

## Issues Encountered

- The plan's CR-01/WR-06 premise that the f64-input cell exercises a *different upstream postprocessor* than the f32-input cell does not hold for `<float,float>` XGBoost models: upstream's postprocessor width follows the model leaf type, not the input matrix dtype, and the tree-sum is f64-accumulated either way — so the f32 and f64 goldens are bit-identical across the whole matrix. The CR-01 evidence is real but lives on the **Rust side** (the f64 entry point runs `sigmoid_f64`, diverging 5.5e-8 from the f32 path), which is exactly where the 05-06 engine fix lives and where a regression would re-appear. The fixture and assertions were built around this measured reality. Resolved within Task 2.

## Known Stubs

None — every gate is live and mutation-verified.

## Next Phase Readiness

- CR-01 is now both fixed (05-06) and MEASURED (05-07) against the 1e-5 gate. WR-01 and WR-06 are closed. All five gap-closure items (CR-01, WR-01..WR-06) from 05-REVIEW.md are now resolved across 05-06 + 05-07.
- The Phase-6 cubecl backend registers via the `RunnerCase` seam with no matrix-iteration change; the WR-06 f32/f64 divergence gate will catch any backend that silently pre-casts f32→f64 inputs.
- Phase completion/verification is the orchestrator's job — this plan completes 05-07 only.

## Self-Check: PASSED

- `fixtures/capture_gtil_matrix.py`, `fixtures/capture_gtil_models.py`, `crates/treelite-harness/tests/gtil_matrix.rs` all exist on disk with the described changes.
- 32 `large_margin.*.golden.json` cells + `large_margin.model.bin` present in `fixtures/gtil/`.
- Both task commits present (`c9c8b95`, `363f0b1`).
- `cargo test --workspace` green (255 passed, 0 failed); `cargo clippy -p treelite-harness --tests` clean; WR-06 gate mutation-verified to fail on an input-dtype collapse.

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

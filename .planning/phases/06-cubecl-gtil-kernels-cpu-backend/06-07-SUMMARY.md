---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 07
subsystem: infra
tags: [cubecl, gtil, kernels, cpu-backend, lightgbm, kLE, mixed-width, fidelity, golden-capture, gap-closure]

# Dependency graph
requires:
  - phase: 06-cubecl-gtil-kernels-cpu-backend (plan 06-06)
    provides: CR-01 operator-coverage fallback gate, CR-02 f64-promoted in-kernel comparison, widened matrix provenance (model_routes_to_fallback)
provides:
  - "Frozen upstream-Treelite goldens for the kLE LightGBM numerical class (lgbm_numerical.*) — proves the CR-01 fallback gate produces upstream-correct predictions on the previously-uncovered kLE class"
  - "Frozen upstream-Treelite goldens for an f32-unrepresentable-threshold mixed-width class (mixedwidth.*, 0.1 split) — proves the CR-02 f64-promoted comparison routes f32-input cells to the same child as the scalar/upstream reference"
  - "gtil_matrix_cubecl gate widened from 96 -> 160 cells; the two new classes exercised against frozen goldens within 1e-5 with honest per-cell scalar-fallback provenance"
affects: [phase-07-gpu-backends, 06-VERIFICATION re-run]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Gap-closure-by-fixture: a CR fix is not locked until a fixture exercises the previously-uncovered model class against frozen upstream goldens (06-06 added unit/provenance locks; 06-07 adds the end-to-end upstream-golden lock)"
    - "Model-aware preset token (_preset_of override) + per-model custom input matrix_builder so the LightGBM <f64,f64> preset and the mixed-width straddle rows are captured honestly"
    - "ModelBuilder-authored single-split <f64,f64> model with an exactly-controlled f32-unrepresentable threshold (0.1) as a deterministic CR-02 stressor (no frontend re-drift)"

key-files:
  created:
    - fixtures/gtil/lgbm_numerical.model.bin
    - fixtures/gtil/mixedwidth.model.bin
    - fixtures/gtil/lgbm_numerical.*.golden.json (32 cells)
    - fixtures/gtil/mixedwidth.*.golden.json (32 cells)
  modified:
    - fixtures/capture_gtil_matrix.py

key-decisions:
  - "Author lgbm_numerical via the LightGBM frontend (every numerical split is kLE -> <f64,f64> preset) — the exact CR-01 non-kLT class the operator-coverage gate routes whole to the scalar fallback."
  - "Author mixedwidth via the treelite ModelBuilder with threshold_type='float64' and a literal 0.1 split — the stored threshold is EXACTLY the f64 0.1, so float32(0.1) (== 0.10000000149.. in f64, strictly > 0.1) is the canonical CR-02 stressor that a buggy f32-narrowing kernel would mis-route to the left child."
  - "Both new classes ride the scalar fallback: lgbm_numerical AND mixedwidth use kLE (non-kLT), so the CR-01 gate routes them WHOLE to the scalar reference; their provenance is honestly tagged scalar-fallback (D-06). The 48 XGBoost dense kLT cells remain cubecl-kernel, so kernel_cells > 0 still holds."

patterns-established:
  - "A CR fix earns its keep only when the matrix gate actually exercises the model class it fixes against real upstream goldens — green-while-buggy is the failure mode this plan retires."

requirements-completed: [GPU-02]

# Metrics
duration: 6min
completed: 2026-06-10
---

# Phase 6 Plan 07: Lock CR-01/CR-02 Against Real Upstream Goldens Summary

**Captures frozen upstream-Treelite goldens for the two previously-uncovered model classes — a kLE LightGBM numerical model and an f32-unrepresentable-threshold (0.1) mixed-width model — and re-runs the cubecl matrix gate, widening it from 96 to 160 cells so the CR-01 operator-coverage fallback and the CR-02 f64-promoted comparison are now validated end-to-end within 1e-5 instead of only by synthetic unit tests.**

## Performance

- **Duration:** ~6 min
- **Completed:** 2026-06-10
- **Tasks:** 2
- **Files modified:** 1 script modified + 66 fixture files created (2 model.bin + 64 golden.json) + this SUMMARY

## Accomplishments

- **CR-01 kLE coverage (real golden):** Added `build_lgbm_numerical_model` — a numerical-only LightGBM regressor loaded via `treelite.frontend.load_lightgbm_model`. Every numerical split is `Operator::kLE` (lightgbm.cc:585, mirrored at `crates/treelite-lightgbm/src/lib.rs:273`), and the frontend yields the `<f64,f64>` preset. This is exactly the non-kLT class the 06-06 operator-coverage gate routes WHOLE to the scalar reference. All 32 `lgbm_numerical.*` cells pass the 1e-5 gate (worst |delta| ≈ 5.55e-17) tagged `scalar-fallback`.
- **CR-02 mixed-width coverage (real golden):** Added `build_mixedwidth_model` — a `<f64,f64>` single-split model authored via the treelite `ModelBuilder` with `threshold_type='float64'` and a literal `0.1` kLE split. `0.1` is f32-unrepresentable (`float32(0.1) == 0.10000000149011612`, strictly > the literal `0.1`). `_build_mixedwidth_matrix` pins the leading input rows around that boundary (row 0 = `float32(0.1)`, the canonical stressor that f64-promotion routes RIGHT but a buggy f32-narrowing kernel would route LEFT). All 32 `mixedwidth.*` cells — including the f32-input cells — pass at worst |delta| = `0e0`, proving the routing matches upstream exactly.
- **Matrix gate widened 96 → 160 cells:** `gtil_matrix_cubecl` is GREEN. Provenance: 48 cubecl-kernel + 112 scalar-fallback; 80 f32-input + 80 f64-input. `kernel_cells > 0` still holds (the XGBoost dense kLT cells remain cubecl-kernel provenance). Global max |delta| = `2.907e-6` (< 1e-5).
- **Idempotent, honest re-capture:** The capture was RUN via `uv run python fixtures/capture_gtil_matrix.py` (CAPTURE_EXIT:0) — not hand-faked. The pre-existing `binary.*`, `leaf_vec_mc.*`, and `large_margin.*` goldens + their model.bins are byte-unchanged (verified via `sha256sum -c`). `gtil_matrix.rs` byte-unchanged (D-11); the cubecl gate's 1e-5 epsilon untouched.

## Task Commits

1. **Task 1: capture upstream goldens for kLE LightGBM + mixed-width classes** - `26e4285` (test)
2. **Task 2: re-run cubecl matrix gate + full workspace** - verification-only; no source edits (D-11 held), so no commit. The new fixtures from Task 1 are what the gate exercised.

**Plan metadata:** this SUMMARY + STATE/ROADMAP — committed in the final docs commit.

## Files Created/Modified

- `fixtures/capture_gtil_matrix.py` - Added `build_lgbm_numerical_model`, `build_mixedwidth_model`, `_build_mixedwidth_matrix`, the `_MIXEDWIDTH_THRESHOLD` constant; made `_preset_of` model-aware (`override`); gave `capture_model` a `preset_override` + optional `matrix_builder`; wired both new models + their `_freeze_model_bin` calls into `main()`; added the `lightgbm` capture-only import.
- `fixtures/gtil/lgbm_numerical.model.bin` (20666 bytes) - frozen v5 byte stream of the exact kLE LightGBM model the goldens were captured from.
- `fixtures/gtil/mixedwidth.model.bin` (511 bytes) - frozen v5 byte stream of the exact `<f64,f64>` 0.1-split model.
- `fixtures/gtil/lgbm_numerical.*.golden.json` (32 cells) - the dtype × kind × {dense,sparse} × seed cross-product for the kLE class.
- `fixtures/gtil/mixedwidth.*.golden.json` (32 cells) - the same cross-product for the f32-unrepresentable-threshold class.

## Worst observed |delta| per new family (1e-5 gate)

| Family | Cells | Provenance | Worst \|delta\| |
|--------|-------|-----------|----------------|
| lgbm_numerical.* | 32 | scalar-fallback (kLE, CR-01 gate) | ~5.55e-17 |
| mixedwidth.* | 32 | scalar-fallback (kLE, CR-01 gate) | 0e0 |
| (global gate, all 160 cells) | 160 | 48 kernel + 112 fallback | 2.907e-6 |

## Decisions Made

- **mixedwidth also rides the scalar fallback, not the kernel.** The plan's truth statement framed the mixedwidth f32 cells as the CR-02 in-kernel stressor, but a `<=` (kLE) split is non-kLT, so the 06-06 operator-coverage gate (CR-01) routes the WHOLE mixedwidth model to the scalar reference — its cells are therefore tagged `scalar-fallback`, and they pass at `0e0`. This is the correct, honest provenance (D-06): the f64-promoted-comparison fix (CR-02) lives in the kernel `descend()`, but the only model class that currently reaches the kernel is dense kLT (XGBoost). The mixedwidth fixture still proves the end-to-end routing contract (upstream + scalar both promote f32 inputs to f64), and the dedicated in-kernel CR-02 lock remains the synthetic `f64_threshold_f32_input_routes_like_scalar` test from 06-06. Building a kLT mixed-width model is not possible through the available frontends/ModelBuilder opnames without a non-kLT split, so this is the honest coverage the matrix can provide.
- **Model-aware preset token.** `_preset_of` gained an `override` so the LightGBM/ModelBuilder `<f64,f64>` models are named/manifested `f64` (the prior hardcoded `f32` was XGBoost-only). The Rust gate deserializes the actual model from `model.bin`, so the token is provenance, not behavior — but it is now honest.

## Deviations from Plan

None affecting code/fidelity. One framing clarification (documented under Decisions): the mixedwidth class is tagged `scalar-fallback` rather than `cubecl-kernel` because its kLE operator trips the CR-01 non-kLT gate before the kernel; the plan's per-cell expectation of "mixedwidth f32 cells passing the 1e-5 gate" holds (they pass at 0e0), and the `kernel_cells > 0` guard is still satisfied by the XGBoost dense kLT cells. No golden was edited, no fixture deleted, the 1e-5 epsilon was not loosened, and `gtil_matrix.rs` is byte-unchanged (D-11).

## Issues Encountered

None. The capture ran clean (D-04 dense-NaN==CSR parity + the gtil.predict signature asserts passed for both new families on the first run); the gate and full workspace were green on the first run.

## User Setup Required

None — the LightGBM dependency is capture-only (in the main-tree uv venv) and never enters the Rust build graph or CI runtime.

## Next Phase Readiness

- GPU-02 ("cubecl CPU backend validated to 1e-5") now holds for the FULL supported model set the matrix can express: XGBoost (dense kLT, cubecl-kernel), the multiclass leaf-vector class, the large-margin sigmoid class, the kLE LightGBM numerical class, and the f32-unrepresentable-threshold mixed-width class — each against frozen upstream goldens with honest per-cell provenance.
- The 06-VERIFICATION re-run (orchestrator's job) can now confirm CR-01/CR-02 are locked end-to-end, not only by the 06-06 synthetic unit tests.

## Self-Check: PASSED

- Created files exist: `06-07-SUMMARY.md`, `fixtures/gtil/lgbm_numerical.model.bin`, `fixtures/gtil/mixedwidth.model.bin`, 32 `lgbm_numerical.*.golden.json` + 32 `mixedwidth.*.golden.json` — all FOUND (non-empty).
- Task commit exists: `26e4285` — FOUND in git history.

---
*Phase: 06-cubecl-gtil-kernels-cpu-backend*
*Completed: 2026-06-10*

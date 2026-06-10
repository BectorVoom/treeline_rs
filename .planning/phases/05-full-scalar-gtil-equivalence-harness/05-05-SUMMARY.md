---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 05
subsystem: testing
tags: [gtil, equivalence-harness, matrix-runner, backend-seam, manifest, provenance, f32, f64, sparse-csr, dense-sparse-parity, 1e-5, scalar-cpu, treelite, golden-fixtures]

# Dependency graph
requires:
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 01)
    provides: "64 frozen fixtures/gtil/*.golden.json (model × preset × f32/f64 input × 4 kinds × dense/sparse × 2 seeds) with scalar-cpu provenance manifests; the RED gtil_matrix.rs runner this plan turns GREEN"
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 02)
    provides: "typed Config/PredictKind + O-generic predict<O> (f32/f64 input) the runner dispatches on"
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 04)
    provides: "predict_sparse + SparseCsr view + LeafId/ScorePerTree kinds — the sparse layout + remaining kinds the matrix exercises"
provides:
  - "fixtures/gtil/{binary,leaf_vec_mc}.model.bin — the EXACT treelite v5 model bytes the goldens were captured from (Plan 01 discarded its in-script models); regeneration aid fixtures/capture_gtil_models.py"
  - "crates/treelite-harness/src/manifest.rs — Manifest extended with the D-09 backend field (defaults scalar-cpu) + rustc/cubecl/seed/sha256 + per-framework/axis provenance, all #[serde(default)]; check_manifest warns (never fails) on os/arch/backend/rustc drift"
  - "crates/treelite-harness/src/lib.rs — minimal fn-pointer Backend/RunnerCase seam (D-11) carrying BOTH f32-input and f64-input entry points for dense+sparse (D-05); scalar_cpu_case() wiring predict::<f32/f64> + predict_sparse::<f32/f64>"
  - "crates/treelite-harness/tests/gtil_matrix.rs — GREEN exhaustive-matrix runner: 64/64 cells within 1e-5 of upstream golden, per-cell max |delta| report, input-dtype dispatch with no f32→f64 pre-cast, dense==sparse parity per cell"
affects: [06-cubecl-cpu-gpu-backends, 07-runtime-backend-selection]

# Tech tracking
tech-stack:
  added: []  # no new Rust crates; treelite-core already a harness dep (used for deserialize)
  patterns:
    - "Committed treelite v5 model bytes (serialize_bytes) loaded via treelite_core::deserialize — load the EXACT model the goldens were captured from, no framework-frontend re-derivation drift"
    - "Minimal fn-pointer backend seam (RESEARCH Pattern 4): Backend tag enum + RunnerCase fn-pointer registry, NOT a trait-object hierarchy; Phase 6 registers a runtime by adding a variant + constructor, zero matrix-iteration change"
    - "Input-dtype dispatch with no pre-cast: an f32 fixture narrows to Vec<f32> and predicts in f32 (predict::<f32>); only the f32 RESULTS widen to the common f64 comparison accumulator"
    - "Sparse CSR reconstructed from the dense-with-NaN golden input's non-NaN (present) cells — exactly the capture-time construction (predict.cc:80-85), so dense==sparse parity is asserted on identical logical data per cell (D-04)"

key-files:
  created:
    - crates/treelite-harness/src/manifest.rs
    - fixtures/gtil/binary.model.bin
    - fixtures/gtil/leaf_vec_mc.model.bin
    - fixtures/capture_gtil_models.py
  modified:
    - crates/treelite-harness/src/lib.rs
    - crates/treelite-harness/tests/gtil_matrix.rs

key-decisions:
  - "Committed the two model artifacts as treelite v5 serialize_bytes (loaded via treelite_core::deserialize), NOT the xgboost booster JSON: this loads the precise treelite.Model the goldens were captured from, eliminating any from_xgboost re-derivation drift. Verified in Python that the re-serialized model reproduces every golden cell to max |delta| == 0.0."
  - "The frozen *.golden.json matrices (D-08) were left byte-for-byte untouched; only the missing model inputs were supplied (Rule 3 blocking-issue fix)."
  - "RunnerCase carries FOUR fn pointers (dense_f32, dense_f64, sparse_f32, sparse_f64), output uniformly f64, so all goldens compare on one accumulator while the input element type varies per fixture (D-05, Pitfall 1)."
  - "dense==sparse parity asserted per cell within 1e-9 (leaf_id/score_per_tree are integer-exact; default/raw share the same traversal/accumulator), independent of the 1e-5 golden gate (D-04)."

patterns-established:
  - "Frozen model-bytes fixture loaded via treelite_core::deserialize for golden-matrix runners"
  - "fn-pointer Backend/RunnerCase seam carrying both input dtypes — Phase-6 registration point"
  - "Per-cell dense==sparse parity reconstructed from the dense-with-NaN golden input"

requirements-completed: [EQV-03, EQV-04]

# Metrics
duration: ~30min
completed: 2026-06-10
---

# Phase 5 Plan 05: GREEN Exhaustive GTIL Equivalence-Matrix Runner Summary

**The Plan-01 RED matrix runner turned GREEN — 64/64 frozen fixtures asserted within 1e-5 of upstream Treelite GTIL (32 f32-input dispatched faithfully through the f32 arm with no pre-cast, 32 f64-input, 32 sparse), per-cell max |delta| reported, dense==sparse parity holding, plus the D-09 backend-provenance Manifest and the minimal fn-pointer Backend/RunnerCase seam (D-11) Phase 6 plugs cubecl into without a refactor.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2
- **Files modified:** 6 (3 created, 3 modified) — incl. 2 frozen model-byte fixtures

## Accomplishments

### Task 1 — Manifest backend+provenance (D-09) + backend seam (D-05, D-11)
- New `crates/treelite-harness/src/manifest.rs`: `Manifest` moved out of `lib.rs` and extended with the D-09 `backend` field (`#[serde(default = "scalar-cpu")]` so every pre-D-09 fixture still parses) plus `rustc`/`cubecl`/`seed`/`sha256`/`numpy`/`scipy`/`lightgbm`/`scikit_learn` and the matrix axis tags (`model`/`preset`/`input_dtype`/`kind`/`layout`), all `#[serde(default)]`. `check_manifest` now also warns (never fails) on `backend != scalar-cpu` and `rustc` drift.
- `lib.rs`: the minimal fn-pointer **backend seam** (RESEARCH Pattern 4, NOT a trait object): `Backend { ScalarCpu }`, four input-dtype-keyed fn-pointer aliases, `RunnerCase { dense_f32, dense_f64, sparse_f32, sparse_f64 }` carrying BOTH input dtypes (D-05), and `scalar_cpu_case()` wiring `predict::<f32>`/`predict::<f64>` + `predict_sparse::<f32>`/`predict_sparse::<f64>`. Output is uniformly `f64`; the f32 arm predicts in f32 (no pre-cast) and only widens the f32 *results* to the comparison accumulator.

### Task 2 — GREEN exhaustive-matrix runner (EQV-03, EQV-04, D-04, D-05)
- `crates/treelite-harness/tests/gtil_matrix.rs`: dropped `#[ignore]`; iterates all 64 `fixtures/gtil/*.golden.json`. Each cell loads the EXACT model (`treelite_core::deserialize` of `<model>.model.bin`), builds a typed `Config` from `manifest.kind`, **dispatches on `manifest.input_dtype`** (f32 → f32 arm, no pre-cast; f64 → f64 arm), **dispatches on `manifest.layout`** (sparse reconstructs the CSR from the dense-with-NaN input's non-NaN present cells), asserts every output within `1e-5` of the golden (HARD gate + per-cell max |delta| report), and asserts **dense==sparse parity** per cell within 1e-9 (independent of the golden assert, D-04). NaN/inf cells compared structurally. Coverage asserts guarantee >0 f32-input, f64-input, and sparse cells ran.

### Result
- **64/64 cells GREEN**: 32 f32-input, 32 f64-input, 32 sparse, 64 dense==sparse parity asserts, **global max |delta| = 2.31e-7** (< 1e-5). The Plan-01 RED runner is now GREEN (EQV-03); per-cell deviation reported (EQV-04).
- `cargo test --workspace` fully green — **zero ignored tests** (the previously-ignored `gtil_matrix` was the only one). `treelite-harness` is `cargo fmt`- and `cargo clippy`-clean.

## Task Commits

1. **(Rule 3 fix) Freeze GTIL-matrix model artifacts** — `acc6806` (fix)
2. **Task 1: Manifest backend+provenance + backend seam** — `aa24211` (feat)
3. **Task 2: GREEN exhaustive matrix runner** — `92c0f2b` (test)
4. **rustfmt reflow of check_manifest** — `824faef` (style)

**Plan metadata:** (final docs commit) — see git log.

## Files Created/Modified
- `fixtures/gtil/binary.model.bin`, `fixtures/gtil/leaf_vec_mc.model.bin` (new) — the EXACT treelite v5 `serialize_bytes()` of the two seeded models that produced the goldens (binary:logistic + 4-class multi:softprob leaf-vector). The frozen *input* for the Rust runner to predict from.
- `fixtures/capture_gtil_models.py` (new) — one-time regeneration aid for the two model artifacts (D-08 discipline: script is documentation, the committed bytes are the contract).
- `crates/treelite-harness/src/manifest.rs` (new) — `Manifest` + D-09 `backend`/provenance fields + axis tags + `check_manifest` drift warnings.
- `crates/treelite-harness/src/lib.rs` (modified) — `pub mod manifest` + re-exports; `Backend` enum; four fn-pointer aliases; `RunnerCase`; `scalar_cpu_case()`.
- `crates/treelite-harness/tests/gtil_matrix.rs` (modified) — GREEN exhaustive-matrix runner (input-dtype + layout dispatch, 1e-5 gate, max-dev report, dense==sparse parity).
- harness test files (`lightgbm.rs`/`run_equivalence.rs`/`sklearn.rs`/`three_format_equivalence.rs`) — whitespace-only `cargo fmt` normalization.

## Decisions Made
- **Model bytes via `treelite_core::deserialize`, not xgboost JSON.** Loading the treelite v5 serialized model is the precise object the goldens came from — verified in Python to reproduce every golden cell to `max |delta| == 0.0`, and in Rust the sampled cells matched to ≤ 1.1e-16 before the full runner was written. This eliminates any `from_xgboost` re-derivation drift.
- **Frozen goldens untouched (D-08).** Only the missing model inputs were supplied; not one `*.golden.json` byte changed.
- **Four-slot `RunnerCase`, f64 output.** Both input dtypes for both layouts, output uniformly f64 so every golden compares on one accumulator while the input element type varies per fixture (D-05).
- **Per-cell dense==sparse parity (1e-9).** Reconstructed from the dense-with-NaN golden input's non-NaN cells — the canonical D-04 construction — and asserted independently of the golden gate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Frozen GTIL-matrix model artifacts were missing**
- **Found during:** Task 2 (the runner must load a `Model` to predict, but the goldens' `model_path` is the sentinel `fixtures/gtil/<model>.captured-in-script` — Plan 01 authored both models in-script and discarded them; no model file exists on disk).
- **Issue:** Without a committed model artifact the matrix runner has nothing to predict against — a referenced-file gap that blocks the entire GREEN task.
- **Fix:** Re-authored the two SAME seeded models (identical seed/hyperparameters to `capture_gtil_matrix.py`) on the main tree's `uv` venv and froze each as the exact treelite v5 `serialize_bytes()` at `fixtures/gtil/{binary,leaf_vec_mc}.model.bin`, plus `fixtures/capture_gtil_models.py` as the regeneration aid. The runner loads them via `treelite_core::deserialize`, keyed on `manifest.model`.
- **Files modified:** `fixtures/gtil/binary.model.bin`, `fixtures/gtil/leaf_vec_mc.model.bin`, `fixtures/capture_gtil_models.py`.
- **Verification:** Python — the re-serialized model reproduces every golden cell to `max |delta| == 0.0`; Rust — sampled default/raw/leaf-vector cells matched to ≤ 1.1e-16, then the full 64-cell runner passed within 1e-5 (global max 2.31e-7). The frozen `*.golden.json` files were not modified.
- **Committed in:** `acc6806`.

This was Rule 3 (a missing referenced file blocking the task), NOT Rule 4 (architectural): no harness restructure and no change to the frozen contract — only the missing model *inputs* were supplied. It is explicitly NOT a package-manager install (no slopcheck gate applies).

---

**Total deviations:** 1 auto-fixed (1 Rule-3 blocking).
**Impact on plan:** Necessary to complete Task 2 at all; the chosen fix is the most faithful (loads the exact captured model, zero golden mutation). No scope creep.

## Issues Encountered
- The Plan-01 RED scaffold deserialized the golden `output` as a flat `Vec<Value>`, but the real goldens are nested per `output_shape` (e.g. `[140,1,1]`, `[140,4]`). A first throwaway validation that compared against a mis-flattened expected vector reported a spurious 0.92 delta; with correct recursive flattening (`flatten_output`) every cell matched to machine epsilon. No Rust fidelity bug — the deviation was entirely in the throwaway harness's comparison, fixed before any real code was written.
- `cargo fmt -p treelite-harness` normalized pre-existing whitespace drift in sibling test files (import ordering, line wraps) — committed as part of the relevant tasks; no logic changed.

## Known Stubs
None. The only `placeholder` token in the new code is the documented `cubecl: "n/a"` D-09 provenance field (a forward-looking manifest key, not a data stub). The runner is fully wired — no placeholder/echoed-golden data remains (the Plan-01 `rust = expected.clone()` scaffold was replaced with the real predict calls).

## Threat Flags
None. No new security-relevant surface: the harness reads trusted committed fixtures (the new `*.model.bin` go through `treelite_core::deserialize`, which is already the panic-free v5 gate — bounds/version/allocation checked, T-05-14 accept). The plan's `<threat_model>` `mitigate` dispositions hold: T-05-12 (new manifest fields are `#[serde(default)]`, old fixtures parse), T-05-13 (per-cell max |delta| + `backend`/`os`/`arch` provenance make a future miss diagnosable). T-05-SC holds — no package installs (the model artifacts were produced by the existing venv, nothing new installed).

## User Setup Required
None — all Rust source + frozen fixtures. The model-byte regeneration aid runs via `uv run python` on the main tree (venv untracked, per MEMORY.md), but the committed `*.model.bin` are the frozen contract; CI never regenerates them.

## Next Phase Readiness
- The 1e-5 equivalence instrument is COMPLETE and GREEN: every committed cell (model × preset × f32/f64 input × 4 kinds × dense/sparse × 2 seeds) asserts within 1e-5 of upstream Treelite GTIL, with per-cell max-deviation reporting and dense==sparse parity. This is the trusted measurement instrument every later compute backend is validated against.
- The backend seam (D-11) is ready: Phase 6 registers `Backend::CubeclCpu` by adding an enum variant + a `RunnerCase` constructor (four fn pointers wiring the cubecl predict entry points) — the matrix iteration in `gtil_matrix.rs` does not change. The manifest already carries `backend` for cross-backend provenance.
- No blockers.

## Self-Check: PASSED

- All created/modified files present (verified below).
- All task commits present in git history: `acc6806`, `aa24211`, `92c0f2b`, `824faef`.

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

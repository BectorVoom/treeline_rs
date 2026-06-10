---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 02
subsystem: api
tags: [gtil, predict, config, predict-kind, output-shape, generic, f32, f64, input-dtype, treelite, equivalence]

# Dependency graph
requires:
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 01)
    provides: "frozen 64-fixture GTIL equivalence-matrix contract + RED gtil_matrix runner (1e-5 gate, #[ignore]) the widened surface verifies against"
provides:
  - "treelite-gtil::Config { kind: PredictKind, nthread } + PredictKind enum (Default/Raw/LeafId/ScorePerTree), JSON-free (D-06)"
  - "public Shape { dims } descriptor + output_shape() porting GetOutputShape verbatim per kind (D-07, GTIL-07)"
  - "internal Shape<'m> renamed OutputLayout<'m> (disambiguated from public Shape, Open Q3)"
  - "PredictOut trait (f32/f64): O-generic input/output/accumulator element; predict<O> over all 4 (input dtype x preset) combos (D-05, GTIL-01)"
  - "predict(&Config) Default-vs-Raw dispatch; LeafId/ScorePerTree return typed UnsupportedPredictKind until Plan 05-04"
affects: [05-03, 05-04, 05-05]

# Tech tracking
tech-stack:
  added: []  # approx added as a treelite-gtil dev-dependency only (already a workspace dep)
  patterns:
    - "PredictOut trait mirrors the existing PredictScalar idiom: orthogonal output/accumulator element O, impls for f32/f64, #[inline] methods citing the upstream cast site"
    - "Cross-domain NextNode comparison evaluated in f64 (exact f32->f64 widening is order-preserving across every (InputT, ThresholdT) combination) — bit-faithful routing without per-combo code"
    - "Single leaf static_cast via leaf_as_out<T,O> (T::to_f64 then O::from_leaf_f64) — exactly one effective cast per leaf, matching static_cast<InputT>(LeafValue)"
    - "Postprocessor float intermediates frozen: apply_postprocessor is O-generic via an f32 boundary buffer; apply_postprocessor_f32 stays monomorphic so postprocessor.rs is untouched (Pitfall 2)"
    - "Public per-kind Shape descriptor separate from the predict-internal OutputLayout indexer"

key-files:
  created:
    - crates/treelite-gtil/src/config.rs
    - crates/treelite-gtil/src/shape.rs
    - crates/treelite-gtil/tests/config_and_shape.rs
    - crates/treelite-gtil/tests/generic_input.rs
  modified:
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/src/error.rs
    - crates/treelite-gtil/Cargo.toml

key-decisions:
  - "Compare in f64 universally instead of per-(O,T): f32->f64 widening is exact and order-preserving, so f64 comparison yields bit-identical routing to the original-domain comparison for all 4 combos (no precision/regression risk on the existing f32 path)"
  - "Updated all 28 existing predict() call sites across gtil/sklearn/xgboost/harness to the new (&Config) signature rather than a back-compat shim, so callers compile against the real D-06 surface (must-have)"
  - "f64 element-wise postprocessor arithmetic (e.g. f64 sigmoid/softmax) deferred to Plan 05-03: this plan's f64 goldens use the precision-exact identity postprocessor; apply_postprocessor routes O through an f32 boundary, leaving postprocessor.rs intermediates untouched per Pitfall 2"
  - "LeafId/ScorePerTree return typed GtilError::UnsupportedPredictKind (not todo!/panic) until Plan 05-04 wires them"

patterns-established:
  - "PredictOut: output/accumulator element O orthogonal to the leaf/threshold domain T (output element == input dtype, not leaf type)"
  - "f64 cross-domain comparison contract: route in f64, accumulate in O"
  - "Public Shape vs internal OutputLayout naming split"

requirements-completed: [GTIL-01, GTIL-03, GTIL-07, GTIL-08]

# Metrics
duration: ~22min
completed: 2026-06-10
---

# Phase 5 Plan 02: Typed Config + O-generic GTIL Predict Surface Summary

**Typed `Config`/`PredictKind` entry surface (D-06), public `Shape`/`output_shape` per-kind descriptor (D-07), and an `O`-generic `predict` (f32/f64 input → matching output element across all 4 input×preset combinations, D-05) — with the f32 binary path byte-identical and serial tree-sum preserved.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 7 (4 created, 3 modified) + 28 caller-signature updates across 9 files

## Accomplishments
- `config.rs`: `PredictKind { Default, Raw, LeafId, ScorePerTree }` (kPredict ordering) + `Config { kind, nthread }` with a `Default` impl mirroring `gtil.h:51-52`; JSON-free (D-06).
- `shape.rs`: public `Shape { dims }` + `output_shape()` porting `output_shape.cc:17-39` verbatim per kind — binary default collapses `num_target==1` to dim `1`; `max_num_class` clamps `>= 1` (T-05-04).
- Internal `Shape<'m>` renamed `OutputLayout<'m>` and all call sites updated (Open Q3 name clash resolved).
- `PredictOut` trait (impls for `f32` and `f64`): the input/output/accumulator element `O`, orthogonal to the leaf/threshold domain `T`. `predict<O>` now monomorphizes over both `ModelVariant::F32`/`F64` arms for both input dtypes — all 4 `(input × preset)` combinations valid; output element == input dtype, NOT leaf type (`predict.cc:236`).
- `predict(&Config)` dispatches `Default` (apply postprocessor) vs `Raw` (skip); `LeafId`/`ScorePerTree` → typed `UnsupportedPredictKind` until Plan 05-04.
- Serial tree-sum (GTIL-08) preserved; f32 binary path byte-identical (full `equivalence`/`lightgbm`/`sklearn`/`three_format_equivalence` suite green, max |delta| unchanged); softmax/sigmoid f32 intermediates untouched (Pitfall 2).

## Task Commits

Each task was committed atomically (TDD: test → feat):

1. **Task 1 RED: Config/PredictKind + output_shape failing tests** - `2372221` (test)
2. **Task 1 GREEN: typed Config/PredictKind + public Shape/output_shape** - `39f8f11` (feat)
3. **Task 2 RED: O-generic input/output path failing tests** - `bfa8342` (test)
4. **Task 2 GREEN: O-generic input/output element path** - `4b76030` (feat)

**Plan metadata:** (final docs commit) — see git log

## Files Created/Modified
- `crates/treelite-gtil/src/config.rs` (new) - `PredictKind` enum + `Config` struct (D-06), JSON-free, doc-cited to `gtil.h`.
- `crates/treelite-gtil/src/shape.rs` (new) - public `Shape { dims }` + `output_shape()` per kind (D-07), porting `output_shape.cc` verbatim with defensive clamps.
- `crates/treelite-gtil/tests/config_and_shape.rs` (new) - 6 tests: `Config::default`, `output_shape` all 4 kinds (binary collapse-to-1), `predict(&Config)` Default-vs-Raw dispatch.
- `crates/treelite-gtil/tests/generic_input.rs` (new) - 6 tests: all 4 input×preset combos, f64 dense golden, cross-domain order-preserving comparison, f64 serial two-tree sum.
- `crates/treelite-gtil/src/lib.rs` (modified) - module registration + re-exports; `Shape<'m>`→`OutputLayout<'m>` rename; `PredictOut` trait + impls; `evaluate_tree`/`predict_preset`/`output_leaf_value`/`output_leaf_vector`/`predict`/`apply_postprocessor` made `O`-generic; `next_node` compares in f64; Default/Raw dispatch.
- `crates/treelite-gtil/src/error.rs` (modified) - `UnsupportedPredictKind { kind }` variant.
- `crates/treelite-gtil/Cargo.toml` (modified) - `approx` dev-dependency.
- Caller updates (signature `+ &Config`): `treelite-sklearn/src/{lib,mixin,histgb}.rs`, `treelite-xgboost/tests/{json,ubjson}.rs`, `treelite-harness/src/lib.rs` + `tests/{lightgbm,sklearn,three_format_equivalence}.rs`, `treelite-gtil/tests/{predict,output_shaping}.rs`.

## Decisions Made
- **Compare in f64 universally.** Rather than carry per-`(O,T)` comparison logic, `next_node` widens both operands to `f64`. f32→f64 is exact and order-preserving, so the boolean result is bit-identical to comparing in the original domain for every combination — including the existing f32/f32 path (no regression) and the cross combos (matches C++ usual arithmetic conversions).
- **Updated all 28 call sites, not a shim.** The must-have requires existing callers to compile against the new `(&Config)` signature; a back-compat shim would have hidden the real D-06 surface. `O` is inferred from the `&[O]` data argument, so no turbofish was needed at any call site.
- **Postprocessor O-genericity scoped out (Plan 05-03).** Upstream's postprocessors are templated on `InputT` (element-wise arithmetic runs in the input width) while keeping `float` reduction intermediates (`softmax` `max_margin`/`t`). Faithfully widening the element-wise arithmetic to `f64` is Plan 05-03's explicit job; this plan keeps `postprocessor.rs` untouched (Pitfall 2 / acceptance grep) and routes `O` through an `f32` boundary buffer in `apply_postprocessor`. The f64 goldens this plan validates use the precision-exact `identity` postprocessor, so no fidelity is lost here.

## Deviations from Plan

None - plan executed exactly as written. Two TDD tasks, each RED→GREEN. One pre-existing out-of-scope item was logged (below), not fixed.

## Issues Encountered
- `cargo fmt --check` reports a pre-existing formatting drift in `crates/treelite-builder/src/lib.rs:676` (a `sum_hess_present` line) unrelated to GTIL work. Per the executor scope boundary this is out of scope and was NOT touched; logged to `.planning/phases/05-full-scalar-gtil-equivalence-harness/deferred-items.md`. The `treelite-gtil` crate itself is `cargo fmt`-clean and `cargo clippy`-clean.

## Known Stubs
- `predict` with `PredictKind::LeafId` or `PredictKind::ScorePerTree` returns `GtilError::UnsupportedPredictKind` — intentional, per the plan's explicit scope (those two kinds are wired in Plan 05-04). This is a typed error, not a silent/wrong-output stub.
- `apply_postprocessor` runs non-identity postprocessors through an `f32` boundary even on the f64-input path — intentional, documented in code; full f64 element-wise postprocessor arithmetic is Plan 05-03. The f64 goldens validated here use `identity` (precision-exact in any width).

## User Setup Required
None - no external service configuration required. All work is Rust source; verified via `cargo test --workspace` on the main tree.

## Next Phase Readiness
- The O-generic predict spine + typed `Config`/`Shape` surface are the foundation the remaining GTIL plans build on. Plan 05-03 widens the categorical full-guard and the 3 new postprocessors (making the f64 element-wise postprocessor path faithful + turning RED scaffolds green); Plan 05-04 adds the sparse CSR path and the `LeafId`/`ScorePerTree` kinds (replacing the `UnsupportedPredictKind` errors); Plan 05-05 registers the backend-parameterized harness seam. The `gtil_matrix.rs` 1e-5 runner stays `#[ignore]` until those plans wire the real `Config`/`predict_sparse` calls.
- No blockers.

## Self-Check: PASSED

- All created/modified files present (verified below).
- All task commits present in git history: `2372221`, `39f8f11`, `bfa8342`, `4b76030`.

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 06
subsystem: gtil
tags: [gtil, postprocessor, f64-precision, cast-order, malformed-model-hardening, thiserror, predict]

# Dependency graph
requires:
  - phase: 05-02
    provides: f64-input predict path with f32 postprocessor boundary (the CR-01 defect this plan fixes)
  - phase: 05-03
    provides: the 10/10 f32 postprocessor surface (sigmoid/exponential/softmax/multiclass_ova/...) the f64 twins mirror
  - phase: 05-04
    provides: ScorePerTree / LeafId predict kinds + output_shape (the WR-02 shape/predict pair)
provides:
  - f64-input postprocessors run in f64 (ApplyPostProcessor<double>), softmax excepted (CR-01 closed engine-side)
  - output_shape(ScorePerTree) and predict_score_by_tree agree element-for-element on the third dim (WR-02)
  - 0-node tree returns GtilError::NodeIndexOutOfBounds { node: 0 }, never an OOB panic (WR-03)
  - malformed category-list / leaf-vector CSR offsets return typed MalformedCategoryList / MalformedLeafVector (WR-04)
  - kNone on a numerical test node returns typed UnrecognizedOperator (WR-05)
affects: [05-07, phase-06-cubecl-backend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-element-type postprocessor dispatch via PredictOut::apply_named_postprocessor (f32 → ApplyPostProcessor<float>, f64 → ApplyPostProcessor<double>)"
    - "f64 postprocessor twins (*_f64) beside the f32 fns; sigmoid_alpha/ratio_c stay f32 model fields cast at the op site"
    - "Checked-vs-empty CSR offset distinction: legitimately-empty (Ok) vs malformed (typed Err), never a silent fallthrough"

key-files:
  created: []
  modified:
    - crates/treelite-gtil/src/postprocessor.rs
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/src/shape.rs
    - crates/treelite-gtil/src/error.rs
    - crates/treelite-gtil/tests/predict.rs
    - crates/treelite-gtil/tests/predict_kinds.rs

key-decisions:
  - "Chose the f64-twin shape (sigmoid_f64/exponential_f64/...) over a generic Float trait — keeps the existing f32 fns byte-identical and the softmax-stays-f32 exception explicit"
  - "softmax has NO f64 form: the f64 arm narrows each (row,target) span to f32, runs the f32 softmax, widens back (postprocessor.cc:59-73 hardcodes float for every InputT)"
  - "hinge runs directly in f64 (its result is exactly 0.0/1.0 in any width) — no f32 intermediate needed"
  - "CR-01 divergence asserted in the ~1e-8 RELATIVE band (not 1e-7 absolute) — that is exactly the regime that masked the defect; tests also assert non-bit-identity"
  - "WR-03 reuses NodeIndexOutOfBounds { node: 0 } (no new variant) per the review's exact naming"
  - "has_leaf_vector distinguishes absent offsets (Ok(false), scalar-leaf path) from present-but-inverted/out-of-range (Err MalformedLeafVector) — preserves the legitimate XGBoost scalar path"

patterns-established:
  - "OutputLayout made pub so the public PredictOut trait method apply_named_postprocessor does not leak a more-private type"
  - "next_node is fallible (Result<i32, GtilError>) and carries the node id so kNone surfaces UnrecognizedOperator { node, op }"

requirements-completed: [GTIL-04, GTIL-05, GTIL-06, GTIL-07, ERR-01]

# Metrics
duration: 22min
completed: 2026-06-10
---

# Phase 5 Plan 6: GTIL Engine Gap Closure (CR-01 + WR-02..WR-05) Summary

**f64-input postprocessors now run in f64 (ApplyPostProcessor<double>, softmax excepted), ScorePerTree shape agrees with predict, and every malformed-Model path (0-node tree, inverted CSR offsets, kNone numerical node) returns a typed GtilError instead of an OOB panic or a silent wrong prediction.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-06-10T08:53Z
- **Completed:** 2026-06-10
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- **CR-01 closed engine-side:** the f64 × {sigmoid, exponential, exponential_standard_ratio, logarithm_one_plus_exp, signed_square, multiclass_ova} surface now runs its element arithmetic in f64, matching upstream `ApplyPostProcessor<double>`. `softmax` correctly stays f32 on every InputT. A unit test proves `sigmoid_f64`/`exponential_f64` diverge from the collapsed-f32 path in the ~1e-8 band that masked the defect. The f32 path is byte-identical (all pre-existing goldens unchanged).
- **WR-02:** `output_shape(ScorePerTree)` third dim now clamps `(a*b).max(1)` with the same `unwrap_or(1).max(0)` defaulting as `predict_score_by_tree`'s `lvs`, so the published shape and the produced buffer agree element-for-element for every (scalar or malformed) `leaf_vector_shape`.
- **WR-03:** a 0-node tree returns `GtilError::NodeIndexOutOfBounds { node: 0 }` via a guard at the top of `evaluate_tree`, instead of an OOB slice panic.
- **WR-04:** malformed category-list / leaf-vector CSR offsets (inverted, out-of-range, or missing) return typed `MalformedCategoryList` / `MalformedLeafVector`; legitimately-empty lists and absent (scalar-leaf) offsets keep their correct non-error behavior.
- **WR-05:** `kNone` (any unrecognized operator) on a numerical test node returns typed `UnrecognizedOperator { node, op }`, matching upstream's `TREELITE_CHECK(false)` fatal path, instead of a silent route-right.
- **No regression:** `cargo test --workspace` green (255 tests pass); clippy clean workspace-wide.

## Task Commits

Each task was committed atomically:

1. **Task 1: CR-01 — f64-input postprocessors run in f64** - `e366dea` (feat)
2. **Task 2: WR-02 + WR-03 — ScorePerTree shape agreement + 0-node guard** - `1a151e3` (fix)
3. **Task 3: WR-04 + WR-05 — typed errors for malformed offsets / kNone** - `a65db82` (fix)

_Task 1 is a tdd="true" task; the divergence test and the f64 twins it exercises were committed together (the test references the twins, so they are inseparable in one compiling commit)._

## Files Created/Modified

- `crates/treelite-gtil/src/postprocessor.rs` - Added f64 twins (`sigmoid_f64`, `exponential_f64`, `exponential_standard_ratio_f64`, `logarithm_one_plus_exp_f64`, `signed_square_f64`, `multiclass_ova_f64`); updated the cast-ordering doc to cite `ApplyPostProcessor<double>` with the softmax exception; added CR-01 divergence + twin-sanity tests.
- `crates/treelite-gtil/src/lib.rs` - `apply_postprocessor` dispatches via `PredictOut::apply_named_postprocessor` (f32/f64 impls); new `apply_postprocessor_f64`; `OutputLayout` made `pub`; `num_nodes == 0` guard in `evaluate_tree`; `next_node` made fallible (UnrecognizedOperator on kNone); `category_list_safe` / `has_leaf_vector` return `Result` distinguishing legitimate-empty/absent from malformed; Results threaded through `evaluate_tree` / `predict_preset` / `predict_score_by_tree_preset`.
- `crates/treelite-gtil/src/shape.rs` - ScorePerTree third dim `(a*b).max(1)` with `unwrap_or(1).max(0)` to agree with predict.
- `crates/treelite-gtil/src/error.rs` - Added `UnrecognizedOperator { node, op }`, `MalformedCategoryList { node }`, `MalformedLeafVector { node }`; imported `treelite_core::Operator`.
- `crates/treelite-gtil/tests/predict.rs` - kNone numerical node, inverted + out-of-range category list, empty-list preserved, inverted leaf-vector vs absent (scalar) preserved.
- `crates/treelite-gtil/tests/predict_kinds.rs` - ScorePerTree shape/predict third-dim agreement (scalar + degenerate shape); 0-node tree typed-error test.

## Decisions Made

- Used the `*_f64` twin shape rather than a generic `Float` bound: keeps the f32 fns byte-identical and makes the softmax-stays-f32 exception explicit and grep-able.
- `softmax` deliberately has no f64 twin; the f64 arm narrows each row to f32, runs the f32 softmax, widens back (upstream hardcodes `float max_margin`/`t` for every `InputT`).
- `hinge` runs directly in f64 (its output is exactly 0.0/1.0) — no f32 round-trip.
- CR-01 divergence tests assert relative divergence `> 1e-8` AND non-bit-identity, because the real f32-vs-f64 gap on a large margin sits at ~1e-8 — exactly the regime that kept the defect inside the 1e-5 gate. A coarser `1e-7` threshold (the plan's behavior text) would never trip and would falsely "pass" a collapsed-f32 path; the stronger bit-identity assertion guarantees the f64 path is genuinely a different computation.
- `OutputLayout` promoted to `pub` so the public `PredictOut::apply_named_postprocessor` method does not leak a more-private type (private-interfaces warning).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] CR-01 divergence test threshold corrected to the empirical f32-vs-f64 band**
- **Found during:** Task 1 (RED→GREEN)
- **Issue:** The plan's behavior/acceptance text specified the f64 path must differ from the collapsed-f32 path by `> 1e-7`. Empirically the divergence on a large-margin sigmoid/exponential is ~1e-8 relative (and saturated values are absolute-identical to 0/1) — a `1e-7` threshold never trips, so the test as literally specified would have failed against a *correct* f64 implementation.
- **Fix:** Asserted relative divergence `> 1e-8` over a probe set AND non-bit-identity (`to_bits()` differ). This is strictly stronger evidence that the f64 path is a genuinely higher-precision computation, and it is the same ~1e-8 band that masked CR-01 inside the 1e-5 gate.
- **Files modified:** crates/treelite-gtil/src/postprocessor.rs
- **Verification:** `sigmoid_f64_diverges_from_collapsed_f32_on_large_margin` and `exponential_f64_diverges_from_collapsed_f32_on_large_arg` pass; the test fails (as intended) if the f64 twin is replaced by the collapsed-f32 path.
- **Committed in:** e366dea (Task 1 commit)

**2. [Rule 3 - Blocking] OutputLayout made pub to avoid a private-interfaces warning**
- **Found during:** Task 1 (wiring the per-type postprocessor dispatch)
- **Issue:** Adding `apply_named_postprocessor` to the public `PredictOut` trait surfaced `OutputLayout` (a private struct) in a public signature → `private_interfaces` warning.
- **Fix:** Promoted `OutputLayout` from private to `pub` (its fields stay private, so it remains opaque to external callers).
- **Files modified:** crates/treelite-gtil/src/lib.rs
- **Verification:** `cargo clippy -p treelite-gtil` clean; no warnings.
- **Committed in:** e366dea (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking; test-threshold correctness + a visibility fix)
**Impact on plan:** Both necessary to land a correct, warning-free implementation. The threshold correction makes the CR-01 proof *stronger* (bit-identity + the actual masking band) rather than weaker. No scope creep — the f32 path stays byte-identical and all pre-existing goldens are unchanged.

## Issues Encountered

- The initial CR-01 divergence test used a `1e-7` absolute threshold and probed fully-saturated margins (±40), where both the f64 and collapsed-f32 sigmoid round to the same value — the test failed. Re-probed the pre-/post-saturation slope (margins ~±10 to ±18) where f32 vs f64 `exp` genuinely differ (~1e-8 relative), and switched to relative-divergence + bit-identity assertions. Resolved within Task 1.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- CR-01 is closed at the engine level. **Plan 05-07 (next wave) must land the large-margin f64 fixture + harness assertion that drives CR-01 against the 1e-5 gate** — this plan deliberately ships only the engine fix and a unit-level divergence proof, not the equivalence fixture.
- WR-01 (sparse harness re-derives CSR from NaN-presence instead of asserting a real captured CSR) remains OPEN and is out of scope for this plan — still tracked in STATE.md blockers for 05-07.
- All malformed-Model paths now return typed `GtilError`s, so the Phase-6 cubecl backend can rely on the same contracts.

## Self-Check: PASSED

- All 6 modified files exist on disk.
- All 3 task commits present in git history (e366dea, 1a151e3, a65db82).
- `cargo test --workspace` green (255 passed, 0 failed); clippy clean.

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

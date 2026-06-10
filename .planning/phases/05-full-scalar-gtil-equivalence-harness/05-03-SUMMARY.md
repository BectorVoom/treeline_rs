---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 03
subsystem: gtil
tags: [gtil, postprocessor, signed-square, hinge, multiclass-ova, categorical, representability-guard, gtil-06, gtil-05, gtil-04, verbatim-port, f32, f64]

# Dependency graph
requires:
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 02)
    provides: "PredictOut O-generic trait (f32/f64) + apply_postprocessor f32-boundary seam + O-generic evaluate_tree the new guard/postprocessors hook into"
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 01)
    provides: "RED unit scaffolds (signed_square/hinge/multiclass_ova postprocessor stubs + categorical_full_guard_red) this plan turns GREEN; the 64 frozen gtil/ goldens incl. the 2**24+1 categorical edge value"
provides:
  - "signed_square / hinge / multiclass_ova postprocessors ported verbatim (postprocessor.cc:22-31,77-82) — all 10 upstream postprocessors now supported (GTIL-04)"
  - "apply_postprocessor dispatch arms for all three (multiclass_ova row-wise like softmax); only genuinely-unknown names hit UnsupportedPostprocessor"
  - "Full O-generic categorical representability guard (predict.cc:135-143): PredictOut::category_match + MANTISSA_BITS const, per-dtype bound 2^24 (f32) / 2^32-1 (f64), reject-before-cast (GTIL-06, T-05-06)"
  - "next_node_categorical made O-generic; evaluate_tree threads the O-domain categorical value (no lossy f32 narrowing on the f64 path)"
affects: [05-04, 05-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Verbatim postprocessor idiom extended: each new fn takes f32, doc-comment quotes the upstream C++ + the 'runs in f32 — no promotion' cast-order contract (Pitfall 2)"
    - "Per-dtype representability bound carried as a PredictOut associated const (MANTISSA_BITS) — deliberately NOT named DIGITS to avoid the inherent f32::DIGITS/f64::DIGITS (decimal-digit) shadowing of Self::DIGITS"
    - "Categorical guard widened to O-generic via a single trait method (category_match) keeping the upstream min()/fabs/truncate formula verbatim per dtype"

key-files:
  created: []
  modified:
    - crates/treelite-gtil/src/postprocessor.rs
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/tests/predict.rs

key-decisions:
  - "PredictOut const named MANTISSA_BITS, not DIGITS: the inherent f32::DIGITS/f64::DIGITS are the DECIMAL digit count (6/15) and shadow a trait const named DIGITS at Self::DIGITS, silently yielding 2^6 instead of 2^24 — caught at test time and renamed"
  - "Corrected the Plan-01 RED categorical assertion to true upstream behavior: float32(2^24+1) rounds to exactly 2^24 (NOT > 2^24), so upstream does NOT magnitude-reject it. The genuine rejected gap value must round strictly above 2^24 (used 2^24+64 = 16_777_280). Frozen goldens were NOT touched (D-08); only the unit-test contract was made numerically faithful."
  - "multiclass_ova dispatched row-wise per (row, target) over num_class cells, copying the softmax arm structure; sigmoid_alpha stays f32 (Pitfall 2)"

patterns-established:
  - "Associated-const per-dtype numeric bounds on PredictOut (MANTISSA_BITS) for representability arithmetic"
  - "O-generic categorical membership via PredictOut::category_match"

requirements-completed: [GTIL-04, GTIL-05, GTIL-06]

# Metrics
duration: ~8min
completed: 2026-06-10
---

# Phase 5 Plan 03: Complete Postprocessor Surface + Full Categorical Guard Summary

**The final three postprocessors (`signed_square`/`hinge`/`multiclass_ova`) ported verbatim with f32 cast order — completing 10/10 (GTIL-04) — plus the full O-generic categorical float-representability guard (`min(u32::MAX, 2^digits)`, 2^24 f32 / 2^32-1 f64, reject-before-cast) with preserved child polarity (GTIL-06) and confirmed NaN→default-child routing (GTIL-05).**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2 (both TDD: RED scaffolds from Plan 01 → GREEN)
- **Files modified:** 3 (all in `treelite-gtil`)

## Accomplishments
- `postprocessor.rs`: ported `signed_square` (`(v*v).copysign(v)`), `hinge` (strict `> 0`), and `multiclass_ova` (per-class independent sigmoid, `sigmoid_alpha` stays f32) verbatim from `postprocessor.cc:22-31,77-82`, each with the upstream-quoting doc-comment and the "runs in f32 — no promotion" contract (Pitfall 2). Turned the three Plan-01 RED postprocessor scaffolds GREEN (un-ignored, now call the real fns). All 10 upstream postprocessors are now supported (GTIL-04).
- `lib.rs::apply_postprocessor_f32`: added `signed_square`/`hinge` per-cell arms and the `multiclass_ova` row-wise arm (per `(row, target)` over `num_class` cells, copying the `softmax` loop structure). The three names no longer fall through to `UnsupportedPostprocessor` — only genuinely-unknown names do.
- `lib.rs` categorical guard: replaced the minimal Phase-4 subset with the full upstream formula (`predict.cc:135-143`), made O-generic. Added `PredictOut::MANTISSA_BITS` (24 f32 / 53 f64) + `PredictOut::category_match`, computing `max_representable_int = min(u32::MAX, 2^MANTISSA_BITS)` and rejecting `fvalue < 0 || fabs(fvalue) > max` BEFORE the `as u32` truncation (T-05-06). `next_node_categorical` is now generic over `O`; `evaluate_tree` threads the categorical value in its `O` domain (previously narrowed lossily to f32, which would have applied the f32 bound on the f64 path).
- Preserved verbatim: `category_list_safe` bounds-safe slice (T-05-07) and the `category_list_right_child` polarity block. Confirmed (unchanged) the NaN→`default_child` route in `evaluate_tree` precedes the categorical dispatch (GTIL-05) — a missing value never reaches the guard.
- Turned the Plan-01 `categorical_full_guard_red` scaffold GREEN and added f32/f64 boundary, negative/±inf, child-polarity, and NaN-gate unit tests (6 categorical-guard tests total).
- Full `cargo test --workspace` green: `lightgbm_categorical` and all existing goldens stay within 1e-5; the f32 binary path is unchanged; `gtil_matrix.rs` remains `#[ignore]` (its real `Config`/`predict_sparse` wiring is Plan 05-04/05-05).

## Task Commits

Each task committed atomically:

1. **Task 1: signed_square / hinge / multiclass_ova + dispatch (GTIL-04)** — `7db3de3` (feat)
2. **Task 2: full O-generic categorical representability guard (GTIL-06, GTIL-05)** — `f3939b9` (feat)

**Plan metadata:** (final docs commit) — see git log.

## Files Created/Modified
- `crates/treelite-gtil/src/postprocessor.rs` (modified) — three new verbatim postprocessors + their GREEN unit tests; updated module docstring (10/10 surface complete).
- `crates/treelite-gtil/src/lib.rs` (modified) — three `apply_postprocessor_f32` arms; `PredictOut::MANTISSA_BITS` + `category_match` (f32/f64 impls); O-generic `next_node_categorical`; `evaluate_tree` passes the `O`-domain value; `red_scaffolds` module renamed `categorical_guard` and expanded to 6 GREEN tests.
- `crates/treelite-gtil/tests/predict.rs` (modified) — `unsupported_postprocessor_is_typed_error` re-pointed from `"hinge"` (now supported) to a genuinely-unknown name.

## Decisions Made
- **`MANTISSA_BITS`, not `DIGITS`.** A trait const named `DIGITS` is shadowed at `Self::DIGITS` by the inherent `f32::DIGITS`/`f64::DIGITS` (the *decimal* digit count, 6/15), silently computing `2^6 = 64` as the f32 bound instead of `2^24`. This surfaced as two failing boundary tests; the const was renamed to `MANTISSA_BITS` (mirroring `f32::MANTISSA_DIGITS = 24`) with an explicit doc-note warning future maintainers.
- **Corrected the RED categorical assertion to true upstream behavior.** Plan 01's RED scaffold (and the capture-script comment) asserted that `2^24 + 1` is magnitude-rejected on the f32 path. It is not: `float32(2^24 + 1) == 2^24` exactly (verified via `uv run python`), so `fabs(fvalue) > 2^24` is false and upstream accepts it (matches category `16_777_216`). The genuine f32-gap rejection requires a value whose f32 representation strictly exceeds `2^24`; the GREEN test uses `2^24 + 64 = 16_777_280`. The frozen goldens (D-08) were NOT regenerated — only the in-code unit-test contract was made numerically faithful to what upstream (and the goldens) actually compute.
- **`multiclass_ova` row-wise.** Dispatched per `(row, target)` over `num_class` cells (same structure as `softmax`), not per-cell — it is the one-vs-all per-class sigmoid; `sigmoid_alpha` is a float model field and stays f32 (Pitfall 2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Trait-const name collision yielded the wrong f32 categorical bound**
- **Found during:** Task 2 (categorical boundary tests failed: f32 bound computed as 2^6 = 64, not 2^24)
- **Issue:** The `PredictOut` associated const was first named `DIGITS`. At `Self::DIGITS` inside `impl PredictOut for f32`, Rust's name resolution prefers the *inherent* `f32::DIGITS` (decimal digits = 6) over the trait const, so `1u64 << Self::DIGITS` evaluated to `2^6` — rejecting nearly all categorical values.
- **Fix:** Renamed the trait const to `MANTISSA_BITS` (no inherent collision) with a doc-note. Both boundary tests then passed.
- **Files modified:** `crates/treelite-gtil/src/lib.rs`
- **Commit:** `f3939b9`

**2. [Rule 1 - Bug] Pre-existing test used a now-supported postprocessor as its "unsupported" probe**
- **Found during:** Task 1 (`unsupported_postprocessor_is_typed_error` in `tests/predict.rs` probed `"hinge"`, which Task 1 makes supported)
- **Issue:** With `hinge` now ported, `predict(...)` returns `Ok`, so the test's `unwrap_err()` panicked. The test's premise (hinge unsupported) became false.
- **Fix:** Re-pointed the probe to `"not_a_real_postprocessor"` — a genuinely-unknown name that still exercises the typed `UnsupportedPostprocessor` path.
- **Files modified:** `crates/treelite-gtil/tests/predict.rs`
- **Commit:** `7db3de3`

**3. [Rule 1 - Bug, documentation-only, NOT fixed in code] Plan-01 RED assertion / capture comment overstated the f32 gap-value behavior**
- **Found during:** Task 2 (the `2^24+1` RED assertion could not be made GREEN against true upstream behavior)
- **Issue:** `float32(2^24 + 1)` rounds to exactly `2^24`, so it is NOT magnitude-rejected; the RED scaffold (and `fixtures/capture_gtil_matrix.py` comments at :39-42,:198-200) describe it as a rejected gap value, which is numerically incorrect.
- **Resolution:** Rewrote the GREEN unit test to assert the *true* upstream contract using `2^24 + 64` (strictly > 2^24, genuinely rejected). The frozen capture script and goldens were deliberately left unchanged (D-08: committed matrices are the contract; their golden VECTORS are correct upstream truth — only the script's explanatory comment is imprecise). Documented here for the verifier; not a runtime defect.

**Total deviations:** 2 auto-fixed bugs (both committed) + 1 documentation-accuracy note (frozen-fixture comment, intentionally not modified per D-08).
**Impact on plan:** No scope change. Both task done-criteria met; the corrected gap value exercises the same GTIL-06 representability boundary the plan intended.

## Issues Encountered
- `cargo fmt --check` flagged drift in the new test code; resolved with `cargo fmt -p treelite-gtil` (the pre-existing `treelite-builder` fmt drift noted in 05-02-SUMMARY remains out of scope, untouched). `treelite-gtil` is `cargo clippy`- and `cargo fmt`-clean.

## Known Stubs
- None introduced by this plan. (Carried from Plan 05-02, unchanged and out of this plan's scope: `LeafId`/`ScorePerTree` predict kinds return `UnsupportedPredictKind` until Plan 05-04; `apply_postprocessor` routes `O` through an f32 boundary buffer — the goldens validated to date use precision-exact `identity`/binary postprocessors.)

## Threat Flags
None — no new security-relevant surface. The plan's `mitigate` dispositions were applied: T-05-06 (full representability guard rejects before the `as u32` cast) is implemented; T-05-07 (`category_list_safe`) and T-05-08 (NaN→default before the guard) are preserved verbatim.

## User Setup Required
None — all work is Rust source; verified via `cargo test --workspace` on the main tree.

## Next Phase Readiness
- The traversal/postprocessor numeric core is now complete: 10/10 postprocessors and the full O-generic categorical guard. Plan 05-04 adds the sparse CSR path and the `LeafId`/`ScorePerTree` kinds (replacing `UnsupportedPredictKind`); Plan 05-05 wires the backend-parameterized harness and un-ignores `gtil_matrix.rs`. The `2^24+1` categorical edge value in the frozen f32/f64 goldens will be asserted by that matrix runner (the f32 cell matches category `2^24`, the f64 cell matches `2^24+1` — exactly what this guard now produces).
- No blockers.

## Self-Check: PASSED

- All modified files present: `crates/treelite-gtil/src/postprocessor.rs`, `crates/treelite-gtil/src/lib.rs`, `crates/treelite-gtil/tests/predict.rs`.
- Both task commits present in git history: `7db3de3` (Task 1), `f3939b9` (Task 2).

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

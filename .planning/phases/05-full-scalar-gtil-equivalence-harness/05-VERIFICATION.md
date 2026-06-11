---
phase: 05-full-scalar-gtil-equivalence-harness
verified: 2026-06-10T12:00:00Z
resolved: 2026-06-11T00:00:00Z
status: passed
score: 12/12
overrides_applied: 0
resolution_note: "All three deferred human_verification items resolved 2026-06-11 during the v1.1 milestone-close (user chose 'stop and resolve'). gtil_matrix + full workspace green (310 passed). See per-item result fields."
human_verification:
  - test: "WR-02 fragile bit-inequality guard (deferred per user decision)"
    expected: "The WR-06 max_div > 0.0 strict bit-inequality assertion catches a true f32->f64 collapse but is fragile against future fixtures where f32 and f64 sigmoid round identically on every row. User deferred fixing to a relative-divergence floor."
    why_human: "User explicitly deferred this (per 05-REVIEW-FIX.md). Verifying whether the current behavior is acceptable or whether the stronger floor should be adopted before Phase 6 ships is a design decision, not a code check."
    result: "RESOLVED 2026-06-11 — root cause found: the `default` (post-sigmoid) axis saturates toward 1.0 in both f32 and f64 for large margins, so saturated rows are bit-identical (that was the fragility). Empirically measured per-kind divergence: raw ≈ 2.9e-6 (never saturates), default ≈ 5.5e-8, leaf_id/score_per_tree = 0. Switched the WR-06 guard to the `raw` margin axis (gtil_matrix.rs WR-06 gate) — stronger signal, saturation-proof, still collapses to 0 under a silent f32→f64 pre-cast. The old 'raw shares the f64-accumulated margin' comment was incorrect and was corrected. User chose 'switch guard to raw margin'."
  - test: "WR-01 comments in gtil_matrix.rs overstate what the 1e-5 gate proves for CR-01"
    expected: "The large_margin f64 sigmoid 1e-5 gate asserts correctness but would NOT catch a reversion to the collapsed-f32 path (deviation ~6e-8, inside 1e-5). The comments at lines 540-554 / 585-594 say the gate 'would have caught' the collapse, which is inaccurate. The real guard is WR-06."
    why_human: "User explicitly deferred rewording (per 05-REVIEW-FIX.md). The code is functionally correct; the comment accuracy is a documentation call."
    result: "RESOLVED 2026-06-11 — reworded the CR-01 comment and eprintln in gtil_matrix.rs to state the 1e-5 gate proves the f64 path matches upstream but does NOT, on its own, catch a collapse-to-f32 (~6e-8, inside 1e-5); the actual collapse guard is the WR-06 paired raw-margin divergence gate."
  - test: "IN-01: UnsupportedPredictKind variant is dead code"
    expected: "error.rs:93-100 declares UnsupportedPredictKind with a doc comment referencing Plan 05-04. No code path constructs it. Should be removed or updated to a forward-compat placeholder. User deferred this as informational."
    why_human: "Dead code is not a correctness issue. Whether to remove it is a code-hygiene call the user has explicitly deferred."
    result: "RESOLVED 2026-06-11 — removed the obsolete `GtilError::UnsupportedPredictKind` variant from treelite-gtil/src/error.rs (never constructed; all four predict kinds are wired). Updated the stale reference comment in predict_kinds.rs. Workspace builds, 310 tests green."
---

# Phase 5: Full Scalar GTIL & Equivalence Harness — Verification Report

**Phase Goal:** Widen the inference spine to the complete scalar GTIL reference — all predict kinds, all postprocessors, sparse input, categoricals, output shaping — and the full seeded equivalence harness that is the 1e-5 measurement instrument for everything after.
**Verified:** 2026-06-10T12:00:00Z (resolved 2026-06-11)
**Status:** passed
**Re-verification:** Yes — 3 deferred human-verification items resolved 2026-06-11 (WR-02 raw-margin guard, WR-01 comment accuracy, IN-01 dead-variant removal); see frontmatter per-item `result` fields

## Goal Achievement

### Observable Truths

All 4 ROADMAP success criteria are verified:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Prediction works over a dense row-major matrix AND a sparse CSR matrix (absent entries as NaN, not 0), with dense↔sparse parity asserted | VERIFIED | `cargo test -p treelite-harness gtil_matrix` runs 96 cells, 96 dense==sparse parity asserts (all within 1e-9), sparse arm loads frozen CSR triple verbatim (WR-01); `predict` and `predict_sparse` both wired in `lib.rs` |
| 2 | All four predict kinds and all ten postprocessors ported verbatim (mixed-precision softmax/exp2/log1p preserved), NaN-only missing-value routing, categorical float-representability guard + child polarity | VERIFIED | `kind_of()` maps all 4 kinds; `apply_postprocessor_f32`/`_f64` cover all 10 postprocessors (identity, identity_multiclass, sigmoid/_f64, signed_square/_f64, hinge, exponential/_f64, exponential_standard_ratio/_f64, logarithm_one_plus_exp/_f64, softmax, softmax_f64, multiclass_ova/_f64); NaN routes via default child; categorical guard via `PredictOut::category_match` with per-dtype mantissa bits (24 for f32, 53 for f64) |
| 3 | Output shaping correct — `GetOutputShape` per kind, leaf-vector broadcast, tree averaging, f64 base-score addition — serial tree_id order | VERIFIED | `shape.rs` implements `output_shape` for all 4 kinds with `(a*b).max(1)` clamp for ScorePerTree; `predict_score_by_tree` uses matching `(a*b).max(1)` lvs; leaf-vector broadcast in `output_leaf_vector`; RF averaging via `average_tree_output` flag; f64 base-score add via `PredictOut::add_base_score`; 258 tests pass, 0 failures |
| 4 | Harness generates seeded dense + sparse CSR inputs, compares against C++-captured goldens (committed with toolchain/libm manifest) across model types, both presets, all predict kinds, asserting within 1e-5 and reporting max observed deviation | VERIFIED | 96 committed fixtures (3 models × 2 presets × 2 seeds × 4 kinds × 2 layouts = matches 96); `gtil_matrix` test: 48 f32-input, 48 f64-input, 48 sparse cells; global max \|delta\| = 2.91e-6 (< 1e-5); EQV-04 max \|delta\| reported per cell via `eprintln!`; CR-01 visibility: 16 large_margin f64 sigmoid cells tracked separately (worst 2.91e-6) |

**Score:** 12/12 truths verified (see per-requirement table below)

### Additional Must-Haves from Plan Frontmatter

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| CR-01-1 | f64-input model's non-softmax postprocessors run in f64, NOT through f32 intermediate | VERIFIED | `apply_postprocessor_f64` dispatches to `*_f64` twins for sigmoid/exponential/exponential_standard_ratio/logarithm_one_plus_exp/signed_square/multiclass_ova; `apply_named_postprocessor` trait method on `impl PredictOut for f64` calls `apply_postprocessor_f64` |
| CR-01-2 | softmax keeps cell in f64 for subtraction/exp/divide (upstream `softmax<double>` cast order) | VERIFIED | `postprocessor::softmax_f64` correctly keeps `*cell` (f64) for `(*cell - max_margin as f64).exp()` and `*cell /= divisor`; `apply_postprocessor_f64` calls `softmax_f64` directly — no narrow-to-f32 dance; 05-REVIEW-FIX.md confirms fix for the CR-01 softmax blocker |
| CR-01-3 | A committed large-margin f64 sigmoid fixture asserts against the 1e-5 gate | VERIFIED | 16 `large_margin.f32.f64.*.golden.json` cells run through `gtil_matrix`; worst \|delta\| = 2.91e-6 (< 1e-5); CR-01 coverage gate at line 585-589 ensures ≥1 large_margin f64 cell ran |
| WR-01 | Sparse cells consume frozen CSR triple, NOT NaN-presence reconstruction | VERIFIED | `FrozenCsr` struct in `gtil_matrix.rs`; `run_cell` loads `golden.csr` verbatim for sparse cells; `build_csr` retained only for D-04 parity probe; all 48 sparse golden files confirmed to carry `"csr": {"data":...,"indices":...,"indptr":...}` |
| WR-02 | ScorePerTree shape and predict_score_by_tree third-dim clamp agree | VERIFIED | `shape.rs:61-64`: `(a*b).max(1)` with `unwrap_or(1).max(0)` per factor; `lib.rs:1122-1124`: matching `unwrap_or(1).max(0)` and `(a*b).max(1)` for `lvs` — identical semantics |
| WR-03 | 0-node tree returns typed GtilError, never OOB panic | VERIFIED | `lib.rs:446-448`: `if num_nodes == 0 { return Err(GtilError::NodeIndexOutOfBounds { node: 0 }) }` before any accessor |
| WR-04 | Malformed category-list/leaf-vector offsets return typed GtilError | VERIFIED | `category_list_safe` returns `Err(GtilError::MalformedCategoryList { node: nid })` for inverted/out-of-range; `has_leaf_vector` returns `Err(GtilError::MalformedLeafVector { node: leaf })` for present-but-malformed; legitimately-empty scalar-leaf case preserved |
| WR-05 | `Operator::kNone` on numerical node returns UnrecognizedOperator typed error | VERIFIED | `lib.rs:352`: `Operator::kNone => return Err(GtilError::UnrecognizedOperator { node, op })` — no bare `=> false` route-right; stale comment removed (grep returns 0) |
| WR-06 | Paired f32/f64 assertion proves dtype axis is a distinct computation | VERIFIED | WR-06 loop at `gtil_matrix.rs:604-648` collects `large_margin` model's `default` f32 and f64 `own` outputs, asserts `max_div > 0.0`; 4 pairs checked, max divergence 5.51e-8; a silent f32→f64 pre-cast would collapse and fail this gate |
| EQV-01 | Harness generates seeded dense + sparse CSR inputs | VERIFIED | 96 fixtures committed covering both seeds (1234, 5678), both layouts (dense, sparse), both input dtypes (f32, f64), 4 kinds |
| EQV-02 | Golden vectors captured from C++ Treelite, committed with manifest | VERIFIED | All 96 goldens have `manifest.backend = "scalar-cpu"`; each carries `os`, `arch`, `seed`, per-fixture sha256 fields |
| EQV-03 | Rust predictions assert within 1e-5 of goldens | VERIFIED | `approx::assert_abs_diff_eq!(g, w, epsilon = 1e-5)` hard gate at `gtil_matrix.rs:412`; 96 cells pass, global max \|delta\| = 2.91e-6 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-gtil/src/postprocessor.rs` | f64-capable postprocessors (sigmoid_f64, exponential_f64, softmax_f64, etc.) + softmax stays f32 | VERIFIED | 10 f32 fns + 6 f64 twins (sigmoid_f64, exponential_f64, exponential_standard_ratio_f64, logarithm_one_plus_exp_f64, signed_square_f64, multiclass_ova_f64, softmax_f64); 14 tests pass |
| `crates/treelite-gtil/src/error.rs` | UnrecognizedOperator, MalformedCategoryList, MalformedLeafVector variants | VERIFIED | All three declared at lines 141, 156, 165; plus SparseColumnOutOfBounds, SparseRowPtrInvalid from Plan 05-04 |
| `crates/treelite-gtil/src/lib.rs` | f64 dispatch, 0-node guard, typed errors wired, no out_to_f32 blanket | VERIFIED | `apply_named_postprocessor` trait routes f64 to `apply_postprocessor_f64`; num_nodes==0 guard at :446; `category_list_safe` + `has_leaf_vector` return Results; `out_to_f32`/`out_from_f32` removed (grep returns nothing) |
| `crates/treelite-gtil/src/shape.rs` | ScorePerTree third-dim `(a*b).max(1)` | VERIFIED | Clamp at :63; `unwrap_or(1).max(0)` per factor matches lib.rs predict |
| `crates/treelite-harness/tests/gtil_matrix.rs` | Frozen-CSR load for sparse cells, WR-06 paired assertion, CR-01 1e-5 gate | VERIFIED | `FrozenCsr` struct; `frozen_csr()` fn; `run_cell` loads verbatim; WR-06 loop at :604; CR-01 counter at :437 with mandatory gate at :585 |
| `fixtures/gtil/*.golden.json` | 96 committed goldens, all 4 kinds, both presets, both dtypes, both seeds, leaf_vec, large_margin | VERIFIED | 96 files; 24 per kind (default/raw/leaf_id/score_per_tree); 32 leaf_vec_mc; 32 large_margin; 48 f64 cells; 48 sparse |
| `fixtures/capture_gtil_matrix.py` | All 4 kinds, scalar-cpu manifest, 2^24 edge, dense==sparse assert, large_margin model, CSR freeze | VERIFIED | grep counts: treelite.gtil.predict (10), predict_leaf/predict_per_tree (4 each), scalar-cpu (3), 2**24 (7), allclose+equal_nan (4), build_large_margin_model (2) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `lib.rs` apply_postprocessor | `postprocessor.rs` f64 twins | `O::apply_named_postprocessor` → `apply_postprocessor_f64` | VERIFIED | `impl PredictOut for f64` at :283 dispatches to `apply_postprocessor_f64`; all 6 non-softmax/non-identity f64 twins called |
| `lib.rs` softmax f64 arm | `postprocessor::softmax_f64` | `postprocessor::softmax_f64(&mut output[start..end])` at :1365 | VERIFIED | Direct call to f64 twin, not narrow-to-f32 dance |
| `lib.rs` evaluate_tree | `error.rs` NodeIndexOutOfBounds | `num_nodes == 0` guard at :446 | VERIFIED | Returns `Err(GtilError::NodeIndexOutOfBounds { node: 0 })` |
| `lib.rs` next_node / evaluate_tree | `error.rs` UnrecognizedOperator | `Operator::kNone => return Err(...)` at :352 | VERIFIED | Wired through numerical branch; stale `returning right_child` comment absent |
| `lib.rs` category_list_safe | `error.rs` MalformedCategoryList | `Err(GtilError::MalformedCategoryList { node: nid })` at :417, :422 | VERIFIED | Inverted and out-of-range cases both handled; legitimately-empty returns Ok(&[]) |
| `lib.rs` has_leaf_vector | `error.rs` MalformedLeafVector | `Err(GtilError::MalformedLeafVector { node: leaf })` at :573 | VERIFIED | Present-but-malformed returns error; absent (scalar-leaf) returns Ok(false) |
| `gtil_matrix.rs` sparse cells | `fixtures/gtil/*.golden.json` csr field | `frozen_csr::<O>(frozen)` → `SparseCsr` at :343-375 | VERIFIED | `FrozenCsr` serde struct; `run_cell` loads verbatim for sparse; `build_csr` kept only for D-04 parity |
| `capture_gtil_matrix.py` | `fixtures/gtil/*.sparse.*.golden.json` | `_freeze_cell` writes `csr` payload | VERIFIED | All sparse goldens carry `"csr": {"data":...,"indices":...,"indptr":...}` confirmed by Python probe |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `gtil_matrix.rs` `gtil_matrix()` test | `golden` / `expected` / `own` | `std::fs::read_to_string(path)` → `serde_json::from_str` → `run_cell` → `predict`/`predict_sparse` | Real fixture files + real Rust prediction engine | FLOWING |
| `lib.rs` `predict` / `predict_sparse` | `output` Vec | `predict_rows` → `predict_preset` → `evaluate_tree` traversal → `apply_postprocessor` | Real model tree traversal + postprocessor | FLOWING |
| `postprocessor::sigmoid_f64` | f64 output cell | `1.0_f64 / (1.0_f64 + (-(sigmoid_alpha as f64) * v).exp())` | Real f64 arithmetic, not collapsed f32 | FLOWING |
| `postprocessor::softmax_f64` | f64 row cells | `(*cell - max_margin as f64).exp()` in f64, `*cell /= divisor (f64)` | Real f64 arithmetic per upstream cast order | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full workspace test suite green | `cargo test --workspace` | 258 tests, 0 failures, 0 ignored on non-deferred | PASS |
| gtil_matrix: all 96 cells within 1e-5 | `cargo test -p treelite-harness gtil_matrix -- --nocapture` | 96 cells, global max \|delta\| = 2.91e-6, ok | PASS |
| CR-01 large_margin f64 cells exercised | Same test, CR-01 gate at :585 | 16 large_margin f64 cells, worst 2.91e-6 | PASS |
| WR-06 f32/f64 paired divergence | Same test, WR-06 gate at :640 | 4 pairs checked, max divergence 5.51e-8 (> 0.0) | PASS |
| softmax_f64 diverges from collapsed-f32 | `cargo test -p treelite-gtil softmax_f64_diverges` | max_rel > 1e-9, any_bit_diff = true | PASS |
| sigmoid_f64 diverges from collapsed-f32 | `cargo test -p treelite-gtil sigmoid_f64_diverges` | max_rel > 1e-8, any_bit_diff = true | PASS |
| UnrecognizedOperator fires on kNone | `knone_operator_on_numerical_node_is_typed_error` | ok | PASS |
| MalformedCategoryList fires | `malformed_category_list_inverted_offsets_is_typed_error` + `malformed_category_list_out_of_range_end_is_typed_error` | ok | PASS |
| MalformedLeafVector fires | `malformed_leaf_vector_inverted_offsets_is_typed_error` | ok | PASS |
| Clippy clean on treelite-gtil | `cargo clippy -p treelite-gtil --all-targets` | 0 errors, 0 warnings | PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| GTIL-01 | 05-02, 05-05 | Dense row-major predict | SATISFIED | `predict<O>` + `predict_preset` wired; harness passes |
| GTIL-02 | 05-04, 05-07 | Sparse CSR predict (absent=NaN) | SATISFIED | `predict_sparse<O>` + `SparseCsr` accessor; absent entries materialized as `O::nan()`; frozen-CSR path in harness |
| GTIL-03 | 05-02, 05-04 | All 4 predict kinds | SATISFIED | `PredictKind` enum; `predict_rows` dispatches Default/Raw/LeafId/ScorePerTree; 24 fixtures per kind |
| GTIL-04 | 05-03, 05-06, 05-07 | All 10 postprocessors, f64 precision | SATISFIED | 10 f32 fns + f64 twins; `softmax_f64` CR-01 fix in 05-REVIEW-FIX; harness passes all |
| GTIL-05 | 05-03 | NaN-only missing-value routing | SATISFIED | `next_node` checks `PredictOut::is_nan_val(fvalue)` → default child; categorical guard rejects NaN via `< 0.0` |
| GTIL-06 | 05-03 | Categorical float-representability guard + child polarity | SATISFIED | `PredictOut::category_match` with `MANTISSA_BITS` (24 f32, 53 f64); polarity via `right_child` inversion; test `categorical_full_guard_rejects_f32_value_past_mantissa_limit` passes |
| GTIL-07 | 05-02, 05-06 | Output shaping, leaf-vector broadcast, tree averaging, f64 base-score | SATISFIED | `output_shape` per kind; `output_leaf_vector` broadcast; `average_tree_output` flag; `PredictOut::add_base_score` in f64; leaf_vec_mc fixtures confirm [140,1,4] shape |
| GTIL-08 | 05-02 | Serial tree_id summation (parallelism only across rows) | SATISFIED | `predict_preset` / `predict_score_by_tree` inner loops are serial over trees per row; no atomics |
| EQV-01 | 05-01, 05-07 | Seeded dense + sparse inputs | SATISFIED | 96 fixtures (both seeds × both layouts); `capture_gtil_matrix.py` uses `np.random.RandomState(seed)` |
| EQV-02 | 05-01, 05-07 | Golden vectors from C++ Treelite, committed with manifest | SATISFIED | All 96 goldens have `backend: scalar-cpu`, `os`/`arch`/`seed`/sha256; captured via `uv run python` on main tree |
| EQV-03 | 05-05, 05-07 | Rust predictions within 1e-5 | SATISFIED | `approx::assert_abs_diff_eq!(epsilon = 1e-5)` hard gate; 96 cells pass; global max 2.91e-6 |
| EQV-04 | 05-05 | Max observed deviation reported | SATISFIED | `eprintln!` per cell; per-cell `max_dev`; CR-01 cell tracking; WR-06 divergence tracking; `global_max_dev` summary |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `error.rs` | 93-100 | `UnsupportedPredictKind` variant — dead code (IN-01) | INFO | No correctness impact; doc comment references closed Plan 05-04. User-deferred. |
| `gtil_matrix.rs` | 540-594 | Comments overstate that the 1e-5 gate would catch a CR-01 reversion (WR-01 review note) | INFO | Functionally correct; comments are inaccurate about what the gate proves vs WR-06. User-deferred. |
| `gtil_matrix.rs` | 628-636 | WR-06 divergence guard is strict bit-inequality (`> 0.0`), not a relative-magnitude floor (WR-02 review note) | WARNING | The guard correctly catches a silent f32→f64 collapse on current fixtures but could be fragile for future fixtures where f32/f64 sigmoid round identically on every row. User-deferred. |

No TBD/FIXME/XXX debt markers found in any phase-5 modified file. No unlinked HACK/PLACEHOLDER markers found.

### Human Verification Required

#### 1. WR-06 Divergence Guard Strength

**Test:** Review `crates/treelite-harness/tests/gtil_matrix.rs:628-636` and decide whether to replace `max_div > 0.0` with a relative-magnitude floor (`max_rel > 1e-9`) before Phase 6 merges a cubecl backend.
**Expected:** Either (a) accept the current bit-inequality guard for now and plan a fixture-independent enforcement mechanism before Phase 6, or (b) adopt the `max_rel > 1e-9` threshold now.
**Why human:** This is a design tradeoff between brittleness of the current guard and the complexity of a per-row relative-divergence assertion. The guard is directionally correct but fragile per the code-review analysis. No code-correctness defect; purely a future-proofing decision.

#### 2. WR-01 Comment Accuracy in gtil_matrix.rs

**Test:** Review the comments at `crates/treelite-harness/tests/gtil_matrix.rs:540-594` (the "would have caught the pre-05-06 collapse-to-f32" claim) and decide whether to reword before the phase is formally closed.
**Expected:** Either (a) reword to state plainly that the 1e-5 gate confirms correctness but does not by itself reject a collapsed-f32 path (the real guard is WR-06), or (b) document explicitly in a code comment that the gate is paired with WR-06 for completeness.
**Why human:** Functionally the test is correct and passes. Whether the comment accuracy matters before Phase 6 is a documentation call.

#### 3. IN-01: UnsupportedPredictKind Dead Variant

**Test:** Review `crates/treelite-gtil/src/error.rs:93-100`. The `UnsupportedPredictKind` variant is never constructed by any reachable code path (all 4 predict kinds are now wired). The stale doc comment references Plan 05-04.
**Expected:** Either remove the variant and its doc comment, or update the comment to mark it as a forward-compat placeholder with no active caller.
**Why human:** No correctness impact. Removing a public enum variant is a semver-relevant call; updating the doc is trivial but requires a human decision on which action to take.

---

## Gaps Summary

No blocking gaps found. All 4 ROADMAP success criteria are verified by the running codebase. The full test suite is green (258 tests, 0 failures). The CR-01 f64-softmax blocker identified in 05-REVIEW.md has been fixed by commits `1e35209` and `6c31263` per 05-REVIEW-FIX.md, verified against the actual `softmax_f64` implementation in `postprocessor.rs`.

Three items remain as human decisions (all user-deferred per 05-REVIEW-FIX.md):
- WR-06 bit-inequality guard strength vs. relative-magnitude floor
- WR-01 comment accuracy around what the 1e-5 gate proves
- IN-01 dead `UnsupportedPredictKind` variant cleanup

These do not block phase goal achievement — the 1e-5 equivalence contract is met, the CR-01 precision defect is closed, all postprocessors and predict kinds are wired, and the harness is green.

---

_Verified: 2026-06-10T12:00:00Z_
_Verifier: Claude (gsd-verifier)_

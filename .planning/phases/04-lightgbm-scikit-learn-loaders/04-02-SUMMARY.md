---
phase: 04-lightgbm-scikit-learn-loaders
plan: 02
subsystem: gtil
tags: [gtil, postprocessor, multiclass, leaf-vector, averaging, base-score]
requires:
  - "treelite_core::Model metadata (target_id, class_id, num_class, leaf_vector_shape, base_scores, average_tree_output, ratio_c)"
  - "treelite-gtil Phase-1 scalar predict (evaluate_tree, PredictScalar, serial tree-sum)"
provides:
  - "GTIL (num_row, num_target, max_num_class) output shaping routed by target_id/class_id"
  - "RF tree averaging (average_tree_output) and f64 per-(target,class) base-score add"
  - "softmax / exponential_standard_ratio / exponential / logarithm_one_plus_exp postprocessors"
affects:
  - "all callers of treelite_gtil::predict (return is now a flat (num_row, num_target, max_num_class) buffer; binary stays (num_row,1,1))"
tech-stack:
  added: []
  patterns:
    - "Flat row-major Array3DView buffer (num_row*num_target*max_num_class) instead of nested Vecs"
    - "Bounds-checked output routing -> typed GtilError (never OOB write), mirroring ERR-01"
    - "Verbatim upstream cast ordering (f32 exp / f64 norm_const accumulate / f32 divide) as the 1e-5 contract"
key-files:
  created:
    - "crates/treelite-gtil/tests/output_shaping.rs"
  modified:
    - "crates/treelite-gtil/src/postprocessor.rs"
    - "crates/treelite-gtil/src/lib.rs"
    - "crates/treelite-gtil/src/error.rs"
    - "crates/treelite-gtil/tests/predict.rs"
decisions:
  - "predict returns a FLAT Vec<f32> of length num_row*num_target*max_num_class (the Array3DView storage); binary (num_row,1,1) stays length num_row, byte-identical to Phase 1 — no new public entry-point name (D per plan)"
  - "softmax/exponential/exp_standard_ratio/log1p_exp ported; signed_square/hinge/identity_multiclass/multiclass_ova deferred to Phase 5 (no multiclassova fixture captured yet)"
  - "has_leaf_vector made bounds-safe in the gtil layer (returns false on absent/short CSR offset columns) so malformed/hand-crafted scalar trees never panic (ERR-01)"
metrics:
  duration: 5min
  completed: 2026-06-10
---

# Phase 4 Plan 02: Minimal GTIL Output Shaping + Postprocessors Summary

Widened `treelite-gtil::predict` from scalar `(num_row,1,1)` to the full
`(num_row, num_target, max_num_class)` output routed by `target_id`/`class_id`,
with RF tree averaging and the f64 per-`(target,class)` base-score add, and
ported four new postprocessors (softmax, exponential, exponential_standard_ratio,
logarithm_one_plus_exp) with verbatim upstream cast ordering — the second Wave-1
1e-5 gate (parallel to the f64 builder), preserving the Phase-1 binary path
byte-for-byte.

## What Was Built

### Task 1 — Four postprocessors (`postprocessor.rs`, commit `5da82ef`)
- `exponential(v)` = `v.exp()`.
- `exponential_standard_ratio(ratio_c, v)` = `(-v / ratio_c).exp2()` — base-**2**
  (`std::exp2`, not `exp`), `ratio_c` threaded through as a `f32` argument.
- `logarithm_one_plus_exp(v)` = `v.exp().ln_1p()` (Rust's `ln_1p` = `std::log1p`).
- `softmax(row: &mut [f32])` — row-wise: `f32` max-subtraction, `f64` `norm_const`
  accumulate (each `f32 t` promoted on add-into), final divide by `norm_const as f32`.
  This mixed f32/f64 reduction order is the 1e-5 contract, ported verbatim from
  `postprocessor.cc:57-75`.
- 5 unit tests against hand-computed references (1e-6/1e-7), incl. empty-row no-op.

### Task 2 — Widened predict (`lib.rs`/`error.rs`/tests, commit `b7fbf98`)
- New `Shape` helper holds `(num_target, max_num_class, num_class, target_id,
  class_id, average_tree_output, base_scores)` and computes flat 3D indices.
- `predict_preset` now accumulates via the four-way `OutputLeafValue` /
  `OutputLeafVector` branch on `(target_id[tree]==-1, class_id[tree]==-1)`
  (`predict.cc:174-229`): both `-1` → leaf-vector broadcast across all cells (RF);
  `class_id>=0` → route into that class column (round-robin multiclass);
  `target_id>=0`/`class_id>=0` → single-cell. Serial tree-sum in `tree_id` order
  (GTIL-08) and the `T::to_f32()` leaf cast are preserved unchanged.
- RF averaging: per-`(target,class)` `average_factor` tree count, then divide
  each cell (`predict.cc:259-293`).
- f64 base-score 2D add per cell with `(acc as f64 + base) as f32` (`:294-304`).
- `apply_postprocessor` dispatches scalar postprocessors per cell and `softmax`
  row-wise over each `(row, target)`'s `num_class` contiguous cells (`:307-323`).
- Two new typed errors (`OutputRouteOutOfBounds`, `LeafVectorTooShort`) so a
  malformed route or short leaf vector never produces an OOB write (T-04-03).
- `tests/output_shaping.rs`: round-robin class routing (distinct columns, no
  collapse), leaf-vector broadcast, RF mean-not-sum, 2D base-score add, scalar
  shape unchanged, softmax normalization, OOB-route typed error.

## How It Differs From Plan

Output return shape: the plan allowed the signature to "gain a shaped-output
return — note the shape change". Chosen representation is the flat row-major
`Array3DView` buffer (`Vec<f32>` of `num_row*num_target*max_num_class`), exactly
mirroring upstream's contiguous `output` array. The binary case
(`num_target=1, max_num_class=1`) yields length `num_row` — byte-identical to the
Phase-1 return — so no existing caller (xgboost json/ubjson tests, harness,
three_format_equivalence) changed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `model_of` test helper under-sized `target_id`/`class_id`**
- **Found during:** Task 2 (workspace regression run)
- **Issue:** `tests/predict.rs::model_of` hard-coded `target_id=vec![0]`,
  `class_id=vec![0]` (length 1). The widened predict reads `target_id[tree_id]`
  per tree; the multi-tree `two_trees_sum_serially` test's second tree fell off
  the end, defaulted to `(-1,-1)`, and hit the leaf-vector broadcast path —
  producing `OutputRouteOutOfBounds` instead of summing into cell 0.
- **Fix:** Size both arrays to the tree count (`vec![0; num_tree]`) so every tree
  routes to `(target=0, class=0)`.
- **Files modified:** crates/treelite-gtil/tests/predict.rs
- **Commit:** b7fbf98

**2. [Rule 1 - Bug] `unsupported_postprocessor_is_typed_error` probed `softmax`**
- **Found during:** Task 2
- **Issue:** The existing test asserted `softmax` was unsupported; Plan 04-02
  added softmax, so the probe was no longer valid.
- **Fix:** Switched the probe to `hinge` (a real upstream postprocessor still
  deferred to Phase 5).
- **Files modified:** crates/treelite-gtil/tests/predict.rs
- **Commit:** b7fbf98

**3. [Rule 2 - Missing safety] Bounds-safe `has_leaf_vector` in the gtil layer**
- **Found during:** Task 2
- **Issue:** `Tree::has_leaf_vector(nid)` indexes `leaf_vector_begin[nid]`
  directly; the widened predict calls it for every tree/leaf. Hand-crafted scalar
  trees (and any malformed model) with empty CSR offset columns would panic on
  the OOB index.
- **Fix:** Added a local `has_leaf_vector(tree, leaf)` that uses
  `as_slice().get(leaf)` and treats absent/short columns as "no leaf vector"
  (falls through to the scalar `OutputLeafValue` path) — consistent with ERR-01
  (never panic on a malformed `Model`).
- **Files modified:** crates/treelite-gtil/src/lib.rs
- **Commit:** b7fbf98

## Verification

- `cargo test -p treelite-gtil` — green: 5 lib (postprocessor) + 7 output_shaping
  + 5 postprocessor (integration) + 11 predict = all pass.
- `cargo test --workspace` — fully green; Phase-1 binary 1e-5 scalar golden and
  three-format equivalence unchanged (no regression; GTIL-08 serial sum preserved).
- `cargo clippy -p treelite-gtil` — no warnings.
- Source assertions: `postprocessor.rs` contains `exp2`, `softmax`,
  `exponential_standard_ratio`, `exponential`, `logarithm_one_plus_exp`;
  `lib.rs` references `average_tree_output` and `class_id` in the shaping path.
- No `panic!`/`anyhow` added (grep clean in `src/`).

## Threat Model Outcome

- **T-04-03 (DoS via output-buffer indexing):** mitigated — `class_id`/`target_id`
  bounds-checked against `max_num_class`/`num_target` before every cell write;
  out-of-range routing returns `GtilError::OutputRouteOutOfBounds`, never an OOB
  write/panic. Covered by `out_of_range_class_route_is_typed_error`.
- **T-04-04 (softmax norm_const division):** accepted — a degenerate all-`-inf`
  row yields NaN, matching upstream; not adversarially reachable from a
  well-formed loaded model.
- **T-04-SC (package installs):** N/A — no new dependencies added.

## Notes for Downstream Plans

- `predict` now returns a flat `(num_row, num_target, max_num_class)` buffer.
  Estimator-slice golden asserts (Plans 03/05/06/08) should index it as
  `out[row*nt*mc + t*mc + c]`. Binary fixtures keep length `num_row`.
- Categorical traversal (`NextNodeCategorical`) for HistGB/LGB is NOT in this
  plan — it lands in the categorical slices (Plans 05/08). This plan covers
  numerical traversal shaping only.
- Deferred postprocessors (Phase 5 complete GTIL surface): `signed_square`,
  `hinge`, `identity_multiclass`, `multiclass_ova` (the latter pending a
  `multiclassova` fixture).

## Self-Check: PASSED

All created/modified files exist on disk; both task commits (5da82ef, b7fbf98) present in git log.

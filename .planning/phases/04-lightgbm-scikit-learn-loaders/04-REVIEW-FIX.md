---
phase: 04-lightgbm-scikit-learn-loaders
fixed_at: 2026-06-10T00:00:00Z
review_path: .planning/phases/04-lightgbm-scikit-learn-loaders/04-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 4: Code Review Fix Report

**Fixed at:** 2026-06-10
**Source review:** .planning/phases/04-lightgbm-scikit-learn-loaders/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (Warnings WR-01..WR-05; Info IN-01..IN-06 out of scope)
- Fixed: 5
- Skipped: 0

All Critical+Warning findings were fixed and committed atomically. The full
`cargo test --workspace` suite passes with ZERO failures (all golden harness
tests included: lightgbm_numerical, lightgbm_categorical, sklearn_rf,
sklearn_gb, sklearn_iforest, sklearn_histgb_numerical,
sklearn_histgb_categorical). `cargo clippy --workspace` is clean.

## Fixed Issues

### WR-01: HistGB leaf detection diverges from upstream for `node.left` in `[2^31, 2^32)`

**Files modified:** `crates/treelite-sklearn/src/histgb.rs`
**Commit:** 8140a23
**Applied fix:** Replaced the raw `node.left == 0` (u32) leaf test with upstream's
signed-cast semantics: `let left_child_id = node.left as i32; ... if
left_child_id <= 0`. This matches `sklearn.cc:317,320`
(`static_cast<int>(node.left) <= 0`). A `node.left` in `[2^31, 2^32)` now
reinterprets as negative and is treated as a LEAF, matching upstream's load
outcome. The internal-node branch reuses the new `left_child_id` /
`right_child_id` i32 bindings.

### WR-02: `multiclass` / `multiclassova` objective skips upstream `num_class` parameter validation

**Files modified:** `crates/treelite-lightgbm/src/objective.rs`,
`crates/treelite-lightgbm/src/error.rs`, `crates/treelite-lightgbm/src/lib.rs`
**Commit:** 99a262f
**Applied fix:** Threaded the global `num_class` into `map_objective(canonical,
obj_param, num_class)`. Added `parse_num_class` (mirrors the `Split(str, ':')`
token loop with the `>= 0` filter at `lightgbm.cc:449-455`/`466-470`) and
`require_matching_num_class`, which rejects a multiclass/multiclassova model
whose embedded `num_class:<n>` token is missing or `!= num_class` — porting
`TREELITE_CHECK(num_class >= 0 && num_class == num_class_)`
(`lightgbm.cc:457,476`). Added a typed `LgbError::InvalidNumClass` variant and
updated the call site plus all `map_objective` unit tests; added coverage for
the new mismatch/missing cases.

### WR-03: Builder accepts negative `split_index`; only caught downstream in GTIL

**Files modified:** `crates/treelite-builder/src/lib.rs`
**Commit:** 5014366
**Applied fix:** Extended the existing metadata-gated check in
`validate_test_children` from `split_index >= meta.num_feature` to `split_index
< 0 || split_index >= meta.num_feature`, reusing the existing
`SplitIndexOutOfRange` variant. Upstream's `TREELITE_CHECK_LT(split_index,
num_feature)` (`model_builder.cc:181-182`) lets a negative signed `split_index`
pass; the builder now rejects it at load time as defense-in-depth. Kept the
`metadata_initialized_`-equivalent gate to mirror upstream.

### WR-04: LightGBM `base_scores` length diverges from upstream when `num_class == 0`

**Files modified:** `crates/treelite-lightgbm/src/lib.rs`
**Commit:** f3df065
**Applied fix:** Dropped the `.max(1)` clamp on `base_scores`:
`vec![0.0; num_class as usize]`, matching upstream's unclamped
`std::vector<double>(num_class_, 0.0)` (`lightgbm.cc:523`). `base_scores.len()`
now tracks `num_class` exactly, keeping it consistent with `metadata.num_class
== [num_class]` even in the reachable degenerate `num_class == 0` case. The
`class_id` modulo retains its own `.max(1)` guard (a separate divide-by-zero
guard, unrelated to shape). Well-formed models (`num_class >= 1`) are unchanged.

### WR-05: `bulk_construct_tree` gain divides by per-node `sample_cnt` with no zero guard

**Files modified:** `crates/treelite-builder/src/bulk.rs`,
`crates/treelite-sklearn/src/mixin.rs`
**Commit:** 8fe5fd4
**Applied fix:** Guarded both gain divisors (`sample_cnt`/`sc` and
`total_sample_cnt`) to yield `0.0` when either is zero, in both the bulk path
(`bulk.rs:152-166`) and the per-node mixin path (`mixin.rs:157-167`). This is a
metadata-only hardening: a well-formed sklearn internal node always has positive
sample counts, so the computed gain for real models is byte-identical, and gain
never enters the prediction path — the 1e-5 fidelity contract is unaffected. The
guard only avoids writing NaN/inf into the `gain` column for a crafted
zero-sample array reaching the documented bulk "validation bypass". Verified
against all sklearn golden harness tests (rf/gb/iforest/histgb), which still
match upstream within 1e-5.

## Skipped Issues

None — all 5 in-scope Warning findings were fixed.

(Info findings IN-01..IN-06 were out of scope for this `critical_and_warning`
run and were not attempted.)

---

_Fixed: 2026-06-10_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_

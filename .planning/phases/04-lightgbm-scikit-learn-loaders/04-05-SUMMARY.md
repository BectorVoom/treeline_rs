---
phase: 04-lightgbm-scikit-learn-loaders
plan: 05
subsystem: model-loader
tags: [lightgbm, categorical, bitset, BitsetToList, gtil, NextNodeCategorical, 1e-5-golden, LGB-02]

# Dependency graph
requires:
  - phase: 04-04
    provides: "treelite-lightgbm crate (parser stores cat_boundaries u64 / cat_threshold u32; build_tree converge-then-build path; categorical branch stubbed with a typed rejection)"
  - phase: 04-02
    provides: "treelite-gtil::predict evaluate_tree numerical NextNode traversal"
  - phase: 04-03
    provides: "frozen fixtures/lightgbm_categorical.txt + fixtures/lightgbm_categorical.golden.json (upstream treelite.gtil.predict, max_cat_to_onehot=1 forced bitset splits)"
  - phase: 04-01
    provides: "f64 ModelBuilder mode"
provides:
  - "BitsetToList categorical bitset decoder (bitset.rs) — word/bit order ported verbatim (lightgbm.cc:210-221)"
  - "treelite-builder categorical_test extended to carry the category list + category_list_right_child polarity; CSR category_list/begin/end columns filled at end_tree"
  - "LightGBM categorical-split emission in load_lightgbm (cat_idx = threshold[node], slice via cat_boundaries, decode via BitsetToList, default_left=false)"
  - "minimal NextNodeCategorical GTIL traversal branch (integer-category membership + polarity)"
  - "lightgbm_categorical 1e-5 golden harness test (max |delta| = 9.54e-7)"
affects: [04-08-histgb-categorical, phase-5-categorical-gtil]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "bitset.rs decodes categorical splits via BitsetToList ported VERBATIM (word = bits[i/32], bit = i%32, LSB-first); NOT shared with the HistGB check(bitmap,...) sibling (different layout)"
    - "categorical per-field precision: cat_threshold u32, cat_boundaries u64; cat_threshold.len() == cat_boundaries.back() validated EXACTLY (not >=) + monotone boundaries → typed LgbError::Bitset before any slicing (T-04-10)"
    - "builder categorical_test is mode-agnostic (sets no leaf/threshold value, no f32/f64 latch) so the f64 LightGBM path uses it directly; the CSR category_list value buffer is built from per-node staging at end_tree, empty for non-categorical trees (matches upstream AllocNode defaults)"

key-files:
  created:
    - crates/treelite-lightgbm/src/bitset.rs
  modified:
    - crates/treelite-lightgbm/src/parse.rs
    - crates/treelite-lightgbm/src/lib.rs
    - crates/treelite-lightgbm/src/error.rs
    - crates/treelite-builder/src/lib.rs
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/tests/predict.rs
    - crates/treelite-harness/tests/lightgbm.rs

key-decisions:
  - "BitsetToList ported verbatim (lightgbm.cc:210-221): bit i lives in word bits[i/32] at position i%32 (LSB-first); the decoder takes bits: &[u32] and derives nslots = bits.len() so the word index is structurally in-bounds (T-04-11 — no path reads past the bitset word array)."
  - "For a categorical node the LightGBM `threshold` field is REPURPOSED as the categorical index: cat_idx = static_cast<int>(threshold[node]) (lightgbm.cc:565). The slice cat_threshold[cat_boundaries[cat_idx]..cat_boundaries[cat_idx+1]] is decoded; cat_idx and both boundary indices are bounds-checked before use → typed LgbError::Bitset (T-04-10)."
  - "Categorical splits ignore the missing_type field: default_left = false and category_list_right_child = false (a category MATCH routes LEFT to left_categories), NaN always → right (lightgbm.cc:569-573)."
  - "treelite-builder categorical_test signature extended to (split_index, default_left, categories: &[u32], category_list_right_child, left_child_key, right_child_key) — matching upstream SetCategoricalTest (detail/tree.h:138-155); guard_f32() removed because a categorical node sets no typed value, so it stays mode-agnostic and is usable from the f64 LightGBM path. end_tree now flattens per-node category staging into the CSR category_list / category_list_begin / category_list_end / category_list_right_child columns (empty/zero-offset for non-categorical trees, preserving the CR-01 column shape)."
  - "Minimal NextNodeCategorical (D-03): integer-category membership + category_list_right_child polarity ported (predict.cc:128-150). The load-bearing subset of the float-representability guard (reject negative/non-finite/>u32::MAX) is applied so the as-u32 cast is well-defined; the EXHAUSTIVE representability matrix (GTIL-06) is explicitly deferred to Phase 5 (noted in a code comment)."
  - "GTIL reads the category list through a bounds-safe category_list_safe wrapper (returns &[] on any out-of-range / inverted CSR slice) so a hand-crafted malformed Model never OOB-panics (T-04-12); an empty list simply makes every category a non-match."

patterns-established:
  - "Categorical bitset decode is a single verbatim port (bitset_to_list) with reference-bitset unit tests pinning the exact word/bit order; reused by future categorical loaders only if the layout matches (HistGB does NOT)."

requirements-completed: [LGB-02]

# Metrics
duration: 5min
completed: 2026-06-10
---

# Phase 4 Plan 5: LightGBM Categorical Bitset Decode (LGB-02) Summary

**LightGBM categorical splits now decode from their bitset into exact category lists via a verbatim `BitsetToList` port, emit through the f64 `ModelBuilder`'s extended `categorical_test`, and traverse via a minimal `NextNodeCategorical` GTIL branch — a categorical LightGBM model loads → predicts → matches its upstream treelite-GTIL golden at max |delta| = 9.54e-7 < 1e-5.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-10T04:55:07Z
- **Completed:** 2026-06-10T05:00Z
- **Tasks:** 2 completed
- **Files modified:** 8 (1 created, 7 modified)

## Accomplishments
- New `crates/treelite-lightgbm/src/bitset.rs` ports `BitsetToList` VERBATIM from `lightgbm.cc:210-221` (word = `bits[i/32]`, bit = `i%32`, LSB-first), with 6 reference-bitset unit tests pinning the exact decode order.
- Parser hardened for the categorical security domain (T-04-10): `cat_threshold.len() == cat_boundaries.back()` is now validated EXACTLY (token count, not `>=`-truncation) and boundaries must be monotone — a mismatch returns the new `LgbError::Bitset`, never an OOB slice.
- `treelite-builder::categorical_test` extended to carry the category list + polarity (matching upstream `SetCategoricalTest`); `end_tree` flattens per-node category staging into the CSR `category_list`/`category_list_begin`/`category_list_end`/`category_list_right_child` columns. Non-categorical trees keep the prior empty/zero-offset shape (CR-01 column-fidelity test still green).
- `load_lightgbm` emits categorical nodes: `cat_idx = threshold[node]`, slice via `cat_boundaries`, decode via `BitsetToList`, `categorical_test(.., default_left=false, &categories, category_list_right_child=false, ..)`, fully bounds-checked.
- `treelite-gtil::evaluate_tree` gained a minimal `NextNodeCategorical` branch (integer membership + polarity), read through a bounds-safe `category_list_safe` wrapper (T-04-12).
- `lightgbm_categorical` 1e-5 golden harness test passes at **max |delta| = 9.54e-7**; full `cargo test --workspace` green (no XGBoost / sklearn / numerical-LightGBM regression); clippy clean on all changed crates.

## Task Commits

Each task was committed atomically (TDD: tests written alongside implementation, all green at commit time):

1. **Task 1: Port BitsetToList + categorical parse + categorical_test emission** — `29eb3cd` (feat)
2. **Task 2: Minimal NextNodeCategorical GTIL branch + lightgbm_categorical 1e-5 golden test** — `7c8a175` (feat)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP/REQUIREMENTS) committed in the final docs commit.

## Files Created/Modified
- `crates/treelite-lightgbm/src/bitset.rs` (created) — `bitset_to_list(&[u32]) -> Vec<u32>` verbatim port; structurally in-bounds; 6 reference unit tests.
- `crates/treelite-lightgbm/src/parse.rs` (modified) — exact `cat_threshold.len() == cat_boundaries.back()` + monotone-boundary validation → `LgbError::Bitset`; per-field-precision + mismatch + non-monotone unit tests.
- `crates/treelite-lightgbm/src/lib.rs` (modified) — categorical branch in `build_tree` replaced the typed rejection with the bitset decode + `categorical_test` emission (bounds-checked cat_idx/slice); categorical-node + out-of-range-index unit tests.
- `crates/treelite-lightgbm/src/error.rs` (modified) — new `LgbError::Bitset { tree, detail }` variant.
- `crates/treelite-builder/src/lib.rs` (modified) — `NodeStaging` gained `categories`/`category_list_right_child`; `categorical_test` signature extended + made mode-agnostic; `end_tree` builds the CSR category columns.
- `crates/treelite-gtil/src/lib.rs` (modified) — `next_node_categorical` + `category_list_safe`; `evaluate_tree` branches on `kCategoricalTestNode`.
- `crates/treelite-gtil/tests/predict.rs` (modified) — `categorical_tree` helper + two polarity routing tests (match→left, right-child→inverted).
- `crates/treelite-harness/tests/lightgbm.rs` (modified) — `lightgbm_categorical` 1e-5 golden test mirroring `lightgbm_numerical`.

## Decisions Made
See `key-decisions` in the frontmatter. Headline: `BitsetToList` ported verbatim (LSB-first word/bit order); the categorical node's `threshold` field is repurposed as the categorical index; categorical splits ignore missing_type (default_left=false, NaN→right); the builder's `categorical_test` is mode-agnostic so the f64 LightGBM path uses it directly; the minimal `NextNodeCategorical` defers the exhaustive float-representability matrix (GTIL-06) to Phase 5.

## Deviations from Plan

None - plan executed exactly as written.

The plan's `<action>` referenced a `categorical_test(split_index, default_left, categories, right_child, left, right)` builder entry; the existing Plan-02 `categorical_test` did NOT yet take a category list (it stored only the node type, with `end_tree` hard-coding empty CSR columns). Extending the builder to accept and persist the category list + polarity is the necessary plumbing the plan's action describes ("emit `builder.categorical_test(..., &categories, ...)`"), implemented as Rule 2 (missing critical functionality — a categorical node with no stored category list would silently mis-predict). This is the planned scope, recorded here for traceability rather than as a true deviation.

## Issues Encountered
None. All unit tests and the golden passed on the first run after wiring; the categorical golden matched at 9.54e-7 (the f32-quantization floor for these leaf magnitudes), well within the 1e-5 gate.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **LGB-02 closed.** LightGBM (numerical + categorical) is fully loaded and 1e-5-verified. The `treelite-lightgbm` slice is complete for Phase 4.
- **Phase 5 categorical GTIL:** the minimal `next_node_categorical` is the seam for the exhaustive categorical matrix + the full float-representability guard (GTIL-06) — the load-bearing subset is already in place and commented.
- **04-08 HistGB categorical:** will need its OWN bitset decoder (`check(bitmap,...)`, different layout) — `bitset.rs` is intentionally NOT shared (04-PATTERNS No-Analog-Found).
- **No blockers.** The 1e-5 categorical gate protects against regression.

## Known Stubs
None. The prior 04-04 stubs (categorical-branch typed rejection; parsed-but-unconsumed cat fields) are now fully resolved — categorical splits decode, emit, and predict.

## Self-Check: PASSED

All listed files exist on disk and both task commits (`29eb3cd`, `7c8a175`) are present in git history. `cargo test --workspace` is fully green; `lightgbm_categorical` passes at max |delta| = 9.54e-7 < 1e-5.

---
*Phase: 04-lightgbm-scikit-learn-loaders*
*Completed: 2026-06-10*

---
phase: 04-lightgbm-scikit-learn-loaders
plan: 08
subsystem: model-loader
tags: [sklearn, histgradientboosting, packed-node, from_le_bytes, features_map, categories_map, bitset, 1e-5, SKL-04]

# Dependency graph
requires:
  - phase: 04-06
    provides: "treelite-sklearn crate + f64 MixIn ModelBuilder path (BuilderMetadata, numerical_test_f64, leaf_scalar_f64, gain, data_count)"
  - phase: 04-05
    provides: "builder categorical_test (category list + polarity + CSR columns) + GTIL NextNodeCategorical branch"
  - phase: 04-03
    provides: "fixtures/sklearn_histgb_{numerical,categorical}.golden.json (frozen treelite.gtil.predict; packed nodes base64, itemsize 56)"
  - phase: 03-xgboost
    provides: "legacy.rs fallible little-endian byte-cursor discipline (from_le_bytes per field, bounds-checked, no transmute — D-08)"
provides:
  - "treelite-sklearn::histgb — packed HistGradientBoostingNode decode (52/56 itemsize) field-by-field via from_le_bytes, NO transmute/bytemuck (D-08)"
  - "load_hist_gradient_boosting_{regressor,classifier} (D-01 array signatures + packed nodes + features_map + categories_map + baseline)"
  - "features_map always applied to split index; categories_map[fid][cat] categorical remap when present (Pitfall 4)"
  - "HistGB categorical check(bitmap,val,row) decode — 8*row 256-bit stride, NOT shared with LightGBM BitsetToList"
  - "SklError::HistGbDecode { offset, detail } typed error (itemsize/buffer/index guards)"
  - "sklearn_histgb_numerical (max |delta| 0e0) + sklearn_histgb_categorical (max |delta| 1.19e-7) 1e-5 golden gates"
affects:
  - "Phase-8 PyO3 will call load_hist_gradient_boosting_* with zero-copy numpy buffers"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Packed-struct (#pragma pack(1)) byte buffers decoded field-by-field at explicit offsets via from_le_bytes (Phase-3 D-08), NEVER reinterpret/transmute — itemsize selects the 52 (i32 feature_idx) vs 56 (i64 feature_idx) offset table"
    - "Security-domain guards run BEFORE any field read: reject itemsize not in {52,56} (T-04-18), nodes buffer < node_count x itemsize (T-04-18), feature_idx out of features_map/categories_map range (T-04-19), and the full 8-word categorical bitset row out of range (T-04-20) — all typed HistGbDecode, never OOB"
    - "HistGB categorical bitset uses the 8*row (8 uint32 = one 256-bit row) stride; it is a SEPARATE function from LightGBM's bitset_to_list (different layout, RESEARCH No-Analog)"

key-files:
  created:
    - crates/treelite-sklearn/src/histgb.rs
  modified:
    - crates/treelite-sklearn/src/lib.rs
    - crates/treelite-sklearn/src/error.rs
    - crates/treelite-harness/tests/sklearn.rs

key-decisions:
  - "Leaf detection is `left == 0` (HistGB uses 0 for the missing child, NOT the sklearn-tree `== -1` rule, sklearn.cc:320). The packed is_leaf/depth/bin_threshold fields are present but deliberately NOT decoded (unused by the loader)."
  - "split_index = features_map[feature_idx] is ALWAYS applied; feature_idx is bounds-checked against features_map.len() before indexing. The packed feature_idx is the model-internal (categorical-first) ordering; features_map permutes it back (Pitfall 4)."
  - "categories_map keyed by node.feature_idx (the un-remapped index): cat_transform = categories_map[fid][cat] when Some, identity when None (empty in the numerical fixture). Both fid and cat are bounds-checked."
  - "num_threshold read DIRECTLY from the struct (Pitfall 5) — no _bin_mapper reconstruction; known_cat_bitsets is accepted by the upstream signature but UNUSED in v4.7.0 (A3), so the Rust loader does not take/consume it."
  - "Both frozen HistGB fixtures are regressors (baseline_prediction len 1, output last-dim 1); the harness drives load_hist_gradient_boosting_regressor. The classifier entry point (binary sigmoid / multiclass softmax round-robin class_id) is implemented and metadata-correct but not golden-exercised this plan (no classifier fixture)."
  - "The harness base64-decodes nodes_b64 with a tiny self-contained standard-base64 decoder rather than adding a third-party base64 crate — keeps the dependency graph minimal and avoids a package-install gate."

patterns-established:
  - "Loader-crate packed-buffer decode: a NodeLayout offset table parameterized by itemsize + per-field from_le_bytes readers (read_u8/u32/i32/i64/f64) that map a short slice to a typed error — the reusable shape for any future #pragma pack(1) importer."

requirements-completed: [SKL-04]

# Metrics
duration: ~8min
completed: 2026-06-10
---

# Phase 4 Plan 8: HistGradientBoosting Import + 1e-5 Verify (SKL-04) Summary

**The phase tentpole: scikit-learn HistGradientBoosting now loads from its raw packed `HistGradientBoostingNode` byte buffer — decoded field-by-field via `from_le_bytes` at the 52/56-byte layout offsets (Phase-3 D-08, no transmute) — with `features_map` always applied to the split index and `categories_map` remapping categorical bit values, and predicts within 1e-5 of its upstream treelite-GTIL golden (numerical max |delta| = 0e0, categorical max |delta| = 1.19e-7).**

## Performance
- **Duration:** ~8 min
- **Started:** 2026-06-10T05:09:52Z
- **Tasks:** 2 completed
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- New `crates/treelite-sklearn/src/histgb.rs`: a `NodeLayout` offset table parameterized by itemsize (52 = i32 feature_idx, 56 = i64 feature_idx) plus fallible little-endian field readers (`read_u8`/`read_u32`/`read_i32`/`read_i64`/`read_f64`) that decode each packed `HistGradientBoostingNode` field at its explicit byte offset — NEVER a `transmute`/`bytemuck` reinterpret (Phase-3 D-08 ban, grep-clean in code).
- Security-domain guards run BEFORE any field read: itemsize not in {52,56} → typed `SklError::HistGbDecode` (T-04-18); `nodes` buffer shorter than `node_count × itemsize` → rejected up front (T-04-18); `feature_idx` out of `features_map`/`categories_map` range → typed error (T-04-19); the full 8-word (256-bit) categorical bitset row out of range → typed error (T-04-20).
- `split_index = features_map[feature_idx]` is ALWAYS applied (Pitfall 4); leaf detection uses `left == 0` (NOT `== -1`); `num_threshold` is read DIRECTLY from the struct (Pitfall 5, no `_bin_mapper` reconstruction).
- Categorical branch: `check(bitmap, val, row) = (bitmap[8*row + val/32] >> (val%32)) & 1` ported verbatim (the `8*row` 8-uint32 256-bit-row stride is load-bearing and is a SEPARATE function from LightGBM's `bitset_to_list` — different layout, RESEARCH No-Analog); each set bit pushes `categories_map[fid][cat]` (or identity) and emits `categorical_test(.., right_child=false, ..)`.
- `load_hist_gradient_boosting_regressor` (`identity` postprocessor) and `load_hist_gradient_boosting_classifier` (binary `sigmoid` / multiclass `softmax` with round-robin `class_id = tree % n_classes`) added with the D-01 array signatures + packed-node + remap inputs.
- New `SklError::HistGbDecode { offset, detail }` variant; module wired into `lib.rs` (re-exports + doc).
- `crates/treelite-harness/tests/sklearn.rs`: `sklearn_histgb_numerical` (max |delta| = **0e0**) and `sklearn_histgb_categorical` (max |delta| = **1.19e-7**) 1e-5 golden gates, plus a self-contained standard-base64 decoder for the frozen `nodes_b64` buffers.
- 13 `histgb` unit tests pin: the 52/56 field offsets, the i64-feature-idx decode, itemsize + short-buffer rejection, the `left == 0` leaf rule, the `features_map` split-index remap, the `8*row` `check_bit` stride, identity-vs-`categories_map` remap, out-of-range `bitset_idx` rejection, category-membership routing, and the malformed-bitmap typed error.

## Task Commits
Each task was committed atomically (TDD: tests written alongside implementation, all green at commit time):

1. **Task 1: Packed-node decode (52/56) + features_map + numerical loader + golden** — `ce26b05` (feat)
2. **Task 2: HistGB categorical decode (check bitmap + categories_map) + golden** — `4bc83e0` (feat)

Plan metadata (this SUMMARY + STATE/ROADMAP/REQUIREMENTS) committed in the final docs commit.

## Verification
- `cargo test -p treelite-sklearn histgb` — 13 unit tests green (offsets, itemsize/buffer guards, leaf rule, features_map remap, categorical stride/remap/bounds).
- `cargo test -p treelite-harness sklearn_histgb_numerical` — green, max |delta| = **0e0**.
- `cargo test -p treelite-harness sklearn_histgb_categorical` — green, max |delta| = **1.19e-7** (the f32-quantization floor, « 1e-5).
- `cargo test --workspace` — fully green (no XGBoost / LightGBM / sklearn / serializer regression).
- `cargo clippy -p treelite-sklearn` (lib) — clean. `histgb.rs` is clippy-clean under `--tests`.
- grep: `histgb.rs` has no `transmute`/`bytemuck` in code (only in the doc comment explaining the ban); `from_le_bytes` present (6 readers).

## Deviations from Plan
None - plan executed exactly as written.

Two implementation choices worth recording for traceability (both inside planned scope, not true deviations):
- **`known_cat_bitsets` omitted from the Rust signatures.** Upstream `LoadHistGradientBoosting` accepts `raw_left_cat_bitsets` AND `known_cat_bitsets`, but `known_cat_bitsets` is UNUSED in v4.7.0 (RESEARCH A3). The plan's action explicitly notes it is "passed but UNUSED", so the Rust loaders take only `raw_left_cat_bitsets` rather than carrying a dead parameter.
- **Self-contained base64 decoder in the harness.** The frozen `nodes_b64` buffers need base64 decoding; rather than add a third-party `base64` crate (a new dependency / potential package-install gate), a tiny standard-alphabet decoder lives in the test file. Test-only, deterministic.

## Threat Model Coverage
- **T-04-18** (itemsize ∉ {52,56} or short buffer → OOB): `NodeLayout::for_itemsize` rejects bad itemsize; `build_tree` verifies `nodes_bytes.len() >= node_count × itemsize` BEFORE any decode → typed `HistGbDecode` (unit-tested: `histgb_decode_rejects_bad_itemsize`, `histgb_decode_rejects_short_buffer_before_field_read`).
- **T-04-19** (`feature_idx` OOB into `features_map`/`categories_map`): `usize::try_from` + `features_map.get(fid)` / `cm.get(fid)` bounds-check before access → typed error (unit-tested: `histgb_feature_idx_out_of_range_is_typed_error`).
- **T-04-20** (`bitset_idx` OOB into the bitmap): the full 256-bit row `[8*bitset_idx, 8*bitset_idx+8)` is range-checked before the bit scan; `check_bit` itself returns `false` on an out-of-range word → no OOB (unit-tested: `histgb_decode_categorical_rejects_out_of_range_bitset_idx`, `histgb_categorical_malformed_bitmap_is_typed_error_not_panic`).
- **T-04-21** (native-endian transmute UB): decode is field-by-field `from_le_bytes`; NO `transmute`/`bytemuck` in code (grep-clean — the only matches are the doc comment explaining the ban).
- **T-04-SC** (package installs): N/A — no new third-party packages (the harness uses an inline base64 decoder; `treelite-sklearn` is an internal path crate).

## Known Stubs
None. The HistGB classifier entry point is real (binary/multiclass metadata implemented per the upstream MixIns); it is simply not golden-exercised this plan because the frozen fixtures are both regressors. The numerical and categorical regressor paths are fully wired and 1e-5-verified.

## Next Phase Readiness
- **SKL-04 closed.** HistGradientBoosting (numerical + categorical) loads → predicts → 1e-5-verified. The `treelite-sklearn` slice (RF/ET, GB, IsolationForest, HistGB) is complete — **Phase 4 is complete**.
- **Phase-8 PyO3:** the D-01 array surface for HistGB (`load_hist_gradient_boosting_*`) is ready for zero-copy numpy buffers; the packed `nodes` come straight from `sklearn._predictors[*].nodes`.
- **No blockers.** The two 1e-5 HistGB gates protect against regression.

## Self-Check: PASSED
- Created file exists on disk: `crates/treelite-sklearn/src/histgb.rs` — FOUND.
- Both task commits present in git history: `ce26b05` (Task 1), `4bc83e0` (Task 2) — FOUND.
- `cargo test --workspace` fully green; both HistGB goldens within 1e-5 (numerical 0e0, categorical 1.19e-7).

---
*Phase: 04-lightgbm-scikit-learn-loaders*
*Completed: 2026-06-10*

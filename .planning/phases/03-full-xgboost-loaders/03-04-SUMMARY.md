---
phase: 03-full-xgboost-loaders
plan: 04
subsystem: model-loader
tags: [xgboost, legacy-binary, from_le_bytes, byte-fidelity, gtil, serializer, def-02-01]

# Dependency graph
requires:
  - phase: 03-03
    provides: "shared build_model_from_parsed convergence path; XgbModelJson struct family; de_f32 sentinel adapter; load_xgboost_ubjson + detect_xgboost_format"
  - phase: 03-02
    provides: "JSON D-10 close-out (sum_hess/gain emission, attributes=None) so the loader path reproduces the upstream golden"
provides:
  - "load_xgboost_legacy: XGBoost legacy-binary loader via explicit little-endian byte cursor (D-07/D-08, no native-endian struct reinterpret)"
  - "XgbModelJson::from_legacy_fields + RegTreeJson::from_legacy_nodes — direct constructors that funnel legacy into the shared build path"
  - "XgbError::Legacy variant (truncation / bs64 / struct-size / bad-booster)"
  - "DEF-02-01 / D-10 closed across all three formats: single-golden cross-format byte-fidelity assertion is fatal and green"
  - "Full three_format_equivalence suite green; cargo test --workspace green (no remaining RED targets)"
affects: [lightgbm-loader, sklearn-loader, c-api, python-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-rolled little-endian byte cursor (from_le_bytes) with Option-bounded reads — zero external deps (D-07)"
    - "Legacy loader converges at build_model_from_parsed by filling XgbModelJson directly (D-01) instead of a second build path"
    - "version-gate mapping: major_version → version=[major_version] reproduces the JSON gate (version[0]>=1) exactly"

key-files:
  created:
    - crates/treelite-xgboost/src/legacy.rs
    - crates/treelite-xgboost/tests/legacy.rs
  modified:
    - crates/treelite-xgboost/src/lib.rs
    - crates/treelite-xgboost/src/error.rs
    - crates/treelite-xgboost/src/json.rs
    - crates/treelite-harness/tests/golden_v5.rs

key-decisions:
  - "GBTreeModelParam is 160 bytes, not 168 — the RESEARCH table had a transcription error; confirmed against mushroom.model (173 + 160 + 1168 == 1501)"
  - "Legacy converges via XgbModelJson::from_legacy_fields (scalars formatted back to strings; f32 base_score round-trips losslessly) rather than a parallel build path, guaranteeing an identical Model to JSON/UBJSON"
  - "Promoted golden_v5.rs loader diagnostic to a fatal assertion since 03-02 closed DEF-02-01 (loader path now byte-identical to golden_v5.bin)"

patterns-established:
  - "Pattern: legacy byte cursor — Cursor{buf,pos} with buf.get(pos..pos+N)->Option, from_le_bytes decode, length/count validation before allocation (T-03-L01..L06)"
  - "Pattern: format-convergence — every XGBoost on-disk format fills the same XgbModelJson and funnels through one build path (D-01)"

requirements-completed: [XGB-03, XGB-05]

# Metrics
duration: 30min
completed: 2026-06-10
---

# Phase 3 Plan 4: XGBoost Legacy-Binary Loader + Cross-Format Close Summary

**The XGBoost legacy-binary format loads via an explicit little-endian byte cursor (no native-endian struct reinterpret), converges at the shared build path to produce a Model byte-identical to the JSON/UBJSON loads, and closes DEF-02-01/D-10 across all three formats — the entire three-format equivalence suite and the full workspace are green.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2 completed
- **Files modified:** 6 (2 created, 4 modified)

## Accomplishments

- **XGB-03 (legacy binary):** `load_xgboost_legacy` ports upstream `ParseStream` onto a fallible `from_le_bytes` cursor + a `PeekableReader` for the `binf`/`bs64` magic peek. Decodes `LearnerModelParam`(136) → length-prefixed objective/booster names → `GBTreeModelParam`(160) → per-tree `TreeParam`(148) + 20-byte Nodes + 16-byte NodeStats + conditional leaf-vector tail → `tree_info` → optional DART `weight_drop`. sindex bit-unpacking (`& 0x7FFFFFFF` / `>> 31`), `cleft == -1` leaf detection, and the `info` union f32 reinterpretation are all ported exactly (Pitfall 6).
- **XGB-05 (legacy leg):** version-gated base_score→margin transform reproduced by mapping `major_version` into `version = [major_version]` so the shared path's gate (`version[0] >= 1`) fires iff `major_version >= 1`. Verified with ± tests.
- **DEF-02-01 / D-10 closed across all three formats:** `three_format_serialize_byte_fidelity` asserts `serialize(load_json) == serialize(load_ubjson) == serialize(load_legacy) == golden_v5_3format.bin`, and `three_format_predicts_within_1e5` holds for all three loaders. The `golden_v5.rs` loader diagnostic was promoted from a non-fatal `println!` to a hard `assert_eq!`.
- **Convergence guaranteed:** legacy fills the same `XgbModelJson`/`RegTreeJson` structs the JSON/UBJSON paths fill and funnels through `build_model_from_parsed`, so the three loaders produce the IDENTICAL Model — proven by the single-golden cross-format assertion.

## Task Commits

1. **Task 1: Legacy LE byte cursor + ParseStream port** — `310fefd` (feat)
2. **Task 2: Close DEF-02-01/D-10 — promote loader diagnostic to fatal assertion** — `dd45d05` (test)

_Task 1 was developed test-first (legacy.rs tests written alongside the loader); committed as a single feat since the test file is part of the same artifact set._

## Files Created/Modified

- `crates/treelite-xgboost/src/legacy.rs` (created, ~535 lines) — little-endian `Cursor`, `PeekableReader`, `ParseStream` port, `load_xgboost_legacy`. No native-endian struct reinterpret; 6 `from_le_bytes` call sites.
- `crates/treelite-xgboost/tests/legacy.rs` (created) — mushroom smoke (1501 B, 2 trees 13/11, binary:logistic, 127 features), sindex unpack, version gate ±, truncation/empty-buffer typed errors, bs64-reject/binf-consume. 8 tests, all green.
- `crates/treelite-xgboost/src/lib.rs` (modified) — `mod legacy;` + `pub use legacy::load_xgboost_legacy;`.
- `crates/treelite-xgboost/src/error.rs` (modified) — added `XgbError::Legacy { pos, detail }`.
- `crates/treelite-xgboost/src/json.rs` (modified) — added `XgbModelJson::from_legacy_fields` and `RegTreeJson::from_legacy_nodes` crate-internal constructors.
- `crates/treelite-harness/tests/golden_v5.rs` (modified) — `loader_path_divergence_diagnostic` → fatal `loader_path_reproduces_golden_v5_byte_for_byte`.

## Verification

- `cargo test -p treelite-xgboost legacy_` → 8/8 green.
- `cargo test -p treelite-harness --test three_format_equivalence` → 2/2 green (both formerly-RED tests now pass).
- `cargo test -p treelite-harness --test golden_v5` → 2/2 green (round-trip + promoted loader assertion).
- `cargo test --workspace` → fully green, no failures, no RED targets.
- `cargo clippy -p treelite-xgboost --all-targets` → clean (0 warnings).
- Grep gates: `grep -nE 'transmute|bytemuck::(cast|from_bytes|pod_read)' crates/treelite-xgboost/src/legacy.rs` → empty; `grep -c 'from_le_bytes' …/legacy.rs` → 6 (≥4).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] GBTreeModelParam size is 160 bytes, not 168**
- **Found during:** Task 1 (mushroom smoke test failed with "GBTreeModelParam size mismatch: consumed 160, expected 168").
- **Issue:** The PLAN/RESEARCH §"Legacy Binary Layout" listed `GBTreeModelParam` as 168 bytes. The upstream struct (`xgboost_legacy.cc:169-178`) is `4×i32 (16) + i64 (8) + 2×i32 (8) + i32[32] (128) = 160` bytes, with no trailing padding (160 is already 8-byte aligned, and upstream has no `static_assert` pinning it to 168).
- **Fix:** Set `SIZE_GBTREE_MODEL_PARAM = 160`; documented the rationale inline. Confirmed arithmetically against `mushroom.model`: header ends at byte 173, two trees + tree_info consume 1168 bytes, `173 + 160 + 1168 == 1501` (the exact file size). The 168 figure was a research transcription error.
- **Files modified:** `crates/treelite-xgboost/src/legacy.rs`, `crates/treelite-xgboost/tests/legacy.rs` (in-memory builder assertion).
- **Commit:** `310fefd`

## Known Stubs

None. The leaf-vector tail bytes are intentionally consumed-and-discarded (scalar-only legacy, ported from upstream `:300-311`); this is upstream-faithful behavior, not a stub. Categorical / leaf-vector parse-wide fields remain unused exactly as in the JSON/UBJSON paths (D-04), gated behind branches not exercised by the verify-narrow numerical path.

## Notes for Future Phases

- The legacy loader is the third and final XGBoost on-disk format; LightGBM/sklearn loaders can reuse the `from_le_bytes` cursor pattern and the `build_model_from_parsed` convergence point.
- DART `weight_drop` fold is implemented and ported but unexercised by the current fixtures (mushroom and the 3-format fixture are both `gbtree`); a DART fixture would harden it.
- The `XgbModelJson::from_legacy_fields` string round-trip relies on `f32::to_string` round-tripping losslessly to `f32::parse` — true for IEEE-754 f32, and validated by the byte-identical cross-format golden.

## Self-Check: PASSED

- Files created/modified all present on disk.
- Both task commits (`310fefd`, `dd45d05`) present in git history.

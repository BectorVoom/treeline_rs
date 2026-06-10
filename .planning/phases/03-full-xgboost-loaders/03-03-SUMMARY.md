---
phase: 03-full-xgboost-loaders
plan: 03
subsystem: xgboost-loader
tags: [xgboost, ubjson, detect, auto-detection, byte-fidelity, d-01, d-03, d-09, d-10, xgb-02, xgb-04]

# Dependency graph
requires:
  - phase: 03-full-xgboost-loaders
    provides: "Plan 03-02 shared build_model_from_parsed convergence path + XgbModelJson structs + de_f32/de_vec_f32 NaN/Inf sentinel adapters (the JSON numeric path the UBJSON decoder reuses, D-01/D-03)"
  - phase: 03-full-xgboost-loaders
    provides: "Plan 03-01 three-format fixtures (xgb_3format.{json,ubj}) + single v5 golden blob (golden_v5_3format.bin) + shared prediction golden (xgb_3format.golden.json)"
  - phase: 02-builder-serialization
    provides: "treelite-core serialize_to_buffer (byte-perfect v5) + treelite-gtil predict for the byte-fidelity and 1e-5 asserts"
provides:
  - "detect_xgboost_format(&[u8]) -> json/ubjson/unknown: DetectXGBoostFormat first/second-byte heuristic ported verbatim (XGB-04/D-09); legacy NOT auto-detected (explicit entry, 03-04)"
  - "load_xgboost_ubjson(&[u8]) -> Result<Model, XgbError>: hand-rolled UBJSON tag decoder converging at the SAME XgbModelJson structs + de_f32 adapter as JSON (XGB-02/D-01/D-03)"
  - "decode_ubjson(&[u8]) -> Result<serde_json::Value, XgbError>: zero-dep recursive-descent UBJSON decoder (14-tag subset, big-endian, $/# optimized-container fast path, non-finite sentinel emission)"
  - "XgbError::Ubjson { pos, detail } typed error for unknown tags / truncation / oversized $/# count (never OOB/OOM — T-03-U01/U02)"
  - "UBJSON byte-fidelity closed: serialize(load_xgboost_ubjson(xgb_3format.ubj)) == serialize(load_xgboost_json(xgb_3format.json)) == golden_v5_3format.bin (D-10)"
affects: [Plan 03-04 legacy loader + full three_format_equivalence close (the JSON+UBJSON legs are now green; legacy leg stays RED until load_xgboost_legacy lands)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Hand-rolled UBJSON tag decoder -> serde_json::Value, then serde_json::from_value into the SHARED XgbModelJson structs + de_f32 adapter — converging at Value (not a second struct path) is what makes the UBJSON and JSON loads produce the IDENTICAL Model (D-01/D-03)"
    - "$/# strongly-typed optimized-container fast path: after a container opener, peek for $<type> then #<count>; with a $-type, per-element tags are omitted and every element is decoded via the shared tag (Pitfall 4)"
    - "Non-finite-float sentinel emission at the UBJSON boundary: a non-finite d/D float emits Value::String(@NaN@/@Inf@/@-Inf@) — the SAME sentinels the JSON pre-lexer emits — so it lands on the same de_f32 adapter (criterion-2 numeric parity, Pitfall 5)"
    - "Fallible byte cursor (bytes.get(..) -> Option) with $/# count validated against remaining bytes BEFORE Vec::with_capacity, so truncation/oversized-count -> typed XgbError::Ubjson, never OOB/OOM (T-03-U01/U02, ASVS V5)"
    - "Verbatim first/second-byte format heuristic; legacy reached via a separate explicit entry point matching upstream's LoadXGBoostModel{JSON,UBJSON,LegacyBinary} API split (D-09)"

key-files:
  created:
    - crates/treelite-xgboost/src/detect.rs
    - crates/treelite-xgboost/src/ubjson.rs
    - crates/treelite-xgboost/tests/detect.rs
    - crates/treelite-xgboost/tests/ubjson.rs
  modified:
    - crates/treelite-xgboost/src/lib.rs
    - crates/treelite-xgboost/src/error.rs

key-decisions:
  - "UBJSON is BIG-ENDIAN (network byte order) — all multi-byte integers/floats decode via from_be_bytes. The xgb_3format.ubj fixture confirmed this empirically (key length `L 00 00 00 00 00 00 00 07` = 7 big-endian, `[$d#L 00 00 00 00 00 00 00 0F` = count 15). The plan/research did not state endianness explicitly; the initial little-endian draft was caught by the byte-identical-to-golden test and corrected (Rule 1)."
  - "decode_ubjson is exposed as `pub fn` inside the private `ubjson` module and re-exported through the existing #[doc(hidden)] pub mod test_support (mirroring how json.rs's de_vec_f32_value/replace_nonfinite are surfaced), so tests/ubjson.rs can unit-test the decoder directly without a second public API. The module itself stays private (`mod ubjson;`), so decode_ubjson is not part of the stable public surface."
  - "Object keys in UBJSON are BARE length-prefixed strings (length-tag + UTF-8, NO leading `S` tag), decoded via decode_string; only `S`-tagged values carry the explicit `S` marker. The $/#-typed object form applies the shared value type to every member value while keys stay bare strings."
  - "DART weight_drop leaf-scaling parse-wide port (D-discretion) was NOT added: no verify-narrow fixture exercises it, the parse-wide weight_drop field already exists on the shared GradientBooster struct from 03-02, and adding gated scaling logic with no test would be unverified surface. Deferred to whenever a DART fixture exists."

patterns-established:
  - "checked_capacity(cursor, count): validate a declared $/# container count against cursor.remaining() before any pre-allocation, since even the smallest element is >= 1 byte — a count exceeding remaining bytes is structurally impossible and is rejected as a typed error rather than a giant Vec::with_capacity (DoS mitigation reusable for any length-prefixed binary format)."

metrics:
  duration: ~22 min
  tasks: 2
  files: 6
  completed: 2026-06-10
---

# Phase 3 Plan 03: XGBoost UBJSON Loader + Format Auto-Detect Summary

The UBJSON + auto-detect vertical slice: a hand-rolled zero-dependency UBJSON tag decoder that emits `serde_json::Value`, converges at the SAME `XgbModelJson` structs and `de_f32` sentinel adapter the JSON slice established (D-01/D-03), plus the ported `DetectXGBoostFormat` JSON-vs-UBJSON heuristic (XGB-04/D-09). A UBJSON model now loads → produces the IDENTICAL Model as the JSON path → serializes byte-for-byte to the single upstream golden blob → predicts within 1e-5.

## What Was Built

### Task 1 — `DetectXGBoostFormat` (detect.rs) — XGB-04 / D-09 — commit `20ba600`
- `pub fn detect_xgboost_format(first_two: &[u8]) -> &'static str` ported verbatim from `detail/xgboost.cc:83-115`: first byte `N` → `"ubjson"`; whitespace → `"json"`; non-`{` → `"unknown"`; `{` then whitespace/`"` → `"json"`; `{` then any of `N $ # i U I l L` → `"ubjson"`; else `"unknown"`.
- Short (<2-byte) slices handled safely via `.first()`/`.get(1)` defaulting to `0`.
- Returns ONLY the three literals — legacy is NOT auto-detected here (explicit entry point in 03-04, matching upstream's API split, D-09).
- `tests/detect.rs` (`detect_` prefix): 8 tests covering all five branch behaviors plus the validated `{L` (`0x7B, 0x4C`) → `"ubjson"` case, `0x00` → `"unknown"`, and short-slice safety.

### Task 2 — Hand-rolled UBJSON decoder (ubjson.rs) — XGB-02 / D-03 — commit `321094b`
- `decode_ubjson(&[u8]) -> Result<serde_json::Value, XgbError>`: recursive-descent over the 14-tag subset (`Z T F i U I l L d D C S [ ] { }` + `N` no-op), big-endian.
- **$/# optimized-container fast path** (Pitfall 4): after `[`/`{`, peek for `$<type>` then `#<count>`; with a `$`-type, per-element tags are omitted and every element decodes via the shared tag. This is how `split_conditions` etc. are stored (`[$d#L<count>`).
- **Non-finite sentinel emission** (Pitfall 5): a non-finite `d`/`D` float emits `Value::String("@NaN@"/"@Inf@"/"@-Inf@")` — the SAME sentinels the JSON pre-lexer produces — routing through the shared `de_f32` adapter for numeric parity (criterion-2).
- **Safety**: a fallible `Cursor` (`bytes.get(..)` → `Option`) makes every read truncation-safe; `$`/`#` counts are validated against remaining bytes before pre-allocation. Truncation / oversized count → typed `XgbError::Ubjson { pos, detail }`, never OOB/OOM (T-03-U01/U02).
- `load_xgboost_ubjson(&[u8]) -> Result<Model, XgbError>` in lib.rs: `decode_ubjson` → `serde_json::from_value::<XgbModelJson>` → `build_model_from_parsed` (the SAME shared path as JSON, D-01).
- New `XgbError::Ubjson` variant; `decode_ubjson` exposed via `test_support`.
- `tests/ubjson.rs` (`ubjson_` prefix): 6 tests — $/# float32 fast path, sentinel-string emission for all three non-finite values, byte-identical-to-golden vs the JSON load (D-10), 1e-5 predict vs the shared golden, oversized-count guard, truncation guard.

## Verification

- `cargo test -p treelite-xgboost detect_` → 8 passed.
- `cargo test -p treelite-xgboost ubjson_` → 6 passed.
- `cargo test -p treelite-xgboost` → all 47 passed (detect 8, ubjson 6, plus 03-02's json/nan_inf/load_fixture/error suites — no regression).
- `cargo test --workspace --exclude treelite-harness` → all green.
- `grep -v '^[[:space:]]*//' crates/treelite-xgboost/src/ubjson.rs | grep -c '@Inf@'` → 1 (non-finite sentinel path present).
- `cargo fmt` + `cargo clippy -p treelite-xgboost --tests` → clean.
- The single golden blob `golden_v5_3format.bin` is byte-identical for BOTH the JSON and UBJSON loads (asserted in `ubjson_loads_identical_model_as_json_byte_for_byte`).

### Known-RED (expected, in scope of 03-04)
The harness test `crates/treelite-harness/tests/three_format_equivalence.rs` references `load_xgboost_legacy`, which is implemented in Plan 03-04. That single test binary does not compile this wave — this is the planned RED for the legacy leg, NOT a regression. Per the plan's acceptance criteria, verification was scoped to `cargo test -p treelite-xgboost detect_ ubjson_`. The JSON and UBJSON legs of the equivalence story are both green.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] UBJSON decoded little-endian instead of big-endian**
- **Found during:** Task 2, GREEN phase (the byte-identical-to-golden test failed with `truncated: need 504403158265495552 bytes` at pos 10).
- **Issue:** The initial decoder used `from_le_bytes` for all multi-byte integers/floats. UBJSON mandates big-endian (network byte order); the real `xgb_3format.ubj` fixture stores e.g. a key length as `L 00 00 00 00 00 00 00 07`. Reading that little-endian yielded `0x0700000000000000`, a giant bogus length. The plan and RESEARCH did not state UBJSON's endianness explicitly.
- **Fix:** Switched all integer/float reads (and the corresponding hand-built test byte vectors) to `from_be_bytes` / `to_be_bytes`; documented big-endian in the module header. Confirmed against the fixture (`[$d#L ...000F` = count 15).
- **Files modified:** crates/treelite-xgboost/src/ubjson.rs, crates/treelite-xgboost/tests/ubjson.rs
- **Commit:** `321094b`

## Known Stubs

None. The DART `weight_drop` parse-wide leaf-scaling (D-discretion, optional) was intentionally NOT ported — the `weight_drop` field already exists on the shared `GradientBooster` struct (parse-wide from 03-02) so it deserializes without loss; only its numerical USE was deferred (no verify-narrow fixture exercises it). This is a documented intentional deferral, not a stub blocking the plan's goal.

## Threat Flags

None. No new network endpoints, auth paths, or schema changes were introduced beyond the UBJSON byte-decode trust boundary already enumerated in the plan's `<threat_model>` (T-03-U01..U04), all of which are mitigated: the fallible cursor + `checked_capacity` count validation (U01/U02), the sentinel-string non-finite path with the D-10 byte-fidelity assert catching desync (U03), and the explicit `$`/`#` typed-container handling with the same-Model-as-JSON assert detecting any type-tag confusion (U04). Zero new crates (T-03-SC accepted).

## Self-Check: PASSED

- All four created source/test files exist on disk.
- Both per-task commits (`20ba600`, `321094b`) exist in git history.
- `cargo test -p treelite-xgboost detect_ ubjson_` exits 0 (the plan's exact verify commands).

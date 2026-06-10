---
phase: 03-full-xgboost-loaders
verified: 2026-06-10T00:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
reverified: 2026-06-10T00:00:00Z
resolution: "human_needed items resolved by fixing the 4 critical code-review findings (CR-01/02 byte-level NaN/Inf pre-lexer, CR-03 checked-add legacy cursor, CR-04 UBJSON depth cap) in commits b1fa45d, 908c5eb, f60eabd. Each item now has a passing regression test proving the typed-error outcome under debug cargo test; cargo test --workspace fully green; 1e-5 + byte-fidelity unchanged."
resolved_human_verification:
  - test: "Non-ASCII byte in JSON value position returns typed error, not a panic."
    status: resolved
    evidence: "tests/nan_inf.rs::nan_inf_non_ascii_byte_in_value_position_does_not_panic and nan_inf_non_ascii_string_contents_round_trip_byte_unchanged pass (CR-01/CR-02, commit b1fa45d)."
  - test: "Crafted leaf-vector length overflow in legacy cursor returns XgbError::Legacy, not an overflow panic."
    status: resolved
    evidence: "tests/legacy.rs::legacy_leaf_vector_length_overflow_returns_typed_err_not_panic passes under debug overflow-checks (CR-03, commit 908c5eb)."
  - test: "Deeply nested UBJSON returns XgbError::Ubjson, not a stack-overflow abort."
    status: resolved
    evidence: "tests/ubjson.rs::ubjson_deeply_nested_input_returns_typed_err_not_stack_overflow passes; MAX_DEPTH=128 (CR-04, commit f60eabd)."
---

# Phase 3: Full XGBoost Loaders Verification Report

**Phase Goal:** Widen the loader layer to the full XGBoost surface — all three formats (JSON, UBJSON, legacy-binary) with auto-detection and the version-gated (major_version >= 1) base_score margin transform — proven across formats against the richest fixture set. Core value: predictions match upstream Treelite within 1e-5, and loader→serialize v5 bytes are byte-identical to a single upstream golden blob (closes DEF-02-01 across all three formats).
**Verified:** 2026-06-10
**Status:** passed
**Re-verification:** Yes — `human_needed` resolved after fixing the 4 critical code-review findings (commits b1fa45d, 908c5eb, f60eabd); regression tests added; `cargo test --workspace` fully green.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Loading XGBoost models in JSON, UBJSON, and legacy-binary form each produces a Model; the same logical model loaded from all three formats predicts within 1e-5 of the shared golden. | VERIFIED | `cargo test -p treelite-harness --test three_format_equivalence` passes 2/2. `three_format_predicts_within_1e5` asserts `approx::assert_abs_diff_eq!(epsilon = 1e-5)` for JSON, UBJSON, and legacy loads of `xgb_3format.*` against `xgb_3format.golden.json`. |
| 2 | The loader auto-detects which XGBoost format a file is; UBJSON path shares the JSON numeric state machine for parity (NaN/Inf accepted); legacy binary read via explicit little-endian decoders (no native-endian struct transmute). | VERIFIED | `detect.rs` ports `DetectXGBoostFormat` verbatim returning "json"/"ubjson"/"unknown" only (legacy is a separate explicit entry point, matching upstream API split). `grep -nE 'transmute|bytemuck::(cast|from_bytes|pod_read)' legacy.rs` returns nothing. `grep -c 'from_le_bytes' legacy.rs` returns 6. UBJSON shares `de_f32` sentinel adapter via identical `"@NaN@"/"@Inf@"/"@-Inf@"` strings. `cargo test -p treelite-xgboost detect_` passes 8/8. |
| 3 | XGBoost objective maps to the correct postprocessor, version-gated (major_version >= 1) base_score probability→margin transform applied (scalar and vector forms), no constant offset. | VERIFIED | `objective.rs::parse_base_score` handles scalar and vector forms with element-wise f64 version-gated transform. JSON path: `apply_transform = parsed.version.is_empty() || parsed.version[0] >= 1`. Legacy path: `major_version` mapped to `version = [major_version]`. `cargo test -p treelite-xgboost json_` (7/7), `objective_` (2/2), and `legacy_` (8/8, includes version-gate negative test) all pass. The byte-fidelity golden proves the entire transform chain is correct end-to-end. |

**Score:** 3/3 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fixtures/xgb_3format.json` | XGBoost-JSON form of the shared logical model | VERIFIED | Present on disk; xgboost 3.2.0 |
| `fixtures/xgb_3format.ubj` | XGBoost-UBJSON form | VERIFIED | Present on disk; xgboost 3.2.0 |
| `fixtures/xgb_3format.model` | XGBoost legacy-binary form | VERIFIED | Present; first bytes `binf` — not `{` (0x7B), not `N` (0x4E) |
| `fixtures/xgb_3format.golden.json` | Shared prediction golden | VERIFIED | Present; loaded by three_format_equivalence |
| `fixtures/golden_v5_3format.bin` | Single upstream v5 byte-fidelity golden | VERIFIED | 7775 bytes; first 12 bytes decode as version triple (4,7,0); sha256 matches manifest `ae53fbf8...` |
| `fixtures/xgb_3format.manifest.json` | Frozen generator manifest | VERIFIED | Records `xgboost_write_legacy=1.7.6`, `xgboost_write_json_ubj=3.2.0`, `treelite=4.7.0`, os/arch/libc/python/sha256/nbytes/source_fixtures |
| `fixtures/generate_xgb_3format.py` | Generation script | VERIFIED | Present; 391+ lines in json.rs alone; generation script exists |
| `crates/treelite-xgboost/src/json.rs` | Widened JSON structs + NaN/Inf + de_f32 | VERIFIED | 391 lines; contains `replace_nonfinite`, `de_f32`, `de_vec_f32`, full recognized key set including `sum_hessian`, `loss_changes`, categorical fields |
| `crates/treelite-xgboost/src/ubjson.rs` | Hand-rolled UBJSON decoder | VERIFIED | 346 lines; `decode_ubjson`, `decode_array`/`decode_object`, `$/# fast path`, non-finite sentinel emission (`@Inf@` confirmed by grep), fallible cursor |
| `crates/treelite-xgboost/src/detect.rs` | Auto-detect first/second-byte heuristic | VERIFIED | 66 lines; verbatim port; returns only "json"/"ubjson"/"unknown" |
| `crates/treelite-xgboost/src/legacy.rs` | LE byte cursor + ParseStream port | VERIFIED | 531 lines; 6 `from_le_bytes` calls; no transmute/bytemuck; PeekableReader for binf/bs64 magic |
| `crates/treelite-xgboost/src/objective.rs` | parse_base_score scalar+vector+version-gate | VERIFIED | `parse_base_score(raw, expand_to, postprocessor, apply_transform)` present with element-wise f64 transform |
| `crates/treelite-harness/tests/three_format_equivalence.rs` | 3-format predict + byte-fidelity test | VERIFIED | Both tests green; real assertions (not println diagnostics) |
| `crates/treelite-harness/tests/golden_v5.rs` | Promoted loader diagnostic to hard assertion | VERIFIED | `loader_path_reproduces_golden_v5_byte_for_byte` uses `assert_eq!` at line 122; 2/2 green |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `json.rs` XgbModelJson structs | `lib.rs` build_model_from_parsed | XgbModelJson fed to shared convergence path | VERIFIED | `lib.rs:183` defines `pub(crate) fn build_model_from_parsed`; `lib.rs:288,311,530` call it from JSON, UBJSON, legacy paths |
| `ubjson.rs` decode_ubjson | `json.rs` XgbModelJson via serde_json::from_value | `from_value::<XgbModelJson>` | VERIFIED | `lib.rs:308-311`: `decode_ubjson(bytes)?` → `serde_json::from_value::<json::XgbModelJson>` → `build_model_from_parsed` |
| `legacy.rs` load_xgboost_legacy | `build_model_from_parsed` | XgbModelJson::from_legacy_fields | VERIFIED | `legacy.rs:520,530`: fills `XgbModelJson` then calls `crate::build_model_from_parsed(parsed)` |
| `lib.rs` build_model_from_parsed | `treelite_builder::ModelBuilder` | `builder.sum_hess(f64)` / `builder.gain(f64)` | VERIFIED | `lib.rs:159,163`: `builder.gain(t.loss_changes[i] as f64)` on internal nodes, `builder.sum_hess(t.sum_hessian[i] as f64)` on every node |
| `three_format_equivalence.rs` | `fixtures/golden_v5_3format.bin` | `treelite_core::serialize_to_buffer` then byte compare | VERIFIED | `three_format_serialize_byte_fidelity` asserts `assert_eq!(produced, golden_blob)` with `first_diff` offset reporting for all three loaders |
| `detect.rs` detect_xgboost_format | `lib.rs` re-export | `pub use detect::detect_xgboost_format` | VERIFIED | `lib.rs:24`: `pub use detect::detect_xgboost_format` |
| `legacy.rs` load_xgboost_legacy | `lib.rs` re-export | `pub use legacy::load_xgboost_legacy` | VERIFIED | `lib.rs:25`: `pub use legacy::load_xgboost_legacy` |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `three_format_equivalence::three_format_predicts_within_1e5` | `rust` (predicted output) | `treelite_gtil::predict(model, &flat, num_row)` against model loaded from real fixtures | Yes — loads real XGBoost files, runs real GTIL prediction, asserts against captured upstream golden | FLOWING |
| `three_format_equivalence::three_format_serialize_byte_fidelity` | `produced` (serialized bytes) | `treelite_core::serialize_to_buffer(model)` on model loaded from real fixtures | Yes — asserts byte equality against `golden_v5_3format.bin` (7775 bytes from upstream Treelite 4.7.0 wheel) | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All three loaders produce identical v5 bytes == upstream golden | `cargo test -p treelite-harness --test three_format_equivalence three_format_serialize_byte_fidelity` | 1/1 passed | PASS |
| All three loaders predict within 1e-5 of shared golden | `cargo test -p treelite-harness --test three_format_equivalence three_format_predicts_within_1e5` | 1/1 passed | PASS |
| Legacy loader uses only from_le_bytes (no transmute) | `grep -nE 'transmute|bytemuck::(cast|from_bytes)' legacy.rs` | No output | PASS |
| Legacy loader has >= 4 from_le_bytes call sites | `grep -c 'from_le_bytes' legacy.rs` | 6 | PASS |
| detect_xgboost_format returns correct verdicts | `cargo test -p treelite-xgboost detect_` | 8/8 passed | PASS |
| Full workspace green | `cargo test --workspace` | All test results ok; 0 failures across all targets | PASS |

---

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase. No probe execution required.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| XGB-01 | 03-02 | User can load an XGBoost JSON model | SATISFIED | `load_xgboost_json` in `lib.rs`; `json_` tests 7/7 green; byte-fidelity + 1e-5 assertions pass |
| XGB-02 | 03-03 | User can load an XGBoost UBJSON model (parser shares the JSON state machine for numeric parity) | SATISFIED | `load_xgboost_ubjson` in `lib.rs`; hand-rolled decoder converges at shared XgbModelJson + de_f32; `ubjson_` tests 6/6 green; non-finite sentinel parity confirmed |
| XGB-03 | 03-04 | User can load an XGBoost legacy binary model (little-endian layout) | SATISFIED | `load_xgboost_legacy` in `lib.rs`; explicit `from_le_bytes` cursor; mushroom smoke (1501 bytes, 2 trees 13/11); `legacy_` tests 8/8 green |
| XGB-04 | 03-03 | The loader auto-detects which XGBoost format a file uses | SATISFIED | `detect_xgboost_format` in `detect.rs`, re-exported from `lib.rs`; ports upstream `DetectXGBoostFormat` verbatim; `detect_` tests 8/8 green |
| XGB-05 | 03-02, 03-04 | XGBoost objective maps to the correct postprocessor, with the version-gated `base_score` margin transform applied | SATISFIED | `parse_base_score` in `objective.rs`; scalar and vector forms; `apply_transform = version.is_empty() || version[0] >= 1`; legacy maps `major_version` to `version[0]`; `objective_` + `legacy_version_gate_*` tests pass |

All 5 requirements in REQUIREMENTS.md §XGBoost Loader (XGB-01 through XGB-05) are satisfied and marked `[x]` in REQUIREMENTS.md with status "Complete". Traceability table maps all 5 to Phase 3.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-xgboost/src/json.rs` | 58, 75, 79, 83, 88 | `replace_nonfinite`: value-position check uses `input[i..]` (str-slice by byte index) and in-string copy uses `out.push(c as char)` where `c: u8`. Non-ASCII byte outside string panics on char-boundary; non-ASCII byte inside string is silently re-encoded as 2-byte UTF-8 sequence (corrupts string content). | WARNING (CR-01/CR-02 from 03-REVIEW.md) | Panic / data corruption on non-ASCII model files; existing test suite covers ASCII only. Current fixture set is pure ASCII so the phase 1e-5 / byte-fidelity goals are not affected. Requires fix before processing real-world models with non-ASCII feature names or attributes. |
| `crates/treelite-xgboost/src/legacy.rs` | 88, 104 | `Cursor::take` and `Cursor::peek` compute `self.pos + n` with unchecked addition (no `checked_add`). Under debug `overflow-checks = on`, a crafted large `n` in the leaf-vector skip path overflows `usize` and panics. | WARNING (CR-03 from 03-REVIEW.md) | Arithmetic-overflow panic in debug builds on adversarially-crafted legacy files. The production fixture set does not trigger this path with extreme lengths. Release builds wrap (harmless). Requires `checked_add` to match the UBJSON cursor's discipline. |
| `crates/treelite-xgboost/src/ubjson.rs` | `decode_value`, `decode_array`, `decode_object` (lines ~110, 168-169, 249, 300) | Recursive descent with no depth cap. `decode_value` → `decode_array`/`decode_object` → `decode_value`. A stream of N `[` bytes produces N-deep native stack frames → stack overflow abort (SIGSEGV). | WARNING (CR-04 from 03-REVIEW.md) | Process abort (uncatchable) on deeply-nested UBJSON. serde_json caps at 128; this decoder has no cap. Requires a depth counter and a MAX_DEPTH guard. |
| `crates/treelite-xgboost/src/legacy.rs` | 518 | `i32::try_from(mparam.major_version).unwrap_or(i32::MAX)` silently substitutes i32::MAX on overflow instead of returning a typed error, forcing the base_score transform gate to always fire for corrupt headers — inconsistent with neighboring `try_from` / typed-error pattern. | WARNING (WR-04 from 03-REVIEW.md) | Silent behavior change for corrupted legacy headers with major_version > i32::MAX. No correctness risk for any real XGBoost file. |

No TBD, FIXME, or XXX markers found in any of the five source files checked — no debt-marker blockers.

---

### Human Verification Required

#### 1. CR-01/CR-02: replace_nonfinite panics / corrupts non-ASCII input

**Test:** Create a minimal XGBoost JSON file where a model attribute or feature name contains a non-ASCII byte (e.g., `"température"` as a feature name). Call `treelite_xgboost::load_xgboost_json` on this file and observe whether it returns an error or panics.
**Expected:** Returns `XgbError::Json` (or propagated serde_json error), not a `thread 'main' panicked ... byte index N is not a char boundary` panic.
**Why human:** `replace_nonfinite` at `json.rs:75,79,83` slices `input[i..]` at a byte index that may fall inside a multi-byte UTF-8 sequence for any non-ASCII byte outside a string literal. The existing test suite (`nan_inf_string_contents_are_byte_unchanged`) only exercises ASCII strings. The panic is confirmed in the source but cannot be safely triggered in a verifier subprocess without a deliberate test harness change.

#### 2. CR-03: legacy Cursor::take arithmetic overflow panic in debug builds

**Test:** Craft a legacy binary buffer that passes the magic/header checks but sets `size_leaf_vector != 0`, `major_version < 2`, and the leaf-vector length `u64` field to a value near `usize::MAX / 4`. Call `treelite_xgboost::load_xgboost_legacy` in a debug-profile binary (`cargo test` mode). Observe whether it returns `XgbError::Legacy` or panics.
**Expected:** Returns `XgbError::Legacy { detail: "..." }`, not an arithmetic-overflow panic (`attempt to add with overflow`) under debug overflow-checks.
**Why human:** `legacy.rs:88` computes `self.buf.get(self.pos..self.pos + n)` with plain `+`. Under `overflow-checks = on` (the default for `cargo test` debug profile), a near-`usize::MAX` value of `n` overflows. The fix is `checked_add`. Cannot trigger this safely in the verifier.

#### 3. CR-04: UBJSON decoder stack-overflow abort on deeply-nested input

**Test:** Construct a UBJSON byte buffer consisting of several thousand nested array openers (repeated `b'['` bytes with no closers), pass it to `load_xgboost_ubjson`, and observe whether the process returns a typed error or aborts.
**Expected:** Returns `XgbError::Ubjson { detail: "nesting too deep" }` (or similar), not a SIGSEGV / stack-overflow abort.
**Why human:** `ubjson.rs` has no recursion depth cap. `decode_value` → `decode_array` → `decode_value` chains unboundedly. A malicious stream of `[` bytes overflows the native stack and triggers an OS abort — uncatchable from Rust code. Cannot verify that the abort does or does not happen by inspection alone; requires a controlled subprocess test.

---

## Gaps Summary

No gaps block the phase success criteria. All three roadmap success criteria are verified:

1. All three format loaders produce a Model; the three-format test asserts 1e-5 prediction parity AND byte-identical serialization against the single upstream golden — both green.
2. Auto-detect (`detect_xgboost_format`) is wired and tested; UBJSON shares the JSON numeric sentinel path; legacy uses explicit `from_le_bytes` (no transmute).
3. `parse_base_score` handles scalar and vector base_score forms with the version-gated f64 margin transform; XGBoost objectives map to correct postprocessors.

The three human-verification items (CR-01/CR-02/CR-04 from 03-REVIEW.md, plus CR-03) are robustness and security defects on the untrusted-parsing surface that are BEYOND the phase success criteria. The test fixture set is ASCII-only and debug-overflow-free, so the phase 1e-5 / byte-fidelity contract holds for the current fixtures. These defects are hardening work that should be tracked as follow-up before any production deployment, but they do not prevent the phase goal from being achieved as defined.

---

_Verified: 2026-06-10_
_Verifier: Claude (gsd-verifier)_

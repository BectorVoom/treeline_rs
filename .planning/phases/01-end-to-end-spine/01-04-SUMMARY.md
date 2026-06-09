---
phase: 01-end-to-end-spine
plan: 04
subsystem: harness
tags: [rust, equivalence-harness, anyhow, approx, serde, golden, 1e-5, spine-test, nan-normalization]

# Dependency graph
requires:
  - phase: 01-01
    provides: treelite-core (Model/ModelVariant/Tree) + committed fixtures/golden.json {input,output,manifest} + binary_logistic.model.json
  - phase: 01-02
    provides: treelite-xgboost::load_xgboost_json(&str) -> Result<Model, XgbError> (F32 variant)
  - phase: 01-03
    provides: treelite-gtil::predict(&Model, &[f32], num_row) -> Result<Vec<f32>, GtilError> (scalar, verbatim cast ordering)
provides:
  - treelite-harness::load_golden(path) -> anyhow::Result<Golden>  (NaN-token-normalizing golden reader)
  - treelite-harness::run_equivalence(model_json_path, &Golden) -> anyhow::Result<f64>  (load -> predict -> assert 1e-5, returns max |delta|)
  - treelite-harness::check_manifest(&Manifest)  (warns, never fails, on OS/arch drift — D-07)
  - Golden/Manifest/NanF32 serde structs
  - the end-to-end equivalence spine test (Success Criterion 4)
affects: [Phase 5 full seeded EQV-04 harness across all model types, all future loader/predict regression checks]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "anyhow for ALL error context in the dev/test crate (ERR-02); typed thiserror errors from the 3 library crates are wrapped with .context(...) — anyhow is NEVER exposed from a library public API"
    - "NaN -> null token normalization on golden read so strict serde_json accepts Python's bare NaN missing-value cell WITHOUT editing the committed golden.json"
    - "custom NanF32 Deserialize (number-or-null -> f32, null => f32::NAN) for the missing-feature row"
    - "max-observed-|delta| reported as f64 return from run_equivalence (Success Criterion 4 spine); approx::assert_abs_diff_eq! is the hard 1e-5 gate"
    - "manifest check warns (eprintln) on env drift, never fails — a 1e-5 failure on a different distro stays diagnosable (D-07, T-04-02)"

key-files:
  created:
    - crates/treelite-harness/tests/run_equivalence.rs
    - crates/treelite-harness/tests/equivalence.rs
  modified:
    - crates/treelite-harness/Cargo.toml
    - crates/treelite-harness/src/lib.rs

key-decisions:
  - "NaN in golden.json is normalized to JSON null on read (serde_json strictly rejects the bare NaN literal); a custom NanF32 deserializer maps null -> f32::NAN. The committed golden.json is never modified."
  - "run_equivalence wraps XgbError/GtilError via anyhow::anyhow!(\"{e}\").context(...) because those typed errors are not std::error::Error-into-anyhow-friendly in all cases and Model is not Debug — string-wrap + context keeps a clean ERR-02 chain."
  - "check_manifest compares the verbose platform.platform() string against the coarse std::env::consts::OS via a case-insensitive substring (right granularity); arch is compared case-insensitively exactly."
  - "The spine test asserts max_dev < 1e-5 explicitly in the test body in addition to the element-wise assert_abs_diff_eq! inside run_equivalence, making the gate visible in the test."

requirements-completed: [ERR-02]

# Metrics
duration: 3min
completed: 2026-06-10
---

# Phase 1 Plan 04: Equivalence Harness Summary

**Implemented `treelite-harness` — the 1e-5 equivalence instrument that closes the walking skeleton: it loads the frozen upstream `fixtures/golden.json`, runs `treelite_xgboost::load_xgboost_json` then `treelite_gtil::predict` over the committed input, asserts every output element is within 1e-5 of the golden, and reports the max observed |delta| — which came out to `0e0` (a bitwise-exact match against upstream Treelite 4.7.0).**

## The Payoff

The entire core value of the project — **predictions match upstream Treelite within 1e-5** — is now proven end-to-end and runnable via `cargo test --workspace`. The spine test (`equivalence_within_1e5`) loads the hand-crafted XGBoost-JSON fixture through the Rust pipeline over the golden's committed 5-row input (including a missing/`NaN` feature row routed via `default_left`) and finds **zero deviation** from the frozen upstream prediction vector. No tolerance loosening, no golden editing — a genuine clean pass.

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-09T21:45:50Z
- **Tasks:** 2 (Task 1 `tdd="true"`; Task 2 plain)
- **Files:** 2 created, 2 modified

## Accomplishments

- `src/lib.rs` (255 lines): `Golden { input: Vec<Vec<NanF32>>, output: Vec<f32>, manifest: Manifest }`, `Manifest { treelite, xgboost: Option, os, arch, libc: serde_json::Value, python: Option }` (keys match `capture_golden.py` exactly), `load_golden`, `run_equivalence` (returns max |delta| as f64), `check_manifest` (warns, never fails).
- **NaN handling:** Python's `json.dump` writes a bare `NaN` for the missing-value row; strict `serde_json` rejects it. Added `normalize_nan_tokens` (token-boundary-safe `NaN` -> `null`) + a custom `NanF32` deserializer (`null` => `f32::NAN`) so the golden round-trips **without editing the committed file**.
- `tests/run_equivalence.rs` (181 lines): a hand-authored single-tree `binary:logistic` model (`base_score=0.5` -> margin transform exactly `0.0`) proves (a) `run_equivalence` returns `max_dev < 1e-5` against a hand-computed `sigmoid(L)` golden, and (b) it **catches** a `>1e-5` perturbation (the 1e-5 assertion fires), plus (c) `load_golden` on a missing path returns an `anyhow` `Err` with a `golden.json` context chain.
- `tests/equivalence.rs` (50 lines): THE spine test — loads the committed fixture, predicts, asserts within 1e-5 of `golden.json`, prints `max observed |delta| = 0e0`.
- `Cargo.toml`: added workspace `serde` (derive).
- Full workspace green: **52 tests pass** (core 18, gtil 12, harness 4, xgboost 11 + doc-tests), `cargo build --workspace` clean, `cargo clippy -p treelite-harness --all-targets` clean.

## Task Commits

1. **Task 1: harness library + run_equivalence unit test vs hand-computed model** — `2f9a1a9` (feat) — `Cargo.toml`, `src/lib.rs`, `tests/run_equivalence.rs`, `Cargo.lock`.
2. **Task 2: end-to-end equivalence spine test (load -> predict -> 1e-5 vs golden)** — `a5df461` (feat) — `tests/equivalence.rs`.

_Note: Task 1 is `tdd="true"`; global `tdd_mode` is `false`, so RED/GREEN were committed together (tests + impl in one commit), consistent with Plans 02/03._

## Files Created/Modified

- `crates/treelite-harness/src/lib.rs` — harness library: golden/manifest structs, NaN-tolerant golden reader, `run_equivalence` (1e-5 gate + max-deviation report), manifest drift warning.
- `crates/treelite-harness/tests/run_equivalence.rs` — unit test of `run_equivalence` against a hand-computed scalar model (catches a >1e-5 deviation; no golden.json dependency).
- `crates/treelite-harness/tests/equivalence.rs` — the end-to-end spine test against the committed golden.
- `crates/treelite-harness/Cargo.toml` — added workspace `serde` (derive); already depended on the 3 library crates + anyhow/approx/serde_json.

## Decisions Made

- **NaN -> null normalization (no golden edit).** `serde_json` rejects bare `NaN`; `normalize_nan_tokens` replaces only standalone `NaN` tokens (bounded by non-identifier chars on both sides) with `null`, and `NanF32` maps `null` -> `f32::NAN`. This faithfully round-trips the golden's missing-feature row while leaving the committed `golden.json` byte-for-byte untouched (the plan's hard constraint).
- **anyhow string-wrap of typed errors.** `XgbError`/`GtilError` are wrapped via `anyhow::anyhow!("{e}").context(...)` rather than `?`-propagated directly — `Model` is intentionally not `Debug` and the wrap keeps a clean ERR-02 context chain without leaking the library error types.
- **Manifest check granularity.** `manifest.os` is the verbose `platform.platform()` descriptor, so it is matched case-insensitively as a substring against the coarse `std::env::consts::OS` (e.g. `"linux"`); `arch` is compared case-insensitively exactly. Warns via `eprintln!`, never fails (D-07/T-04-02).

## Deviations from Plan

None of behavior. One **Rule 3 (blocking, no behavior change)** addition: the plan did not anticipate that strict `serde_json` rejects the bare `NaN` literal Python emits in `golden.json`. Loading the committed golden would otherwise fail at parse time. Resolved by adding `normalize_nan_tokens` + the `NanF32` number-or-null deserializer in `src/lib.rs` (NaN cell => `f32::NAN`), with the committed `golden.json` left unmodified. Two minor clippy/warning cleanups (unused `self` import; `needless_range_loop` -> `iter().enumerate()`) — no behavior change.

## Issues Encountered

The golden's input row 5 is `[NaN, 0.0]` (missing `feature[0]`, routed via `default_left`). This surfaced the serde_json bare-`NaN` rejection described above — handled by normalization, not by editing the fixture. The resulting equivalence delta is exactly `0e0`: the Rust f32-only sigmoid + verbatim cast ordering reproduce the upstream output bit-for-bit on this fixture (no libm divergence on this environment — the manifest matches the running `Linux`/`x86_64`).

## Known Stubs

None. The harness is a complete 1e-5 instrument for the `binary:logistic` walking-skeleton fixture. The full seeded EQV-04 harness across all model types (XGBoost/LightGBM/sklearn, randomized inputs) is explicitly Phase 5 scope per the plan, not a stub in the Phase 1 path.

## Threat Flags

None. The harness introduces no new network/auth surface. It reads two committed local fixture files via `std::fs::read_to_string` with `anyhow` context (T-04-01 mitigated: malformed/missing golden surfaces a context chain, not a panic — exercised by `load_golden_on_missing_path_returns_err_with_context`). T-04-02 (environment divergence) is handled by `check_manifest` warning, not failing. The three crates.io deps (anyhow, approx, serde/serde_json) are the pinned, slop-checked workspace versions (T-04-SC accept).

## Verification Evidence

- `cargo test -p treelite-harness --test run_equivalence` — 3/3 pass (match `max_dev<1e-5`, perturbation `>1e-5` caught, missing-path anyhow `Err`).
- `cargo test -p treelite-harness --test equivalence -- --nocapture` — 1/1 pass; prints `max observed |delta| = 0e0`.
- `cargo build --workspace` — clean.
- `cargo test --workspace` — 52/52 pass, 0 failures (phase gate, Success Criterion 1).
- `cargo clippy -p treelite-harness --all-targets` — clean.
- The committed `fixtures/golden.json` was NOT modified (NaN handled by read-time normalization); the 1e-5 tolerance was NOT loosened.

## Self-Check: PASSED

Both declared created files (`tests/run_equivalence.rs`, `tests/equivalence.rs`) and both modified files (`Cargo.toml`, `src/lib.rs`) exist on disk; both task commits (`2f9a1a9`, `a5df461`) are present in git history.

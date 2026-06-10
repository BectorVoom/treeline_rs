---
phase: 03-full-xgboost-loaders
plan: 01
subsystem: fixtures-and-harness
tags: [fixtures, golden, manifest, three-format, byte-fidelity, red-scaffold, xgboost]

# Dependency graph
requires:
  - phase: 02-builder-serialization
    provides: v5 serializer + golden byte-compare harness (treelite-harness load_golden / Golden / NanF32)
  - phase: 02-builder-serialization
    provides: end_tree AllocNode column emission (CR-01/CR-02) that makes loader byte-fidelity achievable (D-10)
provides:
  - "One logical binary:logistic numerical-split model (base_score=0.5) saved in JSON, UBJSON, and legacy binary from one session (D-05)"
  - "Shared prediction golden (input matrix + output vector) captured from the Treelite 4.7.0 wheel for the 3-format model"
  - "Single upstream v5 byte-fidelity golden blob (golden_v5_3format.bin, 7775 bytes) all three loaders must match (D-10 / DEF-02-01 target)"
  - "Frozen generator manifest (xgboost write versions, treelite, os/arch/libc/python, sha256, nbytes, source_fixtures)"
  - "RED 3-format equivalence + byte-fidelity test scaffold (fails to compile until 03-02..03-04 land)"
  - "Empirically confirmed A1 legacy-write xgboost pin = 1.7.6"
affects: [Plan 03-02 JSON D-10 close, Plan 03-03 UBJSON loader, Plan 03-04 legacy loader + cross-format close]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase ephemeral-env generator: legacy phase (pinned old xgboost) hands a trained-booster JSON spec to the modern phase (xgboost 3.2.0 + treelite 4.7.0) so all three formats come from ONE logical model without re-training"
    - "Generation-only Python tools run via 'uv run --no-project --with', never committed as runtime/CI deps (D-06)"
    - "RED test scaffold references not-yet-existing loader entry points so the phase acceptance bar is executable from day one (MVP failing-test-first)"

key-files:
  created:
    - fixtures/generate_xgb_3format.py
    - fixtures/xgb_3format.json
    - fixtures/xgb_3format.ubj
    - fixtures/xgb_3format.model
    - fixtures/xgb_3format.golden.json
    - fixtures/golden_v5_3format.bin
    - fixtures/xgb_3format.manifest.json
    - crates/treelite-harness/tests/three_format_equivalence.rs
    - .planning/phases/03-full-xgboost-loaders/03-01-SUMMARY.md
  modified: []

key-decisions:
  - "A1 settled empirically: xgboost 1.7.6 writes genuine legacy binary — a 'binf' 4-byte magic prefix followed by the 136-byte LearnerModelParam (base_score=0.5 f32 @0, num_feature=4 u32 @4). The generation spike was resolved AUTONOMOUSLY (no human checkpoint needed); 1.6.2/0.90 fallbacks were not required."
  - "base_score=0.5 (A2) neutralizes the version-gated sigmoid margin transform (sigmoid(0.5)=0 margin), so all three formats serialize to ONE identical v5 blob — proven at generation time by the A2 cross-format same-blob assert before any Rust loader exists."
  - "Legacy binary is loaded through treelite's separate load_xgboost_model_legacy_binary entry point, NOT load_xgboost_model (which only handles JSON/UBJSON and mis-sniffs the binf-prefixed legacy file as UBJSON) — mirrors upstream's API split (D-09)."
  - "The single-golden invariant (golden_v5_3format.bin, sha256 ae53fbf8…, 7775 bytes) is the D-10/DEF-02-01 target: all three Rust loaders must serialize to it byte-identically once 03-02..03-04 land."

patterns-established:
  - "Generator self-routes between two ephemeral xgboost environments via a phase arg ('legacy'|'modern') plus a temp booster-spec handoff file, sidestepping the incompatible-xgboost-version-in-one-interpreter constraint."

metrics:
  duration: ~12 min
  tasks: 3
  files: 8
  completed: 2026-06-10
---

# Phase 3 Plan 01: Wave-0 Three-Format Fixtures, Goldens, Manifest + RED Scaffold Summary

**One-liner:** Stood up the single-logical-model 3-format XGBoost fixtures (JSON/UBJSON/legacy binary, base_score=0.5), the shared prediction golden and the single v5 byte-fidelity golden blob captured from the Treelite 4.7.0 wheel, a frozen generator manifest (A1 legacy pin = xgboost 1.7.6, confirmed empirically), and a RED 3-format equivalence + byte-fidelity test scaffold that fails to compile until the UBJSON/legacy loaders land in 03-03/03-04.

## What Was Built

- **`fixtures/generate_xgb_3format.py`** — one-session, two-phase generator. The `legacy` phase trains ONE deterministic `binary:logistic` numerical-split model (4 features, depth 3, 6 trees, seed 1234, `base_score=0.5`, `tree_method=exact`, no categorical features) with a pinned OLD xgboost, dumps the trained booster spec for handoff, writes the legacy `.model`, and runs the A1 assert. The `modern` phase re-loads that exact booster, writes `.json`/`.ubj`, captures both goldens from the Treelite 4.7.0 wheel, runs the version-triple `(4,7,0)` and A2 cross-format same-blob asserts, and freezes the manifest.
- **Six frozen artifacts** — `xgb_3format.{json,ubj,model}`, `xgb_3format.golden.json` (8-row seeded input + sigmoid output in (0,1)), `golden_v5_3format.bin` (7775 bytes, version triple `(4,7,0)`), `xgb_3format.manifest.json`.
- **`crates/treelite-harness/tests/three_format_equivalence.rs`** — RED scaffold with `three_format_predicts_within_1e5` (all three formats predict within 1e-5 of the shared golden) and `three_format_serialize_byte_fidelity` (all three serialize to the single v5 golden blob byte-for-byte). Both reference `load_xgboost_ubjson`/`load_xgboost_legacy`, which do not yet exist.

## Verification Results

- Task 1: `uv run --no-project python -c "ast.parse(...)"` → PARSE-OK.
- Task 2: A1 assert passed (legacy first byte `b`/`binf`, LearnerModelParam base_score=0.5 num_feature=4); version triple `(4,7,0)`; A2 — all three formats serialize to the SAME v5 blob. All six files on disk; legacy first byte not `{`/`N`; manifest records confirmed legacy pin 1.7.6.
- Task 3: `cargo test -p treelite-harness --test three_format_equivalence` → RED-AS-EXPECTED (4 errors, only `cannot find function load_xgboost_ubjson/load_xgboost_legacy`). No stubs added. Existing `golden_v5.rs` test still compiles and passes (2/2).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] A1 legacy-header assert did not handle the optional `binf` magic prefix**
- **Found during:** Task 2 (first legacy generation run with xgboost 1.7.6).
- **Issue:** xgboost 1.7.6 writes the legacy file with a 4-byte `binf` magic prefix before the 136-byte `LearnerModelParam` (documented in RESEARCH §Legacy Binary Layout step 1; mushroom.model has NO magic, so the original assert read the header at offset 0 and decoded garbage `num_feature=1056964608`). The fixture was genuinely legacy binary — the assert was wrong.
- **Fix:** Skip a 4-byte `binf` prefix (and hard-error on `bs64`) before decoding the 136-byte header.
- **Files modified:** `fixtures/generate_xgb_3format.py`
- **Commit:** 6133ee7

**2. [Rule 1 - Bug] A2 cross-format gate routed the legacy `.model` through the JSON/UBJSON-only loader**
- **Found during:** Task 2 (modern phase A2 loop).
- **Issue:** `treelite.frontend.load_xgboost_model` only handles JSON/UBJSON and mis-sniffed the `binf`-prefixed legacy file as UBJSON → nlohmann sax_parse error. Legacy binary has a SEPARATE upstream entry point (`load_xgboost_model_legacy_binary`), matching the D-09 API split.
- **Fix:** Route JSON via `format_choice="json"`, UBJSON via `format_choice="ubjson"`, and legacy via `load_xgboost_model_legacy_binary`.
- **Files modified:** `fixtures/generate_xgb_3format.py`
- **Commit:** 6133ee7

## Checkpoint Note (Task 2 — resolved autonomously, no human gate hit)

Task 2 was authored as a `checkpoint:human-verify` because installing a pinned OLD xgboost ephemerally was not guaranteed to resolve and the A1 pin needed empirical settling. Per the spawn instructions ("if you can resolve the spike autonomously via `uv run --with xgboost==<ver>` and empirically confirm it writes legacy binary, do so"), the spike was resolved without escalation: `xgboost==1.7.6` installed cleanly via `uv run --no-project --with` and, after the two Rule-1 fixes above, its output passed the A1 legacy-write assert (genuine `binf` + 136-byte LearnerModelParam). The 1.6.2/0.90 fallbacks were not needed. The confirmed pin is recorded in the manifest.

## Notes for Downstream Plans

- **03-02 (JSON D-10 close):** must populate `sum_hess` (every node) + `gain` (internal nodes) and pass `attributes: None` (→ `"{}"`) so `load_xgboost_json` serializes byte-identical to `golden_v5_3format.bin`. The RED `three_format_serialize_byte_fidelity` test will start asserting once the UBJSON/legacy entry points compile.
- **03-03 (UBJSON):** add `treelite_xgboost::load_xgboost_ubjson(&[u8]) -> Result<Model, XgbError>` — the harness test references exactly this signature.
- **03-04 (legacy):** add `treelite_xgboost::load_xgboost_legacy(&[u8]) -> Result<Model, XgbError>` — note the fixture carries the `binf` magic prefix (peekable-reader handling per D-07).
- The temp handoff files (`.xgb_3format.booster.json`, `.xgb_3format.legacy_ver.txt`) are removed after generation and were never committed.

## Self-Check: PASSED

All 8 created files exist on disk and all 3 per-task commits (4bb73e4, 6133ee7, 5d41c1b) are in git history.

---
phase: 03-full-xgboost-loaders
plan: 02
subsystem: xgboost-loader
tags: [xgboost, json, nan-inf, base-score, byte-fidelity, def-02-01, d-10, xgb-01, xgb-05]

# Dependency graph
requires:
  - phase: 03-full-xgboost-loaders
    provides: "Plan 03-01 three-format fixtures + single v5 golden blob (golden_v5_3format.bin) + RED three_format_equivalence scaffold"
  - phase: 02-builder-serialization
    provides: "ModelBuilder sum_hess/gain/commit_model + end_tree AllocNode column emission (CR-01/CR-02) that makes loader byte-fidelity achievable (D-10)"
  - phase: 02-builder-serialization
    provides: "treelite-core serialize_to_buffer (byte-perfect v5) + treelite-gtil predict"
provides:
  - "Widened XGBoost-JSON loader: full recognized key set (loss_changes, sum_hessian, base_weights, leaf_weights, categories_*, boost_from_average, tree_info, cats, weight_drop, num_feature, size_leaf_vector) parse-wide (D-04)"
  - "D-02 NaN/Inf mechanism: string-safe replace_nonfinite pre-lexer + de_f32/de_vec_f32 sentinel adapters (value-position only, string contents byte-unchanged)"
  - "Shared build_model_from_parsed(XgbModelJson) convergence path (D-01) that 03-03 (UBJSON via from_value) and 03-04 (legacy) reuse"
  - "objective::parse_base_score: scalar AND vector base_score, version-gated element-wise f64 margin transform (XGB-05)"
  - "DEF-02-01 closed for the JSON path: serialize(load_xgboost_json(xgb_3format.json)) == golden_v5_3format.bin byte-for-byte (sum_hess/gain emission + attributes:None)"
  - "XgbError::BaseScoreShape typed error for wrong-length vector base_score"
affects: [Plan 03-03 UBJSON loader (converges at json structs + de_f32), Plan 03-04 legacy loader + cross-format close]

# Tech tracking
tech-stack:
  added:
    - "treelite-gtil (dev-dependency of treelite-xgboost — drives the JSON 1e-5 predict assertion in tests/json.rs)"
    - "approx (dev-dependency — abs-diff 1e-5 gate)"
  patterns:
    - "Widen-then-converge serde structs: parse-wide full key set, gate USE of every parse-wide field behind leaf-vector/categorical/multiclass branches so the verify-narrow numerical path is unchanged (D-04)"
    - "String-safe value-position pre-lexer (in-string state tracked) rewriting NaN/Infinity/-Infinity to sentinel STRINGS, recovered by a deserialize_with adapter — never a numeric literal (D-02, avoids serde_json out-of-range rejection)"
    - "Shared build_model_from_parsed convergence path so all three formats produce one identical Model → one identical v5 blob (D-01/D-10)"
    - "Golden parsed locally in the loader crate's own test to avoid a dependency cycle (treelite-harness depends on treelite-xgboost)"

key-files:
  created:
    - crates/treelite-xgboost/src/json.rs
    - crates/treelite-xgboost/tests/nan_inf.rs
    - crates/treelite-xgboost/tests/json.rs
  modified:
    - crates/treelite-xgboost/src/lib.rs
    - crates/treelite-xgboost/src/objective.rs
    - crates/treelite-xgboost/src/error.rs
    - crates/treelite-xgboost/Cargo.toml

key-decisions:
  - "The verify-narrow fixture xgb_3format.json uses the VECTOR base_score form ('[5E-1]'), not scalar — so parse_base_score's vector path is exercised by the real 1e-5/byte-fidelity tests this wave, not just a synthetic unit test. base_score=0.5 makes the sigmoid transform a no-op (margin 0) so all three formats agree (A2)."
  - "expand_to for base_scores = num_target * effective_num_class, where effective_num_class = max(num_class_param, 1) (binary/regressor branch uses 1). This preserves the Phase-1 scalar fixture's single-element base_scores while supporting multiclass parse-wide."
  - "treelite-harness was deliberately NOT added as a dev-dependency of treelite-xgboost (it depends on treelite-xgboost → cycle); the golden is parsed locally in tests/json.rs with a small normalize_nan helper instead."
  - "de_vec_f32_value + replace_nonfinite re-exported via a #[doc(hidden)] pub mod test_support so the integration test tests/nan_inf.rs can exercise the crate-internal D-02 primitives directly."

patterns-established:
  - "BaseScoreVec newtype deserializes the vector base_score string through de_vec_f32 so the same D-02 sentinel recovery applies to base_score as to split_conditions/loss_changes/sum_hessian."

metrics:
  duration: ~18 min
  tasks: 2
  files: 7
  completed: 2026-06-10
---

# Phase 3 Plan 02: XGBoost-JSON Vertical Slice (widen + D-02 + XGB-05 + D-10 close) Summary

**One-liner:** Widened the XGBoost-JSON loader from the Phase-1 minimal subset to the full recognized key set, added the D-02 string-safe NaN/Inf sentinel mechanism, handled the scalar AND vector `base_score` forms with the version-gated f64 margin transform (XGB-05), and closed DEF-02-01 for the JSON path so `serialize(load_xgboost_json(xgb_3format.json))` equals the single upstream `golden_v5_3format.bin` byte-for-byte — establishing the shared `build_model_from_parsed` convergence path that 03-03/03-04 reuse.

## What Was Built

- **`crates/treelite-xgboost/src/json.rs` (new, 301 lines)** — the serde structs moved out of `lib.rs` and widened to the full recognized key set per the `delegated_handler.cc` `is_recognized_key` authority (RegTree/TreeParam/GBTreeModel/GradientBooster/LearnerParam/Learner/XGBoostModel handlers). Parse-wide fields (`loss_changes`, `sum_hessian`, `base_weights`, `leaf_weights`, `categories_*`, `boost_from_average`, `tree_info`, `cats`, `weight_drop`, `num_feature`, `size_leaf_vector`) are carried with `#[serde(default)]` and their use gated behind branches not exercised by the numerical path (D-04). Implements D-02 verbatim from RESEARCH: `replace_nonfinite` (string-state-tracking value-position pre-lexer → sentinel STRINGS), `de_f32`/`de_vec_f32` adapters attached to every `Vec<f32>` field XGBoost may fill non-finite, and a `BaseScoreVec` newtype for the vector base_score string.
- **`crates/treelite-xgboost/src/lib.rs` (refactored)** — `load_xgboost_json` is now `replace_nonfinite` → `serde_json::from_str` → shared `build_model_from_parsed`. `build_tree` emits `sum_hess` on every node (from `sum_hessian`) and `gain` on internal nodes (from `loss_changes`), dimension-checked first; `data_count` is intentionally left unset. The metadata passes `attributes: None` (was `Some(String::new())`) so `commit_model` stamps `"{}"` matching upstream. base_scores now go through `parse_base_score` with `expand_to = num_target * effective_num_class`.
- **`crates/treelite-xgboost/src/objective.rs` (extended)** — added `parse_base_score(raw, expand_to, postprocessor, apply_transform)` porting `ParseBaseScore`: scalar form fills across `expand_to`; vector form (`"[..]"`) parses the inner JSON array through the D-02 sentinel mechanism, asserts `len == expand_to`, casts f32→f64 BEFORE the element-wise transform. `get_postprocessor`/`prob_to_margin_*` unchanged (f64 invariant preserved).
- **`crates/treelite-xgboost/src/error.rs` (extended)** — added `XgbError::BaseScoreShape { expected, got }` for wrong-length vector base_score (T-03-V04).
- **`crates/treelite-xgboost/tests/nan_inf.rs` (new)** — 5 `nan_inf_` tests: sentinel rewrite, string-content byte-safety, f32 sentinel round-trip, Infinity-never-a-numeric-literal, finite pass-through.
- **`crates/treelite-xgboost/tests/json.rs` (new)** — 7 `json_` tests: widened-key load, **byte-fidelity assert vs golden_v5_3format.bin**, 1e-5 predict vs the shared golden, and parse_base_score scalar/vector/version-gate/wrong-length cases.

## Verification Results

- **Task 1:** `cargo test -p treelite-xgboost nan_inf_` → 5/5 pass. `json.rs` exists; `lib.rs` has `mod json;` and no inline structs; `sum_hessian` parsed (non-comment grep ≥ 1); string contents byte-unchanged asserted.
- **Task 2:** `cargo test -p treelite-xgboost json_` → 7/7 pass. **`json_serialize_equals_golden_v5_byte_for_byte` passes** (DEF-02-01 closed for JSON). `json_predicts_within_1e5_of_shared_golden` passes. `cargo test -p treelite-xgboost objective_` → 2/2. `attributes: None` present (count 1), `attributes: Some(String::new())` absent (count 0).
- **Full crate:** `cargo test -p treelite-xgboost` → 6 (error) + 7 (load_fixture) + 7 (json) + 5 (nan_inf) all green; existing Phase-1 fixture tests (scalar base_score) still pass.
- **No workspace regression:** treelite-core (11+2+... ), treelite-builder, treelite-gtil, and the harness `golden_v5`/`equivalence`/`run_equivalence` targets + harness lib all pass. `cargo fmt --check` clean, `cargo clippy` clean for the crate.
- **Intended RED (scoped out this wave):** `treelite-harness --test three_format_equivalence` still fails to COMPILE — only because `load_xgboost_ubjson`/`load_xgboost_legacy` don't exist yet (they land in 03-03/03-04). This is the documented RED state; the JSON byte-fidelity leg is independently proven by `tests/json.rs::json_serialize_equals_golden_v5_byte_for_byte`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] treelite-harness cannot be a dev-dependency (dependency cycle)**
- **Found during:** Task 2 (wiring the 1e-5 predict assertion in tests/json.rs).
- **Issue:** The plan's behavior sketch loaded the golden via `treelite_harness::load_golden`, but `treelite-harness` depends on `treelite-xgboost`; adding it as a dev-dep would create a Cargo dependency cycle (build failure).
- **Fix:** Parse the golden JSON locally in `tests/json.rs` with a small `normalize_nan` helper + a local `Golden` struct (`Vec<Vec<Option<f32>>>` for missing values). Added only `treelite-gtil` + `approx` as dev-deps (neither depends on treelite-xgboost). The same byte-fidelity + 1e-5 assertions remain.
- **Files modified:** crates/treelite-xgboost/tests/json.rs, crates/treelite-xgboost/Cargo.toml
- **Commit:** b1c9626

### Notes / clarifications (not deviations)

- The plan described the scalar form as the common case, but the actual `xgb_3format.json` fixture uses the **vector** base_score form (`"[5E-1]"`). This means the vector path is exercised by the real byte-fidelity/predict tests, not just a synthetic unit test — strictly stronger coverage. The Phase-1 `binary_logistic.model.json` (scalar `"2.5E-1"`) still loads correctly through the new scalar branch (existing `load_fixture.rs` tests green).
- Commit boundary note: because `lib.rs` (Task 1 split) and `objective.rs`/`error.rs` (Task 2 additions) are mutually required for the crate to compile, the Task 1 commit carries the full source refactor (json split + D-02 + shared build path + parse_base_score) and the Task 2 commit carries the `tests/json.rs` slice + Cargo dev-deps. Each commit leaves the crate compiling and its task's verify command green.

## Known Stubs

None. The parse-wide fields (categorical, leaf-vector, DART, multiclass) are intentionally recognized-but-unused per D-04 (verified in Phase 5), not stubs that block this plan's goal — the JSON byte-fidelity + 1e-5 goals are fully achieved with real data.

## Notes for Downstream Plans

- **03-03 (UBJSON):** add `treelite_xgboost::load_xgboost_ubjson(&[u8]) -> Result<Model, XgbError>` that decodes UBJSON → `serde_json::Value` → `serde_json::from_value::<json::XgbModelJson>` → `build_model_from_parsed`. Emit the same `"@NaN@"/"@Inf@"/"@-Inf@"` sentinel STRINGS for non-finite `d`/`D` floats so the shared `de_f32` adapter recovers them (the structs already carry `deserialize_with = de_vec_f32`). `XgbModelJson`/`build_model_from_parsed` are `pub(crate)` — UBJSON code lives in the same crate.
- **03-04 (legacy):** add `treelite_xgboost::load_xgboost_legacy(&[u8]) -> Result<Model, XgbError>`; fill the same logical fields (sum_hess from `NodeStat.sum_hess`, gain from `NodeStat.loss_chg`) and call `build_model_from_parsed` (or replicate its tail). The fixture carries a `binf` magic prefix (peekable-reader per D-07). Once both entry points exist, `three_format_equivalence` goes green WITHOUT changes to it.
- `parse_base_score` is `pub` (re-exported) and ready for the legacy path's `major_version >= 1` gate.

## Self-Check: PASSED

All 3 created files (json.rs, tests/nan_inf.rs, tests/json.rs) exist on disk; both per-task commits (914ac75, b1c9626) are in git history.

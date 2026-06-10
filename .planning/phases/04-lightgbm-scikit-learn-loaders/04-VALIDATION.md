---
phase: 4
slug: lightgbm-scikit-learn-loaders
status: ready
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` / `cargo test` + `anyhow` (ERR-02) in tests |
| **Config file** | none — workspace `Cargo.toml` members |
| **Quick run command** | `cargo test -p treelite-lightgbm -p treelite-sklearn -p treelite-gtil -p treelite-builder` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-edit>` (the quick run for the touched crate)
- **After every plan wave:** Run `cargo test --workspace` (preserves the Phase-3 XGBoost 1e-5 regression gate)
- **Before `/gsd-verify-work`:** Full suite must be green; max|delta| < 1e-5 on every per-estimator golden, recorded (EQV-04 spirit)
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | (enabler) D-05 | T-04-01 | f64 column-fill never OOB; typed BuilderError | unit | `cargo test -p treelite-builder f64` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | (enabler) D-05 | T-04-02 | bulk trust boundary pushed to callers | unit | `cargo test -p treelite-builder bulk_to_model` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 1 | (enabler) D-03 | T-04-04 | softmax/exp2 cast-order = 1e-5 contract | unit | `cargo test -p treelite-gtil postprocessor` | ❌ W0 | ⬜ pending |
| 04-02-02 | 02 | 1 | (enabler) D-03 | T-04-03 | class_id/target_id bounds-checked before output index | unit | `cargo test -p treelite-gtil output_shaping` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 1 | LGB-02/SKL-01..04 D-06/D-07 | T-04-05/06/SC | capture-only deps pinned; golden = treelite.gtil.predict | capture | `test -f fixtures/sklearn_*.golden.json && grep gtil fixtures/capture_sklearn.py` | ❌ W0 | ⬜ pending |
| 04-03-02 | 03 | 1 | LGB-01/LGB-02 D-06/D-07 | T-04-06 | LightGBM goldens from treelite GTIL + version pins | capture | `test -f fixtures/lightgbm_*.golden.json && grep gtil fixtures/capture_lightgbm.py` | ❌ W0 | ⬜ pending |
| 04-04-01 | 04 | 2 | LGB-01, LGB-03 | T-04-07/08/09 | parse counts validated before slice; no OOB/panic | unit | `cargo test -p treelite-lightgbm` | ❌ W0 | ⬜ pending |
| 04-04-02 | 04 | 2 | LGB-01 | T-04-07 | 1e-5 hard gate, never loosened | integration (golden) | `cargo test -p treelite-harness lightgbm_numerical` | ❌ W0 | ⬜ pending |
| 04-05-01 | 05 | 3 | LGB-02 | T-04-10/11 | cat_boundaries validated before BitsetToList slice | unit | `cargo test -p treelite-lightgbm bitset` | ❌ W0 | ⬜ pending |
| 04-05-02 | 05 | 3 | LGB-02 | T-04-12 | category_list slice bounds-checked | integration (golden) | `cargo test -p treelite-harness lightgbm_categorical` | ❌ W0 | ⬜ pending |
| 04-06-01 | 06 | 2 | SKL-01 | T-04-13/14 | children indices + node_count<=INT_MAX guarded | unit | `cargo test -p treelite-sklearn` | ❌ W0 | ⬜ pending |
| 04-06-02 | 06 | 2 | SKL-01, SKL-02 | T-04-15 | no GB leaf re-shrink (capture-side only) | integration (golden) | `cargo test -p treelite-harness sklearn_rf sklearn_gb` | ❌ W0 | ⬜ pending |
| 04-07-01 | 07 | 3 | SKL-03 | T-04-16/17 | ratio_c!=0 guard; golden == -score_samples | integration (golden) | `cargo test -p treelite-harness sklearn_iforest` | ❌ W0 | ⬜ pending |
| 04-08-01 | 08 | 4 | SKL-04 | T-04-18/19/21 | itemsize∈{52,56} + buffer-len guard; no transmute | unit + golden | `cargo test -p treelite-sklearn histgb_decode && cargo test -p treelite-harness sklearn_histgb_numerical` | ❌ W0 | ⬜ pending |
| 04-08-02 | 08 | 4 | SKL-04 | T-04-20 | bitset_idx/feature_idx bounds-checked | integration (golden) | `cargo test -p treelite-harness sklearn_histgb_categorical` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*No 3 consecutive tasks lack an automated verify — every task above carries an `<automated>` command (Dimension 8 satisfied).*

---

## Wave 0 Requirements

The Wave-1 enabler/capture plans (01, 02, 03) ARE the Wave-0 scaffolding — they create the
infrastructure every downstream golden test depends on:

- [ ] `crates/treelite-builder` f64 builder mode + `bulk_to_model` (Plan 01) — gates all f64-preset loaders
- [ ] `crates/treelite-gtil` output-shaping/averaging/base-score + 4 new postprocessors (Plan 02) — gates all golden asserts
- [ ] `fixtures/capture_sklearn.py` + `fixtures/capture_lightgbm.py` + per-estimator golden JSONs/manifests (Plan 03) — needs `scikit-learn`+`lightgbm` installed in the capture env (`uv pip install scikit-learn lightgbm`, capture-only)
- [ ] `crates/treelite-lightgbm/` crate + tests (Plan 04/05)
- [ ] `crates/treelite-sklearn/` crate + tests (Plan 06/07/08)
- [ ] `crates/treelite-harness/tests/{lightgbm,sklearn}.rs` per-estimator assert tests (extend the `golden_v5.rs` pattern)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| One-time golden capture (`uv run python` on main worktree) | D-06 | Capture needs `scikit-learn`+`lightgbm` installed in an untracked venv absent from worktrees; runs once, fixtures committed read-only | `uv pip install scikit-learn lightgbm` then `uv run python fixtures/capture_sklearn.py` and `fixtures/capture_lightgbm.py` on the main tree |

*All asserted phase behaviors (load→predict→1e-5) have automated `cargo test` verification; only the one-time fixture capture is a manual capture-env step (D-06).*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (Plans 01/02/03 are the scaffolding)
- [x] No watch-mode flags
- [x] Feedback latency ~30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-10

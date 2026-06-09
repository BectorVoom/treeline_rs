---
phase: 1
slug: end-to-end-spine
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust built-in) + `approx` for 1e-5 float assertions |
| **Config file** | none — Wave 0 stands up the workspace `Cargo.toml` |
| **Quick run command** | `cargo test -p <crate>` |
| **Full suite command** | `cargo build && cargo test` (all workspace members) |
| **Estimated runtime** | ~30–90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate>` (the crate touched by the task)
- **After every plan wave:** Run `cargo build && cargo test` (full workspace)
- **Before `/gsd-verify-work`:** Full suite must be green, including the equivalence-harness 1e-5 spine test against the committed golden vector
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

> Each phase task maps to an automated `cargo test` command (or, for the manual golden capture, a `test -f` + Python validation gate). The 1e-5 equivalence assertion (CORE-04 / Success Criterion 4) is the spine test (01-04 Task 2) and is automated; 01-04 Task 1 additionally unit-tests `run_equivalence` against a hand-computed scalar model so a wrong impl is caught before the golden gate.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| T1 Golden capture | 01-01 | 1 | CORE-04 / D-06 / D-07 | — | manual capture; sigmoid output in (0,1) | integration | `test -f fixtures/golden.json && test -f fixtures/binary_logistic.model.json && python -c "import json; g=json.load(open('fixtures/golden.json')); assert 'input' in g and 'output' in g and 'manifest' in g; assert all(0.0<v<1.0 for v in g['output'])"` | ⬜ W0 | ⬜ pending |
| T2 Workspace + enums | 01-01 | 1 | FND-01 / FND-02 / ENUM-01 / ERR-01 | T-01-01 | unknown enum string → typed Err, no panic | unit | `cargo build --workspace && cargo test -p treelite-core --test enums` | ⬜ W0 | ⬜ pending |
| T3 TreeBuf/Tree/Model | 01-01 | 1 | CORE-01 / CORE-02 / CORE-03 / CORE-04 | T-01-02 | borrowed-mode raw-ptr slice over valid memory only | unit | `cargo test -p treelite-core --test tree_buf --test tree_model` | ⬜ W0 | ⬜ pending |
| T1 Objective map + transform | 01-02 | 2 | ERR-01 / CORE-04 | T-02-02 | unrecognized objective → typed Err, no panic | unit | `cargo test -p treelite-xgboost --test error` | ⬜ W0 | ⬜ pending |
| T2 Load fixture → Model | 01-02 | 2 | ERR-01 / CORE-04 | T-02-01 | array-length mismatch → DimensionMismatch, no OOB | unit | `cargo test -p treelite-xgboost --test load_fixture` | ⬜ W0 | ⬜ pending |
| T1 Postprocessors | 01-03 | 2 | ERR-01 | T-03-02 | unsupported postprocessor → typed Err | unit | `cargo test -p treelite-gtil --test postprocessor` | ⬜ W0 | ⬜ pending |
| T2 EvaluateTree + predict | 01-03 | 2 | ERR-01 | T-03-01 / T-03-03 | OOB feature idx → typed Err; serial tree-sum | unit | `cargo test -p treelite-gtil --test predict` | ⬜ W0 | ⬜ pending |
| T1 Harness lib + run_equivalence unit | 01-04 | 3 | ERR-02 | T-04-01 | malformed golden → anyhow context; >1e-5 → Err | unit | `cargo test -p treelite-harness --test run_equivalence` | ⬜ W0 | ⬜ pending |
| T2 Spine test (1e-5 vs golden) | 01-04 | 3 | ERR-02 / CORE-04 | T-04-01 / T-04-02 | every output within 1e-5 of golden; manifest warns | integration | `cargo test -p treelite-harness --test equivalence -- --nocapture` | ⬜ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Workspace `Cargo.toml` with `resolver = "3"` and `[workspace.dependencies]` (FND-01, FND-02) — 01-01 Task 2
- [ ] `treelite-harness` crate scaffold + committed golden vector fixture + frozen toolchain/libm manifest (CORE-04) — 01-01 Task 1 (capture) + 01-04 (harness)
- [ ] One simple XGBoost-JSON model fixture captured from `treelite==4.7.0` wheel (golden capture) — 01-01 Task 1

*Every MISSING/❌ verify reference above is stood up by the listed Wave 0/Wave 1 task before the dependent test runs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Golden vector capture from upstream `treelite==4.7.0` | CORE-04 | Requires Python wheel + one-time capture script; output committed as fixture | Run `capture_golden.py` once; commit `golden.json` |

*All other phase behaviors have automated `cargo test` verification (see Per-Task Verification Map).*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (9/9 mapped above)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (every task has an automated command)
- [x] Wave 0 covers all MISSING references (workspace + fixtures + golden stood up in 01-01)
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (Per-Task Verification Map populated; run_equivalence now has a non-golden unit test catching >1e-5 deviation per 01-04 Task 1)

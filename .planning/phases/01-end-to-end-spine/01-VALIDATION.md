---
phase: 1
slug: end-to-end-spine
status: draft
nyquist_compliant: false
wave_0_complete: false
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

> Populated by the planner/executor as tasks are defined. Each task must map to an automated `cargo test` command or a Wave 0 dependency. The 1e-5 equivalence assertion (CORE-04 / Success Criterion 4) is the spine test and must be automated.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | ENUM-01 / CORE-01..04 / ERR-01..02 | — | N/A | unit | `cargo test` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Workspace `Cargo.toml` with `resolver = "3"` and `[workspace.dependencies]` (FND-01, FND-02)
- [ ] `treelite-harness` crate scaffold + committed golden vector fixture + frozen toolchain/libm manifest (CORE-04)
- [ ] One simple XGBoost-JSON model fixture captured from `treelite==4.7.0` wheel (golden capture)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Golden vector capture from upstream `treelite==4.7.0` | CORE-04 | Requires Python wheel + one-time capture script; output committed as fixture | Run `capture_golden.py` once; commit `golden.json` |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

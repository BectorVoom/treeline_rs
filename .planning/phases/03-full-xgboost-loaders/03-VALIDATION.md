---
phase: 3
slug: full-xgboost-loaders
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust edition 2024 workspace) |
| **Config file** | none — Cargo workspace; per-crate `tests/` + harness crate |
| **Quick run command** | `cargo test -p treelite-xgboost` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~{N} seconds (planner to confirm) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p treelite-xgboost`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** {N} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 1 | XGB-{XX} | T-3-01 / — | {expected secure behavior or "N/A"} | unit | `{command}` | ✅ / ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Three-format fixture generation (JSON + UBJSON via current xgboost; legacy binary via pinned old xgboost) — single `binary:logistic` logical model, frozen with generator manifest
- [ ] Shared prediction golden (input matrix + output vector) captured from upstream Treelite wheel
- [ ] Single v5 byte-fidelity golden blob (one upstream-serialized blob all three loaders must match — DEF-02-01 / D-10)
- [ ] Generation spike confirming the pinned old xgboost actually writes legacy binary with `version[0] >= 1` (A1/A2 from RESEARCH.md)

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| {behavior} | XGB-{XX} | {reason} | {steps} |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

---
phase: 5
slug: full-scalar-gtil-equivalence-harness
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust built-in) + `uv run python` for one-time golden capture |
| **Config file** | none — Cargo workspace; capture scripts under `fixtures/` |
| **Quick run command** | `cargo test -p treelite-gtil` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30–90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p treelite-gtil` (or the crate touched by the task)
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite must be green (all 1e-5 equivalence asserts pass)
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _to be filled by planner/executor_ | — | — | — | — | — | — | — | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fixtures/capture_gtil_matrix.py` — seeded dense + sparse CSR capture against upstream Treelite GTIL (absent CSR entries = NaN)
- [ ] Committed frozen golden matrices + output vectors + provenance manifests (with `backend: scalar-cpu` field)
- [ ] `crates/treelite-harness/tests/` — RED equivalence-matrix test that fails until the GTIL surface is widened

*Existing golden+manifest infrastructure (`treelite-harness` `Golden`/`Manifest`/`check_manifest`) is extended, not replaced.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cross-platform libm/RNG drift awareness | EQV-03 | CI asserts against committed matrices only; never re-draws from seed | Manifest records OS/arch/libm/toolchain; a 1e-5 miss is diagnosed by comparing manifests, not re-running capture |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

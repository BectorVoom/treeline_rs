---
phase: 7
slug: gpu-backend-equivalence-report
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> See `07-RESEARCH.md` § "Validation Architecture" for the requirement→test mapping this strategy is derived from.
> **Note:** GPU rows are hardware-gated. ROCm validates on the dev box; CUDA/wgpu absence is a SKIP, not a failure (D-05). The GPU equivalence report is OBSERVATIONAL — measured, not gated (D-01). The 1e-5 hard gate stays on the CPU spine (ScalarCpu/CubeclCpu) and is untouched.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust edition 2024 workspace) |
| **Config file** | none — workspace `Cargo.toml` + per-crate test modules / `crates/treelite-harness/tests/gtil_matrix.rs` |
| **Quick run command** | `cargo test -p treelite-cubecl` |
| **Full suite command** | `cargo test --workspace` (CPU gate); GPU report regenerated via the ROCm-feature harness run on dev hardware |
| **Estimated runtime** | ~planner to fill |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-edit>`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full CPU suite must be green (1e-5 gate untouched); GPU report regenerated on ROCm hardware
- **Max feedback latency:** planner to fill

---

## Per-Task Verification Map

> Planner/nyquist-auditor fills one row per task. GPU-device-dependent rows carry a skip-not-fail note.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 7-01-01 | 01 | 0 | GPU-03 | — | N/A | unit | `cargo test -p treelite-cubecl` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Device-absence spike (research Open Question A3): confirm whether cubecl CUDA/HIP `client()` FFI-aborts on a missing device — determines whether `DeviceUnavailable` (D-05) is catchable or needs a pre-flight device probe.
- [ ] `crates/treelite-cubecl` test stubs for the `R: Runtime` generalization (the cubecl-cpu test must keep compiling unchanged via a `predict_cpu` shim).

*Planner refines from RESEARCH.md § Validation Architecture.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ROCm GPU predictions for the harness model set | GPU-03 | Requires the developer's AMD/ROCm hardware; not CI-runnable | Run the harness with `--features rocm` on the ROCm box; confirm `Backend::Rocm` produces predictions for the frozen golden model set |
| Committed GPU equivalence report regeneration | GPU-04 | Numbers are regenerated only on real ROCm hardware (D-06) | Regenerate the report via the harness on the ROCm box; confirm per-class max-deviation + f64-fallback columns populate and match the committed file |

*Planner refines; CUDA/wgpu rows render "not run — no device" on this hardware (D-05/D-08).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (incl. the device-absence spike)
- [ ] No watch-mode flags
- [ ] Feedback latency target set
- [ ] Skip-not-fail semantics encoded for device-gated rows (D-05)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

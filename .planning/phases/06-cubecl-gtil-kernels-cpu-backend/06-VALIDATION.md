---
phase: 06
slug: cubecl-gtil-kernels-cpu-backend
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 06 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `approx::assert_abs_diff_eq!` (workspace dep `approx 0.5.1`) |
| **Config file** | none — `cargo test` |
| **Quick run command** | `cargo test -p treelite-cubecl` (kernel unit + spike tests) |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30–90 seconds (workspace) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p treelite-cubecl` (fast kernel + spike tests)
- **After every plan wave:** Run `cargo test --workspace` (no regression to the scalar matrix or loader/serializer gates)
- **Before `/gsd-verify-work`:** Full suite must be green; the cubecl matrix run asserts the SAME frozen `fixtures/gtil/` goldens scalar-cpu passes, to 1e-5, plus the SC2 determinism check
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

> Task IDs are assigned by the planner; this map captures the requirement → test-type contract each task must satisfy. `❌ W0` = test artifact created in Wave 0 (new `treelite-cubecl` crate), RED until kernels land.

| Requirement | Behavior | Test Type | Automated Command | File Exists |
|-------------|----------|-----------|-------------------|-------------|
| GPU-01 | Traversal + postproc run as `#[cube(launch)]` kernels; one unit/row, serial trees, no `continue` | unit + integration | `cargo test -p treelite-cubecl` | ❌ W0 (new crate) |
| GPU-02 | `Backend::CubeclCpu` default; frozen matrix passes 1e-5 in CI | integration | `cargo test -p treelite-harness --test gtil_matrix_cubecl` | ❌ W0 (registers `cubecl_cpu_case()`) |
| GPU-02 (determinism, SC2) | `predict` twice → element-wise `.to_bits()` equal | unit | `cargo test -p treelite-cubecl determinism` | ❌ W0 |
| GPU-05 | SoA per-column upload via `TreeBuf::as_bytes()`; one handle/column; forest round-trips | unit | `cargo test -p treelite-cubecl upload` | ❌ W0 |
| D-03 | Each postprocessor `#[cube]` port matches its scalar twin to 1e-5 (softmax f32-max/f64-norm cast order, `exp2`, f64 sigmoid/hinge twins) | unit | `cargo test -p treelite-cubecl postproc` | ❌ W0 |
| D-06 | Per-cell manifest records `cubecl-kernel` vs `scalar-fallback` provenance | integration | matrix test asserts provenance per cell | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/treelite-cubecl/` crate skeleton + `Cargo.toml` (cubecl 0.10.0 `features=["cpu"]`, bytemuck) — register in root `[workspace.members]` and pin in `[workspace.dependencies]`
- [ ] Spike test (break-free descend + f64 in-kernel + one postprocessor cast order) — the D-04 **confirmation** (not a gate), RED until kernels exist
- [ ] `tests/gtil_matrix_cubecl.rs` (sibling test file, or a `cases: &[RunnerCase]` loop that does NOT reshape the per-cell iteration) registering `cubecl_cpu_case()` — RED until kernels + registration land
- [ ] Determinism test (SC2) — two-run `.to_bits()` bit-identity
- [ ] `TreeBuf::as_bytes()` additive accessor — RED until added

*Note (D-11 smell guard): prefer a thin sibling test file or a `RunnerCase` list loop. If parameterizing the existing `gtil_matrix()` would require restructuring its per-cell iteration body, that is a smell against registration-not-refactor — add a sibling instead.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none) | — | All phase behaviors validate through `cargo test` against the frozen Phase-5 golden matrix + manifest | — |

*All phase behaviors have automated verification — the cubecl backend asserts against the identical committed `fixtures/gtil/` goldens scalar-cpu already passes.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (new `treelite-cubecl` crate + matrix registration)
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

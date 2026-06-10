---
phase: 5
slug: full-scalar-gtil-equivalence-harness
status: planned
nyquist_compliant: true
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
| 05-01-T1 | 01 | 1 | EQV-01, EQV-02 | T-05-01 | sha256 provenance per fixture; CI never regenerates | fixture capture | `uv run python fixtures/capture_gtil_matrix.py` | ❌ W0 | ⬜ pending |
| 05-01-T2 | 01 | 1 | EQV-01, EQV-02 | T-05-01 | RED scaffold; hard 1e-5 gate never loosened | scaffold (RED) | `cargo test -p treelite-harness --no-run` | ❌ W0 | ⬜ pending |
| 05-02-T1 | 02 | 2 | GTIL-03, GTIL-07 | T-05-04 | output_shape clamps malformed dims | unit | `cargo test -p treelite-gtil output_shape` | ❌ W0 | ⬜ pending |
| 05-02-T2 | 02 | 2 | GTIL-01, GTIL-08 | T-05-03 | saturating shape guard; serial tree-sum | unit + golden | `cargo test -p treelite-gtil` | ❌ W0 | ⬜ pending |
| 05-03-T1 | 03 | 3 | GTIL-04 | T-05-SC | verbatim cast order, f32 intermediates | unit | `cargo test -p treelite-gtil postprocessor` | ⚠️ partial | ⬜ pending |
| 05-03-T2 | 03 | 3 | GTIL-06, GTIL-05 | T-05-06, T-05-08 | full representability guard before u32 cast; NaN→default | unit | `cargo test -p treelite-gtil categorical` | ❌ W0 | ⬜ pending |
| 05-04-T1 | 04 | 4 | GTIL-02 | T-05-09, T-05-10 | col_ind/row_ptr bounds → typed error; absent=NaN | unit + golden | `cargo test -p treelite-gtil sparse` | ❌ W0 | ⬜ pending |
| 05-04-T2 | 04 | 4 | GTIL-03 | T-05-11 | bounds-safe leaf-vector per-tree access | golden | `cargo test -p treelite-gtil` | ❌ W0 | ⬜ pending |
| 05-05-T1 | 05 | 5 | EQV-04 | T-05-12 | serde(default) old-fixture compat; backend drift warn | build | `cargo test -p treelite-harness --no-run` | ❌ W0 | ⬜ pending |
| 05-05-T2 | 05 | 5 | EQV-03, EQV-04 | T-05-13 | hard 1e-5 gate; dense==sparse parity; max-dev report | golden (matrix) | `cargo test -p treelite-harness gtil_matrix -- --nocapture` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fixtures/capture_gtil_matrix.py` — seeded dense + sparse CSR capture against upstream Treelite GTIL (absent CSR entries = NaN), all 4 kinds, both input dtypes; dense-with-NaN == CSR asserted at capture time (Plan 05-01 Task 1)
- [ ] Committed frozen golden matrices + output vectors + provenance manifests (with `backend: scalar-cpu` field) under `fixtures/gtil/` (Plan 05-01 Task 1)
- [ ] `crates/treelite-harness/tests/gtil_matrix.rs` — RED equivalence-matrix test that fails until the GTIL surface is widened (Plan 05-01 Task 2)
- [ ] RED unit-test scaffolds in `treelite-gtil`: categorical full guard (`2^24+1` rejected), 3 new postprocessors, sparse NaN-fill (Plan 05-01 Task 2)

*Existing golden+manifest infrastructure (`treelite-harness` `Golden`/`Manifest`/`check_manifest`) is extended, not replaced.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cross-platform libm/RNG drift awareness | EQV-03 | CI asserts against committed matrices only; never re-draws from seed | Manifest records OS/arch/libm/toolchain + backend; a 1e-5 miss is diagnosed by comparing manifests, not re-running capture |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** planned (per-task map filled by planner)

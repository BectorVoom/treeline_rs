---
phase: 10
slug: parallel-scalar-inference
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `approx` (1e-5 asserts); pytest for the Python binding |
| **Config file** | none (cargo default test harness) |
| **Quick run command** | `cargo test -p treelite-gtil` |
| **Full suite command** | `cargo test --workspace` then `uv run pytest` (Python venv via `uv run`) |
| **Estimated runtime** | ~60–120 seconds (workspace) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p treelite-gtil`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd-verify-work`:** Full suite + `uv run pytest` green; `golden_v5.bin` / `golden_v5_3format.bin` byte-identical (untouched this phase); full `gtil_matrix` 1e-5 across both presets, both input dtypes, dense + sparse
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 10-W0 | 00 | 0 | PAR-03 | — | N/A | unit (compile) | `cargo test -p treelite-core --test model_invariants` | ❌ W0 (rewrite) | ⬜ pending |
| 10-W0 | 00 | 0 | PAR-04 | T-10-01 (unbounded nthread → thread exhaustion) | `nthread` bounds the pool; `≤0` = all cores | integration | `cargo test -p treelite-gtil --test parallel_nthread` | ❌ W0 (new) | ⬜ pending |
| 10-W0 | 00 | 0 | GTIL-08 | — | N/A | determinism | `cargo test -p treelite-gtil --test determinism` | ❌ W0 (new) | ⬜ pending |
| 10-xx | — | — | PAR-01 | — | N/A | equivalence | `cargo test -p treelite-harness --test gtil_matrix` | ✅ (reuse goldens) | ⬜ pending |
| 10-xx | — | — | PAR-02 | T-10-02 (CSR re-validated per row → DoS) | CSR validated ONCE up front, not per row | equivalence | `cargo test -p treelite-harness --test gtil_matrix` (sparse) | ✅ (reuse goldens) | ⬜ pending |
| 10-xx | — | — | PAR-01/02 | — | N/A | utilization | `cargo test -p treelite-gtil` (>1-core assert, gated on `available_parallelism()>1`) | ❌ W0 | ⬜ pending |
| 10-xx | — | — | PAR-04 | — | N/A | integration (py) | `uv run pytest crates/treelite-py -k nthread` | ❌ W0 (new) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Rewrite `crates/treelite-core/tests/model_invariants.rs` — replace `_assert_not_send` with `requires_sync::<Model>()` (PAR-03). The existing `size_of::<Model>()` budget test stays.
- [ ] `crates/treelite-gtil/tests/parallel_nthread.rs` (new) — `nthread=1` vs `nthread=0` produce identical output; a multi-row run with `nthread≤0` uses `>1` core (skip if `available_parallelism()==1`) (PAR-04, utilization).
- [ ] `crates/treelite-gtil/tests/determinism.rs` (new) — run `predict`/`predict_sparse` N times on a fixed input, assert byte-identical `Vec<O>` (GTIL-08 under parallelism).
- [ ] `treelite-py` pytest (new) — `gtil.predict(..., nthread=2)` returns values identical to `nthread=1` within 1e-5 over a LightGBM/categorical fixture (the scalar-fallback path) (PAR-04 end-to-end).
- [ ] The frozen `fixtures/gtil/*.golden.json` matrix is REUSED verbatim as the parallel-vs-serial 1e-5 gate — no new goldens needed (parallel output must equal the existing serial-captured goldens).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real wall-clock speedup (3–5×) | PAR-01/02 | Throughput is environment-sensitive; CI asserts correctness + >1-core, not a hard speed number | Throwaway bench over a large LightGBM batch; compare serial vs parallel M rows/s (dev box only) |

*The utilization test (`>1` core used) is the automated proxy for "it parallelizes"; absolute speedup stays manual to avoid a flaky timing gate.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

---
phase: 2
slug: builder-serialization
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + `approx` for float tolerance (Phase 1 pattern) |
| **Config file** | none — Cargo convention (`tests/` integration tests per crate) |
| **Quick run command** | `cargo test -p <crate-under-edit>` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30 seconds full suite (no C++ compile; fixtures are tiny) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <crate-under-edit>` (quick).
- **After every plan wave:** Run `cargo test --workspace`.
- **Before `/gsd-verify-work`:** `cargo test --workspace` green + `golden_v5` byte-compare green + the existing 1e-5 equivalence green.
- **Max feedback latency:** ~30 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | SER-01 (setup) | — | N/A (struct fields) | unit | `cargo test -p treelite-core --lib` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | SER-01 (D-02) | T-02-S03 | First 12 bytes == (4,7,0); no mis-versioned golden | manual+script | `python fixtures/capture_golden_v5.py` (one-time, local venv) | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 1 | SER-01 (D-01/D-02) | T-02-S03 | sha256/nbytes manifest consistent; version header (4,7,0) | integration | (verify in golden_v5.bin check, Plan 03) | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 2 | BLD-01 | T-02-B01 | Reject negative/dangling/orphan/duplicate keys with typed error, no panic | unit | `cargo test -p treelite-builder --test validation` | ❌ W0 | ⬜ pending |
| 2-02-02 | 02 | 2 | BLD-02, BLD-03 | T-02-B03 | Concat rejects mismatched variant; bulk bypass trusts pre-validated input | unit | `cargo test -p treelite-builder --test concat --test bulk` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 2 | SER-01, D-01 | T-02-S05 | Byte-for-byte == golden_v5.bin; NaN bits preserved | integration | `cargo test -p treelite-harness --test golden_v5` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 2 | SER-01, SER-02, D-03 | T-02-S01 / S02 / S04 | Bound u64 counts vs buffer; reject truncation/non-v5; no panic/OOM; zero-copy frames | unit | `cargo test -p treelite-core --test serialize_roundtrip --test serialize_pybuffer` | ❌ W0 | ⬜ pending |
| 2-04-01 | 04 | 3 | SER-03, D-04 | T-02-J01 | Structure value-diffable vs json_serializer.cc | integration | `cargo test -p treelite-core --test dump_json` | ❌ W0 | ⬜ pending |
| 2-04-02 | 04 | 3 | SER-04 | T-02-J02 | Read-only fields expose no setter | unit | `cargo test -p treelite-core --test fields` | ❌ W0 | ⬜ pending |
| 2-05-01 | 05 | 3 | BLD-01, D-11 | T-02-X01 / X02 | Builder errors propagate as typed XgbError, no panic crosses loader | unit/integration | `cargo test -p treelite-xgboost` | ✅ partial | ⬜ pending |
| 2-05-02 | 05 | 3 | BLD-01, D-11 | T-02-X02 | Predictions stay within 1e-5 after rewiring | integration | `cargo test -p treelite-harness --test equivalence` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fixtures/golden_v5.bin` + `fixtures/golden_v5.manifest.json` — capture from the installed treelite 4.7.0 wheel (D-02). **Blocks the byte-fidelity test; Plan 01 Task 3 does this FIRST.**
- [ ] `crates/treelite-core/tests/serialize_roundtrip.rs` — SER-01 round-trip + malformed-input rejection (Plan 03).
- [ ] `crates/treelite-core/tests/serialize_pybuffer.rs` — SER-02 zero-copy frame order + as_ptr equality (Plan 03).
- [ ] `crates/treelite-core/tests/dump_json.rs` — SER-03/D-04 structural diff (Plan 04).
- [ ] `crates/treelite-core/tests/fields.rs` — SER-04 accessors (Plan 04).
- [ ] `crates/treelite-harness/tests/golden_v5.rs` — SER-01/D-01/D-02 byte-compare (Plan 03).
- [ ] `crates/treelite-builder/tests/{validation,concat,bulk}.rs` — BLD-01/02/03 (Plan 02).
- [ ] No framework install needed (Rust built-in harness; `approx` already pinned).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Golden v5 blob capture | SER-01 / D-02 | Requires the developer's local Python `.venv` with `treelite==4.7.0` + `xgboost==3.2.0`; CI never compiles/runs the wheel. One-time freeze. | `python fixtures/capture_golden_v5.py`; assert first 12 bytes == (4,7,0); commit blob + manifest. (Plan 01 Task 2 checkpoint.) |
| 1e-5 regression after rewiring | BLD-01 / D-11 | Final human gate confirming the builder rewiring did not perturb predictions before phase verification. | `cargo test --workspace` green + inspect equivalence max |delta| < 1e-5. (Plan 05 Task 2 checkpoint.) |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-06-10

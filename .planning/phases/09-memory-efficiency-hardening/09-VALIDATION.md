---
phase: 9
slug: memory-efficiency-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 09-RESEARCH.md §Validation Architecture. The two hard invariants
> (byte-identical v5 golden + 1e-5 equivalence harness, D-03/D-05/D-11) are the
> real pass/fail gate; everything below samples toward keeping them green.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`cargo test`) + pytest (`treelite-py`) |
| **Config file** | none (workspace default); `crates/treelite-py` pytest conftest |
| **Quick run command** | `cargo test -p <touched-crate>` (e.g. `cargo test -p treelite-core`) |
| **Full suite command** | `cargo test --workspace` + `uv run pytest crates/treelite-py` |
| **Estimated runtime** | ~60–120 seconds (workspace); pytest adds ~tens of seconds |

> Python is invoked via `uv run` (not bare `python`) — venv/pyproject are untracked and absent from worktrees; run on the main tree.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p <touched-crate>` (e.g. `-p treelite-core` after `model.rs`/`tree_buf.rs` edits).
- **After every plan wave:** Run `cargo test --workspace` + `cargo test -p treelite-harness --test golden_v5`.
- **Before `/gsd-verify-work`:** Full suite green + golden byte-compare green + pytest green within 1e-5 (D-11).
- **Max feedback latency:** ~120 seconds.

---

## Per-Task Verification Map

> Seeded from RESEARCH §"Phase Requirements → Test Map". Task IDs are assigned by the planner; the Nyquist auditor reconciles this map against the final PLAN.md task set.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | MEM-02 | — | MEM-02 | — | SmallVec/CompactString deref → byte-identical v5 | integration | `cargo test -p treelite-harness --test golden_v5` | ✅ `tests/golden_v5.rs` | ⬜ pending |
| TBD | MEM-02 | — | MEM-02 | — | workspace compiles + tests green after field-type ripple | build+unit | `cargo test --workspace` | ✅ all crate tests | ⬜ pending |
| TBD | MEM-02 | — | MEM-02 | — | py list/str accessor contract preserved (if getters touched) | integration | `uv run pytest crates/treelite-py` | ✅ pytest suite | ⬜ pending |
| TBD | MEM-02 | — | MEM-02 | — | `Model` stays `!Send`; `size_of::<Model>()` not bloated | unit | `cargo test -p treelite-core` (new `model_stays_not_send`) | ❌ W0 | ⬜ pending |
| TBD | MEM-01 | — | MEM-01 | — | serializer Pod recast keeps emitted bytes identical | integration | `cargo test -p treelite-harness --test golden_v5` | ✅ `tests/golden_v5.rs` (the gate) | ⬜ pending |
| TBD | MEM-01 | — | MEM-01 | — | `TreeBuf::as_bytes` roundtrip after bound change | unit | `cargo test -p treelite-core tree_buf` | ✅ `tree_buf.rs` tests | ⬜ pending |
| TBD | MEM-01 | — | MEM-01/02 | — | full equivalence within 1e-5 | integration | `cargo test -p treelite-harness` (equivalence/matrix) | ✅ `equivalence.rs`, `gtil_matrix*.rs` | ⬜ pending |
| TBD | MEM-03 | — | MEM-03 | — | allocator builds + runs on Linux (jemalloc + mimalloc) | smoke | `cargo run -p treelite-harness --features jemalloc` (and `--features mimalloc`) | ❌ W0 | ⬜ pending |
| TBD | MEM-03 | — | MEM-03 | — | abi3 wheel stays allocator-free | smoke | `cargo tree -p treelite-py \| grep -E "jemalloc\|mimalloc"` empty | ❌ W0 | ⬜ pending |
| TBD | MEM-03 | — | MEM-03 | — | committed `MEMORY_REPORT.md` regenerates | manual/`#[ignore]` | new bench/bin writing the report | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/treelite-core/tests/` — `model_stays_not_send` + `size_of::<Model>()` sanity test (catches the !Send / header-bloat pitfalls).
- [ ] `crates/treelite-harness/benches/memory_report.rs` (or `src/bin/`) — allocator-gated RSS sampler + `MEMORY_REPORT.md` writer (MEM-03 / D-10).
- [ ] `crates/treelite-harness/Cargo.toml` — `jemalloc` / `mimalloc` non-default, mutually-exclusive features + optional deps.
- [ ] Wheel-isolation check (`cargo tree -p treelite-py` grep) wired as a test or script step.
- [ ] Dep install: add the crates (`bytemuck` already pinned; `smallvec`, `compact_str`, `tikv-jemallocator`, `tikv-jemalloc-ctl`, `mimalloc`) to `[workspace.dependencies]`; confirm exact pinned versions via `cargo add --dry-run`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Peak-RSS before/after improvement narrative | MEM-03 / D-10 | Observational report, no brittle CI threshold (D-10) | Run the memory-report bench under each allocator feature; inspect committed `MEMORY_REPORT.md` for the before/after delta |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

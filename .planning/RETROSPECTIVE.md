# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — Parallel Scalar Inference

**Shipped:** 2026-06-11
**Phases:** 1 (Phase 10) | **Plans:** 2 | **Tasks:** 5

### What Was Built
- Row-parallel scalar GTIL (dense + sparse CSR) via rayon `par_chunks_mut`/`map_init`, inner `tree_id` loop kept serial (GTIL-08).
- Sound `unsafe impl Sync` for `Model` and `TreeBuf<T>` (read-only-predict argument), superseding the Phase-9 `!Send` invariant.
- End-to-end `Config.nthread` honoring via a scoped `ThreadPool` (never `build_global`), wired through to the Python `nthread=` kwarg.

### What Worked
- The Wave-0 RED-scaffold / Wave-1 implementation split gave Wave 1 a precise byte-identical-determinism + nthread-equivalence contract to build against.
- Measuring throughput with a real categorical LightGBM model (4M rows) turned the one human-judgment UAT item into a hard number (3.68×) instead of a guess.
- Adversarial passes paid off: code review caught a real overflow guard gap (WR-03) and tightened the `Sync` bound (WR-02); the security audit verified all 5 plan-time threats against actual code with file:line evidence.

### What Was Inefficient
- Worktree isolation is unusable in this repo (dirty untracked vendored trees + untracked Python venv) — both execute-phase and the code-review `--fix` fixer had to be re-verified on `main` after the worktree reported phantom LightGBM fixture failures. Sequential-on-main is the standing workaround.
- The milestone-close pre-flight surfaced stale v1.0 debt (Phase 05 deferred verification items) that had to be resolved inline before closing — a sign v1.0 was never formally closed.

### Patterns Established
- **Throughput UAT via throwaway harness:** for performance-magnitude human-verification items, write a temporary bench (load real fixture, time nthread=1 vs all-cores, assert 1e-5), report the number, then delete it.
- **Saturation-aware divergence guards:** assert f32-vs-f64 distinctness on the *raw margin* (never saturates), not post-sigmoid output (saturates to 1.0 for large margins) — fixed the fragile WR-06 guard.

### Key Lessons
1. Post-subagent IDE diagnostics in this repo are routinely phantom; always confirm with `cargo test --workspace` (+ `uv run pytest`) on `main` before reacting.
2. Close milestones formally as you go — deferred verification items accumulate as cross-milestone debt that blocks the next close.
3. Empirically verify a load-bearing comment before trusting it: the "raw shares the f64-accumulated margin" claim was wrong (raw diverges 2.9e-6).

### Cost Observations
- Model mix: executor/auditor/fixer on opus; verifier/reviewer on sonnet.
- Notable: single-phase milestone, but the close swept the whole project (first formal close) — v1.0 phases 1–9 archived alongside v1.1.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Key Change |
|-----------|--------|------------|
| v1.0 MVP (1–9) | 9 | Vertical-slice spine; shipped incrementally, never formally closed until v1.1. |
| v1.1 (10) | 1 | First formal milestone close; introduced the project's first parallelism + first `unsafe` concurrency, both adversarially reviewed + security-audited. |

### Cumulative Quality

| Milestone | Workspace Tests | 1e-5 Contract | New Runtime Deps |
|-----------|-----------------|---------------|------------------|
| v1.0 | green | held (worst GPU 2.9e-6) | core port deps |
| v1.1 | 310 passed / 0 failed | held (parallel bit-identical to serial) | rayon 1.12.0 |

### Top Lessons (Verified Across Milestones)

1. The 1e-5 contract is the spine — every milestone gates on the golden harness, and it has never regressed.
2. Worktree isolation is unsafe in this repo; run sequentially on `main` and re-verify any sub-agent's pass/fail claims there.

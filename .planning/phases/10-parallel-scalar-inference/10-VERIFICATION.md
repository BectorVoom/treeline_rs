---
phase: 10-parallel-scalar-inference
verified: 2026-06-11T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Observe wall-clock speedup on a LightGBM categorical or kLE model over a large row batch"
    expected: "Multi-core run completes faster than nthread=1; the utilization test asserts rayon::current_num_threads() > 1 but does not measure throughput"
    why_human: "Functional correctness and determinism are proven by automated tests. Actual throughput improvement (the point of the phase) requires running the model on a real scalar-fallback workload with wall-clock timing — not automatable without a benchmark harness that is out of scope for this phase"
    result: "PASSED 2026-06-11 via /gsd-verify-work — categorical LightGBM, 4M rows, 16 cores: nthread=1 0.708s vs all-cores 0.192s = 3.68x speedup; output bit-identical (max diff 0.000e0, within 1e-5). Recorded in 10-UAT.md."
---

# Phase 10: Parallel Scalar Inference — Verification Report

**Phase Goal:** Row-parallelize the single-threaded scalar GTIL fallback engine (treelite_gtil::predict dense and predict_sparse/predict_cpu_sparse) across all available CPU cores — the whole-model path for LightGBM (kLE), categorical, every non-kLT, and all sparse models — so those models stop running on one core, while producing output identical to the current serial path within 1e-5. Honor Config.nthread.

**Verified:** 2026-06-11T00:00:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Step 0: Previous Verification

No previous VERIFICATION.md found. Initial verification mode.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Scalar dense predict runs row-parallel: a multi-row run uses more than one core (PAR-01) | VERIFIED | `par_chunks_mut` at lib.rs:716 (traversal), :775 (RF-averaging), :797 (base-score). `parallel_uses_more_than_one_core` test asserts `rayon::current_num_threads() > 1`. No `par_iter` over trees — inner loop stays serial (GTIL-08). |
| 2 | Scalar sparse predict runs row-parallel under same 1e-5 equivalence; CSR validated once up front (PAR-02) | VERIFIED | `predict_sparse` routes through `predict_rows` → same parallel `predict_preset`/`predict_leaf_preset`/`predict_score_by_tree_preset` as dense. `csr.validate(num_row, num_feature)` at lib.rs:1038 runs before the parallel section. cubecl `predict_cpu_sparse` (lib.rs:369) forwards `cfg` unchanged to `treelite_gtil::predict_sparse`. `determinism_sparse_byte_identical_n_runs` passes. |
| 3 | Per-row tree summation stays serial in tree_id order (GTIL-08) | VERIFIED | No `par_iter` over `trees` found anywhere in lib.rs. `par_chunks_mut` parallelizes only the outer row axis. `determinism_byte_identical_n_runs` (64 rows, 4 kinds, N=4 runs) proves no run-to-run reordering. |
| 4 | Parallel output is byte-identical to the serial path across repeated runs (determinism, PAR-04) | VERIFIED | `determinism_byte_identical_n_runs` (dense, all 4 PredictKinds, N=4 runs, `.to_bits()` assertion) passes. `determinism_sparse_byte_identical_n_runs` (fully-dense CSR, 4 kinds, N=4 runs) passes. |
| 5 | Config.nthread honored end-to-end: <=0 = all cores (global pool), N = bounded scoped pool; never build_global (PAR-02/PAR-04) | VERIFIED | `run_with_nthread` helper at lib.rs:649-663: `nthread <= 0` calls `fill()` on global pool; `nthread > 0` builds `ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| GtilError::ThreadPool(...))?.install(fill)`. Only occurrence of `build_global` in the file is in the doc comment at :646 (not code). `nthread_equivalence` test (nthread 0/1/2 byte-identical over 4 PredictKinds) passes. |
| 6 | Python nthread= kwarg drives the scalar predict path end-to-end (PAR-04) | VERIFIED | `make_config(nthread, pred_margin)` in gtil.rs:62-71 passes nthread directly to `Config { nthread, .. }`. Config is forwarded through `dispatch_backend` → `predict_cpu` → the scalar fallback `treelite_gtil::predict`. `test_nthread.py` (2 tests: nthread=2==1 and nthread=-1==1 within 1e-5 over LightGBM categorical fixture) passes. |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-gtil/src/lib.rs` | rayon par_chunks_mut conversion of all 4 row loops + run_with_nthread + per-row output_leaf_* slice writes | VERIFIED | `par_chunks_mut` appears 8 times (traversal :716, averaging :775, base-score :797, leaf-preset :1162, score-tree :1248 + doc references). `run_with_nthread` helper at :649. `output_leaf_value_row` at :817, `output_leaf_vector_row` at :856 (per-row slice variants using `cell_in_row`). |
| `crates/treelite-gtil/tests/determinism.rs` | N-run byte-identical determinism test (un-ignored, green) | VERIFIED | `determinism_byte_identical_n_runs` (dense) and `determinism_sparse_byte_identical_n_runs` (sparse) — both un-ignored, both green. No `#[ignore]` attributes found. |
| `crates/treelite-gtil/tests/parallel_nthread.rs` | nthread equivalence + >1-core utilization test (un-ignored, green) | VERIFIED | `nthread_equivalence` and `parallel_uses_more_than_one_core` — both un-ignored, both green. |
| `crates/treelite-py/tests/python/test_nthread.py` | pytest: gtil.predict nthread=2 == nthread=1 within 1e-5 over scalar-fallback fixture | VERIFIED | 2 tests pass: `test_nthread_two_equals_one_scalar_fallback` and `test_nthread_all_cores_equals_one_scalar_fallback`. Located at `tests/python/` (plan stated `tests/` — auto-corrected by executor for conftest discovery). |
| `crates/treelite-core/src/model.rs` | unsafe impl Sync for Model with SAFETY comment | VERIFIED | `unsafe impl Sync for Model {}` at :130 with 5-point SAFETY block. No `unsafe impl Send for Model`. |
| `crates/treelite-core/src/tree_buf.rs` | unsafe impl Sync for TreeBuf<T> (blocking-fix for rayon parallel closure) | VERIFIED | `unsafe impl<T: Copy> Sync for TreeBuf<T> {}` at :49 with SAFETY doc mirroring Model. |
| `crates/treelite-core/tests/model_invariants.rs` | requires_sync::<Model>() positive test; size budget retained | VERIFIED | `model_is_sync_for_readonly_predict` (requires_sync + requires_send::<&Model>) and `model_size_not_bloated_by_smallvec` both pass. `_assert_not_send` absent (superseded). |
| `crates/treelite-gtil/src/error.rs` | GtilError::ThreadPool(String) variant | VERIFIED | `ThreadPool(String)` at :179, attributed `#[error("failed to build thread pool: {0}")]`, before the `#[error(transparent)]` catch-all. |
| `crates/treelite-gtil/src/config.rs` | nthread doc corrected (no longer "recorded but never used") | VERIFIED | Stale "recorded but never used" / "ignores it for allocation" doc text absent. Doc now correctly describes the pool semantics (<=0 all cores, N bounded scoped pool). |
| `crates/treelite-py/src/gtil.rs` | make_config doc note corrected (nthread drives scalar path) | VERIFIED | Doc at :57-60 states "nthread now drives the scalar predict path end-to-end (Phase 10, PAR-04)". No "recorded but unused" wording. |
| `Cargo.toml` | rayon = "1.12.0" in [workspace.dependencies] | VERIFIED | Line 39: `rayon = "1.12.0"`. |
| `crates/treelite-gtil/Cargo.toml` | rayon = { workspace = true } | VERIFIED | Line 10: `rayon = { workspace = true }`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/treelite-gtil/src/lib.rs predict_rows` | `predict_preset / predict_leaf_preset / predict_score_by_tree_preset` | nthread threaded from Config | VERIFIED | config.nthread passed at lib.rs:1092, :1095 (predict_preset); :1066, :1069 (predict_leaf); :1216, :1219 (score_by_tree) |
| `crates/treelite-gtil/src/lib.rs` | rayon worker pool | `run_with_nthread` wrapping `ThreadPoolBuilder::num_threads(n).build().install()` when nthread > 0 | VERIFIED | `run_with_nthread` at :649; `ThreadPoolBuilder::new().num_threads(nthread as usize).build()` at :657-660 |
| `crates/treelite-py/src/gtil.rs make_config` | treelite_gtil scalar predict | Config.nthread consumed downstream | VERIFIED | `make_config(nthread, pred_margin)` → `Config { nthread, .. }` → forwarded via `dispatch_backend` → `predict_cpu` → `treelite_gtil::predict` |
| `crates/treelite-cubecl/src/lib.rs predict_cpu` | `treelite_gtil::predict` (scalar fallback) | `cfg` forwarded at line 323 | VERIFIED | `treelite_gtil::predict::<F>(model, data, num_row, cfg)` at cubecl/src/lib.rs:323 |
| `crates/treelite-cubecl/src/lib.rs predict_cpu_sparse` | `treelite_gtil::predict_sparse` | `cfg` forwarded at line 369 | VERIFIED | `treelite_gtil::predict_sparse::<F>(model, csr, num_row, cfg)` at cubecl/src/lib.rs:369 |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `determinism.rs` | `first`/`next` Vec<f64> | `treelite_gtil::predict::<f64>(&model, &data, ...)` | Yes — actual parallel traversal over a 2-tree model with 64 rows | FLOWING |
| `parallel_nthread.rs` | `baseline`/`got` Vec<f64> | `treelite_gtil::predict::<f64>(&model, &data, ...)` with varying nthread | Yes — actual rayon-pool-scoped traversal | FLOWING |
| `test_nthread.py` | `out_n1`/`out_n2` numpy arrays | `gtil.predict(rs_model, data, nthread=...)` over loaded LightGBM categorical model | Yes — real model loaded from fixture, real numpy data | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Determinism tests pass (dense + sparse, 4 kinds, N=4 runs) | `cargo test -p treelite-gtil --test determinism` | 2 passed, 0 failed | PASS |
| nthread equivalence + >1-core utilization | `cargo test -p treelite-gtil --test parallel_nthread` | 2 passed, 0 failed | PASS |
| 1e-5 golden gate (all 4 kinds, both presets, dense + sparse) | `cargo test -p treelite-harness --test gtil_matrix` | 1 passed, 0 failed | PASS |
| Model Sync invariant + size budget | `cargo test -p treelite-core --test model_invariants` | 2 passed, 0 failed | PASS |
| Python nthread pytest | `uv run pytest crates/treelite-py/tests/python/test_nthread.py -q` | 2 passed | PASS |
| Full workspace suite | `cargo test --workspace` | All test results: ok, 0 failed, 0 ignored | PASS |

---

### Probe Execution

No probe files declared in PLAN or VALIDATION for this phase. Step 7c: SKIPPED (phase uses cargo test + pytest as verification gates; no probe-*.sh scripts).

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| PAR-01 | 10-01-PLAN.md | Scalar dense predict runs row-parallel with 1e-5 parity and GTIL-08 serial inner loop | SATISFIED | `par_chunks_mut` at lib.rs:716; inner tree loop serial (no par_iter over trees); `gtil_matrix` golden passes; `parallel_uses_more_than_one_core` green |
| PAR-02 | 10-01-PLAN.md | Scalar sparse predict runs row-parallel under same equivalence; CSR validated once up front | SATISFIED | `predict_sparse` routes same parallel body; `csr.validate` at lib.rs:1038 before parallel section; cubecl sparse forwards cfg; `determinism_sparse_byte_identical_n_runs` passes |
| PAR-03 | 10-00-PLAN.md | Model soundly Sync for read-only predict; _assert_not_send superseded | SATISFIED | `unsafe impl Sync for Model` at model.rs:130; `unsafe impl Sync for TreeBuf<T>` at tree_buf.rs:49; `model_is_sync_for_readonly_predict` passes; `_assert_not_send` absent |
| PAR-04 | 10-01-PLAN.md | Config.nthread honored end-to-end; Python nthread= kwarg drives scalar path | SATISFIED | `run_with_nthread` helper; `nthread_equivalence` green (0/1/2); `test_nthread.py` 2 tests green |
| GTIL-08 | Cross-phase (v1) | Per-row tree summation serial in tree_id order | SATISFIED | No `par_iter` over trees; only row-axis `par_chunks_mut`; determinism tests prove no reordering |

All four PAR requirements covered. No orphaned requirements in REQUIREMENTS.md (traceability table shows PAR-01..PAR-04 mapped to Phase 10 only).

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|---------|--------|
| `crates/treelite-gtil/src/lib.rs` | 646 | `build_global` in doc comment (not code) | Info | Not a code anti-pattern — doc warns against it. Correct. |

No TBD, FIXME, or XXX markers found in any phase-modified file. No placeholder or stub patterns. No empty handlers.

**Code Review Warnings (from 10-REVIEW.md, status: issues_found, critical: 0):**

Three warnings were identified in the code review but none block the phase goal:

- **WR-01** (warning): `unsafe impl Send for SendModelRef` relies on unenforced convention that model has no `Borrowed` TreeBuf columns. Latent (all current loader-produced models use Owned columns). Pre-existing pattern from Phase 8, not a Phase 10 regression.
- **WR-02** (warning): `unsafe impl<T: Copy> Sync for TreeBuf<T>` bound could be `T: Copy + Sync` to prevent a hypothetical `TreeBuf<Cell<f32>>` from auto-claiming Sync. All current T types (f32, f64, i32, u32, u64, bool, Operator, TreeNodeType) are Sync; zero behavioral impact today.
- **WR-03** (warning): `predict_score_by_tree_preset` output allocation (`num_row * num_tree * lvs`) uses unchecked multiplication. Harmless on 64-bit (the target platform per CLAUDE.md). Would require `checked_mul` for 32-bit safety.

These match the code review's own classification (0 critical, 3 warnings). None cause incorrect output or test failures.

---

### Human Verification Required

#### 1. Scalar-path throughput on a real LightGBM or categorical model

**Test:** Load a LightGBM (kLE) or categorical model via the Python binding. Run `gtil.predict(model, large_matrix, nthread=0)` (all cores) and time it. Compare to a prior serial measurement or to `nthread=1`.

**Expected:** Wall-clock time is reduced by roughly the core count (the prototype claimed 3.0-4.6x). The `parallel_uses_more_than_one_core` test asserts `rayon::current_num_threads() > 1` after a predict call, but does not measure throughput.

**Why human:** Throughput measurement requires a benchmarking harness with timing, which is not part of this phase's test suite. The automated suite proves correctness and parallelism structure, but not the actual speedup magnitude.

---

## Gaps Summary

No gaps. All 6 observable truths are VERIFIED. All 12 required artifacts pass existence, substantive content, and wiring checks. All 4 PAR requirements are satisfied. No debt markers. No blocker anti-patterns.

The `human_needed` status is set because one human verification item (throughput measurement) was identified in Step 8. All automated correctness checks pass.

---

_Verified: 2026-06-11T00:00:00Z_
_Verifier: Claude (gsd-verifier)_

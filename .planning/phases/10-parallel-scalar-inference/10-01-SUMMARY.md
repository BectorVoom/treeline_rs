---
phase: 10-parallel-scalar-inference
plan: 01
subsystem: gtil
tags: [rayon, gtil, parallelism, determinism, nthread, sync, par_chunks_mut]

# Dependency graph
requires:
  - phase: 10-parallel-scalar-inference
    plan: 00
    provides: "rayon 1.12.0 pinned + wired, unsafe impl Sync for Model, GtilError::ThreadPool, determinism.rs/parallel_nthread.rs RED scaffolds"
provides:
  - "row-parallel scalar GTIL engine: all 4 row-loop families converted to rayon par_chunks_mut/map_init"
  - "run_with_nthread scoped-pool helper honoring Config.nthread (<=0 all cores, N bounded scoped pool)"
  - "per-row-slice output_leaf_value_row/output_leaf_vector_row (borrow-checker-proven disjoint writes)"
  - "unsafe impl Sync for TreeBuf<T> (read-only predict) so &[Tree<T>] shares across rayon workers"
  - "green determinism (dense+sparse) + nthread-equivalence + >1-core utilization gtil tests"
  - "test_nthread.py: Python nthread kwarg drives the scalar pool end-to-end within 1e-5"
affects: [parallel-scalar-inference, gtil-predict, wave-1, treelite-py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "rayon par_chunks_mut over a row-chunked output buffer + map_init per-worker scratch (Pitfall 2: allocate once per worker)"
    - "scoped rayon::ThreadPool via run_with_nthread helper (never build_global; build Err -> typed GtilError::ThreadPool)"
    - "per-row &mut [O] slice writes (cell_in_row indexing) making per-row disjointness statically provable"
    - "Result<(), GtilError> closure + collect::<Result>()? short-circuit preserving ERR-01 typed errors across the parallel boundary"

key-files:
  created:
    - crates/treelite-py/tests/python/test_nthread.py
  modified:
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/src/config.rs
    - crates/treelite-core/src/tree_buf.rs
    - crates/treelite-gtil/tests/determinism.rs
    - crates/treelite-gtil/tests/parallel_nthread.rs
    - crates/treelite-py/src/gtil.rs

key-decisions:
  - "unsafe impl Sync for TreeBuf<T> added to treelite-core (Rule 3 blocking-fix): predict_preset receives &[Tree<T>] directly, and Tree<T> is !Sync only via TreeBuf::Borrowed { *const T }; the Wave-0 Model: Sync soundness argument applies identically to TreeBuf (read-only predict, backing outlives borrow, no interior mutability). Sync only, NOT Send."
  - "PredictOut gained + Send + Sync supertrait bounds (f32/f64 both satisfy) so all predict entry points thread the parallel O bound with no per-call-site annotation churn."
  - "averaging + base-score post-passes use plain par_chunks_mut(cells_per_row).for_each (no map_init — no per-worker scratch needed); only the traversal loops need map_init."
  - "test_nthread.py placed under tests/python/ (the live conftest/collection root) rather than the plan's literal tests/ path, so `from conftest import FIXTURES` resolves and pytest discovers it."

patterns-established:
  - "run_with_nthread<R,F>(nthread, fill) — the single nthread scoped-pool wrapper reused by all four converted loop families"
  - "fully-dense CSR fixture (dense_csr) to prove sparse-path determinism is byte-identical to the dense logical data"

requirements-completed: [PAR-01, PAR-02, PAR-04]

# Metrics
duration: ~20min
completed: 2026-06-11
---

# Phase 10 Plan 01: Parallel Scalar Inference Summary

**Row-parallelized the scalar GTIL engine — all four serial row loops converted to rayon `par_chunks_mut`/`map_init`, `Config.nthread` threaded through a scoped `ThreadPool`, the inner per-row tree sum kept serial (GTIL-08), and parallel output proven byte-identical to serial across runs and nthread settings, all within the 1e-5 golden gate.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-06-11
- **Tasks:** 3
- **Files modified:** 6 modified, 1 created

## Accomplishments
- **Task 1** — `predict_preset` row-parallelized: serial `for r in 0..num_row` → `output.par_chunks_mut(cells_per_row).enumerate().map_init(|| vec![O::nan(); num_feature], ...)`. Added `use rayon::prelude::*` and the `run_with_nthread` scoped-pool helper. Refactored `output_leaf_value`/`output_leaf_vector` into `_row` variants writing into a disjoint per-row `&mut [O]` slice (`cell_in_row` indexing) — borrow-checker proves non-overlap (T-10-03, no manual unsafe). Inner `for (tree_id, tree)` loop stays serial (GTIL-08); the closure returns `Result<(), GtilError>` and `.collect::<Result>()?` short-circuits on first Err (ERR-01).
- **Task 2** — the three remaining loop families converted: `predict_leaf_preset` (`par_chunks_mut(num_tree)` + map_init), `predict_score_by_tree_preset` (`par_chunks_mut(num_tree*lvs)` + map_init, `LeafVectorTooShort` via Result short-circuit), and the RF-averaging + base-score post-passes (`par_chunks_mut(cells_per_row).for_each`). `Config.nthread` threaded into all call sites. `config.rs` + `treelite-py/src/gtil.rs` docs corrected (nthread now USED). cubecl `predict_cpu`/`predict_cpu_sparse` confirmed to forward `cfg` unchanged (verify-only, no edit).
- **Task 3** — un-ignored `determinism_byte_identical_n_runs` + added `determinism_sparse_byte_identical_n_runs` (fully-dense CSR), un-ignored `nthread_equivalence` + `parallel_uses_more_than_one_core`, and created `test_nthread.py` (nthread=2==nthread=1 and nthread=-1==1 within 1e-5 over the LightGBM categorical scalar-fallback fixture).

## Task Commits

Each task was committed atomically:

1. **Task 1: row-parallelize predict_preset via rayon par_chunks_mut** - `3a074cf` (feat)
2. **Task 2: parallelize leaf/score/averaging/base-score loops, thread nthread** - `d908d03` (feat)
3. **Task 3: un-ignore determinism + nthread tests, add Python nthread pytest** - `167039f` (test)

## Files Created/Modified
- `crates/treelite-gtil/src/lib.rs` - rayon import; `run_with_nthread` helper; all 4 row-loop families parallelized; `output_leaf_*_row` per-row-slice refactor; `cell_in_row` on `OutputLayout`; `PredictOut: + Send + Sync`; nthread threaded through every preset call site.
- `crates/treelite-gtil/src/config.rs` - `Config.nthread` doc corrected (sizes the rayon pool; `<=0` all cores, `N` bounded scoped pool); removed the stale "recorded but never used" claim.
- `crates/treelite-core/src/tree_buf.rs` - `unsafe impl<T: Copy> Sync for TreeBuf<T>` with the read-only-predict SAFETY argument (mirrors Wave-0 `Model: Sync`), so `&[Tree<T>]` shares across rayon workers.
- `crates/treelite-gtil/tests/determinism.rs` - un-ignored dense determinism test; added `determinism_sparse_byte_identical_n_runs` + `dense_csr` helper; RED→GREEN doc refresh.
- `crates/treelite-gtil/tests/parallel_nthread.rs` - un-ignored `nthread_equivalence` + `parallel_uses_more_than_one_core`; RED→GREEN doc refresh.
- `crates/treelite-py/src/gtil.rs` - `make_config` doc note corrected (nthread now drives the scalar path end-to-end).
- `crates/treelite-py/tests/python/test_nthread.py` (new) - two PAR-04 pytest cases over the LightGBM categorical scalar-fallback fixture.

## Decisions Made
- **`unsafe impl Sync for TreeBuf<T>`** (Rule 3 blocking-fix): `predict_preset` takes `&[Tree<T>]` (not `&Model`), and `Tree<T>` is `!Sync` solely because `TreeBuf::Borrowed` holds `*const T`. The Wave-0 `Model: Sync` argument (read-only during predict, backing outlives the borrow, no interior mutability) applies identically at the `TreeBuf` level, so the minimal sound completion is to make `TreeBuf` `Sync`. Sync only, not Send (A4).
- **`PredictOut: + Send + Sync`** supertrait rather than per-fn bounds — f32/f64 trivially satisfy both, and it cleanly propagates the parallel requirement through `predict`/`predict_sparse`/`predict_rows` without touching every signature.
- **Averaging/base-score use plain `par_chunks_mut().for_each`** (no `map_init`) — those passes need no per-worker scratch; `map_init` is reserved for the traversal loops that materialize a row.
- **`test_nthread.py` under `tests/python/`** (the existing collection root with conftest), not the plan's literal `tests/` path, so the fixture import + pytest discovery work as in the rest of the suite.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added `unsafe impl Sync for TreeBuf<T>` to treelite-core**
- **Found during:** Task 1 (first `cargo test -p treelite-gtil`)
- **Issue:** `par_chunks_mut` requires `O: Send` and the closure capturing `&[Tree<T>]` requires `Tree<T>: Sync`. Wave 0 made only `Model` Sync; `predict_preset` receives `&[Tree<T>]` directly, which was still `!Sync` (`TreeBuf::Borrowed { *const T }`). Without it the conversion cannot compile.
- **Fix:** Added `unsafe impl<T: Copy> Sync for TreeBuf<T>` with the same read-only-predict soundness doc block as the Wave-0 `Model: Sync`; added `+ Send + Sync` to the `PredictOut` supertrait.
- **Files modified:** `crates/treelite-core/src/tree_buf.rs`, `crates/treelite-gtil/src/lib.rs`
- **Commit:** `3a074cf`

This is the natural completion of the Wave-0 `Sync` contract (a `Sync` `Model` is only useful if its constituent `Tree`/`TreeBuf` columns can be shared), not a new architectural direction — no Rule-4 checkpoint warranted.

**2. [Rule 3 - Path alignment] test_nthread.py placed under tests/python/**
- **Found during:** Task 3
- **Issue:** The plan named `crates/treelite-py/tests/test_nthread.py`, but the live pytest collection root (conftest + `from conftest import FIXTURES`) is `crates/treelite-py/tests/python/`.
- **Fix:** Created the file under `tests/python/` so it discovers + imports correctly. No behavior change.
- **Commit:** `167039f`

## Authentication Gates
None.

## Verification Evidence
- `cargo test -p treelite-harness --test gtil_matrix` — green (parallel output == frozen serial goldens within 1e-5, all 4 kinds, both presets, both dtypes, dense + sparse).
- `cargo test -p treelite-gtil --test determinism` — green (dense + sparse N=4 byte-identical, all 4 PredictKinds).
- `cargo test -p treelite-gtil --test parallel_nthread` — green (nthread 0/1/2 byte-identical; `rayon::current_num_threads() > 1` on this multi-core runner).
- `cargo test --workspace` — 0 failures (includes `model_is_sync_for_readonly_predict`).
- `uv run pytest -k nthread` — 2 passed (nthread=2==1 and nthread=-1==1 within 1e-5); full pytest suite 41 passed, 1 skipped.
- Grep gates: `par_chunks_mut` count = 8 (≥4, all four families); no `build_global` in code; no tree-axis `par_iter`; scratch alloc only in `map_init` init position; `recorded but never used` doc claim gone; no `#[ignore]` attributes remain.
- Golden v5 bins byte-identical (serialization untouched; no fixture changes in git status).
- `cargo clippy -p treelite-gtil -p treelite-core` — clean.

## Threat Mitigations Applied
- **T-10-01 (DoS, nthread pool):** `run_with_nthread` builds a pool bounded to exactly `nthread` workers via `ThreadPoolBuilder::num_threads(n)`; `build()` Err → `GtilError::ThreadPool` (typed, no panic); `nthread <= 0` uses the core-capped global pool; never `build_global`; per-worker scratch is `num_feature`-sized.
- **T-10-02 (DoS, CSR validation):** `csr.validate(num_row, num_feature)` and the dense buffer-length check remain BEFORE the parallel section — never moved into the per-row closure.
- **T-10-03 (concurrent Model reads):** `par_chunks_mut` yields statically-disjoint output slices (no manual unsafe indexing); workers share `&Model`/`&[Tree<T>]` read-only via the documented `Sync` impls; determinism test proves no run-to-run reordering.
- **T-10-04 (panic across rayon boundary):** each closure returns `Result<(), GtilError>` and `.collect::<Result>()?` short-circuits on first Err (ERR-01 parity); no new panic surface.

## Self-Check: PASSED

- FOUND: crates/treelite-gtil/src/lib.rs
- FOUND: crates/treelite-gtil/tests/determinism.rs
- FOUND: crates/treelite-gtil/tests/parallel_nthread.rs
- FOUND: crates/treelite-py/tests/python/test_nthread.py
- FOUND: crates/treelite-core/src/tree_buf.rs
- FOUND: commit 3a074cf (Task 1)
- FOUND: commit d908d03 (Task 2)
- FOUND: commit 167039f (Task 3)

---
*Phase: 10-parallel-scalar-inference*
*Completed: 2026-06-11*

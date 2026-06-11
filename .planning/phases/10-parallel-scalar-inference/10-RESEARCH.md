# Phase 10: Parallel Scalar Inference - Research

**Researched:** 2026-06-11
**Domain:** CPU data-parallelism (rayon) over the scalar GTIL reference engine; `Send`/`Sync` soundness for a raw-pointer-holding `Model`
**Confidence:** HIGH

## Summary

Phase 10 row-parallelizes the single-threaded scalar GTIL fallback (`treelite_gtil::predict` dense and `predict_sparse`) so LightGBM `kLE`, categorical, non-`kLT`, and all-sparse models stop running on one core. The diagnosis is already MEASURED (99% CPU on the scalar path; a `std::thread::scope` row-partition prototype measured 3.0–4.6×) and is NOT re-derived here. This research resolves the seven rayon-integration UNKNOWNS that constitute the actual planning decision surface.

The work is a near-textbook embarrassingly-parallel transform: the serial bottleneck is exactly one loop — `predict_preset`'s `for r in 0..num_row` at `crates/treelite-gtil/src/lib.rs:661` — and three sibling loops (`predict_leaf_preset`, `predict_score_by_tree_preset`, and the per-row averaging/base-score/postprocessor passes). Every row writes a disjoint output slice and uses a private scratch row, so there is no cross-row shared mutable state. Upstream Treelite does the identical thing under OpenMP (`predict.cc:241` `ParallelFor(0, num_row, ..., Static())` with a per-thread `dense_row_` buffer of `nthread * num_feature`), which is the soundness and behavior precedent to mirror.

**Primary recommendation:** Pin `rayon = "1.12.0"` in `[workspace.dependencies]`. Convert the outer `for r in 0..num_row` loops to `output.par_chunks_mut(cells_per_row).enumerate()` with `map_init` supplying each worker its own scratch row. Honor `Config.nthread` by building a per-call scoped `rayon::ThreadPool` (`ThreadPoolBuilder::num_threads(n).build()` then `pool.install(...)`) ONLY when `nthread > 0`; when `nthread <= 0`, run on the global pool (all cores) — never mutate global pool state. Add a sound, documented `unsafe impl Sync for Model` (NOT `Send`) in `treelite-core`; rayon shares `&Model` across workers, which requires only `Sync`. Replace `_assert_not_send` with a positive `requires_sync::<Model>()` shareability assertion. GTIL-08 is preserved automatically because the parallelized axis is rows; the per-row `tree_id` loop stays serial and untouched.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Row-parallel scalar dense predict | `treelite-gtil` (compute) | — | The serial loop and its scratch live here; rayon is added here |
| Row-parallel scalar sparse predict | `treelite-gtil` (compute) | — | `predict_sparse` funnels into the same `predict_rows`/`predict_preset` bodies |
| `unsafe impl Sync for Model` | `treelite-core` (model) | — | `Model`/`TreeBuf::Borrowed` are defined here; the shareability contract belongs with the type |
| `Config.nthread` → bounded pool | `treelite-gtil` (compute) | `treelite-cubecl` (fallback caller), `treelite-py` (kwarg) | nthread is already a `Config` field; the pool is built where predict runs |
| Python `nthread=` plumbing | `treelite-py` (binding) | `treelite-cubecl` (fallback) | The kwarg already reaches `make_config`; it must flow through `predict_cpu`/`predict_cpu_sparse` to the scalar engine |
| Determinism / 1e-5 equivalence gate | `treelite-harness` (tests) | — | The frozen golden matrix is the gate; parallel output must equal serial within 1e-5 |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `rayon` | 1.12.0 | Data-parallel `par_chunks_mut` / `map_init` / scoped `ThreadPool` over the row loop | The canonical Rust data-parallelism crate; work-stealing scheduler, mature, zero-config global pool. `[VERIFIED: crates.io registry]` via `cargo add --dry-run` (v1.12.0) + slopcheck `[OK]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `std::thread::available_parallelism` | std | Core count for the `nthread <= 0` "all cores" default if ever needed explicitly | Prefer letting rayon's global pool size itself; only call this if you must report/log the effective thread count. `[VERIFIED: std]` |

**Note on `rayon::current_num_threads()`:** rayon exposes `current_num_threads()` to read the active pool's worker count — useful for the "uses >1 core" validation assertion and for honoring `nthread <= 0` (just use the default global pool). `[CITED: docs.rs/rayon]`

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `rayon` | `std::thread::scope` (the prototype) | The prototype proved feasibility (3.0–4.6×) but hand-rolls partitioning, thread-count handling, and work balancing. rayon's work-stealing handles ragged tree depths (categorical/deep LightGBM rows vary in cost) better than a static manual partition. Do NOT ship the hand-rolled prototype. |
| `rayon` | `std::thread` + manual chunking | Same hand-roll objection; this is the "Don't Hand-Roll" item below. |
| per-call scoped pool | mutating the global pool (`ThreadPoolBuilder::build_global`) | `build_global` can be called only ONCE per process and mutates global state — forbidden across calls with differing `nthread`. Use a per-call scoped pool instead. `[CITED: docs.rs/rayon]` |

**Installation:**
```bash
# In [workspace.dependencies]:
rayon = "1.12.0"
# Then in crates/treelite-gtil/Cargo.toml:
# rayon = { workspace = true }
```

**Version verification:** `cargo add rayon --package treelite-gtil --dry-run` → `Adding rayon v1.12.0`. This is the authoritative crates.io resolution (the `npm view rayon` 1.1.5 result is an UNRELATED JavaScript package — a textbook cross-ecosystem confusion; ignore it).

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `rayon` | crates.io | ~10 yrs | >300M total | github.com/rayon-rs/rayon | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

slopcheck 0.6.1 ran `slopcheck install rayon --ecosystem crates.io` → `[OK]`. rayon is the de-facto standard Rust data-parallelism crate maintained by the rayon-rs org. Version 1.12.0 confirmed via `cargo add --dry-run`. No postinstall/build-script risk (rayon has no network/filesystem build script of concern). Approved for `[workspace.dependencies]`.

## Architecture Patterns

### System Architecture Diagram

```
Python predict(nthread=N, backend="cpu")
        │  (numpy zero-copy borrow, GIL released via py.detach)
        ▼
treelite-py::gtil::predict_f32/_f64   ── make_config(nthread, ...) ──► Config{ kind, nthread }
        │
        ▼
treelite-cubecl::predict_cpu / predict_cpu_sparse   (the scalar-fallback caller)
        │   model_routes_to_scalar_fallback(model)?  ── kLE/categorical/non-kLT/sparse ──► YES
        ▼
treelite-gtil::predict / predict_sparse            ◄── &Model (shared, Sync)
        │   validate input ONCE (dense shape | CSR structure)  [up-front, not per-row]
        ▼
predict_rows ─► predict_preset (and leaf/score siblings)
        │
        ▼
   ┌──────────────────────────────────────────────────────────┐
   │  rayon scope honoring Config.nthread:                     │
   │   nthread<=0 → global pool (all cores)                    │
   │   nthread>0  → per-call scoped ThreadPool(n).install(..)  │
   │                                                            │
   │   output.par_chunks_mut(cells_per_row).enumerate()        │
   │     .map_init(|| vec![nan; num_feature],  ← per-worker     │
   │               |scratch, (r, out_cells)| {                  │
   │        rows.materialize(r, scratch);  ← dense copy|CSR NaN │
   │        for (tree_id, tree) in trees.enumerate() {  ← SERIAL│
   │            leaf = evaluate_tree(tree, scratch)?;  (GTIL-08)│
   │            accumulate into out_cells (disjoint per row)    │
   │        }                                                   │
   │     })                                                     │
   └──────────────────────────────────────────────────────────┘
        │
        ▼
   per-row RF averaging  +  f64 base-score add  +  postprocessor
   (each is a per-row independent pass — also row-parallel, optional)
        │
        ▼
   Vec<O>  (flat row-major, identical bytes to the serial path)
```

File-to-implementation mapping is in the Component Responsibilities table below, not the diagram.

### Component Responsibilities
| File | Change |
|------|--------|
| `crates/treelite-gtil/Cargo.toml` | Add `rayon = { workspace = true }` |
| `crates/treelite-gtil/src/lib.rs` | Parallelize the 4 outer row loops: `predict_preset` (:661), `predict_leaf_preset` (:1082), `predict_score_by_tree_preset` (:1149), and the per-row averaging/base-score passes (:709, :728). Add the `nthread` scoped-pool wrapper in `predict_rows`/`predict`/`predict_sparse`. |
| `crates/treelite-gtil/src/config.rs` | Update the doc comment: `nthread` is now USED (no longer "recorded but never used"). No struct change — the field already exists. |
| `crates/treelite-core/src/model.rs` | Add `unsafe impl Sync for Model` with a soundness comment. (See PAR-03 analysis for whether `Send` is also needed — it is NOT for the rayon `&Model` path.) |
| `crates/treelite-core/tests/model_invariants.rs` | Replace `_assert_not_send` with `requires_sync::<Model>()` positive assertion. |
| `crates/treelite-cubecl/src/lib.rs` | `predict_cpu_sparse` already forwards `cfg` to `predict_sparse` — verify nthread flows. `predict_cpu` → `predict::<CpuRuntime>` → for kLE/categorical it hits the kernel-vs-fallback gate; confirm `cfg` reaches the scalar fallback. |
| `crates/treelite-py/src/gtil.rs` | The `nthread` kwarg already reaches `make_config`. Update the doc note (currently says "recorded but unused"). No signature change needed. |
| `crates/treelite-harness/tests/` | Add a determinism test (run N times, assert identical bytes) and a >1-core utilization check; reuse the existing frozen `gtil_matrix.rs` goldens unchanged. |

### Pattern 1: `par_chunks_mut` + `map_init` per-worker scratch (THE core pattern)
**What:** Replace the serial `for r in 0..num_row { materialize(r, &mut scratch); ... }` with a parallel iterator over disjoint output chunks, where each worker gets its own scratch row via `map_init`.
**When to use:** The dense `predict_preset` and the leaf/score-per-tree siblings.
**Example:**
```rust
// Source: rayon docs (par_chunks_mut + map_init) — docs.rs/rayon
use rayon::prelude::*;

// `output` is the flat Vec<O> of length num_row * cells_per_row.
// par_chunks_mut yields one disjoint &mut [O] of width cells_per_row per row.
let result: Result<(), GtilError> = output
    .par_chunks_mut(cells_per_row)
    .enumerate()
    .map_init(
        // init: called once per rayon worker thread — each gets its OWN scratch.
        || vec![O::nan(); num_feature],
        |scratch, (r, out_cells)| {
            rows.materialize(r, scratch);          // dense copy | CSR NaN-fill
            let row: &[O] = scratch;
            for (tree_id, tree) in trees.iter().enumerate() {
                let leaf = evaluate_tree(tree, row)?;   // ? inside the closure
                let target_id = shape.target_id.get(tree_id).copied().unwrap_or(-1);
                let class_id  = shape.class_id.get(tree_id).copied().unwrap_or(-1);
                // NOTE: out_cells is THIS ROW's slice — output_leaf_* must be
                // refactored to write into a per-row &mut [O] at a local offset,
                // not into the global `output` at shape.idx(r, t, c).
                if has_leaf_vector(tree, leaf)? {
                    output_leaf_vector_local(out_cells, &shape, tree, leaf, target_id, class_id)?;
                } else {
                    output_leaf_value_local(out_cells, &shape, tree, leaf, target_id, class_id)?;
                }
            }
            Ok(())
        },
    )
    .collect();   // collect Result<(), E> — short-circuits on the first Err
result?;
```
**Key refactor:** `output_leaf_value`/`output_leaf_vector` currently index the GLOBAL `output` via `shape.idx(row_id, t, c)`. To use `par_chunks_mut`, they must be adapted to write into a per-row `&mut [O]` slice indexed by `t * max_num_class + c` (i.e. `shape.idx` minus the `r * cells_per_row` row offset). This is a mechanical change that also makes the disjointness statically obvious to the borrow checker. `[ASSUMED]` that this refactor is the cleanest route — an alternative (`par_iter_mut().chunks()` or manual index math) exists but is less idiomatic.

### Pattern 2: Honoring `Config.nthread` without touching global pool state
**What:** `nthread <= 0` → use the global pool (all cores). `nthread > 0` → build a per-call scoped pool of exactly `nthread` workers and `install` the parallel work onto it.
**When to use:** Wrap the parallel section in `predict_rows` (and the leaf/score bodies).
**Example:**
```rust
// Source: rayon ThreadPoolBuilder docs — docs.rs/rayon
fn run_parallel<F: FnOnce() -> R + Send, R: Send>(nthread: i32, work: F) -> Result<R, GtilError> {
    if nthread <= 0 {
        // Upstream MaxNumThread() semantics: all available threads (the global pool).
        Ok(work())
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthread as usize)
            .build()
            .map_err(|e| GtilError::ThreadPool(e.to_string()))?; // new typed variant
        Ok(pool.install(work))
    }
}
```
**Anti-pattern guarded:** Do NOT call `ThreadPoolBuilder::build_global()` per request — it can only succeed once per process and is a global mutation. A scoped `pool.install(...)` is per-call and side-effect-free. `[CITED: docs.rs/rayon — build_global "may be called at most once"]`

**Cost note:** Building a scoped `ThreadPool` per call spawns OS threads each time. For the bounded-`nthread` path this is acceptable (matches the "explicit nthread bound" use case), but if profiling shows churn, a `nthread`-keyed memoized pool is a v2 optimization — out of scope for Phase 10. `[ASSUMED]`

### Pattern 3: Row-parallel averaging / base-score / postprocessor passes
**What:** The RF-averaging pass (`:709`), the base-score add (`:728`), and `apply_postprocessor` are independent per-row writes (upstream runs each as its own `ParallelFor`, `predict.cc:284`, `:297`, `:315`). They CAN be parallelized the same way (`par_chunks_mut(cells_per_row)`), but are cheap relative to traversal.
**When to use:** Parallelize the traversal loop first (the measured bottleneck); the post-passes are optional and lower-value. Match upstream by parallelizing them too for consistency, but the 1e-5 contract does not require it. `[CITED: predict.cc:284-323]`

### Anti-Patterns to Avoid
- **Parallelizing the tree axis:** NEVER. The inner `for (tree_id, tree)` loop must stay serial in `tree_id` order — float add is non-associative; reordering breaks GTIL-08 and the 1e-5 contract. Only the OUTER row loop is parallelized.
- **One shared scratch buffer across workers:** A data race + wrong results. Each worker needs its own scratch via `map_init` (mirrors upstream's `dense_row_` sized `nthread * num_feature`).
- **`unsafe impl Send for TreeBuf`/blanket Send:** Over-broad. The minimal sound change is `unsafe impl Sync for Model` (see PAR-03). Adding `Send` is unnecessary for the rayon `&Model`-shared path and widens the contract beyond what is justified.
- **Mutating the global rayon pool per call:** `build_global` once-only; use scoped `install`.
- **Re-validating CSR per row:** `predict_sparse` already validates the whole CSR ONCE up front (`:969 csr.validate(...)`). Keep it up front — do not move validation inside the parallel closure (PAR-02 requirement).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Row partitioning across cores | The `std::thread::scope` prototype's manual chunk math | `rayon::par_chunks_mut` | Work-stealing balances ragged per-row cost (deep categorical/LightGBM rows vary); manual static partition leaves cores idle on skewed workloads |
| Per-worker scratch lifecycle | Manual `Vec<Vec<O>>` indexed by `thread_id` (the upstream OpenMP `dense_row_` shape) | `rayon` `map_init` | `map_init` ties scratch to the worker lifetime with zero index bookkeeping and no `thread_id` plumbing |
| Bounded thread count | A hand-rolled thread pool / semaphore | `rayon::ThreadPoolBuilder` scoped `install` | Battle-tested, integrates with the parallel iterators, no global state mutation |
| Core count detection | Parsing `/proc/cpuinfo` | rayon global pool default (or `available_parallelism`) | rayon already sizes its pool; upstream `MaxNumThread()` is the same intent |

**Key insight:** This phase is one `rayon` dependency plus a ~4-loop mechanical conversion plus one `unsafe impl Sync`. The prototype proved the algorithm; rayon replaces the hand-rolled scaffolding the prototype used. Resist re-implementing scheduling.

## Runtime State Inventory

This is a code-only parallelism change — no stored data, services, OS registrations, secrets, or build artifacts carry parallelism state. The inventory is included for completeness because the change touches a soundness invariant (`Send`/`Sync`).

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore encodes thread/parallelism state. Verified: grep for nthread/rayon shows only `Config.nthread` (a runtime arg) and doc comments. | none |
| Live service config | None — no external service. | none |
| OS-registered state | None — rayon's pool is process-local and ephemeral. | none |
| Secrets/env vars | rayon reads `RAYON_NUM_THREADS` env var for its DEFAULT global pool size. This is a behavior note, not a secret: when `nthread <= 0`, the effective core count can be capped by `RAYON_NUM_THREADS` if set in the environment. Document this; do not depend on it. `[CITED: docs.rs/rayon]` | document only |
| Build artifacts | The `_assert_not_send` test in `model_invariants.rs` is a compiled invariant that MUST be replaced (not deleted) — see PAR-03. The `treelite-py` `#[pyclass(unsendable)]` stays as-is (GIL-bound, independent of core Sync). | replace test; leave pyclass |

**The canonical question — after the code change, what still carries the old invariant?** Only the `_assert_not_send` test (superseded by `requires_sync`) and the doc comments in `config.rs`/`gtil.rs`/`model.rs` that say "recorded but unused" / "must stay !Send". All must be updated in the same phase.

## Common Pitfalls

### Pitfall 1: Edition-2024 disjoint-closure capture grabbing the bare `&Model`
**What goes wrong:** A `move` closure that captures `model_wrapper.0` (a bare `&Model`) instead of the wrapper triggers the `!Send`/`!Sync` of the inner reference, failing the rayon bound — even though the wrapper is shareable.
**Why it happens:** Edition-2024 disjoint closure capture grabs the specific field `.0`, not the whole struct (this exact trap bit the prototype).
**How to avoid:** The codebase ALREADY has the fix pattern: `SendModelRef::into_ref` in `treelite-py/src/gtil.rs:42-55` consumes the WHOLE wrapper by value inside the closure. For rayon, the cleaner route is to make `Model: Sync` directly (PAR-03) so `&Model` is `Send`-shareable across rayon workers with no wrapper at all. With `unsafe impl Sync for Model`, `&Model: Send` follows automatically (`&T: Send` iff `T: Sync`) and rayon closures capture `&model` without ceremony.
**Warning signs:** `E0277: &Model cannot be shared between threads safely` or `... is not Send`.

### Pitfall 2: Allocation contention under jemalloc/mimalloc (a MEASURED sub-linearity cause)
**What goes wrong:** A per-row or per-chunk `Vec` allocation inside the hot loop serializes on the global allocator, capping speedup below the 3.0–4.6× ceiling.
**Why it happens:** Phase 9 wired jemalloc/mimalloc (benchmarks/binaries). High allocation churn (one scratch `Vec` per row) hammers the allocator from many threads.
**How to avoid:** `map_init` allocates the scratch ONCE PER WORKER (not per row) — this is the allocation-light idiom. The scratch is reused across all rows that worker processes. Do NOT allocate inside the per-row closure body. Confirm no `vec!`/`Vec::with_capacity` appears in the inner row closure.
**Warning signs:** Speedup plateaus well below core count; `perf` shows time in `malloc`/`free`.

### Pitfall 3: `?` and error propagation across the parallel boundary
**What goes wrong:** `evaluate_tree` returns `Result`; the serial loop uses `?`. Inside a rayon closure, `?` returns from the CLOSURE, not the function, so errors must be collected.
**Why it happens:** Parallel iterators don't propagate `?` to the outer fn.
**How to avoid:** Have the `map_init` closure return `Result<(), GtilError>` and `.collect::<Result<(), GtilError>>()?` at the end — rayon short-circuits on the first `Err` and returns it deterministically. This preserves the exact typed-error behavior (ERR-01) the serial path has.
**Warning signs:** Compile error "`?` cannot be used in a closure that returns `()`", or swallowed errors.

### Pitfall 4: Tiny-batch thread overhead
**What goes wrong:** A 1-row or few-row predict pays pool/scheduling overhead for no gain.
**Why it happens:** rayon splits work even for tiny inputs (though its adaptive splitting is cheap).
**How to avoid:** Upstream does NOT add an explicit row-count threshold — OpenMP's `ParallelFor` simply doesn't benefit on tiny loops and the overhead is small (`predict.cc` uses `ParallelSchedule::Static()` unconditionally). rayon's `par_chunks_mut` similarly has low fixed cost and adaptive splitting. RECOMMENDATION: do NOT add a cutoff in v1 (match upstream simplicity); if a micro-benchmark shows regression on 1-row calls, a `if num_row < THRESHOLD { serial } else { parallel }` guard is a trivial, low-risk addition. Flag as an Open Question to confirm during planning. `[CITED: predict.cc — no threshold; ASSUMED no cutoff needed]`
**Warning signs:** Single-row latency regresses vs. the serial baseline.

### Pitfall 5: Forgetting the leaf/score/averaging siblings
**What goes wrong:** Parallelizing only `predict_preset` leaves `predict_leaf_preset` (LeafId), `predict_score_by_tree_preset` (ScorePerTree), and the averaging/base-score passes serial — PAR-01 says "scalar dense predict runs row-parallel," which the harness exercises across all 4 predict kinds.
**Why it happens:** The Default/Raw path is the obvious one; the other kinds have their own row loops.
**How to avoid:** Convert all four outer row loops (lines 661, 1082, 1149, and the averaging/base-score passes). The harness matrix covers all kinds — a missed loop is caught by the >1-core utilization assertion only if it tests that kind.

## Code Examples

### Honoring nthread with a scoped pool (full integration)
```rust
// Source: composed from rayon ThreadPoolBuilder + par_chunks_mut docs (docs.rs/rayon)
fn predict_preset<T: PredictScalar + PartialOrd, O: PredictOut>(
    trees: &[Tree<T>],
    shape: &OutputLayout<'_>,
    rows: &RowSource<'_, O>,
    num_row: usize,
    num_feature: usize,
    nthread: i32,                       // threaded through from Config
) -> Result<Vec<O>, GtilError> {
    let cells_per_row = shape.cells_per_row();
    let mut output = vec![O::zero(); num_row * cells_per_row];

    let fill = || -> Result<(), GtilError> {
        output
            .par_chunks_mut(cells_per_row)
            .enumerate()
            .map_init(
                || vec![O::nan(); num_feature],          // per-worker scratch
                |scratch, (r, cells)| {
                    rows.materialize(r, scratch);
                    for (tree_id, tree) in trees.iter().enumerate() {
                        let leaf = evaluate_tree(tree, scratch)?;
                        // ... write into `cells` (this row's disjoint slice), SERIAL in tree_id
                    }
                    Ok(())
                },
            )
            .collect::<Result<(), GtilError>>()
    };

    if nthread <= 0 {
        fill()?;                                          // global pool = all cores
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthread as usize)
            .build()
            .map_err(|e| GtilError::ThreadPool(e.to_string()))?;
        pool.install(fill)?;
    }
    // ... averaging / base-score / (postprocessor at caller) unchanged or par'd similarly
    Ok(output)
}
```

### The sound `unsafe impl Sync for Model`
```rust
// Source: Rust reference (Send/Sync rules) + upstream OpenMP precedent (predict.cc)
// SAFETY: `Model` is `!Sync` only because `TreeBuf::Borrowed { ptr: *const T }`
// (tree_buf.rs:27) holds a raw const pointer. Sharing `&Model` across threads is
// sound for the prediction path because:
//   1. predict takes `&Model` (SHARED, never `&mut`) — no field is mutated;
//   2. the borrowed foreign buffer is read-only (`from_raw_parts` const slice,
//      tree_buf.rs:63) and its backing memory is guaranteed (by the
//      `from_borrowed` SAFETY contract) to outlive the `TreeBuf` and to NOT be
//      mutated while borrowed — therefore concurrent reads from many threads are
//      data-race-free;
//   3. `Model` contains NO interior mutability (no Cell/RefCell/atomics on the
//      predict path) — the serialization-bookkeeping scalars are plain values,
//      and `stage_serialization_fields` takes `&mut self` so it cannot run
//      concurrently with a shared-`&` predict.
// This mirrors upstream Treelite, which shares `Model const&` across OpenMP
// threads in `PredictRaw` (predict.cc:241). Only `Sync` is asserted — NOT `Send`;
// rayon shares `&Model` (which is `Send` iff `Model: Sync`), and the model is
// never MOVED to another thread.
unsafe impl Sync for Model {}
```

### Replacement shareability test (supersedes `_assert_not_send`)
```rust
// crates/treelite-core/tests/model_invariants.rs
/// PAR-03: `Model` is soundly SHAREABLE across threads for read-only predict
/// (documented `unsafe impl Sync`, mirroring upstream OpenMP). This SUPERSEDES
/// the prior `_assert_not_send` invariant — the type intentionally became `Sync`.
#[test]
fn model_is_sync_for_readonly_predict() {
    fn requires_sync<T: Sync>() {}
    requires_sync::<Model>();          // compiles iff Model: Sync — the new contract
    // &Model: Send follows from Model: Sync, which is what rayon requires.
    fn requires_send<T: Send>() {}
    requires_send::<&Model>();
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Serial scalar `for r in 0..num_row` | rayon `par_chunks_mut` + `map_init` | This phase | LightGBM/categorical/sparse stop running on 1 core |
| `Model: !Send + !Sync` (`_assert_not_send` pinned) | `Model: Sync` via documented `unsafe impl` | This phase | `&Model` shareable across rayon workers, mirroring upstream OpenMP |
| `Config.nthread` recorded-but-unused | `nthread` drives a scoped pool (`<=0` all cores, `N` bounded) | This phase | PAR-04 — Python `nthread=` kwarg becomes effective |

**Deprecated/outdated:**
- `_assert_not_send` test: superseded — DELETE and replace with `requires_sync` (do not merely remove).
- Doc comments asserting "nthread accepted and recorded but never used" (`config.rs:41-43`) and "recorded but unused" (`gtil.rs:58-59`): now false, must be updated.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `output_leaf_value`/`output_leaf_vector` refactor to per-row `&mut [O]` slices is the cleanest `par_chunks_mut` route | Pattern 1 | Low — an alternative (manual offset into `par_iter_mut`) works; only ergonomics differ |
| A2 | No explicit small-batch row-count cutoff is needed (matching upstream) | Pitfall 4 | Low — adding a `num_row < N → serial` guard later is trivial and non-breaking |
| A3 | Per-call scoped pool churn for `nthread > 0` is acceptable (no memoization needed in v1) | Pattern 2 | Low — memoizing a pool is a v2 perf tweak; correctness is unaffected |
| A4 | Only `unsafe impl Sync` (not `Send`) is required for the rayon `&Model` path | PAR-03 / Code Examples | Medium — if any future caller MOVES a `Model` to another thread, `Send` would also be needed; but rayon shares, it does not move. Confirm no `std::thread::spawn(move || ... model ...)` ownership-move appears in the plan |
| A5 | The averaging/base-score/postprocessor per-row passes are independent and parallelizable | Pattern 3 | Low — verified against upstream `predict.cc:284-323` (each is its own ParallelFor) |

## Open Questions

1. **Small-batch threshold (Pitfall 4 / A2)**
   - What we know: Upstream uses no explicit cutoff; rayon's fixed overhead is low.
   - What's unclear: Whether single-row Python predict calls (common in serving) regress measurably.
   - Recommendation: Ship without a cutoff (match upstream); add a micro-benchmark of 1-row latency to the validation set, and only add a `num_row < THRESHOLD` serial guard if it regresses.

2. **Parallelize the post-passes, or only traversal? (Pattern 3 / A5)**
   - What we know: Traversal is the measured bottleneck; the post-passes are cheap and per-row-independent.
   - What's unclear: Whether parallelizing them adds meaningful speedup vs. code churn.
   - Recommendation: Parallelize traversal (the 4 main loops) for PAR-01/02; parallelize the post-passes to MIRROR upstream for consistency, but treat as low-priority — the 1e-5 contract does not require it.

3. **`Send` necessity (A4)**
   - What we know: rayon needs `&Model: Send`, which follows from `Model: Sync`.
   - What's unclear: Whether the chosen implementation moves a `Model` (then `Send` is also needed).
   - Recommendation: Plan for `unsafe impl Sync` only; if the implementation surfaces a move requirement, add `unsafe impl Send` with its own soundness note (the read-only argument covers both).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rayon (crates.io) | PAR-01/02/04 | ✓ | 1.12.0 | — |
| Rust stable, edition 2024 | whole workspace | ✓ | (existing toolchain) | — |
| Multi-core CPU | >1-core utilization test | ✓ | 16 cores (per measured diagnosis) | A 1-core CI runner would make the utilization assertion vacuous — see Validation |

**Missing dependencies with no fallback:** none — rayon resolves on crates.io and the toolchain already builds the workspace.
**Missing dependencies with fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `approx` (1e-5 asserts); pytest for the Python binding |
| Config file | none (cargo default test harness) |
| Quick run command | `cargo test -p treelite-gtil` |
| Full suite command | `cargo test --workspace` then `uv run pytest` (Python venv via `uv run`, per project memory) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAR-01 | Scalar dense predict row-parallel, output == serial within 1e-5 | equivalence | `cargo test -p treelite-harness --test gtil_matrix` | ✅ (reuse frozen goldens) |
| PAR-02 | Scalar sparse predict row-parallel, same guarantee; CSR validated once up front | equivalence | `cargo test -p treelite-harness --test gtil_matrix` (sparse cells) | ✅ (reuse frozen CSR goldens) |
| PAR-03 | `Model: Sync`; `_assert_not_send` superseded by `requires_sync` | unit (compile) | `cargo test -p treelite-core --test model_invariants` | ❌ Wave 0 (rewrite test) |
| PAR-04 | `nthread<=0` all cores; `N` bounded; Python kwarg drives it | integration | new `cargo test -p treelite-gtil` nthread test + `uv run pytest -k nthread` | ❌ Wave 0 (new tests) |
| PAR-01/02 | Parallel run actually uses >1 core | utilization | new test asserting `rayon::current_num_threads() > 1` on a multi-row run (gated on `available_parallelism() > 1`) | ❌ Wave 0 |
| GTIL-08 | Determinism: N repeated runs produce byte-identical output | determinism | new test: `predict` ×N, assert `out[0] == out[i]` bytes | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-gtil`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** `cargo test --workspace` + `uv run pytest` green; golden_v5.bin / golden_v5_3format.bin byte-identical (untouched this phase); full `gtil_matrix` 1e-5 across both presets, both input dtypes, dense + sparse.

### Wave 0 Gaps
- [ ] Rewrite `crates/treelite-core/tests/model_invariants.rs` — replace `_assert_not_send` with `requires_sync::<Model>()` (PAR-03). The existing `model_size_not_bloated_by_smallvec` test stays.
- [ ] New `crates/treelite-gtil/tests/parallel_nthread.rs` — assert `nthread=1` vs `nthread=0` produce identical output; assert a multi-row run with `nthread<=0` uses `>1` core (skip if `available_parallelism() == 1`) (PAR-04, utilization).
- [ ] New `crates/treelite-gtil/tests/determinism.rs` — run `predict`/`predict_sparse` N times on a fixed input, assert byte-identical `Vec<O>` (GTIL-08 under parallelism).
- [ ] New `treelite-py` pytest — `gtil.predict(..., nthread=2)` returns identical values to `nthread=1` within 1e-5 (PAR-04 end-to-end), reusing an existing LightGBM/categorical fixture (the scalar-fallback path).
- [ ] The frozen `fixtures/gtil/*.golden.json` matrix is REUSED verbatim as the parallel-vs-serial 1e-5 gate — no new goldens needed (parallel output must equal the existing serial-captured goldens).

*(The Validation section reuses the existing equivalence harness — `gtil_matrix.rs`, both presets, dense+sparse, golden vectors within 1e-5 — as the primary correctness gate; only the nthread/determinism/utilization tests are net-new.)*

## Security Domain

Phase 10 is internal CPU compute with no new external input surface. `security_enforcement` is enabled (`security_asvs_level: 1`).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Library compute, no access tiers |
| V5 Input Validation | yes | EXISTING up-front validation preserved: dense `InvalidInputShape` check (`lib.rs:902-926`); CSR `validate` ONCE before parallel rows (`lib.rs:969`); Python `check_feature_count` (`gtil.rs:94`). Parallelism must NOT move these inside the per-row closure. |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for rayon + raw-pointer Model

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data race on shared `Model` foreign buffer | Tampering / Info Disclosure | `unsafe impl Sync` is sound ONLY because predict is read-only (`&Model`, no interior mutability) — the documented soundness argument is the control; concurrent reads of an immutable `*const T` slice are race-free |
| Unbounded thread allocation via `nthread` | Denial of Service | `nthread > 0` is bounded by the value the caller passes; `nthread <= 0` is capped by core count (rayon global pool / `RAYON_NUM_THREADS`). No `nthread`-driven heap allocation (the old config.rs note about "no DoS amplification" still holds — the per-worker scratch is `num_feature`-sized, not `nthread`-sized growth per row) |
| OOB write under parallel row indexing | Tampering | `par_chunks_mut` yields statically-disjoint slices — the borrow checker proves non-overlap; no manual unsafe indexing into the global output |
| Panic crossing the rayon boundary | DoS | rayon propagates a worker panic to the calling thread; the existing `treelite-py` `guard_assert` catch_unwind (`gtil.rs:247`) still traps it into a `TreeliteError`. Prefer `Result` propagation (Pitfall 3) over panics. |

## Sources

### Primary (HIGH confidence)
- `cargo add rayon --package treelite-gtil --dry-run` → rayon v1.12.0 (authoritative crates.io resolution)
- `slopcheck install rayon --ecosystem crates.io` → `[OK]` (legitimacy)
- `treelite-mainline/src/gtil/predict.cc:230-323` — upstream OpenMP `ParallelFor(0, num_row, Static())` over rows, per-thread `dense_row_`, serial tree loop (the behavioral + soundness precedent)
- `treelite-mainline/include/treelite/detail/threading_utils.h:60-115` — `MaxNumThread()` / `ThreadConfig` (`nthread <= 0` = all threads) semantics
- `crates/treelite-gtil/src/lib.rs:643-741, 1056-1181` — the exact serial loops to convert
- `crates/treelite-core/src/tree_buf.rs:25-63` — the `*const T` `Borrowed` variant that is the source of `!Sync`
- `crates/treelite-core/tests/model_invariants.rs` — the `_assert_not_send` invariant to supersede
- `crates/treelite-py/src/gtil.rs:35-70, 220-279` — `SendModelRef::into_ref` disjoint-capture pattern + `make_config(nthread,...)` + `py.detach`
- `crates/treelite-gtil/src/config.rs` — `Config.nthread` field (recorded, to be wired)
- `crates/treelite-cubecl/src/lib.rs:261-371` — `model_routes_to_scalar_fallback`, `predict_cpu`, `predict_cpu_sparse` (the fallback callers)

### Secondary (MEDIUM confidence)
- docs.rs/rayon — `par_chunks_mut`, `map_init`, `ThreadPoolBuilder` (`build_global` once-only), `current_num_threads`, `RAYON_NUM_THREADS` (standard library docs, cross-checked against the API surface used)

### Tertiary (LOW confidence)
- none — all claims are grounded in the codebase, upstream source, or rayon's authoritative docs.

## Metadata

**Confidence breakdown:**
- Standard stack (rayon 1.12.0): HIGH — verified via `cargo add --dry-run` + slopcheck, single canonical crate
- Architecture (par_chunks_mut + map_init + scoped pool): HIGH — directly mirrors upstream OpenMP structure; pattern is rayon-idiomatic
- `unsafe impl Sync` soundness: HIGH — read-only predict + no interior mutability + upstream precedent; the one nuance (Sync vs Send) is flagged in A4
- Pitfalls: HIGH — allocation contention and disjoint-capture are both MEASURED/known in this codebase
- nthread plumbing: HIGH — the chain (kwarg → make_config → Config → predict) already exists end-to-end; only the scalar engine's use of it is new

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (rayon is stable/slow-moving; the codebase facts are version-pinned to the current tree)

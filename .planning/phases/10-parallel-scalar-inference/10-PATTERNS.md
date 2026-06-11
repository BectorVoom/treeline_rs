# Phase 10: Parallel Scalar Inference - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 7 (5 modify, 2 create)
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-gtil/src/lib.rs` (MODIFY) | service (compute engine) | batch / transform | itself (the serial loops at :661, :1082, :1149, :709, :728) + rayon docs | self / before-after |
| `crates/treelite-gtil/Cargo.toml` (MODIFY) | config | — | own `[dependencies]` block + `[workspace.dependencies]` pins | exact |
| `Cargo.toml` (root, MODIFY) | config | — | Phase-9 `smallvec`/`compact_str` pin entries (lines 26-32) | exact |
| `crates/treelite-core/src/model.rs` (MODIFY) | model | — | `SendModelRef` `unsafe impl Send` (`treelite-py/src/gtil.rs:35-40`) | role + soundness analog |
| `crates/treelite-core/tests/model_invariants.rs` (MODIFY) | test (compile invariant) | — | itself — the `_assert_not_send` fn being rewritten (:29-34) | self / rewrite |
| `crates/treelite-cubecl/src/lib.rs` (MODIFY) | service (fallback caller) | request-response | `predict_cpu` / `predict_cpu_sparse` (:347-371) — already forward `cfg` | self / verify-only |
| `crates/treelite-gtil/tests/parallel_nthread.rs` (CREATE) | test (integration) | batch | `treelite-cubecl/tests/determinism.rs` fixtures (`split_tree`/`scalar_model`) | exact (fixture reuse) |
| `crates/treelite-gtil/tests/determinism.rs` (CREATE) | test (determinism) | batch | `treelite-cubecl/tests/determinism.rs` (whole file) | exact |

## Pattern Assignments

### `crates/treelite-gtil/src/lib.rs` (compute engine, batch transform)

**Analog:** the four serial row loops in this same file (the before/after target). RESEARCH §"Code Examples" supplies the rayon-side shape; the loops below are the exact bodies to convert.

**Loop 1 — `predict_preset` (the measured bottleneck), current serial body** (lines 659-675):
```rust
let mut scratch = vec![O::nan(); num_feature];

for r in 0..num_row {
    rows.materialize(r, &mut scratch);
    let row: &[O] = &scratch;
    // Serial tree accumulation in tree_id order — do NOT parallelize/reorder.
    for (tree_id, tree) in trees.iter().enumerate() {
        let leaf = evaluate_tree(tree, row)?;
        let target_id = shape.target_id.get(tree_id).copied().unwrap_or(-1);
        let class_id = shape.class_id.get(tree_id).copied().unwrap_or(-1);
        if has_leaf_vector(tree, leaf)? {
            output_leaf_vector(&mut output, shape, tree, leaf, r, target_id, class_id)?;
        } else {
            output_leaf_value(&mut output, shape, tree, leaf, r, target_id, class_id)?;
        }
    }
}
```

**Conversion target** (RESEARCH Pattern 1 + §"Code Examples"):
- Replace `for r in 0..num_row` with `output.par_chunks_mut(cells_per_row).enumerate().map_init(|| vec![O::nan(); num_feature], |scratch, (r, cells)| { ... })`.
- The `init` closure (`|| vec![O::nan(); num_feature]`) replaces the single hoisted `scratch` at line 659 — one scratch per rayon worker (Pitfall 2: allocate once per worker, NEVER inside the per-row body).
- INNER `for (tree_id, tree)` loop stays SERIAL and untouched (GTIL-08; float add non-associative).
- `output_leaf_value` / `output_leaf_vector` currently index the GLOBAL `output` via `shape.idx(r, t, c)` (see signatures at :747-755). They must be refactored to write into the per-row `&mut [O]` slice `cells`, indexed by `t * max_num_class + c` (the `shape.idx` value minus the `r * cells_per_row` row offset). RESEARCH Pattern 1 "Key refactor" + A1.
- Closure returns `Result<(), GtilError>`; terminate with `.collect::<Result<(), GtilError>>()?` (Pitfall 3 — `?` inside the closure returns from the closure, not the fn; rayon short-circuits on first `Err`, preserving ERR-01 typed-error behavior).

**Loop 2 — RF averaging pass** (lines 709-720): per-row independent writes; `par_chunks_mut(cells_per_row)`-parallelizable (Pattern 3, optional — mirror upstream `predict.cc:284`). The `average_factor` table (built once at :680-708) stays serial and is shared read-only across workers.

**Loop 3 — base-score add pass** (lines 728-738): per-row independent; same `par_chunks_mut` treatment (Pattern 3, upstream `predict.cc:297`).

**Loop 4a — `predict_leaf_preset`** (lines 1082-1091): outer `for r in 0..num_row` over a `num_row * num_tree` buffer; chunk width is `num_tree`. Same `par_chunks_mut(num_tree).map_init` conversion.

**Loop 4b — `predict_score_by_tree_preset`** (lines 1149-1180): outer `for r in 0..num_row` over a `num_row * num_tree * lvs` buffer; chunk width is `num_tree * lvs`. Same conversion; the `LeafVectorTooShort` error (:1167) propagates via the `collect::<Result>` short-circuit.

**nthread scoped-pool wrapper** (NEW — RESEARCH Pattern 2 + §"Code Examples"; wrap the parallel section). `predict_preset`/`predict_leaf_preset`/`predict_score_by_tree_preset` must thread `nthread: i32` through from `Config` (currently NOT passed — see `predict_rows` at :980-1038 which holds `config` and the call sites at :1021/:1024 which drop it):
```rust
// nthread <= 0 → global pool (all cores); nthread > 0 → per-call scoped pool.
if nthread <= 0 {
    fill()?;                                       // global pool = all cores
} else {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthread as usize)
        .build()
        .map_err(|e| GtilError::ThreadPool(e.to_string()))?;  // new typed variant
    pool.install(fill)?;
}
```
**Anti-pattern guarded:** NEVER `ThreadPoolBuilder::build_global()` per call (once-only global mutation). Use scoped `install`.

**Up-front validation stays up front** (V5 / PAR-02): dense buffer-length check at `predict` (:902-926) and CSR `csr.validate(num_row, num_feature)` at `predict_sparse` (:969) MUST remain BEFORE the parallel section — never move validation into the per-row closure.

**Imports to add** at top of `lib.rs`:
```rust
use rayon::prelude::*;
```

---

### `crates/treelite-gtil/src/error.rs` (new error variant — supporting change for the scoped pool)

**Analog:** the existing `#[error(...)]` variants in `GtilError` (e.g. `InvalidInputShape` at error.rs:42-56, `LeafVectorTooShort` at :80-86).

The scoped-pool builder needs a typed error variant. Follow the existing `thiserror` pattern (one `#[error("...")]` attr per variant, a struct or tuple payload):
```rust
#[error("failed to build thread pool: {0}")]
ThreadPool(String),
```
Insert it before the `#[error(transparent)]` catch-all at error.rs:174-176. (RESEARCH §"Code Examples" — `GtilError::ThreadPool(e.to_string())`.)

---

### `crates/treelite-gtil/src/config.rs` (config — doc update only, no struct change)

**Analog:** the existing `Config` struct (config.rs:44-51). No field change — `nthread: i32` already exists (:50). Update ONLY the doc comments at lines 41-43 and :48-49 that currently say "accepted and recorded but never used" / "the scalar reference ignores it for allocation (recorded only)" — those become FALSE this phase (`nthread` now drives the scoped pool). RESEARCH §"State of the Art" / "Deprecated".

---

### `crates/treelite-gtil/Cargo.toml` (config)

**Analog:** the crate's own `[dependencies]` block + the workspace pin convention.

Current `[dependencies]`:
```toml
[dependencies]
treelite-core = { path = "../treelite-core" }
thiserror = { workspace = true }
```
Add (workspace-pin form, matching `thiserror = { workspace = true }`):
```toml
rayon = { workspace = true }
```

---

### `Cargo.toml` (root workspace, config)

**Analog:** the Phase-9 pin entries added to `[workspace.dependencies]` (lines 26-32 — `smallvec`, `compact_str`), which follow exactly the comment-then-pin convention.

Add to `[workspace.dependencies]` (RESEARCH §"Standard Stack" — pin `1.12.0`, verified via `cargo add --dry-run` + slopcheck `[OK]`):
```toml
# Phase 10 parallel scalar inference (PAR-01/02/04). Verified rayon v1.12.0 via
# `cargo add rayon --package treelite-gtil --dry-run` + slopcheck [OK]
# (10-RESEARCH §Package Legitimacy Audit). Row-parallel par_chunks_mut/map_init
# over the scalar GTIL row loop; scoped ThreadPool honors Config.nthread.
rayon = "1.12.0"
```

---

### `crates/treelite-core/src/model.rs` (model — add `unsafe impl Sync for Model`)

**Analog:** `SendModelRef`'s documented `unsafe impl Send` in `treelite-py/src/gtil.rs:35-40` — the existing precedent for asserting cross-thread shareability of a `&Model` with a soundness comment grounded in "predict is read-only, no interior mutability, borrow outlives use."

The Phase-10 change is the *minimal sound* version of that same argument, placed on the type itself (so rayon shares `&Model` with no wrapper). The `Model` struct (model.rs:56-117) is `!Sync` only because `TreeBuf::Borrowed` holds a `*const T`. Add, after the struct definition, the documented impl from RESEARCH §"Code Examples":
```rust
// SAFETY: Model is !Sync only because TreeBuf::Borrowed { ptr: *const T } holds
// a raw const pointer. Sharing &Model across threads is sound on the predict path:
//   1. predict takes &Model (SHARED, never &mut) — no field is mutated;
//   2. the borrowed foreign buffer is a read-only const slice whose backing memory
//      (by the from_borrowed SAFETY contract) outlives the TreeBuf and is not
//      mutated while borrowed → concurrent reads are data-race-free;
//   3. Model has NO interior mutability on the predict path; stage_serialization_fields
//      takes &mut self (model.rs:162) so it cannot run concurrently with a shared-& predict.
// Mirrors upstream Treelite sharing Model const& across OpenMP threads (predict.cc:241).
// Only Sync is asserted — NOT Send (A4); rayon shares &Model (Send iff Model: Sync),
// the model is never MOVED to another thread.
unsafe impl Sync for Model {}
```
**Anti-pattern guarded** (RESEARCH §"Anti-Patterns"): do NOT add blanket `Send` or `unsafe impl Send for TreeBuf` — over-broad. `Sync` only (A4). If the impl later surfaces a `Model` move requirement, add a separately-justified `unsafe impl Send`.

**Note on `stage_serialization_fields`:** confirmed `&mut self` (model.rs:162) — the soundness argument's point 3 holds against the live code.

---

### `crates/treelite-core/tests/model_invariants.rs` (test — rewrite `_assert_not_send`)

**Analog:** the function being replaced — `_assert_not_send` (model_invariants.rs:29-34) and its doc block (:25-28). The OTHER test in the file, `model_size_not_bloated_by_smallvec` (:39-49), STAYS unchanged.

Current (the thing to supersede, NOT merely delete):
```rust
#[allow(dead_code)]
fn _assert_not_send() {
    fn requires_send<T: Send>() {}
    // requires_send::<Model>();  // ← must NOT compile; this comment IS the invariant.
    let _ = requires_send::<i32>; // keep the helper referenced (i32: Send).
}
```
Replace with the positive shareability assertion (RESEARCH §"Replacement shareability test", lines 327-341):
```rust
/// PAR-03: `Model` is soundly SHAREABLE across threads for read-only predict
/// (documented `unsafe impl Sync`, mirroring upstream OpenMP). SUPERSEDES the
/// prior `_assert_not_send` invariant — the type intentionally became `Sync`.
#[test]
fn model_is_sync_for_readonly_predict() {
    fn requires_sync<T: Sync>() {}
    requires_sync::<Model>();          // compiles iff Model: Sync — the new contract
    fn requires_send<T: Send>() {}
    requires_send::<&Model>();         // &Model: Send follows from Model: Sync (what rayon needs)
}
```
Also update the module doc block (:1-14) which currently documents the `!Send` invariant — it now describes the `Sync` contract. The `MODEL_SIZE_BUDGET` const (:23) and its test are untouched.

---

### `crates/treelite-cubecl/src/lib.rs` (fallback caller — VERIFY ONLY, likely no code change)

**Analog:** `predict_cpu` (:347-354) and `predict_cpu_sparse` (:360-371) — the scalar-fallback callers.

Verification (already confirmed by reading the file):
- `predict_cpu` → `predict::<CpuRuntime, F>(model, data, num_row, cfg)` (:353) → the scalar-fallback gate at :322-325 calls `treelite_gtil::predict::<F>(model, data, num_row, cfg)` — `cfg` (carrying `nthread`) IS forwarded.
- `predict_cpu_sparse` (:360-371) → `treelite_gtil::predict_sparse::<F>(model, csr, num_row, cfg)` — `cfg` IS forwarded.

So `Config.nthread` already flows through to the scalar engine end-to-end. This file likely needs NO change beyond a possible doc-comment refresh; the actual nthread *use* is entirely inside `treelite-gtil` (Pattern 2). Confirm during planning that no intermediate drops `cfg` (RESEARCH Component Responsibilities row for cubecl).

---

### `crates/treelite-gtil/tests/determinism.rs` (CREATE — determinism, GTIL-08 under parallelism)

**Analog:** `crates/treelite-cubecl/tests/determinism.rs` (whole file) — copy its structure wholesale, retargeting `predict_cpu` → `treelite_gtil::predict` / `predict_sparse`.

**Fixtures to copy verbatim** (cubecl determinism.rs:24-62):
```rust
fn split_tree<T: Copy + Default>(feature: i32, threshold: T, left: T, right: T) -> Tree<T> { ... }
fn scalar_model<T, W>(trees: Vec<Tree<T>>, wrap: W, num_feature: i32) -> Model where W: Fn(ModelPreset<T>) -> ModelVariant { ... }
```
**Assertion pattern to copy** (cubecl determinism.rs:84-97) — `.to_bits()` equality, NOT `==`, so `+0.0`/`-0.0` and NaN-payload differences fail (T-06-13):
```rust
for kind in [PredictKind::Default, PredictKind::Raw, PredictKind::LeafId, PredictKind::ScorePerTree] {
    let cfg = Config { kind, nthread: 0 };
    let a = treelite_gtil::predict::<f64>(&model, &data, num_row, &cfg).unwrap();
    let b = treelite_gtil::predict::<f64>(&model, &data, num_row, &cfg).unwrap();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "...");
    }
}
```
**Adaptations for Phase 10:** (a) call `treelite_gtil::predict` directly (not `predict_cpu`); (b) run N times (the RESEARCH spec says "N repeated runs"), not just 2 — loop `for _ in 0..N` and compare each against `a`; (c) use MORE rows than the 4-row fixture so rayon actually splits work across workers (a determinism test on 1 row proves nothing about parallel reordering); (d) add a sparse variant calling `predict_sparse` if a `SparseCsr` fixture is cheap to build. Imports mirror cubecl determinism.rs:16-18 minus `treelite_cubecl`.

---

### `crates/treelite-gtil/tests/parallel_nthread.rs` (CREATE — nthread equivalence + >1-core utilization)

**Analog:** the SAME `split_tree` / `scalar_model` fixtures from `treelite-cubecl/tests/determinism.rs:24-62` (reuse them — duplicate the two helper fns, as the cubecl test itself notes it mirrors `predict_kinds.rs`).

**Test 1 — nthread equivalence** (PAR-04): assert `nthread=1` and `nthread=0` (and e.g. `nthread=2`) produce byte-identical output:
```rust
let a = treelite_gtil::predict::<f64>(&model, &data, num_row, &Config { kind, nthread: 0 }).unwrap();
let b = treelite_gtil::predict::<f64>(&model, &data, num_row, &Config { kind, nthread: 1 }).unwrap();
for (x, y) in a.iter().zip(b.iter()) { assert_eq!(x.to_bits(), y.to_bits()); }
```

**Test 2 — >1-core utilization** (PAR-01/02): gate on `std::thread::available_parallelism()` and assert the parallel run uses >1 worker via `rayon::current_num_threads()` (RESEARCH §"Standard Stack" note on `current_num_threads`):
```rust
if std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) > 1 {
    // run a multi-row predict and assert rayon::current_num_threads() > 1
    // (skip / vacuous-pass on a 1-core runner — Environment Availability note)
}
```
Use a multi-row `data` input so the parallel split is real. `nthread` defaults follow `Config::default()` (nthread: 0 → global pool). This file is the only NET-NEW test surface beyond determinism (Wave 0 gaps).

---

## Shared Patterns

### `unsafe impl <marker>` with a SAFETY doc block
**Source:** `crates/treelite-py/src/gtil.rs:37-40` (`unsafe impl Send for SendModelRef`).
**Apply to:** `crates/treelite-core/src/model.rs` (`unsafe impl Sync for Model`).
```rust
// SAFETY: see the type doc — the reference is only read for pure compute inside
// the detached region; no `TreeBuf::Borrowed` pointer is mutated or sent onward,
// and the underlying model + numpy borrow both outlive the closure.
unsafe impl Send for SendModelRef<'_> {}
```
The Phase-10 `unsafe impl Sync for Model` follows the identical "read-only, no interior mutability, backing outlives borrow" justification, generalized from the wrapper to the type (RESEARCH Pitfall 1: with `Model: Sync`, `&Model: Send` follows automatically and rayon needs no `SendModelRef` wrapper).

### `thiserror` typed-error variant
**Source:** `crates/treelite-gtil/src/error.rs` — every variant is `#[error("...")] Name { fields } | Name(payload)`.
**Apply to:** the new `GtilError::ThreadPool(String)` variant (scoped-pool build failure). Errors propagate via `?`/`collect::<Result>` — never panic (ERR-01), matching the existing serial path.

### `.to_bits()` bit-identity assertion (determinism / equivalence)
**Source:** `crates/treelite-cubecl/tests/determinism.rs:88-95`.
**Apply to:** BOTH new gtil tests (`determinism.rs`, `parallel_nthread.rs`). `assert_eq!(x.to_bits(), y.to_bits(), ...)` — distinguishes `±0.0` and NaN payloads so a reordered-accumulation result that happens to be numerically equal still fails (T-06-13).

### Workspace dependency pin convention
**Source:** root `Cargo.toml:23-46` — a comment block citing the RESEARCH legitimacy audit, then `name = "x.y.z"` (exact pin) in `[workspace.dependencies]`, consumed by crates via `name = { workspace = true }`.
**Apply to:** the `rayon = "1.12.0"` pin (root) + `rayon = { workspace = true }` (gtil Cargo.toml).

### Test fixture reuse (`split_tree` / `scalar_model`)
**Source:** `crates/treelite-cubecl/tests/determinism.rs:24-62`.
**Apply to:** both new gtil test files. These build a minimal `(num_row, 1, 1)` sigmoid model over single-split numerical trees — the smallest fixture that exercises the dense numerical traversal path. Duplicate the two helper fns into each test file (cargo integration tests don't share a module).

## No Analog Found

None. Every file maps to a concrete in-repo analog (the serial loops, the `SendModelRef` soundness precedent, the cubecl determinism test, the Phase-9 pin entries, the `_assert_not_send` test being rewritten). rayon's `par_chunks_mut`/`map_init`/`ThreadPoolBuilder` API shapes come from RESEARCH §"Code Examples" (docs.rs/rayon), which is the planner's reference for the net-new parallel idiom — but every call SITE has a serial-body analog in `lib.rs`.

## Metadata

**Analog search scope:** `crates/treelite-gtil/src/` (lib.rs serial loops, config.rs, error.rs), `crates/treelite-core/src/model.rs` + `tests/model_invariants.rs`, `crates/treelite-py/src/gtil.rs`, `crates/treelite-cubecl/src/lib.rs` + `tests/determinism.rs`, root + gtil `Cargo.toml`.
**Files scanned:** 9
**Pattern extraction date:** 2026-06-11

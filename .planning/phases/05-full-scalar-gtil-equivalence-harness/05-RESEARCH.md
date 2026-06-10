# Phase 5: Full Scalar GTIL & Equivalence Harness - Research

**Researched:** 2026-06-10
**Domain:** Reference tree-ensemble inference (GTIL) port from C++ to scalar Rust + seeded equivalence harness
**Confidence:** HIGH (upstream source is vendored read-only and was read line-by-line; the existing Rust crates were read in full; treelite 4.7.0 is installed in the local venv for capture)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: Exhaustive cross-product.** The matrix is the full product of (model types × **2 presets** `<f32,f32>`/`<f64,f64>` × **2 input dtypes** f32/f64 × **4 predict kinds** × **dense + sparse CSR** × seeds). Goldens captured once and frozen.
- **D-02: Few seeds, wide edge-seeded matrices.** ~2–3 seeds per cell, each a **wide** matrix (≈100–500 rows) deliberately seeded with edge values (NaN, ±inf, boundary thresholds, missing entries). Each seed → one committed input matrix + golden output vector.
- **D-03: Run every cell literally — no invariance pruning.** Even provably-invariant cells (`leaf_id` postprocessor/preset-invariant; `score_per_tree` postprocessor-invariant) are captured and asserted. Redundant fixtures accepted.
- **D-04: Dense↔sparse parity is an additional assertion (SC1).** Every dense AND sparse cell carries its own upstream Treelite golden, AND the harness asserts Rust-dense == Rust-sparse on identical logical data (absent CSR entries materialized as **NaN, not 0**).
- **D-05: Support both f32 and f64 input matrices**, orthogonal to the model preset — faithful to upstream's generic `Predict<InputT>` (float/double). Current Rust path is f32-input only; Phase 5 adds the f64-input path. All 4 (input × preset) combinations valid; input dtype is **not** constrained to match the preset.
- **D-06: Idiomatic Rust config struct, not upstream JSON config.** Entry points `predict(model, input, &config)` + `predict_sparse(model, csr, &config)` taking a typed `Config { kind: PredictKind, nthread, … }`, `PredictKind` a Rust enum (`Default`/`Raw`/`LeafId`/`ScorePerTree`). JSON-config parsing stays OUT of the compute crate.
- **D-07: Flat buffer + Shape descriptor return.** `predict` returns a flat `Vec` plus a `Shape` descriptor computed per kind (`GetOutputShape`). No richer typed-output abstraction.
- **D-08: Committed matrices are the contract; seeds are documentation.** The actual input matrices AND golden output vectors are the frozen source of truth. CI asserts against committed matrices and **never re-draws from a seed**.
- **D-09: Full-provenance manifest + backend field.** Manifest: OS+arch, libm/libc identity, rustc+cubecl version, every capture framework version, seed, sha256 per fixture, **and a `backend` field** (`scalar-cpu` | `cubecl-cpu` | `cuda` | `wgpu` | `rocm`). Phase 5 records `scalar-cpu`. Extends existing `Manifest` + `check_manifest`.
- **D-10 (forward only, NOT implemented this phase):** End user selects/switches compute backend at runtime across `{scalar-cpu, cubecl-cpu, cuda, wgpu, rocm}` via cubecl generics over `R: Runtime`. Implementation is Phase 6/7.
- **D-11 (in scope now):** The equivalence harness is built **backend-parameterized** — the same frozen golden matrices drive any `R: Runtime`, backend identity recorded via manifest `backend` field. Phase-5 scalar GTIL is the plain-Rust **reference/fallback** (`scalar-cpu`) that every future backend is measured against to 1e-5.
- **Verbatim port (pre-decided):** All postprocessors ported verbatim with upstream mixed-precision cast order. Per-row tree summation serial in `tree_id` order (GTIL-08); parallelism only across rows. Bounds-checked output routing → typed `GtilError`.

### Claude's Discretion

- **Sparse CSR Rust representation** — exact CSR type, provided absent entries materialize as NaN and dense↔sparse parity (D-04) holds.
- **`Config`/`PredictKind` exact field set and module placement** — derive from `gtil.h` + `config.cc`, keep core crate JSON-free (D-06).
- **Harness fixture layout / capture-script structure** under `fixtures/` — extend existing per-loader pattern.
- **Which concrete models populate each capability axis** — pick representative models per axis from test corpus + existing per-loader fixtures.

### Deferred Ideas (OUT OF SCOPE)

- cubecl GTIL kernels generic over `R: Runtime` — Phase 6 (NO cubecl kernel work this phase; only the harness seam design).
- Runtime-selectable GPU backends + per-model-class equivalence report — Phase 7.
- ROCm as v1 user-selectable backend — implementation Phase 6/7 (recorded for lineage).
- PyO3 JSON-config compatibility shim — Phase 8 PyO3 edge.
- Memory-efficiency (bytemuck/smallvec/compact_str) — Phase 9.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GTIL-01 | Predict over dense row-major input matrix | Already exists (f32-input). Needs: `Config`/kind plumbing (D-06), f64-input path (D-05), output type = `InputT`. `predict.cc:42-56` (DenseMatrixAccessor), `:398-404` (`Predict`). |
| GTIL-02 | Predict over sparse CSR (absent = NaN) | NEW. `predict.cc:58-97` (SparseMatrixAccessor materializes a dense per-thread scratch row, fills NaN then writes nonzeros). `gtil.h:85-88` (`PredictSparse` signature, `data`/`col_ind`/`row_ptr`). |
| GTIL-03 | All 4 predict kinds (`default`/`raw`/`leaf_id`/`score_per_tree`) | `default`/`raw` exist (via `pred_margin`). NEW: `leaf_id` (`predict.cc:325-345` PredictLeaf), `score_per_tree` (`:347-378` PredictScoreByTree). `PredictImpl` dispatch `:380-396`. |
| GTIL-04 | All 10 postprocessors verbatim, mixed precision preserved | 7/10 done. NEW: `signed_square`, `hinge`, `multiclass_ova`. `postprocessor.cc:19-115`. |
| GTIL-05 | Missing-value routing fires on NaN only, via default direction | Already correct (`evaluate_tree` NaN → `default_child`). `predict.cc:158-159`. Verified by sparse path (NaN materialization). |
| GTIL-06 | Categorical float-representability guard + child polarity | Partial (minimal subset shipped 04-05). NEW: full guard. `predict.cc:127-150` (`NextNodeCategorical` exact guard formula). |
| GTIL-07 | Output shaping: `GetOutputShape`, leaf-vector broadcast, averaging, f64 base-score add | Mostly done; needs per-kind `GetOutputShape` + leaf-vector broadcast for all 4 cases. `output_shape.cc:17-39`, `predict.cc:174-216` (4-way broadcast). |
| GTIL-08 | Per-row tree summation serial in `tree_id` order | Already correct (serial loop). Preserve — no tree-axis parallelism. |
| EQV-01 | Harness generates random seeded dense + sparse CSR inputs | NEW. Python capture side (numpy RNG); committed matrices are the contract (D-08). |
| EQV-02 | Golden vectors from C++ Treelite, committed with toolchain/libm manifest | NEW capture scripts. treelite 4.7.0 in local venv confirmed. |
| EQV-03 | Rust asserts within 1e-5 across model types, presets, kinds | NEW exhaustive matrix harness (D-01/D-03). |
| EQV-04 | Harness reports max observed deviation | Pattern exists (`run_equivalence` returns max `|delta|`). Generalize across matrix. |
</phase_requirements>

## Summary

Phase 5 completes the GTIL surface that has been pulled forward in minimal slices across Phases 1–4, and builds the exhaustive seeded equivalence harness that is the 1e-5 measurement instrument every later phase (cubecl/GPU) is validated against. The work is **a verbatim port, not a design exercise** — the upstream C++ at `treelite-mainline/src/gtil/` is the source of truth, it is vendored read-only, and it was read line-by-line for this research. The existing `crates/treelite-gtil` already implements the hard parts correctly (serial tree-sum, four-way leaf routing, RF averaging, f64 base-score add, bounds-checked routing, 7/10 postprocessors, softmax mixed precision). The gaps are well-bounded: sparse CSR input, the f64-input path, two new predict kinds (`leaf_id`, `score_per_tree`), three new postprocessors, the full categorical representability guard, per-kind `GetOutputShape`, and a typed `Config`/`PredictKind` entry surface.

The single largest correctness subtlety the planner must internalize: **the output buffer and accumulator type is `InputT`, not the model's leaf-output type.** Upstream instantiates `Predict<float>` and `Predict<double>` and the output pointer is the same type as the input pointer (confirmed in `c_api/gtil.cc:50-55`). So the f64-input path (D-05) must accumulate and return `f64`, while the existing f32-input path returns `f32`. Two postprocessors hardcode `float` intermediates *regardless of InputT* (`softmax` uses `float max_margin`/`float t` with a `double norm_const`; this is already ported correctly for f32). The leaf value is cast `static_cast<InputT>(...)` *before* accumulation, and base scores are always `double` added into the `InputT` accumulator. These cast orderings ARE the 1e-5 contract.

The harness is a deliberate, large, one-time investment (D-01/D-03 exhaustive, no pruning). Capture runs once via `uv run python` on the main tree against the locally-installed treelite 4.7.0; CI only ever reads committed matrices and never re-draws from seeds (D-08). The harness must be built **backend-parameterized now** (D-11) so Phase 6 plugs a cubecl `R: Runtime` in as a registration, not a refactor — but no cubecl code is written this phase.

**Primary recommendation:** Widen `crates/treelite-gtil` in place (do not rewrite). Make `predict`/`predict_sparse` generic over the input/output element type `O: PredictOut` (f32/f64), add a typed `Config { kind: PredictKind, nthread: i32 }`, port the three missing postprocessors and the full categorical guard verbatim, add a `SparseCsr` view with NaN materialization, and add a `output_shape(model, num_row, &config) -> Shape` function mirroring `output_shape.cc`. Build the harness as a data-driven runner over a committed fixture manifest that names a `backend` and iterates the exhaustive matrix, asserting dense==sparse and golden within 1e-5, reporting max deviation per cell.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dense/sparse predict, kinds, postprocessors | `treelite-gtil` (compute library) | — | The GTIL reference engine owns all inference logic; stays plain idiomatic Rust (it is the fallback every backend is measured against). |
| `Config`/`PredictKind` types | `treelite-gtil` (public API) | — | Idiomatic struct, JSON-free (D-06). JSON shim, if ever needed, lives at the Phase-8 PyO3 edge. |
| `Shape`/`output_shape` | `treelite-gtil` (public API) | — | Mirrors upstream `GetOutputShape`; consumed by the harness for buffer sizing and by Phase-8 PyO3 for numpy reshape. |
| Sparse CSR representation | `treelite-gtil` (input view) | — | A thin borrowed view (`data`/`col_ind`/`row_ptr`); materialized to NaN dense rows internally (D-04). |
| Golden capture (matrices + vectors + manifest) | Python capture scripts (`fixtures/capture_*.py`) | upstream treelite 4.7.0 | Goldens are C++-truth; captured once via `uv run python`, frozen, never regenerated in CI (D-08). |
| Equivalence assertion + max-deviation report | `treelite-harness` (dev/test crate) | `treelite-gtil` + loaders | The harness is the 1e-5 instrument; uses `anyhow`, consumes typed library errors. |
| Backend-parameterized seam | `treelite-harness` (D-11 seam design) | — | A runner abstraction keyed on a `backend` field so Phase 6/7 register a runtime without a harness refactor. NO cubecl code this phase. |

## Standard Stack

This phase adds **no new third-party runtime dependencies to the library crates.** The port is plain `std` Rust plus the already-pinned `thiserror`. The capture side is Python (`uv run`), already established.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `thiserror` | 2.0.18 (already pinned) | Typed `GtilError` at the library boundary | Established project convention (CLAUDE.md). Extend existing enum for new sparse/kind/categorical errors. |
| `std` (f32/f64 intrinsics) | rustc 1.95.0 | `exp`, `exp2`, `ln_1p`, `copysign`, `is_nan`, `min` | Direct analogs of `<cmath>` `std::exp`/`std::exp2`/`std::log1p`/`std::copysign`. Already used. |

### Supporting (capture-side / dev-only — already present)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| treelite (Python) | 4.7.0 (in local venv, verified) | Generate golden vectors via `treelite.gtil.predict` / `predict_leaf` / `predict_per_tree` | Capture scripts only; never in the Rust build graph. |
| numpy | (in venv) | Seeded random matrices, CSR construction (`scipy.sparse.csr_matrix`) | Capture scripts only. |
| scipy | (in venv) | `csr_matrix` for sparse golden capture | Capture scripts only. |
| `anyhow` | 1.0.102 (pinned) | Harness error context (ERR-02) | Dev/test crate `treelite-harness` only. |
| `approx` | 0.5.1 (pinned) | `assert_abs_diff_eq!(epsilon = 1e-5)` | Harness assertion. |
| `serde`/`serde_json` | pinned | Parse committed golden + manifest fixtures | Harness fixture I/O. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Generic `O: PredictOut` trait for output type | A `predict_f32` + `predict_f64` pair of monomorphic fns | A trait + one generic body keeps the assembly-order code single-sourced (the 1e-5 contract lives in one place); two copies risk drift. Recommend the trait, mirroring the existing `PredictScalar` pattern. |
| `rand` crate for Rust-side matrix generation | None (capture is Python-side) | D-08 forbids Rust re-drawing from seeds; the committed matrices are the contract. **Do NOT add `rand` to any Rust crate.** Generation is numpy in capture scripts. |
| Thin borrowed `SparseCsr<'a, O>` view | Owned CSR struct | Borrowed view is zero-copy-friendly for Phase-8 PyO3 (numpy `indptr`/`indices`/`data`). Materialize to a per-row NaN dense scratch internally, exactly as upstream `SparseMatrixAccessor`. |

**Installation:** No `cargo add` needed for library crates. Capture-side (one-time, on main tree):
```bash
uv run python fixtures/capture_gtil_matrix.py   # new exhaustive-matrix capture script(s)
```

**Version verification:** treelite confirmed `4.7.0` via `uv run python -c "import treelite; print(treelite.__version__)"`. rustc `1.95.0`, cargo `1.95.0`. No new crates.io packages introduced.

## Package Legitimacy Audit

> No external packages are installed by this phase. The library crates gain no new dependencies; the capture side uses Python packages already present in the repo venv (treelite 4.7.0, numpy, scipy, optionally xgboost/lightgbm/scikit-learn for model authoring). slopcheck/registry audit is **not applicable** — there is nothing to install.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none) | — | No installs in this phase |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────────────────────────────┐
   committed fixtures     │            treelite-harness (dev)            │
   (frozen, D-08)         │                                             │
  ┌──────────────────┐    │  for each matrix cell in the exhaustive     │
  │ <model>.<kind>.  │    │  matrix (model × preset × in-dtype ×        │
  │  <indtype>.<seed>│───▶│  kind × {dense,sparse} × seed):             │
  │  .dense.golden   │    │   1. load committed input matrix            │
  │  .sparse.golden  │    │   2. run Rust predict / predict_sparse      │
  │  .manifest(backend)   │   3. assert |Rust - golden| < 1e-5          │
  └──────────────────┘    │   4. assert Rust-dense == Rust-sparse (D-04)│
            ▲             │   5. record max |delta|, report per cell    │
            │             └───────────────────┬─────────────────────────┘
   one-time capture                           │ calls
   (uv run python,                            ▼
    treelite 4.7.0)        ┌──────────────────────────────────────────┐
  ┌──────────────────┐     │           treelite-gtil (library)         │
  │ capture_gtil_*.py│     │                                          │
  │  numpy seeded X   │     │  predict(model, &[O], num_row, &Config)   │
  │  scipy csr_matrix │     │  predict_sparse(model, SparseCsr, &Config)│
  │  treelite.gtil.   │     │  output_shape(model, num_row, &Config)    │
  │   predict/_leaf/  │     │                                          │
  │   _per_tree       │     │  PredictImpl dispatch on Config.kind ─────┼──┐
  └──────────────────┘     │   ├─ default → PredictRaw + postprocess    │  │
                           │   ├─ raw     → PredictRaw                  │  │
                           │   ├─ leaf_id → PredictLeaf                 │  │
                           │   └─ per_tree→ PredictScoreByTree          │  │
                           │                                          │  │
   loaded Model            │  per row (parallel-safe across rows):     │  │
  ┌──────────────────┐     │   accessor.get_row(r) ──▶ &[O] (NaN-filled│  │
  │ treelite-core     │────▶│      for absent sparse entries)           │  │
  │  ModelVariant     │     │   serial over tree_id (GTIL-08):          │  │
  │  F32 / F64        │     │     evaluate_tree → leaf_id               │  │
  └──────────────────┘     │     OutputLeafValue / OutputLeafVector     │  │
                           │   RF average → f64 base-score add          │  │
                           └────────────────────────────────────────────┘ │
                                                                           │
                            postprocessor.rs (10 verbatim) ◀───────────────┘
```

The diagram traces the primary use case (golden assert) from frozen fixture → harness runner → `treelite-gtil` predict → `treelite-core` Model → postprocessor, and the capture path that produced the goldens. The `Config.kind` dispatch and the dense/sparse accessor abstraction are the two structural seams.

### Recommended Project Structure
```
crates/treelite-gtil/src/
├── lib.rs            # predict/predict_sparse entry points (generic O), PredictImpl dispatch
├── config.rs         # NEW: Config { kind: PredictKind, nthread }, PredictKind enum (D-06)
├── shape.rs          # NEW: Shape descriptor + output_shape() (D-07, mirrors output_shape.cc)
├── accessor.rs       # NEW: DenseAccessor + SparseCsr view + get_row → &[O] NaN-fill (GTIL-02)
├── postprocessor.rs  # extend: + signed_square, hinge, multiclass_ova (GTIL-04)
└── error.rs          # extend: + sparse/kind/categorical variants

crates/treelite-harness/src/
├── lib.rs            # generalize: matrix runner, backend-parameterized seam (D-11)
└── manifest.rs       # NEW: extend Manifest with `backend` field (D-09)

crates/treelite-harness/tests/
└── gtil_matrix.rs    # NEW: drives the exhaustive matrix; dense==sparse parity; max-dev report

fixtures/
├── capture_gtil_matrix.py    # NEW: exhaustive seeded capture (dense + CSR, all kinds, both dtypes)
├── gtil/                      # NEW subdir for the matrix fixtures (keeps root tidy)
│   ├── <model>.<preset>.<indtype>.<kind>.<dense|sparse>.s<seed>.golden.json
│   └── <model>...manifest.json   # carries `backend: "scalar-cpu"`
```

### Pattern 1: InputT-typed output (the central correctness pattern)
**What:** The output buffer/accumulator type equals the *input* element type, NOT the leaf-output type. `Predict<InputT>` writes `InputT*`. Leaf values are `static_cast<InputT>` before accumulation; base scores are always `double` added in.
**When to use:** Everywhere in the predict body. The current code returns `Vec<f32>`; D-05 requires the body be generic over `O ∈ {f32, f64}` so f64 input yields `Vec<f64>`.
```rust
// Source: predict.cc:236 (Array3DView<InputT> output_view), :228 static_cast<InputT>(LeafValue),
//         :301 output += base_score_view (double), c_api/gtil.cc:50-55 (output type == input type)
pub trait PredictOut: Copy + PartialOrd {
    fn nan() -> Self;
    fn zero() -> Self;
    fn from_f32_threshold(v: Self, t: f32) -> bool; // not needed if threshold compared in T domain
    fn from_leaf_f32(v: f32) -> Self;   // static_cast<InputT>(leaf as f32-ish)
    fn from_leaf_f64(v: f64) -> Self;   // static_cast<InputT>(leaf)
    fn add_base_score(self, base: f64) -> Self; // (self as f64 + base) as Self
    // ... exp/exp2/ln_1p/copysign forwarded to the concrete type for postprocessors
}
```
Note the existing `PredictScalar` trait is about the *tree's* threshold/leaf domain `T`; the new `O` (input/output) trait is orthogonal. There are now potentially 4 (O × T) combinations to instantiate: {f32,f64} input × {f32,f64} preset. Comparison `NextNode<InputT,ThresholdT>` promotes `InputT` to the wider of the two for the comparison (C++ usual arithmetic conversions) — preserve this.

### Pattern 2: Sparse accessor with per-row NaN materialization
**What:** A CSR matrix is read by, per row, filling a scratch dense row with NaN, then writing the row's nonzeros at their column indices. This is exactly how absent = NaN (not 0) is realized, and it makes the dense and sparse paths share `evaluate_tree` verbatim (D-04 parity is then structural).
**When to use:** `predict_sparse` only.
```rust
// Source: predict.cc:58-97 (SparseMatrixAccessor::GetRow)
fn get_row_sparse<O: PredictOut>(
    data: &[O], col_ind: &[u64], row_ptr: &[u64],
    row_id: usize, num_feature: usize, scratch: &mut [O],
) -> &[O] {
    for v in scratch.iter_mut() { *v = O::nan(); }           // absent = NaN
    let (b, e) = (row_ptr[row_id] as usize, row_ptr[row_id + 1] as usize);
    for i in b..e { scratch[col_ind[i] as usize] = data[i]; } // write nonzeros
    scratch
}
```
For single-threaded scalar (Phase 5), one scratch row suffices; upstream allocates `nthread * num_feature` for thread safety. Bounds-check `col_ind[i] < num_feature` and `row_ptr` monotonicity → typed error (do not panic).

### Pattern 3: Per-kind dispatch + GetOutputShape
**What:** `PredictImpl` branches on `config.kind`. Output shapes differ per kind (`output_shape.cc`): `default`/`raw` → `(num_row, num_target≥1, max_num_class)`; `leaf_id` → `(num_row, num_tree)` of integer leaf IDs; `score_per_tree` → `(num_row, num_tree, leaf_vector_shape[0]*leaf_vector_shape[1])`.
**When to use:** Compute `output_shape()` first (callers size the buffer), then `predict` fills it.
```rust
// Source: output_shape.cc:17-39, predict.cc:380-396 (PredictImpl)
match config.kind {
    PredictKind::Default => { predict_raw(...); apply_postprocessor(...); }
    PredictKind::Raw     => { predict_raw(...); }
    PredictKind::LeafId  => { predict_leaf(...); }       // NEW: output_view(row, tree) = leaf_id
    PredictKind::ScorePerTree => { predict_score_by_tree(...); } // NEW: per-tree leaf scalar/vector
}
```
Note `leaf_id` output holds *integer node IDs cast to `O`* (upstream stores them in the same `InputT` buffer: `output_view(row_id, tree_id) = leaf_id;`). `score_per_tree` writes the raw leaf scalar OR each leaf-vector element with NO postprocessing, NO averaging, NO base score.

### Pattern 4: Backend-parameterized harness seam (D-11, design only)
**What:** A trait/enum that names which runtime produced/asserted a vector, so the same frozen goldens drive any backend. Phase 5 implements exactly one variant (`scalar-cpu`); the seam is the registration point Phase 6 fills.
**When to use:** Harness runner. Recommend the lightest seam that satisfies "adding a backend is a registration, not a refactor":
```rust
// A predict function pointer + a backend tag is sufficient (no cubecl this phase).
pub enum Backend { ScalarCpu /* , CubeclCpu, Cuda, Wgpu, Rocm (Phase 6/7) */ }
type PredictFn = fn(&Model, &[f64], usize, &Config) -> anyhow::Result<Vec<f64>>;
struct RunnerCase { backend: Backend, predict: PredictFn, /* + sparse fn */ }
```
Avoid over-engineering into a full trait object hierarchy now — the manifest `backend` field (D-09) plus a registry of `(Backend, predict_fn)` pairs is the minimal seam. Document that Phase 6 registers `(CubeclCpu, cubecl_predict)` without touching the matrix iteration or fixture loading.

### Anti-Patterns to Avoid
- **Parallelizing the tree-axis sum.** Float add is non-associative; GTIL-08 mandates serial `tree_id` order. Parallelism is allowed only across rows. The existing code is correct — keep it.
- **Doing softmax/sigmoid math in f64 for an f32 input.** `softmax` hardcodes `float max_margin`/`float t` with `double norm_const` regardless of `InputT`; `sigmoid`/`multiclass_ova` use `model.sigmoid_alpha` which is `float`. For f64 input, the leaf accumulation is f64 but the postprocessor intermediates that upstream hardcodes as `float` must stay `float` (cast in, cast out). Mirror upstream exactly — do not "promote for consistency."
- **Treating absent CSR entries as 0.** They are NaN (SC1). The NaN then routes via `default_child` (GTIL-05). Zero would silently mis-route.
- **Re-drawing input matrices from a seed in Rust/CI.** D-08: committed matrices are the contract. Seeds are documentation only.
- **Rewriting `crates/treelite-gtil`.** Widen in place; the serial-sum/routing/averaging/base-score core is the 1e-5-proven spine.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Margin transforms (softmax, sigmoid, exp2, log1p, signed_square, hinge, ova) | Custom math | Verbatim port of `postprocessor.cc:19-115` | Cast order IS the 1e-5 contract; any reordering shifts ULPs past tolerance. |
| Categorical membership guard | Ad-hoc `as u32` cast | Verbatim `NextNodeCategorical` guard (`predict.cc:135-144`) | The exact `max_representable_int = min(u32::MAX, 1<<digits)` formula and the `fvalue < 0 \|\| fabs > max` rejection are load-bearing for GTIL-06. |
| Output shape per kind | Inferring shapes ad-hoc | Port `GetOutputShape` (`output_shape.cc`) | Callers (and Phase-8 numpy) depend on the exact dims; `num_target>1 ? (r,nt,mc) : (r,1,mc)` branch is subtle. |
| Sparse→dense row | Custom sparse traversal in the tree walk | NaN-materialized scratch row + shared `evaluate_tree` | Makes dense==sparse parity structural (D-04) and reuses the proven traversal. |
| Random input + golden generation | Rust RNG + a Rust "reference" | numpy seeded capture + `treelite.gtil.predict` C++ golden | The golden must be C++-truth, captured once (D-08/EQV-02). A Rust reference would be circular. |
| Provenance/manifest | Free-form notes | Extend the existing `Manifest` struct + `check_manifest` (+ `backend` field) | The drift-warning machinery already exists and is the diagnosability mechanism (D-09). |

**Key insight:** Every numeric operation in this phase already has a canonical upstream implementation in vendored read-only source. The job is faithful transcription with explicit cast ordering, not invention. The only genuinely new *design* surface is (a) the `O`-generic output type plumbing and (b) the harness matrix runner + backend seam — and both have a clear minimal shape.

## Common Pitfalls

### Pitfall 1: Output type confused with leaf-output type
**What goes wrong:** Assuming the f64-input path returns the leaf-output type, or that f32-input always returns f32 even for an f64 model. The output is `InputT` (= the *input* dtype), independent of the model preset.
**Why it happens:** `model.output_type` in the Python layer is the *leaf* type; the C++ `Predict<InputT>` output is `InputT`. These differ.
**How to avoid:** Make the predict body generic over `O` (input/output element). f32 input → `Vec<f32>`, f64 input → `Vec<f64>`, for *either* preset. Confirmed by `c_api/gtil.cc:50-55` (output cast to the same type as input).
**Warning signs:** A 4-cell input×preset matrix where the `(f32-input, f64-preset)` or `(f64-input, f32-preset)` cells fail or won't compile.

### Pitfall 2: softmax/postprocessor intermediate precision drift on the f64 path
**What goes wrong:** Generalizing `softmax` to use `O` for `max_margin`/`t` makes the f64 path use f64 intermediates — but upstream hardcodes `float max_margin`/`float t` *even when `InputT == double`* (`postprocessor.cc:59-61`). Same for `sigmoid_alpha`/`ratio_c` which are `float` model fields.
**Why it happens:** "Make it generic" instinct over-promotes.
**How to avoid:** Keep the upstream-hardcoded `float` intermediates as `f32` in BOTH instantiations. Only the row buffer element type is `O`; the reduction scalars follow upstream's literal types. The current f32 port is already correct; the f64 instantiation must NOT change these to f64.
**Warning signs:** f64-input goldens drift at ~1e-6–1e-5 only for `softmax`/`sigmoid`-family postprocessors.

### Pitfall 3: Categorical guard subset vs. full guard
**What goes wrong:** 04-05 shipped a *minimal* guard (`fvalue < 0 || !finite || fvalue > u32::MAX`). The full upstream guard is `max_representable_int = min(static_cast<InputT>(u32::MAX), static_cast<InputT>(1<<numeric_limits<InputT>::digits))` and rejects `fvalue < 0 || fabs(fvalue) > max_representable_int`. For f32, `1<<24` (mantissa) is smaller than u32::MAX, so large-but-u32-fitting floats that are NOT exactly representable must be rejected.
**Why it happens:** The minimal subset passed the one Phase-4 categorical fixture.
**How to avoid:** Port the exact formula (`predict.cc:135-138`). `numeric_limits<float>::digits == 24`, `numeric_limits<double>::digits == 53`. For f32 input: `max = min(2^32-1 as f32, 2^24 as f32) = 2^24`. For f64 input: `max = min(2^32-1 as f64, 2^53 as f64) = 2^32-1`. Seed the categorical matrix with values in the gap (e.g., `2^24 + 1` for f32) to actually exercise it.
**Warning signs:** Categorical golden passes for small categories, fails only on edge-seeded large category values.

### Pitfall 4: libm divergence across capture vs. CI environment
**What goes wrong:** A 1e-5 failure that is really a glibc/libm `exp`/`exp2`/`log1p` ULP difference between the capture machine and the running machine.
**Why it happens:** Transcendental functions are not bit-identical across libm versions/platforms.
**How to avoid:** The manifest records os/arch/libc (existing) — extend with rustc + (placeholder) cubecl version + per-framework versions + `backend` (D-09). `check_manifest` warns (never fails) on drift. The 1e-5 tolerance is the absorber; never loosen it to mask a real gap. Capture on the same Linux/glibc 2.39 x86_64 environment that CI runs (current machine).
**Warning signs:** Failures appear only on a different distro/arch; `check_manifest` warning printed.

### Pitfall 5: `score_per_tree` shape with scalar-leaf models
**What goes wrong:** `score_per_tree` output third dim is `leaf_vector_shape[0] * leaf_vector_shape[1]`. For scalar-leaf models (the common case) that product is 1, and upstream writes `output_view(row, tree, 0) = LeafValue`. Treating it as always-leaf-vector OOBs or mis-sizes.
**Why it happens:** The kind name suggests vectors.
**How to avoid:** Port `PredictScoreByTree` (`predict.cc:347-378`) literally: `if HasLeafVector → write each element; else → write scalar at index 0`. No postprocessing/averaging/base-score for this kind (nor for `leaf_id`).
**Warning signs:** `score_per_tree` fails on RF/multiclass (leaf-vector) models or on scalar models depending on which assumption was baked in.

### Pitfall 6: nthread semantics
**What goes wrong:** Misinterpreting `nthread <= 0` as "single thread."
**Why it happens:** `Config` default is `nthread{0}`.
**How to avoid:** Upstream `ThreadConfig`: `nthread <= 0` ⇒ use ALL threads (`threading_utils.h:74-80`). For the Phase-5 scalar reference, the engine is single-threaded by design (the deterministic fallback); `nthread` is honored *semantically* in the `Config` type (carried for Phase-6) but the scalar path computes serially regardless. Document that `nthread` is accepted-and-recorded but the scalar reference is single-threaded (parallelism arrives with cubecl in Phase 6). This does NOT affect correctness because GTIL-08 mandates serial tree-sum and row-parallelism is numerically identical to row-serial.

## Code Examples

Verified patterns from vendored upstream source.

### The three missing postprocessors (GTIL-04)
```rust
// Source: postprocessor.cc:22-31, 77-82
// signed_square: copysign(margin*margin, margin)  — runs in InputT
pub fn signed_square(v: f32) -> f32 { (v * v).copysign(v) }

// hinge: 1 if >0 else 0
pub fn hinge(v: f32) -> f32 { if v > 0.0 { 1.0 } else { 0.0 } }

// multiclass_ova: row-wise sigmoid with model.sigmoid_alpha (float), over num_class cells
// (NOT softmax — independent per-class sigmoid)
pub fn multiclass_ova(sigmoid_alpha: f32, row: &mut [f32]) {
    for c in row.iter_mut() {
        *c = 1.0_f32 / (1.0_f32 + (-sigmoid_alpha * *c).exp());
    }
}
```
`multiclass_ova` is per-`(row, target)` over that target's `num_class` cells (same loop structure as `softmax` in `apply_postprocessor`). `signed_square`/`hinge` are per-cell. Add `"signed_square"`, `"hinge"`, `"multiclass_ova"` arms to `apply_postprocessor`'s match (removing them from `UnsupportedPostprocessor`).

### Full categorical guard (GTIL-06)
```rust
// Source: predict.cc:127-150 — exact representability formula
// digits: f32 = 24, f64 = 53
fn next_node_categorical_full<O>(fvalue: O, category_list: &[u32],
    category_list_right_child: bool, left: i32, right: i32) -> i32
where O: /* float */ {
    // max_representable_int = min( O(u32::MAX), O(1u64 << O::DIGITS) )
    let max_repr = o_min(O::from_u32_max(), O::from_pow2(O::DIGITS));
    let matched = if fvalue < O::zero() || o_fabs(fvalue) > max_repr {
        false
    } else {
        let cv = o_truncate_to_u32(fvalue);
        category_list.contains(&cv)
    };
    if category_list_right_child {
        if matched { right } else { left }
    } else if matched { left } else { right }
}
```
For f32: `max_repr = min(4294967295.0f32, 16777216.0f32) = 16777216.0` (2^24). For f64: `min(4294967295.0, 9007199254740992.0) = 4294967295.0`. This must be exercised by edge-seeded categorical matrices.

### Per-kind output shape (GTIL-07, D-07)
```rust
// Source: output_shape.cc:17-39
pub fn output_shape(model: &Model, num_row: u64, config: &Config) -> Shape {
    let num_tree = model.get_num_tree();
    let max_num_class = model.num_class.iter().copied().max().unwrap_or(1).max(1) as u64;
    match config.kind {
        PredictKind::Default | PredictKind::Raw => {
            let nt = if model.num_target > 1 { model.num_target as u64 } else { 1 };
            Shape::dims3(num_row, nt, max_num_class)
        }
        PredictKind::LeafId => Shape::dims2(num_row, num_tree),
        PredictKind::ScorePerTree => {
            let lvs = model.leaf_vector_shape[0] as u64 * model.leaf_vector_shape[1] as u64;
            Shape::dims3(num_row, num_tree, lvs)
        }
    }
}
```
Note `default`/`raw` collapse `num_target==1` to dim `1` (not omitted). The current internal `Shape` (in lib.rs) is the *predict-internal* `(num_row, num_target≥1, max_num_class)` indexer; the *public* `Shape` returned to callers (D-07) is the per-kind dims vector. Keep both, or unify carefully — the public one is what Phase-8 numpy reshapes against.

### Capture script skeleton (EQV-01/EQV-02, D-08/D-09)
```python
# Source: extends fixtures/capture_lightgbm.py pattern; uses scipy CSR + the 3 GTIL kinds
import numpy as np, scipy.sparse, treelite, hashlib, json, platform
SEED = 1234
rng = np.random.RandomState(SEED)
X = rng.uniform(-5, 5, size=(200, n_feat))
# inject edge values: NaN (missing), +-inf, boundary thresholds, large categoricals
X[0, 0] = np.nan; X[1, 1] = np.inf; X[2, 2] = 2**24 + 1  # f32 categorical-gap value
for dtype in (np.float32, np.float64):
    Xc = X.astype(dtype)
    Xs = scipy.sparse.csr_matrix(np.where(np.isnan(Xc), 0.0, Xc))  # NOTE: see caveat
    default = treelite.gtil.predict(model, Xc)                       # kind=default
    raw     = treelite.gtil.predict(model, Xc, pred_margin=True)     # kind=raw
    leaf    = treelite.gtil.predict_leaf(model, Xc)                  # kind=leaf_id
    pertree = treelite.gtil.predict_per_tree(model, Xc)              # kind=score_per_tree
    # ... freeze each as a committed matrix+vector with manifest{backend:"scalar-cpu", ...}
```
**Caveat to resolve in planning (Open Q1):** scipy `csr_matrix` cannot natively store NaN as "present"; absent = NaN is a GTIL *semantic* (the C API materializes NaN for absent columns). To exercise dense↔sparse parity (D-04), the sparse fixture's *absent* columns are the ones that become NaN — so the dense matrix used for the sparse golden must have NaN exactly where the CSR has no entry. The capture must build the CSR from the *nonzero/present* mask and the matching dense matrix with NaN in the absent positions, then assert `treelite.gtil.predict(dense_with_nan) == treelite.gtil.predict(csr)` at capture time. This is the canonical D-04 construction; nail it in the capture script.

## Runtime State Inventory

> This is a greenfield compute/test phase (no rename/refactor, no stored data, no live services, no OS registration). The only persisted artifacts are the committed fixture files. Inventory included for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores. Goldens are flat JSON fixtures committed to git. | None |
| Live service config | None — verified, no external services. | None |
| OS-registered state | None — verified, no scheduled tasks/daemons. | None |
| Secrets/env vars | None — verified, no secrets; capture uses local venv only. | None |
| Build artifacts | Capture-only Python packages (xgboost/lightgbm/scikit-learn) in the untracked `uv` venv; treelite 4.7.0 present. NOT in the Rust build graph. | Ensure capture runs on the main tree (venv/pyproject untracked, absent from worktrees — per MEMORY.md). |

## Validation Architecture

> nyquist_validation is enabled (config.json `workflow.nyquist_validation: true`). This section maps phase requirements to automated tests so VALIDATION.md can be generated.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]` + `cargo test`), `approx` for float asserts |
| Config file | none — workspace `cargo test` is the runner |
| Quick run command | `cargo test -p treelite-gtil` (unit tests: postprocessors, categorical guard, shape, sparse NaN-fill) |
| Full suite command | `cargo test --workspace` (includes the exhaustive harness matrix in `treelite-harness/tests/gtil_matrix.rs`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GTIL-01 | Dense predict, f32 + f64 input | unit + golden | `cargo test --workspace gtil_matrix` | ❌ Wave 0 (matrix test) |
| GTIL-02 | Sparse CSR predict, absent=NaN | unit + golden | `cargo test -p treelite-gtil sparse` ; `... gtil_matrix` | ❌ Wave 0 |
| GTIL-03 | 4 predict kinds | golden | `cargo test --workspace gtil_matrix` | ❌ Wave 0 |
| GTIL-04 | 10 postprocessors verbatim | unit | `cargo test -p treelite-gtil postprocessor` | ⚠️ partial (7/10 tested) |
| GTIL-05 | NaN-only missing routing | unit | `cargo test -p treelite-gtil missing_value` | ❌ Wave 0 |
| GTIL-06 | Categorical guard + polarity | unit | `cargo test -p treelite-gtil categorical_guard` | ❌ Wave 0 |
| GTIL-07 | Output shaping per kind + broadcast | unit | `cargo test -p treelite-gtil output_shape` | ❌ Wave 0 |
| GTIL-08 | Serial tree-sum order | unit (existing behavior) | `cargo test -p treelite-gtil` | ✅ (serial loop in place) |
| EQV-01 | Seeded dense+sparse inputs committed | fixture presence | capture script + committed fixtures | ❌ Wave 0 (capture) |
| EQV-02 | C++ goldens + manifest committed | fixture presence | committed `*.golden.json` + `*.manifest.json` | ❌ Wave 0 (capture) |
| EQV-03 | 1e-5 across matrix | golden | `cargo test --workspace gtil_matrix` | ❌ Wave 0 |
| EQV-04 | Max-deviation report | golden (asserts + prints) | `cargo test --workspace gtil_matrix -- --nocapture` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-gtil` (fast unit gate: postprocessors, guard, shape, sparse NaN-fill).
- **Per wave merge:** `cargo test --workspace` (full matrix + no-regression on Phases 1–4 goldens).
- **Phase gate:** Full suite green before `/gsd-verify-work`; existing binary `(num_row,1,1)` path stays byte-identical (no regression).

### Wave 0 Gaps
- [ ] `fixtures/capture_gtil_matrix.py` — one-time exhaustive seeded capture (dense + CSR, both dtypes, all 4 kinds) via `uv run python` on main tree; freezes matrices + C++ goldens + manifests with `backend: "scalar-cpu"`.
- [ ] Committed `fixtures/gtil/*.golden.json` + `*.manifest.json` — the frozen contract (D-08).
- [ ] `crates/treelite-harness/tests/gtil_matrix.rs` — the matrix runner (RED until the gtil widening lands), dense==sparse parity (D-04), max-dev report (EQV-04).
- [ ] Unit-test scaffolds in `treelite-gtil` for the categorical guard edge values, sparse NaN-fill, and the 3 new postprocessors.
- [ ] Decide concrete corpus models per axis (see Open Q2) before capture.

## Security Domain

> security_enforcement is enabled (config.json), ASVS level 1. This is a numeric compute library with no network/auth/session surface; the relevant ASVS category is input validation (malformed model / malformed input must never cause OOB or panic).

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface. |
| V3 Session Management | no | No sessions. |
| V4 Access Control | no | Library, no access control. |
| V5 Input Validation | **yes** | Bounds-checked traversal + routing → typed `GtilError`, never OOB/panic (existing pattern; extend for sparse `col_ind`/`row_ptr` and new kinds). |
| V6 Cryptography | no (sha256 used only as a fixture integrity tag, not a security control) | `hashlib.sha256` in capture for fixture provenance only. |

### Known Threat Patterns for {Rust scalar GTIL + untrusted Model + untrusted input}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed `num_feature`/negative dims → OOB slice / overflow | Tampering / DoS | Existing `predict` saturating-mul + `InvalidInputShape` guard; extend to sparse `row_ptr[num_row]` length checks. |
| Sparse `col_ind[i] >= num_feature` → OOB write into scratch row | Tampering | Bounds-check each `col_ind[i] < num_feature` → new `GtilError` variant (do NOT panic). |
| Non-monotonic / out-of-range `row_ptr` → OOB slice | Tampering / DoS | Validate `row_ptr` monotone non-decreasing and `row_ptr[num_row] <= data.len()` → typed error. |
| Malformed `leaf_vector_shape` → `score_per_tree` OOB | Tampering | Bounds-safe leaf-vector access (existing `has_leaf_vector`/`LeafVectorTooShort` pattern) extended to the per-tree kind. |
| Categorical `as u32` cast on out-of-range float → UB-ish truncation | Tampering | Full representability guard (GTIL-06) rejects before cast. |
| `nthread` integer extremes | DoS | Scalar path is single-threaded; `nthread` recorded but not used to allocate — no amplification. |

All mitigations follow the **existing** crate posture: every upstream fatal/unchecked path becomes a returned `GtilError`. The new sparse and per-tree-kind code paths are the only places that need *new* bounds checks; reuse the established error-variant style.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Minimal pulled-forward GTIL slice (Phases 1–4): f32-input dense only, `default`/`raw` only, 7/10 postprocessors, minimal categorical guard | Complete scalar GTIL surface | Phase 5 (this) | The reference engine becomes the full 1e-5 instrument every backend is measured against. |
| Ad-hoc per-loader goldens (`lightgbm_*`, `sklearn_*`, `xgb_3format`) with bespoke local serde structs in each test | Exhaustive matrix harness driven by a uniform fixture/manifest schema with a `backend` field | Phase 5 | One harness, one schema, backend-ready (D-09/D-11). |
| Manifest without backend identity | Manifest with `backend` field | Phase 5 (D-09) | Future 1e-5 misses are diagnosable per backend. |

**Deprecated/outdated:** Nothing deprecated — upstream treelite 4.7.0 is the frozen porting target; v5 wire format only (per CLAUDE.md). The existing per-loader golden tests stay (they remain valid regression gates); the new matrix harness is additive.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Output element type == input element type (`InputT`), independent of preset, for ALL kinds | Pattern 1 / Pitfall 1 | If wrong, the f64-input path returns the wrong type and every f64 cell fails. **Mitigation:** confirmed by reading `c_api/gtil.cc:50-55` and `predict.cc:236` — output_view is `Array3DView<InputT>`. Confidence HIGH. |
| A2 | softmax/sigmoid `float` intermediates stay `float` even for f64 input | Pitfall 2 | f64 softmax goldens drift. **Mitigation:** `postprocessor.cc:59-61` literally declares `float max_margin`/`float t`; `sigmoid_alpha`/`ratio_c` are `float` fields. Confidence HIGH (read source). |
| A3 | `nthread` does not affect numeric output (row-parallel == row-serial), so scalar single-threaded reference is bit-equivalent to upstream multithreaded for the goldens | Pitfall 6 | If upstream introduced any cross-row reduction, single-threaded could differ. **Mitigation:** `PredictRaw` parallelizes only over `row_id`; per-row tree-sum is serial. No cross-row reduction exists. Confidence HIGH. |
| A4 | `leaf_id` output stores integer node IDs cast into the `InputT` buffer (not a separate integer buffer) | Pattern 3 | Wrong buffer type for `leaf_id`. **Mitigation:** `predict.cc:340` `output_view(row,tree) = leaf_id;` where output_view is `Array2DView<InputT>`. Confidence HIGH. |
| A5 | scipy CSR + matching NaN dense can reproduce absent=NaN parity at capture time | Code Examples / Open Q1 | If the construction is subtly wrong, the sparse golden won't equal the dense-with-NaN golden and D-04 parity is untestable. **Mitigation:** assert `predict(dense_with_nan)==predict(csr)` AT CAPTURE TIME; flagged as Open Q1. Confidence MEDIUM (needs the capture-time assert to confirm). |
| A6 | No new third-party Rust crate is needed (no `rand`) | Standard Stack | If the planner adds `rand`, it violates D-08 (CI must not re-draw). **Mitigation:** generation is numpy-side; Rust only reads committed matrices. Confidence HIGH. |
| A7 | The full categorical representability guard uses `digits = 24 (f32) / 53 (f64)` giving `max_repr = 2^24 (f32) / 2^32-1 (f64)` | Pitfall 3 / Code Examples | Wrong threshold mis-classifies edge categoricals. **Mitigation:** `predict.cc:135-138` + IEEE-754 `numeric_limits::digits`. Confidence HIGH (read source + standard constants). |

## Open Questions

1. **Sparse-with-NaN capture construction (D-04).**
   - What we know: absent CSR entries become NaN via the C API's `SparseMatrixAccessor`; scipy `csr_matrix` stores nonzeros, not NaN-as-present.
   - What's unclear: the exact capture-script recipe that produces a CSR whose *absent* columns equal the NaN positions of the dense matrix used for the dense golden, so dense-with-NaN and sparse predict identically.
   - Recommendation: in `capture_gtil_matrix.py`, build a *presence mask*, construct the dense matrix with NaN in absent positions and the CSR from present positions, then assert `treelite.gtil.predict(dense_nan) == treelite.gtil.predict(csr)` at capture time and freeze both goldens. Resolve during Wave 0 capture (it is a capture-script detail, not a Rust-design blocker).

2. **Concrete corpus models per capability axis (Claude's Discretion).**
   - What we know: the vendored examples are sparse (LightGBM categorical: `toy_categorical`, `sparse_categorical`; LightGBM numerical: `deep_lightgbm`; XGBoost: `mushroom` + `xgb_3format`). The existing per-loader goldens cover XGBoost/LightGBM/sklearn families. There is no vendored multiclass/leaf-vector or true-sparse XGBoost example.
   - What's unclear: which exact models populate the multiclass/leaf-vector-broadcast and sparse axes for the matrix.
   - Recommendation: reuse existing fixtures where they cover an axis (XGBoost `xgb_3format` for binary scalar; LightGBM `toy_categorical`/`sparse_categorical` for categorical+sparse; sklearn RF/GB goldens already exist for multiclass leaf-vector). For axes not covered (e.g., XGBoost multiclass `multi:softmax`/`softprob`, a multi-target RF for the 4-way leaf-vector broadcast), author small fresh models in the capture script (the LightGBM/sklearn capture scripts already author models). Pick ONE representative model per (sparse, categorical, multiclass/leaf-vector, missing-value) axis to keep the frozen fixture size bounded (D-02 wide matrices, few seeds).

3. **Public `Shape` vs. internal predict-`Shape` collision.**
   - What we know: `lib.rs` already has a private `Shape<'m>` struct used as the predict-internal indexer.
   - What's unclear: whether to reuse that name for the public per-kind shape descriptor (D-07).
   - Recommendation: rename the internal one (e.g., `OutputLayout`) and introduce a public `Shape { dims: SmallVec/Vec<u64> }` (or a fixed-size enum) returned by `output_shape()`. Avoids a name clash and keeps the public contract clean for Phase-8 numpy reshape. Pure planning decision — low risk.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo | Build + test | ✓ | 1.95.0 | — |
| uv (Python runner) | Golden capture | ✓ | 0.11.8 | — |
| Python treelite | Golden capture (C++ truth) | ✓ | 4.7.0 (in venv) | — |
| numpy | Seeded matrices | ✓ | in venv | — |
| scipy (`csr_matrix`) | Sparse capture | likely (venv) — confirm at capture | TBD | construct CSR arrays by hand from numpy if scipy absent |
| xgboost/lightgbm/scikit-learn | Authoring fresh axis models | partly (capture-only, installed ad-hoc per prior phases) | — | reuse vendored/existing fixtures where possible |
| cubecl | NOT this phase (forward only, manifest placeholder) | n/a | record version as "n/a" or placeholder in manifest | — |

**Missing dependencies with no fallback:** none — treelite 4.7.0 confirmed present.
**Missing dependencies with fallback:** scipy (hand-build CSR arrays if absent); ML framework packages for fresh axis models (prefer reusing existing fixtures).

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/src/gtil/predict.cc` (vendored read-only) — full predict: DenseMatrixAccessor (:42-56), SparseMatrixAccessor (:58-97), NextNode (:99-125), NextNodeCategorical (:127-150), EvaluateTree (:152-172), OutputLeafVector 4-way broadcast (:174-216), OutputLeafValue (:218-229), PredictRaw + averaging + base-score (:231-305), ApplyPostProcessor (:307-323), PredictLeaf (:325-345), PredictScoreByTree (:347-378), PredictImpl dispatch (:380-396), Predict/PredictSparse (:398-423).
- `treelite-mainline/src/gtil/postprocessor.cc` (vendored) — all 10 postprocessors (:19-115), softmax mixed precision (:57-75), exp2 in exponential_standard_ratio (:44-47).
- `treelite-mainline/src/gtil/output_shape.cc` (vendored) — GetOutputShape per kind (:17-39).
- `treelite-mainline/src/gtil/config.cc` (vendored) — Configuration JSON parsing semantics (:16-46), predict_type strings.
- `treelite-mainline/include/treelite/gtil.h` (vendored) — PredictKind enum + dims docs (:26-47), Configuration (:50-55), Predict/PredictSparse signatures (:66-88).
- `treelite-mainline/src/c_api/gtil.cc` (vendored) — output type == input type dispatch (:44-76) [confirms A1].
- `treelite-mainline/python/treelite/gtil/gtil.py` (vendored) — predict/predict_leaf/predict_per_tree → predict_type strings (the capture API).
- `treelite-mainline/include/treelite/detail/threading_utils.h` (vendored) — ThreadConfig nthread semantics (:66-80) [A3/Pitfall 6].
- `crates/treelite-gtil/src/{lib.rs,postprocessor.rs,error.rs}` (in-repo) — existing spine to widen.
- `crates/treelite-harness/src/lib.rs` + `tests/lightgbm.rs` (in-repo) — Golden/Manifest/check_manifest + golden-assert pattern to generalize.
- `fixtures/capture_lightgbm.py` (in-repo) — capture-script pattern (numpy seed, manifest, sha256, treelite.gtil.predict golden).
- Local environment probes — treelite 4.7.0, rustc/cargo 1.95.0, uv 0.11.8 (verified via Bash).

### Secondary (MEDIUM confidence)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Cubecl_conditionals.md` — cubecl control-flow constraints (if-as-statement, no if-expression, thread divergence) — informs the D-11 seam scope (no kernel work this phase).
- crates.io API — `rand` 0.10.1 (2026-04-11) — noted only to confirm it is NOT needed (D-08).

### Tertiary (LOW confidence)
- None — all factual claims are grounded in vendored source or local probes.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new deps; everything is std + existing pinned crates, verified.
- Architecture / port surface: HIGH — every behavior read line-by-line in vendored read-only source; the gap vs. existing Rust is precisely enumerated.
- Pitfalls: HIGH — the four central numeric subtleties (InputT output type, float postprocessor intermediates, full categorical guard, per-kind shapes) are confirmed against source line numbers.
- Harness/capture: MEDIUM — the sparse-with-NaN capture construction (Open Q1) and concrete corpus model selection (Open Q2) are resolvable during Wave 0 but not yet executed.

**Research date:** 2026-06-10
**Valid until:** ~2026-07-10 (stable — the porting target is frozen treelite 4.7.0; only the harness/capture details are open, and those are internal decisions).

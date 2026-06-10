# Phase 6: cubecl GTIL Kernels (CPU Backend) - Research

**Researched:** 2026-06-10
**Domain:** cubecl `#[cube(launch)]` kernel authoring (tree traversal + postprocessors), CPU runtime, zero-copy ragged-SoA host→device upload, backend registration into the Phase-5 harness seam
**Confidence:** HIGH (cubecl authoring + harness seam + GTIL semantics all read from vendored sources); MEDIUM on two cubecl API surface details flagged in the Assumptions Log (the exact `exp2` intrinsic name and the `client.create_from_slice` method name).

## Summary

Phase 6 reimplements the GTIL hot path as `cubecl` `#[cube(launch)]` kernels generic over `R: Runtime`, defaulting to the cubecl **CPU** runtime, and registers it as `Backend::CubeclCpu` in the Phase-5 `RunnerCase` seam — validated to 1e-5 against the frozen scalar golden matrix. The work is genuinely a *registration plus a kernel crate*, not a harness refactor: `crates/treelite-harness/src/lib.rs` already declares the four-fn-pointer `RunnerCase` and reserves the `CubeclCpu` enum variant (lib.rs:59), and `tests/gtil_matrix.rs` already dispatches every cell through `scalar_cpu_case()` with zero hard-coded backend assumptions in its iteration. The matrix runner is reused verbatim.

The cubecl side is well-bounded by the vendored manual (`/home/user/Documents/workspace/cubecl_manual/`). cubecl **0.10.0** is the current published version (confirmed via `cargo search`), uses `cubecl::cpu::CpuRuntime` / `cubecl::cpu::CpuDevice` behind the `"cpu"` feature, supports `f64` on the CPU runtime, has the documented "no `continue`" control-flow constraint (use an `if`-wrapped tail) and the "helpers must be `#[cube]`" rule, and the CPU runtime is the standard deterministic reference backend the manual uses for every verification example. The tree descent maps to a **break-free bounded `while !is_leaf` loop** (one unit per row, trees looped serially), and the 10 postprocessors port directly to `#[cube]` helpers — but with two authoring gotchas that MUST be honored or the kernels will not compile: (1) call math intrinsics as **associated functions** `F::exp(x)`, never `x.exp()`; (2) never use `if` as a value-returning expression — initialize a `mut` and assign inside an `if` statement.

The zero-copy SoA upload (SC3/GPU-05) is a small **additive** `as_bytes()` accessor on the existing `TreeBuf<T>` (which is already a `T: Copy` POD enum with an `as_slice()`), feeding `bytemuck::cast_slice` → `cubecl::bytes::Bytes` → `client.create(...)`. The forest uploads as **one device handle per column** built by concatenating each column across all trees, with a parallel per-tree `(offset, len)` index — no per-tree handle explosion. Per D-04, this is confirmation, not a precision gamble: the user has validated cubecl reproduces scalar f64/mixed precision to 1e-5; the mandatory spike confirms the control-flow shape and one postprocessor's cast order, and is NOT a go/no-go gate.

**Primary recommendation:** Add a new `crates/treelite-cubecl` library crate (kernels + host launchers + a `cubecl_cpu_case()` constructor), depend on `cubecl = { version = "0.10.0", features = ["cpu"] }` and `bytemuck`, register `Backend::CubeclCpu` in the harness, run a minimal one-postprocessor + break-free-descent spike first, then port the dense numerical core across all 4 predict kinds with sparse/categorical routed to the existing scalar fallback (D-02). Use **separate kernels per predict kind** (default/raw fused-with-postproc, leaf_id, score_per_tree) rather than one mega-kernel.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Kernelize the numerical-dense core across all 4 predict kinds + leaf-vector broadcast.** cubecl kernels cover dense numerical-split traversal for `default`, `raw`, `leaf_id`, `score_per_tree`, including multiclass leaf-vector broadcast.
- **D-02: Sparse CSR and categorical splits ride the scalar fallback this phase.** Their cubecl port is explicitly deferred — NOT abandoned — and recorded as a Deferred Idea. Phase is `Mode: mvp`; this is the vertical slice.
- **D-03: Both traversal AND postprocessors run as `#[cube]` kernels, verbatim mixed-precision cast order.** softmax = f32 max-subtraction + f64 norm accumulate + f32 divide; `exponential_standard_ratio` uses `exp2` (base-2); the f64 sigmoid/hinge twins from 05-06 run in f64.
- **D-04: In-kernel precision is NOT treated as a fidelity risk.** The mandatory kernel spike is a **confirmation step, not a go/no-go gate expected to fail**. Plans MUST NOT pad with "if cubecl can't hit 1e-5, fall back to host postprocessors" contingency branches — that risk is retired. (Memory `cubecl-precision-validated`.)
- **D-05: f64 input + the `<f64,f64>` preset run in-kernel on CubeclCpu — no f64 scalar fallback.** All 4 input×preset combinations of the harness matrix execute through cubecl kernels on the CPU backend.
- **D-06: Per-cell provenance in the harness manifest.** Every harness cell records which path actually executed — `cubecl-kernel` vs `scalar-fallback` — extending the Phase-5 manifest `backend` field (D-09) to **per-cell granularity**.
- **Kernel shape (SC1):** one compute unit per row, looping over trees **serially** in `tree_id` order — no `atomicAdd`/reduce over the tree axis, no `continue`. (Matches GTIL-08 serial tree-sum; parallelism only across rows.)
- **SoA upload (SC3):** host→device via `TreeBuf::as_bytes()` + `client.create_from_slice`, with **per-column ragged-SoA concatenation across the forest** (one handle per column for the whole forest — no per-tree handle explosion). A plain-Rust fallback exists for any unimplemented cubecl op.
- **Determinism (SC2):** output bit-identical across two runs of the same input on the CPU backend.
- **Registration seam (05-05, D-11):** Phase 6 = add `Backend::CubeclCpu` variant + a `RunnerCase` constructor wiring the four fn-pointer slots through cubecl. The matrix iteration does NOT change — adding the backend is a registration, not a refactor.
- **Scalar reference is the measuring stick (D-11):** the Phase-5 plain-Rust scalar GTIL stays the reference/fallback every backend is measured against to 1e-5.

### Claude's Discretion
- **Kernel granularity** — single fused traversal+postproc kernel vs separate kernels per predict-kind; how leaf-vector broadcast maps into the unit-per-row shape. Derive from the cubecl manual + the spike.
- **Exact ragged-SoA concatenation layout** — per-column offset/length bookkeeping for the concatenated forest buffers, provided SC3's "one handle per column, no per-tree explosion" holds and upload stays zero-copy via `as_bytes()`.
- **Which concrete cubecl CPU runtime** backs `Backend::CubeclCpu` (subject to SC2 bit-identical determinism). Researcher confirms against the cubecl manual.
- **Spike scope** — the minimal kernel that confirms control-flow shape (no `continue`), f64 in-kernel, and a single postprocessor's cast order before the full port. (Confirmation per D-04, not a gate.)

### Deferred Ideas (OUT OF SCOPE)
- **Sparse CSR input in cubecl kernels** — deferred (rides scalar fallback per D-02). A later cubecl-coverage pass ports the NaN-materialized ragged CSR row path into kernels.
- **Categorical splits in cubecl kernels** — deferred likewise (D-02): integer-membership bitset + float-representability guard + child polarity; scalar fallback this phase, kernel port later.
- **GPU backends (CUDA/wgpu/ROCm)** — Phase 7 (GPU-03/04). Routes through the same generic `R: Runtime` seam this phase establishes.
- **f16/bf16 half-precision in-kernel fast path** — v2 PERF-v2-01 / Phase-9.
- **Input-buffer bytemuck zero-copy recast beyond the SoA model upload** — Phase 9 (MEM-01).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPU-01 | The GTIL inference hot path (traversal + postprocessors) is implemented as cubecl kernels | §"Architecture Patterns" (break-free descent kernel + postprocessor `#[cube]` helpers, derived from `crates/treelite-gtil/src/lib.rs::evaluate_tree`/`predict_preset` and `postprocessor.rs`); cubecl loop-control / conditionals / algebra manuals confirm the constructs exist. |
| GPU-02 | The cubecl CPU backend is the default and is validated to 1e-5 | §"Standard Stack" pins `cubecl 0.10.0` `cpu` feature → `cubecl::cpu::CpuRuntime`; §"Common Pitfalls" covers determinism (SC2); the frozen Phase-5 matrix (`tests/gtil_matrix.rs`) is the 1e-5 instrument, driven via the `RunnerCase` seam. |
| GPU-05 | SoA model buffers upload host→device zero-copy | §"Don't Hand-Roll" + §"Code Examples" (additive `TreeBuf::as_bytes()` → `bytemuck::cast_slice` → `cubecl::bytes::Bytes` → `client.create`, per-column ragged concatenation); `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `ZERO_COPY_ARROW_CUBECL.md`, `Backend-Agnostic_Buffer_Slicing...md`. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tree traversal (numerical dense) | cubecl kernel (`#[cube(launch)]`, CPU runtime) | scalar GTIL (fallback) | GPU-01 hot path; one unit per row, serial trees (SC1). |
| Postprocessors (10) | cubecl `#[cube]` helpers | scalar `postprocessor.rs` twins | D-03: postprocessors literally run as kernels; the scalar twins are the reference. |
| Sparse-CSR predict | scalar GTIL fallback | — | D-02: ragged NaN-materialized rows deferred; routed to `treelite_gtil::predict_sparse`. |
| Categorical-split traversal | scalar GTIL fallback | — | D-02: bitset + representability guard deferred; the cubecl numerical kernel must detect a categorical node and route the whole row to fallback, OR the model is fallback-only if any tree has a categorical split (see Open Q1). |
| SoA model→device upload | host launcher (new `treelite-cubecl` crate) | `TreeBuf::as_bytes()` in `treelite-core` | GPU-05 zero-copy; per-column concatenation + offset index. |
| Backend registration | `treelite-harness` (`Backend::CubeclCpu` + `cubecl_cpu_case()`) | — | D-11 registration-not-refactor; the matrix iteration is untouched. |
| Per-cell provenance | `treelite-harness` manifest (`backend` per cell) | capture script | D-06: `cubecl-kernel` vs `scalar-fallback` recorded as data. |
| 1e-5 measurement | frozen Phase-5 golden matrix (`fixtures/gtil/`) | — | D-11: scalar reference is the measuring stick; CubeclCpu asserts against identical goldens. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `cubecl` | `0.10.0` (features = `["cpu"]`) | `#[cube(launch)]` kernel authoring, `CpuRuntime`/`CpuDevice`, `ComputeClient`, `Array<F>`/`ArrayArg`, `cubecl::bytes::Bytes` | The project's mandated compute spine (CLAUDE.md: "GTIL inference hot path via cubecl"). Current published version `[VERIFIED: crates.io via cargo search → cubecl = "0.10.0"]`. The vendored manual cites `cubecl = { version = "0.10.0", features = ["cpu"] }` in every example `[CITED: cubecl_manual/.../Cubecl_loop_control.md, Cubecl_generics.md, Cubecl_conditionals.md]`. |
| `bytemuck` | `1` (latest 1.x; features = `["derive"]` if a `#[repr(C)]` POD struct is later introduced) | Zero-copy `cast_slice::<T,u8>` for the SoA upload and `cast_slice::<u8,T>` on read-back | The documented cubecl ingestion bridge `[CITED: ZERO_COPY_TRANSMUTATION_CUBECL.md, ZERO_COPY_ARROW_CUBECL.md]`. Already foreseen by `treelite-core` ("the `bytemuck::Pod` seam is deferred to Phase 9" — `tree_buf.rs:12`); Phase 6 only needs `cast_slice` on existing `Copy` columns, not a `Pod` derive on `TreeBuf` itself. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl-cpu` | `0.10.0` | The standalone CPU runtime crate, if the umbrella `cubecl` `"cpu"` feature is not preferred | Prefer the umbrella `cubecl` with `features=["cpu"]` (what every manual example uses); only depend on `cubecl-cpu` directly if a transitive-feature conflict appears. `[VERIFIED: crates.io via cargo search → cubecl-cpu = "0.10.0"]` |
| `anyhow` / `thiserror` | workspace-pinned (`1.0.102` / `2.0.18`) | Errors: `thiserror` in the new `treelite-cubecl` lib crate, `anyhow` at the harness boundary | CLAUDE.md error-handling contract; the four `RunnerCase` fn pointers return `anyhow::Result<Vec<f64>>` (lib.rs:65-73). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| cubecl `"cpu"` runtime as default | cubecl `"wgpu"` with a CPU adapter (lavapipe) | Rejected: the manual's deterministic reference backend is `CpuRuntime`; wgpu-on-CPU adds a driver layer and is not bit-identical-deterministic by contract. SC2 requires CPU-backend determinism. |
| Separate kernels per predict kind | One fused mega-kernel branching on kind | Rejected: kind is host-known (from `Config`), so branching belongs on the host (kernel selection), not in-kernel; separate kernels keep each kernel's control flow simple and avoid dead in-kernel branches. |
| New `treelite-cubecl` crate | Kernels inside `treelite-gtil` | Rejected: `treelite-gtil` is the plain-Rust reference/fallback (D-11) and must stay dependency-light and scalar; adding a heavy GPU dep there couples the reference to cubecl. A separate crate keeps the seam clean. |

**Installation (root `Cargo.toml` `[workspace.dependencies]`):**
```toml
cubecl = { version = "0.10.0", features = ["cpu"] }
bytemuck = { version = "1", features = ["derive"] }
```
Then in `crates/treelite-cubecl/Cargo.toml`: `cubecl = { workspace = true }`, `bytemuck = { workspace = true }`, `treelite-core = { path = ... }`, `treelite-gtil = { path = ... }` (for the fallback), `thiserror = { workspace = true }`.

**Version verification:** `cargo search cubecl` → `cubecl = "0.10.0"`; `cargo search cubecl-cpu` → `cubecl-cpu = "0.10.0"`. Both authoritatively cross-confirmed by the vendored cubecl manual (the project's canonical reference) which pins `0.10.0` throughout. The live `crates.io` HTTP API was unavailable from the sandbox; the `cargo search` registry index + the manual are the two corroborating sources.

## Package Legitimacy Audit

> The Phase Legitimacy Gate was run. NOTE the ecosystem-confusion finding below — it is the single most important entry in this table.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `cubecl` | **crates.io (Rust)** | est. ~2 yr (Tracel/Burn ecosystem) | high (Burn dependency) | github.com/tracel-ai/cubecl | **N/A — see note** | Approved (`[VERIFIED: crates.io via cargo search; CITED: vendored cubecl_manual]`) |
| `cubecl-cpu` | **crates.io (Rust)** | est. ~1 yr | moderate | github.com/tracel-ai/cubecl | **N/A — see note** | Approved (`[VERIFIED: crates.io via cargo search]`) |
| `bytemuck` | **crates.io (Rust)** | 6+ yr | very high (ecosystem-standard) | github.com/Lokathor/bytemuck | **N/A — see note** | Approved (`[VERIFIED: crates.io via cargo search]`) |

**slopcheck note (ecosystem-confusion vector — DO NOT misread):** `slopcheck install cubecl cubecl-cpu bytemuck` returned `[SLOP]` for ALL THREE with "does not exist on **pypi**". This is a **false positive caused by registry mismatch**: slopcheck 0.6.1 checks **PyPI**, but these are **Rust crates on crates.io**, not Python packages. This is exactly the documented cross-ecosystem hallucination vector (~9% rate) — *in reverse*: a tool checking the wrong registry. The correct registry verification is `cargo search`, which confirms all three exist at the stated versions. **Disposition: all three APPROVED.** The packages are NOT removed; the slopcheck verdict is discarded as a wrong-ecosystem artifact. The planner should still gate the first `cargo add` behind the spike (the spike compiles + runs a kernel, which is itself the strongest possible legitimacy proof).

**Packages removed due to slopcheck [SLOP] verdict:** none (the three [SLOP] verdicts are wrong-registry false positives, documented above).
**Packages flagged as suspicious [SUS]:** none.

## Architecture Patterns

### System Architecture Diagram

```
                    treelite-harness  tests/gtil_matrix.rs  (UNCHANGED — D-11)
                                 │  iterates frozen fixtures/gtil/*.golden.json
                                 │  dispatches each cell via a RunnerCase
                                 ▼
        ┌──────────────── cubecl_cpu_case() : RunnerCase ────────────────┐
        │  dense_f32  dense_f64  sparse_f32  sparse_f64  (4 fn pointers)  │
        └───────┬───────────┬──────────┬───────────┬─────────────────────┘
        dense f32/f64       │   sparse f32/f64 (D-02 → scalar fallback)
                │           │          └────────────► treelite_gtil::predict_sparse  (REFERENCE)
                ▼           ▼
   ┌─────────── treelite-cubecl  (NEW crate) ───────────────────────────┐
   │  host launcher:                                                     │
   │   1. has any tree a categorical split? ─yes─► scalar fallback (D-02)│
   │   2. upload SoA forest:                                             │
   │        per column → concat across trees → bytemuck::cast_slice      │
   │        → cubecl::bytes::Bytes → client.create  (ONE handle/column)  │
   │        + per-tree (offset,len) index buffers                        │
   │   3. upload row-major input matrix (f32 or f64) → client.create     │
   │   4. select kernel by Config.kind:                                  │
   │        default/raw → traversal+accumulate (+postproc for default)   │
   │        leaf_id     → traversal → write leaf node id                 │
   │        score_per_tree → traversal → write raw per-tree leaf data    │
   │   5. launch::<F, CpuRuntime>(CubeCount per-row-blocks, CubeDim)     │
   │   6. client.read → bytemuck::cast_slice → Vec<f32|f64> → widen f64  │
   └────────────────────────────────────────────────────────────────────┘
                │  #[cube(launch)] kernels generic over F: Float, R: Runtime
                ▼
   ┌─────────── kernel: one unit per row (ABSOLUTE_POS = row) ───────────┐
   │  if row < num_row {                                                 │
   │    for tree_id in 0..num_tree {            // SERIAL over trees(SC1) │
   │       nid = 0;                                                      │
   │       while cleft[base+nid] != -1 {        // break-free descent    │
   │          fi = split_index[base+nid];                                │
   │          fv = input[row*num_feat + fi];                             │
   │          // NaN → default child; else compare (no `continue`)       │
   │          ... assign next via if-STATEMENTS (no if-expr value) ...   │
   │          nid = next;                                                │
   │       }                                                             │
   │       accumulate leaf into output cell(s)  // leaf-vector broadcast │
   │    }                                                               │
   │    // base-score add + postprocessor (default kind) via #[cube] fns │
   │  }                                                                  │
   └────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/treelite-cubecl/          # NEW crate
├── Cargo.toml                    # cubecl 0.10.0 (cpu), bytemuck, treelite-core, treelite-gtil, thiserror
└── src/
    ├── lib.rs                    # cubecl_cpu_case()-style constructor exports + host predict entry points
    ├── upload.rs                 # SoA ragged concatenation + as_bytes() upload + offset index
    ├── kernels/
    │   ├── traversal.rs          # #[cube] break-free descent + next-node helpers
    │   ├── postproc.rs           # #[cube] ports of the 10 postprocessors (cast order verbatim)
    │   ├── default_raw.rs        # #[cube(launch)] traversal+accumulate(+postproc) kernel
    │   ├── leaf_id.rs            # #[cube(launch)] leaf-node-id kernel
    │   └── score_per_tree.rs     # #[cube(launch)] raw per-tree leaf-data kernel
    └── error.rs                  # thiserror CubeclError
crates/treelite-core/src/tree_buf.rs   # ADD `as_bytes()` (additive)
crates/treelite-harness/src/lib.rs     # ADD Backend::CubeclCpu + cubecl_cpu_case()
crates/treelite-harness/src/manifest.rs# per-cell `backend` already present; add provenance recording
```

### Pattern 1: Break-free data-dependent tree descent (no `continue`)
**What:** The scalar `evaluate_tree` (`crates/treelite-gtil/src/lib.rs:432-502`) is a `while !is_leaf(nid)` loop with early data-dependent routing. cubecl forbids `continue` but allows `while`/`for` + `break` `[CITED: Cubecl_loop_control.md]`. The descent is naturally break-free: it has no `continue` — every iteration computes the next node and assigns it. The only adaptation is the NaN/categorical branch must use `if`-**statements** assigning a `mut next`, never an `if`-expression returning a value.
**When to use:** Every traversal kernel.
**Example:**
```rust
// Source: derived from treelite-gtil/src/lib.rs:449-500 + Cubecl_loop_control.md + Cubecl_conditionals.md
#[cube]
fn descend<F: Float>(
    cleft: &Array<i32>, cright: &Array<i32>, split_index: &Array<i32>,
    threshold: &Array<F>, default_left: &Array<u32>, // bool-as-u32 (see Pitfall 4)
    base: u32, row_off: u32, input: &Array<F>,
) -> u32 {
    let mut nid: u32 = 0;
    // Bounded by tree depth; `break` is allowed, `continue` is NOT.
    while cleft[(base + nid) as usize] != -1 {
        let fi = split_index[(base + nid) as usize];
        let fv = input[(row_off + fi as u32) as usize];
        let mut next: i32 = cright[(base + nid) as usize]; // default = right
        // NaN routes to the default child (predict.cc:158-159).
        if F::is_nan(fv) {                                  // associated fn, NOT fv.is_nan()
            let mut dch = cright[(base + nid) as usize];
            if default_left[(base + nid) as usize] == 1u32 { dch = cleft[(base + nid) as usize]; }
            next = dch;
        } else {
            // XGBoost always kLT: fvalue < threshold ? left : right.
            if fv < threshold[(base + nid) as usize] { next = cleft[(base + nid) as usize]; }
        }
        nid = next as u32;
    }
    nid
}
```
**Note:** `descend` must be `#[cube]` (a helper called from the launch kernel) — a plain Rust fn would fail with E0433 `[CITED: cubecl_error_solution_guide/calling a "normal" Rust function...md]`.

### Pattern 2: One unit per row, serial trees (SC1)
**What:** `ABSOLUTE_POS` is the row index; the kernel loops `for tree_id in 0..num_tree` serially and accumulates into the row's output cell(s). No reduction/atomic over the tree axis (GTIL-08 forbids it; float add is non-associative).
**When to use:** default/raw/score_per_tree kernels.
**Example:**
```rust
// Source: Cubecl_multi_threading.md (ABSOLUTE_POS, bounds check) + predict_preset (lib.rs:643-741)
#[cube(launch)]
fn predict_default<F: Float>(/* SoA columns, offsets, input, output, scalars */) {
    let row = ABSOLUTE_POS;
    if row < num_row {                       // mandatory bounds check
        // ... serial tree loop, accumulate, base-score add, postproc ...
    }
}
```
Host launch: `CubeDim { x: 256, y: 1, z: 1 }`, `CubeCount::Static((num_row + 255)/256, 1, 1)` (the manual's ceiling-division idiom `[CITED: Cubecl_multi_threading.md]`).

### Pattern 3: Per-column ragged-SoA forest upload (SC3 / GPU-05)
**What:** For each Tree column (`cleft`, `cright`, `split_index`, `threshold`, `leaf_value`, `default_left`, `leaf_vector`, the leaf-vector CSR offsets, …) concatenate that column across every tree into one host `Vec`, `bytemuck::cast_slice` it to `&[u8]`, wrap in `cubecl::bytes::Bytes`, and `client.create` → **one device handle per column for the whole forest**. A parallel `tree_node_offset[tree_id]` (prefix sum of `num_nodes`) and `tree_leafvec_offset[tree_id]` index let the kernel address tree `t`'s node `n` at `concat[tree_node_offset[t] + n]`.
**When to use:** model upload, once per `predict` call (or cache per model — see Open Q2).
**Anti-pattern avoided:** one handle per tree (handle explosion) — explicitly forbidden by SC3.

### Anti-Patterns to Avoid
- **Calling `x.exp()` / `x.sqrt()` / `x.is_nan()` as methods inside `#[cube]`:** fails with `no method named __expand_exp_method` (E0599). Use associated functions `F::exp(x)`, `F::sqrt(x)`, `F::is_nan(x)` `[CITED: cubecl_error_solution_guide/mismatched types.md:113-235]`.
- **`let v = if cond { a } else { b };` inside `#[cube]`:** fails with `ExpandElementTyped vs {float}` (E0308). Use `let mut v = default; if cond { v = a; }` `[CITED: same file:9-109]`.
- **`continue` in a kernel loop:** compile error in cubecl 0.10.0. Wrap the loop tail in `if` `[CITED: Cubecl_loop_control.md:42-65]`.
- **Calling a plain Rust helper from a kernel:** E0433. Mark helpers `#[cube]` or inline them `[CITED: cubecl_error_solution_guide/calling a "normal" Rust function...md]`.
- **`usize`/`u64` inside `#[cube]`:** prefer `u32`/`i32` for indices/counters; cast to `usize` only at the array-index site (`arr[idx as usize]`) `[CITED: Batch-Tree_Reorganization_Algorithm.md:142-144, calling-normal-rust...md:269-276]`.
- **Reducing/atomic over the tree axis:** violates SC1/GTIL-08 and breaks 1e-5 (non-associative float sum). Loop trees serially per row.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Host→device byte conversion | A manual `unsafe` transmute of column `Vec<T>` to `&[u8]` | `bytemuck::cast_slice` | Validates alignment/size, no UB; the documented cubecl bridge `[CITED: ZERO_COPY_TRANSMUTATION_CUBECL.md]`. |
| Device buffer wrapper | A custom byte container | `cubecl::bytes::Bytes::from_elems(vec)` / `Bytes::from_bytes_vec(bytes.to_vec())` + `client.create(...)` | The runtime's native ingestion type `[CITED: Cubecl_generics.md:79, ZERO_COPY_TRANSMUTATION_CUBECL.md:96]`. |
| Read-back typing | Manual byte slicing | `bytemuck::cast_slice::<u8, F>(&client.read_one(handle))` | O(1), validated `[CITED: ZERO_COPY_TRANSMUTATION_CUBECL.md:111-114]`. |
| Sparse/categorical traversal | A cubecl ragged-CSR / bitset kernel | `treelite_gtil::predict_sparse` / scalar fallback | D-02: deferred. The fallback already exists, green, and is the reference. |
| The 1e-5 instrument | A new cubecl test harness | The frozen `tests/gtil_matrix.rs` + `RunnerCase` | D-11: registration-not-refactor; the matrix iteration is reused unchanged. |
| Postprocessor math | A "simplified" all-f32 or all-f64 rewrite | Verbatim cast-order ports of `postprocessor.rs` (incl. the `*_f64` twins, softmax's f32-max/f64-norm split) | The cast order IS the 1e-5 contract (CR-01). |

**Key insight:** Almost everything novel in this phase is *plumbing* (upload + launch + read-back) plus a *faithful re-expression* of code that already exists and is green in `treelite-gtil`. The kernels must reproduce, not reinvent, `evaluate_tree`/`predict_preset`/the 10 postprocessors. Treat `crates/treelite-gtil/src/lib.rs` and `postprocessor.rs` as the line-by-line spec.

## Common Pitfalls

### Pitfall 1: cubecl math intrinsics are associated functions, not methods
**What goes wrong:** Porting `(-alpha * v).exp()` or `fv.is_nan()` verbatim into `#[cube]` fails to compile (`__expand_exp_method` E0599).
**Why it happens:** cubecl lowers `f64::exp(x)` / `Exp::exp(x)`, not the method form.
**How to avoid:** Write `F::exp(x)`, `F::sqrt(x)`, `F::is_nan(x)`, `F::abs(x)`, `F::powf(b, e)`. Import traits via `use cubecl::prelude::*;`.
**Warning signs:** Any `.exp()`/`.ln()`/`.is_nan()`/`.copysign()` call inside a `#[cube]` block.

### Pitfall 2: `exp2` (base-2) for `exponential_standard_ratio` — verify the intrinsic
**What goes wrong:** `exponential_standard_ratio` uses `std::exp2` (base-2), NOT `exp` (`postprocessor.rs:124`, `predict.cc`). The algebra manual lists `exp`, `ln`, `powf`, `sin`, etc. but does **not** list `exp2` `[CITED: Cubecl_algebra.md:24-38]`.
**Why it happens:** cubecl's `Float` may not expose `exp2` directly.
**How to avoid:** First try `F::exp2(x)`. If it does not exist in 0.10.0, use the exact algebraic identity `exp2(x) == exp(x * ln(2))` → `F::exp(x * F::new(core::f64::consts::LN_2 as f32))` **computed in the element's own width**, or `F::powf(F::new(2.0), x)`. The spike (D-04) MUST exercise `exponential_standard_ratio` specifically to lock this down. Document the chosen form; verify 1e-5 against the `exponential_standard_ratio`/`_f64` scalar twins. (Flagged in Assumptions Log A1.)
**Warning signs:** A 1e-5 miss localized to IsolationForest / `survival:aft` fixtures (the exp2 consumers).

### Pitfall 3: Softmax mixed-precision cast order in-kernel
**What goes wrong:** Collapsing softmax to all-f32 or all-f64 shifts ULPs past 1e-5 on large margins (this was CR-01).
**Why it happens:** Upstream `softmax<InputT>` keeps `max_margin` and `t` and the final divisor in **f32** for EVERY `InputT`, but for the `double` instantiation the row cells stay **f64** (`postprocessor.rs:156-253`). The subtraction is `f64 - f32 -> f64`, `exp` in f64, narrow result to f32 `t`, accumulate `norm_const` in f64, divide `f64 /= (f32 cast back to f64)`.
**How to avoid:** Mirror `softmax`/`softmax_f64` exactly. In the f64 kernel keep cells as `f64`, but introduce explicit `f32`-typed locals for `max_margin`, `t`, `divisor` and cast at the exact sites the scalar twin does. Note cubecl `f32`/`f64` mixed arithmetic in one kernel — confirm the spike can hold both (Assumptions Log A2). Per D-04 the user has validated this; the spike confirms.
**Warning signs:** `leaf_vec_mc` / softprob fixtures drifting at ~1e-8–1e-6 between the cubecl and scalar paths.

### Pitfall 4: `bool` columns are not a cubecl `Array` element
**What goes wrong:** `default_left` and `category_list_right_child` are `TreeBuf<bool>` (`tree.rs:28,36`). cubecl `Array<F>` wants `Numeric`/`Float`/`CubeElement` elements; `bool` is not a natural array element for upload.
**Why it happens:** Bytemuck/cubecl operate on POD numeric scalars.
**How to avoid:** On the host, materialize `default_left` as a `Vec<u32>` (or `i32`) of 0/1 during upload (it stays zero-copy in spirit — one small contiguous numeric column). Compare `default_left[i] == 1u32` in-kernel. Same for any bool routing column the numerical kernel needs. `TreeNodeType` (an enum) similarly uploads as a small `i32`/`u32` discriminant column so the kernel can detect `kCategoricalTestNode` and route to fallback.
**Warning signs:** A compile error on `Array<bool>` or a `CubeElement` trait bound failure.

### Pitfall 5: Determinism check (SC2) — read-back stability, not re-launch nondeterminism
**What goes wrong:** Asserting bit-identical output across two runs could spuriously fail if the CPU runtime parallelizes the per-row units with a non-deterministic *reduction* — but per SC1 there is NO reduction (each row writes its own disjoint output cells). So determinism holds structurally as long as no atomic/reduce is introduced.
**Why it happens:** The risk is only self-inflicted (if someone adds a tree-axis atomic).
**How to avoid:** Keep the kernel write pattern disjoint-per-row. Add a test that runs the same input twice and asserts `out_a.to_bits() == out_b.to_bits()` element-wise (the SC2 gate). The CPU runtime is the manual's deterministic reference backend.
**Warning signs:** Any introduction of `Atomic*` or `plane_*`/`sync_cube` reductions over trees.

### Pitfall 6: Output element type follows INPUT dtype, not the preset
**What goes wrong:** Producing `Vec<f32>` for an f64-input cell (or vice-versa) breaks the harness's no-pre-cast discipline (`gtil_matrix.rs:322-381`).
**Why it happens:** The `PredictOut` element `O` equals the *input* element type, independent of the model preset (`lib.rs:87-95`). All four input×preset combos are valid (D-05).
**How to avoid:** Launch the f32-input kernel with `F = f32` and the f64-input kernel with `F = f64`. The kernel reads `threshold[T]`/`leaf_value[T]` in the *preset's* `T` and the input in `F` — so the kernel is generic over BOTH the input element and the leaf/threshold element (two type params, mirroring `predict_preset<T, O>`). The `dense_f32` fn pointer always returns `Vec<f64>` only by widening *after* the f32 kernel runs (exactly what `scalar_cpu_case` does at lib.rs:114-117).

## Runtime State Inventory

> Phase 6 is additive new code + one additive accessor; it is NOT a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a renamed string.
>
> - **Stored data:** None — no datastore keys change. The frozen `fixtures/gtil/` goldens are reused verbatim (CubeclCpu asserts against the identical committed vectors).
> - **Live service config:** None.
> - **OS-registered state:** None.
> - **Secrets/env vars:** None new (no `CUBECL_*` env required for the CPU runtime; GPU device-selection env is Phase 7).
> - **Build artifacts:** Adding `crates/treelite-cubecl` to `[workspace.members]` and the new `[workspace.dependencies]` entries will trigger a fresh `Cargo.lock` resolution pulling the cubecl dependency tree — expected and correct, not stale state.

## Code Examples

### Additive `TreeBuf::as_bytes()` (SC3 enabler — `treelite-core/src/tree_buf.rs`)
```rust
// Source: tree_buf.rs already exposes as_slice() over a `T: Copy` POD buffer.
// Add an additive byte-view; bytemuck::cast_slice requires T: Pod (true for the
// numeric columns we upload). For the f32/f64/i32/u32/u64 columns this is exact.
impl<T: Copy + bytemuck::Pod> TreeBuf<T> {
    /// Zero-copy byte view of the column for the cubecl SoA upload (GPU-05/SC3).
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
}
```
Note: keep this behind the existing `T: Copy` API; only the numeric columns we actually upload need `Pod`. `bool`/enum columns are converted to a numeric column on the host (Pitfall 4), so they don't need `as_bytes`.

### Per-column ragged concatenation + upload (`treelite-cubecl/src/upload.rs`)
```rust
// Source: ZERO_COPY_TRANSMUTATION_CUBECL.md (cast_slice → Bytes → client.create) +
//         Batch-Tree_Reorganization_Algorithm.md (offset arithmetic).
// One handle per column for the whole forest; offsets address tree t's node n.
fn upload_threshold<F: bytemuck::Pod>(client: &ComputeClient<R>, trees: &[Tree<F>]) -> (Handle, Vec<u32>) {
    let mut concat: Vec<F> = Vec::new();
    let mut node_off: Vec<u32> = Vec::with_capacity(trees.len() + 1);
    node_off.push(0);
    for t in trees {
        concat.extend_from_slice(t.threshold.as_slice());
        node_off.push(concat.len() as u32);          // prefix sum of num_nodes
    }
    let bytes = bytemuck::cast_slice::<F, u8>(&concat).to_vec();
    let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(bytes));
    (handle, node_off)
}
```
(Repeat per column: `cleft`/`cright`/`split_index` as `i32`, `threshold`/`leaf_value` as `F`, `default_left` as `u32`, `node_type` as `i32`, plus the `leaf_vector` value column + its CSR `begin/end` offset columns for the multiclass broadcast.)

### `cubecl_cpu_case()` registration (`treelite-harness/src/lib.rs`)
```rust
// Source: mirrors scalar_cpu_case() (lib.rs:107-134); adds the Backend::CubeclCpu variant.
pub fn cubecl_cpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::CubeclCpu,
        dense_f32: |model, data, num_row, cfg| {
            let out = treelite_cubecl::predict_cpu::<f32>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;     // kernel path (D-01)
            Ok(out.into_iter().map(|v| v as f64).collect()) // widen AFTER predict (no pre-cast)
        },
        dense_f64: |model, data, num_row, cfg| {
            treelite_cubecl::predict_cpu::<f64>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))
        },
        // D-02: sparse rides the scalar fallback this phase (recorded as
        // `scalar-fallback` provenance, D-06).
        sparse_f32: |model, csr, num_row, cfg| {
            let out = treelite_gtil::predict_sparse::<f32>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        sparse_f64: |model, csr, num_row, cfg| {
            treelite_gtil::predict_sparse::<f64>(model, csr, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))
        },
    }
}
```

### Postprocessor `#[cube]` port (cast-order verbatim — `treelite-cubecl/src/kernels/postproc.rs`)
```rust
// Source: postprocessor.rs:74-93 (sigmoid / sigmoid_f64). Associated-fn exp.
#[cube]
fn sigmoid<F: Float>(sigmoid_alpha: F, v: F) -> F {
    // 1 / (1 + exp(-alpha * v)); F::exp NOT v.exp() (Pitfall 1).
    F::new(1.0) / (F::new(1.0) + F::exp(F::new(0.0) - sigmoid_alpha * v))
}
```
For f64-input, launch with `F = f64` so `F::exp` is the double `exp` (D-05). `sigmoid_alpha`/`ratio_c` are `f32` model fields cast into `F` at the call site (mirrors `sigmoid_f64`'s `sigmoid_alpha as f64`).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-tree GPU buffers (AoS handle-per-tree) | Per-column ragged-SoA concat, one handle/column | this phase (SC3) | No handle explosion; coalesced reads. |
| Method-style kernel math (`x.exp()`) | Associated-fn math (`F::exp(x)`) | cubecl IR design | Required for compilation. |
| `if`-expression values in kernels | `mut` + `if`-statement assignment | cubecl 0.10.0 limitation | Required; affects every routing branch. |

**Deprecated/outdated:** None relevant — cubecl 0.10.0 is current.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | cubecl `Float` exposes `exp2`; if not, `exp(x*ln2)` or `powf(2,x)` reproduces base-2 to 1e-5 | Pitfall 2, D-03 | `exponential_standard_ratio` cells miss 1e-5 on IsolationForest/aft fixtures. **Mitigation:** the spike MUST exercise this postprocessor specifically. LOW residual risk (algebraic identity is exact-enough; user has validated cubecl precision per D-04). |
| A2 | Mixed f32/f64 locals in one f64-input `#[cube]` kernel (softmax's f32 `max_margin`/`t`/divisor with f64 cells) compile and run on CpuRuntime | Pitfall 3, D-03 | softmax<double> cast order can't be expressed in-kernel → would need a host-side softmax for f64 (NOT a fidelity risk per D-04, but a structural one for "postprocessors run as kernels", SC1). **Mitigation:** spike a softmax_f64 micro-kernel. The generics manual confirms f64 on CpuRuntime; mixed-width locals are the open detail. |
| A3 | `client.create(Bytes)` is the upload entry the CONTEXT's `client.create_from_slice` refers to | SC3 wording, Code Examples | Method name mismatch only; the manual uniformly shows `client.create(cubecl::bytes::Bytes::...)` and `client.read`/`read_one`. `create_from_slice` may be a thin convenience over the same path; the planner should grep the installed cubecl 0.10.0 API at spike time. LOW risk (cosmetic). |
| A4 | The CPU runtime's `ComputeClient` type path is `cubecl::cpu::{CpuRuntime, CpuDevice}` and `R::client(&device)` | Standard Stack | Wrong import path → trivial compile fix. Confirmed across 5 manual examples. VERY LOW risk. |

**These four assumptions are exactly what the mandatory spike (D-04 / Open Q3) retires.** None is a precision risk — they are cubecl API-surface details. Do NOT design 1e-5 fallback contingencies around them (D-04).

## Open Questions (RESOLVED)

> All three questions are resolved — the recommendation below each is adopted by the Phase-6 plans (Q1 → 06-04, Q2 → 06-04, Q3 → 06-02). No open decision reaches execution.

1. **Categorical-split detection granularity (route-to-fallback boundary).**
   - What we know: D-02 sends categorical splits to the scalar fallback. Models are tagged per-tree via `Tree::has_categorical_split` (`tree.rs:70`) and per-node via `node_type == kCategoricalTestNode`.
   - What's unclear: whether the cubecl path should fall back for the *whole model* if ANY tree has a categorical split (simplest, coarsest), or handle numerical trees in-kernel and route only categorical-node rows to fallback (finer, more complex).
   - **RESOLVED:** **whole-model fallback if `any tree.has_categorical_split`** for this MVP slice — simplest, keeps the kernel purely numerical, and the per-cell provenance (D-06) records `scalar-fallback` honestly. The numerical-dense matrix cells (the dominant path, D-01) still exercise real kernels. Finer routing is a later cubecl-coverage pass. Adopted by plan 06-04 (the `has_categorical_split` gate in `predict_cpu`).

2. **Model upload caching vs per-call upload.**
   - What we know: the harness calls `predict` per cell; uploading the forest every call is correct but redundant across cells sharing a model.
   - What's unclear: whether to cache device handles keyed by model identity.
   - **RESOLVED:** **upload per `predict` call** for the MVP (correctness first; the matrix is small). Note caching as a Phase-9/perf follow-up — it does not affect 1e-5 or SC1/SC2/SC3. Adopted by plan 06-04 (per-call upload in `predict_cpu`).

3. **Spike scope (D-04 confirmation — NOT a gate).**
   - What we know: the spike must confirm (a) the break-free / no-`continue` descent compiles & runs, (b) f64 in-kernel arithmetic on CpuRuntime, (c) one postprocessor's exact cast order (recommend `exponential_standard_ratio` to also retire A1, plus a `softmax_f64` micro-probe to retire A2).
   - What's unclear: nothing blocking — this is confirmation.
   - **RESOLVED:** a single `crates/treelite-cubecl` test that uploads a 2-tree numerical forest + a tiny input matrix, runs the default kernel, and asserts within 1e-5 of `treelite_gtil::predict` on the same model — plus a standalone `softmax_f64`/`exponential_standard_ratio` micro-kernel asserting against the scalar twins. Small, time-boxed; per D-04 it is expected to pass. Adopted by plan 06-02 (the mandatory confirmation spike).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cubecl` crate (cpu) | GPU-01/02/05 kernels | ✓ (crates.io) | 0.10.0 | none needed |
| `cubecl-cpu` runtime | CPU backend default | ✓ (via `cubecl` `cpu` feature) | 0.10.0 | direct `cubecl-cpu` dep |
| `bytemuck` | Zero-copy upload | ✓ (crates.io, ecosystem-standard) | 1.x | none needed |
| Rust stable, edition 2024 | Whole workspace | ✓ | per project toolchain | none |
| GPU device (CUDA/ROCm/wgpu) | — | N/A this phase | — | CPU runtime is the target (GPU is Phase 7) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — the CPU runtime requires no GPU hardware (it is a software runtime), so the developer's ROCm-only box (memory `gpu-hardware-rocm-only`) is irrelevant to Phase 6; GPU validation is Phase 7.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `approx::assert_abs_diff_eq!` (workspace dep `approx 0.5.1`) |
| Config file | none — `cargo test` |
| Quick run command | `cargo test -p treelite-cubecl` (kernel unit + spike tests) |
| Full suite command | `cargo test --workspace` |
| Phase gate | `cargo test --workspace` green with `Backend::CubeclCpu` asserting the frozen `fixtures/gtil/` matrix to 1e-5 |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GPU-01 | Traversal+postproc run as `#[cube(launch)]` kernels; one unit/row, serial trees, no continue | unit + integration | `cargo test -p treelite-cubecl` | ❌ Wave 0 (new crate) |
| GPU-02 | CubeclCpu default; frozen matrix passes 1e-5; output bit-identical across 2 runs (SC2) | integration | `cargo test -p treelite-harness --test gtil_matrix_cubecl` | ❌ Wave 0 (new test registering `cubecl_cpu_case()`) |
| GPU-02 (determinism) | `predict` twice → element-wise `.to_bits()` equal | unit | `cargo test -p treelite-cubecl determinism` | ❌ Wave 0 |
| GPU-05 | SoA per-column upload via `as_bytes()`; one handle/column; round-trips a forest | unit | `cargo test -p treelite-cubecl upload` | ❌ Wave 0 |
| D-03 | Each postprocessor `#[cube]` port matches its scalar twin to 1e-5 (incl. softmax cast order, exp2) | unit | `cargo test -p treelite-cubecl postproc` | ❌ Wave 0 |
| D-06 | Per-cell manifest records `cubecl-kernel` vs `scalar-fallback` | integration | the matrix test asserts provenance per cell | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-cubecl` (fast kernel + spike tests).
- **Per wave merge:** `cargo test --workspace` (no regression to the scalar matrix or the loader/serializer gates).
- **Phase gate:** Full suite green; the cubecl matrix run asserts the SAME frozen goldens scalar-cpu passes, to 1e-5, plus the SC2 determinism check.

### Wave 0 Gaps
- [ ] `crates/treelite-cubecl/` crate skeleton + `Cargo.toml` (cubecl 0.10.0 cpu, bytemuck) — register in `[workspace.members]`.
- [ ] Spike test (descend + f64 + one postprocessor) — the D-04 confirmation, RED until kernels exist.
- [ ] `tests/gtil_matrix_cubecl.rs` (or parameterize the existing runner over a `RunnerCase` list) registering `cubecl_cpu_case()` — RED until the kernels + registration land.
- [ ] Determinism test (SC2) — two-run bit-identity.
- [ ] `TreeBuf::as_bytes()` additive accessor — RED until added.
- [ ] Decision on whether to parameterize `tests/gtil_matrix.rs` over a backend list vs add a sibling test file (D-11: prefer a sibling test file or a `cases: &[RunnerCase]` loop that does NOT reshape the per-cell iteration; if the existing `gtil_matrix()` body would need restructuring, that is a smell — add a thin sibling instead).

## Security Domain

> `security_enforcement` is not set in `.planning/config.json` (workspace has no auth/network/crypto surface — it is a numerical library). The relevant "security" posture here is **memory safety + no-panic at boundaries**, already enforced by the project's `thiserror`/bounds-checked discipline.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (malformed `Model` / input shapes) | The kernel host path MUST validate `num_feature`/`num_row`/buffer lengths up front (as `treelite_gtil::predict` does at lib.rs:892-930) before any `client.create`/launch — never an OOB device write. `unsafe { ArrayArg::from_raw_parts(...) }` is the one `unsafe` site; its lengths must be the validated values. |
| V6 Cryptography | no | — |

### Known Threat Patterns for {Rust + cubecl numerical kernels}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OOB device read/write from a malformed model (bad `split_index`, child id, leaf-vector offset) | Tampering / DoS | Validate on the host before launch; the kernel's `if row < num_row` bound + pre-validated offset columns keep indices in range. Mirror the scalar path's typed-error guards (`FeatureIndexOutOfBounds`, `NodeIndexOutOfBounds`). Categorical/sparse already route to the checked scalar fallback (D-02). |
| `unsafe` transmute UB in upload | Tampering | Use `bytemuck::cast_slice` (validates alignment/size), never a hand-rolled transmute. |
| Non-determinism masquerading as a 1e-5 pass | Repudiation | SC2 two-run bit-identity test; disjoint-per-row writes (no tree-axis atomics). |

## Sources

### Primary (HIGH confidence)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` + `Cubecl_loop_control.md`, `Cubecl_multi_threading.md`, `Cubecl_conditionals.md`, `Cubecl_algebra.md`, `Cubecl_basic_operations.md`, `Cubecl_generics.md` — kernel authoring, `ABSOLUTE_POS`/`CubeCount`/`CubeDim`, no-`continue`, `if`-statement rule, `Float` f64 on CpuRuntime, `client.create`/`read`/`empty`, `launch::<F, R>`.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/mismatched types.md` + `calling a "normal" Rust function...md` — associated-fn math intrinsics, `if`-expr E0308, helpers-must-be-`#[cube]` E0433, `u64`-avoidance.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/Backend-Agnostic_Buffer_Slicing...md`, `Batch-Tree_Reorganization_Algorithm.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `ZERO_COPY_ARROW_CUBECL.md` — per-column/offset SoA layout, `bytemuck::cast_slice → Bytes → client.create` zero-copy upload.
- In-repo: `crates/treelite-gtil/src/lib.rs` (evaluate_tree :432, predict_preset :643, next_node :333, PredictScalar/PredictOut, output routing), `crates/treelite-gtil/src/postprocessor.rs` (10 postprocessors + `*_f64` twins, exact cast order), `crates/treelite-core/src/tree.rs`/`tree_buf.rs`/`model.rs` (SoA columns, `TreeBuf` POD enum, `ModelVariant`), `crates/treelite-harness/src/lib.rs` (Backend enum :54, RunnerCase :90, scalar_cpu_case :107), `crates/treelite-harness/src/manifest.rs` (backend field :60), `crates/treelite-harness/tests/gtil_matrix.rs` (the reused matrix runner), `treelite-mainline/src/gtil/output_shape.cc` (GetOutputShape per kind).
- `cargo search cubecl` / `cargo search cubecl-cpu` — version confirmation (0.10.0).

### Secondary (MEDIUM confidence)
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` — dtype/feature-gating context (f16/bf16 are Phase-9, but confirms cubecl 0.10.0 dtype handling + `client.properties().features` API).

### Tertiary (LOW confidence)
- slopcheck 0.6.1 PyPI check on `cubecl`/`bytemuck` — returned [SLOP] but is a **wrong-registry false positive** (these are crates.io packages); discarded in favor of `cargo search`.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — cubecl 0.10.0 cross-confirmed by `cargo search` + the vendored manual; `cpu` feature + `CpuRuntime` are the manual's universal reference backend.
- Architecture: HIGH — the kernel shape is a direct re-expression of the green scalar `evaluate_tree`/`predict_preset`; the harness seam already reserves `Backend::CubeclCpu`.
- Pitfalls: HIGH on the authoring gotchas (associated-fn math, no-`continue`, `if`-statement, bool-as-u32) — all cited from the error guides; MEDIUM on the two API-surface details (A1 `exp2`, A2 mixed-width locals) which the spike retires.
- Zero-copy upload: HIGH — three independent vendored zero-copy manuals + the existing `TreeBuf::as_slice` make `as_bytes()` a one-line additive.

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (cubecl is fast-moving — re-verify the published version and the `exp2`/`create_from_slice` API surface if the phase slips past ~30 days).

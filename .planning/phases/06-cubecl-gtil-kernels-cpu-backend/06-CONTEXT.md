# Phase 6: cubecl GTIL Kernels (CPU Backend) - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Reimplement the GTIL hot path (tree traversal + postprocessors) as `cubecl` `#[cube(launch)]` kernels generic over `R: Runtime`, with the **cubecl CPU backend as the deterministic default**, validated to 1e-5 against the green plain-Rust scalar reference (Phase 5). This widens the proven compute spine onto cubecl by registering `Backend::CubeclCpu` into the existing Phase-5 `RunnerCase` seam — it does NOT touch loaders, builder, or serialization (those stay plain idiomatic Rust per the project Out-of-Scope boundary).

**Requirements covered:** GPU-01 (GTIL hot path as cubecl kernels), GPU-02 (cubecl CPU backend is default, validated to 1e-5), GPU-05 (SoA model buffers upload host→device zero-copy).

In scope (HOW, not WHETHER):
- Kernelizing **numerical-split, dense-input** traversal across all 4 predict kinds (`default`/`raw`/`leaf_id`/`score_per_tree`) + leaf-vector (multiclass) broadcast.
- Porting both traversal AND the postprocessor set as `#[cube]` kernels with verbatim mixed-precision cast order.
- f32 AND f64 input dtypes + both `<f32,f32>`/`<f64,f64>` presets running in-kernel.
- Zero-copy ragged-SoA host→device upload of the forest.
- Per-cell provenance recording (kernel vs fallback) in the harness manifest.
- A mandatory minimal-kernel spike before the full port (research flag already set).

Out of scope (this phase):
- **Sparse CSR input** and **categorical splits** in cubecl — they ride the plain-Rust scalar fallback this phase (the two raggedest data-dependent shapes; deferred to a later cubecl-coverage pass).
- Any **GPU backend** (CUDA/wgpu/ROCm) — Phase 7 (GPU-03/04).
- PyO3 binding / numpy marshalling — Phase 8.
- Memory-efficiency hardening (bytemuck recast of input buffers beyond the SoA upload, smallvec/compact_str, custom allocator) — Phase 9.

</domain>

<decisions>
## Implementation Decisions

### MVP kernel slice scope (GPU-01)
- **D-01: Kernelize the numerical-dense core across all 4 predict kinds + leaf-vector broadcast.** cubecl kernels cover dense numerical-split traversal for `default`, `raw`, `leaf_id`, `score_per_tree`, including multiclass leaf-vector broadcast. This proves cubecl on the dominant inference path without taking on the raggedest shapes first.
- **D-02: Sparse CSR and categorical splits ride the scalar fallback this phase.** They are the two most data-dependent / ragged shapes (NaN-materialized CSR rows; integer-membership categorical bitset with float-representability guard). Their cubecl port is explicitly deferred — NOT abandoned — and is recorded as a Deferred Idea for a later cubecl-coverage pass. The phase is `Mode: mvp`; this is the vertical slice.

### Postprocessor execution location (GPU-01, the 1e-5 spine)
- **D-03: Both traversal AND postprocessors run as `#[cube]` kernels, verbatim mixed-precision cast order.** Honors Success Criterion 1 literally (postprocessors listed as kernels). The exact cast order is preserved in-kernel: softmax = f32 max-subtraction + f64 norm accumulate + f32 divide; `exponential_standard_ratio` uses `exp2` (base-2); the f64 sigmoid/hinge twins from 05-06 run in f64.
- **D-04: In-kernel precision is NOT treated as a fidelity risk.** The user has repeatedly tested that cubecl reproduces scalar precision in-kernel, including f64 and mixed-precision arithmetic. Therefore the mandatory kernel spike is a **confirmation step, not a go/no-go gate expected to fail**. Plans MUST NOT pad with elaborate "if cubecl can't hit 1e-5, fall back to host postprocessors" contingency branches — that risk is considered retired. (See memory `cubecl-precision-validated`.)

### f64 path under cubecl (GPU-02)
- **D-05: f64 input + the `<f64,f64>` preset run in-kernel on CubeclCpu — no f64 scalar fallback.** All 4 input×preset combinations of the harness matrix (f32/f64 input × `<f32,f32>`/`<f64,f64>` preset) execute through cubecl kernels on the CPU backend. Justified by D-04 (the user has tested f64 in cubecl directly). This keeps the Phase-8 PyO3 zero-copy numpy `float64` path on real kernels.

### Fallback honesty / auditability (GPU-02 claim integrity)
- **D-06: Per-cell provenance in the harness manifest.** Every harness cell records which path actually executed — `cubecl-kernel` vs `scalar-fallback` — extending the Phase-5 manifest `backend` field (D-09 from 05-CONTEXT) to **per-cell granularity**. This makes SC2 ("full equivalence harness passes within 1e-5 on cubecl-cpu") auditable cell-by-cell: the equivalence report shows exactly what cubecl executed vs what fell back (sparse/categorical this phase). The CubeclCpu backend is allowed to mix kernel + fallback paths internally, but the mix is never hidden — it is recorded as data.

### Locked upstream by Success Criteria + Phase 5 (carried forward, NOT re-litigated)
- **Kernel shape (SC1):** one compute unit per row, looping over trees **serially** in `tree_id` order — no `atomicAdd`/reduce over the tree axis, no `continue`. (Matches GTIL-08 serial tree-sum; parallelism only across rows.)
- **SoA upload (SC3):** host→device via `TreeBuf::as_bytes()` + `client.create_from_slice`, with **per-column ragged-SoA concatenation across the forest** (one handle per column for the whole forest — no per-tree handle explosion). A plain-Rust fallback exists for any unimplemented cubecl op (the mechanism D-02/D-06 rely on).
- **Determinism (SC2):** output bit-identical across two runs of the same input on the CPU backend.
- **Registration seam (05-05, D-11):** Phase 6 = add `Backend::CubeclCpu` variant + a `RunnerCase` constructor wiring the four fn-pointer slots (dense/sparse × f32/f64) through cubecl. The matrix iteration in the harness does NOT change — adding the backend is a registration, not a refactor.
- **Scalar reference is the measuring stick (D-11):** the Phase-5 plain-Rust scalar GTIL stays the reference/fallback every backend is measured against to 1e-5.

### Claude's Discretion (for research/planner)
- **Kernel granularity** — single fused traversal+postproc kernel vs separate kernels per predict-kind; how leaf-vector broadcast maps into the unit-per-row shape. Derive from the cubecl manual + the spike.
- **Exact ragged-SoA concatenation layout** — per-column offset/length bookkeeping for the concatenated forest buffers, provided SC3's "one handle per column, no per-tree explosion" holds and upload stays zero-copy via `as_bytes()`.
- **Which concrete cubecl CPU runtime** backs `Backend::CubeclCpu` (subject to SC2 bit-identical determinism). Researcher confirms against the cubecl manual.
- **Spike scope** — the minimal kernel that confirms control-flow shape (no `continue`), f64 in-kernel, and a single postprocessor's cast order before the full port. (Confirmation per D-04, not a gate.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### cubecl authoring (primary references for this phase)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — cubecl kernel authoring + backend selection (generic `R: Runtime`); the primary reference for `#[cube(launch)]`, the CPU runtime, control-flow constraints (no `continue`, helpers must be `#[cube]`), and `client.create_from_slice` upload.
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_ARROW_CUBECL.md`, `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_TRANSMUTATION_CUBECL.md` — zero-copy host→device patterns relevant to the SoA `as_bytes()` upload (GPU-05/SC3).
- `/home/user/Documents/workspace/optimisor/manual/HALF_PRECISION_CUBECL.md` — half-precision context (f16 is Phase-9/v2, but informs cubecl dtype handling).

### Upstream GTIL (porting source of truth — the behavior the kernels must reproduce to 1e-5)
- `treelite-mainline/src/gtil/predict.cc` — tree traversal, `NextNode`, the 4 predict kinds, output routing (`OutputLeafValue`/`OutputLeafVector`), RF averaging, f64 2D base-score add, serial tree-sum (GPU-01 traversal kernel target).
- `treelite-mainline/src/gtil/postprocessor.cc` + `treelite-mainline/src/gtil/postprocessor.h` — all postprocessors verbatim incl. the mixed-precision contracts the in-kernel port (D-03) must preserve.
- `treelite-mainline/src/gtil/output_shape.cc` — `GetOutputShape` per predict kind (output buffer sizing for the kernels).

### In-repo assets to extend (the seam Phase 6 plugs into)
- `crates/treelite-harness/src/lib.rs` — the **backend-parameterized seam**: `Backend` enum (lines ~54), `RunnerCase` (line ~90) with four fn-pointer slots (`DensePredictF32Fn`/`DensePredictF64Fn`/sparse twins, lines ~65–73), and `scalar_cpu_case()` (line ~107). Phase 6 adds `Backend::CubeclCpu` + a `cubecl_cpu_case()` constructor here.
- `crates/treelite-harness/src/manifest.rs` — `Manifest` + `check_manifest`; the `backend` field (line ~56) extended to per-cell provenance (D-06).
- `crates/treelite-harness/tests/gtil_matrix.rs` — the exhaustive matrix runner; reused unchanged per the registration-not-refactor design (D-11).
- `crates/treelite-gtil/src/lib.rs` (1553 lines) — the scalar reference: `evaluate_tree` (~432), `next_node` (~333), `predict_preset` (~643), `predict`/`predict_sparse` (~892/949), `RowSource` (~594), the `PredictScalar`/`PredictOut` traits (`to_f32`/`to_f64`/`threshold_to_f64`). The kernels reproduce this logic; this file stays the fallback.
- `crates/treelite-gtil/src/postprocessor.rs` (608 lines) — the 10 verbatim postprocessors (incl. softmax f32-max/f64-norm, exp2, f64 sigmoid/hinge twins) the in-kernel port (D-03) mirrors.
- `crates/treelite-core` — `TreeBuf<T>` (`Owned`/`Borrowed` POD enum) and the SoA `Tree<T>` columns; needs an `as_bytes()` surface for the SoA upload (SC3/GPU-05). Confirm/add the column byte-view accessor here.
- Root `Cargo.toml` `[workspace.dependencies]` — cubecl is NOT yet declared; Phase 6 adds the pinned cubecl crate(s) here (latest published version per project constraint).

### Prior context (precedent inherited)
- `.planning/phases/05-full-scalar-gtil-equivalence-harness/05-CONTEXT.md` — D-09 (manifest `backend` field), D-10/D-11 (generic `R: Runtime` runtime selection + backend-parameterized harness seam this phase implements), the scalar-reference-as-fallback contract.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — SoA/variant pattern, thiserror translation, golden-harness test layout.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase-5 `RunnerCase` seam** (`treelite-harness/src/lib.rs`) — purpose-built for this phase: a `Backend` enum + four fn-pointer slots + `scalar_cpu_case()`. Phase 6 registers `Backend::CubeclCpu` with a parallel constructor; the matrix iteration is untouched (D-11, registration-not-refactor).
- **Scalar GTIL** (`treelite-gtil`, 2603 LoC) — the complete, green, plain-Rust reference (`evaluate_tree`, `predict_preset`, `RowSource`, 10 verbatim postprocessors). It IS the fallback (D-02/D-06) and the 1e-5 measuring stick (D-11) — kernels reproduce it, never replace it.
- **Exhaustive frozen golden matrix + manifest** (Phase 5) — drives any `R: Runtime` from the same committed fixtures; CubeclCpu asserts against the identical goldens scalar-cpu already passes.
- **`TreeBuf<T>` POD columns** (`treelite-core`) — `Copy`-bounded owned/borrowed flat buffers; the natural source for the `as_bytes()` zero-copy SoA upload (SC3/GPU-05).

### Established Patterns
- **Verbatim port with exact mixed-precision cast order** — the 1e-5 contract; the in-kernel postprocessor port (D-03) must reproduce softmax (f32-max/f64-norm), exp2, and the f64 sigmoid/hinge twins exactly.
- **Serial tree-sum, parallel-over-rows** (GTIL-08 / SC1) — the kernel shape is one unit per row, trees looped serially; no reduce/atomic over the tree axis, no `continue`.
- **thiserror typed errors, bounds-checked routing** — no panic/OOB in library crates; the cubecl path and the fallback dispatch both preserve this.
- **Frozen goldens, CI never regenerates** — CubeclCpu is validated against the same committed fixtures; the manifest gains per-cell provenance (D-06).

### Integration Points
- `Backend::CubeclCpu` + `cubecl_cpu_case()` in `treelite-harness` is the single registration point; the four fn-pointer slots route dense f32/f64 through kernels and sparse f32/f64 through the scalar fallback this phase (D-02).
- `treelite-core` needs a column `as_bytes()` byte-view for the ragged-SoA upload (SC3/GPU-05) — likely a small additive accessor on `TreeBuf`/`Tree`.
- The cubecl dependency is new to the workspace — declared pinned in root `Cargo.toml`; the cubecl-cpu runtime choice must satisfy SC2 bit-identical determinism.
- CubeclCpu must coexist with the existing 1e-5 regression gates without regression — the scalar path stays byte-identical; the binary `(num_row,1,1)` path is unaffected.

</code_context>

<specifics>
## Specific Ideas

- **Precision is settled, not speculative.** The user has tested cubecl in-kernel precision (incl. f64 + mixed-precision) repeatedly — it reproduces scalar within 1e-5. The spike confirms control-flow shape, not whether the numbers land. Plans should reflect confidence, not hedging (D-04).
- **Honesty over a green checkmark.** The per-cell provenance (D-06) exists specifically so "1e-5 validated on cubecl-cpu" can never quietly mean "validated on scalar fallback." If sparse/categorical fall back, the report says so per cell.
- **Vertical slice, not horizontal stub.** D-01/D-02 take a real, end-to-end-kernelized slice (dense numerical, all 4 kinds, f32+f64, leaf-vector broadcast) rather than a thin all-surface stub — the cubecl proof is genuine on the path it covers.
- **Registration, not refactor.** The whole Phase-5 seam was built so Phase 6 is `Backend::CubeclCpu` + a constructor. If planning finds itself reshaping the matrix runner, that's a smell against D-11.

</specifics>

<deferred>
## Deferred Ideas

- **Sparse CSR input in cubecl kernels** — deferred from this phase's cubecl coverage (rides scalar fallback per D-02). A later cubecl-coverage pass (post-Phase-7 or a dedicated slice) ports the NaN-materialized ragged CSR row path into kernels. Recorded so it isn't lost.
- **Categorical splits in cubecl kernels** — deferred likewise (D-02): integer-membership bitset + float-representability guard + child polarity is the raggedest branch; scalar fallback this phase, kernel port later.
- **GPU backends (CUDA/wgpu/ROCm) runtime-selectable + per-model-class equivalence report** — Phase 7 (GPU-03/04). ROCm is the hardware-validated v1 GPU backend (developer's AMD device); CUDA build-supported, validated only where NVIDIA hardware exists. Everything routes through the same generic `R: Runtime` seam this phase establishes.
- **f16/bf16 half-precision in-kernel fast path** — v2 PERF-v2-01 / Phase-9 territory; off the 1e-5 equivalence path. `HALF_PRECISION_CUBECL.md` noted for when it lands.
- **Input-buffer bytemuck zero-copy recast beyond the SoA model upload** — Phase 9 memory hardening (MEM-01).

None of the above is scope creep out of Phase 6 — all recorded so they aren't lost.

</deferred>

---

*Phase: 6-cubecl-gtil-kernels-cpu-backend*
*Context gathered: 2026-06-10*

# Phase 5: Full Scalar GTIL & Equivalence Harness - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Widen the inference layer to the **complete scalar GTIL reference** — all 4 predict kinds (`default`, `raw`, `leaf_id`, `score_per_tree`), all 10 postprocessors, dense **and** sparse CSR input, categorical-split evaluation, NaN-only missing-value routing, and full output shaping — and build the **full seeded equivalence harness** that is the 1e-5 measurement instrument every later phase (Phases 6–7 cubecl/GPU) is validated against.

**Requirements covered:** GTIL-01, GTIL-02, GTIL-03, GTIL-04, GTIL-05, GTIL-06, GTIL-07, GTIL-08, EQV-01, EQV-02, EQV-03, EQV-04.

In scope (HOW, not WHETHER): completing the GTIL surface on top of the Phase-1..4 pulled-forward slice; the scalar reference stays **plain idiomatic Rust** (it is the reference/fallback all backends are later measured against); the seeded golden harness with committed input matrices + provenance manifests; dense↔sparse parity; both numeric presets `<f32,f32>`/`<f64,f64>`; **both f32 and f64 input dtypes**.

Out of scope: any cubecl kernel work (Phase 6); GPU backends (Phase 7); PyO3 binding / live numpy marshalling (Phase 8); memory-efficiency hardening — bytemuck/smallvec/allocator (Phase 9). The user-selectable runtime backend capability is recorded below as a **forward constraint** that shapes the Phase-5 harness seam but is **not implemented** this phase.

</domain>

<decisions>
## Implementation Decisions

### Equivalence harness coverage (EQV-01..04)
- **D-01: Exhaustive cross-product.** The matrix is the full product of (model types × **2 presets** `<f32,f32>`/`<f64,f64>` × **2 input dtypes** f32/f64 × **4 predict kinds** × **dense + sparse CSR** × seeds). Goldens are captured once and frozen, so the cost is one-time capture + committed fixture size + CI assert time — accepted for maximum confidence in the instrument every later phase trusts.
- **D-02: Few seeds, wide edge-seeded matrices.** ~2–3 seeds per cell, each a **wide** matrix (≈100–500 rows) deliberately seeded with edge values (NaN, ±inf, boundary thresholds, missing entries) so one matrix exercises many leaf paths and the missing/categorical guards in one shot. Each seed produces one committed input matrix + golden output vector.
- **D-03: Run every cell literally — no invariance pruning.** Even cells that are provably invariant across an axis (e.g. `leaf_id` is postprocessor- and preset-invariant; `score_per_tree` is postprocessor-invariant) are captured and asserted. "We tested everything" is the chosen posture; redundant fixtures are accepted.
- **D-04: Dense↔sparse parity is an additional assertion (SC1).** Every dense **and** sparse cell carries its own upstream Treelite golden (per D-03), AND the harness asserts Rust-dense == Rust-sparse on identical logical data (absent CSR entries materialized as **NaN, not 0**).

### Input dtype scope (GTIL-01/02)
- **D-05: Support both f32 and f64 input matrices**, orthogonal to the model preset — faithful to upstream's generic `Predict<InputT>` (float/double). The current Rust path is f32-input only; Phase 5 adds the f64-input path. This gives the exhaustive matrix its input-dtype axis and lets Phase-8 PyO3 hand numpy `float32` **and** `float64` zero-copy without a cast. The 4 (input × preset) combinations are all valid; input dtype is **not** constrained to match the preset.

### Predict API surface (what Phase-8 PyO3 wraps)
- **D-06: Idiomatic Rust config struct, not upstream JSON config.** Entry points are `predict(model, input, &config)` + `predict_sparse(model, csr, &config)` taking a typed `Config { kind: PredictKind, nthread, … }`, where `PredictKind` is a Rust enum (`Default`/`Raw`/`LeafId`/`ScorePerTree`). The behavior of upstream `gtil/config.cc` (`pred_type`, `nthread`) is honored, but JSON-config **parsing stays out of the compute crate** — any JSON-string compatibility shim, if ever needed, lives at the PyO3 edge.
- **D-07: Flat buffer + Shape descriptor return.** `predict` returns a flat `Vec` plus a `Shape` descriptor computed per kind (`GetOutputShape`: `default`/`raw` → (rows, targets, classes); `leaf_id` → (rows, trees); `score_per_tree` → (rows, trees, …)). Extends the current contiguous-buffer contract; zero-copy-friendly for PyO3 numpy reshaping in Phase 8. No richer typed-output abstraction (would be unwrapped back to a flat buffer anyway and departs from upstream's flat-buffer contract).

### Golden reproducibility rigor (the flagged blocker)
- **D-08: Committed matrices are the contract; seeds are documentation.** The actual input matrices AND golden output vectors are the frozen source of truth. Seeds + the capture script are committed too, but as a **regeneration aid only** — CI asserts against the committed matrices and **never re-draws from a seed** (cross-platform RNG/libm drift can't silently change inputs). Continues the existing discipline (`fixtures/golden.json` is never hand-edited).
- **D-09: Full-provenance manifest + backend field.** Each golden set commits a manifest with: OS + arch, libm/libc identity, rustc + cubecl version, **every** capture framework version (xgboost, lightgbm, scikit-learn, numpy, upstream treelite), seed, a sha256 per fixture, **and a `backend` field** (`scalar-cpu` | `cubecl-cpu` | `cuda` | `wgpu` | `rocm`) recording which `R: Runtime` produced/asserted the vector. One schema across the whole harness — Phase 5 records `scalar-cpu`. Extends the existing `Manifest` struct + `check_manifest` (`crates/treelite-harness/src/lib.rs`) and the per-loader manifest format already frozen in Phases 1/3/4.

### Verbatim port (carried from prior phases — pre-decided, not re-litigated)
- All postprocessors ported **verbatim with upstream mixed-precision cast order** (04-02 already shipped `identity`, `identity_multiclass`, `sigmoid`, `exponential`, `exponential_standard_ratio`, `logarithm_one_plus_exp`, `softmax`; Phase 5 adds the remaining set incl. `signed_square`, `hinge`, `multiclass_ova`). `softmax` = f32 max-subtraction + f64 norm accumulate + f32 divide; `exponential_standard_ratio` uses `exp2` (base-2).
- Per-row tree summation is **serial in `tree_id` order** (GTIL-08); parallelism only across rows.
- Bounds-checked output routing → typed `GtilError` (never OOB/panic on malformed trees).

### Forward constraint — user-selectable runtime backend (NOT implemented this phase)
- **D-10: The end user selects and switches the compute backend at runtime** across `{ scalar-cpu, cubecl-cpu, cuda, wgpu, rocm }`, implemented via **cubecl generics over `R: Runtime`** (uniform runtime selection, not a recompile). **Implementation is Phase 6** (generic seam + cubecl-CPU default) **and Phase 7** (runtime-selectable GPU backends). ROCm is part of the intended user-facing set even though it currently sits in v2 (PERF-v2-02) — see Deferred Ideas for the roadmap reconciliation.
- **D-11: Phase-5 effect of D-10 (in scope now):** the equivalence harness is built **backend-parameterized** — the same frozen golden matrices drive any `R: Runtime`, with backend identity recorded via the manifest `backend` field (D-09). The Phase-5 scalar GTIL is the plain-Rust **reference/fallback** (`scalar-cpu`) that sits behind that seam and that every future backend is measured against to 1e-5.

### Claude's Discretion
- **Sparse CSR Rust representation** — the exact CSR type (indptr/indices/data slices vs a thin view), provided absent entries materialize as NaN and dense↔sparse parity (D-04) holds.
- **`Config`/`PredictKind` exact field set and module placement** — derive from `gtil.h` + `config.cc`, keeping the core crate JSON-free (D-06).
- **Harness fixture layout / capture-script structure** under `fixtures/` — extend the existing per-loader pattern; planner/research decide file naming and the capture-script organization.
- **Which concrete models populate each capability axis** of the exhaustive matrix (sparse, categorical, multiclass/leaf-vector broadcast, missing-value) — pick representative models per axis from the test corpus + existing per-loader fixtures.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Upstream GTIL (porting source of truth)
- `treelite-mainline/src/gtil/predict.cc` — full GTIL predict: tree traversal, `NextNode`/`NextNodeCategorical`, the 4 predict kinds, output routing (`OutputLeafValue`/`OutputLeafVector`), RF averaging, f64 2D base-score add, serial tree-sum (GTIL-01/02/03/05/06/07/08).
- `treelite-mainline/src/gtil/postprocessor.cc` + `treelite-mainline/src/gtil/postprocessor.h` — all 10 postprocessors verbatim incl. the mixed-precision contracts (GTIL-04).
- `treelite-mainline/src/gtil/output_shape.cc` — `GetOutputShape` per predict kind (GTIL-07; informs the `Shape` descriptor, D-07).
- `treelite-mainline/src/gtil/config.cc` — `Configuration` (`pred_type`, `nthread`) semantics that the idiomatic Rust `Config` honors (D-06; JSON parsing stays at the PyO3 edge).
- `treelite-mainline/include/treelite/gtil.h` — public GTIL API surface (`Predict<InputT>` generic over f32/f64 input → D-05; `PredictSparse`; predict-kind enum).

### In-repo assets to extend
- `crates/treelite-gtil/src/lib.rs` — existing scalar `predict` (flat `(num_row, num_target, max_num_class)` buffer, four-way leaf routing, RF averaging, f64 base-score add, serial tree-sum); the entry point widened to `Config`/`PredictKind` + sparse + f64 input (D-05/06/07).
- `crates/treelite-gtil/src/postprocessor.rs` — 7 of 10 postprocessors already verbatim; add the remaining set (`signed_square`, `hinge`, `multiclass_ova`, …).
- `crates/treelite-gtil/src/error.rs` — typed `GtilError` (extend for new routing/sparse/kind errors).
- `crates/treelite-harness/src/lib.rs` — `Golden { input, output, manifest }` struct, `Manifest`, `check_manifest` (extend for the exhaustive matrix + `backend` field, D-08/D-09).
- `crates/treelite-harness/tests/{equivalence,run_equivalence,lightgbm,sklearn,three_format_equivalence}.rs` — existing golden-assert test pattern to generalize across the matrix.
- `fixtures/` (workspace root) — `golden.json`, `golden_v5.manifest.json`, `xgb_3format.manifest.json`, per-loader `*.golden.json`, and the `capture_*.py` scripts: the frozen-golden + manifest discipline (D-06/D-07 lineage) the Phase-5 harness extends. Capture runs via `uv run python` on the main tree.

### Test corpus (capability-axis fixtures)
- `treelite-mainline/tests/examples/toy_categorical/`, `treelite-mainline/tests/examples/sparse_categorical/` — categorical-split + sparse material (GTIL-02/GTIL-06 float-representability guard + child polarity).
- `treelite-mainline/tests/examples/` — broader example models usable to populate the multiclass / leaf-vector-broadcast / missing-value axes.
- `treelite-mainline/tests/cpp/`, `treelite-mainline/tests/python/` — document expected GTIL behavior per kind.

### Prior context (precedent inherited)
- `.planning/phases/04-lightgbm-scikit-learn-loaders/04-CONTEXT.md` — D-03/D-06/D-07: pulled-forward GTIL + "golden frozen from upstream **Treelite GTIL**, not the framework" discipline this phase completes.
- `.planning/codebase/{ARCHITECTURE,CONVENTIONS,TESTING}.md` — SoA/variant pattern, thiserror translation, test layout.

### Forward (Phase 6/7 backend work — informs D-10/D-11 seam only)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — cubecl kernel authoring + backend selection (generic `R: Runtime`); referenced so the Phase-5 harness seam is backend-ready.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`crates/treelite-gtil` scalar predict** (Phases 1–4) — flat-buffer predict with four-way leaf routing, RF averaging, f64 base-score add, serial tree-sum, and 7/10 verbatim postprocessors; the spine Phase 5 widens rather than rewrites.
- **`treelite-harness` golden+manifest harness** — `Golden`/`Manifest`/`check_manifest` already enforce os/arch provenance; extended to the exhaustive matrix + `backend` field.
- **`fixtures/capture_*.py`** — the `uv run python` one-time capture pattern (input matrix + upstream-Treelite golden + frozen manifest) generalized to the seeded matrix.

### Established Patterns
- **Parse-wide / verify-narrow → now verify-complete:** Phases 1–4 pulled forward a minimal GTIL slice; Phase 5 completes the surface and does NOT backfill loader basic parity.
- **Golden frozen from upstream Treelite GTIL; CI never regenerates** (D-08) — asserted target is upstream Treelite, not the source framework.
- **Verbatim postprocessor port with exact cast order** — the 1e-5 contract; mixed-precision (f32 max / f64 norm in softmax, `exp2` base-2) preserved.
- **thiserror typed errors, bounds-checked routing** — no panic/OOB on malformed trees in library crates.

### Integration Points
- Widened GTIL must coexist with the existing 1e-5 XGBoost/LightGBM/sklearn regression gates without regression (binary `(num_row,1,1)` path stays byte-identical).
- The f64-input path (D-05) and sparse path (D-04) are new code exercising the `<f64,f64>` preset and CSR materialization end-to-end.
- The backend-parameterized harness seam (D-11) is the contract Phase 6 plugs cubecl `R: Runtime` into — design it so adding a backend is a registration, not a harness refactor.

</code_context>

<specifics>
## Specific Ideas

- The harness is explicitly "the 1e-5 measurement instrument for everything after" — its thoroughness (exhaustive, D-01) is a deliberate investment, not gold-plating.
- Sparse semantics: **absent CSR entries = NaN, never 0** (SC1) — and NaN-only missing routing (GTIL-05) fires via the node default direction.
- Categorical evaluation must apply the **float-representability guard + correct child polarity** (GTIL-06) — the minimal Phase-4 `NextNodeCategorical` branch is completed to the exhaustive matrix here.
- Manifest must make a future 1e-5 miss diagnosable **on any backend** — hence the `backend` field added now (D-09) even though only `scalar-cpu` exists this phase.
- End-user backend selection is uniform via generic `R: Runtime` (D-10) — the design intent is "select a feature / switch backend at runtime," not per-backend recompiles.

</specifics>

<deferred>
## Deferred Ideas

- **cubecl GTIL kernels (traversal + postprocessors) generic over `R: Runtime`, CPU backend default** — Phase 6 (GPU-01/02/05). Phase 5 ships the plain-Rust scalar reference + the backend-parameterized harness seam they plug into.
- **Runtime-selectable GPU backends + per-model-class equivalence report** — Phase 7 (GPU-03/04).
- **ROCm as a v1 user-selectable backend (roadmap reconciliation).** The user wants `{scalar-cpu, cubecl-cpu, cuda, wgpu, rocm}` all runtime-selectable (D-10). Today ROCm sits in **v2 (PERF-v2-02)** while wgpu+CUDA are v1 (GPU-03). Action: reconcile at the **roadmap level** — either pull ROCm into v1 GPU-03 or confirm v2 with the generic `R: Runtime` seam kept ROCm-ready. Not a Phase-5 decision; recorded so it isn't lost. Because everything routes through generic `R: Runtime`, adding `cubecl-rocm` later is a backend registration, not a refactor.
- **PyO3 JSON-config compatibility shim** (if upstream's JSON `Configuration` string is ever needed) — lives at the Phase-8 PyO3 edge, not the compute crate (D-06).
- **Memory-efficiency (bytemuck zero-copy recast of SoA columns for input buffers, smallvec/compact_str)** — Phase 9.

None of the above is scope creep out of Phase 5 — all recorded so they aren't lost.

</deferred>

---

*Phase: 5-full-scalar-gtil-equivalence-harness*
*Context gathered: 2026-06-10*

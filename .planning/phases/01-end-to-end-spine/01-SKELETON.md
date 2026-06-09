# Walking Skeleton — treelite-rs

**Phase:** 1
**Generated:** 2026-06-10

## Capability Proven End-to-End

One simple XGBoost-JSON `binary:logistic` model is loaded into the core `Model` representation, predicted with a scalar single-threaded GTIL pass (identity/sigmoid only), and the output is asserted to match a frozen golden vector captured from the upstream `treelite==4.7.0` Python wheel within **1e-5** — proving the project's core fidelity value (load → represent → predict → verify) through the whole pipeline on day one.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Build system | Cargo virtual workspace, `resolver = "3"`, `edition = "2024"` via `[workspace.package]` | FND-01; resolver 3 is the edition-2024 default; one root holds all member crates (D-01/D-02) |
| Dependency management | Single root `[workspace.dependencies]` table, every crate pinned to current stable (no pre-release) | FND-02; `approx` pinned `0.5.1` NOT `0.6.0-rc2` (RESEARCH §Standard Stack) |
| Crate layout (spine-only) | `treelite-core`, `treelite-xgboost`, `treelite-gtil`, `treelite-harness` | D-01/D-02 — only the crates Phase 1 exercises; later phases widen one layer each |
| In-memory model | Two-variant `enum ModelVariant { F32(ModelPreset<f32>), F64(ModelPreset<f64>) }`, header metadata on `Model` outside the variant | CORE-01; mirrors upstream `std::variant` (tree.h:437); no mixed threshold/leaf types |
| Tree storage | Struct-of-Arrays: every node field is a parallel `TreeBuf<T>` column, never a `Node` struct | CORE-02; cache-friendly + zero-copy serialization story (tree.h:97-132) |
| Buffer primitive | `TreeBuf<T> { Owned(Vec<T>), Borrowed { ptr, len } }`, `T: Copy` | CORE-03; mirrors `ContiguousArray` owned/`UseForeignBuffer`; `bytemuck` POD seam deferred to Phase 9 |
| Errors | Per-crate `thiserror` enums (`CoreError`, `XgbError`, `GtilError`); `anyhow` only in `treelite-harness`/tests | ERR-01/ERR-02; upstream `TREELITE_LOG(FATAL)` paths become typed `Err`, never panic |
| Predict altitude | Plain `fn predict(&Model, &[f32], num_row) -> Result<Vec<f32>, GtilError>` — NO Predictor/backend trait | D-08; the cubecl seam is deferred to Phase 6 (research-flagged); designing the abstraction now risks the wrong boundary |
| Fixture | Hand-crafted XGBoost-JSON `binary:logistic` literal, `base_score = 0.25`, `version: [4,7,0]` | D-04/D-05/D-05a; loadable by the upstream wheel; 0.25 (not 0.5) genuinely exercises the sigmoid margin transform |
| Golden artifact | `fixtures/golden.json` = `{input, output, manifest}` captured once from `treelite==4.7.0` GTIL, frozen, never regenerated in CI | D-06/D-07; manifest records treelite/xgboost/OS/arch/glibc/python for libm-divergence diagnosability |
| Numerical contract | Serial tree-sum in tree_id order; f64 base_score margin transform; f32 accumulator with f64 base_score added in; f32 sigmoid_alpha + `exp` on f32 | The 1e-5 contract — cast/ordering ported verbatim from `predict.cc` + `postprocessor.cc` + `xgboost.cc` |
| Directory layout | `crates/treelite-*/src/*.rs` (one file per concern); `fixtures/` at repo root; `treelite-mainline/` + `xgboost-master/` vendored read-only | RESEARCH §Recommended Project Structure |

## Stack Touched in Phase 1

- [x] Project scaffold — virtual Cargo workspace, edition 2024, resolver 3, pinned `[workspace.dependencies]`, `cargo build`/`cargo test` (Plan 01-01)
- [x] Vocabulary layer — four enums with exact upstream string round-trip (Plan 01-01)
- [x] Core model — `Model`/`Tree<T>`/`TreeBuf<T>` SoA representation, both owned + borrowed modes, full header metadata (Plan 01-01)
- [x] Loader — one real XGBoost-JSON model parsed into a `Model` (Plan 01-02)
- [x] Inference — one real scalar predict (identity/sigmoid) (Plan 01-03)
- [x] Verification — frozen golden + 1e-5 equivalence harness end-to-end test (Plan 01-01 golden capture, Plan 01-04 assertion)

## Out of Scope (Deferred to Later Slices)

Explicit — this list prevents later phases from re-litigating Phase 1's minimalism:

- `ModelBuilder`, `ConcatenateModelObjects`, bulk construction, v5 binary/PyBuffer/JSON serialization, field accessors — **Phase 2**
- XGBoost UBJSON + legacy binary formats, format auto-detect, full base_score (scalar+vector) handling, NaN/Inf JSON literals — **Phase 3**
- LightGBM text + scikit-learn (RF/ET/GBM/IsolationForest/HistGradientBoosting) loaders — **Phase 4**
- Full GTIL surface: 4 predict kinds (`raw`/`leaf_id`/`score_per_tree`), all 10 postprocessors, sparse CSR, categorical splits, output shaping/leaf-vector broadcast, tree averaging, seeded golden harness — **Phase 5**
- cubecl CPU kernels + the Predictor/backend trait seam — **Phase 6**
- GPU backend + equivalence report — **Phase 7**
- PyO3 binding, zero-copy numpy I/O, abi3 wheel — **Phase 8**
- `bytemuck` Pod recast, `smallvec`/`compact_str`, custom global allocator — **Phase 9**
- GitHub Actions CI — discretionary, may be added here or later (success criteria only require `cargo build`/`cargo test`)

## Subsequent Slice Plan

Each later phase widens ONE layer of this proven spine without altering its architectural decisions; each ends runnable + 1e-5-tested:

- **Phase 2:** validated `ModelBuilder` (+ concatenate + bulk path) and full v5 serialization — loaders get a construction target; models round-trip.
- **Phase 3:** full XGBoost loaders — JSON + UBJSON + legacy binary with auto-detect and version-gated base_score transform across all formats.
- **Phase 4:** LightGBM text + the full scikit-learn estimator set (incl. HistGradientBoosting).
- **Phase 5:** complete scalar GTIL — all predict kinds, all postprocessors, sparse CSR, categoricals, output shaping + the full seeded equivalence harness (the 1e-5 instrument for everything after).
- **Phase 6:** GTIL hot path reimplemented as cubecl kernels, CPU backend default, validated to 1e-5.
- **Phase 7:** runtime-selectable GPU backend + documented per-model-class deviation report.
- **Phase 8:** PyO3 binding — load/predict/serialize/dump from Python, zero-copy numpy, abi3 wheel.
- **Phase 9:** memory-efficiency hardening (bytemuck/smallvec/compact_str/jemalloc) without regressing 1e-5.

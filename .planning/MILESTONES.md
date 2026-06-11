# Milestones

## v1.1 Parallel Scalar Inference (Shipped: 2026-06-11)

**Scope:** 1 phase (Phase 10), 2 plans, 5 tasks. Requirements PAR-01..04.

**Delivered:** Row-parallelized the single-core scalar GTIL fallback (the whole-model path for LightGBM `kLE`, categorical, every non-`kLT`, and all sparse models) without regressing the 1e-5 contract.

**Key accomplishments:**

- Pinned rayon 1.12.0, made `Model` (and `TreeBuf<T>`) soundly `Sync` for read-only predict via a documented `unsafe impl`, added `GtilError::ThreadPool`, and replaced the Phase-9 `!Send` invariant with a `requires_sync::<Model>()` contract test (PAR-03).
- Converted all four scalar GTIL row-loop families to rayon `par_chunks_mut`/`map_init` (per-worker scratch), keeping the inner per-row `tree_id` sum serial (GTIL-08) (PAR-01/02).
- Threaded `Config.nthread` end-to-end through a scoped `ThreadPool` (`run_with_nthread`: ≤0 = all cores, N bounded, never `build_global`) and wired the Python `nthread=` kwarg through to the scalar path (PAR-04).
- Proven parallel output byte-identical to serial across runs and nthread settings, all within the 1e-5 golden gate; measured 3.68× throughput on a categorical LightGBM model (4M rows, 16 cores).

**Quality gates:** verified 6/6 must-haves + human throughput UAT (3.68×); code review 5/5 findings fixed; security 5/5 threats closed (ASVS high).

**Archives:** `milestones/v1.1-ROADMAP.md`, `milestones/v1.1-REQUIREMENTS.md` (cumulative roadmap/requirements at v1.1 close).

**Note:** This was the first *formal* GSD milestone close. v1.0 (below) had shipped its phases incrementally but was never separately archived, so the cumulative roadmap/requirements snapshots live alongside this entry.

---

## v1.0 MVP (Shipped: 2026-06-09 → 2026-06-11, archived at v1.1 close)

**Scope:** 9 phases (1–9), 49 plans. The from-scratch Rust port spine: load → predict → serialize → GPU → Python → memory, all validated to 1e-5 vs upstream Treelite 4.7.0.

**Key accomplishments:**

- **Core + harness (Ph 1–2):** Cargo workspace, `treelite-core` (four upstream-exact enums, `TreeBuf<T>` SoA primitive, two-variant `Model`), validated `ModelBuilder` + bulk path, byte-for-byte v5 binary/JSON/PyBuffer serialization, and the seeded 1e-5 equivalence harness (bitwise-exact vs upstream on the spine fixture).
- **Loaders (Ph 3–4):** Full XGBoost (JSON/UBJSON/legacy-binary with auto-detect + version-gated base_score margin transform), LightGBM text (numerical + categorical bitset), and scikit-learn (RF/ET/GB/IsolationForest/HistGradientBoosting) — all within 1e-5.
- **Full GTIL (Ph 5):** All 4 predict kinds, 10 postprocessors, sparse CSR, categoricals, output shaping; the 64+ frozen golden matrix that is the 1e-5 measurement instrument for everything after.
- **cubecl kernels + GPU (Ph 6–7):** Traversal + postprocessor kernels (CPU backend default), then runtime-selectable ROCm/CUDA/wgpu backends with an on-hardware (AMD/ROCm) per-model-class deviation report — 160 cells within 1e-5, worst 2.9e-6.
- **Python + memory (Ph 8–9):** PyO3 abi3 wheel (load/predict/serialize/dump with zero-copy numpy I/O, GIL released), then memory hardening (bytemuck Pod recast, SmallVec/CompactString metadata, runtime-selectable jemalloc/mimalloc) — zero behavior change, golden + 1e-5 green.

---

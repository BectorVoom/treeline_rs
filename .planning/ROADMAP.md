# Roadmap: treelite-rs

## Overview

treelite-rs is a strict numerical port of Treelite 4.7.0 (C++) to a Rust Cargo workspace, where the single non-negotiable invariant is that predictions match upstream within **1e-5**. The work was structured as vertical MVP slices laid along the dependency spine: enums → core model → builder/serialize → loaders → scalar GTIL + equivalence harness → cubecl CPU kernels → GPU backend → PyO3 → memory hardening, then row-parallel scalar inference.

## Milestones

- ✅ **v1.0 MVP** — Phases 1–9 (shipped 2026-06-09 → 2026-06-11)
- ✅ **v1.1 Parallel Scalar Inference** — Phase 10 (shipped 2026-06-11)

Full phase details are archived per milestone in `.planning/milestones/`.

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1–9) — SHIPPED 2026-06-11</summary>

- [x] Phase 1: End-to-End Spine (4/4 plans) — completed 2026-06-09
- [x] Phase 2: Builder & Serialization (6/6 plans) — completed 2026-06-10
- [x] Phase 3: Full XGBoost Loaders (4/4 plans) — completed 2026-06-10
- [x] Phase 4: LightGBM & scikit-learn Loaders (8/8 plans) — completed 2026-06-10
- [x] Phase 5: Full Scalar GTIL & Equivalence Harness (7/7 plans) — completed 2026-06-10
- [x] Phase 6: cubecl GTIL Kernels (CPU Backend) (7/7 plans) — completed 2026-06-10
- [x] Phase 7: GPU Backend & Equivalence Report (4/4 plans) — completed 2026-06-10
- [x] Phase 8: PyO3 Python Binding (5/5 plans) — completed 2026-06-11
- [x] Phase 9: Memory-Efficiency Hardening (4/4 plans) — completed 2026-06-11

</details>

<details>
<summary>✅ v1.1 Parallel Scalar Inference (Phase 10) — SHIPPED 2026-06-11</summary>

- [x] Phase 10: Parallel Scalar Inference (2/2 plans) — completed 2026-06-11
  Row-parallelize scalar dense + sparse GTIL across all cores via a sound shareable `Model` and honored `Config.nthread`, output identical to the serial path within 1e-5. (PAR-01..04)

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. End-to-End Spine | v1.0 | 4/4 | Complete | 2026-06-09 |
| 2. Builder & Serialization | v1.0 | 6/6 | Complete | 2026-06-10 |
| 3. Full XGBoost Loaders | v1.0 | 4/4 | Complete | 2026-06-10 |
| 4. LightGBM & scikit-learn Loaders | v1.0 | 8/8 | Complete | 2026-06-10 |
| 5. Full Scalar GTIL & Equivalence Harness | v1.0 | 7/7 | Complete | 2026-06-10 |
| 6. cubecl GTIL Kernels (CPU Backend) | v1.0 | 7/7 | Complete | 2026-06-10 |
| 7. GPU Backend & Equivalence Report | v1.0 | 4/4 | Complete | 2026-06-10 |
| 8. PyO3 Python Binding | v1.0 | 5/5 | Complete | 2026-06-11 |
| 9. Memory-Efficiency Hardening | v1.0 | 4/4 | Complete | 2026-06-11 |
| 10. Parallel Scalar Inference | v1.1 | 2/2 | Complete | 2026-06-11 |

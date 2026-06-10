# Phase 4: LightGBM & scikit-learn Loaders - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 4-lightgbm-scikit-learn-loaders
**Areas discussed:** sklearn input contract, Verify-narrow scope, Numeric preset, Fixtures/goldens, Golden source, HistGB scope

---

## sklearn input contract

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror C-API array signature | Rust fns 1:1 with upstream `namespace sklearn` array signatures; Phase 8 PyO3 calls them with zero-copy numpy buffers; tested via frozen array-dump fixtures | ✓ |
| Parse a serialized array-dump format | Intermediate on-disk JSON/npz the Rust loader parses; decoupled from Python contract | |
| You decide | Defer to research/planning | |

**User's choice:** Mirror C-API array signature
**Notes:** Faithful 1:1 port of `treelite-mainline/include/treelite/model_loader.h` `namespace sklearn`. No intermediate file format introduced.

---

## Verify-narrow scope (Phase 4 vs Phase 5)

| Option | Description | Selected |
|--------|-------------|----------|
| Pull forward minimal GTIL per loader | Port just enough GTIL per estimator to verify 1e-5 this phase; Phase 5 widens to complete surface | ✓ |
| Parse-wide, verify on GTIL-ready subset | Load all, verify only what current scalar GTIL supports; defer the rest's prediction parity to Phase 5 | |
| You decide | Defer to research | |

**User's choice:** Pull forward minimal GTIL per loader
**Notes:** Phase 4 proves real prediction parity, not just structural load. Phase 5 does not have to backfill basic parity for Phase-4 loaders.

---

## Numeric preset mapping

| Option | Description | Selected |
|--------|-------------|----------|
| f64 preset for both | LightGBM + sklearn → `<f64,f64>` ModelPreset, matching upstream per-field precision; XGBoost stays `<f32,f32>` | ✓ |
| Match upstream exactly per loader | Defer to whatever each upstream loader emits; research confirms exact types first | |
| You decide | Defer to research | |

**User's choice:** f64 preset for both
**Notes:** First end-to-end exercise of the f64 variant. Planner/research must confirm exact upstream ThresholdType/LeafOutputType and not silently downcast.

---

## Fixtures & goldens

| Option | Description | Selected |
|--------|-------------|----------|
| One-time Python capture, freeze all | `uv run python` fits estimators, dumps arrays + input + golden + manifest, committed frozen; CI never regenerates | ✓ |
| Capture against upstream Treelite too | Same, plus capture golden through upstream Treelite GTIL for stronger fidelity | (refined below) |
| You decide | Defer to research/planning | |

**User's choice:** One-time Python capture, freeze all — refined by the "Golden source" question below to capture from upstream Treelite GTIL.
**Notes:** Mirrors D-05/D-06 golden discipline; pin sklearn + lightgbm + treelite versions in the manifest.

---

## Golden source (refinement)

| Option | Description | Selected |
|--------|-------------|----------|
| Upstream Treelite GTIL | Golden = `treelite.gtil.predict`; the actual port target; framework predict only a sanity cross-check | ✓ |
| Framework predict only | Capture from sklearn/LightGBM's own predict(); risks verifying the wrong target where Treelite diverges | |
| You decide | Defer to research | |

**User's choice:** Upstream Treelite GTIL
**Notes:** The 1e-5 contract is against upstream Treelite. IsolationForest's golden = `-clf.score_samples` and deliberately differs from the framework's predict.

---

## HistGradientBoosting scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full import + verify this phase | Port complete HistGB path (bin_mapper, features_map, packed nodes) and verify 1e-5 vs upstream-Treelite golden in Phase 4 | ✓ |
| Full import, verify if GTIL-ready | Build full loader, verify only if pulled-forward GTIL covers it; else defer prediction parity to Phase 5 | |
| You decide | Defer to research | |

**User's choice:** Full import + verify this phase
**Notes:** Honors SKL-04 fully now; the research-flagged risk and largest single chunk of phase work. Reinforces the pull-forward-GTIL decision.

---

## Claude's Discretion

- Crate organization (per-format crates vs combined) — planner's call, following the `treelite-xgboost` per-format pattern.
- LightGBM text-parse mechanics (streaming/line parser, categorical-bitset decode, string_utils analog).
- HistGB packed-node decode mechanics.

## Deferred Ideas

- Complete GTIL surface (4 predict kinds, 10 postprocessors, sparse CSR, full categorical/output shaping) — Phase 5.
- PyO3 marshalling of live fitted estimators — Phase 8.
- Multi-target sklearn coverage beyond captured fixtures — Phase 5 harness.
- LightGBM categorical-split prediction parity beyond the captured fixture — Phase 5 categorical GTIL.

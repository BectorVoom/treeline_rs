# Phase 1: End-to-End Spine - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-09
**Phase:** 1-End-to-End Spine
**Areas discussed:** Crate scaffolding scope, Phase 1 fixture model, Golden-vector capture, Predict-path altitude

---

## Crate Scaffolding Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Spine-only, grow per phase | Create only the crates Phase 1 exercises (treelite-core, treelite-gtil, treelite-xgboost, treelite-harness); add crates when later phases need them. | ✓ |
| Scaffold all 9-phase crates now | Empty stubs for every eventual member up front. | |
| Single crate now, split later | Start as one crate, extract members as it grows (conflicts with FND-01). | |

**User's choice:** Spine-only, grow per phase.
**Notes:** Matches the MVP-slice roadmap ("widen one layer per phase"); avoids dead empty crates. FND-01 still satisfied (multi-crate workspace from Phase 1).

---

## Phase 1 Fixture Model

| Option | Description | Selected |
|--------|-------------|----------|
| Generate tiny model via xgboost Python, commit the .json | Train minimal binary:logistic model, save_model() to JSON, commit. | |
| Hand-craft the JSON literal by hand | Author a minimal valid XGBoost-JSON document directly in the repo. | ✓ |
| You decide | Defer fixture approach to planning. | |

**User's choice:** Hand-craft the JSON literal by hand.
**Notes:** Surfaced critical coupling — the hand-crafted JSON must still be parseable by the upstream Treelite wheel (it produces the golden), so it must conform to the real XGBoost-JSON schema; use binary:logistic to exercise the sigmoid path. Context: vendored tests/examples/ ships NO XGBoost-JSON model (mushroom is legacy binary), so the fixture had to be created either way.

---

## Golden-Vector Capture

| Option | Description | Selected |
|--------|-------------|----------|
| Capture from upstream Treelite Python wheel GTIL + manifest | pip install treelite, load fixture, run gtil.predict, freeze output + manifest. | ✓ |
| Build C++ Treelite from treelite-mainline once, capture | Compile vendored v4.7.0 source, run GTIL, freeze. | |
| Hand-compute expected from the known tiny model | Compute identity/sigmoid by formula, cross-check later. | |

**User's choice:** Capture from upstream Treelite Python wheel GTIL + manifest.
**Notes:** Authoritative GTIL source without a C++ source compile; satisfies PROJECT's "golden frozen from upstream Treelite." Manifest records treelite/xgboost versions, OS/arch, libm/glibc. Context: no checked-in golden vectors exist upstream, so a golden had to be produced.

---

## Predict-Path Altitude

| Option | Description | Selected |
|--------|-------------|----------|
| Simplest scalar fn, refactor at Phase 6 | Plain single-threaded scalar predict, no backend abstraction. | ✓ |
| Build a Predictor/backend trait seam now | Define the cubecl swap point at Phase 1. | |

**User's choice:** Simplest scalar fn, refactor at Phase 6.
**Notes:** Phase 6 is research-flagged and will spike kernel shape before porting; designing the seam now risks the wrong abstraction. Keeps the spine genuinely thin (YAGNI).

---

## Claude's Discretion

- Exact crate names/granularity within the spine-only constraint (default: enums live in `treelite-core`).
- `TreeBuf<T>` owned-vs-borrowed Rust representation mechanism (must support zero-copy borrow per CORE-03).
- Error-enum granularity (default: per-crate `thiserror` enums).
- `DType` numeric-type coverage for Phase 1 (must match upstream `TypeInfo` strings).
- Whether GitHub Actions CI is wired in Phase 1 or deferred.
- Manifest file format and location.

## Deferred Ideas

- Backend/`Predictor` trait abstraction → Phase 6 (cubecl).
- Real-world example fixtures (mushroom binary, LightGBM text) → Phase 3/4.
- GitHub Actions CI → here or later, Claude's discretion.
- serde_json NaN/Inf handling → Phase 3 (Phase 1 fixture avoids NaN/Inf literals).

# Phase 3: Full XGBoost Loaders - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 3-full-xgboost-loaders
**Areas discussed:** Parsing architecture, Three-format fixtures & golden, Legacy binary & auto-detect, DEF-02-01 byte-fidelity scope

---

## Parsing architecture

### Q1 — JSON/UBJSON shared parsing

| Option | Description | Selected |
|--------|-------------|----------|
| Converge at serde structs | Decode UBJSON into the same serde_json::Value / XgbModelJson structs the JSON path uses; parity at deserialize level; keeps Phase-1 code | ✓ |
| Port upstream SAX handler | Faithfully port DelegatedHandler + two adapters; replaces Phase-1 serde-DOM loader with streaming | |
| Separate UBJSON decoder | Standalone independent UBJSON decoder; two numeric paths risk drift | |

**User's choice:** Converge at serde structs

### Q2 — NaN/Inf literal handling

| Option | Description | Selected |
|--------|-------------|----------|
| Lock requirement, research picks | Lock invariant (bare NaN/Inf → f32, value-position only); research chooses mechanism | ✓ |
| Tolerant pre-lex sentinel | String-safe byte pass rewrites NaN/Inf to sentinel; custom serde deserializer restores | |
| Swap to NaN/Inf parser | Replace JSON front-end with a NaN/Inf-tolerant parser | |

**User's choice:** Lock requirement, research picks

### Q3 — UBJSON decode mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Hand-rolled tag decoder → Value | Hand-write UBJSON type-tag decoder emitting serde_json::Value | |
| UBJSON serde data-format crate | Pull a UBJSON-as-serde crate; Deserialize straight into XgbModelJson | |
| Research picks | Lock convergence at serde structs; research chooses hand-rolled vs crate | ✓ |

**User's choice:** Research picks

### Q4 — Loader surface width (given GTIL is scalar until Phase 5)

| Option | Description | Selected |
|--------|-------------|----------|
| Parse-wide, verify-narrow | Parse full structure (multiclass, categorical fields, vector base_score, DART); verify 1e-5 only on what scalar GTIL predicts | ✓ |
| Strictly what GTIL predicts | Only parse features the scalar GTIL can predict end-to-end | |
| Full surface + pull GTIL forward | Parse AND predict full set; pull categorical/multiclass GTIL forward from Phase 5 | |

**User's choice:** Parse-wide, verify-narrow

---

## Three-format fixtures & golden

### Q1 — Same logical model across 3 formats

| Option | Description | Selected |
|--------|-------------|----------|
| Round-trip via xgboost Python | Load vendored mushroom.model, re-save JSON/UBJSON, capture golden from wheel | |
| Fresh-train one, save 3 ways | Train one small binary:logistic model, save all 3 formats from one session, capture golden | ✓ |
| Research designs it | Lock requirement; research chooses generation path after testing compat | |

**User's choice:** Fresh-train one, save 3 ways

### Q2 — Legacy-binary generation toolchain

| Option | Description | Selected |
|--------|-------------|----------|
| Pin old xgboost for gen only | Pin an older xgboost that writes legacy binary in the generation script only; never runtime/CI | ✓ |
| Spike to confirm first | Research/spike tests whether pinned xgboost emits legacy binary, then selects fallback | |
| Record generation manifest | Freeze exact generator versions in a fixture manifest (D-07 discipline) | |

**User's choice:** Pin old xgboost for gen only
**Notes:** Generator-manifest discipline (D-07) is retained regardless; folded into CONTEXT D-06.

---

## Legacy binary & auto-detect

### Q1 — Auto-detect scope

| Option | Description | Selected |
|--------|-------------|----------|
| Unified 3-way sniff | Port JSON-vs-UBJSON heuristic + add leading-byte branch for legacy; one entry detects all three | |
| Mirror upstream's split | Port DetectXGBoostFormat (JSON/UBJSON only); legacy binary is a separate explicit loader entry | ✓ |
| Research picks | Lock that any of three formats must load; research chooses sniff strategy | |

**User's choice:** Mirror upstream's split
**Notes:** Criterion-2 "auto-detect which format" satisfied at JSON-vs-UBJSON level; legacy via explicit entry — recorded as a deliberate fidelity interpretation for the verifier (CONTEXT D-09).

### Q2 — Legacy decoder mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Hand-rolled LE cursor | from_le_bytes byte-cursor + peekable reader mirroring PeekableInputStream; zero deps | ✓ |
| byteorder crate | Standard byteorder crate; one more pinned dep | |
| Research picks | Lock no-transmute / explicit-LE; research chooses implementation | |

**User's choice:** Hand-rolled LE cursor

---

## DEF-02-01 byte-fidelity scope

### Q1 — Loader→serialize byte-fidelity disposition

| Option | Description | Selected |
|--------|-------------|----------|
| Close on verify-narrow fixture | Byte-fidelity on the narrow verified fixture only | |
| Close across all 3 formats | All three loaders → identical Model → identical v5 bytes == single upstream golden blob | ✓ |
| Keep deferred | Verify loaders via 1e-5 prediction only; push byte-fidelity later | |

**User's choice:** Close across all 3 formats
**Notes:** Brittleness risk (column ordering / bookkeeping, not float formatting) flagged for research; builds on 02-06 AllocNode groundwork.

---

## Claude's Discretion

- Module/file layout within `crates/treelite-xgboost`.
- `XgbError` additions for new formats (idiomatic thiserror, transparent builder propagation).
- Peekable-reader / byte-cursor internal representation.
- Full objective→postprocessor mapping table extent for XGB-05.
- Whether DART `weight_drop` leaf-scaling is applied at parse time (no verify-narrow fixture yet).

## Deferred Ideas

- Categorical-split & multiclass PREDICTION parity → Phase 5 (GTIL widening).
- DART `weight_drop` verified prediction → later (no verify-narrow fixture this phase).
- Vector `base_score` verified prediction → aligns with Phase 5 multi-output.
- NaN/Inf and UBJSON-decode concrete mechanisms → research phase (D-02, D-03).

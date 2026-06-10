# Phase 3: Full XGBoost Loaders - Research

**Researched:** 2026-06-10
**Domain:** XGBoost model deserialization (JSON / UBJSON / legacy-binary) → `treelite_core::Model`, porting C++ Treelite v4.7.0
**Confidence:** HIGH (the porting source of truth is vendored read-only at `treelite-mainline/`; all format layouts validated empirically against the local toolchain)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 (shared numeric path):** JSON and UBJSON converge at the existing serde structs. UBJSON is decoded into the same `serde_json::Value` / typed `XgbModelJson` structs the Phase-1 JSON path uses, so both formats share all downstream numeric coercion. The convergence point is the serde structs (our analog of upstream's shared `DelegatedHandler`).
- **D-02 (NaN/Inf — requirement LOCKED, mechanism deferred to research):** Bare `NaN`/`Infinity`/`-Infinity` literals MUST round-trip into f32 thresholds/leaf values, **value-position only, never inside string contents**. Mechanism chosen by this research.
- **D-03 (UBJSON decode mechanism — deferred to research):** Locked invariant: converge at the same serde structs (D-01). Research chooses hand-rolled type-tag decoder vs a UBJSON serde crate.
- **D-04 (parse-wide / verify-narrow — KEY BOUNDARY):** PARSE the full XGBoost structure faithfully (multiclass grouping, categorical split fields, vector `base_score`, DART `weight_drop`) so structs are future-proof. VERIFY 1e-5 only on fixtures today's scalar GTIL can predict (numerical splits, identity/sigmoid). Categorical/multiclass *prediction* parity lands in Phase 5.
- **D-05 (one logical model, three formats):** Fresh-train one small `binary:logistic` model in a single xgboost session, save it in all three formats from that one session. Capture the shared prediction golden from the upstream Treelite Python wheel. Keep numerical splits only so today's scalar GTIL verifies it.
- **D-06 (legacy-binary generation toolchain):** Pin an older xgboost that still WRITES legacy binary in the fixture-generation script only — never a runtime/CI dep. Freeze the generator manifest beside the golden.
- **D-07 (legacy decoder mechanism):** Hand-rolled little-endian byte-cursor using `from_le_bytes`, plus a peekable reader mirroring upstream's `PeekableInputStream` (1024-byte window). Zero external dependencies.
- **D-08 (NO native-endian transmute):** Legacy fields read field-by-field via explicit LE conversions. Never `transmute` a byte buffer onto a native struct. Hard criterion-2 invariant.
- **D-09 (auto-detect scope):** Port `DetectXGBoostFormat` exactly — JSON-vs-UBJSON only, returning "json"/"ubjson"/"unknown". Legacy binary is reached through a separate explicit loader entry point (matching upstream's API split).
- **D-10 (close DEF-02-01 across all three formats):** Load the fresh model in JSON, UBJSON, AND legacy binary; serialize each to v5; byte-compare against the SINGLE upstream-Treelite-wheel-serialized v5 golden blob. All three loaders → identical `Model` → identical v5 bytes == upstream's v5 bytes.

### Claude's Discretion

- Exact module/file layout within `crates/treelite-xgboost` (e.g. `legacy.rs`, `ubjson.rs`, `detect.rs`, widened `json` structs).
- Error-enum additions to `XgbError` for the new formats (idiomatic `thiserror`, transparent builder propagation).
- Internal representation of the peekable reader / byte cursor.
- Exact full objective→postprocessor mapping table extent for XGB-05 (port upstream's map; verify-narrow fixture exercises sigmoid).
- Whether DART `weight_drop` leaf-scaling is applied at parse time given it has no verify-narrow fixture yet.

### Deferred Ideas (OUT OF SCOPE)

- Categorical-split & multiclass PREDICTION parity (parsed now, verified in Phase 5).
- DART `weight_drop` verified prediction (parsed parse-wide; no verify-narrow fixture this phase).
- Vector `base_score` verified prediction (transform handled in XGB-05; multi-output fixture aligns with Phase 5).
- NaN/Inf & UBJSON-decode concrete mechanisms were deferred to this research (D-02, D-03) — now resolved below.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| XGB-01 | User can load an XGBoost JSON model | Widen Phase-1 serde structs to the full recognized key set (§Standard Stack, §Pattern 1); upstream key authority = `delegated_handler.cc` `is_recognized_key` methods enumerated in §Recognized Key Map |
| XGB-02 | User can load an XGBoost UBJSON model (parser shares the JSON state machine for numeric parity) | Hand-rolled UBJSON type-tag decoder → `serde_json::Value` → same `XgbModelJson` structs (§Pattern 2, D-03 resolution); exact tag set validated empirically (§UBJSON Type-Tag Map) |
| XGB-03 | User can load an XGBoost legacy binary model (little-endian layout) | Byte-exact legacy layout map validated against `mushroom.model` (§Legacy Binary Layout); `from_le_bytes` cursor + peekable reader (D-07/D-08) |
| XGB-04 | The loader auto-detects which XGBoost format a file uses | Port `DetectXGBoostFormat` first/second-char heuristic verbatim (§Pattern 3, D-09) |
| XGB-05 | XGBoost objective maps to the correct postprocessor, with the version-gated `base_score` margin transform applied | `objective.rs` map already ported; extend for scalar+vector `base_score` and the `version[0] >= 1` gate (§XGB-05 Mapping) |
</phase_requirements>

## Summary

This phase widens `crates/treelite-xgboost` from the Phase-1 minimal JSON loader to the full XGBoost surface across three on-disk formats, converging all of them at the existing typed serde structs so a single downstream build/validate/emit path serves every format (D-01). The upstream porting source — vendored read-only at `treelite-mainline/src/model_loader/` — is the authority for every recognized key, every binary offset, and the objective→postprocessor map; this research reads it line-by-line and validates the binary layout empirically against the only vendored fixture (`mushroom.model`).

Three research-flagged decisions are now resolved with empirical backing. **D-02 (NaN/Inf):** the off-the-shelf "lenient JSON" crate `serde_json_lenient` was disqualified — its source explicitly errors on `+/- infinity` and its lenience covers only comments/trailing-commas/escapes, not non-finite literals. The recommended mechanism is a **zero-dependency string-safe pre-lexer** that rewrites bare `NaN`/`Infinity`/`-Infinity` in value position to sentinel strings, paired with a `deserialize_with` adapter that recovers them into f32 — validated end-to-end (string contents preserved, all three non-finite values recovered). **D-03 (UBJSON):** the only serde-UBJSON crate (`serde_ubjson`) is abandoned since 2017 with ~7K downloads; the recommendation is a **hand-rolled type-tag decoder** emitting `serde_json::Value`, against the exact 14-tag subset XGBoost emits (validated by decoding a live `.ubj` file). **D-06 (legacy generation):** empirically confirmed that current xgboost 3.2.0 **cannot write legacy binary** — `.model`/`.bin`/`.deprecated` extensions silently emit UBJSON — so the generator manifest must pin an old xgboost (recommend `1.7.6`) for the write step only.

The legacy-binary layout is the highest-fidelity-risk surface and is mapped to the byte: a 136-byte `LearnerModelParam`, length-prefixed objective/booster names, a 168-byte `GBTreeModelParam`, then per-tree a 148-byte `TreeParam` + `num_nodes × 20`-byte nodes + `num_nodes × 16`-byte stats, followed by a `num_trees × 4`-byte `tree_info` tail. This was verified by parsing `mushroom.model` end-to-end (1501 bytes consumed exactly). DEF-02-01 (D-10) is achievable because Phase 2's `end_tree` already emits upstream's exact AllocNode column layout; the binding risk is column presence/ordering and bookkeeping-scalar emission (NOT float formatting — v5 stores raw float bytes), so the loader must populate `sum_hess`/`gain` and leave `data_count` empty exactly as upstream's loaders do.

**Primary recommendation:** Add zero runtime dependencies. Implement D-02 via a string-safe pre-lex + custom f32 deserializer over the existing `serde_json` front-end; D-03 via a hand-rolled UBJSON tag decoder → `serde_json::Value`; D-07 via a `from_le_bytes` cursor + 1024-byte peekable reader. Converge JSON/UBJSON/legacy at one `build_model_from_parsed` path emitting through the existing `ModelBuilder`. Pin xgboost `1.7.6` + treelite `4.7.0` in the generation-only manifest; freeze one prediction golden and one v5 byte golden.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Format auto-detection (JSON vs UBJSON) | `treelite-xgboost` (loader) | — | First/second-byte heuristic; pure string inspection, no model construction (D-09) |
| JSON tokenizing + NaN/Inf tolerance | `treelite-xgboost` (loader) | `serde_json` (front-end) | Pre-lex wraps the existing serde_json front-end; no new crate (D-02) |
| UBJSON byte decoding | `treelite-xgboost` (loader) | — | Hand-rolled tag decoder → `serde_json::Value`; zero deps (D-03) |
| Legacy binary byte decoding | `treelite-xgboost` (loader) | — | `from_le_bytes` cursor + peekable reader; zero deps (D-07/D-08) |
| Typed model structs (parse-wide) | `treelite-xgboost` (loader) | — | Single `XgbModelJson` struct family all three formats converge on (D-01) |
| Objective→postprocessor + base_score transform | `treelite-xgboost::objective` | — | Already ported; extend for vector base_score + version gate (XGB-05) |
| Node/tree topology validation + Tree column emission | `treelite-builder::ModelBuilder` | — | Loader emits through builder; builder owns AllocNode columns (CR-01/CR-02, D-10) |
| v5 serialization (byte-fidelity target) | `treelite-core` serializer | `treelite-harness` (golden) | Loader produces `Model`; serializer is already byte-perfect (Phase 2 round-trip) |
| 1e-5 prediction verification | `treelite-gtil` (scalar slice) + `treelite-harness` | — | Verify-narrow: only numerical-split sigmoid/identity (D-04); full GTIL is Phase 5 |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `serde` | 1.0.228 (workspace-pinned) | Derive the `XgbModelJson` struct family | Already the convergence point; `deserialize_with` hook implements the D-02 adapter |
| `serde_json` | 1.0.150 (workspace-pinned) | JSON front-end; `serde_json::Value` is the UBJSON decode target | Already in the crate; UBJSON converges here (D-01/D-03) |
| `thiserror` | 2.0.18 (workspace-pinned) | `XgbError` new variants for the new formats | Project ERR-01 constraint; already used |

**No new runtime dependencies are recommended.** `from_le_bytes` (std), a hand-rolled pre-lexer (std `str`/`u8`), and a hand-rolled UBJSON decoder (std) cover D-02, D-03, D-07 with zero crates. This is consistent with the CONTEXT rejections of `byteorder` (D-07) and a standalone UBJSON decoder drift risk (D-03), and with the project constraint "all crates pinned to latest published versions" (fewer pins = less surface).

### Supporting (generation-only, NOT runtime/CI dependencies)

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| `xgboost` (Python) | `1.7.6` `[ASSUMED]` | WRITE the legacy-binary fixture | Fixture-generation script only, run once, frozen (D-06) |
| `xgboost` (Python) | `3.2.0` `[VERIFIED: local venv]` | WRITE the JSON + UBJSON fixtures (same logical model) | Fixture-generation script only |
| `treelite` (Python wheel) | `4.7.0` `[VERIFIED: local venv]` | Capture the shared prediction golden + the single v5 byte golden | Fixture-generation script only (D-05/D-10) |
| `numpy` | `>=2.4.6` `[VERIFIED: pyproject.toml]` | Build the input matrix for the prediction golden | Generation script |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| String-safe pre-lex (D-02) | `serde_json_lenient` 0.2.4 | **DISQUALIFIED** — does not parse NaN/Infinity at all; source line `de.rs:972` errors on `+/- infinity`; lenience is comments/trailing-commas/escapes only. Earlier crates.io blurb ("handling NaN and Infinity") refers to `Value` *serialization* (→null), not input parsing. |
| Hand-rolled UBJSON decoder (D-03) | `serde_ubjson` 0.2.1 | **REJECTED** — abandoned since 2017-08-31, ~6.9K total downloads, no recent maintenance; two numeric paths risk drift vs the JSON path (works against criterion 2). |
| Hand-rolled UBJSON decoder (D-03) | `ubjson` 0.1.0 (nom-based) | **REJECTED** — last updated 2021-01, ~1.7K downloads / 9 recent; nom adds a heavy transitive parser dep for a 14-tag subset; not a serde data-format (would still need conversion). |
| `from_le_bytes` cursor (D-07) | `byteorder` crate | REJECTED in CONTEXT — extra pin for boilerplate `from_le_bytes` already covers. |
| `from_le_bytes` cursor (D-07) | `binread`/`nom` parser crate | REJECTED in CONTEXT — heavier than needed for a fixed-layout header. |

**Installation:** No `cargo add`. Generation-only Python tools (run once, never committed as deps):
```bash
# In a THROWAWAY environment — never the project venv, never CI:
uv run --with 'xgboost==1.7.6' --with 'treelite==4.7.0' --with numpy python fixtures/generate_xgb_3format.py
```

**Version verification performed:**
- `xgboost 3.2.0` and `treelite 4.7.0` confirmed installed in the project venv via `uv run python -c "import xgboost; import treelite"`.
- `serde_json_lenient` latest is `0.2.4` (not yanked) — confirmed via crates.io index. (Disqualified, not used.)
- xgboost `1.7.6` is `[ASSUMED]` as the legacy-binary-writing version — see Assumptions Log A1; the generation spike MUST confirm it before locking.

## Package Legitimacy Audit

> All runtime additions are zero. The only externally-sourced artifacts are generation-only Python tools, vetted by metadata below. slopcheck has no cargo mode in this environment; the runtime crate set is unchanged from Phase 2 (all already vetted), so no new runtime legitimacy risk is introduced.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `serde_json_lenient` | crates.io | updated 2024-12-10 | 895K total / 321K recent | github.com/google/serde_json_lenient | n/a (cargo) | Evaluated then REJECTED (no NaN/Inf parse support) — not installed |
| `serde_ubjson` | crates.io | last 2017-08-31 | 6.9K / 3.2K | github.com/milesand/serde-ubjson | n/a (cargo) | REJECTED (abandoned) — not installed |
| `ubjson` | crates.io | last 2021-01-28 | 1.7K / 9 | github.com/OtaK/ubjson | n/a (cargo) | REJECTED (abandoned, nom-heavy) — not installed |
| `xgboost` (PyPI) | PyPI | mature | very high | github.com/dmlc/xgboost | OK (well-known) | Generation-only; pin 1.7.6 (write) + 3.2.0 (write) |
| `treelite` (PyPI) | PyPI | mature | high | github.com/dmlc/treelite | OK (well-known) | Generation-only; 4.7.0 (golden capture) |

**Packages removed due to slopcheck [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none. The two rejected UBJSON crates are low-download but not slop — they are real, just abandoned; we decline them on maintenance/drift grounds, not legitimacy.

*No new runtime crate is added, so no `checkpoint:human-verify` gate is required for runtime deps. The xgboost-1.7.6 generation pin is `[ASSUMED]` (A1) and the generation spike doubles as its human-verify gate.*

## Architecture Patterns

### System Architecture Diagram

```
                        ┌─────────────────────────────┐
   model file bytes ───▶│  DetectXGBoostFormat (D-09)  │  (JSON vs UBJSON only;
                        │  first/second byte heuristic │   legacy = explicit entry)
                        └──────────────┬───────────────┘
            ┌──────────────────────────┼───────────────────────────┐
            ▼ "json"                   ▼ "ubjson"                   ▼ (explicit)
   ┌──────────────────┐      ┌──────────────────────┐     ┌────────────────────────┐
   │ pre-lex NaN/Inf  │      │ UBJSON tag decoder   │     │ legacy LE byte cursor  │
   │ (string-safe)    │      │ → serde_json::Value  │     │ + PeekableReader(1024) │
   │ → serde_json     │      │ (D-03)               │     │ (D-07/D-08)            │
   │ + de_f32 adapter │      └──────────┬───────────┘     └───────────┬────────────┘
   │ (D-02)           │                 │                             │
   └────────┬─────────┘                 │                             │
            │        ┌──────────────────▼─────────┐                   │
            └───────▶│  XgbModelJson struct family │◀──────────────── │ (legacy fills the
                     │  (parse-wide, D-01/D-04)    │   same logical fields directly)
                     └──────────────┬──────────────┘
                                    ▼
              ┌──────────────────────────────────────────┐
              │ build_model_from_parsed:                 │
              │  • objective→postprocessor (XGB-05)      │
              │  • version-gated base_score→margin       │
              │  • per-tree: numerical_test / leaf_scalar│
              │    via ModelBuilder (sum_hess, gain)     │
              └──────────────────┬───────────────────────┘
                                 ▼
                      ┌──────────────────────┐
                      │ ModelBuilder         │  AllocNode columns (CR-01/CR-02)
                      │ .commit_model()      │  → identical Model for all 3 formats
                      └──────────┬───────────┘
                                 ▼
                ┌────────────────────────────────────────┐
                │ Model ──▶ serialize v5 ──▶ byte-compare │ (D-10: ALL THREE == one
                │          (treelite-core)   golden blob  │  upstream golden)
                │      ──▶ scalar GTIL ──▶ 1e-5 vs golden │ (D-04/D-05 verify-narrow)
                └────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/treelite-xgboost/src/
├── lib.rs           # public entry points + the shared build_model_from_parsed path
├── json.rs          # widened XgbModelJson struct family + pre-lex + de_f32 (D-01/D-02)
├── ubjson.rs        # hand-rolled tag decoder → serde_json::Value (D-03)
├── legacy.rs        # LE byte cursor + PeekableReader + ParseStream port (D-07/D-08)
├── detect.rs        # DetectXGBoostFormat first/second-byte heuristic (D-09)
├── objective.rs     # existing; extend for vector base_score (XGB-05)
└── error.rs         # existing XgbError; add Ubjson / Legacy / FormatDetect variants
```

### Pattern 1: Widen-then-converge serde structs (D-01)
**What:** Extend the Phase-1 `RegTreeJson`/`Learner` structs to the full recognized key set; all three formats deserialize into them.
**When to use:** Always — it is the single downstream path that makes UBJSON parity (criterion 2) and three-format `Model` equality (D-10) automatic.
**Recognized Key Map (from `delegated_handler.cc` `is_recognized_key` methods — port these EXACTLY):**

| Handler | Recognized keys | Source |
|---------|----------------|--------|
| `RegTreeHandler` | `loss_changes`, `sum_hessian`, `base_weights`, `leaf_weights`, `categories_segments`, `categories_sizes`, `categories_nodes`, `categories`, `leaf_child_counts`(ignored), `left_children`, `right_children`, `parents`, `split_indices`, `split_type`, `split_conditions`, `default_left`, `tree_param`, `id` | `delegated_handler.cc:484-491` |
| `TreeParamHandler` | `num_feature`, `num_nodes`, `size_leaf_vector`, `num_deleted`(ignored) | `:331-334` |
| `GBTreeModelHandler` | `trees`, `tree_info`, `gbtree_model_param`(ignored), `iteration_indptr`(ignored), `cats` | `:548-551` |
| `GradientBoosterHandler` | `name`, `model`, `gbtree`(dart nesting), `weight_drop` | `:721-723` |
| `ObjectiveHandler` | `name` + 9 ignored `*_param` keys | `:753-758` |
| `LearnerParamHandler` | `num_target`, `base_score`, `num_class`, `num_feature`, `boost_from_average` | `:781-784` |
| `LearnerHandler` | `learner_model_param`, `gradient_booster`, `objective`, `attributes`(ignored), `feature_names`(ignored), `feature_types`(ignored) | `:916-919` |
| `XGBoostModelHandler` | `version`, `learner`, `Config`(ignored), `Model` | `:963-965` |

> **Scalar-as-string note (verified by inspecting live JSON):** XGBoost-JSON stores numeric learner/tree scalars as JSON *strings* (`"num_feature":"127"`, `"base_score":"5E-1"`, `"num_nodes":"13"`). The Phase-1 code already deserializes these as `String` + `str::parse`. Parallel node arrays (`split_conditions`, `left_children`, …) are real JSON arrays of numbers. **In UBJSON these same scalars are also strings** (`S` tag), so the same `String`+parse logic applies after decode — do NOT special-case UBJSON numeric scalars.

**Parse-wide additions (D-04) the structs must carry but the scalar GTIL won't yet predict:** `split_type` (already present), `categories_segments/sizes/nodes/categories`, `leaf_weights` (vector-leaf path), `tree_info`+`num_class` (multiclass grouping), vector `base_score`, DART `weight_drop`. Build them into the structs now; gate their *use* behind the leaf-vector / categorical / multiclass branches so the verify-narrow fixture path is unchanged.

### Pattern 2: Hand-rolled UBJSON tag decoder → `serde_json::Value` (D-03)
**What:** A recursive-descent decoder over the UBJSON byte stream that emits `serde_json::Value`, then the existing `serde_json::from_value::<XgbModelJson>` (with the D-02 adapter on float arrays) finishes the job.
**When to use:** For the UBJSON entry point only. Converging at `Value` (not at a second struct path) satisfies the D-01 "same structs" invariant.
**UBJSON Type-Tag Map (validated empirically by decoding a live xgboost `.ubj` — markers observed: `{ } [ ] $ # S C U i I l L d D`):**

| Tag | Meaning | Decode to `serde_json::Value` |
|-----|---------|-------------------------------|
| `Z` | null | `Value::Null` |
| `T` / `F` | true / false | `Value::Bool` |
| `i` | int8 | `Value::Number` (i64) |
| `U` | uint8 | `Value::Number` (u64) |
| `I` | int16 | `Value::Number` (i64) |
| `l` | int32 | `Value::Number` (i64) |
| `L` | int64 | `Value::Number` (i64) |
| `d` | float32 | `Value::Number` (f64 — **see NaN/Inf note**) |
| `D` | float64 | `Value::Number` (f64) |
| `C` | char (1 byte) | `Value::String` (1-char) |
| `S` | string: length-tag + UTF-8 bytes | `Value::String` |
| `[` … `]` | array (may be `$`type + `#`count optimized) | `Value::Array` |
| `{` … `}` | object (may be `$`/`#` optimized) | `Value::Object` |
| `$` | (inside container) element type marker | drives typed-array fast path |
| `#` | (inside container) count marker | drives typed-array fast path |
| `N` | no-op | skip (this is the byte D-09 keys on for UBJSON detection) |

**Optimized containers (XGBoost emits these heavily — `$`=44, `#`=51 in the sampled file):** `[$<type>#<count>` means "array of `count` elements all of `type`, with element tags omitted." The decoder MUST handle this strongly-typed-container form (e.g. `[$d#l<int32 count>` = float32 array) — this is how `split_conditions` etc. are stored compactly. A naive "every element has a tag" decoder will mis-parse. nlohmann's `sax_parse(..., input_format_t::ubjson)` handles this; the hand-rolled port must too.

**UBJSON NaN/Inf note:** Because UBJSON stores floats as raw IEEE-754 bytes (`d`/`D`), a NaN/Inf value arrives as actual `f32::NAN`/`INFINITY` *before* it ever becomes a `serde_json::Value`. `serde_json::Value::Number` cannot hold non-finite f64 (it maps to `Null` on construction). **Therefore the UBJSON path needs the SAME sentinel treatment**: when a decoded `d`/`D` value is non-finite, emit the sentinel `Value::String` (`"@NaN@"`/`"@Inf@"`/`"@-Inf@"`) instead of a Number, so the shared `de_f32` adapter recovers it identically to the JSON path. This is the concrete realization of criterion-2 "UBJSON shares the JSON numeric state machine."

### Pattern 3: `DetectXGBoostFormat` heuristic (D-09 — port `detail/xgboost.cc:83-115` verbatim)
**What:** Read the first 2 bytes; classify JSON vs UBJSON vs unknown.
**Exact logic (no interpretation — copy it):**
```
read buf[0..2]
is_space(c) = c in { ' ', '\n', '\r', '\t' }
if buf[0] == 'N'        -> "ubjson"   // UBJSON no-op code; never in JSON
else if is_space(buf[0])-> "json"     // whitespace only in JSON
else if buf[0] != '{'   -> "unknown"
// buf[0] == '{'; look at buf[1]:
if is_space(buf[1]) || buf[1] == '"'              -> "json"
else if buf[1] in { 'N','$','#','i','U','I','l','L' } -> "ubjson"
else                                               -> "unknown"
```
> Validated against the live UBJSON file whose first two bytes are `7B 4C` = `'{' 'L'` → `buf[1]=='L'` → "ubjson". Correct.

### Pattern 4: Legacy `ParseStream` port (D-07/D-08 — see §Legacy Binary Layout for the byte map)
**What:** A `from_le_bytes`-based cursor reads the fixed-layout header and per-tree structures field by field, plus a `PeekableReader` (1024-byte window) for the `binf` magic peek.
**When to use:** The legacy entry point only.
**Default-direction nuance (from `xgboost_legacy.cc:211-218`):** `sindex` packs `default_left` in the top bit: `split_index = sindex & 0x7FFFFFFF`, `default_left = (sindex >> 31) != 0`. A leaf is `cleft == -1`. The leaf's value lives in the `info` union (reinterpret the same 4 bytes as f32). Port these bit ops exactly.

### Anti-Patterns to Avoid
- **Replacing the working Phase-1 JSON loader with a SAX port** — CONTEXT D-01 explicitly rejects porting upstream's `DelegatedHandler`; converge at serde structs instead.
- **Pre-lexing `Infinity` → `1e400` (a huge number literal)** — serde_json rejects out-of-range numbers ("number out of range"); validated empirically. Use a sentinel STRING for all three non-finite tokens.
- **A second independent UBJSON numeric path** — drift risk vs the JSON path (works against criterion 2). Converge at `serde_json::Value` + the shared `de_f32` adapter.
- **`transmute`/`bytemuck::cast` of a byte slice onto a `#[repr(C)]` LearnerModelParam struct** — forbidden by D-08 (native-endian + padding hazards). Read field by field.
- **Treating UBJSON numeric scalars as numbers** — like JSON, XGBoost stores learner scalars as UBJSON strings (`S` tag); keep the `String`+parse path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON tokenizing | A full JSON tokenizer | `serde_json` + a thin pre-lex pass | The pre-lex is ~30 lines over a byte scanner; the actual JSON grammar stays serde_json's job |
| f32 array deserialization | Manual `Value` walking | `#[serde(deserialize_with = de_vec_f32)]` | serde already drives the array; the adapter only intercepts the sentinel strings |
| Tree topology validation (orphans, dangling children) | Re-validate in the loader | `ModelBuilder` (already does it) | Phase 2 builder ports upstream's orphan/dangling checks; loader just emits |
| v5 serialization | Any loader-side serialization | `treelite_core::serialize_to_buffer` | Already byte-perfect (Phase 2 golden round-trip) |
| LE integer/float decode | A `byteorder` dependency | `u32::from_le_bytes` etc. (std) | std covers every width the header uses (D-07) |

**Key insight:** The *one* thing genuinely worth hand-rolling here is the UBJSON tag decoder — not because decoding is hard, but because every published Rust UBJSON crate is abandoned, and a 14-tag recursive decoder emitting `serde_json::Value` is both small and the only way to honor the D-01 "same structs" invariant without taking on dead-dependency risk.

## Legacy Binary Layout (D-07/D-08 — the from_le_bytes cursor spec)

> **Empirically validated** by parsing `treelite-mainline/tests/examples/mushroom/mushroom.model` (1501 bytes) end-to-end: every field below decoded to sane values and the cursor consumed exactly 1501 bytes (1493 after the two trees + 8-byte `tree_info` tail = 1501). All multi-byte values are **little-endian**. `[VERIFIED: parsed mushroom.model]`

**Read order (top to bottom):**

1. **Optional magic peek (PeekableReader, 1024-byte window):** peek 4 bytes.
   - If `== "bs64"` → ERROR (base64 no longer supported).
   - If `== "binf"` → consume 4 bytes. Otherwise do NOT consume (mushroom has NO magic: first byte `0x00`). Source: `xgboost_legacy.cc:328-336`.
2. **`LearnerModelParam` — exactly 136 bytes** (`static_assert(sizeof==136)`), field-by-field:

   | Field | Type | Bytes | Notes |
   |-------|------|-------|-------|
   | `base_score` | f32 | 4 | global bias (mushroom: `-0.0`) |
   | `num_feature` | u32 | 4 | mushroom: 127 |
   | `num_class` | i32 | 4 | mushroom: 0 → `max(num_class,1)`=1 |
   | `contain_extra_attrs` | i32 | 4 | |
   | `contain_eval_metrics` | i32 | 4 | |
   | `major_version` | u32 | 4 | **gates base_score transform** (mushroom: 0 → NO transform) |
   | `minor_version` | u32 | 4 | |
   | `num_target` | u32 | 4 | 0 → treated as 1 |
   | `pad2[26]` | i32×26 | 104 | reserved; skip |

3. **Objective name:** `u64` length (8 bytes) + that many UTF-8 bytes. (mushroom: len 15 = `"binary:logistic"`). Source: `:340-349`.
4. **Booster name:** `u64` length (8 bytes) + bytes. (mushroom: len 6 = `"gbtree"`). MUST be `"gbtree"` or `"dart"`. Source: `:351-360`.
5. **`GBTreeModelParam` — 168 bytes**, field-by-field:

   | Field | Type | Bytes | Notes |
   |-------|------|-------|-------|
   | `num_trees` | i32 | 4 | mushroom: 2 |
   | `num_roots` | i32 | 4 | mushroom: 1 |
   | `num_feature` | i32 | 4 | mushroom: 127 |
   | `pad1` | i32 | 4 | |
   | `pad2` | i64 | 8 | |
   | `num_output_group` | i32 | 4 | mushroom: 1 |
   | `size_leaf_vector` | i32 | 4 | mushroom: 0 |
   | `pad3[32]` | i32×32 | 128 | reserved; skip |

6. **Per tree** (`num_trees` times), each via `XGBTree::Load` (`:288-315`):
   - **`TreeParam` — 148 bytes (37 × i32):** `num_roots`, `num_nodes`, `num_deleted`, `max_depth`, `num_feature`, `size_leaf_vector`, then `reserved[31]`. (mushroom tree 0: num_nodes=13; tree 1: num_nodes=11.) `num_nodes` MUST be > 0.
   - **Nodes: `num_nodes × 20 bytes`**, each `Node` (`static_assert(sizeof==20)`):

     | Field | Type | Bytes | Notes |
     |-------|------|-------|-------|
     | `parent` | i32 | 4 | top bit = is-left-child flag; mask with `0x7FFFFFFF` for parent id |
     | `cleft` | i32 | 4 | `-1` ⇒ leaf |
     | `cright` | i32 | 4 | |
     | `sindex` | u32 | 4 | `split_index = sindex & 0x7FFFFFFF`; `default_left = (sindex>>31)!=0` |
     | `info` | union f32/f32 | 4 | leaf: `leaf_value`; internal: `split_cond` (same 4 bytes, reinterpret as f32) |

   - **Stats: `num_nodes × 16 bytes`**, each `NodeStat` (`sizeof==16`):

     | Field | Type | Bytes | Notes |
     |-------|------|-------|-------|
     | `loss_chg` | f32 | 4 | → builder `Gain` (internal nodes only) |
     | `sum_hess` | f32 | 4 | → builder `SumHess` (all nodes) |
     | `base_weight` | f32 | 4 | (not emitted) |
     | `leaf_child_cnt` | i32 | 4 | (not emitted) |

   - **Leaf-vector tail (conditional):** if `param.size_leaf_vector != 0 && major_version < 2`: read `u64 len`, then if `len>0` consume `len × 4` bytes (discarded — scalar-only legacy). If `major_version == 2`: assert `size_leaf_vector == 1`. (mushroom: size_leaf_vector=0, so skip.) Source: `:300-311`.
   - Assert `param.num_roots == 1`.

7. **`tree_info`:** `num_trees × i32` (mushroom: 2×4 = the final 8 bytes). Source: `:379-384`.
8. **DART `weight_drop` (only if booster == "dart"):** `u64 sz` (must equal num_trees), then `sz × f32`. Folded into leaf values at build time: `leaf *= weight_drop[tree_id]`. Source: `:386-397, 478-482`.

**Struct-size constants to encode as asserts in the Rust port:** `LearnerModelParam=136`, `GBTreeModelParam=168`, `TreeParam=148`, `Node=20`, `NodeStat=16`. These let the cursor advance by fixed strides and fail loudly on a malformed file.

**Metadata mapping (`xgboost_legacy.cc:399-465`):** `num_class = max(mparam.num_class, 1)`; `num_target = (mparam.num_target==0 ? 1 : mparam.num_target)`; `leaf_vector_shape = {1,1}`; task_type from `num_class>1 ? kMultiClf` else by objective-name prefix (`binary:`→kBinaryClf, `rank:`→kLearningToRank, else kRegressor). `base_scores` length = `num_target * num_class`, all = the (possibly transformed) scalar base_score. `average_tree_output = false`.

## XGB-05 Mapping (objective → postprocessor + base_score transform)

**Objective→postprocessor table (already ported verbatim in `objective.rs`; confirm unchanged):** Source `detail/xgboost.cc:28-50`.

| Objectives | Postprocessor |
|------------|---------------|
| `multi:softmax`, `multi:softprob` | `softmax` |
| `reg:logistic`, `binary:logistic` | `sigmoid` |
| `count:poisson`, `reg:gamma`, `reg:tweedie`, `survival:cox`, `survival:aft` | `exponential` |
| `binary:hinge` | `hinge` |
| `reg:squarederror`, `reg:linear`, `reg:squaredlogerror`, `reg:pseudohubererror`, `binary:logitraw`, `rank:pairwise`, `rank:ndcg`, `rank:map` | `identity` |
| anything else | ERROR (`XgbError::UnrecognizedObjective`) |

**Base_score transform (`TransformBaseScoreToMargin`, `xgboost.cc:52-60` + `ProbToMargin`, `xgboost.h:16-23`):**
- `sigmoid` → `-ln(1/p - 1)` (f64)
- `exponential` → `ln(p)` (f64)
- everything else → identity (pass through)

**Version gate (`delegated_handler.cc:893` / `xgboost_legacy.cc:456`):** apply the transform iff `version.empty() || version[0] >= 1` (JSON) / `major_version >= 1` (legacy). The Phase-1 code already implements this for JSON. **CRITICAL:** the transform stays in **f64** end-to-end (it is the #1 silent 1e-5 break — already documented in `objective.rs`). mushroom (major_version=0) is the negative-gate case: NO transform.

**Scalar vs vector base_score (NEW for Phase 3 — `delegated_handler.cc:877-889` + `ParseBaseScore` `xgboost.cc:62-79`):**
- **Scalar (XGBoost <3.1):** `base_score` is a JSON string like `"5E-1"` → `vec![parse]`, then filled across `num_target*num_class` entries.
- **Vector (XGBoost 3.1+):** `base_score` is a JSON string that *starts with `[`* → parse it as a nested JSON array of floats (upstream re-parses the string with `kParseNanAndInfFlag`). Its length must equal `num_target*num_class`. Each element f32→f64, then the version-gated transform applies element-wise.
- The transform is applied to EVERY element of `base_scores` after expansion (D-04: handle both forms even though the verify-narrow fixture is scalar).

## DEF-02-01 / D-10: loader→serialize byte-fidelity

**What must hold:** all three loaders → identical `Model` → identical v5 bytes == the single upstream-wheel v5 golden.

**Why it's now achievable (it wasn't in Phase 1):** Phase 1's loader left stat/CSR columns empty (the documented gap that made `golden_v5.rs::loader_path_divergence_diagnostic` non-fatal). Phase 2's `end_tree` (02-06) now emits upstream's exact AllocNode column layout: the five per-node CSR/category columns at length `num_nodes` (CR-01) and empty-unless-set stat columns (CR-02). So the remaining work is in the **loader**, not the builder.

**Exact columns/scalars the loader must populate to byte-match (binding risk is presence/ordering, NOT float formatting — v5 stores raw float bytes):**

| Item | What upstream's loader sets | What the Rust loader must do |
|------|----------------------------|------------------------------|
| `sum_hess` (per node) | set on EVERY node from `sum_hessian` (JSON) / `NodeStat.sum_hess` (legacy) | call `builder.sum_hess(...)` for every node → triggers CR-02 `any_sum_hess` → column emitted at length num_nodes |
| `gain` (per node) | set on internal nodes from `loss_changes` / `NodeStat.loss_chg` | call `builder.gain(...)` on internal nodes → CR-02 emits the gain column |
| `data_count` | upstream does NOT set it | do NOT call `builder.data_count(...)` → CR-02 leaves it empty (matches upstream) |
| leaf `split_index` | `-1` for leaves | leaf nodes leave split_index at the builder default `-1` |
| `attributes` | upstream stamps `"{}"` | the loader currently passes `Some(String::new())` (empty) — **change to `None`** so `commit_model` defaults it to `"{}"` to match upstream's serialized attributes |
| per-node CSR/category columns | emitted at length num_nodes, all-zero/false | already handled by builder CR-01 — no loader action |

> **Phase-1 fidelity gap to close:** the current `load_xgboost_json` sets `attributes: Some(String::new())` and does NOT emit `sum_hess`/`gain`. Both diverge from the upstream golden. Phase 3 must (a) pass `attributes: None` (→ `"{}"`), and (b) emit `sum_hess` on every node and `gain` on internal nodes — exactly as `delegated_handler.cc:471-477` and `xgboost_legacy.cc:488-490` do. The fixture's `sum_hessian`/`loss_changes` arrays must be parsed (they're in the recognized key set) and threaded through.

**The single-golden invariant:** because all three formats are the SAME logical model (D-05) and the builder is deterministic, all three produce the same `Model`, hence the same v5 bytes. The test asserts `serialize(load_json(j)) == serialize(load_ubjson(u)) == serialize(load_legacy(b)) == golden_v5_3format.bin`. This one assertion simultaneously proves three-loader equality AND upstream-serialization match.

## Runtime State Inventory

> This phase is **additive loader work**, not a rename/refactor/migration. No stored data, live service config, OS-registered state, secrets, or pre-existing build artifacts embed a string this phase renames.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — verified by scope (no datastore; fixtures are flat files committed to the repo) | none |
| Live service config | None — verified (no external services; xgboost/treelite are generation-only Python tools) | none |
| OS-registered state | None — verified (no daemons/tasks; tests run via `cargo test`) | none |
| Secrets/env vars | None — verified (no secrets; `uv run python` venv is untracked but carries no phase secret) | none |
| Build artifacts | New fixtures only: `fixtures/*.json/.ubj/.model` + `*.bin` golden + manifest — created fresh, frozen, committed (D-05/D-06) | generate once, commit, never regenerate in CI |

## Common Pitfalls

### Pitfall 1: `Infinity` → numeric-literal substitution overflows serde_json
**What goes wrong:** Replacing `Infinity` with `1e400` produces "number out of range" — serde_json rejects non-finite-after-parse numbers.
**Why it happens:** serde_json validates numeric range during tokenization (validated empirically: `1e999` → "number out of range").
**How to avoid:** Replace ALL THREE non-finite tokens (`NaN`, `Infinity`, `-Infinity`) with sentinel STRINGS, recovered by the `de_f32` adapter. Never emit a numeric literal for them.
**Warning signs:** A test with an Inf split_condition fails to parse rather than producing `f32::INFINITY`.

### Pitfall 2: Pre-lex matching `NaN`/`Infinity` inside string contents
**What goes wrong:** A model attribute like `"feature_name":"NaN_count"` or an objective string gets corrupted.
**Why it happens:** A naive find-and-replace ignores JSON string state.
**How to avoid:** The pre-lexer MUST track in-string state (toggle on unescaped `"`, honor `\` escapes) and only substitute in value position. **Validated:** the reference scanner leaves `"NaN_in_string"` and `"has Infinity inside"` untouched.
**Warning signs:** Round-trip of a model with NaN-containing strings changes the string.

### Pitfall 3: f64→f32→f64 base_score precision (the silent 1e-5 break)
**What goes wrong:** Doing the sigmoid/exponential margin transform in f32 shifts the last ULPs past 1e-5.
**Why it happens:** XGBoost stores base_score as f32, but the transform and `Model.base_scores` are f64.
**How to avoid:** Keep the transform in f64 throughout (already enforced in `objective.rs`; `prob_to_margin_*` have zero f32 tokens). For vector base_score, cast each f32 element to f64 BEFORE transforming.
**Warning signs:** Prediction off by ~1e-6–1e-5 only when base_score != 0.5.

### Pitfall 4: UBJSON strongly-typed containers parsed as untyped
**What goes wrong:** `[$d#<count>` (typed float32 array) decoded as if every element carries its own tag → garbage / desync.
**Why it happens:** UBJSON's `$`/`#` optimization omits per-element tags; a decoder that always reads a tag per element mis-parses.
**How to avoid:** Implement the `$`type + `#`count fast path in the array/object decoder (observed `$`=44, `#`=51 in a tiny model — XGBoost uses it everywhere).
**Warning signs:** UBJSON load produces wrong array lengths or trailing-byte errors; JSON load of the same model is fine.

### Pitfall 5: UBJSON NaN/Inf bypasses the JSON sentinel path
**What goes wrong:** A non-finite f32 decoded from a `d` tag becomes `Value::Null` (serde_json can't hold non-finite), silently losing the value.
**Why it happens:** UBJSON floats are raw IEEE-754; they're already non-finite before reaching `Value`.
**How to avoid:** In the UBJSON decoder, when a decoded float is non-finite, emit the sentinel `Value::String` instead of a Number, so the shared `de_f32` adapter recovers it (criterion-2 numeric parity).
**Warning signs:** UBJSON and JSON loads of the same NaN-containing model disagree.

### Pitfall 6: Legacy `sindex` bit-unpacking
**What goes wrong:** Using `sindex` directly as the split index, or reading default_left from the wrong bit.
**Why it happens:** `sindex` packs `default_left` in bit 31.
**How to avoid:** `split_index = sindex & 0x7FFFFFFF`; `default_left = (sindex >> 31) != 0` (`xgboost_legacy.cc:214-219`). Leaf detection is `cleft == -1`, not `sindex`.
**Warning signs:** Legacy predictions diverge from JSON/UBJSON for the same model; wrong default-direction routing on missing values.

### Pitfall 7: Current xgboost silently writes UBJSON when you ask for legacy
**What goes wrong:** The generation script "saves a `.model`" expecting legacy binary but gets UBJSON.
**Why it happens:** **Verified:** xgboost 3.2.0 emits UBJSON for `.model`/`.bin`/`.deprecated` (with a warning), only honoring `.json`/`.ubj` explicitly.
**How to avoid:** Use a pinned OLD xgboost (recommend `1.7.6`) to write the legacy fixture; assert the first byte is NOT `{`/`N` and the header decodes as a 136-byte `LearnerModelParam`.
**Warning signs:** The "legacy" fixture starts with `{L` (UBJSON magic) instead of a `LearnerModelParam`.

## Code Examples

### D-02: string-safe pre-lex + custom f32 deserializer (validated end-to-end)
```rust
// Source: empirically validated in this research session (zero new deps).
// Rewrites bare NaN/Infinity/-Infinity in VALUE position to sentinel strings.
fn replace_nonfinite(input: &str) -> String {
    let b = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let (mut i, mut in_str, mut escaped) = (0usize, false, false);
    while i < b.len() {
        let c = b[i];
        if in_str {
            out.push(c as char);
            if escaped { escaped = false; }
            else if c == b'\\' { escaped = true; }
            else if c == b'"' { in_str = false; }
            i += 1; continue;
        }
        match c {
            b'"' => { in_str = true; out.push('"'); i += 1; }
            _ if input[i..].starts_with("-Infinity") => { out.push_str("\"@-Inf@\""); i += 9; }
            _ if input[i..].starts_with("Infinity")  => { out.push_str("\"@Inf@\"");  i += 8; }
            _ if input[i..].starts_with("NaN")        => { out.push_str("\"@NaN@\"");  i += 3; }
            _ => { out.push(c as char); i += 1; }
        }
    }
    out
}

fn de_f32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    use serde::de;
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = f32;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("f32 or a NaN/Inf sentinel")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f32, E> { Ok(v as f32) }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f32, E> {
            match v {
                "@NaN@" => Ok(f32::NAN),
                "@Inf@" => Ok(f32::INFINITY),
                "@-Inf@" => Ok(f32::NEG_INFINITY),
                other => other.parse().map_err(de::Error::custom),
            }
        }
    }
    d.deserialize_any(V)
}
// Applied as: #[serde(deserialize_with = "de_vec_f32")] split_conditions: Vec<f32>
// where de_vec_f32 wraps each element through de_f32 (see research session sketch).
```
> **Note:** the JSON path applies `replace_nonfinite` to the input string before `serde_json::from_str`. The UBJSON path instead emits the sentinel strings directly from the tag decoder when a `d`/`D` value is non-finite — both converge on the same `de_f32` adapter.

### D-09: DetectXGBoostFormat (port verbatim)
```rust
// Source: treelite-mainline/src/model_loader/detail/xgboost.cc:83-115
fn detect_xgboost_format(first_two: &[u8]) -> &'static str {
    let is_space = |c: u8| matches!(c, b' ' | b'\n' | b'\r' | b'\t');
    let (b0, b1) = (first_two.first().copied().unwrap_or(0),
                    first_two.get(1).copied().unwrap_or(0));
    if b0 == b'N' { return "ubjson"; }
    if is_space(b0) { return "json"; }
    if b0 != b'{' { return "unknown"; }
    if is_space(b1) || b1 == b'"' { return "json"; }
    if matches!(b1, b'N' | b'$' | b'#' | b'i' | b'U' | b'I' | b'l' | b'L') { return "ubjson"; }
    "unknown"
}
```

### D-07: legacy header decode (cursor pattern)
```rust
// Source: treelite-mainline/src/model_loader/xgboost_legacy.cc:154-167, 337-360
// Validated field offsets against mushroom.model.
struct Cursor<'a> { buf: &'a [u8], pos: usize }
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32, XgbError> {
        let b = self.buf.get(self.pos..self.pos+4).ok_or(/* Legacy truncation */)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, XgbError> { Ok(self.u32()? as i32) }
    fn f32(&mut self) -> Result<f32, XgbError> { Ok(f32::from_le_bytes(/* 4 bytes */)) }
    fn u64(&mut self) -> Result<u64, XgbError> { /* 8 bytes from_le_bytes */ }
    fn skip(&mut self, n: usize) { self.pos += n; }
    fn bytes(&mut self, n: usize) -> &'a [u8] { let s=&self.buf[self.pos..self.pos+n]; self.pos+=n; s }
}
// LearnerModelParam (136 bytes), then objective/booster (u64 len + bytes),
// then GBTreeModelParam (168 bytes), then per-tree TreeParam(148)+nodes(20·n)+stats(16·n),
// then tree_info (4·num_trees). sindex bit-unpacking on each node.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| XGBoost saves legacy binary by default | XGBoost saves UBJSON by default; legacy binary write **deprecated/removed** | xgboost ~1.0 (2020) deprecated; **3.x silently emits UBJSON for `.model`** (verified 3.2.0) | Must pin an OLD xgboost to WRITE the legacy fixture (D-06) |
| Scalar `base_score` (JSON string) | Vector `base_score` (JSON-array-in-a-string) | XGBoost 3.1 | XGB-05 must handle both forms (`ParseBaseScore` `xgboost.cc:62-79`) |
| Standalone XGBoost-JSON schema doc | Schema doc removed (structure unchanged) | XGBoost 3.2 (`saving_model.rst:L313`) | The vendored `delegated_handler.cc` key set remains the authority |

**Deprecated/outdated:**
- `serde_json_lenient` as a NaN/Inf solution — it does NOT parse non-finite literals (source-verified). Do not adopt it for D-02.
- Base64 (`bs64`) legacy model framing — upstream hard-errors on it (`xgboost_legacy.cc:331`); no need to support.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | xgboost **1.7.6** is a version that still WRITES legacy binary | Standard Stack / D-06 | If 1.7.x already refuses legacy write, the generation spike picks another (e.g. 1.6.x or 0.90). LOW risk: the spike is the gate, and any version that emits a non-`{`/non-`N` first byte + 136-byte LearnerModelParam works. Verified that 3.2.0 does NOT write it, so an older pin is definitely required. |
| A2 | The fresh `binary:logistic` model's `version` triple has `version[0] >= 1` so the sigmoid base_score transform fires (matching the existing fixture's `[4,7,0]`) | XGB-05 / D-05 | If the chosen old-xgboost stamps `major_version==0` in the legacy header (like mushroom), the legacy fixture would NOT transform while JSON/UBJSON (written by 3.2.0, version>=1) WOULD — breaking the single-golden invariant. The generation spike MUST verify all three formats yield the same transformed base_score, or set `base_score=0.5` (sigmoid(0.5)=0 margin, transform is a no-op) to sidestep the gate entirely. **Recommend base_score=0.5 for the fixture.** |
| A3 | XGBoost emits NaN/Inf only in float-array value positions (never as object keys or in strings) | D-02 | If NaN could appear as a key, the value-position pre-lex would still be safe (keys are quoted strings, untouched). LOW risk. |
| A4 | The hand-rolled UBJSON decoder's `$`/`#` typed-container handling matches nlohmann's exactly for the tags XGBoost emits | D-03 | Mis-handling desyncs the byte stream. MEDIUM risk — mitigated by the D-10 byte-fidelity test (UBJSON path must produce the same Model as JSON) which would catch any desync. |

## Open Questions

1. **Which exact old xgboost version writes legacy binary cleanly with `version[0]>=1`?**
   - What we know: 3.2.0 cannot write it; mushroom (an old model) has `major_version==0`.
   - What's unclear: the precise oldest/cleanest pin (1.7.6 assumed).
   - Recommendation: A generation spike (Wave 0) installs candidate versions in a throwaway env, writes a `binary:logistic` model with `base_score=0.5`, and asserts (a) legacy first byte is a `LearnerModelParam` not `{`, (b) all three formats round-trip to the same Treelite-4.7.0 v5 blob. Lock the version that passes. A2's `base_score=0.5` recommendation removes the transform-gate divergence risk.

2. **Does the fresh 3-format model need `tree_info`/multiclass coverage to exercise the parse-wide paths, given verify-narrow?**
   - What we know: D-04 says parse-wide, verify-narrow; the fixture is single-class binary:logistic.
   - What's unclear: whether to add a *separate* parse-only multiclass/categorical fixture now.
   - Recommendation: Keep ONE verified fixture (binary:logistic). Add parse-only unit tests for the categorical/multiclass/vector-base_score struct branches using small hand-crafted JSON snippets (no golden needed) so the parse-wide code is covered without a Phase-5 GTIL dependency.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `uv` + project venv | Fixture generation, golden capture | ✓ | uv 0.11.8, Python 3.13.13 | — |
| `xgboost` (current, write JSON/UBJSON) | D-05 fixture generation | ✓ | 3.2.0 | — |
| `treelite` wheel (golden capture) | D-05/D-10 goldens | ✓ | 4.7.0 | — |
| `numpy` | prediction golden input matrix | ✓ | >=2.4.6 | — |
| `xgboost` 1.7.6 (write legacy binary) | D-06 legacy fixture | ✗ | not installed | Install in throwaway env via `uv run --with`; A1 gate |
| `mushroom.model` (legacy smoke test) | XGB-03 independent parse check | ✓ | vendored, 1501 B | — |
| Rust toolchain (edition 2024) | all loader code | ✓ | workspace builds | — |

**Missing dependencies with no fallback:** none (the legacy-write xgboost is generation-only and installable on demand).
**Missing dependencies with fallback:** xgboost 1.7.6 — install ephemerally for the one-time generation spike; never a runtime/CI dep (D-06).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `cargo test`; `approx` 0.5.1 for 1e-5 float asserts; `anyhow` in tests (ERR-02) |
| Config file | none — standard `crates/*/tests/*.rs` integration tests + `#[cfg(test)]` units |
| Quick run command | `cargo test -p treelite-xgboost` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| XGB-01 | Load widened XGBoost-JSON | integration | `cargo test -p treelite-xgboost json_` | ❌ Wave 0 |
| XGB-02 | Load UBJSON → same Model as JSON | integration | `cargo test -p treelite-xgboost ubjson_` | ❌ Wave 0 |
| XGB-03 | Load legacy binary (mushroom smoke + fresh fixture) | integration | `cargo test -p treelite-xgboost legacy_` | ❌ Wave 0 |
| XGB-04 | DetectXGBoostFormat json/ubjson/unknown | unit | `cargo test -p treelite-xgboost detect_` | ❌ Wave 0 |
| XGB-05 | objective→postprocessor + scalar/vector base_score + version gate | unit | `cargo test -p treelite-xgboost objective_` | ⚠️ partial (objective.rs has units) |
| D-02 | NaN/Inf round-trip into f32, string-safe | unit | `cargo test -p treelite-xgboost nan_inf_` | ❌ Wave 0 |
| D-05 | 3 formats predict within 1e-5 of one golden | integration (harness) | `cargo test -p treelite-harness three_format_` | ❌ Wave 0 |
| D-10 | 3 loaders → identical v5 bytes == upstream golden | integration (harness) | `cargo test -p treelite-harness loader_byte_fidelity` | ❌ Wave 0 (extends golden_v5.rs) |

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-xgboost`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** Full suite green + the D-10 three-format byte-fidelity assertion (no longer a non-fatal diagnostic) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `fixtures/generate_xgb_3format.py` — one-session 3-format generator (D-05/D-06); covers the A1/A2 spike
- [ ] `fixtures/<name>.json`, `.ubj`, `.model` (fresh binary:logistic, base_score=0.5) + frozen golden vectors + 3-format manifest
- [ ] `fixtures/golden_v5_3format.bin` + manifest — the single upstream v5 blob for D-10
- [ ] `crates/treelite-xgboost/tests/json.rs` — covers XGB-01 (widened keys, sum_hess/gain emission)
- [ ] `crates/treelite-xgboost/tests/ubjson.rs` — covers XGB-02 (+ same-Model-as-JSON assertion)
- [ ] `crates/treelite-xgboost/tests/legacy.rs` — covers XGB-03 (mushroom smoke + fresh fixture)
- [ ] `crates/treelite-xgboost/tests/detect.rs` — covers XGB-04 (all three verdicts + edge bytes)
- [ ] `crates/treelite-xgboost/tests/nan_inf.rs` — covers D-02 (value recovery + string safety)
- [ ] `crates/treelite-harness/tests/three_format_equivalence.rs` — D-05 1e-5 across formats
- [ ] Promote `golden_v5.rs::loader_path_divergence_diagnostic` to a fatal D-10 assertion once the loader closes the gap

## Security Domain

> `security_enforcement: true`, ASVS level 1. This phase parses **untrusted binary/JSON model files** — input-validation is the dominant security surface.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | library, no access control |
| V5 Input Validation | **yes** | Every length/count read from the file MUST be bounds-checked before allocation/indexing. `num_nodes`, `u64` string lengths, array dims, `num_trees` are attacker-controlled. Return typed `XgbError`, never panic/OOB/OOM. |
| V6 Cryptography | no | no crypto |

### Known Threat Patterns for {untrusted model parsing}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Oversized length prefix (`u64` name/leaf-vector len) → giant allocation (OOM/DoS) | Denial of Service | Validate `len` against remaining buffer length before allocating/reading; the cursor's `buf.get(pos..pos+n)` returns `Option` → typed error, not panic |
| `num_nodes`/`num_trees` huge or negative → OOM or capacity overflow | Denial of Service | `require_non_negative` (exists) + cap against remaining bytes; the legacy `TREELITE_CHECK_GT(num_nodes,0)` and dimension checks port to typed errors |
| Array-length mismatch (`split_conditions.len() != num_nodes`) → OOB index | Tampering | `check_dim` (exists) runs BEFORE builder emission (`delegated_handler.cc:398-432`) |
| NaN/Inf injected to crash the float parser | Tampering | Handled deliberately by the D-02 sentinel mechanism — non-finite values are *expected* and recovered, not crashes |
| Truncated file mid-struct | Denial of Service | Cursor `.get()` returns `Option`; every read is fallible → `XgbError::LegacyTruncated` |
| UBJSON `$`/`#` count larger than the stream | Denial of Service | Validate declared container count against remaining bytes before pre-allocating the `Vec`/`Map` |

> **Net:** the existing `require_non_negative` + `check_dim` validators (Phase 1) plus a "every read is `Option`-checked" cursor discipline (D-07) satisfy ASVS V5 for this surface. New error variants: `XgbError::LegacyTruncated`, `XgbError::UbjsonDecode`, `XgbError::FormatDetect`/`UnknownFormat`.

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/src/model_loader/xgboost_legacy.cc` — full legacy `ParseStream`, `PeekableInputStream`, struct layouts (D-07/D-08, XGB-03).
- `treelite-mainline/src/model_loader/xgboost_json.cc` + `.../xgboost_ubjson.cc` — JSON/UBJSON entry points, DART weight-drop post-processing.
- `treelite-mainline/src/model_loader/detail/xgboost.cc` — `DetectXGBoostFormat`, `GetPostProcessor`, `TransformBaseScoreToMargin`, `ParseBaseScore` (D-09, XGB-05).
- `treelite-mainline/src/model_loader/detail/xgboost_json/delegated_handler.{h,cc}` — recognized key map, EndObject metadata finalize, scalar/vector base_score (XGB-01/05, D-01).
- `treelite-mainline/src/model_loader/detail/xgboost_json/sax_adapters.h` — RapidJSON/nlohmann adapter surface (parity reference).
- `crates/treelite-xgboost/src/{lib,objective,error}.rs`, `crates/treelite-builder/src/lib.rs` — existing Rust base (widening point, builder API, CR-01/CR-02 columns).
- **Empirical (this session):** parsed `mushroom.model` byte-for-byte (legacy layout validated, 1501 B exact); confirmed xgboost 3.2.0 cannot write legacy binary; confirmed serde_json rejects bare NaN/Inf and `1e999`; validated the pre-lex + `de_f32` mechanism end-to-end; decoded a live `.ubj` to enumerate emitted tags; confirmed `serde_json_lenient` source rejects `+/- infinity`.

### Secondary (MEDIUM confidence)
- crates.io metadata for `serde_json_lenient` (895K dl, Google, 2024-12), `serde_ubjson` (6.9K, 2017), `ubjson` (1.7K, 2021) — maintenance/legitimacy assessment.
- `xgboost-master/doc/tutorials/saving_model.rst` — saved-model JSON structure; schema-doc-removed note (L313).

### Tertiary (LOW confidence)
- xgboost 1.7.6 as the legacy-write pin (A1) — training-knowledge assumption; the generation spike is the verifier.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new runtime deps; the disqualification of `serde_json_lenient` and the abandonment of the UBJSON crates are source/metadata-verified.
- Architecture / legacy layout: HIGH — every byte offset validated against a real vendored fixture; recognized-key map read directly from the porting source.
- Pitfalls: HIGH — Pitfalls 1, 2, 5, 7 were each reproduced or directly observed this session.
- D-06 version pin: MEDIUM — the *need* for an old xgboost is verified; the *exact* version (1.7.6) is assumed and gated by the Wave-0 spike.

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (stable — the porting source is vendored/frozen; only the xgboost-1.7.6 pin and crate-maintenance facts could drift)

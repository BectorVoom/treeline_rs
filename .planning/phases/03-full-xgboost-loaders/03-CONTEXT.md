# Phase 3: Full XGBoost Loaders - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Widen the loader layer to the **complete XGBoost surface** — JSON, UBJSON, and legacy-binary formats — with format auto-detection and the version-gated (`major_version >= 1`) `base_score` probability→margin transform (scalar and vector forms). All three formats must load the same logical model and predict within **1e-5** of a single shared golden.

**In scope (Phase 3):**
- Full XGBoost-JSON loader (XGB-01) — widen the Phase-1 minimal loader to the full saved-model structure.
- XGBoost UBJSON loader (XGB-02) sharing the JSON numeric path for parity.
- XGBoost legacy-binary loader (XGB-03) via explicit little-endian decoding.
- Format auto-detection (XGB-04) — see interpretation note in D-09 below.
- Objective→postprocessor mapping + version-gated `base_score` margin transform, scalar and vector base_score forms (XGB-05). No constant prediction offset.
- Close DEF-02-01: XGBoost loader→serialize v5 byte-fidelity across all three formats.

**Out of scope (later phases):**
- Full GTIL prediction surface — 4 predict kinds, 10 postprocessors, sparse CSR, **categorical split evaluation**, multi-class output shaping (Phase 5). Phase 3's predict path remains the Phase-1 scalar slice (numerical splits, identity/sigmoid). See D-04 (parse-wide / verify-narrow).
- LightGBM / scikit-learn loaders (Phase 4).
- cubecl kernels (Phase 6), GPU (Phase 7), PyO3 (Phase 8), memory hardening (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### Parsing Architecture (JSON + UBJSON)
- **D-01 (shared numeric path):** JSON and UBJSON **converge at the existing serde structs**. UBJSON is decoded into the same `serde_json::Value` / typed `XgbModelJson` structs the Phase-1 JSON path already uses, so both formats share all downstream numeric coercion. This keeps the working Phase-1 serde-DOM JSON loader (already rewired through `treelite_builder::ModelBuilder` per 02-05) and achieves criterion-2 parity at the deserialize level rather than at the SAX level. Upstream's parity mechanism is the shared `DelegatedHandler` SAX handler with per-format adapters; our equivalent convergence point is the serde structs.
  - **Rejected:** porting upstream's full SAX `DelegatedHandler` + adapter pair (replaces working Phase-1 loader with streaming; more work); a standalone independent UBJSON decoder (two numeric paths risk drift — works against criterion 2).
- **D-02 (NaN/Inf — requirement locked, mechanism deferred):** Bare `NaN` / `Infinity` / `-Infinity` literals **MUST round-trip into f32 thresholds/leaf values**, value-position only (never inside string contents). `serde_json`'s tokenizer rejects these before any custom deserializer runs, and XGBoost emits them in float arrays (e.g. `split_conditions`); upstream accepts them via RapidJSON's `kParseNanAndInfFlag`. The **concrete mechanism is left to the research phase** (candidates: string-safe tolerant pre-lex + custom serde float deserializer, vs swapping the JSON front-end to a NaN/Inf-tolerant parser). This is research-flagged in ROADMAP.
- **D-03 (UBJSON decode mechanism — deferred to research):** Locked invariant is convergence at the same serde structs (D-01). Research chooses **hand-rolled type-tag decoder → `serde_json::Value`** vs a **UBJSON serde data-format crate** after surveying the Rust UBJSON ecosystem and vetting any crate's maintenance/quality. The research flag explicitly calls out "UBJSON type-tag decoding."
- **D-04 (parse-wide / verify-narrow — KEY BOUNDARY):** **PARSE** the full XGBoost structure faithfully — multiclass grouping (`tree_info`/`num_class`), categorical split fields, vector `base_score`, DART `weight_drop` — so the structs are complete and future-proof. But **VERIFY 1e-5 only on fixtures today's scalar GTIL can predict** (numerical splits, identity/sigmoid). Categorical/multiclass *prediction* parity lands when Phase 5 widens GTIL. Rationale: criterion 1 demands 1e-5 prediction, but full GTIL (categoricals, output shaping, 10 postprocessors) is owned by Phase 5 — Phase 3 cannot verify shapes it cannot yet predict.
  - **Rejected:** parsing only what scalar GTIL predicts (forces struct re-widening in Phase 4/5); pulling categorical/multiclass GTIL forward from Phase 5 (scope creep into GTIL's territory).

### Three-Format Fixtures & Golden
- **D-05 (one logical model, three formats):** **Fresh-train one small `binary:logistic` model** in a single xgboost session and save it in all three formats (JSON, UBJSON, legacy binary) from that one session — strongest "same logical model" guarantee for criterion 1. Capture the shared prediction golden from the **upstream Treelite Python wheel** (carries forward Phase-1 D-06; no C++ compile, never regenerated in CI). Keep the model `binary:logistic` + numerical splits so today's scalar GTIL verifies it (consistent with D-04 verify-narrow).
- **D-06 (legacy-binary generation toolchain):** Modern xgboost has **deprecated writing** legacy binary (read-mostly). **Pin an older xgboost version that still writes legacy binary in the fixture-GENERATION script only** — never a runtime or CI dependency. The fresh model is saved legacy-binary by the pinned old xgboost, JSON/UBJSON by current xgboost; all three remain the same logical model. **Freeze the generator manifest** (xgboost version(s), Treelite wheel version, OS/arch, libm/glibc) beside the golden — carries forward Phase-1 D-07 discipline.
  - **Research/spike note:** empirically confirm the pinned xgboost actually emits legacy binary, and that current xgboost reads back the same logical model into JSON/UBJSON, before locking the exact versions.

### Legacy Binary & Auto-Detect
- **D-07 (legacy decoder mechanism):** **Hand-rolled little-endian byte-cursor** helper using `from_le_bytes` (u32/i32/f32/u64/…), plus a peekable reader mirroring upstream's `PeekableInputStream` (1024-byte peek window). Zero external dependencies; literally satisfies criterion 2's "explicit little-endian decoders (no native-endian struct transmute)."
  - **Rejected:** `byteorder` crate (extra pinned dep for boilerplate that `from_le_bytes` already covers); a parser crate (`binread`/`nom`) — heavier than needed.
- **D-08 (NO native-endian transmute):** Legacy binary fields are read field-by-field via explicit LE conversions. Never `transmute` a byte buffer onto a native struct. This is a hard criterion-2 invariant.
- **D-09 (auto-detect scope — mirror upstream's API split):** **Port `DetectXGBoostFormat` exactly** — it disambiguates **JSON vs UBJSON only** (first char `{`; second char / no-op + type markers distinguish UBJSON), returning "json"/"ubjson"/"unknown". **Legacy binary is reached through a separate explicit loader entry point**, matching upstream's API shape (upstream does NOT auto-detect legacy-vs-JSON in one call). 
  - **Criterion-2 interpretation note for the verifier:** "auto-detects which XGBoost format a file is" is satisfied at the JSON-vs-UBJSON level via the ported heuristic; legacy binary is selected by the explicit legacy entry point. This is a deliberate fidelity choice, not a gap.
  - **Rejected:** a unified 3-way content sniff (non-`{` ⇒ legacy) — would extend beyond upstream's API split.

### DEF-02-01 — Loader→Serialize Byte-Fidelity
- **D-10 (close across all three formats):** Phase 2 proved the *serializer* is byte-faithful (`serialize(deserialize(golden_v5.bin)) == blob`) and 02-06 made the builder emit upstream's exact AllocNode column layout (CR-01/CR-02), but **loader→serialize byte-fidelity was never proven** (deferred as DEF-02-01). Phase 3 closes it **across all three formats**: for the fresh model in JSON, UBJSON, AND legacy binary — load in Rust, serialize to v5, and byte-compare against the **single** upstream-Treelite-wheel-serialized v5 golden blob (same logical model ⇒ one upstream blob). 
  - **Cross-format invariant established:** all three loaders → identical `Model` → identical v5 bytes == upstream's v5 bytes. This simultaneously proves the three loaders produce identical Model layout AND that the Rust loader matches upstream's serialization.
  - **Research note (brittleness):** the binding risk is column ordering / bookkeeping-scalar emission, not float formatting (v5 binary stores raw float bytes). Research should map exactly which Model columns/scalars the loader must populate to match upstream's serialized layout, building on the 02-06 AllocNode groundwork.

### Claude's Discretion
- Exact module/file layout within `crates/treelite-xgboost` (e.g., `legacy.rs`, `ubjson.rs`, `detect.rs`, widened `json` structs).
- Error-enum additions to `XgbError` for the new formats (idiomatic `thiserror`, transparent builder propagation as established in 02-05).
- Internal representation of the peekable reader / byte cursor.
- Exact full objective→postprocessor mapping table extent for XGB-05 (port upstream's map; the verify-narrow fixture exercises sigmoid).
- Whether DART `weight_drop` leaf-scaling is applied at parse time (parse-wide) given it has no verify-narrow fixture yet.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level (this milestone)
- `.planning/PROJECT.md` — core value (1e-5 equivalence), constraints, Key Decisions, Out of Scope (no C-API, v5-only serialization, golden frozen from upstream wheel).
- `.planning/REQUIREMENTS.md` — Phase 3 IDs **XGB-01..XGB-05** (full text + traceability table).
- `.planning/ROADMAP.md` § "Phase 3: Full XGBoost Loaders" — goal + 3 success criteria + research flag (the authoritative acceptance bar).
- `.planning/STATE.md` — Phase 3 blockers (serde_json NaN/Inf) and the DEF-02-01 deferred item carried here.

### Upstream porting source of truth (`treelite-mainline/`, C++ v4.7.0)
- `treelite-mainline/src/model_loader/xgboost_json.cc` — JSON loader entry; objective handling.
- `treelite-mainline/src/model_loader/xgboost_ubjson.cc` — UBJSON loader; nlohmann SAX `ubjson` parse; DART `weight_drop` leaf-scaling (applied post-CommitModel).
- `treelite-mainline/src/model_loader/xgboost_legacy.cc` — legacy-binary `ParseStream`, `PeekableInputStream` (1024-byte peek), full binary header/tree layout — the authority for D-07/D-08.
- `treelite-mainline/src/model_loader/detail/xgboost.cc` — **`DetectXGBoostFormat` heuristic** (first/second char JSON-vs-UBJSON; lines ~60–115) — the authority for D-09. Also shared loader helpers.
- `treelite-mainline/src/model_loader/detail/xgboost.h` — `HandlerConfig`, shared loader declarations.
- `treelite-mainline/src/model_loader/detail/xgboost_json/delegated_handler.{h,cc}` — upstream's shared SAX handler + the recognized per-tree key subset (parity reference for the widened structs; convergence-point analog for D-01).
- `treelite-mainline/src/model_loader/detail/xgboost_json/sax_adapters.{h,cc}` — RapidJSON + nlohmann adapters feeding the shared handler (how upstream achieves JSON/UBJSON parity).
- `treelite-mainline/src/model_loader/detail/string_utils.h` — `StringStartsWith` etc. used by detection/legacy.
- `treelite-mainline/include/treelite/model_loader.h` — public loader API: `LoadXGBoostModelJSON`, `...UBJSON`, `...LegacyBinary`, `DetectXGBoostFormat` (returns json/ubjson/unknown — confirms D-09's split).

### XGBoost format authority (`xgboost-master/`, v3.3.0-dev — read-only reference)
- `xgboost-master/doc/tutorials/saving_model.rst` (~L180–260) — worked saved-model JSON structure (`learner → gradient_booster → gbtree`, `objective`, `base_score`, version). Note L313: standalone JSON schema removed in 3.2 (structure unchanged).

### Existing Rust code (Phase 1/2 — the widening base)
- `crates/treelite-xgboost/src/lib.rs` — Phase-1 serde-DOM JSON loader + intermediate structs (`XgbModelJson`, `Learner`, `RegTreeJson`, `TreeParam`); the convergence point D-01 widens. Builds the `F32` variant only.
- `crates/treelite-xgboost/src/objective.rs` — existing objective→postprocessor map + `transform_base_score_to_margin` / `prob_to_margin_*` (the XGB-05 version gate is partially built here).
- `crates/treelite-xgboost/src/error.rs` — `XgbError` (extend for new formats).
- `crates/treelite-builder/` — `ModelBuilder` / `BuilderMetadata` the loader emits through (02-05 rewiring; 02-06 AllocNode column emission CR-01/CR-02 — the groundwork D-10 builds on).
- `crates/treelite-harness/tests/golden_v5.rs` + `fixtures/golden_v5.bin` + `fixtures/golden_v5.manifest.json` — existing v5 golden + manifest pattern that D-10's loader→serialize byte-fidelity test extends.
- `fixtures/golden.json` — Phase-1 prediction golden pattern (D-05 extends to the 3-format model).

### Test corpus
- `treelite-mainline/tests/examples/mushroom/mushroom.model` — the **only** vendored XGBoost fixture (legacy binary, `binary:logistic`, ~1.5KB). Useful as a real legacy-binary parse smoke test even though D-05 fresh-trains its own 3-format fixture. No XGBoost JSON/UBJSON or golden vectors ship in the tree.

### Codebase maps
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md` — SoA/variant pattern, naming/error translation, test layout.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase-1 serde-DOM JSON loader** (`crates/treelite-xgboost/src/lib.rs`) — the convergence point for D-01; widen its intermediate structs rather than rewrite. Already builds through `ModelBuilder`.
- **`objective.rs`** — objective→postprocessor map + version-gated `base_score`→margin transform already exist; XGB-05 extends the map and handles the vector base_score form.
- **`ModelBuilder` + 02-06 AllocNode column emission** — the loader emits the upstream-faithful column layout that makes D-10's byte-fidelity achievable.
- **v5 golden + manifest harness** (`treelite-harness`, `fixtures/golden_v5.*`) — extend for the loader→serialize byte-fidelity test (D-10) and the 3-format prediction golden (D-05).

### Established Patterns
- **Converge-then-build:** parse → typed structs → validators (`require_non_negative`/`check_dim`) → `ModelBuilder` emission → `CommitModel` (from 02-05). New formats plug in at the struct layer (D-01).
- **F32-only variant:** XGBoost always yields `<f32,f32>` (Phase-1 decision); unchanged for all three formats.
- **thiserror transparent propagation:** builder errors surface as `XgbError::Builder` — no panic, no anyhow in the library crate.
- **Golden frozen from upstream wheel; CI never regenerates** (D-06/D-07; mirrors D-06/D-07 of Phase 1).

### Integration Points
- New format entry points (`LoadXGBoostModelUBJSON`-analog, `LoadXGBoostModelLegacyBinary`-analog, `DetectXGBoostFormat`-analog) join the existing JSON entry in `treelite-xgboost`.
- D-10 connects the loader crate to the serializer/harness: loaded `Model` → v5 serialize → byte-compare vs upstream golden blob.

</code_context>

<specifics>
## Specific Ideas

- The shared golden should be one logical `binary:logistic` model with **numerical splits only** so the existing scalar GTIL verifies all three formats at 1e-5 today (verify-narrow, D-04). Vendored `mushroom.model` is a good independent legacy-binary parse smoke test.
- D-10 yields a single upstream v5 golden blob that ALL THREE Rust loaders must serialize to byte-identically — a deliberately strong cross-format equality invariant, not three separate goldens.
- Carry the Phase-1 golden discipline verbatim: artifact = output vector + input matrix + generator manifest, frozen together.

</specifics>

<deferred>
## Deferred Ideas

- **Categorical-split & multiclass PREDICTION parity** — parsed now (parse-wide, D-04) but 1e-5 verification waits for Phase 5 GTIL widening (categorical evaluation, output shaping, full postprocessor set).
- **DART `weight_drop` verified prediction** — parsed parse-wide; no verify-narrow fixture exercises it this phase.
- **Vector `base_score` verified prediction** — transform handled (XGB-05) but a multi-output verifying fixture aligns with Phase 5.
- **NaN/Inf & UBJSON-decode concrete mechanisms** — deliberately left to the research phase (D-02, D-03), not lost.

None of the above is scope creep into Phase 3 — all are recorded so they aren't lost.

</deferred>

---

*Phase: 3-full-xgboost-loaders*
*Context gathered: 2026-06-10*

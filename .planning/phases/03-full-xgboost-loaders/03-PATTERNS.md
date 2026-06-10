# Phase 3: Full XGBoost Loaders - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 11 (5 new source modules, 3 modified source files, 3 new/modified test+fixture artifacts)
**Analogs found:** 11 / 11 (every new file has a strong in-repo analog — this is a widening of an existing crate, not greenfield)

## How to read this map

This phase widens the existing `crates/treelite-xgboost` crate. Almost every new file is a *split-out and extension* of patterns already present in `crates/treelite-xgboost/src/lib.rs`, `objective.rs`, and `error.rs`, plus the harness golden discipline in `crates/treelite-harness`. The planner should treat the **existing Phase-1 loader as the literal template** — converge-then-build, F32-only, thiserror-transparent, golden+manifest. Excerpts below are the exact lines to copy from.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-xgboost/src/lib.rs` (MODIFY) | loader / orchestrator | transform (parse→build) | itself (Phase-1 `load_xgboost_json` + `build_tree`) | exact (in-place widen) |
| `crates/treelite-xgboost/src/json.rs` (NEW, split from lib.rs) | model (serde structs) + utility (pre-lex/de_f32) | transform (file-I/O→struct) | `lib.rs` `XgbModelJson`/`RegTreeJson` structs (lines 35-90); `harness/src/lib.rs` `NanF32` visitor (lines 36-69) | exact |
| `crates/treelite-xgboost/src/ubjson.rs` (NEW) | utility (decoder) | transform (bytes→`serde_json::Value`) | `harness/src/lib.rs::normalize_nan_tokens` byte-scanner (lines 132-173); `json.rs` `de_f32` sentinel adapter | role-match (byte cursor) |
| `crates/treelite-xgboost/src/legacy.rs` (NEW) | loader / utility (LE cursor) | file-I/O (binary decode) | `build_tree` per-node loop (`lib.rs` 155-191); RESEARCH §Legacy Binary Layout cursor sketch | role-match (no existing binary reader; closest is the builder-emission loop) |
| `crates/treelite-xgboost/src/detect.rs` (NEW) | utility (format sniff) | transform (bytes→tag) | `harness/src/lib.rs` `is_ident_byte`/`preceded_by_ident` byte-classifier helpers (lines 163-173) | role-match |
| `crates/treelite-xgboost/src/objective.rs` (MODIFY) | service (mapping table) | transform | itself (Phase-1 `get_postprocessor` + `transform_base_score_to_margin`) | exact (extend in place) |
| `crates/treelite-xgboost/src/error.rs` (MODIFY) | error enum | — | itself (Phase-1 `XgbError`) | exact (add variants) |
| `crates/treelite-xgboost/tests/load_fixture.rs` or new `tests/three_format.rs` | test | request-response | `tests/load_fixture.rs` (existing per-assertion tests) | exact |
| `crates/treelite-harness/tests/golden_v5.rs` (MODIFY/extend) | test (byte-fidelity) | request-response | itself + `loader_path_divergence_diagnostic` (lines 99-128) | exact |
| `fixtures/generate_xgb_3format.py` (NEW) | config (generation script) | batch | `fixtures/capture_golden_v5.py`; `fixtures/capture_golden.py` | exact |
| `fixtures/*.manifest.json` + golden artifacts (NEW) | config (frozen fixture) | — | `fixtures/golden_v5.manifest.json` | exact |

## Pattern Assignments

### `crates/treelite-xgboost/src/json.rs` (model structs + pre-lex + de_f32) — XGB-01, D-01, D-02

**Analogs:** `crates/treelite-xgboost/src/lib.rs` lines 35-90 (struct family); `crates/treelite-harness/src/lib.rs` lines 39-69 (`NanF32` deserialize_with visitor); RESEARCH §Code Examples lines 451-503 (validated pre-lex + `de_f32`).

**Struct-family pattern to copy and WIDEN** (`lib.rs:35-90`) — keep the exact `#[derive(Deserialize)]` + `String`-scalar style; the RESEARCH §Recognized Key Map lists every key to add:
```rust
#[derive(Deserialize)]
struct RegTreeJson {
    tree_param: TreeParam,
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<i32>,
    split_type: Vec<i32>,             // 0 numeric / 1 categorical
    split_conditions: Vec<f32>,       // <-- attach #[serde(deserialize_with="de_vec_f32")] (D-02)
    default_left: Vec<i32>,
}
```
Widen-then-converge additions (parse-wide, D-04) the planner must add to these structs but gate behind leaf-vector/categorical/multiclass branches: `loss_changes`, `sum_hessian` (REQUIRED for D-10 — see byte-fidelity table), `base_weights`, `leaf_weights`, `categories_segments`/`sizes`/`nodes`/`categories`, plus learner-level `boost_from_average`, vector `base_score`, DART `weight_drop`. Source authority: RESEARCH §Recognized Key Map (`delegated_handler.cc` `is_recognized_key`).

**Scalar-as-string note (load-bearing):** XGBoost stores learner/tree numeric scalars as JSON *strings* even in UBJSON (`S` tag). Keep the Phase-1 `String` + `parse_scalar` path (`lib.rs:96-107`); do NOT special-case UBJSON scalars.

**Pre-lex + de_f32 (D-02) — copy verbatim from RESEARCH lines 451-503.** The string-state-tracking scanner mirrors the harness's existing `normalize_nan_tokens` discipline (only-in-value-position, string-safe). The `de_f32` visitor mirrors the harness `NanF32` visitor structure exactly (`harness/src/lib.rs:39-69`):
```rust
// harness NanF32 visitor — the structural template for de_f32:
fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> { Ok(NanF32(v as f32)) }
fn visit_unit<E>(self) -> Result<Self::Value, E> { Ok(NanF32(f32::NAN)) }
```
The new `de_f32` swaps the `null`→NaN rule for sentinel-string recovery (`"@NaN@"`/`"@Inf@"`/`"@-Inf@"`). JSON path applies `replace_nonfinite` to the input string before `serde_json::from_str`; UBJSON path emits the sentinels directly (see `ubjson.rs`).

---

### `crates/treelite-xgboost/src/ubjson.rs` (hand-rolled tag decoder → `serde_json::Value`) — XGB-02, D-03

**Analogs:** `crates/treelite-harness/src/lib.rs::normalize_nan_tokens` (lines 132-173) — the established byte-cursor-with-state idiom in this repo; the new `de_f32` adapter in `json.rs` (shared convergence point).

**Pattern:** recursive-descent over the byte stream emitting `serde_json::Value`, then `serde_json::from_value::<XgbModelJson>` finishes (converges at the SAME structs — D-01). The byte-walking loop should follow the harness's `while i < bytes.len()` cursor shape:
```rust
// harness normalize_nan_tokens — the repo's byte-cursor template (lib.rs:140-156):
let mut i = 0;
while i < bytes.len() {
    if bytes[i] == b'N' && raw[i..].starts_with("NaN") { /* ... */ i += 3; }
    else { out.push(bytes[i]); i += 1; }
}
```
**Critical (RESEARCH Pitfall 4):** implement the `$`type + `#`count optimized-container fast path (`[$d#<count>` = typed float32 array). A naive per-element-tag decoder mis-parses. Tag set = RESEARCH §UBJSON Type-Tag Map (14 tags).

**Critical (RESEARCH Pitfall 5):** when a decoded `d`/`D` float is non-finite, emit `Value::String("@NaN@"/"@Inf@"/"@-Inf@")` NOT `Value::Number` (serde_json maps non-finite f64 to `Null`). This routes UBJSON floats through the same `de_f32` adapter as JSON — the criterion-2 numeric-parity guarantee.

---

### `crates/treelite-xgboost/src/legacy.rs` (LE byte cursor + PeekableReader) — XGB-03, D-07, D-08

**Analogs:** `crates/treelite-xgboost/src/lib.rs::build_tree` (lines 155-191) — the per-node emission loop the legacy decoder feeds into; RESEARCH §Legacy Binary Layout (lines 277-344, the byte-exact map validated against `mushroom.model`) and §Code Examples lines 521-541 (cursor sketch).

**Cursor pattern (copy from RESEARCH 521-541):** each accessor slices `self.buf.get(pos..pos+N)`, advances `pos`, and uses `from_le_bytes` — never `transmute` (hard D-08 invariant). Truncation → typed `XgbError` (mirror the `?`-on-`get` ergonomics already used in the builder's `nodes[..]` access).

**Emission convergence:** after decoding nodes/stats, drive the SAME `ModelBuilder` calls `build_tree` already uses (`lib.rs:170-189`):
```rust
builder.start_node(i as i32)?;
if is_leaf { builder.leaf_scalar(leaf_value)?; }
else { builder.numerical_test(split_index, threshold, default_left, Operator::kLT, cleft, cright)?; }
builder.end_node()?;
```
Bit-unpacking (RESEARCH Pitfall 6, authority `xgboost_legacy.cc:214-219`): `split_index = sindex & 0x7FFFFFFF`; `default_left = (sindex >> 31) != 0`; leaf iff `cleft == -1`; leaf value = `info` union reinterpreted as f32. Struct-size asserts to encode: `LearnerModelParam=136`, `GBTreeModelParam=168`, `TreeParam=148`, `Node=20`, `NodeStat=16`.

**D-10 stat emission (see Shared Pattern below):** legacy fills `sum_hess` on every node from `NodeStat.sum_hess` and `gain` on internal nodes from `NodeStat.loss_chg`.

---

### `crates/treelite-xgboost/src/detect.rs` (`DetectXGBoostFormat`) — XGB-04, D-09

**Analog:** `crates/treelite-harness/src/lib.rs` byte-classifier helpers `is_ident_byte`/`preceded_by_ident` (lines 163-173) — same "tiny pure byte-predicate" idiom.

**Copy verbatim from RESEARCH lines 506-518** (the ported `detail/xgboost.cc:83-115` heuristic). Returns `"json"`/`"ubjson"`/`"unknown"` over the first two bytes. JSON-vs-UBJSON only; legacy is reached through a separate explicit entry point (matches upstream's `model_loader.h` API split: `LoadXGBoostModelJSON`/`...UBJSON`/`...LegacyBinary` are distinct functions, line 31/47/65; `DetectXGBoostFormat` at line 83 only disambiguates JSON/UBJSON).

---

### `crates/treelite-xgboost/src/objective.rs` (MODIFY) — XGB-05

**Analog:** itself (lines 23-70). The objective→postprocessor map and `transform_base_score_to_margin` are already ported verbatim and pass `tests/error.rs`. **Do not rewrite** — confirm the table unchanged vs RESEARCH §XGB-05 table, then ADD:
- scalar-vs-vector `base_score` handling (RESEARCH lines 366-369): if the `base_score` string starts with `[`, parse as a JSON array of floats; element-count must equal `num_target*num_class`; apply the transform element-wise.
- The **f64 invariant is already enforced** (`prob_to_margin_sigmoid`/`_exponential` are pure f64, lines 50-58). For vector base_score, cast each f32 element to f64 BEFORE transforming (RESEARCH Pitfall 3 — the #1 silent 1e-5 break).

The version gate (`version.is_empty() || version[0] >= 1`) is already implemented in `lib.rs:253`; reuse it for legacy via `major_version >= 1`.

---

### `crates/treelite-xgboost/src/error.rs` (MODIFY)

**Analog:** itself (the existing `XgbError` enum, lines 14-82). Follow the established `thiserror` idiom exactly:
- `#[from]` for transparent propagation (existing: `Json(#[from] serde_json::Error)` line 20; `Builder(#[from] ...BuilderError)` line 81; `Core(#[from] ...CoreError)` line 72).
- structured fields with `#[error("...{field}...")]` (existing: `DimensionMismatch`, lines 50-62).

**Add variants** (D-discretion): `Ubjson { ... }` (decode/tag errors), `Legacy { ... }` (truncation, bad magic `bs64`, struct-size mismatch), `FormatDetect { ... }` or reuse a generic. Keep builder errors flowing transparently through the existing `Builder(#[from] BuilderError)` — no new wrapping.

---

### `crates/treelite-harness/tests/golden_v5.rs` (extend) + new 3-format test — DEF-02-01, D-10

**Analog:** itself — `serializer_reproduces_golden_v5_byte_for_byte` (lines 63-88) for the byte-compare + `first_diff` machinery (lines 46-57); `loader_path_divergence_diagnostic` (lines 99-128) is the EXACT scaffold to promote from non-fatal diagnostic to hard assertion once the loader gap closes.

**Pattern:** the diagnostic currently prints divergence (lines 112-126). Phase 3 closes the gap (emit `sum_hess`/`gain`, `attributes: None`) and turns it into the single-golden cross-format assertion:
```rust
// promote the existing diagnostic shape to a hard assertion across 3 formats:
let m_json   = treelite_xgboost::load_xgboost_json(&json)?;
let m_ubj    = treelite_xgboost::load_xgboost_ubjson(&ubj_bytes)?;
let m_legacy = treelite_xgboost::load_xgboost_legacy(&bin_bytes)?;
let golden = std::fs::read(fixture_path("golden_v5_3format.bin"))?;
for mut m in [m_json, m_ubj, m_legacy] {
    assert_eq!(treelite_core::serialize_to_buffer(&mut m), golden); // first_diff on failure
}
```
Reuse `fixture_path` (lines 37-43) and `first_diff` (lines 46-57) verbatim.

---

### `fixtures/generate_xgb_3format.py` (NEW generation script) — D-05, D-06

**Analog:** `fixtures/capture_golden_v5.py` (entire file) and `fixtures/capture_golden.py`. Copy the script structure exactly:
- HERE-relative path resolution (capture_golden_v5.py:44-48).
- write raw bytes + a sibling `*.manifest.json` with `{treelite, xgboost, os, arch, libc, python, sha256, nbytes, source_fixture}` (capture_golden_v5.py:62-79).
- empirically-settling asserts in the script itself (capture_golden_v5.py:82-88 asserts the version triple) — here, assert the legacy fixture's first byte is NOT `{`/`N` and the 136-byte `LearnerModelParam` decodes (RESEARCH Pitfall 7).

**Generation-only pins (NEVER runtime/CI):** xgboost `1.7.6` `[ASSUMED A1]` writes the legacy fixture; xgboost `3.2.0` writes JSON/UBJSON; treelite `4.7.0` captures the prediction golden + single v5 byte golden. Run via `uv run --with 'xgboost==1.7.6' --with 'treelite==4.7.0' ...` in a throwaway env (RESEARCH §Installation). **Set `base_score=0.5` (RESEARCH A2)** so the version-gate transform is a no-op and all three formats agree regardless of the legacy header's `major_version`.

## Shared Patterns

### Converge-then-build (the spine of every loader path)
**Source:** `crates/treelite-xgboost/src/lib.rs` `load_xgboost_json` (lines 201-293) + `build_tree` (155-191).
**Apply to:** `json.rs`, `ubjson.rs`, `legacy.rs` — all three converge at `XgbModelJson` (or fill the same logical fields) and emit through ONE shared `build_model_from_parsed` path.
```rust
let parsed: XgbModelJson = serde_json::from_str(json)?;   // (UBJSON: from_value; legacy: fill directly)
// validate scalars → require_non_negative / check_dim → ModelBuilder → commit_model
let mut builder = ModelBuilder::new(metadata)?;
for (i, t) in booster.trees.iter().enumerate() { build_tree(&mut builder, i, t)?; }
let mut model = builder.commit_model()?;
model.sigmoid_alpha = 1.0; model.ratio_c = 1.0;
```

### F32-only variant
**Source:** `lib.rs:280-285` + builder `ModelBuilder` only produces `Tree<f32>` (`builder/src/lib.rs:128-135`).
**Apply to:** all three formats — XGBoost always yields `ModelVariant::F32`. No variant branching.

### thiserror transparent propagation (no panic, no anyhow in lib crates)
**Source:** `crates/treelite-xgboost/src/error.rs:19-82`.
**Apply to:** every new error path. Builder errors surface via the existing `Builder(#[from] BuilderError)` (line 81); add `Ubjson`/`Legacy` variants in the same idiom. Tests assert typed errors, not panics (template: `tests/load_fixture.rs:81-149`).

### D-10 byte-fidelity column emission (THE close-out work)
**Source:** RESEARCH §DEF-02-01 table (lines 379-388); builder CR-01/CR-02 already done (`builder/src/lib.rs:457-518`).
**Apply to:** all three loaders. The builder side is finished; the LOADER must now:
| Item | Loader action |
|------|---------------|
| `sum_hess` (every node) | `builder.sum_hess(...)` on every node (from `sum_hessian` JSON / `NodeStat.sum_hess` legacy) → triggers CR-02 `any_sum_hess` |
| `gain` (internal nodes) | `builder.gain(...)` on internal nodes (from `loss_changes` / `NodeStat.loss_chg`) |
| `data_count` | do NOT call `builder.data_count(...)` (upstream leaves it empty) |
| `attributes` | pass `attributes: None` (NOT `Some(String::new())`) so `commit_model` defaults to `"{}"` (`builder/src/lib.rs:569`) — **fixes the Phase-1 `lib.rs:277` `Some(String::new())`** |
| leaf `split_index` | leave at builder default `-1` (already the case) |
The current Phase-1 `lib.rs` does NOT emit `sum_hess`/`gain` and sets `attributes: Some(String::new())` — both diverge from the golden and MUST change (RESEARCH lines 385, 388).

### Golden + manifest discipline (frozen, CI never regenerates)
**Source:** `fixtures/capture_golden_v5.py`; `fixtures/golden_v5.manifest.json`; harness `check_manifest` (`harness/src/lib.rs:242-263`) and `Manifest` struct (lines 92-108).
**Apply to:** the 3-format generation script + its manifest + the prediction golden. Manifest keys are fixed; `check_manifest` warns (never fails) on environment drift.

## No Analog Found

No file in this phase is without an in-repo analog. The two lowest-precedent areas — the UBJSON tag decoder and the legacy LE binary cursor — have no *direct* byte-format reader in the Rust tree yet, but both reuse the repo's established byte-cursor idiom (`harness::normalize_nan_tokens`) for structure and the existing `build_tree`/`ModelBuilder` emission loop for output. The authoritative byte/tag layout is in RESEARCH (§Legacy Binary Layout, §UBJSON Type-Tag Map) and the vendored C++ (`xgboost_legacy.cc`, `xgboost_ubjson.cc`).

| File | Role | Data Flow | Note |
|------|------|-----------|------|
| `ubjson.rs` | utility decoder | bytes→Value | No prior UBJSON reader; structure from `normalize_nan_tokens`, spec from RESEARCH §UBJSON Type-Tag Map |
| `legacy.rs` | LE binary cursor | file-I/O | No prior binary reader; cursor sketch in RESEARCH lines 521-541, byte map lines 277-344, validated vs `mushroom.model` |

## Metadata

**Analog search scope:** `crates/treelite-xgboost/src/`, `crates/treelite-harness/{src,tests}/`, `crates/treelite-builder/src/`, `fixtures/`, plus upstream `treelite-mainline/include/treelite/model_loader.h` (API-shape confirmation only).
**Files scanned:** 12 (lib.rs, objective.rs, error.rs, load_fixture.rs, error.rs[test], golden_v5.rs, equivalence.rs, harness/src/lib.rs, builder/src/lib.rs, capture_golden_v5.py, golden_v5.manifest.json, model_loader.h).
**Pattern extraction date:** 2026-06-10
**Key upstream porting authorities (read-only, for the planner/executor):** `treelite-mainline/src/model_loader/xgboost_legacy.cc` (legacy), `.../xgboost_ubjson.cc` (UBJSON/DART), `.../detail/xgboost.cc` (DetectXGBoostFormat + objective map), `.../detail/xgboost_json/delegated_handler.cc` (recognized key set + base_score gate).

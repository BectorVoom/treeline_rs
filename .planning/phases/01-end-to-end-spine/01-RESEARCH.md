# Phase 1: End-to-End Spine - Research

**Researched:** 2026-06-10
**Domain:** Rust Cargo-workspace scaffolding + porting a C++ tree-ensemble representation/loader/inference slice to Rust with 1e-5 numerical equivalence
**Confidence:** HIGH (every representation claim anchored to a concrete vendored upstream file/line; stack versions verified against the crates.io sparse index this session)

## Summary

Phase 1 stands up the thinnest vertical slice of treelite-rs: a multi-crate Cargo workspace (edition 2024 / resolver 3), the four upstream enums with exact string round-trip, a two-variant SoA `Model`/`Tree<T>` core carrying full header metadata, a minimal XGBoost-JSON loader producing one model, a scalar single-threaded predict with identity/sigmoid only, and an equivalence harness asserting the output is within 1e-5 of a golden vector captured from the upstream `treelite==4.7.0` Python wheel. The whole point is to prove the load→predict→verify pipeline end-to-end, not to be complete in any one layer.

The good news: the upstream source needed for Phase 1 is small and fully readable. The enum string values, the ~20-column Tree SoA field set, the Model header field set, the XGBoost-JSON field list, the objective→postprocessor map, the base_score→margin transform, and the scalar traversal + sigmoid postprocessor are all in vendored files I read this session — every representation decision in this document cites a concrete `treelite-mainline/` line. The subtle risks are numerical, not structural: (1) XGBoost JSON always builds a `ModelPreset<float,float>` model and the `binary:logistic` `base_score` is run through `ProbToMargin::Sigmoid = -log(1/p - 1)` in **f64** before being added to the tree sum, so the golden's exact value depends on f32/f64 cast ordering that must be ported verbatim; (2) `sigmoid` postprocessor uses `model.sigmoid_alpha` (f32) and `std::exp` — libm divergence is the only thing that can break 1e-5, hence the frozen toolchain/libm manifest (D-07).

**Primary recommendation:** Build 4 crates (`treelite-core`, `treelite-xgboost`, `treelite-gtil`, `treelite-harness`) under one root `[workspace.dependencies]`. Port the enum strings, SoA columns, Model header, the `LearnerHandler::EndObject` metadata math, the `EvaluateTree` traversal, and the `sigmoid`/`identity` postprocessors **verbatim** from the cited upstream lines. Capture the golden from `treelite==4.7.0` + `xgboost==3.2.0` once, commit it with a manifest, and never regenerate in CI. Keep predict a plain function — no trait/backend abstraction (D-08).

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Workspace is spine-only, grown one layer per phase — create only the crates Phase 1 exercises, not stubs for all 9 phases.
- **D-02:** Initial members (names indicative, planner may refine):
  - `treelite-core` — enums (`TaskType`, `TreeNodeType`, `Operator`, `DType`) + `Model` enum + `Tree<T>` + `TreeBuf<T>` SoA columns + header metadata.
  - `treelite-gtil` — scalar single-threaded predict (identity/sigmoid only).
  - `treelite-xgboost` — minimal XGBoost-JSON loader.
  - `treelite-harness` — 1e-5 equivalence harness (dev/test-facing).
- **D-03:** Single root `[workspace.dependencies]` table; every third-party crate pinned to a current stable version, no pre-release on the critical path (FND-01/FND-02).
- **D-04:** The Phase 1 fixture is a hand-crafted XGBoost-JSON literal committed to the repo — no runtime dependency on the `xgboost` package to produce the model itself.
- **D-05 (CONSTRAINT — critical coupling):** Because the golden is captured by loading this fixture into the upstream Treelite Python wheel (D-06), the hand-crafted JSON must be valid enough for upstream Treelite/XGBoost to parse. It must conform to the XGBoost-JSON model structure (`learner → gradient_booster → gbtree`, `objective`, `base_score`, tree arrays) closely enough that `treelite` can load it. Use `binary:logistic` so the sigmoid postprocessor path is genuinely exercised; keep it minimal (1–2 shallow trees).
- **D-05a (schema authority):** XGBoost source vendored at `xgboost-master/` (v3.3.0-dev). XGBoost removed the standalone JSON schema file in 3.2 (structure unchanged). Hand-craft against the worked JSON example in `xgboost-master/doc/tutorials/saving_model.rst` cross-checked with Treelite's own XGBoost-JSON loader (`treelite-mainline/src/model_loader/`). Set JSON `version` to a value the wheel accepts; avoid NaN/Inf literals.
- **D-06:** First golden captured from the upstream Treelite Python wheel's GTIL (`pip install treelite==<matching 4.x>`, load fixture, run `treelite.gtil.predict`), then committed and frozen. No C++ source compile; CI never regenerates it.
- **D-07:** Commit a toolchain/libm manifest alongside the golden: treelite wheel version, OS/arch, libm/glibc version, and (if used) xgboost version. The manifest is part of the frozen artifact.
- **D-08:** Phase 1 predict is the simplest plain scalar single-threaded function — no backend/`Predictor` trait abstraction. The cubecl seam is deferred to Phase 6.

### Claude's Discretion

- Exact crate names/granularity within the spine-only constraint (enums default: in `treelite-core`).
- Rust representation mechanism for `TreeBuf<T>` owned-vs-borrowed mode (`Cow`, enum, custom) — must support zero-copy borrow (CORE-03).
- Error-enum granularity (per-crate `thiserror` enums vs shared) — default: per-crate, idiomatic.
- `DType` numeric-type coverage for Phase 1 (must match upstream `TypeInfo` string values; ENUM-01).
- Whether CI (GitHub Actions) is wired in Phase 1 or deferred — success criterion only requires `cargo build`/`cargo test` to pass.
- Exact manifest file format (TOML/JSON/markdown) and location.

### Deferred Ideas (OUT OF SCOPE)

- Backend/`Predictor` trait abstraction — deferred to Phase 6 (cubecl).
- Real-world example fixtures (mushroom legacy binary, LightGBM text) — Phase 3/4.
- GitHub Actions CI — not required by Phase 1 success criteria.
- serde_json NaN/Inf handling — Phase 3 XGBoost-JSON blocker; Phase 1 fixture must avoid NaN/Inf literals.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-01 | Cargo workspace (edition 2024, resolver "3") builds all member crates from a single pinned `[workspace.dependencies]` table | Standard Stack + Architecture Patterns (workspace layout); resolver 3 is the edition-2024 default. |
| FND-02 | All third-party crates pinned to current latest-stable, no pre-release on critical path | Standard Stack table — all versions verified this session; `approx` flagged (latest is `0.6.0-rc2` pre-release → pin `0.5.1`). |
| ENUM-01 | `TaskType`, `TreeNodeType`, `Operator`, `DType` with string conversions matching upstream | Exact strings extracted from `src/enum/*.cc` (see Enum String Table). `DType` ↔ upstream `TypeInfo`. |
| CORE-01 | `Model` as two-variant enum over `<f32,f32>`/`<f64,f64>` (no mixed types) | Upstream `ModelPresetVariant` (`tree.h:437`); Rust `enum` with two variants. |
| CORE-02 | `Tree<T>` stores all upstream node fields as parallel SoA columns | Full ~20-column field set extracted from `tree.h:97-127` (see Tree SoA Field Set). |
| CORE-03 | `TreeBuf<T>` supports owned + zero-copy borrowed (foreign-buffer) modes | Upstream `ContiguousArray` owned/`UseForeignBuffer` semantics (`contiguous_array.h:29,58-62`). |
| CORE-04 | Model carries full header metadata | Full Model header field set extracted from `tree.h:535-553` (see Model Header Field Set). |
| ERR-01 | Library crates expose typed `thiserror` errors at API boundaries | `thiserror` 2.0.18; per-crate error enums. |
| ERR-02 | Binaries/tests use `anyhow` for context | `anyhow` 1.0.102 in `treelite-harness` + integration tests. |

> Phase 1 also exercises a **minimal subset** of XGB-01 (one XGBoost-JSON model), GTIL-01 (scalar dense predict), EQV-01/EQV-02 (harness skeleton + one golden). Per REQUIREMENTS.md note, those requirements remain *owned* by their dedicated phases (3 and 5); Phase 1 must not claim to complete them.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Enum string round-trip | `treelite-core` (vocabulary layer) | — | Upstream keeps enums as a foundation layer (`include/treelite/enum/`) used by all others. |
| `Model`/`Tree<T>`/`TreeBuf<T>` SoA representation | `treelite-core` (model layer) | — | Central in-memory object; everything else borrows it. Mirrors upstream `tree.h`. |
| Typed errors | each library crate (boundary) | — | `thiserror` enum per crate at its public API (ERR-01). |
| XGBoost-JSON parse → `Model` | `treelite-xgboost` (loader layer) | `treelite-core` (construction target) | Loader depends on core; produces a `Model`. Mirrors upstream `src/model_loader/`. |
| Scalar traversal + base_score + postprocessor | `treelite-gtil` (inference layer) | `treelite-core` (reads model) | Pure read-over-model predict. Mirrors upstream `src/gtil/predict.cc` + `postprocessor.cc`. |
| Golden capture + 1e-5 assertion + manifest | `treelite-harness` (test/dev layer) | all above | Dev-facing; uses `anyhow`. Mirrors the role of upstream's test harness but sources golden from the wheel (D-06). |

**Note:** This is a single-binary/library Rust project — there is no client/server/CDN tiering. "Tier" here means architectural *layer* within the workspace, matching the upstream layer model in `.planning/codebase/ARCHITECTURE.md`.

## Standard Stack

All versions verified 2026-06-10 against the crates.io sparse index (`index.crates.io`) and `cargo search`. Edition 2024 requires Rust ≥ 1.85; local toolchain is 1.95.0 (verified), so all crates below are compatible.

### Core (library crates)

| Crate | Version | Purpose | Why Standard |
|-------|---------|---------|--------------|
| `thiserror` | `2.0.18` | Typed error enums at library API boundaries (ERR-01) | The de-facto derive-macro error crate; v2 is current stable. `[VERIFIED: crates.io sparse index + slopcheck OK]` |
| `serde` | `1.0.228` | Derive for the loader's intermediate structs + manifest (de)serialize | Ecosystem standard; `derive` feature. `[VERIFIED: crates.io + slopcheck OK]` |
| `serde_json` | `1.0.150` | Parse the hand-crafted XGBoost-JSON fixture | Standard JSON parser. NaN/Inf rejection is a *Phase 3* concern; Phase 1 fixture avoids those literals (D-05a / deferred). `[VERIFIED: crates.io + slopcheck OK]` |

### Supporting (harness / dev-test)

| Crate | Version | Purpose | When to Use |
|-------|---------|---------|-------------|
| `anyhow` | `1.0.102` | Error context in `treelite-harness` + integration tests (ERR-02) | Binaries/tests only — never in library public APIs. `[VERIFIED: crates.io + slopcheck OK]` |
| `approx` | `0.5.1` | `assert_abs_diff_eq!`/`assert_relative_eq!` for the 1e-5 check | **Pin 0.5.1, NOT 0.6.0-rc2.** Latest published is a pre-release; FND-02 forbids pre-release on the critical path. `[VERIFIED: crates.io — 0.5.1 is latest stable; 0.6.0-rc1/rc2 are pre-release]` |
| `toml` | `1.1.2` | Read/write the toolchain/libm manifest if TOML chosen (D-07; discretion) | Only if manifest format is TOML. `serde_json` already covers a JSON manifest. `[VERIFIED: crates.io + slopcheck OK]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `approx` | hand-rolled `(a-b).abs() < 1e-5` | Fine for a scalar; `approx` gives better failure messages and is the standard. Either is acceptable for Phase 1's tiny output. |
| `serde_json` for fixture | a bespoke minimal JSON reader | Don't hand-roll JSON. `serde_json` is correct here; the only XGBoost-JSON gotcha (NaN/Inf) is explicitly deferred to Phase 3. |
| `insta` (snapshot) `1.47.2` | committed golden vector | Snapshot testing is for code-generated text, not numerical goldens. The golden is a committed data artifact (D-06), not an `insta` snapshot. Do **not** add `insta` in Phase 1. |
| `half` `2.7.1` | — | f16/bf16 is Phase 9 (PERF-v2), off the equivalence path. **Do not add in Phase 1.** Listed only to mark it explicitly out of scope. |
| `rand` `0.10.1` | — | Seeded random input matrices are EQV-01 (Phase 5). Phase 1's single fixed input vector needs no RNG. **Do not add in Phase 1.** |

**Installation (Phase 1 critical path only):**
```bash
# In each crate's Cargo.toml, reference workspace deps:
# treelite-core:    thiserror, serde (derive)
# treelite-xgboost: thiserror, serde, serde_json
# treelite-gtil:    thiserror
# treelite-harness: anyhow, approx, serde_json (+ toml if manifest is TOML)
```
Root `Cargo.toml`:
```toml
[workspace]
resolver = "3"
members = ["crates/treelite-core", "crates/treelite-xgboost", "crates/treelite-gtil", "crates/treelite-harness"]

[workspace.package]
edition = "2024"

[workspace.dependencies]
thiserror = "2.0.18"
anyhow = "1.0.102"
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.150"
approx = "0.5.1"
toml = "1.1.2"   # only if manifest format is TOML
```

**Golden-capture environment (NOT Cargo deps — a one-time Python script, D-06):**
```bash
pip install treelite==4.7.0 xgboost==3.2.0 numpy
```
`treelite==4.7.0` exactly matches the vendored C++ source `treelite-mainline/` v4.7.0 — this is the correct wheel for the golden. `[VERIFIED: pypi.org — treelite latest 4.7.0; xgboost latest 3.2.0]`

## Package Legitimacy Audit

slopcheck 1.4-class available locally and run this session (`slopcheck install -e crates.io …` and `-e pypi …`).

| Package | Registry | Disposition | slopcheck |
|---------|----------|-------------|-----------|
| thiserror 2.0.18 | crates.io | Approved | [OK] |
| anyhow 1.0.102 | crates.io | Approved | [OK] |
| serde 1.0.228 | crates.io | Approved | [OK] |
| serde_json 1.0.150 | crates.io | Approved | [OK] |
| approx 0.5.1 | crates.io | Approved (pin stable, not 0.6.0-rc2) | [OK] |
| toml 1.1.2 | crates.io | Approved (conditional on TOML manifest) | [OK] |
| half 2.7.1 | crates.io | OUT OF SCOPE (Phase 9) | [OK] |
| rand 0.10.1 | crates.io | OUT OF SCOPE (Phase 5) | [OK] |
| insta 1.47.2 | crates.io | NOT RECOMMENDED for Phase 1 | [OK] |
| treelite 4.7.0 | PyPI (golden capture only) | Approved | [OK] |
| xgboost 3.2.0 | PyPI (golden capture only) | Approved | [OK] |
| numpy | PyPI (golden capture only) | Approved | [OK] |

**Packages removed due to slopcheck [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** none. All packages are long-lived, high-download, official-source crates/wheels.

## Upstream Anchor Reference (the porting spec)

### Enum String Table — port these EXACTLY (ENUM-01)

The string forms are **not uniform** across the four enums — this is the single highest-risk ENUM detail. `TaskType` uses `kXxx`-style strings; the others use lowercase/symbolic. `[VERIFIED: read from src/enum/*.cc this session]`

| Enum | Variant | Exact string | Source |
|------|---------|--------------|--------|
| `TaskType` | kBinaryClf / kRegressor / kMultiClf / kLearningToRank / kIsolationForest | `"kBinaryClf"`, `"kRegressor"`, `"kMultiClf"`, `"kLearningToRank"`, `"kIsolationForest"` | `src/enum/task_type.cc:15-47` |
| `TreeNodeType` | kLeafNode / kNumericalTestNode / kCategoricalTestNode | `"leaf_node"`, `"numerical_test_node"`, `"categorical_test_node"` | `src/enum/tree_node_type.cc:15-39` |
| `Operator` | kEQ / kLT / kLE / kGT / kGE | `"=="`, `"<"`, `"<="`, `">"`, `">="` (note: `kNone` → `""`) | `src/enum/operator.cc:16-49` |
| `DType` (≡ upstream `TypeInfo`) | kInvalid / kUInt32 / kFloat32 / kFloat64 | `"invalid"`, `"uint32"`, `"float32"`, `"float64"` | `src/enum/typeinfo.cc:15-42` |

Underlying integer reprs (for the `#[repr]` to match wire-format expectations later): `TaskType: u8 {0..4}` (`task_type.h:19`), `TreeNodeType: i8 {0,1,2}` (`tree_node_type.h:17`), `Operator: i8 {kNone=0,kEQ=1,kLT=2,kLE=3,kGT=4,kGE=5}` (`operator.h:17`), `TypeInfo: u8 {kInvalid=0,kUInt32=1,kFloat32=2,kFloat64=3}` (`typeinfo.h:21`). `FromString` on an unknown value is a fatal error upstream → Rust returns a `thiserror` `Err`, not a panic.

**DType Phase 1 coverage (discretion, ENUM-01):** include all four (`invalid/uint32/float32/float64`) — the set is tiny and round-tripping all four against the upstream strings is the cleanest way to satisfy "asserted against values read from `treelite-mainline`."

### Tree SoA Field Set — the ~20 parallel columns (CORE-02)

`[VERIFIED: tree.h:97-132]`. `Tree<ThresholdType, LeafOutputType>` where in Phase 1 ThresholdType == LeafOutputType ∈ {f32, f64} (the `static_assert` at `tree.h:81-86` forbids mixed types). XGBoost-JSON always yields `<f32,f32>` (see Loader note). Each is a `ContiguousArray<...>` column indexed by node id:

| Column | Element type | Notes |
|--------|--------------|-------|
| `node_type_` | `TreeNodeType` | leaf / numerical / categorical |
| `cleft_` | `i32` | left child; `-1` ⇒ leaf (see `IsLeaf`, `tree.h:204`) |
| `cright_` | `i32` | right child |
| `split_index_` | `i32` | feature index |
| `default_left_` | `bool` | missing-value direction |
| `leaf_value_` | `LeafOutputType` | scalar leaf output |
| `threshold_` | `ThresholdType` | numerical split threshold |
| `cmp_` | `Operator` | comparison op (XGBoost always `kLT`) |
| `category_list_right_child_` | `bool` | categorical polarity (unused in Phase 1 fixture) |
| `leaf_vector_` | `LeafOutputType` | flattened leaf vectors |
| `leaf_vector_begin_` / `leaf_vector_end_` | `u64` | CSR-style offsets into `leaf_vector_` |
| `category_list_` | `u32` | flattened category lists |
| `category_list_begin_` / `category_list_end_` | `u64` | CSR offsets into `category_list_` |
| `data_count_` | `u64` | node statistic (optional) |
| `sum_hess_` | `f64` | node statistic (optional) |
| `gain_` | `f64` | node statistic (optional) |
| `data_count_present_` / `sum_hess_present_` / `gain_present_` | `bool` | presence flags |

Plus scalar tree fields: `has_categorical_split_: bool` (`tree.h:126`), `num_nodes: i32` (`tree.h:158`), and the serialization-recomputed `num_opt_field_per_tree_`/`num_opt_field_per_node_` (`tree.h:131-132` — can be deferred, they're serialization bookkeeping, Phase 2).

**Phase 1 minimum to be correct:** `node_type_`, `cleft_`, `cright_`, `split_index_`, `default_left_`, `leaf_value_`, `threshold_`, `cmp_`, plus `num_nodes`. The leaf-vector and category columns must *exist* (CORE-02 says "all upstream node fields") but for the `binary:logistic` fixture they are empty (`leaf_vector_begin_ == leaf_vector_end_` ⇒ `HasLeafVector` false, `tree.h:233-235`). Node statistics columns can be present-but-empty.

### Model Header Field Set (CORE-04)

`[VERIFIED: tree.h:535-553]`. **Important deviation from the ROADMAP wording:** in v4.7.0 `num_class` and `leaf_vector_shape` are `ContiguousArray<i32>` (arrays), and `target_id`/`class_id` are per-tree `ContiguousArray<i32>` — they are NOT scalars.

| Field | Type | Source | Phase 1 value for `binary:logistic` |
|-------|------|--------|--------------------------------------|
| `num_feature` | `i32` | `tree.h:535` | from fixture `learner_model_param.num_feature` |
| `task_type` | `TaskType` | `tree.h:537` | `kBinaryClf` (objective starts `binary:`) |
| `average_tree_output` | `bool` | `tree.h:539` | `false` (XGBoost loader hardcodes false, `delegated_handler.cc:814`) |
| `num_target` | `i32` | `tree.h:542` | `1` |
| `num_class` | `ContiguousArray<i32>` | `tree.h:543` | `[1]` (binary clf, num_class field ≤ 1 ⇒ branch at `delegated_handler.cc:856`) |
| `leaf_vector_shape` | `ContiguousArray<i32>` | `tree.h:544` | `[1, 1]` |
| `target_id` | `ContiguousArray<i32>` (per tree) | `tree.h:546` | `tree_info[i]` (= `[0]` for single-target) |
| `class_id` | `ContiguousArray<i32>` (per tree) | `tree.h:547` | `[0]` |
| `postprocessor` | `String` | `tree.h:549` | `"sigmoid"` (from objective map) |
| `sigmoid_alpha` | `f32` | `tree.h:550` | `1.0` (default; not in fixture) |
| `ratio_c` | `f32` | `tree.h:551` | `1.0` (default) |
| `base_scores` | `ContiguousArray<f64>` | `tree.h:552` | margin-transformed base score (see below) |
| `attributes` | `String` | `tree.h:553` | `""` (JSON string blob, may be empty) |

Also-present-but-private (serialization bookkeeping, Phase 2): `num_tree_`, `num_opt_field_per_model_`, `major_ver_/minor_ver_/patch_ver_`, `threshold_type_`, `leaf_output_type_` (`tree.h:556-567`). The version triple comes from `TREELITE_VER_*` — for Phase 1 a fixed `{4,7,0}` is acceptable.

### Two-Variant Model (CORE-01)

Upstream: `using ModelPresetVariant = std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` (`tree.h:437`); `Model` holds `variant_` and dispatches via `std::visit`. Rust port: a two-variant `enum ModelVariant { F32(ModelPreset<f32>), F64(ModelPreset<f64>) }` (or generic `Tree<T>` behind it) with a `match` instead of `std::visit`. The header metadata lives on `Model` itself (outside the variant), exactly as upstream (`tree.h:454-573`). XGBoost-JSON only ever produces the `F32` variant (`xgboost_json.cc:145` casts to `ModelPreset<float,float>`).

### Owned/Borrowed TreeBuf (CORE-03)

Upstream `ContiguousArray<T>` has a `bool owned_buffer_` and `UseForeignBuffer(void* prealloc_buf, std::size_t size)` for non-owning zero-copy aliasing (`contiguous_array.h:29,58-62`); `static_assert(std::is_pod<T>)`. Rust port options (discretion, D-02 area): an `enum TreeBuf<T> { Owned(Vec<T>), Borrowed(*const T, usize) }`, or `Cow<'a, [T]>`. The borrowed mode's real consumer is the Python buffer protocol (Phase 8, MEM-04) — Phase 1 only needs the *type to support both modes* and round-trip a tiny owned buffer; a borrowed-mode unit test (construct a `TreeBuf` over a borrowed slice, read it back) satisfies CORE-03 without needing Python. Note `std::is_pod` ⇒ Rust elements should be `Copy`/plain-old-data; this is the `bytemuck::Pod` seam deferred to Phase 9 (MEM-01) — don't pull in `bytemuck` yet.

## Architecture Patterns

### System Architecture Diagram

```
  committed fixture                committed golden artifact
  model.json (XGBoost-JSON)        golden.{json}: { input[], output[], manifest{} }
        │                                   │
        ▼                                   │
 ┌─────────────────────┐                    │
 │  treelite-xgboost   │  serde_json parse  │
 │  LoadXGBoostJSON    │  → intermediate    │
 │  (port of           │    structs         │
 │   delegated_handler │                    │
 │   + LearnerHandler  │                    │
 │   ::EndObject math) │                    │
 └─────────┬───────────┘                    │
           │ builds                          │
           ▼                                  │
 ┌─────────────────────┐                     │
 │   treelite-core     │  Model{F32 variant, │
 │   Model + Tree<f32> │   header metadata,  │
 │   SoA TreeBuf cols  │   base_scores[],    │
 │   (enums, header)   │   postprocessor=    │
 └─────────┬───────────┘   "sigmoid"}        │
           │ borrowed-read                    │
           ▼                                   │
 ┌─────────────────────┐    input[] ──────────┤
 │   treelite-gtil     │◄─────────────────────┘
 │  predict(model,     │
 │   input) →          │  per row:
 │  EvaluateTree       │   sum over trees (serial in tree_id)
 │  + base_score add   │   + base_scores[target,class]  (f64)
 │  + sigmoid/identity │   → sigmoid(alpha * margin)
 └─────────┬───────────┘
           │ output[]
           ▼
 ┌─────────────────────┐
 │  treelite-harness   │  assert |output[i] - golden[i]| < 1e-5
 │  (anyhow)           │  report max |Δ|; check manifest matches env
 └─────────────────────┘
```

### Recommended Project Structure

```
treeline_rs/
├── Cargo.toml                      # [workspace], resolver "3", [workspace.dependencies]
├── crates/
│   ├── treelite-core/
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── enums.rs            # TaskType, TreeNodeType, Operator, DType + to/from string
│   │       ├── tree_buf.rs         # TreeBuf<T> owned + borrowed
│   │       ├── tree.rs             # Tree<T> SoA columns + getters
│   │       ├── model.rs            # Model enum + header metadata
│   │       └── error.rs            # thiserror CoreError
│   ├── treelite-xgboost/
│   │   └── src/
│   │       ├── lib.rs              # load_xgboost_json(&str) -> Result<Model, XgbError>
│   │       ├── objective.rs        # objective -> postprocessor map + ProbToMargin
│   │       └── error.rs            # thiserror XgbError
│   ├── treelite-gtil/
│   │   └── src/
│   │       ├── lib.rs              # predict(&Model, &[f32]) -> Vec<f32>
│   │       ├── postprocessor.rs    # identity, sigmoid
│   │       └── error.rs            # thiserror GtilError
│   └── treelite-harness/
│       ├── src/lib.rs              # load golden, assert 1e-5, report max dev
│       └── tests/equivalence.rs    # the end-to-end test (anyhow)
├── fixtures/
│   ├── binary_logistic.model.json  # hand-crafted XGBoost-JSON (D-04)
│   ├── golden.json                 # { input, output, manifest } (D-06/D-07)
│   └── capture_golden.py           # one-time script, treelite==4.7.0 (committed for provenance)
└── treelite-mainline/  xgboost-master/   (vendored, read-only)
```
The existing `src/main.rs` stub is removed; the package becomes a virtual workspace root (no `[package]` at root).

### Pattern 1: Struct-of-Arrays Tree (port verbatim)
**What:** Store every node field as a separate `TreeBuf<T>` column indexed by node id, not a `Node` struct.
**When to use:** Always — it's the upstream invariant (cache-friendly traversal + zero-copy serialization). Anti-pattern to deviate.
**Example (Rust shape, derived from `tree.h:97-127` + getters `tree.h:169-335`):**
```rust
// Derived from treelite-mainline/include/treelite/tree.h:97-206
pub struct Tree<T> {
    node_type: TreeBuf<TreeNodeType>,
    cleft: TreeBuf<i32>,
    cright: TreeBuf<i32>,
    split_index: TreeBuf<i32>,
    default_left: TreeBuf<bool>,
    leaf_value: TreeBuf<T>,
    threshold: TreeBuf<T>,
    cmp: TreeBuf<Operator>,
    // ... leaf_vector / category / stat columns (present, empty for binary:logistic)
    pub num_nodes: i32,
}
impl<T: Copy> Tree<T> {
    #[inline] pub fn is_leaf(&self, nid: usize) -> bool { self.cleft[nid] == -1 }     // tree.h:204
    #[inline] pub fn left_child(&self, nid: usize) -> i32 { self.cleft[nid] }          // tree.h:169
    #[inline] pub fn default_child(&self, nid: usize) -> i32 {                         // tree.h:183
        if self.default_left[nid] { self.cleft[nid] } else { self.cright[nid] }
    }
}
```

### Pattern 2: Scalar traversal (port verbatim from `EvaluateTree`)
**What:** Walk from node 0; at each internal node read the feature, route on NaN→default child else compare.
**Example (derived from `predict.cc:152-172`):**
```rust
// Derived from treelite-mainline/src/gtil/predict.cc:152-172
fn evaluate_tree<T: Copy + Into<f64>>(tree: &Tree<T>, row: &[f32]) -> usize {
    let mut nid = 0usize;
    while !tree.is_leaf(nid) {
        let fvalue = row[tree.split_index(nid) as usize];
        nid = if fvalue.is_nan() {
            tree.default_child(nid) as usize          // missing → default direction
        } else {
            // XGBoost numerical: Operator::kLT, threshold is f32
            next_node(fvalue, tree.threshold(nid), tree.comparison_op(nid),
                      tree.left_child(nid), tree.right_child(nid)) as usize
        };
    }
    nid
}
```
The Phase 1 fixture has no categorical nodes, so the `NextNodeCategorical` branch (`predict.cc:127-150`) need not be ported (it's Phase 5 / GTIL-06).

### Pattern 3: Predict assembly order (the numerical contract)
The exact arithmetic order from `PredictRaw` (`predict.cc:231-305`) then `ApplyPostProcessor` (`predict.cc:307-323`) — this ordering IS the 1e-5 contract:
1. Output buffer filled with `InputT{}` zeros (`predict.cc:238`).
2. For each row, for each tree **serial in tree_id order**, add the leaf value to `output[row, target_id[tree], class_id[tree]]` (`predict.cc:245-254`, `OutputLeafValue` cast `static_cast<InputT>` at `:228`).
3. (Skip averaging — `average_tree_output == false` for XGBoost.)
4. Add `base_scores[target,class]` — **base_scores is f64**, added into the InputT accumulator (`predict.cc:294-304`).
5. Postprocessor: `default` ⇒ apply sigmoid; `raw` ⇒ skip. Sigmoid: `1/(1 + exp(-sigmoid_alpha * val))` with `sigmoid_alpha` f32 (`postprocessor.cc:33-37`).

For `binary:logistic`: `num_target=1, num_class=[1]`, so output is shape `(num_row, 1, 1)` (`output_shape.cc:27`) — effectively one scalar per row.

### Anti-Patterns to Avoid
- **A `Node` struct instead of SoA columns** — breaks the upstream representation and the future zero-copy/serialization story. Use parallel `TreeBuf` columns.
- **Copying `Model`/`Tree`** — upstream deletes copy ctors (`tree.h:90-91,462-463`); Rust should make them move-only / explicit `clone()`, not derive `Clone` casually.
- **Introducing a `Predictor`/backend trait now** — explicitly deferred (D-08). A plain `fn predict` is correct for Phase 1.
- **Computing the sigmoid base_score transform in f32** — `ProbToMargin::Sigmoid` is `-log(1/p - 1)` in **f64** (`xgboost.h:17-19`); porting it in f32 will miss 1e-5.
- **Reordering tree summation** — must be serial in `tree_id` (GTIL-08); float addition is non-associative.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parse the fixture JSON | a custom JSON tokenizer | `serde_json` | Correct, fast, standard. (NaN/Inf is a Phase 3 concern, deferred.) |
| Error types | bespoke `enum` + manual `Display`/`Error` impls | `thiserror` derive | ERR-01 mandates it; less boilerplate, correct `source()`. |
| Error context in harness | string concatenation | `anyhow` + `.context()` | ERR-02 mandates it. |
| Float approx assert | ad-hoc epsilon logic with poor messages | `approx` (`assert_abs_diff_eq!`) | Better diagnostics; standard. (Either approach is acceptable for one scalar.) |
| The enum string maps | guessing the strings | the Enum String Table above | The strings are non-uniform and asserted against upstream (ENUM-01). |
| objective→postprocessor + base_score transform | re-deriving the math | port `xgboost.cc:28-60` + `xgboost.h:16-23` verbatim | Getting the margin transform wrong silently breaks 1e-5. |

**Key insight:** In this phase the danger is never "missing a library" — it's *re-deriving numerical logic that already exists verbatim upstream*. Every arithmetic step has a cited source line; copy it, don't reinvent it.

## Runtime State Inventory

> Phase 1 is greenfield code creation, not a rename/refactor/migration. This section is included only to record the one piece of pre-existing state being replaced.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores in this project. | None. |
| Live service config | None — no running services. | None. |
| OS-registered state | None. | None. |
| Secrets/env vars | None. | None. |
| Build artifacts | `src/main.rs` (`fn main()` stub) + root `[package]` `Cargo.toml` (`treeline_rs` 0.1.0, edition 2024, empty deps) — verified by reading both this session. | Replace root `Cargo.toml` with a `[workspace]` virtual manifest; delete/relocate `src/main.rs` (no longer a package root). |

**Note:** `treelite-mainline/` and `xgboost-master/` are vendored read-only reference; they are not built and must not be modified.

## Common Pitfalls

### Pitfall 1: Wrong enum string forms
**What goes wrong:** Using `"kBinaryClf"`-style strings for `Operator`/`TreeNodeType`/`DType`, or symbolic for `TaskType`. They're inconsistent across the four enums.
**Why:** Natural to assume uniformity.
**How to avoid:** Use the Enum String Table verbatim; assert each against the cited `.cc` line. (ENUM-01's "asserted against values read from `treelite-mainline`" — read the four `.cc` files in the test or hardcode the values from them with the line cited.)
**Warning sign:** ENUM round-trip test fails for exactly one enum family.

### Pitfall 2: base_score not transformed to margin
**What goes wrong:** Predictions are off by a constant. For `binary:logistic` with `base_score=0.5`, the margin is `-log(1/0.5 - 1) = 0`, which *masks* the bug; pick a fixture `base_score != 0.5` (e.g. 0.25) so the transform is genuinely exercised.
**Why:** XGBoost ≥1.0 stores the user-space probability; Treelite transforms it to margin space at load (`delegated_handler.cc:891-897`, gated on `version[0] >= 1`).
**How to avoid:** Port `TransformBaseScoreToMargin` (`xgboost.cc:52-60`) + `ProbToMargin::Sigmoid` (`xgboost.h:17-19`) in f64; apply when `version` is empty or `version[0] >= 1`. Set the fixture `"version": [4, 7, 0]` (or any major ≥ 1) so the gate fires.
**Warning sign:** Constant offset between Rust and golden.

### Pitfall 3: f32/f64 cast ordering drift
**What goes wrong:** Max deviation hovers just above 1e-5.
**Why:** The accumulator is `InputT` (f32 for the float predict path), leaf values cast `static_cast<InputT>`, but `base_scores` is f64 added into the f32 accumulator, and `sigmoid_alpha` is f32. Doing the whole chain in f64 (or all f32) changes the last ULPs.
**How to avoid:** Mirror the exact types: f32 accumulator, f32 leaf/threshold, f64 base_scores added in, f32 sigmoid_alpha, `exp` on the f32 value. The model is `<f32,f32>`; the predict `InputT` for an f32 fixture is f32.
**Warning sign:** Deviation ~1e-6..1e-5 that won't go away.

### Pitfall 4: libm divergence between Rust and the Treelite wheel
**What goes wrong:** `sigmoid` uses `exp`; Rust's `f32::exp`/`f64::exp` and the wheel's `std::exp` may differ in the last ULP across libm versions.
**Why:** Transcendental functions aren't bit-identical across libms.
**How to avoid:** This is exactly why D-07 mandates a manifest (glibc/libm version captured). 1e-5 tolerance comfortably absorbs single-ULP `exp` divergence for one scalar; the manifest documents the environment so a future failure is diagnosable. Capture and compare on the same OS/arch where practical.
**Warning sign:** Passes locally, fails on a different distro — check the manifest.

### Pitfall 5: Fixture not loadable by the upstream wheel (D-05 coupling)
**What goes wrong:** `treelite.gtil` can't load the hand-crafted JSON, so no golden can be captured.
**Why:** The loader requires a specific nesting and a specific set of per-tree arrays (`left_children`, `right_children`, `split_indices`, `split_conditions`, `default_left`, `split_type`, plus `tree_param.num_nodes`/`size_leaf_vector`) — `delegated_handler.cc:484-490` lists the recognized keys; `:425-432` validates array lengths == num_nodes.
**How to avoid:** Author the fixture against the *recognized-key list and length checks* in `delegated_handler.cc`, not the `saving_model.rst` config example (which is a config dump, not a model). Required nesting: `{"learner": {"learner_model_param": {num_feature, num_class, num_target, base_score}, "gradient_booster": {"name":"gbtree","model": {"trees":[...], "tree_info":[0], "gbtree_model_param":{num_trees,...}}}, "objective": {"name":"binary:logistic"}}, "version":[4,7,0]}`. Each tree object needs `tree_param:{num_nodes, size_leaf_vector:"0"}`, the parallel arrays, and `split_type` (all 0 = numerical). Validate by actually running `capture_golden.py` against `treelite==4.7.0` — if it loads, the fixture is correct.
**Warning sign:** Python raises in `treelite.frontend.load_xgboost_model_legacy_binary`/`from_xgboost`.

## Code Examples

### objective → postprocessor map + base_score transform (port verbatim)
```rust
// Derived from treelite-mainline/src/model_loader/detail/xgboost.cc:28-60
//                + .../detail/xgboost.h:16-23
fn get_postprocessor(objective: &str) -> &'static str {
    match objective {
        "multi:softmax" | "multi:softprob" => "softmax",
        "reg:logistic" | "binary:logistic" => "sigmoid",
        "count:poisson" | "reg:gamma" | "reg:tweedie"
        | "survival:cox" | "survival:aft" => "exponential",
        "binary:hinge" => "hinge",
        _ => "identity", // reg:squarederror, reg:linear, binary:logitraw, rank:*, ...
    }
}
fn prob_to_margin_sigmoid(base_score: f64) -> f64 { -((1.0 / base_score) - 1.0).ln() }
fn transform_base_score_to_margin(postproc: &str, base_score: f64) -> f64 {
    match postproc {
        "sigmoid"     => prob_to_margin_sigmoid(base_score),
        "exponential" => base_score.ln(),
        _             => base_score,
    }
}
```

### sigmoid / identity postprocessor (port verbatim)
```rust
// Derived from treelite-mainline/src/gtil/postprocessor.cc:20,33-37
fn identity(_alpha: f32, v: f32) -> f32 { v }
fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 {
    1.0f32 / (1.0f32 + (-sigmoid_alpha * v).exp())
}
```

### Golden-capture script (one-time, committed for provenance — D-06)
```python
# fixtures/capture_golden.py  — run once, never in CI
import json, platform, treelite, treelite.gtil, numpy as np, xgboost, ctypes.util
model = treelite.frontend.load_xgboost_model("fixtures/binary_logistic.model.json")
X = np.array([[ ... ]], dtype=np.float32)          # the committed input matrix
y = treelite.gtil.predict(model, X, pred_margin=False)  # default => sigmoid applied
json.dump({
    "input": X.tolist(),
    "output": np.asarray(y).ravel().tolist(),
    "manifest": {
        "treelite": treelite.__version__,          # expect 4.7.0
        "xgboost": xgboost.__version__,            # 3.2.0
        "os": platform.platform(),
        "arch": platform.machine(),
        "libc": platform.libc_ver(),               # glibc version
        "python": platform.python_version(),
    },
}, open("fixtures/golden.json", "w"), indent=2)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cargo `resolver = "2"` opt-in | `resolver = "3"` is the **default** for edition-2024 crates | Rust 1.84 (edition 2024 stable, early 2025) | Phase 1 can rely on resolver 3 via `edition = "2024"`; explicit `resolver = "3"` in the virtual workspace is still recommended (workspaces don't inherit edition). `[CITED: doc.rust-lang.org/edition-guide]` |
| `thiserror` 1.x | `thiserror` 2.x | 2024 | v2 is current; minor API tightening around `#[from]`/display. Use 2.0.18. `[VERIFIED: crates.io]` |
| XGBoost scalar `base_score` | vector `base_score` | XGBoost 3.1 | Phase 1 fixture uses scalar form (simpler); loader handles both (`delegated_handler.cc:878-889`). `[CITED: saving_model.rst:307-314]` |
| XGBoost JSON schema file | removed in 3.2 (structure unchanged) | XGBoost 3.2 | Hand-craft against Treelite's loader key list, not a schema file. `[CITED: saving_model.rst:313]` |

**Deprecated/outdated for Phase 1:**
- `approx` `0.6.0-rc2`: pre-release — forbidden by FND-02. Use `0.5.1`.
- The root `src/main.rs` stub + `[package]` `Cargo.toml`: replaced by the workspace manifest.

## Validation Architecture

> nyquist_validation: no `.planning/config.json` was found, so this section is included (absent key ⇒ enabled).

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]` + `cargo test`) + `approx` for float asserts |
| Config file | none — standard Cargo layout (per-crate `tests/` + inline `#[cfg(test)]`) |
| Quick run command | `cargo test -p treelite-core` (enum + model unit tests) |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| FND-01/02 | Workspace builds; deps pinned, no pre-release | smoke | `cargo build --workspace` | ❌ Wave 0 |
| ENUM-01 | Each enum round-trips to/from exact upstream string | unit | `cargo test -p treelite-core enums` | ❌ Wave 0 |
| CORE-01 | `Model` is two-variant; constructs both f32/f64 | unit | `cargo test -p treelite-core model` | ❌ Wave 0 |
| CORE-02 | `Tree<T>` exposes all SoA columns; getters match upstream semantics | unit | `cargo test -p treelite-core tree` | ❌ Wave 0 |
| CORE-03 | `TreeBuf<T>` round-trips in both owned and borrowed modes | unit | `cargo test -p treelite-core tree_buf` | ❌ Wave 0 |
| CORE-04 | Header metadata fields present + settable | unit | `cargo test -p treelite-core header` | ❌ Wave 0 |
| ERR-01 | Unknown enum string / malformed model ⇒ typed `Err`, no panic | unit | `cargo test -p treelite-xgboost error` | ❌ Wave 0 |
| ERR-02 | Harness surfaces context via `anyhow` | integration | `cargo test -p treelite-harness` | ❌ Wave 0 |
| XGB-01 (subset) | Fixture loads into a `Model` (f32 variant, postprocessor=sigmoid) | integration | `cargo test -p treelite-xgboost load_fixture` | ❌ Wave 0 |
| GTIL-01 (subset) | Scalar predict over the fixture input | integration | `cargo test -p treelite-gtil predict` | ❌ Wave 0 |
| EQV-02 (subset) | **The spine test:** load → predict → assert within 1e-5 of golden; report max |Δ| | integration | `cargo test -p treelite-harness equivalence` | ❌ Wave 0 |

### Equivalence-Harness Design (the 1e-5 instrument)
1. **Golden artifact** = a single committed file (`fixtures/golden.json`) holding `{ input[][], output[], manifest{} }` — captured once from `treelite==4.7.0` GTIL (D-06), never regenerated in CI. The manifest (D-07) records treelite/xgboost versions, OS/arch, glibc/libm version, python version.
2. **Harness** (`treelite-harness`, uses `anyhow`): reads the fixture model + the golden file, runs `treelite-xgboost::load` then `treelite-gtil::predict`, then asserts `abs_diff_eq!(rust[i], golden.output[i], epsilon = 1e-5)` for each element and **reports the max observed |Δ|** (forward-looking to EQV-04, even though EQV-04 is owned by Phase 5).
3. **Manifest check:** harness reads `manifest` and emits a warning (not a failure) if the running glibc/OS differs from capture-time — so a libm-divergence failure is immediately diagnosable.
4. **Why default (not raw) predict:** capturing the default-kind prediction exercises the sigmoid postprocessor end-to-end (the real fidelity check); `pred_margin=False` in the capture script.

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-under-edit>` (sub-second for unit crates).
- **Per wave merge:** `cargo test --workspace`.
- **Phase gate:** `cargo build --workspace && cargo test --workspace` green (Success Criterion 1) + the `treelite-harness` equivalence test green within 1e-5 (Success Criterion 4) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `fixtures/binary_logistic.model.json` — hand-crafted, validated by loading in `capture_golden.py` (D-04/D-05).
- [ ] `fixtures/golden.json` + `fixtures/capture_golden.py` — golden + manifest, captured from `treelite==4.7.0` (D-06/D-07).
- [ ] `crates/treelite-core/tests/` — enum round-trip, model variant, tree SoA, tree_buf owned/borrowed, header.
- [ ] `crates/treelite-xgboost/tests/` — load fixture, error path.
- [ ] `crates/treelite-gtil/tests/` — predict over fixture, sigmoid/identity unit tests.
- [ ] `crates/treelite-harness/tests/equivalence.rs` — the end-to-end 1e-5 spine test.
- [ ] No framework install needed — Rust's built-in test harness is sufficient; `approx` is the only test-time dep.

## Security Domain

> No `.planning/config.json` found ⇒ `security_enforcement` treated as enabled. This is an offline numerical library with no network/auth/session surface; most ASVS categories are N/A.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface. |
| V3 Session Management | no | No sessions. |
| V4 Access Control | no | No multi-user access. |
| V5 Input Validation | yes | The XGBoost-JSON loader parses untrusted-shaped input. Validate array-length invariants (mirror `delegated_handler.cc:425-432`) and return typed `Err` on malformed input rather than indexing out of bounds. `serde_json` handles malformed-JSON safely. |
| V6 Cryptography | no | No crypto. |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds node index from a malformed/adversarial fixture | Tampering / DoS | Bounds-checked indexing or explicit length validation before traversal; the harness fixture is trusted, but the loader API should not panic on bad input (ERR-01). |
| Panic across a future FFI boundary (Phase 8) | DoS | Out of scope for Phase 1 (no PyO3 yet); noted so the typed-error discipline established now carries forward. |
| Slopsquatted dependency | Tampering | All deps verified via slopcheck [OK] this session + pinned exact versions in `[workspace.dependencies]`. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The hand-crafted fixture nesting described in Pitfall 5 is sufficient for `treelite==4.7.0` to load — derived from the loader's recognized-key list, not executed end-to-end this session. | Common Pitfalls / Code Examples | Golden capture fails until the JSON matches what the wheel's loader expects; mitigated by running `capture_golden.py` during implementation (it either loads or it doesn't — fast feedback). |
| A2 | `treelite.gtil.predict(model, X, pred_margin=False)` applies the sigmoid postprocessor for a `binary:logistic` model (default predict kind). | Validation Architecture / capture script | If the wheel's default differs, the golden would be raw margins; verify the captured output is in (0,1) for sigmoid. |
| A3 | The local glibc/libm (2.39) is close enough to the golden-capture environment that `exp` divergence stays < 1e-5 for the chosen scalar. | Pitfall 4 | A1-class environment drift; D-07 manifest exists precisely to diagnose this. Capture on the same machine where the harness runs to eliminate it. |
| A4 | Rust `f32::exp`/`f64::ln` map to the platform libm closely enough for 1e-5. | Pitfall 4 | Same as A3; one scalar within 1e-5 is a very loose bar for single-ULP `exp` differences. |

**These four are the only non-verified claims.** Every enum string, SoA field, header field, traversal step, postprocessor formula, and crate version in this document was verified against a vendored file or the live registry this session.

## Open Questions

1. **Exact `treelite.gtil.predict` API signature in 4.7.0 (pred kind keyword).**
   - What we know: 4.7.0 wheel is installed and matches the C++ source; GTIL exposes default/raw/leaf_id/score_per_tree (`config.cc:24-31`).
   - What's unclear: whether the Python wrapper uses `pred_margin=` vs a `predict_type=` string in 4.7.0.
   - Recommendation: the planner's golden-capture task should `help(treelite.gtil.predict)` first and use whatever 4.7.0 exposes; the captured numbers are what matter, not the keyword.

2. **Manifest format (TOML vs JSON vs markdown) — D-07 discretion.**
   - Recommendation: embed the manifest *inside* `golden.json` (as the capture script does) so the golden and its provenance are one atomic file. This avoids adding the `toml` crate to the critical path. If a standalone TOML manifest is preferred, add `toml = "1.1.2"` to harness deps only.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo | All crates (edition 2024 needs ≥1.85) | ✓ | 1.95.0 | — |
| Python 3 | Golden capture (D-06), one-time | ✓ | 3.12.3 | — |
| `treelite` wheel | Golden capture | ✗ (not yet installed) | target 4.7.0 (on PyPI ✓) | `pip install treelite==4.7.0` |
| `xgboost` wheel | Golden capture (optional; fixture is hand-crafted) | ✗ | 3.2.0 on PyPI ✓ | Not strictly required — D-04 fixture is hand-written; xgboost only needed if validating the fixture trains/loads in xgboost itself |
| `numpy` | Golden capture | ✗ | on PyPI ✓ | `pip install numpy` |
| glibc / libm | Runtime `exp` for sigmoid; manifest provenance | ✓ | glibc 2.39 | — |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** the three PyPI packages are install-on-demand for the one-time golden capture; all verified present on PyPI and slopcheck [OK]. They are NOT Cargo dependencies and never enter CI.

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/include/treelite/enum/{task_type,tree_node_type,operator,typeinfo}.h` + `src/enum/*.cc` — exact enum string values and integer reprs (ENUM-01).
- `treelite-mainline/include/treelite/tree.h` (lines 78-573) — Tree SoA columns, ModelPreset variant, Model header field set (CORE-01/02/04).
- `treelite-mainline/include/treelite/contiguous_array.h` — owned/borrowed buffer semantics (CORE-03).
- `treelite-mainline/src/model_loader/xgboost_json.cc` + `detail/xgboost_json/delegated_handler.{h,cc}` + `detail/xgboost.{h,cc}` — XGBoost-JSON field list, metadata math, objective map, base_score margin transform.
- `treelite-mainline/src/gtil/{predict.cc,postprocessor.cc,config.cc,output_shape.cc}` — scalar traversal, sigmoid/identity, predict kinds, output shape.
- `xgboost-master/doc/tutorials/saving_model.rst` (lines 175-315) — JSON format history, schema-removal note, base_score versioning.
- crates.io sparse index (`index.crates.io`) + `cargo search` — all crate versions (this session, 2026-06-10).
- PyPI JSON API — `treelite` 4.7.0, `xgboost` 3.2.0 availability (this session).
- slopcheck (local) — legitimacy [OK] for all 12 packages (this session).
- Local toolchain probe — rustc 1.95.0, Python 3.12.3, glibc 2.39 (this session).

### Secondary (MEDIUM confidence)
- `doc.rust-lang.org/edition-guide` — resolver 3 default for edition 2024 `[CITED, not re-fetched this session — training knowledge of the edition guide]`.

### Tertiary (LOW confidence)
- None — no claim in this document rests on unverified web search.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every version verified against the live crates.io index + slopcheck this session.
- Architecture / representation: HIGH — every field, string, and arithmetic step cited to a vendored upstream line read this session.
- Numerical equivalence (1e-5): MEDIUM — the *logic* is verified verbatim; the *bit-level outcome* depends on libm/cast ordering (Assumptions A3/A4), which the manifest (D-07) is designed to manage.
- Fixture loadability: MEDIUM — structure derived from the loader's recognized-key list (A1); confirmed only by running `capture_golden.py` during implementation.

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (stable domain; vendored C++ is frozen at v4.7.0, so the porting spec never drifts — only crate versions might; re-check before pinning if planning slips past this window).

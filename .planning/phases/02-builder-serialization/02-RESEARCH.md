# Phase 2: Builder & Serialization - Research

**Researched:** 2026-06-10
**Domain:** Faithful C++→Rust port of Treelite v4.7.0 model construction + v5 binary/JSON serialization
**Confidence:** HIGH (the entire spec is vendored read-only at `treelite-mainline/` and was read line-by-line; the upstream wheel that produces the golden blob is already installed at `.venv/.../treelite-4.7.0`)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01 (binary v5 = byte-identical cross-tool):** v5 binary output MUST be byte-for-byte interoperable with upstream Treelite v5 — match exact field order, type tags, `PyBufferFrame` framing, version header, opt-field counts that `serializer.cc` emits. Strongest expression of the faithful-port promise; yields a strong golden-blob test.
- **D-02 (byte-fidelity validation artifact):** Validate D-01 against a **committed upstream-wheel-produced v5 blob + a toolchain/libm manifest**, captured once and frozen — mirroring Phase 1's golden-vector approach. CI never compiles C++. (Claude's discretion on exact capture mechanics; artifact + manifest mandatory.)
- **D-03 (deserialize rejects non-v5):** Deserializer accepts only v5; non-v5 input (v3.9/v4.0 headers) is rejected with a typed error. Do NOT port `DeserializeHeaderAndCreateModelV3`.
- **D-04 (DumpAsJSON matches upstream structure):** `DumpAsJSON` (SER-03) mirrors field names, nesting, and types of `src/json_serializer.cc` output so a Rust dump is diffable against a C++ dump. Not a free-form Rust-native debug format.
- **D-05 (real zero-copy now):** Build a true zero-copy PyBuffer in Phase 2 — frames hold **borrowed references directly into the `Model`'s SoA `TreeBuf<T>` columns** (no copy). Frame list's lifetime is tied to `&Model`.
- **D-06 (Rust-native frame enum representation):** Represent the PyBuffer frame as an **idiomatic Rust enum over borrowed slices** (not a raw `void*` POD struct in Phase 2). Converted to the C buffer-protocol POD layout only at the **Phase 8 (PyO3)** boundary. Frame list contents/order are pinned by D-01 — the enum is a representation choice over already-fixed framing.
- **D-07 (eager validation):** Validate per-node well-formedness at `EndNode` (leaf-vs-test mutual exclusivity, duplicate node keys, valid args), and per-tree topology — orphans, dangling child keys, reachability — at `EndTree` (deferred to tree close because child keys may be **forward references**). Errors surface at the offending call site with good locality where possible.
- **D-08 (always-strict, no opt-out):** Fluent node-by-node builder is always strict — do NOT port `SetValidationFlag("check_orphaned_nodes", ...)`. Validation cannot be turned off on the fluent path.
- **D-09 (BulkConstructTree is a separate pre-validated path):** `BulkConstructTree` (BLD-03) is NOT "the strict builder with checks disabled." It is a distinct fast constructor consuming pre-validated bulk input that bypasses node-by-node validation by construction. Document the bypass to match upstream.
- **D-10 (builder in new crate, serialize in core):** New `treelite-builder` crate. Put **serialization** (v5 binary, PyBuffer, `DumpAsJSON`) and **field accessors** (SER-04) as a **module inside `treelite-core`**, co-located with `Model` internals — avoids leaking private serialization fields across a crate boundary.
- **D-11 (loader rewiring dependency):** `treelite-xgboost` loader rewired to build through `treelite-builder` (success criterion 1), gaining a dependency on it. Must still load the Phase 1 fixture and verify within 1e-5.

### Claude's Discretion
- `ConcatenateModelObjects` (BLD-02) merge semantics — header-compat checks (matching `num_feature` / `task_type` / threshold+leaf types), tree concatenation, `target_id`/`class_id` handling — follow `src/model_concat.cc`.
- Field-accessor surface (SER-04) — which `Model`/`Tree` fields exposed and accessor API shape — follow `GetHeaderField` / tree-field accessors; keep idiomatic Rust.
- Typed error-enum granularity for builder and serializer (per-crate `thiserror` vs shared) — default per-crate, idiomatic, consistent with Phase 1.
- Exact mechanics of capturing the D-02 upstream v5 golden blob + manifest format/location (mirror Phase 1's manifest conventions).
- Internal representation of the builder's in-progress node/tree state and node-key resolution data structure.

### Deferred Ideas (OUT OF SCOPE)
- **Actual Python buffer-protocol / C POD conversion of the frame enum** — Phase 8 (PyO3). Phase 2 proves zero-copy borrowing into `TreeBuf`.
- **v3.9 / v4.0 read/write and cross-version migration** — PROJECT out-of-scope. Phase 2 is v5-only; deserialize rejects older formats (D-03).
- **`SetValidationFlag` configurable validation toggle** — intentionally not ported (D-08).
- **bytemuck `Pod` zero-copy recast of SoA columns** — Phase 9 (MEM-01); Phase 2 borrows via `TreeBuf` borrowed mode without bytemuck.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BLD-01 | Fluent validated `ModelBuilder` constructing node-by-node, rejecting orphaned/ill-formed topologies with typed error | Full port of `model_builder.cc` state machine + `EndTree` orphan/forward-ref resolution documented in §Architecture Patterns Pattern 1 & §Common Pitfalls; the loader rewiring proves it end-to-end |
| BLD-02 | `ConcatenateModelObjects` merges built models | `model_concat.cc` semantics (header-compat checks, `target_id`/`class_id` Extend) documented in §Architecture Patterns Pattern 4 |
| BLD-03 | `BulkConstructTree` fast path bypassing per-node validation | `sklearn_bulk.cc` `BulkConstructTree` documented in §Architecture Patterns Pattern 3 + §Open Questions Q1 (signature is sklearn-shaped — scope decision needed) |
| SER-01 | v5 binary serialize+deserialize round-trip (identical model) | Complete byte-level wire spec in §The v5 Wire Format (authoritative field order + framing) |
| SER-02 | v5 zero-copy PyBuffer representation | §Architecture Patterns Pattern 5 (frame enum over borrowed `TreeBuf` slices); frame order == binary field order per D-01 |
| SER-03 | `DumpAsJSON` emits model as JSON | `json_serializer.cc` structure documented in §Architecture Patterns Pattern 6 + §Code Examples |
| SER-04 | model/tree field accessors | `field_accessor.cc` `GetHeaderField`/`GetTreeField` documented in §Architecture Patterns Pattern 7 |
</phase_requirements>

## Summary

This is a **pure faithful-port phase**: the byte-level v5 wire format, the builder state machine, concat semantics, and the JSON dump structure are all fully specified in the vendored C++ at `treelite-mainline/` and were read line-by-line for this research. There is **no library to choose and no design space to explore** — the spec is fixed, and the only freedoms are the idiomatic-Rust representation of already-pinned structures (D-06 frame enum, error-enum granularity, builder internal state). The dominant risk is silent byte-level divergence from upstream, which is exactly what the D-02 golden blob defends against.

Three findings dominate planning:

1. **The "v5" format writes version bytes `4, 7, 0`, NOT `5, x, x`.** `TREELITE_VER_MAJOR/MINOR/PATCH` are `4.7.0` (vendored CMake `project(... VERSION 4.7.0)`). The serializer writes the *current Treelite version* into the header (`serializer.cc:93-95`), and the deserializer's only hard version gate is `major_ver == TREELITE_VER_MAJOR` i.e. `== 4` (`serializer.cc:192`). "v5" is the *wire-format generation name* (the `>=5.0` compatibility row in `tree.h`), but this Treelite 4.7.0 build emits a `major_ver=4` header that it accepts as the modern format. **D-03 ("reject non-v5") in practice means: accept `major_ver==4` only; reject `major_ver==3`; warn (not reject) on `major_ver==4, minor_ver>7`.** This must be reconciled with the CONTEXT wording before planning — see §Open Questions Q2. The golden blob captured from the installed wheel will settle it definitively.

2. **The binary stream framing is dead-simple and fixed-endian-by-host.** Every scalar is `sizeof(T)` raw little-endian bytes (x86-64 host). Every array is a `uint64` element-count prefix followed by `count * sizeof(T)` raw bytes. Every string is a `uint64` byte-length prefix followed by raw bytes. Empty arrays/strings write *just the `uint64` zero count* and no payload. There is **no magic number, no endianness marker, no alignment/padding** between fields. The PyBuffer path produces the *same logical sequence* as a list of `{ptr, format, itemsize, nitem}` frames; the binary path is the concatenation of those frames with the count/length prefixes inlined.

3. **`BulkConstructTree` (BLD-03) is sklearn-shaped, not generic.** The only upstream `BulkConstructTree` lives in `src/model_loader/sklearn_bulk.cc` and takes scikit-learn's exact CSR-tree array layout (`children_left/right`, `feature`, `threshold`, `value`, `n_node_samples`, `weighted_n_node_samples`, `impurity`, `is_classifier`) plus sklearn-specific gain computation and probability normalization. There is no generic "bulk build a tree from pre-validated columns" API in upstream. This is a scope tension with Phase 2 (sklearn is Phase 4) — see §Open Questions Q1.

**Primary recommendation:** Capture the D-02 golden v5 blob from the installed `treelite==4.7.0` wheel *first* (it is the ground truth for every byte-order/framing decision), then port the serializer header→trees field walk verbatim, then the builder state machine, then rewire the XGBoost loader through the builder while holding the existing 1e-5 harness green.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fluent `ModelBuilder` (BLD-01) | `treelite-builder` (new crate, D-10) | `treelite-core` (`Model`/`Tree` build target) | One responsibility per crate; builder is a construction front-end over core types |
| `ConcatenateModelObjects` (BLD-02) | `treelite-builder` or `treelite-core` | — | Free function over `&[&Model]`; clones trees, merges headers. Co-locate with builder (it is a construction operation) unless it needs `Model` privates (it does not — uses public fields + `Clone`) |
| `BulkConstructTree` (BLD-03) | `treelite-builder` | `treelite-core` (mutates `Tree` columns) | Bulk constructor; needs write access to `Tree` SoA columns (today `pub`, so no privacy issue) |
| v5 binary serialize/deserialize (SER-01) | `treelite-core` serialize module (D-10) | — | Touches `Model` private bookkeeping fields (`num_tree_`, `major_ver_`, …) — must be in-crate |
| Zero-copy PyBuffer frames (SER-02) | `treelite-core` serialize module | — | Frames borrow into `TreeBuf` columns of `&Model`; lifetime-tied to core types |
| `DumpAsJSON` (SER-03) | `treelite-core` serialize module | — | Reads `Model`/`Tree` public + private (`threshold_type` via `GetThresholdType`) |
| Field accessors (SER-04) | `treelite-core` serialize/accessor module | — | `GetHeaderField` touches `major_ver_`, `num_tree_`, etc. (private) |
| Loader rewiring (D-11) | `treelite-xgboost` | `treelite-builder` (new dep) | Loader emits builder calls instead of hand-assembling `Tree` |
| 1e-5 regression gate | `treelite-harness` | — | Unchanged; proves rewiring preserves predictions |

## Standard Stack

This is a port phase with **no new external dependencies**. The crate graph grows; the dependency set does not.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `thiserror` | 2.0.18 (workspace-pinned) | Typed error enums for `treelite-builder` and the core serialize module | Phase 1 convention; library crates use `thiserror`, never `anyhow` (CLAUDE.md) |
| `serde_json` | 1.0.150 (workspace-pinned) | `GetModelBuilder(json_str)` metadata parse (if ported) and `DumpAsJSON` emission | Already in workspace; Phase 1 loader uses it |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `anyhow` | 1.0.102 (workspace-pinned) | Error context in `treelite-harness` round-trip tests | Test/dev crates only (ERR-02) |
| `approx` | 0.5.1 (workspace-pinned) | 1e-5 float comparison in round-trip + rewiring tests | Already used by harness |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written little-endian byte writes (`to_le_bytes`) | `bytemuck` `Pod` cast of whole columns | bytemuck is explicitly **deferred to Phase 9** (D / Deferred Ideas). Phase 2 writes bytes via `&[T] → to_le_bytes` loops or `slice::align_to` — but note the SoA columns are already contiguous, so a plain `std::slice::from_raw_parts` reinterpret to `&[u8]` (or per-element `to_le_bytes`) suffices |
| `serde` derive for the binary format | Manual field walk mirroring `serializer.cc` | The wire format is **not** a serde-shaped format (no self-describing structure; fixed positional field walk with inlined length prefixes). A manual walk is the faithful port and is trivially auditable against `serializer.cc` |

**Installation:** No new crates. New crate manifest only:

```toml
# crates/treelite-builder/Cargo.toml
[package]
name = "treelite-builder"
edition.workspace = true
version.workspace = true
license.workspace = true

[dependencies]
treelite-core = { path = "../treelite-core" }
thiserror.workspace = true
serde_json.workspace = true   # only if GetModelBuilder(json_str) is ported
```

Add `"crates/treelite-builder"` to the root `[workspace] members`. Add `treelite-builder = { path = "../treelite-builder" }` to `crates/treelite-xgboost/Cargo.toml` (D-11).

**Version verification:** `thiserror 2.0.18`, `anyhow 1.0.102`, `serde_json 1.0.150`, `approx 0.5.1` are already resolved in the committed workspace and `uv.lock`/`Cargo` — confirmed by reading root `Cargo.toml`. No registry lookup needed (no new deps). `[VERIFIED: workspace Cargo.toml]`

## Package Legitimacy Audit

> No external packages are installed in this phase. The only manifest change adds a path dependency on the in-repo `treelite-core` crate and re-uses workspace-pinned `thiserror`/`serde_json` already present in Phase 1. slopcheck/registry verification is **N/A** — no third-party package is introduced.

| Package | Registry | Disposition |
|---------|----------|-------------|
| `treelite-core` (path dep) | in-repo | Approved (local crate) |
| `thiserror` 2.0.18 | crates.io (already pinned in Phase 1) | Approved (pre-existing) |
| `serde_json` 1.0.150 | crates.io (already pinned in Phase 1) | Approved (pre-existing) |

## The v5 Wire Format (SER-01 authoritative spec)

> This is the single most important section for D-01. It is the byte-for-byte contract the Rust serializer must reproduce. Source: `src/serializer.cc` (field order), `include/treelite/detail/serializer.h` (framing primitives), `include/treelite/detail/serializer_mixins.h` (binary writes). All `[VERIFIED: vendored serializer.cc/.h]`.

### Framing primitives (host little-endian; x86-64 manifest)

| Kind | Bytes written | Source |
|------|---------------|--------|
| **Scalar** `T` | exactly `sizeof(T)` raw LE bytes, no prefix | `WriteScalarToStream` / `BufferSerializerMixIn::SerializeScalar` (`serializer.h:163`, `mixins.h:171`) |
| **Array** `ContiguousArray<T>` | `uint64` element count (8 bytes LE) **then** `count * sizeof(T)` raw bytes. If count == 0: **only the 8-byte zero**, no payload | `WriteArrayToStream` (`serializer.h:181`), `BufferSerializerMixIn::SerializeArray` (`mixins.h:188`) |
| **String** `std::string` | `uint64` byte length (8 bytes LE) **then** `length` raw bytes (no NUL terminator). If length == 0: **only the 8-byte zero** | `WriteStringToStream` (`serializer.h:201`), `mixins.h:177` |

There is **NO** magic number, format version tag distinct from the version triple, endianness marker, field-name tags, or inter-field alignment/padding. The stream is a bare positional concatenation. `SerializeToBuffer` does a size-calc pass then a write pass — identical bytes to the stream path (`serializer.cc:482-501`). `TreeliteSerializeModelToBytes` (the C API the Python wheel calls) routes to `SerializeToBuffer`.

### Type-tag byte widths (critical — these are NOT all 4 bytes)

| Field type | Underlying repr | Bytes | Source |
|------------|-----------------|-------|--------|
| `TypeInfo` (threshold_type, leaf_output_type) | `std::uint8_t` | **1** | `typeinfo.h:21` — `kInvalid=0, kUInt32=1, kFloat32=2, kFloat64=3` |
| `TaskType` | `std::uint8_t` | **1** | `task_type.h:20` — `kBinaryClf=0, kRegressor=1, kMultiClf=2, kLearningToRank=3, kIsolationForest=4` |
| `TreeNodeType` | `std::int8_t` | **1** | `tree_node_type.h:18` — `kLeafNode=0, kNumericalTestNode=1, kCategoricalTestNode=2` |
| `Operator` | `std::int8_t` | **1** | `operator.h:17` — `kNone=0, kEQ=1, kLT=2, kLE=3, kGT=4, kGE=5` |
| `bool` (scalar `average_tree_output`, `has_categorical_split_`) | `bool` | **1** | direct `sizeof(bool)==1` |
| `ContiguousArray<bool>` (`default_left_`, `*_present_`, `category_list_right_child_`) | `bool*` real byte buffer (NOT `std::vector<bool>` bit-pack) | **1 byte per element** | `contiguous_array.h:59` — manual `T* buffer_`. Rust `Vec<bool>`/`Vec<u8>` matches; one byte per element |

The existing Rust enums (`enums.rs`) already carry the correct `#[repr(u8)]` / `#[repr(i8)]` matching these widths and integer values. **Reuse them directly — no remapping.** The PyBuffer format strings upstream infers (`InferFormatString`, `serializer.h:37`) are `=B/=b/=H/=h/=L/=l/=Q/=q/=f/=d` and `=c` for strings — needed only at the Phase 8 boundary, but the frame enum (D-06) should carry enough type info to reproduce them.

### Field order — Header (SerializeHeader, serializer.cc:91-126)

Emit in EXACTLY this order:

| # | Field | Wire type | Width / framing | Notes |
|---|-------|-----------|-----------------|-------|
| 1 | `major_ver` | scalar `int32` | 4 bytes | = `4` (TREELITE_VER_MAJOR). **Recomputed at serialize time**, not stored from load |
| 2 | `minor_ver` | scalar `int32` | 4 bytes | = `7` |
| 3 | `patch_ver` | scalar `int32` | 4 bytes | = `0` |
| 4 | `threshold_type` | scalar `TypeInfo`/`uint8` | 1 byte | `=2` (float32) for the XGBoost fixture |
| 5 | `leaf_output_type` | scalar `TypeInfo`/`uint8` | 1 byte | `=2` (float32) |
| 6 | `num_tree` | scalar `uint64` | 8 bytes | recomputed from `GetNumTree()` |
| 7 | `num_feature` | scalar `int32` | 4 bytes | |
| 8 | `task_type` | scalar `TaskType`/`uint8` | 1 byte | |
| 9 | `average_tree_output` | scalar `bool` | 1 byte | |
| 10 | `num_target` | scalar `int32` | 4 bytes | |
| 11 | `num_class` | array `int32` | u64 count + payload | |
| 12 | `leaf_vector_shape` | array `int32` | u64 count + payload | `[1,1]` for binary clf |
| 13 | `target_id` | array `int32` | u64 count + payload | length == num_tree |
| 14 | `class_id` | array `int32` | u64 count + payload | length == num_tree |
| 15 | `postprocessor` | string | u64 len + bytes | e.g. `"sigmoid"` |
| 16 | `sigmoid_alpha` | scalar `float32` | 4 bytes | |
| 17 | `ratio_c` | scalar `float32` | 4 bytes | |
| 18 | `base_scores` | array `double` | u64 count + payload | f64 elements |
| 19 | `attributes` | string | u64 len + bytes | e.g. `"{}"` |
| 20 | `num_opt_field_per_model` | scalar `int32` | 4 bytes | **always written as `0`** (extension slot 1) |

### Field order — Per tree (SerializeTree, serializer.cc:140-175)

For each of the `num_tree` trees, in tree order, emit EXACTLY:

| # | Field | Wire type | Element type | Notes |
|---|-------|-----------|--------------|-------|
| 1 | `num_nodes` | scalar `int32` | — | |
| 2 | `has_categorical_split_` | scalar `bool` | — | 1 byte |
| 3 | `node_type_` | array | `TreeNodeType`/int8 (1 B/elem) | |
| 4 | `cleft_` | array | `int32` | |
| 5 | `cright_` | array | `int32` | |
| 6 | `split_index_` | array | `int32` | |
| 7 | `default_left_` | array | `bool` (1 B/elem) | |
| 8 | `leaf_value_` | array | `LeafOutputType` (f32 or f64) | |
| 9 | `threshold_` | array | `ThresholdType` (f32 or f64) | |
| 10 | `cmp_` | array | `Operator`/int8 (1 B/elem) | |
| 11 | `category_list_right_child_` | array | `bool` (1 B/elem) | |
| 12 | `leaf_vector_` | array | `LeafOutputType` | empty for binary:logistic |
| 13 | `leaf_vector_begin_` | array | `uint64` | |
| 14 | `leaf_vector_end_` | array | `uint64` | |
| 15 | `category_list_` | array | `uint32` | |
| 16 | `category_list_begin_` | array | `uint64` | |
| 17 | `category_list_end_` | array | `uint64` | |
| 18 | `data_count_` | array | `uint64` | |
| 19 | `data_count_present_` | array | `bool` (1 B/elem) | |
| 20 | `sum_hess_` | array | `double` | |
| 21 | `sum_hess_present_` | array | `bool` (1 B/elem) | |
| 22 | `gain_` | array | `double` | |
| 23 | `gain_present_` | array | `bool` (1 B/elem) | |
| 24 | `num_opt_field_per_tree` | scalar `int32` | — | **always `0`** (extension slot 2) |
| 25 | `num_opt_field_per_node` | scalar `int32` | — | **always `0`** (extension slot 3) |

**Note on column order vs. the CONTEXT "Specific Ideas" list:** the CONTEXT prose lists the tree columns in declaration order from `tree.h`. The **serializer emits them in the order above** (`serializer.cc:142-174`), which differs slightly from struct declaration order (e.g. `leaf_value_` before `threshold_`, then `cmp_`, then `category_list_right_child_`, then the leaf-vector group, then category group, then the stats group). **Use `serializer.cc` order, not declaration order.** This is a prime divergence landmine.

### Deserialize gates (D-03)

`DeserializeHeaderAndCreateModel` (`serializer.cc:186-245`) reads the 3 version int32s first, then:
- `major_ver == 3 && minor_ver == 9` → routes to `DeserializeHeaderAndCreateModelV3` — **DO NOT PORT this branch (D-03)**; instead return a typed error.
- `major_ver != TREELITE_VER_MAJOR` (i.e. `!= 4`) and not the 3.9 case → upstream `TREELITE_CHECK` fails fatally. Rust: typed error.
- `major_ver == 4 && minor_ver > 7` → upstream logs a WARNING and continues (forward-compat). Decide whether to warn or accept silently (Q2).
- Reads remaining header in the same order; for `num_opt_field_per_model > 0` it calls `SkipOptionalField` that many times (each skip = 2 frames / a name-string + a {elem_size,nelem,payload} block in the stream — see `SkipOptionalFieldInStream`, `serializer.h:211`). Since we always write `0`, round-trip never exercises skip, but the deserializer must still implement the skip loop to consume a forward-version blob without corrupting the stream position.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
   XGBoost-JSON string   │           treelite-xgboost              │
   ────────────────────► │  load_xgboost_json (REWIRED, D-11)      │
                         │  parse JSON ─► emit builder calls       │
                         └──────────────┬──────────────────────────┘
                                        │ StartTree/StartNode/
                                        │ NumericalTest/LeafScalar/
                                        │ EndNode/EndTree/CommitModel
                                        ▼
                         ┌─────────────────────────────────────────┐
                         │            treelite-builder (NEW)       │
                         │  ModelBuilder state machine (D-07/D-08) │
                         │   ├ EndNode: per-node validity          │
                         │   ├ EndTree: forward-ref resolve +      │
                         │   │          orphan/reachability check  │
                         │   └ CommitModel ─► Model                │
                         │  ConcatenateModelObjects (BLD-02)       │
                         │  BulkConstructTree (BLD-03, bypass)     │
                         └──────────────┬──────────────────────────┘
                                        │ produces / mutates
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                        treelite-core                         │
        │  Model { variant: F32|F64, header fields,                    │
        │          + NEW private: num_tree_, num_opt_field_per_model_, │
        │            major/minor/patch_ver_, threshold/leaf_output_ty }│
        │  Tree<T> { SoA TreeBuf columns + num_opt_field_per_tree/node}│
        │                                                              │
        │   serialize module (D-10) ──────────────────────────────┐   │
        │    SerializerBackend trait (stream / buffer / pybuffer) │   │
        │     ├ serialize_header(&Model) ─► bytes / frames        │   │
        │     ├ serialize_trees(&Model)                           │   │
        │     ├ deserialize (v5-only gate, D-03) ─► Model         │   │
        │     ├ dump_as_json(&Model) (D-04 structure)             │   │
        │     └ get_header_field / get_tree_field (SER-04)        │   │
        └──────────┬─────────────────────────────┬────────────────────┘
                   │ binary bytes                 │ frame list (borrows
                   ▼   (SER-01)                   ▼  into TreeBuf, D-05)
        ┌──────────────────────┐      ┌────────────────────────────────┐
        │ round-trip:          │      │ PyBufferFrame enum (D-06)      │
        │ serialize→deserialize│      │ {slice + type tag}             │
        │ == identical Model   │      │ (→ C POD only at Phase 8)      │
        └──────────────────────┘      └────────────────────────────────┘
                   │
                   ▼ byte-compare against
        ┌──────────────────────────────────────────────┐
        │ D-02 golden v5 blob (frozen) + manifest       │
        │ captured from treelite==4.7.0 wheel (.venv)   │
        └──────────────────────────────────────────────┘
                   ▲
                   │ also: 1e-5 prediction gate
        ┌──────────────────────────────────────────────┐
        │ treelite-harness (unchanged) — proves rewire │
        └──────────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/
├── treelite-core/
│   └── src/
│       ├── model.rs        # + add 6 private bookkeeping fields
│       ├── tree.rs         # + num_opt_field_per_tree/node (default 0)
│       ├── tree_buf.rs     # zero-copy substrate (unchanged; frames borrow here)
│       └── serialize/      # NEW module (D-10)
│           ├── mod.rs      # SerializerBackend trait + serialize_header/trees
│           ├── binary.rs   # stream/buffer byte writes (SER-01)
│           ├── pybuffer.rs # frame enum over borrowed slices (SER-02, D-06)
│           ├── json.rs     # DumpAsJSON (SER-03, D-04)
│           └── fields.rs   # GetHeaderField/GetTreeField (SER-04)
├── treelite-builder/       # NEW crate (D-10)
│   └── src/
│       ├── lib.rs          # ModelBuilder state machine (BLD-01, D-07/D-08)
│       ├── concat.rs       # ConcatenateModelObjects (BLD-02)
│       ├── bulk.rs         # BulkConstructTree (BLD-03, D-09)
│       └── error.rs        # BuilderError (thiserror)
└── treelite-xgboost/       # rewired (D-11) — depends on treelite-builder
```

### Pattern 1: Builder state machine with forward-reference resolution (BLD-01, D-07)
**What:** A 5-state machine (`kExpectTree → kExpectNode → kExpectDetail → kNodeComplete`, plus `kModelComplete`) gating which methods are legal at each point. Child keys are *user-supplied integers*, stored raw during node creation, then translated to internal node indices at `EndTree` once all nodes exist.
**When to use:** All fluent construction.
**Upstream source:** `src/model_builder/model_builder.cc:50-388`.

Key behaviors to port verbatim:
- `StartNode(node_key)`: `node_key >= 0` required; `node_id_map_[node_key]` must not already exist (duplicate-key check, `model_builder.cc:162`); allocate an internal id.
- `NumericalTest/CategoricalTest`: child keys `>= 0`; current node key must differ from both child keys; `left_child_key != right_child_key`; if metadata initialized, `split_index < num_feature`. Children stored as **raw keys** via `SetChildren` (translated later).
- `LeafScalar/LeafVector`: mutually exclusive with the test methods by virtue of the state machine (after either, state = `kNodeComplete`). `LeafScalar` requires `expected_leaf_size_ == 1`; `LeafVector` requires `leaf_vector.size() == expected_leaf_size_` and the vector element type to match `LeafOutputT` (float↔double mismatch is fatal).
- `Gain/DataCount/SumHess`: legal in both `kExpectDetail` and `kNodeComplete` (optional stats can come after the test/leaf call).
- `EndTree` (`model_builder.cc:104-153`): tree must have `num_nodes > 0`; build an `orphaned` bool vector (all true except root index 0); for each non-leaf node, look up `node_id_map_.at(left_key)` / `.at(right_key)` — **a missing key is the "dangling child key" fatal error** ("Node with key K not found"); call `SetChildren(i, cleft, cright)` to write *internal* ids and mark both children non-orphaned. After the loop, if any node is still orphaned → fatal ("Node with key K is orphaned — it cannot be reached from the root node"). **D-08: this check is always on; do not port the `flag_check_orphaned_nodes_` toggle.**
- `CommitModel` (`model_builder.cc:279`): requires metadata initialized and `GetNumTree() == expected_num_tree_`; returns the `Model`.

**Rust mapping:** every upstream `TREELITE_LOG(FATAL)`/`TREELITE_CHECK` becomes a `Result<_, BuilderError>` arm with a `thiserror` variant carrying the offending key/index for locality (D-07 "errors at the offending call site"). Builder internal state (the `node_id_map_`) — discretion: a `BTreeMap<i32,i32>` mirrors upstream `std::map<int,int>` ordering (matters for the orphan-error message which iterates the map to find the offending key); a `HashMap` is fine if the error message exact text is not byte-compared.

### Pattern 2: Trait-based serializer backend (replaces the C++ MixIn template) (SER-01/02)
**What:** Upstream parameterizes `Serializer<MixIn>` over four mixins (Stream, Buffer, SizeCalculator, PyBuffer) that each implement `SerializeScalar/SerializeString/SerializeArray`. Port to a Rust trait with those three methods; the header/tree field walk is written once against the trait.
**When to use:** All serialize paths.
**Upstream source:** `include/treelite/detail/serializer_mixins.h` + `serializer.cc:86-179`.

```rust
// Source: ported from serializer_mixins.h (PyBufferSerializerMixIn et al.)
trait SerializerBackend {
    fn scalar<T: Pod1OrLE>(&mut self, value: &T);      // sizeof(T) LE bytes / 1-frame
    fn array<T: LeBytes>(&mut self, slice: &[T]);      // u64 count + payload / 1-frame
    fn string(&mut self, s: &str);                     // u64 len + bytes / 1-frame
}
// Implementors: BufferBackend(Vec<u8>), StreamBackend<W: Write>, FrameBackend<'a>(Vec<Frame<'a>>)
```

The size-calculator pass (upstream `SizeCalculatorMixIn`) is an optimization, not a correctness requirement — a Rust `Vec<u8>` that grows is fine for Phase 2; the byte output is identical. Keep it simple unless a benchmark demands pre-sizing.

### Pattern 3: BulkConstructTree bypass path (BLD-03, D-09)
**What:** A single-pass tree builder that takes pre-validated bulk arrays and `PushBack`s every column directly, skipping the state machine and validation entirely.
**Upstream source:** `src/model_loader/sklearn_bulk.cc:36-211`.
**Bypass semantics to document (D-09):** it does NOT run orphan/reachability/duplicate-key checks; it trusts the input arrays. Leaf detection is `children_left[i] == -1`; leaves get `split_index=-1, cmp=kNone, threshold=0`; internal nodes get `cmp=kLE, default_left=true`. Gain for internal nodes is computed (sklearn impurity-reduction formula); leaves get `gain=0, gain_present=false`. `data_count`/`sum_hess` always present. `has_categorical_split_ = false` (no categorical splits in the sklearn path).
**Scope caveat:** the only upstream signature is sklearn-array-shaped (see §Open Questions Q1) — planning must decide the Phase 2 surface.

### Pattern 4: ConcatenateModelObjects (BLD-02)
**What:** Merge `&[&Model]` into one model by copying header from `objs[0]` and deep-cloning every tree, extending `target_id`/`class_id`.
**Upstream source:** `src/model_concat.cc:19-71`.
**Semantics to port verbatim:**
- Empty input → return `None`/empty (upstream returns `{}`).
- Header (`num_feature`, `task_type`, `average_tree_output`, `num_target`, `num_class`, `leaf_vector_shape`, `postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores`, `attributes`) all copied from `objs[0]`.
- For each model: must be the **same variant** (`F32`/`F64`) as `objs[0]` (else error "different type than the first model object"); must match `num_target`, `num_class`, `leaf_vector_shape` (else error). Trees are `Clone()`d (deep copy — `TreeBuf::deep_copy`) and appended. `target_id`/`class_id` are `Extend`ed (concatenated).
- Post: assert `target_id.len() == class_id.len() == total_num_tree`.
**Note:** upstream does NOT check `postprocessor`/`sigmoid_alpha`/`base_scores` equality across inputs — it silently takes `objs[0]`'s. Port that behavior (do not add checks upstream lacks).

### Pattern 5: Zero-copy PyBuffer frame enum (SER-02, D-05/D-06)
**What:** `SerializeToPyBuffer` produces a `Vec<PyBufferFrame>` where each frame is `{buf ptr, format, itemsize, nitem}` aliasing live model memory (upstream POD struct, `c_api.h:53`). D-06 represents this as a Rust enum over borrowed slices instead of raw pointers.
**Upstream source:** `serializer.cc:458-463` + `serializer.h:30-110`.
**Frame order == binary field order** (the header table then per-tree table above) — pinned by D-01.

```rust
// Source: ported from PyBufferSerializerMixIn + GetPyBufferFrom{Scalar,Array,String}
// D-06: idiomatic enum over borrowed slices; lifetime tied to &Model (D-05).
enum Frame<'a> {
    U8(&'a [u8]),   I8(&'a [i8]),
    U32(&'a [u32]), I32(&'a [i32]),
    U64(&'a [u64]), I64(&'a [i64]),
    F32(&'a [f32]), F64(&'a [f64]),
    Str(&'a str),                    // "=c", itemsize 1
}
// A scalar is just a 1-element slice (upstream GetPyBufferFromScalar == nitem 1).
// Enums (TypeInfo/TaskType/...) are reinterpreted to their 1-byte underlying slice.
```

The frame borrows directly into `TreeBuf::as_slice()` for the `Owned` columns. **Lifetime constraint:** the returned `Vec<Frame<'a>>` borrows `&'a Model`, so the model must outlive the frames (D-05). The format-string mapping (`=B/=l/=Q/=f/=d/=c`) is reproduced from the variant at the Phase 8 boundary; it does not need to live in the enum if the variant already encodes the type — but carry `itemsize`/`format` derivation logic alongside.
**Caveat:** the version triple, `num_tree`, `threshold_type`/`leaf_output_type`, and the opt-field `0`s are *recomputed scalars* — they are not stored as slices in the model today. The frame for those must borrow a scalar that lives somewhere for `'a`. Upstream sidesteps this by writing the recomputed value into a `Model` member field first (`serializer.cc:93-106` assigns `model.major_ver_ = …` then takes `&model.major_ver_`). **Mirror that:** add the private bookkeeping fields, populate them at serialize time, then borrow them. This is why D-10 puts serialize in-core (needs `&mut` access to private fields to stage the recomputed scalars).

### Pattern 6: DumpAsJSON (SER-03, D-04)
**What:** Emit the model as a JSON object whose keys/nesting/types mirror `json_serializer.cc` exactly, so a Rust dump diffs cleanly against a C++ dump.
**Upstream source:** `src/json_serializer.cc:135-229`.
**Exact structure (D-04 — match key names and order):**
```
{ threshold_type, leaf_output_type, num_feature, task_type, average_tree_output,
  num_target, num_class, leaf_vector_shape, target_id, class_id,
  postprocessor, sigmoid_alpha, ratio_c, base_scores, attributes,
  trees: [ { num_nodes, has_categorical_split, nodes: [ <node>, ... ] }, ... ] }
```
Per node (`WriteNode`, json_serializer.cc:81-133):
- always: `node_id`.
- leaf: `leaf_value` (scalar OR array if `HasLeafVector`).
- internal: `split_feature_id`, `default_left`, `node_type` (string form via `TreeNodeTypeToString`), then for numerical: `comparison_op` (string) + `threshold`; for categorical: `category_list_right_child` + `category_list`; then `left_child`, `right_child`.
- conditional tail: `data_count` (if `HasDataCount`), `sum_hess` (if `HasSumHess`), `gain` (if `HasGain`).
- enum string forms come from `TaskTypeToString` (`kBinaryClf` etc.), `TypeInfoToString` (`float32` etc.), `TreeNodeTypeToString` (`leaf_node`/`numerical_test_node`/`categorical_test_node`), `OperatorToString` (`<`, `<=`, …) — **all already implemented in `enums.rs` `as_str()`** with the correct non-uniform spellings. Reuse them.
**Float formatting caveat:** upstream uses RapidJSON's `Double()` writer; Rust `serde_json` float formatting may differ in trailing-digit representation. For diffability, compare *parsed JSON values* (numeric equality) rather than raw bytes, OR pin a float formatter. Byte-identical JSON is NOT required by D-04 (only structural fidelity) — but flag this so the planner picks a comparison strategy (see §Validation Architecture).

### Pattern 7: Field accessors (SER-04)
**What:** `GetHeaderField(name)` / `GetTreeField(tree_id, name)` return a frame view of a named field; setters write into a field. Used by Python for inspection.
**Upstream source:** `src/field_accessor.cc:16-249`.
**Surface (discretion — keep idiomatic Rust):** the upstream string-dispatch over ~20 header names + ~25 tree-field names. A Rust port can expose typed accessor methods (`model.num_feature()`, `tree.threshold()`) instead of string dispatch, OR a `get_header_field(name) -> Frame` mirroring upstream for Phase 8 compatibility. Recommendation: provide the typed methods now (idiomatic, used by tests/concat/dump) and defer the string-dispatch `GetHeaderField(name)->Frame` shape to whenever Phase 8 needs the buffer-protocol seam — but at minimum cover the read accessors the success criteria name ("expose model/tree fields for inspection"). Note upstream read-only fields: `major_ver/minor_ver/patch_ver/threshold_type/leaf_output_type/num_tree/num_opt_field_*` reject `Set` (fatal); preserve that.

### Anti-Patterns to Avoid
- **Emitting columns in `tree.h` declaration order instead of `serializer.cc` order.** The serializer reorders (`leaf_value_` before `threshold_`, stats group last). Always follow `SerializeTree` order. A single transposed column corrupts every downstream byte.
- **Writing the version header as `5, x, x`.** It is `4, 7, 0`. The golden blob will catch this immediately.
- **Bit-packing bool arrays.** `ContiguousArray<bool>` is a real 1-byte-per-element buffer; a Rust `Vec<bool>` (1 byte/elem) matches. Do not use a bitset.
- **Adding cross-input equality checks to Concatenate that upstream lacks** (postprocessor/base_scores). Port the exact check set only.
- **Building a `Node` struct.** SoA is mandatory (Phase 1 anti-pattern, preserved); a node struct breaks zero-copy framing.
- **Porting the V3 deserialize path** (`DeserializeHeaderAndCreateModelV3`, `DeserializeTreeV3`). D-03 forbids it; reject `major_ver==3` with a typed error.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LE byte emission for scalars/arrays | A custom endian abstraction | `i32::to_le_bytes()` etc. in the trait impl, or `slice::from_raw_parts` reinterpret of the already-contiguous `TreeBuf` column | std has it; the format is host-LE on the x86-64 manifest |
| JSON emission for DumpAsJSON | A hand-written JSON string builder | `serde_json` (already in workspace) | escaping/number formatting is a foot-gun; but compare values, not bytes (D-04) |
| Wire-format spec | Reverse-engineering from the golden blob | The vendored `serializer.cc`/`serializer.h` (the literal source) + golden blob as the *check* | The spec is in-tree; the blob only validates the port |
| Builder validation rules | Inventing "reasonable" checks | The exact check set in `model_builder.cc` | Adding/removing a check diverges from upstream behavior tests |

**Key insight:** This phase has *no* deceptively-complex sub-problem to delegate to a library — the complexity is entirely "reproduce the vendored source exactly." The discipline is fidelity, not abstraction.

## Runtime State Inventory

> This phase is a feature-add port, not a rename/refactor. No runtime state migration applies. Listed for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore exists; the only "stored" artifact is the committed golden blob/fixtures (created fresh this phase) | none |
| Live service config | None — library crate, no services | none |
| OS-registered state | None | none |
| Secrets/env vars | None | none |
| Build artifacts | The new `treelite-builder` crate adds a workspace member; `treelite-xgboost` gains a dep edge. `cargo` rebuilds incrementally — no stale artifact carries an old name (greenfield crate) | none beyond normal `cargo build` |

**Nothing found requiring migration — verified by: no persistent datastore exists in the repo, and the phase only adds code + a frozen test fixture.**

## Common Pitfalls

### Pitfall 1: Version header mismatch (4.7.0 vs "v5")
**What goes wrong:** Writing `5,0,0` (or any non-`4,7,0`) into the version triple; the golden blob byte-compare fails on byte 0.
**Why it happens:** The format is *named* "v5" (the `>=5.0` compatibility-matrix generation), but Treelite 4.7.0 stamps its own version `4,7,0`.
**How to avoid:** Hardcode `MAJOR=4, MINOR=7, PATCH=0` constants matching the vendored `CMakeLists.txt VERSION 4.7.0`; verify against the captured golden blob's first 12 bytes.
**Warning signs:** Golden byte-compare diverges at offset 0; or upstream `deserialize_bytes` rejects the Rust output with a version error.

### Pitfall 2: Tree column emission order
**What goes wrong:** Columns serialized in struct-declaration order instead of `SerializeTree` order.
**Why it happens:** `tree.h` declares fields in a different order than `serializer.cc` emits them.
**How to avoid:** Follow the §Per tree table (from `serializer.cc:142-174`) literally; add a code comment citing the line.
**Warning signs:** Round-trip works (symmetric bug) but golden byte-compare diverges at the first reordered column; or deserialize succeeds but field values are transposed.

### Pitfall 3: Empty-array / empty-string framing
**What goes wrong:** Writing a payload (or skipping the count) for a zero-length array/string.
**Why it happens:** The "empty" branch writes *only* the 8-byte zero count and returns early (`serializer.h:185,205`).
**How to avoid:** For count==0, emit the 8-byte zero and stop — no payload. (binary:logistic exercises this heavily: `leaf_vector_`, all category columns, and possibly the stats columns are empty.)
**Warning signs:** Byte length off by the size of a missing/extra count word.

### Pitfall 4: NaN / inf in float columns (raw bit copy, not parsed)
**What goes wrong:** Treating float framing as text or normalizing NaN.
**Why it happens:** Phase 1 already established that missing-value rows produce `f32::NAN` thresholds in *prediction inputs*; in *serialization* floats are raw `to_le_bytes` of the IEEE-754 bits — NaN/inf serialize as their exact bit patterns with no special handling (`memcpy` upstream).
**How to avoid:** Use `f32::to_le_bytes`/`f64::to_le_bytes` (bit-exact); never compare floats for "equality" in round-trip via `==` on NaN — compare *bytes* for the binary round-trip, and use the 1e-5 harness only for the prediction gate. For a NaN-containing column, byte round-trip is exact (bits preserved); the model "identical after round-trip" check must be byte/bit-level, not float-`==` (NaN != NaN).
**Warning signs:** A round-trip "identical model" assertion fails only on rows with NaN despite bytes matching.

### Pitfall 5: 1-byte enum/bool widths assumed 4 bytes
**What goes wrong:** Serializing `TypeInfo`/`TaskType`/`Operator`/`TreeNodeType`/`bool` as 4-byte ints.
**Why it happens:** Habit; many enums default to `int`.
**How to avoid:** Use the `#[repr(u8)]`/`#[repr(i8)]` already on `enums.rs`; write `as u8`/`as i8` single bytes. `bool` arrays are 1 byte/element.
**Warning signs:** Header length is 3 bytes too long per enum scalar; golden diverges right after `patch_ver`.

### Pitfall 6: Forward-reference child keys resolved too early
**What goes wrong:** Trying to validate/resolve child node existence at `EndNode` instead of `EndTree`.
**Why it happens:** Intuitively a node "references" children when created.
**How to avoid:** Store child keys raw at node-creation; resolve via `node_id_map_.at()` only at `EndTree` (D-07). A child key may legally reference a node declared *later* in the same tree.
**Warning signs:** A valid tree with a parent-before-child declaration order is wrongly rejected as "child not found."

## Code Examples

### Binary scalar / array / string framing (the three primitives)
```rust
// Source: serializer.h WriteScalarToStream / WriteArrayToStream / WriteStringToStream
fn scalar_le(out: &mut Vec<u8>, bytes: &[u8]) { out.extend_from_slice(bytes); } // no prefix

fn array_le<T>(out: &mut Vec<u8>, slice: &[T], elem_to_le: impl Fn(&T)->[u8; N]) {
    out.extend_from_slice(&(slice.len() as u64).to_le_bytes()); // u64 count
    if slice.is_empty() { return; }                              // empty: count only
    for e in slice { out.extend_from_slice(&elem_to_le(e)); }
}

fn string_le(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());      // u64 byte-length
    if s.is_empty() { return; }                                  // empty: length only
    out.extend_from_slice(s.as_bytes());                         // no NUL
}
```

### Header field walk (mirrors SerializeHeader exactly)
```rust
// Source: serializer.cc:91-126 — order is load-bearing (D-01)
fn serialize_header(m: &mut Model, b: &mut impl SerializerBackend) {
    m.major_ver_ = 4; m.minor_ver_ = 7; m.patch_ver_ = 0;   // recomputed, then borrowed
    b.scalar(&m.major_ver_); b.scalar(&m.minor_ver_); b.scalar(&m.patch_ver_);
    m.threshold_type_ = m.threshold_type();  m.leaf_output_type_ = m.leaf_output_type();
    b.scalar_u8(m.threshold_type_ as u8); b.scalar_u8(m.leaf_output_type_ as u8);
    m.num_tree_ = m.num_tree() as u64;       b.scalar(&m.num_tree_);
    b.scalar(&m.num_feature); b.scalar_u8(m.task_type as u8);
    b.scalar_bool(m.average_tree_output); b.scalar(&m.num_target);
    b.array(&m.num_class); b.array(&m.leaf_vector_shape);
    b.array(&m.target_id); b.array(&m.class_id);
    b.string(&m.postprocessor); b.scalar(&m.sigmoid_alpha); b.scalar(&m.ratio_c);
    b.array(&m.base_scores); b.string(&m.attributes);
    m.num_opt_field_per_model_ = 0; b.scalar(&m.num_opt_field_per_model_); // always 0
}
```

### D-02 golden blob capture (Python, run once from the installed wheel)
```python
# Source: treelite/model.py serialize_bytes (calls TreeliteSerializeModelToBytes -> SerializeToBuffer)
# Run from the repo .venv (treelite==4.7.0 already installed). Mirrors Phase 1 golden capture.
import json, platform, hashlib, treelite, xgboost
m = treelite.frontend.from_xgboost(
        xgboost.Booster(model_file="fixtures/binary_logistic.model.json"))  # same fixture as Phase 1
blob = m.serialize_bytes()                       # the authoritative v5 byte stream
open("fixtures/golden_v5.bin", "wb").write(blob)
manifest = {                                     # mirror fixtures/golden.json manifest keys
    "treelite": treelite.__version__, "xgboost": xgboost.__version__,
    "os": platform.platform(), "arch": platform.machine(),
    "libc": list(platform.libc_ver()), "python": platform.python_version(),
    "sha256": hashlib.sha256(blob).hexdigest(), "nbytes": len(blob),
    "source_fixture": "fixtures/binary_logistic.model.json",
}
json.dump(manifest, open("fixtures/golden_v5.manifest.json", "w"), indent=2)
```
The Rust test then asserts `serialize(model) == read("fixtures/golden_v5.bin")` byte-for-byte, and `deserialize(blob)` reproduces the model. (Confirm the exact `from_xgboost`/`Booster` load API against `treelite/frontend.py` when capturing — the fixture is XGBoost-JSON; load it the same way Phase 1's golden was produced.)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v3.9 serialization (`Node` struct array, `ModelParamV3`, `TaskParamV3`, `pred_transform[256]`) | v5 SoA-column framing with extension slots | Treelite 4.0 → present | Phase 2 ports ONLY the modern path; v3 is read-reject (D-03) |
| `SetValidationFlag` toggle | Always-strict fluent + separate bulk bypass | upstream still has the toggle; this port drops it | D-08/D-09 — cleaner two-path design |
| C-API buffer-protocol POD struct | Rust frame enum over borrowed slices | this port (D-06) | Safe zero-copy now; POD conversion deferred to Phase 8 |

**Deprecated/outdated:**
- `DeserializeHeaderAndCreateModelV3` / `DeserializeTreeV3` (`serializer.cc:317-453`): do NOT port (D-03).
- `TaskTypeV3::kMultiClfCategLeaf`: removed in 4.0; irrelevant.
- `max_index` postprocessor → forced to `softmax` in V3 load path only; irrelevant to v5 round-trip.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | "v5" wire generation, as produced by the installed treelite 4.7.0 wheel, stamps version bytes `4,7,0` and the modern (non-V3) field walk; D-03 in practice = "accept major_ver==4, reject major_ver==3". | Summary finding 1, §Wire Format, §Open Q2 | If the wheel actually emits a `5,x,x` header (e.g. a repackaged build), the version constants and the D-03 gate are wrong. **Mitigated:** the golden blob's first 12 bytes settle this empirically during capture — capture before finalizing the gate. |
| A2 | Host is little-endian x86-64 (per Phase 1 manifest `x86_64`), so raw `memcpy`/`to_le_bytes` reproduces upstream bytes; the format has no endianness marker and is therefore host-endian-dependent. | §Wire Format primitives | On a big-endian host the bytes would differ — but the manifest pins x86-64 and CI runs there. Document the manifest dependency (already a D-02 requirement). |
| A3 | The Phase 2 `BulkConstructTree` (BLD-03) should expose the sklearn-shaped signature found in `sklearn_bulk.cc`, since that is the only upstream `BulkConstructTree`. | §Pattern 3, §Open Q1 | If planners intended a generic bulk-from-columns API, the signature differs. **Needs user/planner decision (Q1).** |
| A4 | `DumpAsJSON` (D-04) requires *structural* fidelity (key names/nesting/types), not byte-identical float formatting; value-level JSON comparison is the intended check. | §Pattern 6 | If byte-identical dump is required, a custom float formatter matching RapidJSON is needed. CONTEXT D-04 says "diffable"/"field names, nesting, types" — structural reading is supported but not explicit on float bytes. |
| A5 | `ContiguousArray<bool>` is a real 1-byte-per-element buffer (manual `T* buffer_`), so a Rust `Vec<bool>` matches byte-for-byte (no bit-packing). | §Wire Format type tags, §Pitfall 5 | If any column used `std::vector<bool>` it would bit-pack — but upstream uses `ContiguousArray`, not `vector<bool>` (verified `contiguous_array.h:59`). Low risk. |

## Open Questions

1. **BulkConstructTree scope/signature (BLD-03).**
   - What we know: The only upstream `BulkConstructTree` is in `src/model_loader/sklearn_bulk.cc`, with a scikit-learn-specific array signature (`children_left/right`, `feature`, `threshold`, `value`, `n_node_samples`, `weighted_n_node_samples`, `impurity`, `total_sample_cnt`, `n_targets`, `max_num_class`, `is_classifier`) and sklearn gain/probability logic. sklearn loading itself is Phase 4.
   - What's unclear: Does Phase 2 BLD-03 mean (a) port that exact sklearn-shaped function now (so Phase 4 can call it), or (b) define a more generic "bulk build from validated columns" entry that sklearn later adapts to? Upstream has no generic version.
   - Recommendation: Port the sklearn-shaped `BulkConstructTree` verbatim into `treelite-builder` now (it is self-contained and the only spec), and document the bypass (D-09). Defer wiring it to an actual sklearn loader to Phase 4. Flag for the planner to confirm against the BLD-03 requirement text in REQUIREMENTS.md.

2. **D-03 "reject non-v5" exact gate vs. the 4.7.0 version header.**
   - What we know: The deserializer's hard gate is `major_ver == TREELITE_VER_MAJOR (==4)`; `major_ver==3 && minor==9` routes to V3 (which we reject); `major==4 && minor>7` warns and continues.
   - What's unclear: CONTEXT phrases D-03 as "accepts only v5; non-v5 (v3.9/v4.0 headers) rejected." But the *wire generation* "v5" corresponds to `major_ver==4` here (the `>=5.0` compatibility row applies to format capability, not the stamped version). Does "reject v4.0 headers" mean reject `major==4,minor==0`? Upstream does NOT reject 4.0 — it reads 4.0+ with the modern walk. The CONTEXT's "v3.9/v4.0" likely means the *old wire generations*, of which only the 3.9 path has distinct deserialize code.
   - Recommendation: Implement the gate as: reject `major_ver != 4` (covers 3.x and any future-major); within `major==4`, accept all minors (warn if `>7`). This matches upstream's actual behavior and the golden blob (which is `4,7,0`). Confirm wording with the planner; the golden capture makes the real header bytes authoritative.

3. **`DumpAsJSON` float-formatting comparison strategy (D-04).**
   - What we know: Upstream uses RapidJSON `Double()`; Rust `serde_json` may format floats differently.
   - What's unclear: Whether the diffability requirement is structural (value-equal) or byte-equal.
   - Recommendation: Compare parsed JSON values (numeric tolerance where needed), not raw text. If byte-identical is later required, pin a Grisu/Ryū formatter to match RapidJSON.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable, edition 2024 | all crates | ✓ (Phase 1 builds) | stable | — |
| `cargo` | build/test | ✓ | — | — |
| `treelite` Python wheel | D-02 golden v5 blob capture (one-time) | ✓ | 4.7.0 (`.venv/lib/python3.13/site-packages/treelite-4.7.0`) | — (this IS the ground-truth tool) |
| `xgboost` Python wheel | D-02 capture (load fixture to build the Model to serialize) | ✓ | 3.2.0 (`.venv/.../xgboost-3.2.0`) | Could instead build the treelite Model via treelite's own model_builder in Python if the from_xgboost path is awkward |
| C++ compiler / CMake | NOT needed | n/a | — | CI never compiles C++ (D-02) — the wheel is pre-built |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None blocking — both wheels are present at the exact versions the Phase 1 manifest froze.

## Validation Architecture

> nyquist_validation is enabled (config: `workflow.nyquist_validation: true`). This section drives a derivable VALIDATION.md.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `approx` for float tolerance; Phase 1 pattern |
| Config file | none (Cargo convention: `tests/` integration tests per crate) |
| Quick run command | `cargo test -p treelite-builder` / `cargo test -p treelite-core --lib` |
| Full suite command | `cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SER-01 | Binary round-trip: serialize→deserialize == identical model (byte/bit-level for NaN safety) | unit | `cargo test -p treelite-core serialize::roundtrip` | ❌ Wave 0 |
| SER-01 / D-01 / D-02 | Rust serialize(fixture model) == frozen `golden_v5.bin` byte-for-byte; deserialize(golden)==model | integration | `cargo test -p treelite-harness golden_v5` | ❌ Wave 0 (needs blob captured) |
| SER-02 | PyBuffer frame list order/contents match the binary field walk; frames borrow (no copy) | unit | `cargo test -p treelite-core serialize::pybuffer` | ❌ Wave 0 |
| SER-03 / D-04 | DumpAsJSON structure (keys/nesting/types) matches upstream dump (value-level) | integration | `cargo test -p treelite-harness dump_json` | ❌ Wave 0 (optional: capture a golden JSON dump from wheel) |
| SER-04 | Field accessors return correct model/tree field values | unit | `cargo test -p treelite-core fields` | ❌ Wave 0 |
| BLD-01 / D-07 / D-08 | Builder rejects: orphan node, dangling child key, duplicate key, leaf+test, empty tree, child<0; accepts forward-ref child order; always-strict | unit | `cargo test -p treelite-builder validation` | ❌ Wave 0 |
| BLD-02 | Concatenate merges trees, extends target_id/class_id, rejects type/num_target/num_class/leaf_vector_shape mismatch, empty→none | unit | `cargo test -p treelite-builder concat` | ❌ Wave 0 |
| BLD-03 / D-09 | BulkConstructTree builds a tree from bulk arrays bypassing validation; output matches per-node build | unit | `cargo test -p treelite-builder bulk` | ❌ Wave 0 |
| D-11 | XGBoost loader rewired through builder still verifies within 1e-5 | integration | `cargo test -p treelite-harness equivalence` | ✅ exists (must stay green) |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate-under-edit>` (quick).
- **Per wave merge:** `cargo test --workspace`.
- **Phase gate:** `cargo test --workspace` green + golden_v5 byte-compare green + existing 1e-5 equivalence green, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `fixtures/golden_v5.bin` + `fixtures/golden_v5.manifest.json` — capture from the installed treelite 4.7.0 wheel (D-02). **Blocks the byte-fidelity test; do this first.**
- [ ] `crates/treelite-core/tests/serialize_roundtrip.rs` — covers SER-01.
- [ ] `crates/treelite-core/tests/serialize_pybuffer.rs` — covers SER-02.
- [ ] `crates/treelite-core/tests/fields.rs` — covers SER-04.
- [ ] `crates/treelite-harness/tests/golden_v5.rs` — covers SER-01/D-01/D-02 byte-compare.
- [ ] `crates/treelite-builder/tests/{validation,concat,bulk}.rs` — covers BLD-01/02/03.
- [ ] (optional) golden JSON dump fixture for SER-03/D-04 structural diff.
- [ ] No framework install needed (Rust built-in harness; `approx` already pinned).

## Security Domain

> security_enforcement is enabled (config: `true`), ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Library crate; no auth surface |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | **yes** | Deserializing untrusted bytes (SER-01) and untrusted builder input (BLD-01) MUST validate: array count prefixes (a malicious `u64` count could request a huge allocation or over-read), string length prefixes, `num_tree`/`num_nodes` bounds, child-key bounds, and reject truncated/over-long streams with a typed error — never panic, never over-read. Phase 1 already establishes the typed-error-not-panic discipline (WR-02). |
| V6 Cryptography | no | No crypto; never hand-roll any (none needed) |

### Known Threat Patterns for the Rust serializer/builder

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious deserialize: oversized `u64` array/string count triggers huge `Vec::with_capacity` (DoS / OOM) | Denial of Service | Bound the count against remaining buffer length before allocating; reject if `count * sizeof(T) > remaining_bytes` with a typed error (do NOT trust the prefix). |
| Malicious deserialize: truncated stream → out-of-bounds read | Tampering / Info disclosure | Read into checked slices; every read verifies `offset + n <= buf.len()`; return `Err` on short read (safe Rust slicing panics rather than over-reads — convert to `Result`). |
| Deserialize accepts a non-v5/V3 header and mis-parses | Tampering | D-03 reject-non-v5 gate is itself the mitigation; reject `major_ver != 4` early. |
| Builder: negative/huge `num_feature`, child keys, or `num_tree` (cast to `usize`) | Tampering / DoS | Phase 1 `require_non_negative` pattern; bound-check split_index < num_feature (upstream does), keys >= 0. |
| Deserialize allocation amplification via opt-field skip loop | DoS | `num_opt_field_*` loop count is itself from the stream; bound it (we always write 0; reject absurd counts on read). |

## Sources

### Primary (HIGH confidence)
- `treelite-mainline/src/serializer.cc` — authoritative field order (SerializeHeader/SerializeTrees/SerializeTree), V3-reject boundary, all four serialize entry points.
- `treelite-mainline/include/treelite/detail/serializer.h` — byte-level framing primitives (scalar/array/string read+write, empty-handling, format-string inference, opt-field skip).
- `treelite-mainline/include/treelite/detail/serializer_mixins.h` — the four MixIn backends (stream/buffer/size/pybuffer) → Rust trait.
- `treelite-mainline/include/treelite/pybuffer_frame.h` + `include/treelite/c_api.h:53` — `TreelitePyBufferFrame{buf,format,itemsize,nitem}` POD.
- `treelite-mainline/src/model_builder/model_builder.cc` — builder state machine, validation timing, forward-ref resolution, orphan check.
- `treelite-mainline/include/treelite/model_builder.h` — ModelBuilder interface, Metadata/TreeAnnotation/PostProcessorFunc.
- `treelite-mainline/src/model_concat.cc` — ConcatenateModelObjects semantics.
- `treelite-mainline/src/model_loader/sklearn_bulk.cc` — the only BulkConstructTree (sklearn-shaped).
- `treelite-mainline/src/json_serializer.cc` — DumpAsJSON structure.
- `treelite-mainline/src/field_accessor.cc` — GetHeaderField/GetTreeField/SetHeaderField surface.
- `treelite-mainline/include/treelite/tree.h:380-580` — Model/Tree field types, private bookkeeping fields, compatibility matrix, serialize signatures.
- `treelite-mainline/include/treelite/enum/{typeinfo,task_type,tree_node_type,operator}.h` — exact enum underlying types + values.
- `treelite-mainline/cmake/version.h.in` + `CMakeLists.txt` (`VERSION 4.7.0`) — version header values.
- Existing Rust: `crates/treelite-core/src/{model,tree,tree_buf,enums}.rs`, `crates/treelite-xgboost/src/lib.rs`, `crates/treelite-harness/src/lib.rs`, root `Cargo.toml` — the build target and reusable assets.
- `.venv/lib/python3.13/site-packages/treelite/model.py:250` (`serialize_bytes` → `TreeliteSerializeModelToBytes`), `VERSION` (`4.7.0`) — the D-02 capture tool, installed.
- `fixtures/golden.json` — Phase 1 manifest pattern to mirror for D-02.

### Secondary (MEDIUM confidence)
- None — every claim is grounded in vendored source or installed artifacts read in-session.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Wire format (SER-01/D-01): HIGH — read the literal serializer source byte-by-byte; the only residual is the 4.7.0-vs-"v5" version-header naming (A1/Q2), which the golden blob capture settles empirically.
- Builder (BLD-01/D-07/D-08): HIGH — full state machine + validation source read.
- Concat (BLD-02): HIGH — short, complete source read.
- BulkConstructTree (BLD-03): MEDIUM — source is clear, but its sklearn-specific shape creates a scope question for Phase 2 (Q1).
- PyBuffer (SER-02/D-05/D-06): HIGH — framing source read; the recomputed-scalar lifetime subtlety is documented (Pattern 5).
- DumpAsJSON (SER-03/D-04): HIGH structure, MEDIUM on float-byte equality (A4/Q3).
- D-02 capture: HIGH — exact wheel/version present and the serialize API confirmed.

**Research date:** 2026-06-10
**Valid until:** Stable — the spec is a frozen vendored read-only tree (treelite 4.7.0) and pinned wheels; only re-research if the vendored `treelite-mainline/` version or the `.venv` treelite wheel changes.

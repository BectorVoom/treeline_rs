# Phase 2: Builder & Serialization - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 17 new/modified files
**Analogs found:** 16 / 17 (1 capture-script analog; no in-repo serializer/builder analog yet — those are first-of-kind, but every Phase 1 crate supplies structural analogs)

All analogs are Phase 1 Rust output under `crates/`. Upstream `treelite-mainline/` is the **spec** (read-only porting source), not a code analog — the wire-format/builder semantics live in RESEARCH.md, while *how to shape the Rust* comes from the Phase 1 analogs below.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-builder/Cargo.toml` | config | — | `crates/treelite-xgboost/Cargo.toml` | exact (path-dep crate manifest) |
| `crates/treelite-builder/src/lib.rs` | builder (state machine) | event-driven (Begin/End calls) | `crates/treelite-xgboost/src/lib.rs` | role-match (constructs `Tree`/`Model` from a driver) |
| `crates/treelite-builder/src/error.rs` | error | — | `crates/treelite-xgboost/src/error.rs` | exact (thiserror enum w/ locality fields) |
| `crates/treelite-builder/src/concat.rs` | utility (free fn) | transform (`&[&Model]`→`Model`) | `crates/treelite-xgboost/src/lib.rs` (`load_xgboost_json` builds a `Model`) | role-match |
| `crates/treelite-builder/src/bulk.rs` | builder (bulk ctor) | batch (columns→`Tree`) | `crates/treelite-xgboost/src/lib.rs` (`build_tree`) | exact (column-fill → `TreeBuf::from_owned`) |
| `crates/treelite-core/src/serialize/mod.rs` | serializer (trait) | request-response (header/tree walk) | none in-repo (first serializer) | no analog — see "No Analog Found" |
| `crates/treelite-core/src/serialize/binary.rs` | serializer backend | streaming (byte emit) | `crates/treelite-harness/src/lib.rs` (byte-level `normalize_nan_tokens`) | partial (byte buffer building) |
| `crates/treelite-core/src/serialize/pybuffer.rs` | serializer backend | streaming (borrowed frames) | `crates/treelite-core/src/tree_buf.rs` (`as_slice`, `Borrowed`) | role-match (zero-copy borrow substrate) |
| `crates/treelite-core/src/serialize/json.rs` | serializer backend | transform (`Model`→JSON) | `crates/treelite-xgboost/src/lib.rs` (serde_json usage) | partial |
| `crates/treelite-core/src/serialize/fields.rs` | accessor | request-response | `crates/treelite-core/src/tree.rs` (getter methods) | exact (getter convention) |
| `crates/treelite-core/src/serialize/error.rs` (or reuse `error.rs`) | error | — | `crates/treelite-gtil/src/error.rs` | exact (multi-variant thiserror w/ bounds fields) |
| `crates/treelite-core/src/model.rs` (modify) | model | — | itself (add 6 private fields) | exact (self-extension) |
| `crates/treelite-core/src/tree.rs` (modify) | model | — | itself (add `num_opt_field_per_tree/node`) | exact (self-extension) |
| `crates/treelite-core/src/lib.rs` (modify) | config (module wiring) | — | itself (`pub mod` + `pub use`) | exact |
| `crates/treelite-xgboost/src/lib.rs` (rewire) | loader | event-driven (emits builder calls) | itself (current direct-assembly form) | exact (self-rewire) |
| `crates/treelite-xgboost/Cargo.toml` (modify) | config | — | itself (add builder dep) | exact |
| `fixtures/capture_golden_v5.py` + `fixtures/golden_v5.bin` + manifest | test fixture | file-I/O | `fixtures/capture_golden.py` + `fixtures/golden.json` | exact (D-02 mirrors D-06/D-07) |
| `crates/treelite-builder/tests/{validation,concat,bulk}.rs` | test | — | `crates/treelite-core/tests/tree_model.rs` | exact |
| `crates/treelite-core/tests/serialize_*.rs` + `crates/treelite-harness/tests/golden_v5.rs` | test | — | `crates/treelite-harness/tests/equivalence.rs` | exact |

---

## Pattern Assignments

### `crates/treelite-builder/Cargo.toml` (config)

**Analog:** `crates/treelite-xgboost/Cargo.toml` (whole file — a path-dep-on-core crate manifest using workspace-pinned deps).

Copy the exact `workspace = true` inheritance shape. Per RESEARCH §Standard Stack the new manifest is:

```toml
[package]
name = "treelite-builder"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
treelite-core = { path = "../treelite-core" }
thiserror = { workspace = true }
serde_json = { workspace = true }   # only if GetModelBuilder(json_str) is ported
```

Then add `"crates/treelite-builder"` to root `Cargo.toml` `[workspace] members` (currently lists the 4 Phase 1 crates).

---

### `crates/treelite-builder/src/error.rs` (error, thiserror)

**Analog:** `crates/treelite-xgboost/src/error.rs` (whole file) + `crates/treelite-gtil/src/error.rs` for the bounds-carrying variants.

**Header doc-comment convention** (`error.rs:1-12`) — cite the upstream fatal paths the variants replace:

```rust
//! Typed errors for the model builder (ERR-01).
//!
//! Every upstream fatal path (`TREELITE_LOG(FATAL)` / `TREELITE_CHECK`) in
//! `treelite-mainline/src/model_builder/model_builder.cc` becomes a returned
//! `Err` here rather than a panic.

use thiserror::Error;
```

**Enum-with-locality-fields pattern** (xgboost `error.rs:47-62` `DimensionMismatch`) — each variant carries the offending key/index for D-07 "errors at the offending call site":

```rust
#[derive(Debug, Error)]
pub enum BuilderError {
    #[error("node with key {key} not found (dangling child reference)")]
    DanglingChildKey { key: i32 },

    #[error("node with key {key} is orphaned — unreachable from the root")]
    OrphanedNode { key: i32 },

    #[error("duplicate node key {key}")]
    DuplicateNodeKey { key: i32 },

    // ... leaf-vs-test mutual exclusion, child_key < 0, split_index >= num_feature, etc.

    /// Bubbled up from `treelite-core`.
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),
}
```

**`#[error(transparent)]` + `#[from]` for core bubble-up** is the established cross-crate convention (xgboost `error.rs:70-72`, gtil `error.rs:61-63`). Reuse it verbatim.

---

### `crates/treelite-builder/src/lib.rs` (builder state machine, BLD-01)

**Analog:** `crates/treelite-xgboost/src/lib.rs` — specifically `build_tree` (`lib.rs:151-207`) shows the canonical "fill `Vec`s per column, then `TreeBuf::from_owned` each into a `Tree::new()`" finalize. The builder's `EndTree` produces a `Tree<T>` the same way.

**Column-fill → finalize pattern to copy** (`lib.rs:195-206`):

```rust
let mut tree = Tree::<f32>::new();
tree.node_type = TreeBuf::from_owned(node_type);
tree.cleft = TreeBuf::from_owned(cleft);
// ... every column ...
tree.has_categorical_split = false;
tree.num_nodes = num_nodes as i32;
```

**`Result<_, BuilderError>` return on every fallible step** (xgboost `load_xgboost_json` `lib.rs:217`, `?`-propagation throughout). Every upstream `TREELITE_CHECK` (RESEARCH Pattern 1, `model_builder.cc:50-388`) becomes a `BuilderError` arm.

**Module structure / doc-comment header** mirrors xgboost `lib.rs:1-23` (`//! ... Ports treelite-mainline/...` then `pub mod`/`pub use`, then `use treelite_core::{...}`).

**Builder internal `node_id_map_`** — discretion (RESEARCH Pattern 1 Rust-mapping note): `BTreeMap<i32,i32>` if the orphan-error message text is byte-compared (mirrors upstream `std::map` iteration order); `HashMap` otherwise. No in-repo analog (greenfield state machine) — semantics come from RESEARCH Pattern 1, shape from xgboost loader.

---

### `crates/treelite-builder/src/concat.rs` (BLD-02, free fn over `&[&Model]`)

**Analog:** `crates/treelite-xgboost/src/lib.rs:240-285` (the `Model::new(variant)` + header-field assignment block) shows how a `Model` is assembled field-by-field. Concat copies header from `objs[0]` the same way.

**`ModelVariant` match for the "same-variant" check** — copy the dispatch shape from `model.rs:44-49` (`ModelVariant::F32`/`F64`). RESEARCH Pattern 4: each input must be the same variant as `objs[0]` (else error), trees `deep_copy`'d (use `Tree::deep_copy` / `TreeBuf::deep_copy`, `tree_buf.rs:80-83`), `target_id`/`class_id` `Extend`ed.

**Deep-copy primitive** (`tree_buf.rs:81-83`): `TreeBuf::Owned(self.as_slice().to_vec())` — concat appends `Clone`d trees via this.

---

### `crates/treelite-builder/src/bulk.rs` (BLD-03, bulk ctor)

**Analog:** `crates/treelite-xgboost/src/lib.rs` `build_tree` (`lib.rs:151-207`) — an **exact** structural match: BulkConstructTree is the same "pre-validated columns in, single fill loop, `TreeBuf::from_owned` per column" shape, minus the `check_dim` validation (D-09 bypass).

The leaf/internal branch in `build_tree` (`lib.rs:172-192`) is the template for the bulk leaf/internal split (`children_left[i] == -1` → leaf; else `cmp=kLE, default_left=true` per RESEARCH Pattern 3).

---

### `crates/treelite-core/src/serialize/mod.rs` + `binary.rs` (SER-01, the SerializerBackend trait)

**Analog:** No prior serializer exists in-repo (first-of-kind). Closest substrate analogs:

- **Byte-buffer building / byte-level care:** `crates/treelite-harness/src/lib.rs` `normalize_nan_tokens` (`lib.rs:132-161`) builds a `Vec<u8>` via `extend_from_slice` / `push` with careful byte handling — the same primitive the binary backend uses for `to_le_bytes` emission (RESEARCH §Code Examples `scalar_le`/`array_le`/`string_le`).
- **Reading columns to serialize:** `crates/treelite-core/src/tree_buf.rs` `as_slice` (`tree_buf.rs:58-65`) is the source of every column's bytes. The backend walks `tree.node_type.as_slice()` etc. in `serializer.cc` order (RESEARCH §Field order — Per tree).

**Trait shape** (RESEARCH Pattern 2):

```rust
trait SerializerBackend {
    fn scalar<T>(&mut self, value: &T);   // sizeof(T) LE bytes / 1 frame
    fn array<T>(&mut self, slice: &[T]);  // u64 count + payload / 1 frame
    fn string(&mut self, s: &str);        // u64 len + bytes / 1 frame
}
```

**Enum 1-byte widths are already correct** — `enums.rs` carries `#[repr(u8)]` (`TaskType` `enums.rs:24`, `DType` `enums.rs:160`) and `#[repr(i8)]` (`TreeNodeType` `enums.rs:68`, `Operator` `enums.rs:106`) matching the wire tags. Serialize via `as u8`/`as i8` (RESEARCH Pitfall 5). **Do not invent new type tags — reuse the existing enum reprs.**

**Recomputed-scalar staging (Pattern 5 lifetime subtlety):** the version triple / `num_tree` / type tags / opt-field `0`s are recomputed at serialize time and written into the **new private `Model` fields** (added this phase) so a borrowed-slice frame can point at them. This is *why* serialize lives in-core (D-10) — it needs `&mut` access to the private bookkeeping fields. See RESEARCH §Code Examples `serialize_header`.

---

### `crates/treelite-core/src/serialize/pybuffer.rs` (SER-02, zero-copy frames, D-05/D-06)

**Analog:** `crates/treelite-core/src/tree_buf.rs` — the **direct** zero-copy substrate. The frame enum borrows into `TreeBuf::as_slice()` (`tree_buf.rs:58-65`); the `Borrowed { ptr, len }` mode (`tree_buf.rs:24-31`) and `unsafe from_borrowed` (`tree_buf.rs:50-55`) are the existing borrow mechanism.

**Zero-copy assertion pattern to reuse in tests** — `crates/treelite-core/tests/tree_buf.rs:22-31` proves "borrowed slice points at the same memory" via `as_ptr()` equality. The SER-02 test asserts each frame's slice `.as_ptr()` matches the owning column's `.as_ptr()` (no copy).

**Frame enum** (RESEARCH Pattern 5) — idiomatic enum over borrowed slices, lifetime tied to `&'a Model`:

```rust
enum Frame<'a> {
    U8(&'a [u8]), I8(&'a [i8]), U32(&'a [u32]), I32(&'a [i32]),
    U64(&'a [u64]), I64(&'a [i64]), F32(&'a [f32]), F64(&'a [f64]),
    Str(&'a str),
}
```

A scalar is a 1-element slice; enum columns are reinterpreted to their 1-byte underlying slice. Frame order == binary field order (D-01).

---

### `crates/treelite-core/src/serialize/json.rs` (SER-03, DumpAsJSON, D-04)

**Analog:** `crates/treelite-xgboost/src/lib.rs` (serde-based JSON handling) for the serde_json dependency pattern; **enum string forms are already implemented** — reuse `as_str()`:
- `TaskType::as_str` (`enums.rs:40-48`) → `"kBinaryClf"` etc.
- `TreeNodeType::as_str` (`enums.rs:80-86`) → `"leaf_node"`/`"numerical_test_node"`/`"categorical_test_node"`.
- `Operator::as_str` (`enums.rs:124-133`) → `"<"`, `"<="`, … (`kNone` → `""`).
- `DType::as_str` (`enums.rs:174-181`) → `"float32"` etc.

RESEARCH Pattern 6 confirms these non-uniform spellings match upstream `*ToString`. **Compare parsed JSON values, not raw bytes** (D-04 structural fidelity; float formatting differs — RESEARCH A4/Q3).

---

### `crates/treelite-core/src/serialize/fields.rs` (SER-04, accessors)

**Analog:** `crates/treelite-core/src/tree.rs` getter convention (`tree.rs:107-158`): small `pub fn name(&self, nid) -> T { self.col[nid] }` methods, each doc-commented with the upstream `tree.h:NNN` line. SER-04 typed accessors (`model.num_feature()`, `tree.threshold()`) follow this exact shape (RESEARCH Pattern 7 recommends typed methods now, string-dispatch deferred to Phase 8).

Preserve upstream's read-only fields (`major_ver`/`num_tree`/… reject `Set`) — expose read accessors only for those.

---

### `crates/treelite-core/src/model.rs` (modify — add 6 private bookkeeping fields)

**Analog:** itself. The fields are already documented as a deferred-to-Phase-2 comment at `model.rs:85-87`:

```rust
// private serialization bookkeeping (tree.h:556-567) — deferred to Phase 2:
// num_tree_, num_opt_field_per_model_, major/minor/patch_ver_,
// threshold_type_, leaf_output_type_.
```

Add them as private fields (staged at serialize time per Pattern 5), and initialize in `Model::new` (`model.rs:93-110`) following the existing default-init block. `threshold_type_`/`leaf_output_type_` use the existing `DType` enum (`enums.rs:159-170`).

---

### `crates/treelite-core/src/tree.rs` (modify — add `num_opt_field_per_tree/node`)

**Analog:** itself. Documented deferred at `tree.rs:73-74`:

```rust
// num_opt_field_per_tree_/_per_node_ (tree.h:131-132) are serialization
// bookkeeping — deferred to Phase 2.
```

Add as scalar fields defaulting to `0`; initialize in `Tree::new` (`tree.rs:79-104`). Both always serialize as `0` (RESEARCH §Per tree #24/#25).

---

### `crates/treelite-core/src/lib.rs` (modify — wire the serialize module)

**Analog:** itself (`lib.rs:8-18`). Add `pub mod serialize;` to the `pub mod` block and re-export the public surface (e.g. `pub use serialize::{...}`) following the existing `pub use` block.

---

### `crates/treelite-xgboost/src/lib.rs` (rewire through builder, D-11)

**Analog:** itself (current form). The current `build_tree` (`lib.rs:151-207`) hand-assembles columns and `load_xgboost_json` (`lib.rs:217-288`) hand-assembles the `Model`. Rewiring replaces the direct `TreeBuf::from_owned` assembly with builder calls (`StartTree`/`StartNode`/`NumericalTest`/`LeafScalar`/`EndNode`/`EndTree`/`CommitModel`) per RESEARCH §System Architecture. The per-node leaf/internal branch (`lib.rs:172-192`) maps 1:1 to `LeafScalar` vs `NumericalTest` calls.

**Existing validation helpers stay** (`require_non_negative` `lib.rs:116-121`, `check_dim` `lib.rs:128-143`) — they run before emitting builder calls. The 1e-5 harness (`equivalence.rs`) is the regression gate; it must stay green after the rewire.

---

### `crates/treelite-xgboost/Cargo.toml` (modify — add builder dep, D-11)

**Analog:** itself (`Cargo.toml:7-12`). Add one line to `[dependencies]`:

```toml
treelite-builder = { path = "../treelite-builder" }
```

---

### `fixtures/capture_golden_v5.py` + `fixtures/golden_v5.bin` + `fixtures/golden_v5.manifest.json` (D-02)

**Analog:** `fixtures/capture_golden.py` (whole file) + `fixtures/golden.json` — an **exact** procedural + manifest analog. D-02 explicitly mirrors Phase 1's D-06/D-07.

**Reuse from `capture_golden.py`:**
- Path-resolution block (`capture_golden.py:32-35`): `HERE = os.path.dirname(os.path.abspath(__file__))`.
- Wheel load API (`capture_golden.py:47`): `treelite.frontend.load_xgboost_model(MODEL_PATH)` on the **same** `binary_logistic.model.json` fixture (RESEARCH §Specific Ideas: keep sigmoid exercised).
- **Manifest keys** (`capture_golden.py:71-78`) — copy exactly: `treelite`, `xgboost`, `os` (`platform.platform()`), `arch` (`platform.machine()`), `libc` (`platform.libc_ver()`), `python`. RESEARCH §Code Examples adds `sha256` + `nbytes` for the blob.

**Manifest deser analog** in Rust — `crates/treelite-harness/src/lib.rs` `Manifest` struct (`lib.rs:92-108`) is the exact shape to mirror when the golden_v5 test reads the manifest (`#[serde(default)]` on optional keys, `libc` as `serde_json::Value`).

**Capture call (RESEARCH §Code Examples, line 514-532):** `m.serialize_bytes()` → write raw bytes to `golden_v5.bin`. **Run once, commit, CI never regenerates** (`capture_golden.py:4-5` discipline).

---

## Shared Patterns

### thiserror error-enum convention
**Source:** `crates/treelite-xgboost/src/error.rs`, `crates/treelite-gtil/src/error.rs`, `crates/treelite-core/src/error.rs`
**Apply to:** `treelite-builder/src/error.rs`, the core serialize error enum.
- Module doc-comment cites the upstream `TREELITE_LOG(FATAL)`/`TREELITE_CHECK` path each variant replaces (xgboost `error.rs:1-9`).
- `#[derive(Debug, Error)]`; add `PartialEq, Eq` when variants are all comparable (core `error.rs:10`, gtil `error.rs:12`) — omit when a variant carries a `Box<dyn Error>` (xgboost `error.rs:14`).
- Variants carry the offending **value/index/key** for locality (gtil `FeatureIndexOutOfBounds` `error.rs:17-25`; xgboost `DimensionMismatch` `error.rs:53-62`).
- Cross-crate bubble-up via `#[error(transparent)] Core(#[from] treelite_core::CoreError)` (xgboost `error.rs:70-72`, gtil `error.rs:61-63`).
- Library crates use `thiserror` only — never `anyhow` (anyhow is test/harness-only, ERR-02).

### Input-validation-not-panic discipline (ASVS V5)
**Source:** `crates/treelite-xgboost/src/lib.rs` (`require_non_negative` `lib.rs:116-121`, `check_dim` `lib.rs:128-143`), `crates/treelite-gtil/src/error.rs` (`InvalidInputShape`, overflow-aware `required`).
**Apply to:** the deserializer (bound `u64` count/length prefixes against remaining buffer before allocating — RESEARCH §Security), and the builder (child-key/`split_index`/`num_feature` bounds). Never trust a stream-supplied count; return a typed `Err`, never panic/over-read.

### Module + doc-comment header convention
**Source:** every Phase 1 `src/*.rs` (e.g. `tree_buf.rs:1-12`, `xgboost/lib.rs:1-23`).
**Apply to:** all new files.
- `//!` module doc opening with the responsibility + the upstream `Ports treelite-mainline/...:NN-NN` citation.
- Per-public-item doc comments cite the upstream `tree.h:NNN` / `serializer.cc:NNN` line. **Critical for serialize order** (RESEARCH Pitfall 2): annotate each emitted column with the `serializer.cc:142-174` line so reviewers can audit field order against the spec.

### Zero-copy borrow + assertion
**Source:** `crates/treelite-core/src/tree_buf.rs` (`as_slice`, `Borrowed`, `from_borrowed`); test `crates/treelite-core/tests/tree_buf.rs:22-31`.
**Apply to:** SER-02 PyBuffer frames and their test (assert frame `.as_ptr()` == column `.as_ptr()`).

### Fixture-path resolution in tests
**Source:** `crates/treelite-harness/tests/equivalence.rs:21-27` (`Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name)`).
**Apply to:** `golden_v5.rs` byte-compare test and any serialize test reading a fixture. The golden_v5 test asserts `serialize(model) == read("fixtures/golden_v5.bin")` byte-for-byte and `deserialize(blob)` round-trips.

### Test construction style
**Source:** `crates/treelite-core/tests/tree_model.rs:8-38`, `crates/treelite-core/tests/tree_buf.rs`.
**Apply to:** `treelite-builder/tests/{validation,concat,bulk}.rs`, `treelite-core/tests/serialize_*.rs`.
- Build a small `Tree`/`Model` inline via `Tree::new()` + `TreeBuf::from_owned(vec![...])` (tree_model `tree_model.rs:11-17`).
- Plain `#[test]` + `assert_eq!`/`assert!` for unit tests; `-> anyhow::Result<()>` with `?` for integration tests that read fixtures (equivalence `equivalence.rs:30`).

---

## No Analog Found

| File | Role | Data Flow | Reason | Planner Direction |
|------|------|-----------|--------|-------------------|
| `crates/treelite-core/src/serialize/mod.rs` (the `SerializerBackend` trait + header/tree field walk) | serializer | request-response | No serializer/trait-dispatch code exists in Phase 1 — this is the first one. | Use **RESEARCH Pattern 2 + §Wire Format + §Code Examples** as the primary source; borrow byte-emission mechanics from harness `normalize_nan_tokens` and column reads from `TreeBuf::as_slice`. Field order is load-bearing — follow `serializer.cc:91-126` (header) and `:142-174` (per-tree) literally, NOT struct declaration order (Pitfall 2). |
| `crates/treelite-builder/src/lib.rs` (the state machine itself) | builder | event-driven | No fluent Begin/End state machine exists in Phase 1. | Structural shape (column-fill→`from_owned`, `Result`-propagation, doc headers) from `xgboost/lib.rs`; the 5-state machine + forward-ref/orphan semantics from **RESEARCH Pattern 1** (`model_builder.cc:50-388`). Internal `node_id_map_` data structure is discretion. |

(Both are first-of-kind in the Rust tree but have strong *structural* analogs in the table above; the missing analog is only the domain logic, which RESEARCH fully specifies from the vendored C++.)

---

## Metadata

**Analog search scope:** `crates/treelite-core/`, `crates/treelite-xgboost/`, `crates/treelite-gtil/`, `crates/treelite-harness/`, `fixtures/`, root `Cargo.toml`.
**Files scanned:** model.rs, tree.rs, tree_buf.rs, enums.rs, error.rs (core), lib.rs (core), xgboost lib.rs + error.rs + Cargo.toml, gtil error.rs, harness lib.rs + equivalence.rs + Cargo.toml, core tests (tree_model.rs, tree_buf.rs), core Cargo.toml, fixtures/capture_golden.py + golden.json, root Cargo.toml. (19 files.)
**Pattern extraction date:** 2026-06-10

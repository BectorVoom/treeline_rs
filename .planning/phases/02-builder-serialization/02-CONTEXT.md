# Phase 2: Builder & Serialization - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Widen the **construction** and **persistence** layers along the proven 1e-5 spine. Deliver a fluent, validated `ModelBuilder` (node-by-node), `ConcatenateModelObjects`, and a `BulkConstructTree` fast path; plus full **v5** serialization — binary round-trip, zero-copy PyBuffer, `DumpAsJSON`, and field accessors. The Phase 1 XGBoost-JSON loader is rewired to build *through* the new builder and must still verify within 1e-5.

**In scope:** fluent `ModelBuilder` with topology/orphan validation (BLD-01); `ConcatenateModelObjects` (BLD-02); `BulkConstructTree` fast path (BLD-03); v5 binary serialize+deserialize round-trip (SER-01); v5 zero-copy PyBuffer representation (SER-02); `DumpAsJSON` (SER-03); model/tree field accessors (SER-04); rewiring the Phase 1 XGBoost-JSON loader through the builder while holding 1e-5; the previously-deferred `Model` private serialization bookkeeping fields (`num_tree_`, `num_opt_field_per_model_`, `major/minor/patch_ver_`, `threshold_type_`, `leaf_output_type_`).

**Out of scope (later phases or PROJECT boundaries):** v3.9 / v4.0 wire formats (PROJECT out-of-scope — v5 only); UBJSON + legacy-binary XGBoost + auto-detect (Phase 3); LightGBM / sklearn loaders (Phase 4); full GTIL surface — 4 predict kinds, 10 postprocessors, sparse CSR, categoricals, output shaping (Phase 5); cubecl kernels (Phase 6); GPU (Phase 7); PyO3 — the actual Python buffer-protocol/C POD wiring (Phase 8); memory hardening — bytemuck/smallvec/compact_str/allocator (Phase 9). No C-API (PROJECT — PyO3 is the only binding).

</domain>

<decisions>
## Implementation Decisions

### Serialization Fidelity
- **D-01 (binary v5 = byte-identical cross-tool):** The v5 binary output MUST be byte-for-byte interoperable with upstream Treelite v5 — a model serialized by `treelite-rs` is readable by upstream C++ Treelite and vice-versa. This requires matching the exact field order, type tags, `PyBufferFrame` framing, version header, and opt-field counts that upstream's `serializer.cc` emits. This is the strongest expression of the "faithful port" promise and yields a strong golden-blob test.
- **D-02 (byte-fidelity validation artifact):** Validate D-01 against a **committed upstream-wheel-produced v5 blob + a toolchain/libm manifest**, captured once from the upstream Treelite Python wheel and frozen — mirroring Phase 1's golden-vector approach (Phase 1 D-06/D-07). CI never compiles C++. (Claude's discretion on exact capture mechanics; the artifact + manifest are mandatory.)
- **D-03 (deserialize rejects non-v5):** The deserializer accepts only v5; non-v5 input (e.g. v3.9/v4.0 headers) is rejected with a typed error. Aligns with PROJECT's v5-only boundary — do NOT port upstream's `DeserializeHeaderAndCreateModelV3` back-compat path.
- **D-04 (DumpAsJSON matches upstream structure):** `DumpAsJSON` (SER-03) mirrors the field names, nesting, and types of upstream `src/json_serializer.cc` output so a Rust dump is diffable against a C++ dump (enables JSON-level equivalence checking). Not a free-form Rust-native debug format.

### PyBuffer Zero-Copy Seam
- **D-05 (real zero-copy now):** Build a true zero-copy PyBuffer in Phase 2 — frames hold **borrowed references directly into the `Model`'s SoA `TreeBuf<T>` columns** (no copy). The zero-copy contract (SER-02, and the foundation for MEM-04) is proven now, not deferred. The frame list's lifetime is tied to `&Model`.
- **D-06 (Rust-native frame enum representation):** Represent the PyBuffer frame as an **idiomatic Rust enum over borrowed slices** (not a raw `void*` POD struct in Phase 2). It is converted/adapted to the C buffer-protocol POD layout (`TreelitePyBufferFrame`-equivalent) only at the **Phase 8 (PyO3)** boundary. Keeps Phase 2 safe (no unsafe raw pointers) while still being genuinely zero-copy.
- **Note (consistency with D-01):** Because the binary stream is byte-identical and is the concatenation of these frames in fixed field order, the *frame list contents/order* are pinned by D-01 — the enum is a representation choice over already-fixed framing, not a freedom to reorder.

### Builder Validation
- **D-07 (eager validation, as early as correctness allows):** Validate per-node well-formedness at `EndNode` (leaf-vs-test mutual exclusivity, duplicate node keys, valid args), and per-tree topology — orphans, dangling child keys, reachability — at `EndTree` (deferred to tree close because child keys may be **forward references** resolved only when all nodes in the tree are known). Errors should surface at the offending call site with good locality where possible.
- **D-08 (always-strict, no opt-out):** The fluent node-by-node builder is always strict — do NOT port upstream's `SetValidationFlag("check_orphaned_nodes", ...)` configurable toggle. Validation cannot be turned off on the fluent path.
- **D-09 (BulkConstructTree is a separate pre-validated path):** `BulkConstructTree` (BLD-03) is NOT "the strict builder with checks disabled." It is a distinct fast constructor that consumes pre-validated bulk input and bypasses node-by-node validation by construction. Document the bypass behavior to match upstream. This is how the always-strict fluent path (D-08) and a validation-bypassing bulk path coexist.

### Crate Layout
- **D-10 (builder in new crate, serialize in core):** Create a new `treelite-builder` crate (one responsibility, per Phase 1 D-01's per-phase crate growth). Put **serialization** (v5 binary, PyBuffer, `DumpAsJSON`) and **field accessors** (SER-04) as a **module inside `treelite-core`**, co-located with the `Model` internals they encode — avoids leaking `Model`'s private serialization fields across a crate boundary.
- **D-11 (loader rewiring dependency):** The existing `treelite-xgboost` loader is rewired to build through `treelite-builder` (success criterion 1) and therefore gains a dependency on it. It must still load the Phase 1 fixture and verify within 1e-5 after rewiring.

### Claude's Discretion
- `ConcatenateModelObjects` (BLD-02) merge semantics — header-compatibility checks (matching `num_feature` / `task_type` / threshold+leaf types), tree concatenation, and `target_id`/`class_id` handling — follow upstream `src/model_concat.cc`.
- Field-accessor surface (SER-04) — which `Model`/`Tree` fields are exposed and the accessor API shape — follow upstream `GetHeaderField` / tree-field accessors; keep idiomatic Rust.
- Typed error-enum granularity for the builder and serializer (per-crate `thiserror` enums vs. shared) — default per-crate, idiomatic, consistent with Phase 1.
- Exact mechanics of capturing the D-02 upstream v5 golden blob + manifest format/location (mirror Phase 1's manifest conventions).
- Internal representation of the builder's in-progress node/tree state and node-key resolution data structure.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level (this milestone)
- `.planning/PROJECT.md` — core value (1e-5 equivalence), v5-only serialization boundary, no C-API, Key Decisions, Out of Scope.
- `.planning/REQUIREMENTS.md` — Phase 2 IDs: BLD-01, BLD-02, BLD-03, SER-01, SER-02, SER-03, SER-04 (full text + traceability).
- `.planning/ROADMAP.md` § "Phase 2: Builder & Serialization" — goal + 4 success criteria (the authoritative acceptance bar).
- `.planning/phases/01-end-to-end-spine/01-CONTEXT.md` — Phase 1 decisions carried forward: D-01 spine-only crate growth, D-08 defer-the-abstraction philosophy, D-06/D-07 golden+manifest capture pattern.

### Upstream porting source of truth (`treelite-mainline/`, C++ v4.7.0) — Builder
- `treelite-mainline/include/treelite/model_builder.h` — `ModelBuilder` interface (StartTree/EndTree, StartNode/EndNode, NumericalTest/CategoricalTest, LeafScalar/LeafVector, Gain/DataCount/SumHess, InitializeMetadata, CommitModel), `Metadata`, `TreeAnnotation`, `PostProcessorFunc`, `GetModelBuilder` factory variants, `SetValidationFlag` (NOTE: D-08 intentionally drops the toggle).
- `treelite-mainline/src/model_builder/` (esp. `model_builder.cc`) — concrete builder: node-key resolution, orphan/topology validation timing (informs D-07), `BulkConstructTree` bulk path behavior (BLD-03 / D-09).
- `treelite-mainline/src/model_concat.cc` — `ConcatenateModelObjects` merge semantics (BLD-02).

### Upstream porting source of truth — Serialization
- `treelite-mainline/src/serializer.cc` — `Serializer<MixIn>::SerializeHeader`/`SerializeTrees`/`SerializeTree` — **the authoritative v5 field order, type tags, num_tree, opt-field counts** (D-01 byte fidelity). Also `DeserializeHeaderAndCreateModelV3` (do NOT port — D-03).
- `treelite-mainline/include/treelite/detail/serializer.h` + `treelite-mainline/include/treelite/detail/serializer_mixins.h` — mixin-based `Serializer`/`Deserializer` template → port to a trait-based Rust serializer; defines the stream/PyBuffer/buffer I/O backends.
- `treelite-mainline/include/treelite/pybuffer_frame.h` + `treelite-mainline/include/treelite/c_api.h:53` (`TreelitePyBufferFrame`) — POD frame layout {buf ptr, format, itemsize, nitems} the Rust frame enum (D-06) must map onto at Phase 8.
- `treelite-mainline/src/json_serializer.cc` — `DumpAsJSON` structure (D-04 fidelity target).
- `treelite-mainline/include/treelite/tree.h:480-520` — version fields, compatibility matrix, `SerializeToStream`/`SerializeToPyBuffer`/`SerializeToBuffer`/`DumpAsJSON` signatures, `GetHeaderField` accessor (SER-04).

### Existing Rust code (Phase 1 output — the build target)
- `crates/treelite-core/src/model.rs` — `Model` + `ModelVariant` + header metadata; the deferred private serialization fields are listed in a comment to be added in Phase 2.
- `crates/treelite-core/src/tree.rs` + `crates/treelite-core/src/tree_buf.rs` — `Tree<T>` SoA columns + `TreeBuf<T>` owned/borrowed buffer (frames borrow into these per D-05).
- `crates/treelite-xgboost/src/lib.rs` — Phase 1 loader to be rewired through `treelite-builder` (D-11).
- `crates/treelite-harness/` — 1e-5 equivalence harness that must stay green after rewiring.

### Codebase maps
- `.planning/codebase/ARCHITECTURE.md` — SoA + variant pattern, `ModelBuilder` validation anti-pattern (no direct `Tree` mutation), mixin serializer pattern, no-copy `Model`/`Tree`.
- `.planning/codebase/CONVENTIONS.md` — naming + `thiserror`/`anyhow` translation notes.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Model` / `ModelVariant` / header metadata** (`crates/treelite-core/src/model.rs`) already exist; Phase 2 adds the deferred private bookkeeping fields (`num_tree_`, `num_opt_field_per_model_`, `major/minor/patch_ver_`, `threshold_type_`, `leaf_output_type_`) needed by the serializer header.
- **`TreeBuf<T>` owned/borrowed SoA buffer** is the zero-copy substrate — PyBuffer frames borrow directly into its columns (D-05). Borrowed mode (CORE-03) is the mechanism.
- **`thiserror` per-crate error pattern** from Phase 1 carries into `treelite-builder` and the core serialize module.
- **Upstream is spec, not reusable code** — `treelite-mainline/` is read-only porting reference.

### Established Patterns (preserve from upstream)
- Struct-of-Arrays tree storage (parallel columns) → serializer walks columns in fixed order (D-01).
- Type-erased `Model` over `<f32,f32>`/`<f64,f64>` — serializer dispatches over the variant; threshold/leaf type tags are written into the header.
- Mixin `Serializer<MixIn>` (stream / PyBuffer / buffer backends) → **trait-based Rust serializer** with pluggable I/O backends.
- Fluent builder with Begin/End pairing + `CommitModel()` finalize.

### Integration Points
- `treelite-builder` (new) depends on `treelite-core`; `treelite-xgboost` rewired to depend on `treelite-builder` (D-11).
- Serialize module lives in `treelite-core` (D-10), with access to `Model` private fields.
- The 1e-5 harness is the regression gate for the loader rewiring.

</code_context>

<specifics>
## Specific Ideas

- The v5 binary stream is the concatenation of `PyBufferFrame`s in a fixed field order: version (major/minor/patch) → threshold_type tag → leaf_output_type tag → num_tree → header-2 block (num_feature, task_type, average_tree_output, num_target, num_class[], leaf_vector_shape[], target_id[], class_id[], postprocessor, sigmoid_alpha, ratio_c, base_scores[], attributes) → opt-field count (0) → per-tree (num_nodes, has_categorical_split, then each SoA column). Match this exactly for D-01.
- Use `binary:logistic` Phase 1 fixture as the round-trip + rewiring subject so sigmoid stays exercised end-to-end through the builder.
- Capture the upstream v5 golden blob from the same Treelite wheel used for the Phase 1 golden, committed with a manifest (treelite version, OS/arch, libm/glibc) — a frozen unit like the Phase 1 golden.

</specifics>

<deferred>
## Deferred Ideas

- **Actual Python buffer-protocol / C POD conversion of the frame enum** — Phase 8 (PyO3). Phase 2 proves zero-copy borrowing into `TreeBuf`; Phase 8 maps the frame enum onto `TreelitePyBufferFrame` and the buffer protocol (D-06, MEM-04).
- **v3.9 / v4.0 read/write and cross-version migration** — PROJECT out-of-scope (v2 requirement SER-v2-01). Phase 2 is v5-only; deserialize rejects older formats (D-03).
- **`SetValidationFlag` configurable validation toggle** — intentionally not ported (D-08); could be revisited only if a future loader path needs it (it doesn't — the bulk path covers bypass per D-09).
- **bytemuck `Pod` zero-copy recast of SoA columns** — Phase 9 (MEM-01); Phase 2 borrows via `TreeBuf` borrowed mode without bytemuck.

</deferred>

---

*Phase: 2-Builder & Serialization*
*Context gathered: 2026-06-10*

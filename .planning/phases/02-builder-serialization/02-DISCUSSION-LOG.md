# Phase 2: Builder & Serialization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 2-Builder & Serialization
**Areas discussed:** Serialization fidelity, PyBuffer seam timing, Builder validation, Crate layout

---

## Serialization fidelity — Binary v5

| Option | Description | Selected |
|--------|-------------|----------|
| Byte-identical cross-tool | Rust v5 output byte-for-byte readable by upstream C++ Treelite v5 and vice-versa; match exact field order, type tags, PyBufferFrame framing, version, opt-field counts | ✓ |
| Self-consistent round-trip | Rust serialize→deserialize identical, no cross-tool byte guarantee | |
| Byte-identical, defer reverse | Output byte-identical to upstream; reading arbitrary upstream blobs only smoke-tested now | |

**User's choice:** Byte-identical cross-tool
**Notes:** Validate against a committed upstream-wheel-produced v5 blob + manifest (mirrors Phase 1 D-06/D-07). Deserialize rejects non-v5 per PROJECT v5-only boundary — accepted as Claude's discretion.

## Serialization fidelity — DumpAsJSON

| Option | Description | Selected |
|--------|-------------|----------|
| Match upstream JSON structure | Field names/nesting/types mirror json_serializer.cc; diffable against C++ dump | ✓ |
| Rust-native inspection format | Clean serde-derived JSON for debug only, not cross-comparable | |
| You decide | Defer to Claude | |

**User's choice:** Match upstream JSON structure
**Notes:** Enables JSON-level equivalence checking; keeps faithful-port property.

---

## PyBuffer seam timing — Zero-copy realness

| Option | Description | Selected |
|--------|-------------|----------|
| Real zero-copy frames now | Frames borrow pointers/slices directly into Model SoA TreeBuf columns; Phase 8 hands them to Python | ✓ |
| Frame shape now, copy ok, borrow later | Define frame struct + list now, copied spans ok, defer true borrow to Phase 8 | |
| You decide | Defer to Claude | |

**User's choice:** Real zero-copy frames now
**Notes:** Frame list lifetime tied to &Model; zero-copy contract (SER-02, MEM-04 foundation) proven in Phase 2.

## PyBuffer seam timing — Frame representation

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror TreelitePyBufferFrame | Plain POD struct {buf ptr, format, itemsize, nitems}, 1:1 with Python buffer protocol | |
| Rust-native frame enum | Idiomatic enum/slice-based repr, adapted to C buffer layout at Phase 8 boundary | ✓ |
| You decide | Defer to Claude | |

**User's choice:** Rust-native frame enum
**Notes:** Combined with "real zero-copy now": enum over borrowed slices into TreeBuf (safe, no raw pointers in Phase 2); POD conversion at Phase 8. Frame contents/order still pinned by the byte-identical binary decision.

---

## Builder validation — Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Mostly at CommitModel | Cheap inline checks; full topology validation at CommitModel | |
| Eagerly per node/tree | Validate as early as possible at EndNode/EndTree for error locality | ✓ |
| You decide | Match upstream timing | |

**User's choice:** Eagerly per node/tree
**Notes:** Refined to: per-node well-formedness at EndNode; per-tree topology/orphan/reachability at EndTree (child keys can be forward references resolved only at tree close).

## Builder validation — Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Keep SetValidationFlag | Port upstream's configurable check_orphaned_nodes toggle (default on) | |
| Always strict | Always validate, no opt-out | ✓ |
| You decide | Follow upstream + loader/bulk needs | |

**User's choice:** Always strict
**Notes:** BulkConstructTree (BLD-03) captured as a separate pre-validated fast path, not the strict builder with checks disabled — that is how bypass coexists with always-strict.

---

## Crate layout

| Option | Description | Selected |
|--------|-------------|----------|
| Two new crates | treelite-builder + treelite-serialize; needs Model serialization fields exposed cross-crate | |
| Builder crate, serialize in core | New treelite-builder; serialize as a module in treelite-core with natural access to Model internals | ✓ |
| Both modules in core | Both in treelite-core; tightest coupling, multi-responsibility core | |
| You decide | Honor D-01 without visibility leaks | |

**User's choice:** Builder crate, serialize in core
**Notes:** Avoids leaking Model's private serialization fields across a crate boundary; serializer co-located with the data it encodes. treelite-xgboost gains a dependency on treelite-builder after rewiring.

---

## Claude's Discretion

- `ConcatenateModelObjects` (BLD-02) merge semantics — follow upstream `model_concat.cc`.
- Field-accessor surface (SER-04) — follow upstream `GetHeaderField`/tree accessors, idiomatic Rust.
- Builder/serializer typed error-enum granularity — default per-crate `thiserror`.
- Upstream v5 golden-blob capture mechanics + manifest format/location.
- Builder in-progress node/tree state + node-key resolution data structure.
- Deserialize rejects non-v5 (no v3 back-compat port).

## Deferred Ideas

- Actual Python buffer-protocol / C POD conversion of the frame enum — Phase 8 (PyO3).
- v3.9 / v4.0 read/write + cross-version migration — PROJECT out-of-scope (v2 SER-v2-01).
- `SetValidationFlag` configurable toggle — intentionally not ported.
- bytemuck `Pod` zero-copy recast of SoA columns — Phase 9 (MEM-01).

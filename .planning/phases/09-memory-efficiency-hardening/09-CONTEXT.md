# Phase 9: Memory-Efficiency Hardening - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Apply the project's memory-efficiency techniques across the already-proven,
equivalence-tested workspace **without regressing the 1e-5 contract or the
byte-identical v5 golden** — the final v1 phase, closing MEM-01/02/03.

Three deliverables, all scoped maximally per this discussion:
1. **MEM-01** — `bytemuck` `Pod` zero-copy recasting pushed everywhere layout
   allows (core SoA columns, serializer, loaders), LE-host-only.
2. **MEM-02** — `smallvec` + `compact_str` replacing the public small-collection
   and metadata-string field types on the `Model` header.
3. **MEM-03** — both `jemalloc` and `mimalloc` wired as runtime-selectable
   global allocators in benchmarks/binaries only (never the abi3 wheel), plus a
   committed observational memory report demonstrating the win.

**Not in scope:** new functionality, format changes, GPU work, or any change
that alters prediction output, the Python API contract, or the v5 wire bytes.
</domain>

<decisions>
## Implementation Decisions

### MEM-01 — bytemuck Pod zero-copy recast
- **D-01:** Push Pod recasting **everywhere feasible** — extend the existing
  `TreeBuf<T: Copy + bytemuck::Pod>::as_bytes()` seam (`tree_buf.rs:96`, today
  confined to device upload) across core SoA columns, the v5 serializer, and
  loaders wherever the buffer layout permits. Maximum zero-copy reach.
- **D-02:** **Endianness: little-endian-only zero-copy; big-endian hosts are
  out of scope for v1.** `bytemuck::cast_slice` yields native-endian bytes; the
  dev box + CI are x86/ROCm (LE) and the frozen golden was captured LE, so
  native-endian casts on the serialize path are acceptable. **No portable
  BE byte-swap fallback** is to be built. The LE assumption MUST be documented
  at every recast site that touches serialized output.
- **D-03 (invariant):** Despite the aggressive reach, the v5 serializer MUST
  still emit **byte-identical** output to `fixtures/golden_v5.bin` /
  `fixtures/golden_v5_3format.bin`. Any recast that would change emitted bytes
  is not "where layout allows" — the golden compare is the gate.

### MEM-02 — smallvec / compact_str
- **D-04:** **Change the public `Model` field types directly** — the metadata
  `Vec<i32>` fields (`num_class`, `leaf_vector_shape`, `target_id`, `class_id`)
  and `base_scores` (`Vec<f64>`) become `SmallVec`-backed; `postprocessor` and
  `attributes` (`String`) become `CompactString`-backed. The ripple through
  serializer, builder, loaders, and GTIL is accepted and must be updated in
  lockstep.
- **D-05:** **Storage-only at the boundaries.** The change is internal storage
  even though the Rust field types change: the PyO3 accessors (Phase 8) MUST
  still return `list`/`str` to Python, and the v5 serializer MUST still emit
  byte-identical bytes (SmallVec/CompactString deref to the same payload). The
  Phase-8 binding contract and the frozen golden are both preserved. **Planner
  must verify both explicitly** (binding tests + golden byte-compare).
- **D-06 (research detail):** Inline sizing for the SmallVecs and the
  compact_str benefit on the potentially-large `attributes` JSON blob are left
  to the researcher/planner to profile (CompactString is harmless on large
  strings — heap-equivalent above its inline threshold — so applying it
  uniformly is safe even where it gives no win).

### MEM-03 — custom global allocator
- **D-07:** Wire **both jemalloc and mimalloc**, runtime-selectable via
  **mutually-exclusive Cargo features**, **neither default**. Benchmarks can
  then compare the two. (Researcher picks the concrete crates, e.g.
  `tikv-jemallocator` / `mimalloc`, per the manuals.)
- **D-08:** Allocator is installed as `#[global_allocator]` **only in harness
  binaries + the new benchmark targets**. Library crates and the **abi3 cpu
  wheel keep the system default allocator** — the allocator must never be
  reachable from the wheel build. Matches SC3 verbatim.
- **D-09:** Both allocators must be validated to **build + import/run on Linux**
  (the AMD/ROCm dev box).

### Validation bar
- **D-10:** **Committed observational memory report** — a new benchmark target
  measures peak RSS / allocation (leveraging jemalloc stats) and writes a
  committed `MEMORY_REPORT.md` showing before/after, following the Phase-7
  `GPU_EQUIVALENCE_REPORT.md` observational precedent. **No brittle hard CI
  threshold gate.**
- **D-11 (implicit floor):** Independent of the report, the phase is not done
  until the **full equivalence harness stays green within 1e-5** and **all
  existing workspace + Python tests pass** after the type/allocator changes.
  This is the real pass/fail gate; the report is evidence the hardening worked.

### Claude's Discretion
- SmallVec inline capacities, the exact jemalloc/mimalloc crate selection, the
  benchmark model set, and the report's exact columns — all delegated to
  research/planning (see D-06, D-07, D-10).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Memory-efficiency playbook (optimiser manual)
- `/home/user/Documents/workspace/optimisor/manual/SMALLVEC_MANUAL.md` — smallvec usage + inline-size guidance (MEM-02)
- `/home/user/Documents/workspace/optimisor/manual/COMPACT_STR_OPTIMIZATION_EN.md` — compact_str adoption + inline threshold semantics (MEM-02)
- `/home/user/Documents/workspace/optimisor/manual/JEMALLOC_MANUAL.md` — jemalloc wiring + stats for the memory report (MEM-03, D-10)
- `/home/user/Documents/workspace/optimisor/manual/MIMALLOC_MANUAL.md` — mimalloc wiring (MEM-03)
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_TRANSMUTATION_CUBECL.md` — bytemuck Pod transmute patterns + alignment/endianness pitfalls (MEM-01)
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_ARROW_CUBECL.md` — related zero-copy buffer patterns (MEM-01 background)

### Requirements + invariants
- `.planning/REQUIREMENTS.md` §MEM-01/MEM-02/MEM-03 — the three requirements this phase closes
- `.planning/ROADMAP.md` §"Phase 9" — goal + the 3 success criteria
- `fixtures/golden_v5.bin`, `fixtures/golden_v5_3format.bin` — the byte-identical v5 goldens that MUST still match after MEM-01/MEM-02 (D-03/D-05)

### Code touch-points
- `crates/treelite-core/src/tree_buf.rs` — existing `TreeBuf` + `as_bytes()` Pod seam to extend (MEM-01)
- `crates/treelite-core/src/model.rs` — the public metadata fields (`num_class`/`leaf_vector_shape`/`target_id`/`class_id`/`base_scores`/`postprocessor`/`attributes`) to convert (MEM-02)
- `crates/treelite-cubecl/src/lib.rs` — `PredictCpuElem` already bounds `bytemuck::Pod` (reference for the Pod element constraints)
- `crates/treelite-harness/src/bin/` — where the allocator + benchmark targets live (MEM-03, D-08/D-10)
- `crates/treelite-py/` — abi3 wheel that must stay allocator-free and keep list/str accessors (D-05/D-08)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TreeBuf<T: Copy + bytemuck::Pod>::as_bytes()` already exists and is tested
  (`tree_buf.rs:96`, roundtrip tests at :122) — Phase 9 extends this seam rather
  than introducing bytemuck from scratch. `bytemuck` is already a pinned
  `[workspace.dependencies]` entry.
- `PredictCpuElem: Float + CubeElement + bytemuck::Pod` in treelite-cubecl shows
  the Pod-bound element pattern already in use on the hot path.
- The Phase-7 `GPU_EQUIVALENCE_REPORT.md` writer is the template/precedent for
  the new committed observational `MEMORY_REPORT.md` (D-10).

### Established Patterns
- **Struct-of-Arrays** columns (`TreeBuf`) are the primary Pod-recast target —
  flat `Vec<T>`/borrowed slices of plain numeric `T`.
- **Byte-identical v5 serialization** is a hard, tested invariant — the goldens
  are fatal compares. MEM-01's serializer reach is bounded by it (D-03).
- **`Model` is `!Send`** (TreeBuf::Borrowed raw pointers) — unchanged by this
  phase; the allocator/type changes must not touch that property.
- Allocators are feature-gated and confined to bins (the abi3 cpu wheel never
  pulls cubecl/allocator deps — same discipline as Phase 8's optional cubecl).

### Integration Points
- MEM-02 field-type changes ripple into: treelite-core serializer, treelite-builder,
  the loaders (xgboost/lightgbm/sklearn write these fields), treelite-gtil (reads
  them), and treelite-py accessors — all must compile + keep tests green.
- MEM-03 allocator wiring connects only at binary/bench entry points via
  `#[global_allocator]`; no library crate gains the dependency.

</code_context>

<specifics>
## Specific Ideas

- The user consistently chose the **maximal** option on every fork (push
  bytemuck everywhere, change public types directly, ship both allocators,
  add a real benchmark) — bias the plan toward thorough coverage, bounded only
  by the two hard invariants (1e-5 harness green + byte-identical golden).
- Endianness is the one deliberate scope cut: LE-only, BE explicitly out of
  scope for v1 (D-02). Document it; do not build a swap path.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Big-endian portability is the one
explicitly out-of-scope item, recorded as decision D-02 rather than a deferred
idea, since it bounds MEM-01 directly.)

</deferred>

---

*Phase: 9-Memory-Efficiency Hardening*
*Context gathered: 2026-06-11*

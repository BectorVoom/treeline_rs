# Phase 9: Memory-Efficiency Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 9-Memory-Efficiency Hardening
**Areas discussed:** bytemuck scope, MEM-02 blast radius, allocator choice, validation bar, endianness, benchmark role, FFI/wire boundary

---

## bytemuck scope (MEM-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Pod seam only (minimal) | Confine recasting to the existing as_bytes() upload path | |
| Extend into serialization | Also replace serializer byte copies with bytemuck casts | |
| Push everywhere feasible | Aggressively recast every Pod-eligible buffer across core/serializer/loaders | ✓ |

**User's choice:** Push everywhere feasible
**Notes:** Bounded by the byte-identical golden compare (D-03) and the LE-only endianness cut (D-02).

---

## MEM-02 blast radius

| Option | Description | Selected |
|--------|-------------|----------|
| Change public field types | Swap public Model fields to SmallVec/CompactString; ripple through all consumers | ✓ |
| Internal storage only | Keep Vec/String public API, change only internal storage | |
| Selective by field | Apply per field where profiling shows benefit | |

**User's choice:** Change public field types
**Notes:** Reconciled with the FFI/wire follow-up — change is storage-only at the boundaries (Python returns list/str, v5 bytes unchanged). See D-05.

---

## Allocator (MEM-03)

| Option | Description | Selected |
|--------|-------------|----------|
| jemalloc, feature-gated in bins/benches | Single allocator, tikv-jemallocator, out of wheel | |
| jemalloc + mimalloc, selectable | Both behind mutually-exclusive features for comparison | ✓ |
| Defer allocator choice to research | Lock wiring location, let researcher pick | |

**User's choice:** jemalloc + mimalloc, selectable
**Notes:** Mutually-exclusive features, neither default, only in bins + benches; abi3 cpu wheel stays default-allocator (D-07/D-08).

---

## Validation bar

| Option | Description | Selected |
|--------|-------------|----------|
| No-regression only | Harness green within 1e-5 + existing tests pass; no new infra | |
| Add a memory benchmark | Benchmark measures allocation/peak RSS to demonstrate the reduction | ✓ |
| Lightweight size assertions | static size_of/capacity assertions only | |

**User's choice:** Add a memory benchmark
**Notes:** Refined to a committed observational report (Phase-7 precedent), not a hard CI gate. The 1e-5 harness + existing tests remain the real pass/fail floor (D-10/D-11).

---

## Endianness (MEM-01 follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| LE-only zero-copy, BE out of scope | Native-endian bytemuck casts; big-endian hosts out of scope for v1 | ✓ |
| Zero-copy on LE, explicit swap on BE | Fast LE path + byte-swap fallback on BE | |
| Keep serializer portable, recast elsewhere | Leave serializer LE encoding; confine recast to in-memory/upload | |

**User's choice:** LE-only zero-copy, BE out of scope
**Notes:** Dev box + CI are x86/ROCm (LE), golden captured LE. Document the LE assumption at recast sites; no swap path built (D-02).

---

## Benchmark role (validation follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Committed observational report | MEMORY_REPORT.md before/after via jemalloc stats; no hard gate | ✓ |
| Hard regression gate | Assert a concrete reduction threshold, fail CI otherwise | |
| Report + cheap size_of guards | Observational report plus static size_of/inline-capacity guards | |

**User's choice:** Committed observational report
**Notes:** Mirrors Phase-7 GPU_EQUIVALENCE_REPORT.md observational precedent (D-10).

---

## FFI/wire boundary (MEM-02 follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Storage-only; Python+wire unchanged | Field types change but py accessors return list/str and v5 bytes stay identical | ✓ |
| Let new types surface in Python | Allow new types into the Python API where convenient | |

**User's choice:** Storage-only; Python+wire unchanged
**Notes:** Phase-8 binding contract + frozen golden both preserved; planner must verify both explicitly (D-05).

---

## Claude's Discretion

- SmallVec inline capacities; concrete jemalloc/mimalloc crate selection; benchmark model set; MEMORY_REPORT.md columns (D-06/D-07/D-10).

## Deferred Ideas

None — discussion stayed within phase scope. Big-endian portability is recorded as in-scope-cut decision D-02 rather than a deferred idea.

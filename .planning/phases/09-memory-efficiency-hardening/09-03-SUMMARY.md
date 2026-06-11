---
phase: 09-memory-efficiency-hardening
plan: 03
subsystem: core-serializer
tags: [mem-01, bytemuck, cast_slice, pod-bound, golden-byte-compat, unsafe-removal, serialize-emit]

# Dependency graph
requires:
  - phase: 09-memory-efficiency-hardening
    plan: 02
    provides: "Model metadata migrated to SmallVec/CompactString; the migrated fields deref to &[i32]/&[f64] Pod slices, ready for the recast; serialize EMIT seam left untouched"
  - phase: 02-core-model-serializer
    provides: "v5 serializer le_bytes_of emit + golden_v5 byte-compare gate (the MEM-01 recast target / invariant)"
provides:
  - "le_bytes_of<T: bytemuck::Pod> routed through bytemuck::cast_slice (the tested tree_buf.rs as_bytes seam); no unsafe/from_raw_parts remains in the fn"
  - "serialize_tree<T> bound tightened Copy -> Copy + bytemuck::Pod (f32/f64 presets are Pod); concrete callers unchanged"
  - "D-02 LE-host-only assumption documented at the recast site (no big-endian byte-swap path)"
  - "scalar_le / enum (as u8) / bool_bytes emits and the untrusted deserialize read path (binary.rs Reader::array) NOT recast (V5 security)"
affects: [09-04 (MEM-03 allocator RSS report — the final Phase-9 plan; MEM-01 closes here)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pod-recast through the validated bytemuck::cast_slice seam: replace a hand-rolled &[T]->&[u8] from_raw_parts transmute with bytemuck::cast_slice on the EMIT direction only, bounded by the byte-identical golden compare"
    - "Bound propagation: narrowing le_bytes_of to T: bytemuck::Pod forces serialize_tree<T> to carry the Pod bound; only f32/f64 instantiate it, so callers (serialize_trees match arms) compile unchanged"

key-files:
  created: []
  modified:
    - crates/treelite-core/src/serialize/mod.rs

key-decisions:
  - "le_bytes_of body is now bytemuck::cast_slice(slice) with bound T: bytemuck::Pod (was T: Copy + hand-rolled unsafe from_raw_parts) — the exact tree_buf.rs:102 as_bytes form (D-01). No unsafe remains in le_bytes_of."
  - "serialize_tree<T> bound tightened Copy -> Copy + bytemuck::Pod (Rule 3 blocking): the per-tree TreeBuf<T> columns (leaf_value/threshold/leaf_vector) feed le_bytes_of, so T must satisfy Pod. The only two instantiations are f32/f64 (both Pod), so serialize_trees' F32/F64 match arms compile verbatim — mechanical, not architectural."
  - "bool_bytes (default_left / *_present / category_list_right_child) and the explicit enum `as u8` maps (node_type / cmp) are NOT recast — they define their own wire byte (D-01 'where layout allows' boundary, T-09-10). The untrusted deserialize read path (binary.rs Reader::array) is NOT recast (T-09-D); grep confirms 0 cast_slice in binary.rs."

requirements-completed: [MEM-01]

# Metrics
duration: 2min
completed: 2026-06-11
---

# Phase 9 Plan 03: le_bytes_of bytemuck::cast_slice Recast (MEM-01) Summary

**Replaced the serializer's hand-rolled `unsafe { from_raw_parts }` in `le_bytes_of` with the existing, tested `bytemuck::cast_slice` seam (the exact `tree_buf.rs::as_bytes` form), tightening the bound `T: Copy` → `T: bytemuck::Pod` and documenting the little-endian-host-only assumption (D-02) at the site — a one-function change that removes an `unsafe` block while emitting byte-identical bytes, proven by `golden_v5.bin` + `golden_v5_3format.bin` (D-03) and the 1e-5 equivalence harness (D-11). The recast is restricted to the safe `&[T] → &[u8]` EMIT direction; scalar/enum/bool emits and the untrusted deserialize read path are untouched (V5 security).**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-06-11T03:01:53Z
- **Completed:** 2026-06-11
- **Tasks:** 2
- **Files modified:** 1 production file

## Accomplishments

- **Task 1 — `le_bytes_of` routed through `bytemuck::cast_slice` (the recast):** the body became `bytemuck::cast_slice(slice)` with the signature `fn le_bytes_of<T: bytemuck::Pod>(slice: &[T]) -> &[u8]` — the verbatim `tree_buf.rs:102` template. No `unsafe`/`from_raw_parts` remains in the fn. The bound narrowing forced `serialize_tree<T>`'s bound to `T: Copy + bytemuck::Pod` (the per-tree `TreeBuf<T>` value columns feed `le_bytes_of`); the only two instantiations are `f32`/`f64`, both `Pod`, so the `serialize_trees` F32/F64 match arms compile unchanged. The D-02 LE-host-only doc (no big-endian byte-swap path, gated by the golden) is at the site. `cargo build/test -p treelite-core` (tree_buf + serialize_roundtrip) green.
- **Task 2 — both hard invariants green (the gate):** no production edit needed — the golden never diverged. `golden_v5.bin` AND `golden_v5_3format.bin` byte-identical (HARD INVARIANT 1); full harness equivalence/matrix/three-format within 1e-5 (HARD INVARIANT 2); `cargo test --workspace` 0 failures; `cargo test -p treelite-core --test model_invariants` green (no MEM-02 regression); `uv run pytest crates/treelite-py` 39 passed / 1 skipped. MEM-01 closed.

## Task Commits

Each task was committed atomically:

1. **Task 1: Route le_bytes_of through bytemuck::cast_slice (Pod bound, LE-documented)** — `2e1e288` (refactor)
2. **Task 2: Prove both invariants green (golden + 1e-5 + pytest)** — gate task, no production edit (golden byte-identical); proven against `2e1e288`, no commit of its own.

_Task 1 is `tdd="true"`; the RED/GREEN guard is the pre-existing `golden_v5` byte-compare + `serialize_roundtrip` + `tree_buf` cast_slice-roundtrip suite (the Phase-2 + Plan-01 baseline). This is a transmute→`cast_slice` mechanical swap with ZERO new behavior — like Plan 02, the existing green tests are the gate; the byte image is identical on the LE host, so there is no separate failing-then-passing behavior to author. Every gate stayed green._

## Files Created/Modified

- `crates/treelite-core/src/serialize/mod.rs` — `le_bytes_of` body → `bytemuck::cast_slice`, bound `T: Copy` → `T: bytemuck::Pod`, D-02 LE doc; `serialize_tree<T>` bound `Copy` → `Copy + bytemuck::Pod` (bound propagation). `scalar_le`/`bool_bytes`/enum `as u8` emits and the deserialize read path left verbatim.

## Decisions Made

- **`cast_slice` over the EMIT direction only (D-01 boundary):** the recast is `&[T] → &[u8]` on owned aligned Pod columns (always divisible/aligned). The untrusted deserialize `Reader::array` keeps its element-wise bounds-checked decode — a bulk `cast_slice` there would panic on misaligned/short hostile input instead of returning `SerializeError` (T-09-D / RESEARCH Pitfall 4). Grep confirms 0 `cast_slice` added to `binary.rs`.
- **`serialize_tree<T>` Pod bound (Rule 3 blocking, anticipated by the plan via the call-site inventory):** narrowing `le_bytes_of` to `T: bytemuck::Pod` made the 3 generic-`T` call sites (`leaf_value`/`threshold`/`leaf_vector`, all `TreeBuf<T>`) fail E0277 until `serialize_tree<T>` carried the Pod bound. Tightening it is mechanical: `f32`/`f64` are the only instantiations and both are `Pod`, so the `serialize_trees` match arms and all external callers compile verbatim. Not architectural — no API/wire change.
- **enum/bool columns NOT recast (T-09-10):** `node_type`/`cmp` (explicit `as i8 as u8` maps) and `default_left`/`*_present`/`category_list_right_child` (`bool_bytes`) define their own wire byte; they are outside D-01's "where layout allows" boundary and stay untouched. `bool_bytes` keeps its `from_raw_parts` (bool→u8, valid 0/1 bit patterns) — out of MEM-01's `&[T]→&[u8]` numeric scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `serialize_tree<T>` bound tightened `Copy` → `Copy + bytemuck::Pod`**
- **Found during:** Task 1 (compiling treelite-core after the `le_bytes_of` bound narrowing)
- **Issue:** narrowing `le_bytes_of` to `T: bytemuck::Pod` left the 3 generic-`T` call sites inside `serialize_tree<T>` (`leaf_value`/`threshold`/`leaf_vector`, each `TreeBuf<T>`) failing `E0277: the trait bound T: Pod is not satisfied` — `serialize_tree`'s own bound was still `T: Copy`.
- **Fix:** tightened `serialize_tree<T: Copy, B>` → `serialize_tree<T: Copy + bytemuck::Pod, B>`. The plan's call-site inventory only listed the header (`m.num_class` etc.) sites; the per-tree value columns are the same `le_bytes_of` seam and need the bound too. `f32`/`f64` are the only instantiations (both `Pod`), so `serialize_trees`' F32/F64 arms and all callers compile unchanged — same emitted bytes (golden byte-identical confirms).
- **Files modified:** `crates/treelite-core/src/serialize/mod.rs`
- **Commit:** `2e1e288`

## Issues Encountered

None blocking. One tooling note (carried from 09-02, pre-existing environment state): `uv run maturin develop` warns it cannot set rpath (`patchelf` not installed) but still builds + installs the abi3 wheel successfully and pytest passes — not introduced by this plan.

## Known Stubs

None. This plan is a complete one-function recast; every emit call site is wired through the migrated `le_bytes_of`. No placeholder/empty-value/TODO surface introduced.

## Threat Flags

None. No new network/auth/file-access surface. The recast is restricted to the trusted in-memory `&[T] → &[u8]` EMIT direction over owned aligned Pod columns (always divisible/aligned, T-09-D mitigated); the untrusted v5 deserialize read path (`binary.rs Reader::array`) is unchanged — still the element-wise bounds-checked decode, with 0 `cast_slice` added (grep-confirmed). The LE-host assumption is documented and byte-identity is proven by the golden compare on BOTH fixtures (T-09-09); enum/bool wire bytes stay defined by their explicit maps (T-09-10).

## Next Phase Readiness

- MEM-01 closed: `le_bytes_of` routed through the tested `bytemuck::cast_slice` seam (Pod bound), one `unsafe` block removed, both hard invariants green.
- Plan 04 (MEM-03, the final Phase-9 plan) can now run the allocator RSS report + record the `size_of::<Model>()` before/after row; MEM-01/02 leave the serializer + Model in their final v1 memory-hardened shape (byte-identical wire, no struct-size cost).
- All three remaining MEM-01/02 invariants (golden byte-identical both fixtures, 1e-5, workspace, pytest) are green — no deferred items, no blockers from this plan.

## Self-Check: PASSED

- `crates/treelite-core/src/serialize/mod.rs` exists; `le_bytes_of` body is `bytemuck::cast_slice(slice)` with bound `T: bytemuck::Pod`, no `unsafe`/`from_raw_parts` in the fn (grep: `cast_slice` x3 — 1 body + 2 doc; `from_raw_parts` only in `bool_bytes` + doc).
- Commit `2e1e288` present in git history.
- golden_v5 (both fixtures byte-identical), full harness (1e-5), workspace (0 failures), model_invariants (size 248 ≤ 512, !Send), pytest (39/1) all green; binary.rs read path has 0 `cast_slice`.

---
*Phase: 09-memory-efficiency-hardening*
*Completed: 2026-06-11*

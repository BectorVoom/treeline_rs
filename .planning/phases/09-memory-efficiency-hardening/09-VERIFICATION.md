---
phase: 09-memory-efficiency-hardening
verified: 2026-06-11T05:00:00Z
status: passed
score: 15/15 must-haves verified
overrides_applied: 0
gaps: []
deferred: []
---

# Phase 9: Memory-Efficiency Hardening Verification Report

**Phase Goal:** Apply the memory-efficiency techniques across the proven, equivalence-tested workspace without regressing the 1e-5 contract — closing out the last v1 requirements (MEM-01/02/03).
**Verified:** 2026-06-11T05:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SoA columns use `bytemuck` `Pod` zero-copy recasting where layout allows, with the full equivalence harness still green within 1e-5 | VERIFIED | `le_bytes_of` body is `bytemuck::cast_slice(slice)` with bound `T: bytemuck::Pod` (serialize/mod.rs:87-88); golden_v5 both fixtures pass; `cargo test --workspace` 0 failures |
| 2 | `smallvec` and `compact_str` back small collections and metadata strings, verified by existing tests still passing | VERIFIED | model.rs has 16 SmallVec + 7 CompactString occurrences; builder/lib.rs has 11 SmallVec/CompactString occurrences; all loader ripple sites carry `.into()`; full workspace tests pass |
| 3 | A custom global allocator (jemalloc) is wired into benchmarks/binaries and validated to import/run on Linux, not enabled in a way that breaks the abi3 wheel | VERIFIED | `memory_report` bin builds under `--features jemalloc` (exit 0); `cargo tree -p treelite-py grep jemalloc/mimalloc` prints nothing; MEMORY_REPORT.md records jemalloc RSS=5.43 MiB on Linux |
| 4 | v5 serializer still emits byte-identical golden_v5.bin / golden_v5_3format.bin | VERIFIED | `cargo test -p treelite-harness --test golden_v5` exits 0: both `serializer_reproduces_golden_v5_byte_for_byte` and `loader_path_reproduces_golden_v5_byte_for_byte` pass |
| 5 | Full equivalence harness green within 1e-5; `cargo test --workspace` + pytest pass | VERIFIED | `cargo test --workspace` all result lines show `ok. N passed; 0 failed`; `uv run pytest crates/treelite-py` shows `39 passed, 1 skipped` |
| 6 | harness has non-default, mutually-exclusive jemalloc/mimalloc features that build | VERIFIED | Builds with default, `--features jemalloc`, `--features mimalloc` all exit 0; `--features jemalloc,mimalloc` emits `compile_error!(...)` (exit 101) |
| 7 | abi3 wheel (treelite-py) has ZERO allocator deps in its tree | VERIFIED | `cargo tree -p treelite-py grep -E "jemalloc|mimalloc"` prints nothing |
| 8 | model_invariants test asserts size_of::<Model>() bound and documents !Send | VERIFIED | `cargo test -p treelite-core --test model_invariants` exits 0; test `model_size_not_bloated_by_smallvec` passes with size=248 <= 512; `_assert_not_send` documented in file |
| 9 | le_bytes_of uses bytemuck::cast_slice (the tested tree_buf.rs seam) instead of hand-rolled from_raw_parts | VERIFIED | `grep cast_slice serialize/mod.rs` >= 3; `le_bytes_of` body has no `unsafe` or `from_raw_parts`; the only remaining `from_raw_parts` is in `bool_bytes` (intentional, bool columns define their own wire byte) |
| 10 | The LE-host-only assumption (D-02) is documented at the recast site | VERIFIED | serialize/mod.rs lines 72-89 carry a full doc comment naming D-02 explicitly |
| 11 | The deserialize read path is NOT bulk-recast (stays element-wise bounds-checked) | VERIFIED | `grep cast_slice binary.rs` returns nothing; Reader::array keeps element-wise decode |
| 12 | enum/bool tree columns emitted via explicit as-u8/bool_bytes maps are NOT recast | VERIFIED | `bool_bytes` uses `from_raw_parts` for bool->u8 (not `cast_slice`); enum `as u8` maps are explicit; these lines were not touched by MEM-01 |
| 13 | jemalloc and mimalloc each build + run on Linux as #[global_allocator] in the memory_report bin | VERIFIED | Both `--features jemalloc` and `--features mimalloc` builds exit 0; MEMORY_REPORT.md shows jemalloc RSS=5.43 MiB and mimalloc RSS=10.40 MiB captured on linux x86_64 |
| 14 | docs/MEMORY_REPORT.md is committed with Observational banner + provenance + all 3 allocator blocks | VERIFIED | File exists; carries "Observational — NOT a CI gate" banner; contains rows for jemalloc, mimalloc, and system; includes size_of::<Model>()=248 row and D-11 attestation |
| 15 | Model metadata fields are SmallVec/CompactString-backed and deref byte-identically through the serializer | VERIFIED | model.rs has SmallVec<[i32;1]>/SmallVec<[i32;2]>/SmallVec<[f64;1]>/CompactString fields; golden v5 byte-compare passes confirming deref-transparency |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | 5 workspace dependency pins (smallvec 1.15.1, compact_str 0.9.1, tikv-jemallocator 0.7.0, tikv-jemalloc-ctl 0.7.0, mimalloc 0.1.52) | VERIFIED | All 5 present; smallvec on 1.x line (not 2.0); jemalloc/jemalloc-ctl carry `stats` feature; workspace metadata resolves |
| `crates/treelite-harness/Cargo.toml` | jemalloc/mimalloc non-default mutually-exclusive features + optional deps | VERIFIED | Lines 29-30: `jemalloc`/`mimalloc` features; lines 48-50: `optional = true` on all 3 allocator deps |
| `crates/treelite-harness/src/bin/memory_report.rs` | Allocator-gated bin skeleton with #[global_allocator] cfg arms + compile_error guard | VERIFIED | `global_allocator` count=4; 3 cfg-gated arms present; `compile_error!` guard present; fleshed-out main() loads benchmark set, drives prediction, samples RSS, merges report |
| `crates/treelite-core/tests/model_invariants.rs` | size_of::<Model>() budget test + documented !Send invariant | VERIFIED | `size_of::<Model>` count=5; `_assert_not_send` fn with commented `requires_send::<Model>()` present; test passes |
| `crates/treelite-core/src/model.rs` | 7 metadata fields as SmallVec<[i32;N]>/SmallVec<[f64;N]>/CompactString | VERIFIED | 16 SmallVec + 7 CompactString occurrences; explicit field types verified in lines 72-95 |
| `crates/treelite-builder/src/lib.rs` | Metadata struct fields migrated in lockstep with Model | VERIFIED | 11 SmallVec/CompactString occurrences |
| `crates/treelite-core/src/serialize/mod.rs` | le_bytes_of<T: bytemuck::Pod> via bytemuck::cast_slice with LE doc | VERIFIED | `cast_slice` count=3; signature has `T: bytemuck::Pod`; doc comment documents D-02 LE assumption |
| `crates/treelite-harness/src/memory.rs` | RSS/allocated sampler + markdown report writer | VERIFIED | `sample_rss` with jemalloc epoch-advance and statm paths; `render_markdown`/`emit`; Observational banner; Manifest provenance |
| `docs/MEMORY_REPORT.md` | Committed observational before/after memory report | VERIFIED | Exists with "Observational" banner; 3 allocator blocks (system/jemalloc/mimalloc); size_of::<Model>() attestation |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/treelite-harness/Cargo.toml` | tikv-jemallocator, tikv-jemalloc-ctl, mimalloc | optional workspace deps gated behind features | WIRED | Lines 48-50 confirm `optional = true`; features lines 29-30 reference `dep:tikv-jemallocator` etc. |
| `Cargo.toml` | smallvec, compact_str, tikv-jemallocator, tikv-jemalloc-ctl, mimalloc | [workspace.dependencies] pins | WIRED | All 5 entries present in workspace deps table; `smallvec` pattern found |
| `crates/treelite-core/src/serialize/mod.rs le_bytes_of` | `crates/treelite-core/src/tree_buf.rs as_bytes` | same `bytemuck::cast_slice` call form; bound Copy → Pod | WIRED | `le_bytes_of` body is `bytemuck::cast_slice(slice)` matching tree_buf.rs:102 form |
| `le_bytes_of call sites (mod.rs serialize_header + serialize_tree)` | i32/f64 Pod columns | unchanged call sites; bound tightened to Pod | WIRED | serialize_header lines 110-121 call `le_bytes_of(&m.num_class)` etc.; serialize_tree calls le_bytes_of on tree column slices |
| `crates/treelite-harness/src/bin/memory_report.rs` | tikv_jemalloc_ctl epoch/stats + /proc/self/statm | epoch.advance() then stats reads (jemalloc); statm field 2 (system/mimalloc) | WIRED | memory.rs sample_rss() has both paths; jemalloc path calls epoch advance before stats reads (Pitfall 5) |
| `crates/treelite-harness/src/memory.rs` | `crates/treelite-harness/src/manifest.rs Manifest` | report header provenance (os/arch/rustc) | WIRED | `use crate::manifest::Manifest;` on line 39; render_markdown signature takes `manifest: &Manifest` |

---

### Data-Flow Trace (Level 4)

Not applicable for non-rendering crate files. All artifacts are utilities or binaries, not dynamic-rendering components. The key data flow is: `memory_report` bin loads real models → drives prediction → samples RSS → writes MEMORY_REPORT.md. Verified by the committed docs/MEMORY_REPORT.md containing non-zero RSS values (5.43 MiB jemalloc, 10.40 MiB mimalloc, 9.71 MiB system).

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| memory_report bin builds (default) | `cargo build -p treelite-harness --bin memory_report` | exit 0, `Finished dev profile` | PASS |
| memory_report bin builds with jemalloc | `cargo build -p treelite-harness --bin memory_report --features jemalloc` | exit 0 | PASS |
| memory_report bin builds with mimalloc | `cargo build -p treelite-harness --bin memory_report --features mimalloc` | exit 0 | PASS |
| both-on is a compile_error | `cargo build ... --features jemalloc,mimalloc` | exit non-zero; `error: features jemalloc and mimalloc are mutually exclusive (D-07)` | PASS |
| model_invariants passes | `cargo test -p treelite-core --test model_invariants` | `1 passed; 0 failed` | PASS |
| golden v5 byte-identical | `cargo test -p treelite-harness --test golden_v5` | `2 passed; 0 failed` | PASS |
| workspace suite green | `cargo test --workspace` | All result lines `ok. N passed; 0 failed` | PASS |
| pytest 1e-5 | `uv run pytest crates/treelite-py -q` | `39 passed, 1 skipped` | PASS |
| wheel has no allocator deps | `cargo tree -p treelite-py grep jemalloc/mimalloc` | empty output | PASS |

---

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes declared for this phase.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MEM-01 | 09-03 | SoA columns use `bytemuck` Pod zero-copy recasting where layout allows | SATISFIED | `le_bytes_of` uses `bytemuck::cast_slice` with Pod bound; no unsafe in fn; golden byte-identical |
| MEM-02 | 09-02 | `smallvec` and `compact_str` used for small collections and metadata strings | SATISFIED | 7 Model fields + builder::Metadata fields migrated; size_of stays 248 B; tests green |
| MEM-03 | 09-04 | A custom global allocator (jemalloc) wired into benchmarks/binaries | SATISFIED | `#[global_allocator]` in memory_report bin only; both allocators build+run on Linux; abi3 wheel clean; MEMORY_REPORT.md committed |

All 3 phase requirements satisfied. No orphaned requirements (REQUIREMENTS.md maps MEM-01/02/03 to Phase 9).

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-core/src/serialize/mod.rs` | 248 | `unsafe { from_raw_parts }` in `bool_bytes` | INFO | Intentional — `bool_bytes` handles `bool -> u8` recast, which `bytemuck::cast_slice` cannot do because `bool` does not implement `Pod`. The SAFETY comment is present; this is explicitly excluded from MEM-01 scope ("enum/bool columns define their own wire byte"). Not a regression. |

No TBD/FIXME/XXX markers in any phase-modified file. No stub patterns found in production code. The `bool_bytes unsafe` is a pre-existing intentional pattern (per plan design), not introduced by this phase.

---

### Human Verification Required

None. All phase deliverables are amenable to automated verification. The MEMORY_REPORT.md is an observational artifact (explicitly NOT a CI gate per D-10); its existence and content have been verified programmatically.

---

## Gaps Summary

No gaps. All 15 must-have truths verified, all 9 required artifacts pass all 4 verification levels (exists, substantive, wired, data-flowing), all 3 requirements (MEM-01/02/03) satisfied, all spot-checks pass.

The phase goal is achieved: memory-efficiency techniques applied without regressing the 1e-5 contract. The hard invariants hold:
- HARD INVARIANT 1: v5 serializer emits byte-identical golden_v5.bin / golden_v5_3format.bin (both golden_v5 tests pass).
- HARD INVARIANT 2: cargo test --workspace (0 failures) + uv run pytest (39 passed/1 skipped) green within 1e-5.

---

_Verified: 2026-06-11T05:00:00Z_
_Verifier: Claude (gsd-verifier)_

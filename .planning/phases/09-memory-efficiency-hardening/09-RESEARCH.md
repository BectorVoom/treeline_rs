# Phase 9: Memory-Efficiency Hardening - Research

**Researched:** 2026-06-11
**Domain:** Rust memory-efficiency hardening (zero-copy byte recast, small-collection/string optimization, custom global allocators) over a numerically-frozen tree-ensemble workspace
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**MEM-01 — bytemuck Pod zero-copy recast**
- **D-01:** Push Pod recasting **everywhere feasible** — extend the existing `TreeBuf<T: Copy + bytemuck::Pod>::as_bytes()` seam (`tree_buf.rs:96`, today confined to device upload) across core SoA columns, the v5 serializer, and loaders wherever the buffer layout permits. Maximum zero-copy reach.
- **D-02:** **Endianness: little-endian-only zero-copy; big-endian hosts are out of scope for v1.** `bytemuck::cast_slice` yields native-endian bytes; dev box + CI are x86/ROCm (LE) and the frozen golden was captured LE, so native-endian casts on the serialize path are acceptable. **No portable BE byte-swap fallback** is to be built. The LE assumption MUST be documented at every recast site that touches serialized output.
- **D-03 (invariant):** Despite the aggressive reach, the v5 serializer MUST still emit **byte-identical** output to `fixtures/golden_v5.bin` / `fixtures/golden_v5_3format.bin`. Any recast that would change emitted bytes is not "where layout allows" — the golden compare is the gate.

**MEM-02 — smallvec / compact_str**
- **D-04:** **Change the public `Model` field types directly** — the metadata `Vec<i32>` fields (`num_class`, `leaf_vector_shape`, `target_id`, `class_id`) and `base_scores` (`Vec<f64>`) become `SmallVec`-backed; `postprocessor` and `attributes` (`String`) become `CompactString`-backed. The ripple through serializer, builder, loaders, and GTIL is accepted and must be updated in lockstep.
- **D-05:** **Storage-only at the boundaries.** The change is internal storage even though the Rust field types change: the PyO3 accessors (Phase 8) MUST still return `list`/`str` to Python, and the v5 serializer MUST still emit byte-identical bytes (SmallVec/CompactString deref to the same payload). The Phase-8 binding contract and the frozen golden are both preserved. **Planner must verify both explicitly** (binding tests + golden byte-compare).
- **D-06 (research detail):** Inline sizing for the SmallVecs and the compact_str benefit on the potentially-large `attributes` JSON blob are left to the researcher/planner to profile (CompactString is harmless on large strings — heap-equivalent above its inline threshold — so applying it uniformly is safe even where it gives no win).

**MEM-03 — custom global allocator**
- **D-07:** Wire **both jemalloc and mimalloc**, runtime-selectable via **mutually-exclusive Cargo features**, **neither default**. Benchmarks can then compare the two.
- **D-08:** Allocator is installed as `#[global_allocator]` **only in harness binaries + the new benchmark targets**. Library crates and the **abi3 cpu wheel keep the system default allocator** — the allocator must never be reachable from the wheel build. Matches SC3 verbatim.
- **D-09:** Both allocators must be validated to **build + import/run on Linux** (the AMD/ROCm dev box).

**Validation bar**
- **D-10:** **Committed observational memory report** — a new benchmark target measures peak RSS / allocation (leveraging jemalloc stats) and writes a committed `MEMORY_REPORT.md` showing before/after, following the Phase-7 `GPU_EQUIVALENCE_REPORT.md` observational precedent. **No brittle hard CI threshold gate.**
- **D-11 (implicit floor):** Independent of the report, the phase is not done until the **full equivalence harness stays green within 1e-5** AND **all existing workspace + Python tests pass** after the type/allocator changes. This is the real pass/fail gate; the report is evidence the hardening worked.

### Claude's Discretion
- SmallVec inline capacities, the exact jemalloc/mimalloc crate selection, the benchmark model set, and the report's exact columns — all delegated to research/planning (see D-06, D-07, D-10).

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope. Big-endian portability is the one explicitly out-of-scope item, recorded as decision D-02 (it bounds MEM-01 directly), **not** a deferred idea.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MEM-01 | SoA columns use `bytemuck` `Pod` zero-copy recasting where layout allows | §MEM-01 Recast Site Inventory below distinguishes "safe everywhere (in-memory/device)" from "serializer-path, golden-gated"; the serializer's existing hand-rolled `le_bytes_of` unsafe transmute is the primary replace target — byte-identical by construction. |
| MEM-02 | `smallvec` and `compact_str` are used for small collections and metadata strings | §MEM-02 Field-Type Migration: inline-capacity recommendations from real model shapes, exact 6-field ripple list (serializer/builder/loaders/gtil/concat), and the two preservation gates (golden byte-compare + Python list/str). |
| MEM-03 | A custom global allocator (jemalloc) is wired into benchmarks/binaries | §MEM-03 Allocator Wiring: `tikv-jemallocator 0.7` + `mimalloc 0.1` mutually-exclusive non-default features, `#[global_allocator]` only in new harness bins/benches, `tikv-jemalloc-ctl` for RSS/allocated stats, abi3-wheel isolation discipline. |
</phase_requirements>

## Summary

This phase is a **carefully bounded hardening sweep over an already-frozen numerical contract.** The two invariants (byte-identical `golden_v5.bin`/`golden_v5_3format.bin` and the 1e-5 equivalence harness) are non-negotiable gates, and the good news from the code audit is that **the existing architecture was deliberately built to absorb all three changes with near-zero fidelity risk:**

1. **MEM-01** mostly *replaces hand-rolled `unsafe`* rather than adding new behavior. The serializer already reinterprets columns as bytes via a private `le_bytes_of<T: Copy>` helper (`serialize/mod.rs:77-81`) that does an `unsafe { std::slice::from_raw_parts(... as *const u8, ...) }` transmute. Swapping that for `bytemuck::cast_slice` is byte-identical by construction (both emit the native-LE image), removes `unsafe`, and adds bytemuck's alignment/size validation. The `TreeBuf::as_bytes()` Pod seam (`tree_buf.rs:96-105`) already exists and is tested — it is widened, not introduced.

2. **MEM-02** changes six public `Model` field types, but the serializer's emit path is **already generic over `&[T]`/`&str`** (`le_bytes_of<T: Copy>(slice: &[T])` and `b.string(&str)`), so `SmallVec` and `CompactString` deref straight through with **no serializer change and identical bytes**. The Python binding currently exposes **no getter at all** for these metadata fields (verified — `treelite-py/src/model.rs` only exposes `num_tree`/`num_feature`/`input_type`/`output_type`/serialize/dump), so the D-05 "keep list/str" concern is mostly forward-looking; the lockstep ripple is into the `builder::Metadata` struct, `bulk.rs`, `concat.rs`, and the loaders.

3. **MEM-03** is purely additive: new feature-gated `#[global_allocator]` statics in **new** harness bins/benches. The project already has the exact discipline pattern from Phase 7/8 (the abi3 wheel never pulls cubecl via `optional = true` + non-default features). The allocator follows the same shape, one tier stricter (it must not appear in *any* library crate's dependency graph reachable from the wheel).

**Primary recommendation:** Sequence the work MEM-02 → MEM-01 → MEM-03. Do MEM-02 first (it touches the most files and its lockstep ripple is the riskiest compile surface); land the golden byte-compare and `cargo test --workspace` green; then MEM-01 (largely mechanical `unsafe`→`bytemuck` swaps gated by the same golden); then MEM-03 (fully isolated new targets). Every step ends green against the two invariants.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Zero-copy column→bytes recast (MEM-01) | Core (`treelite-core`: `tree_buf.rs`, `serialize/mod.rs`) | Loaders (legacy-binary LE decode), cubecl upload | The SoA columns and the serializer emit path live in core; loaders only *produce* columns, GPU upload *consumes* `as_bytes()`. |
| Small-collection / string storage (MEM-02) | Core (`model.rs` field types) | Builder (`Metadata`, `bulk`, `concat`), loaders (write), GTIL (read), py (read) | The field types are owned by `Model`; every other crate touches them through assignment/read and must update in lockstep but owns no policy. |
| Global allocator selection (MEM-03) | Harness bins + new bench targets only | — (explicitly NOT any library crate, NOT the wheel) | `#[global_allocator]` is a binary-level decision; placing it anywhere in a lib crate's graph would leak into the wheel. D-08. |
| Memory measurement / report (MEM-03/D-10) | New harness bench target + `report`-style writer | jemalloc stats (`tikv-jemalloc-ctl`) | Mirrors the Phase-7 `report.rs` → `docs/GPU_EQUIVALENCE_REPORT.md` observational precedent. |
| 1e-5 + byte-identical gates (D-03/D-05/D-11) | `treelite-harness` tests (`golden_v5.rs`, `equivalence.rs`, matrix tests) + py pytest | — | These existing tests are the pass/fail oracle; the phase adds no new tolerance logic, it must keep them green. |

## Project Constraints (from CLAUDE.md)

- **Edition 2024**, Cargo workspace, resolver "3"; single pinned `[workspace.dependencies]` table — new crates go there.
- **All crates pinned to their latest published versions**, **no pre-release on the critical path** (FND-02). → `smallvec 2.0.0-alpha` is FORBIDDEN; use the `1.x` line.
- **Error handling:** `thiserror` in library crates, `anyhow` in binaries/tests. New bench/bin targets use `anyhow`.
- **`Model` is `!Send`** (TreeBuf::Borrowed raw pointers) — MUST stay so. `SmallVec`/`CompactString` are `Send`-neutral (they don't add `Send`; the raw-pointer `Borrowed` variant keeps `Model` `!Send` regardless). Verify the auto-trait is unchanged.
- **Predictions match upstream within 1e-5** — the core value. No recommendation may alter prediction output.
- **Serialization: current (v5) format generation only.** No format work.
- **`rustfmt` + `clippy` default settings** — run `cargo fmt` / `cargo clippy` after the type changes (clippy may suggest `.into()` simplifications at the new conversion sites).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `bytemuck` | `1` (already pinned, `features=["derive"]`) | Zero-copy `cast_slice` recast for MEM-01 | Already a `[workspace.dependencies]` entry and used by `TreeBuf::as_bytes` + `treelite-cubecl`. No new dep. `[CITED: tree_buf.rs:96, Cargo.toml]` |
| `smallvec` | `1.15.1` (latest stable `1.x`; **NOT** `2.0.0-alpha.x`) | SVO backing for the 5 small metadata Vec fields (MEM-02) | De-facto Rust SVO crate; `1.x` is the stable line, derefs to `&[T]` so serializer is untouched. `[VERIFIED: ctx7 /websites/rs_smallvec_smallvec — SmallVec<[T;N]> array-syntax API, serde feature, const_new]` + `[VERIFIED: slopcheck OK on crates.io]` |
| `compact_str` | `0.9.1` (latest stable; manual cites `0.8` — registry has `0.9.1`) | SSO backing for `postprocessor` + `attributes` (MEM-02) | 24-byte inline SSO, `Deref<Target=str>` so serializer's `b.string(&str)` is untouched; heap-equivalent above threshold (D-06 — harmless on large `attributes`). `[VERIFIED: cargo search — 0.9.1]` + `[VERIFIED: slopcheck OK]` |
| `tikv-jemallocator` | `0.7.0` | jemalloc `#[global_allocator]` for benches/bins (MEM-03) | The **maintained** jemalloc binding (TiKV fork); plain `jemallocator 0.5.4` is effectively unmaintained. Linux-native, builds on the ROCm box. `[VERIFIED: cargo search — 0.7.0]` + `[VERIFIED: slopcheck OK]` |
| `mimalloc` | `0.1.52` | mimalloc `#[global_allocator]` alternative (MEM-03, D-07) | The canonical Rust mimalloc crate (Microsoft allocator). `[VERIFIED: cargo search — 0.1.52]` + `[VERIFIED: slopcheck OK]` + `[CITED: MIMALLOC_MANUAL.md]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tikv-jemalloc-ctl` | `0.7.0` | Read jemalloc `stats.allocated` / `stats.resident` (RSS) for `MEMORY_REPORT.md` (D-10) | Only in the jemalloc-feature bench target; pairs version-locked with `tikv-jemallocator 0.7`. `[VERIFIED: cargo search — 0.7.0]` + `[VERIFIED: slopcheck OK]` |
| `criterion` | `0.8.2` | (Optional) statistical micro-benchmark harness | Only if the planner wants timing alongside RSS. A custom RSS sampler (below) is simpler and sufficient for the observational report. `[VERIFIED: cargo search — 0.8.2]` + `[VERIFIED: slopcheck OK]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `tikv-jemallocator 0.7` | `jemallocator 0.5.4` | The manual's `jemallocator 0.5` is the **unmaintained original**; tikv fork is current and the community-standard. Prefer tikv. |
| `smallvec 1.15` | `smallvec 2.0.0-alpha.12` | 2.0 uses the new `SmallVec<T, N>` const-generic API and is **pre-release** — forbidden by FND-02. Stay on `1.x` (`SmallVec<[T; N]>`). |
| `compact_str` | `smol_str` / `ecow` / `Box<str>` | `compact_str` was named in CONTEXT (D-04) and has the cleanest `Deref<Target=str>` drop-in. Don't substitute. |
| custom RSS sampler | `criterion` + `criterion-perf-events` | Criterion measures *time*, not *peak RSS*; for an observational memory report the jemalloc-ctl `epoch`+`resident` read (or `/proc/self/statm`) is more directly on-target. Criterion is optional, not required. |
| `mimalloc 0.1` (default v2/v3) | `mimalloc` with `secure` feature | `secure` adds guard pages + overhead; off-target for a perf report. Use default features. |

**Installation (add to `[workspace.dependencies]`):**
```toml
smallvec = { version = "1.15.1", features = ["serde", "const_new", "union"] }
compact_str = { version = "0.9.1", features = ["serde"] }
# allocator crates — referenced ONLY by harness bin/bench targets behind features:
tikv-jemallocator = "0.7.0"
tikv-jemalloc-ctl = "0.7.0"
mimalloc = "0.1.52"
# bytemuck already present.
```
> Notes: `serde` features are optional — only add them if the migrated fields cross a serde boundary (they currently do **not**; the v5 serializer is hand-framed, and `serde_json` is only used for JSON *dump*, which reads via getters, not serde-derive on `Model`). Recommend adding `union` for `smallvec` (smaller footprint via untagged storage) and `const_new` only if `const` construction is wanted; otherwise drop them to minimize features. **Verify each feature exists for the pinned version at plan time** (`cargo add --dry-run`).

**Version verification performed:** `cargo search` (crates.io) confirmed each version above on the **crates.io (Rust) ecosystem** — the correct registry. `slopcheck install` ran `cargo add` against crates.io and returned `6 OK`. `smallvec`'s `1.x` array-syntax API was cross-checked via ctx7 (`/websites/rs_smallvec_smallvec`).

## Package Legitimacy Audit

> All packages verified via `slopcheck install` (ran `cargo add` against **crates.io**, the correct ecosystem) — all clean. `bytemuck` is pre-existing.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `smallvec` (1.15.1) | crates.io | ~10 yrs | very high (servo) | github.com/servo/rust-smallvec | [OK] | Approved |
| `compact_str` (0.9.1) | crates.io | ~5 yrs | high | github.com/ParkMyCar/compact_str | [OK] | Approved |
| `tikv-jemallocator` (0.7.0) | crates.io | ~5 yrs | high (TiKV) | github.com/tikv/jemallocator | [OK] | Approved |
| `tikv-jemalloc-ctl` (0.7.0) | crates.io | ~5 yrs | high | github.com/tikv/jemallocator | [OK] | Approved |
| `mimalloc` (0.1.52) | crates.io | ~6 yrs | high | github.com/purpleprotocol/mimalloc_rust | [OK] | Approved |
| `criterion` (0.8.2) | crates.io | ~8 yrs | very high | github.com/bheisler/criterion.rs | [OK] | Approved (optional) |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

> Note on transitive native builds: `tikv-jemallocator`/`tikv-jemalloc-sys` and `mimalloc`/`libmimalloc-sys` compile a vendored C allocator via a `build.rs` (a `cc` build, not a network postinstall). This is expected and benign on the Linux dev box (D-09 requires validating exactly this build). A working C toolchain (`cc`) must be present — it already is (the upstream C++ tree builds locally).

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────────┐
   external model files  │              LOADERS (write fields)          │
   (xgb/lgbm/sklearn) ──▶│  xgboost / lightgbm / sklearn / builder      │
                         │  ┌────────────────────────────────────────┐  │
                         │  │ Metadata{ num_class, target_id, ...,    │  │  MEM-02
                         │  │           base_scores, postprocessor }  │  │  type change
                         │  └───────────────────┬────────────────────┘  │  ripples here
                         └──────────────────────┼───────────────────────┘
                                                ▼ assign (lockstep .into())
                  ┌───────────────────────────────────────────────────────┐
                  │  treelite-core :: Model  (header metadata fields)       │
                  │   num_class/leaf_vector_shape/target_id/class_id        │ MEM-02
                  │     : Vec<i32>  →  SmallVec<[i32; N]>                    │ (field types)
                  │   base_scores  : Vec<f64> → SmallVec<[f64; N]>          │
                  │   postprocessor/attributes : String → CompactString     │
                  │                                                         │
                  │  Tree<T> :: SoA columns  (TreeBuf<T>)  ◀── MEM-01 ───┐  │
                  └──────────┬───────────────────────────┬──────────────┼──┘
                             │ deref &[T]/&str            │ as_slice()   │ as_bytes()
                             ▼ (transparent)              ▼              ▼ (Pod cast_slice)
        ┌────────────────────────────┐   ┌──────────────────┐   ┌────────────────────┐
        │  v5 serializer (mod.rs)    │   │   GTIL (read)    │   │  cubecl SoA upload │
        │  le_bytes_of<T>  ──MEM-01──│   │  shape/predict   │   │  (already Pod)     │
        │   unsafe transmute →       │   └──────────────────┘   └────────────────────┘
        │   bytemuck::cast_slice     │            │                       │
        │  MUST stay byte-identical  │            ▼                       ▼
        │  to golden_v5.bin (D-03)   │     1e-5 harness gate        (device memory)
        └────────────┬───────────────┘     (D-11)
                     ▼
            fixtures/golden_v5.bin  ◀── byte-compare GATE (golden_v5.rs)

   ── MEM-03 (fully isolated, additive) ──────────────────────────────────────────
        crates/treelite-harness/{benches,src/bin}/  (NEW targets)
           #[global_allocator] static A = Jemalloc | MiMalloc   (feature-gated)
           ─ tikv-jemalloc-ctl reads stats.{allocated,resident} ─▶ docs/MEMORY_REPORT.md
        ⚠ NEVER reachable from crates/treelite-py (abi3 wheel keeps system malloc)
```

### Recommended Project Structure (additions only)
```
crates/treelite-harness/
├── Cargo.toml              # + [features] jemalloc/mimalloc (mutually exclusive, neither default)
│                           # + optional deps tikv-jemallocator / mimalloc / tikv-jemalloc-ctl
├── benches/                # NEW (or src/bin/) — the only place #[global_allocator] lives
│   └── memory_report.rs    # RSS/alloc sampler → docs/MEMORY_REPORT.md (D-10)
└── src/
    └── memory.rs           # (optional) shared sampler helper + report writer (mirrors report.rs)
docs/
└── MEMORY_REPORT.md        # NEW committed observational artifact (Phase-7 precedent)
```

### Pattern 1: Serializer recast — replace hand-rolled `unsafe` with `bytemuck::cast_slice` (MEM-01)
**What:** The serializer already transmutes columns to bytes manually; swap to bytemuck for the same bytes minus the `unsafe`.
**When to use:** Every `le_bytes_of` call site in `serialize/mod.rs` whose element `T: bytemuck::Pod` (all numeric columns: `i32`, `f64`, `u64`, `T∈{f32,f64}`).
```rust
// Source: current crates/treelite-core/src/serialize/mod.rs:77-81 (the REPLACE target)
fn le_bytes_of<T: Copy>(slice: &[T]) -> &[u8] {
    // SAFETY: T: Copy POD ... (hand-rolled transmute)
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}
// MEM-01 form — byte-identical, no unsafe, with size/align validation:
fn le_bytes_of<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
    bytemuck::cast_slice(slice)   // native-LE image == upstream memcpy image on the LE manifest host (D-02)
}
```
> `bytemuck::cast_slice` does **NOT** reorder, pad, or insert framing — it only rewrites the fat-pointer (ptr, len*size) metadata; the byte content is exactly the source memory image. On x86-64 (LE) this equals `to_le_bytes` concatenation, which is what `golden_v5.bin` was captured as. **Document the LE assumption at this site (D-02).** The bound tightens from `T: Copy` to `T: bytemuck::Pod`; all numeric column element types already satisfy Pod (proven by `TreeBuf::as_bytes`). Enum/bool columns (`node_type`, `cmp`, `default_left`, …) are **already** converted to `u8` via explicit `.map(|n| *n as i8 as u8)` / `bool_bytes(...)` loops *before* emission — they are **not** Pod-recast candidates and stay as-is (they are not `repr`-guaranteed Pod and the explicit map is what preserves the wire byte).

### Pattern 2: Transparent deref through SmallVec/CompactString (MEM-02)
**What:** Because the serializer emits `&[T]` (via `le_bytes_of(&m.num_class)`) and `&str` (via `b.string(&m.postprocessor)`), changing the field type to `SmallVec<[i32; N]>` / `CompactString` requires **no serializer edit** — both deref to the same slice/str.
**When to use:** All six migrated fields. The serializer is the single most fidelity-critical consumer and it needs zero changes.
```rust
// model.rs field-type change (the ONLY type edits):
pub num_class:         SmallVec<[i32; 1]>,   // [1] for binary clf
pub leaf_vector_shape: SmallVec<[i32; 2]>,   // [1,1] for binary clf
pub target_id:         SmallVec<[i32; 1]>,   // [0] single-target
pub class_id:          SmallVec<[i32; 1]>,   // [0] binary clf (grows with num_tree for multiclass)
pub base_scores:       SmallVec<[f64; 1]>,   // scalar base_score is the common case
pub postprocessor:     CompactString,        // names like "sigmoid","identity" ≤ 24B → inline
pub attributes:        CompactString,        // "{}" inline; large JSON spills to heap (harmless, D-06)
// serializer UNCHANGED — le_bytes_of(&m.num_class) and b.string(&m.postprocessor) both deref.
```

### Pattern 3: Allocator isolation via optional dep + non-default mutually-exclusive features (MEM-03)
**What:** The same isolation shape the workspace already uses for cubecl GPU backends, one tier stricter: the allocator crates are referenced **only** by the harness's *own* bin/bench targets, never by a `pub fn` in any library crate.
```toml
# crates/treelite-harness/Cargo.toml
[features]
default = []
jemalloc = ["dep:tikv-jemallocator", "dep:tikv-jemalloc-ctl"]
mimalloc = ["dep:mimalloc"]
[dependencies]
tikv-jemallocator = { workspace = true, optional = true }
tikv-jemalloc-ctl = { workspace = true, optional = true }
mimalloc          = { workspace = true, optional = true }
```
```rust
// benches/memory_report.rs (or src/bin/) — the ONLY #[global_allocator] site.
#[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Mutual exclusion guard (neither default; both-on is a config error):
#[cfg(all(feature = "jemalloc", feature = "mimalloc"))]
compile_error!("features `jemalloc` and `mimalloc` are mutually exclusive (D-07)");
```
> `#[global_allocator]` is only honored at the **final binary/bench crate root** (`main.rs`/the bench's root), never from a library. Placing the static in a `#[bench]`/`#[bin]` target guarantees the wheel (`crate-type=["cdylib"]`, no such static) is unaffected.

### Pattern 4: jemalloc RSS/allocation read for the report (MEM-03 / D-10)
```rust
// Source: tikv-jemalloc-ctl API (epoch must be advanced before each stats read).
use tikv_jemalloc_ctl::{epoch, stats};
fn sample_rss() -> (usize, usize) {
    epoch::advance().unwrap();                 // refresh cached stats
    let allocated = stats::allocated::read().unwrap();
    let resident  = stats::resident::read().unwrap();  // ≈ peak RSS proxy
    (allocated, resident)
}
```
> For the **mimalloc** and **system-malloc** rows (which have no jemalloc-ctl), fall back to a platform RSS read: parse `/proc/self/statm` (field 2 × page size = resident pages) on Linux — adequate for the observational report. The report should make the measurement method per-row explicit (jemalloc-ctl vs `/proc`) for honesty, mirroring the Phase-7 report's provenance discipline.

### Anti-Patterns to Avoid
- **Pod-recasting enum/bool columns.** `TreeNodeType`/`Operator`/`bool` columns are emitted via explicit `as u8` / `bool_bytes` maps that define the wire byte. `cast_slice` on a `#[repr(i8)]` enum is neither valid nor needed; leave those loops untouched.
- **Adding `#[global_allocator]` to any library crate (or to `treelite-py`).** Breaks D-08; the abi3 wheel must keep system malloc. The static belongs only in new bin/bench targets.
- **Bumping `smallvec` to `2.0.0-alpha`.** Pre-release → violates FND-02 and changes the type syntax (`SmallVec<T,N>`).
- **Over-large SmallVec inline N.** Inlining `N=16` for a field that's almost always length 1 bloats `Model`'s size and hurts cache locality (the opposite goal). Size N to the *common* case, not the max (see §MEM-02 sizing).
- **Touching `Model`'s `!Send` property.** Don't add `unsafe impl Send`; the change is storage-only.
- **Inserting a big-endian byte-swap path.** Explicitly out of scope (D-02). Document LE, build no swap.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| `&[T]` → `&[u8]` recast | `unsafe from_raw_parts(... as *const u8 ...)` (current `le_bytes_of`) | `bytemuck::cast_slice` | Adds size/alignment validation, removes `unsafe`, byte-identical on LE. This *is* MEM-01. |
| Small-vector inline storage | manual `enum { Inline([T;N]), Heap(Vec<T>) }` | `smallvec::SmallVec<[T;N]>` | Battle-tested spill logic, `Deref<[T]>`, no fidelity risk. |
| Small-string inline storage | manual SSO union | `compact_str::CompactString` | 24-byte SSO with correct UTF-8 discriminant; heap-equivalent above threshold. |
| jemalloc binding | raw FFI to `je_mallctl` | `tikv-jemallocator` + `tikv-jemalloc-ctl` | Maintained safe wrappers; vendored build. |
| RSS sampling | shelling out to `ps`/`top` | `tikv-jemalloc-ctl` stats + `/proc/self/statm` | In-process, deterministic, no subprocess. |

**Key insight:** Almost nothing in this phase is *new logic* — it is replacing hand-rolled primitives (unsafe transmute, plain `Vec`/`String`) with the project's standard memory-efficiency crates, plus one additive isolated allocator/report target. The fidelity surface is therefore tiny and fully covered by the two existing gates.

## Runtime State Inventory

> This is a code/type/dependency change, not a rename or data migration. No stored data, live-service config, OS-registered state, secret, or persisted build artifact embeds a string this phase changes. The only "wire" surface — the v5 byte format — is **deliberately held byte-identical** (D-03/D-05), so even serialized blobs on disk (`fixtures/*.bin`) remain valid and unchanged.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — the v5 wire bytes are held byte-identical by design (D-03); no DB/keystore involved. | none |
| Live service config | None — verified: no external service stores treelite metadata field types. | none |
| OS-registered state | None. | none |
| Secrets/env vars | `MALLOC_CONF` (jemalloc) is an *optional runtime* env var a user could set to tune the allocator/profiling; it is read by jemalloc itself, not by our code, and is not required. Note in report docs only. | none (documentation only) |
| Build artifacts | The `*-sys` allocator crates compile a vendored C library on first build (expected per D-09). A stale `target/` does not embed the old field types — a normal `cargo build` rebuilds. The maturin abi3 wheel must be rebuilt after MEM-02 (its py crate recompiles against the new core), but its *contents* (no allocator) are unchanged. | rebuild via normal `cargo build` / `maturin develop` |

**The canonical question — "after every file is updated, what runtime systems still have the old types cached?":** None. The on-disk goldens stay byte-valid (invariant), and there is no external/persisted representation of the Rust field types.

## Common Pitfalls

### Pitfall 1: Lockstep type ripple breaks compilation across 5 crates (MEM-02)
**What goes wrong:** Changing `Model.num_class: Vec<i32>` to `SmallVec<...>` breaks every `model.num_class = some_vec;` assignment and every `vec_field == other_vec_field` comparison until *all* of them are updated.
**Why it happens:** The exact ripple set is: `builder::Metadata` struct fields (`lib.rs:59-71`), `bulk.rs:229-235`, `lib.rs:768-774`, `concat.rs:79-150` (which also does `m.num_class != out.num_class` comparisons and builds `Vec<i32>` locals then assigns), and the loaders that populate `Metadata` (lightgbm/sklearn/xgboost). The serializer and GTIL **read** via deref and need no change.
**How to avoid:**
- Decide whether `builder::Metadata`'s fields *also* become SmallVec/CompactString (cleanest — no conversion at assignment) or stay `Vec`/`String` with a `.into()` at the `model.x = metadata.x.into()` boundary. **Recommendation:** change `Metadata` too (consistent, zero per-call conversions, matches D-04's "ripple accepted").
- `concat.rs` builds `let mut target_id: Vec<i32> = Vec::new();` then `out.target_id = target_id;` — change the local to `SmallVec` or `.into()` at assignment. The `m.num_class != out.num_class` comparison works unchanged (`SmallVec: PartialEq`).
- `bulk.rs:235` does `metadata.attributes.unwrap_or_else(|| "{}".to_string())` → becomes `.unwrap_or_else(|| CompactString::from("{}"))` (or `Option<CompactString>`).
- Compile the **whole workspace** after the type change before running tests; the compiler is the checklist.
**Warning signs:** `expected SmallVec, found Vec` / `expected CompactString, found String` errors — each is a ripple site to convert with `.into()`.

### Pitfall 2: Choosing inline N too large bloats `Model` and pessimizes cache (MEM-02)
**What goes wrong:** `SmallVec<[i32; 16]>` for a field that's length-1 in 99% of models adds 60 bytes of dead inline storage per `Model`, hurting the cache-locality goal.
**Why it happens:** Instinct to size N to the worst case (large multiclass). But `class_id`/`target_id` grow with `num_tree` (can be thousands) — they will spill to heap for any non-trivial model regardless of N, so a large N only wastes space on the common small case.
**How to avoid:** Size N to the **dominant** shape (see §MEM-02 sizing table). For fields that scale with tree count (`class_id`, `target_id`), a small N (1) is correct — they spill when big, inline when trivial. CompactString needs no sizing (fixed 24B).
**Warning signs:** `size_of::<Model>()` jumps substantially after the change; a `Model`-size assertion in the report sanity-check catches it.

### Pitfall 3: Allocator leaks into the abi3 wheel (MEM-03, breaks D-08/SC3)
**What goes wrong:** If the allocator crate is added as a non-optional dep of any crate in `treelite-py`'s dependency graph, or `#[global_allocator]` is declared in a lib, the wheel links a non-system allocator.
**Why it happens:** Adding `tikv-jemallocator` to `[workspace.dependencies]` is fine (workspace deps are not auto-included), but adding it to a *crate's* `[dependencies]` without `optional = true`, or enabling the feature by default, pulls it in.
**How to avoid:** Keep allocator deps `optional = true`, features non-default and only on `treelite-harness`'s bin/bench targets. **Verify** with `cargo tree -p treelite-py --no-default-features --features cpu | grep -E "jemalloc|mimalloc"` returning empty, and a maturin wheel build that succeeds with no allocator symbol. Add this as an explicit verification task.
**Warning signs:** `cargo tree` on the py crate shows `tikv-jemalloc-sys` or `libmimalloc-sys`.

### Pitfall 4: `bytemuck::cast_slice` panics on a misaligned/odd-length recast (MEM-01)
**What goes wrong:** `cast_slice` panics (not UB) if the source slice's byte length isn't divisible by `size_of::<T>()` or alignment is violated. For `&[u8] → &[T]` directions this matters; for the `&[T] → &[u8]` direction used by the serializer it is always safe (u8 has align 1, length always divisible).
**Why it happens:** Only relevant if MEM-01 also adds a `&[u8] → &[T]` recast in a loader's deserialize path. The current serializer goes `&[T] → &[u8]` only (always safe).
**How to avoid:** For MEM-01, restrict recasts to the **emit** direction (`column → bytes`), which is unconditionally safe. The *deserialize* reader (`binary.rs Reader::array`) already decodes element-by-element from validated offsets — do **not** replace that with a bulk `cast_slice` over the borrowed input buffer (untrusted input may be misaligned/short → would panic instead of returning the typed `SerializeError`). Keep the bounds-checked element decode on the read path.
**Warning signs:** A new `cast_slice` over `&[u8]` input on the deserialize side; `golden_v5.rs` round-trip panicking instead of asserting.

### Pitfall 5: jemalloc-ctl stats read without advancing `epoch` returns stale numbers (D-10)
**What goes wrong:** jemalloc caches `stats.*` and only refreshes them when `epoch` is advanced; reading without `epoch::advance()` reports the value from a previous sample, making the before/after report wrong.
**Why it happens:** It's a documented jemalloc API contract that's easy to miss.
**How to avoid:** Call `tikv_jemalloc_ctl::epoch::advance()` immediately before each `stats::allocated::read()` / `stats::resident::read()` (see Pattern 4).
**Warning signs:** Before and after RSS numbers are suspiciously identical or don't move with workload.

## Code Examples

### Verifying the abi3 wheel stays allocator-free (the D-08 gate, as a test/CI step)
```bash
# Source: project Cargo graph + maturin workflow (Phase 8 precedent)
cargo tree -p treelite-py --no-default-features --features cpu \
  | grep -Eq "jemalloc|mimalloc" && echo "FAIL: allocator leaked into wheel" || echo "OK: wheel is allocator-free"
```

### Golden byte-compare is the MEM-01/MEM-02 gate (already exists, must stay green)
```rust
// Source: crates/treelite-harness/tests/golden_v5.rs
// serialize(deserialize(golden_v5.bin)) == golden_v5.bin  byte-for-byte.
// After MEM-01 (cast_slice) and MEM-02 (SmallVec/CompactString deref), this MUST still pass.
```

### Memory report target shape (mirrors Phase-7 report.rs → docs/)
```rust
// Source: crates/treelite-harness/src/report.rs precedent (observational, #[ignore], writes docs/)
// New bench/bin: load a model set, predict, sample (allocated, resident) per allocator,
// write a markdown table to docs/MEMORY_REPORT.md. Observational — never a hard CI gate (D-10).
```

## MEM-01 — Recast Site Inventory (the specific question)

| Site | Direction | Pod-safe? | Class | Action |
|------|-----------|-----------|-------|--------|
| `tree_buf.rs:96` `TreeBuf::as_bytes` | `&[T]→&[u8]` | yes (already `T: Pod`) | safe everywhere (in-memory/device) | already done; reused by cubecl upload — no change |
| `serialize/mod.rs:77` `le_bytes_of<T: Copy>` (header arrays: `num_class`/`leaf_vector_shape`/`target_id`/`class_id` i32, `base_scores` f64) | `&[T]→&[u8]` | yes | **serializer-path, golden-gated** | replace `unsafe from_raw_parts` with `bytemuck::cast_slice`; bound → `T: Pod`; byte-identical; document LE |
| `serialize/mod.rs` tree columns via `le_bytes_of` (`cleft`/`cright`/`split_index` i32, `leaf_value`/`threshold` T, `leaf_vector` T, `leaf_vector_begin/end` u64, category u32/u64, stats u64/f64) | `&[T]→&[u8]` | yes | **serializer-path, golden-gated** | same `cast_slice` swap; byte-identical |
| `serialize/mod.rs` enum/bool columns (`node_type`, `cmp` via `as u8`; `default_left`, `category_list_right_child`, `*_present` via `bool_bytes`) | explicit `as u8` map | **NO** (not repr-Pod) | excluded | leave as-is — the explicit map defines the wire byte |
| `binary.rs Reader::array` (deserialize read) | `&[u8]→T` | conditionally | **do NOT bulk-recast** | keep element-wise bounds-checked decode (untrusted input; cast_slice would panic not error — Pitfall 4) |
| legacy-binary XGBoost loader LE cursor | `&[u8]→T` | n/a | excluded by design | Phase-3 deliberately uses `from_le_bytes` cursor (no transmute); MEM-01 does not regress that to a bulk recast |

**Answer to "does `cast_slice` ever reorder/pad vs the current emission?":** No. `cast_slice` only rewrites slice fat-pointer metadata (ptr + len·size); it never reorders elements, never inserts padding, never changes endianness (it's native-endian = LE on the manifest host). The emitted bytes are identical to the current `from_raw_parts`-based `le_bytes_of`. Alignment/Pod constraints: every numeric column element (`i32`,`f64`,`u64`,`u32`,`f32`) is already `bytemuck::Pod`; `&[u8]` target has align 1 so the emit direction never fails the alignment check.

## MEM-02 — Field Sizing & Ripple (the specific question)

### Recommended inline capacities (Claude's discretion per D-06)
| Field | Type → | Inline N | Rationale (real model shapes) |
|-------|--------|----------|-------------------------------|
| `num_class` | `SmallVec<[i32; 1]>` | 1 | `[1]` for binary clf / regression (the overwhelming majority); multi-target spills (rare). |
| `leaf_vector_shape` | `SmallVec<[i32; 2]>` | 2 | Always a 2-tuple `[rows, cols]` (`[1,1]` binary) — N=2 inlines the universal case exactly. |
| `target_id` | `SmallVec<[i32; 1]>` | 1 | Length == num_tree; `[0]` for single-target single-tree but grows with forests → spills when big. Small N is correct (don't pre-bloat). |
| `class_id` | `SmallVec<[i32; 1]>` | 1 | Same as `target_id` — scales with num_tree; inline only the trivial case. |
| `base_scores` | `SmallVec<[f64; 1]>` | 1 | Scalar base score is the dominant case (binary/regression); vector base_score (multiclass) spills. |
| `postprocessor` | `CompactString` | — | Names (`"identity"`,`"sigmoid"`,`"softmax"`,`"exponential_standard_ratio"`=26B) — most ≤24B inline; the longest spills harmlessly. |
| `attributes` | `CompactString` | — | `"{}"` inline; arbitrary JSON blob spills to heap = exactly `String` behavior (D-06: harmless, applied uniformly for consistency). |

> **Caveat on `exponential_standard_ratio` (26 chars):** it exceeds the 24-byte inline threshold and will heap-allocate as a `CompactString` — identical to `String`, no regression, just no SVO win for that one name. CompactString is never worse than `String`. Confirmed heap-equivalent above threshold `[CITED: COMPACT_STR_OPTIMIZATION_EN.md §1, §3]`.

### Exact ripple list (what each consumer needs)
| Crate / file | Role | Needs |
|--------------|------|-------|
| `treelite-core/src/model.rs` | owns field types | change 7 field type declarations + `Model::new()` initializers (`Vec::new()`→`SmallVec::new()`, `String::new()`→`CompactString::new("")`/`CompactString::default()`) |
| `treelite-core/src/serialize/mod.rs` | emit | **no change** — `le_bytes_of(&m.x)` and `b.string(&m.postprocessor)` deref through |
| `treelite-core/src/serialize/mod.rs` (deserialize, ~line 303-363) | read-back assign | `model.num_class = r.array(...)?` returns `Vec` → wrap `.into()` (or have `array` return into the target). 7 assignment sites. |
| `treelite-builder/src/lib.rs` (`Metadata` struct + assign 768-774) | construct | change `Metadata` field types to match (recommended) OR `.into()` at assignment |
| `treelite-builder/src/bulk.rs` (229-235) | construct | `.into()` / type change; `attributes.unwrap_or_else(|| "{}".to_string())` → CompactString |
| `treelite-builder/src/concat.rs` (79-150) | merge | `Vec<i32>` locals → SmallVec or `.into()`; `.clone()` works; `!=` comparison works (PartialEq) |
| `treelite-lightgbm/src/lib.rs`, `treelite-sklearn/src/*`, `treelite-xgboost/src/*` | loaders write | populate `Metadata`/`Model` fields — `.into()` where they build `Vec`/`String` literals |
| `treelite-gtil/src/{shape,lib,error}.rs` | read | **no change** — read via deref/iter/index (SmallVec & Vec share the slice API) |
| `treelite-py/src/model.rs` | Python accessors | **no metadata getter exists today** (verified). If the planner ADDS list/str getters they must build `PyList`/`str` from the field via `.as_slice()` / `.as_str()`. D-05's "keep list/str" = if/when exposed, return Python `list`/`str`, never a custom type. Existing getters (`num_tree` etc.) are unaffected. |

**Two preservation gates the planner MUST add explicit verification tasks for (D-05):**
1. `golden_v5.rs` byte-compare stays green (SmallVec/CompactString deref → identical bytes).
2. The full pytest suite + `cargo test --workspace` stay green; if metadata getters are added, assert they return `list`/`str`.

## MEM-03 — Allocator Wiring & Report (the specific question)

- **Crate selection:** `tikv-jemallocator 0.7.0` (maintained, over the manual's unmaintained `jemallocator 0.5`) + `mimalloc 0.1.52`, both as `optional` workspace deps. `tikv-jemalloc-ctl 0.7.0` (version-locked to the allocator) for stats.
- **Wiring:** non-default mutually-exclusive `jemalloc`/`mimalloc` features on `treelite-harness`; `#[global_allocator]` static lives **only** in a new bench (`benches/memory_report.rs`) or `src/bin/` target, with `compile_error!` guarding the both-on case (Pattern 3). Edition-2024 compatible (all three are plain crates, no edition coupling).
- **abi3-wheel isolation:** allocator deps never reach `treelite-py`; verify with the `cargo tree -p treelite-py` grep (Pitfall 3 / Code Examples). Add as a checked task.
- **Stats for the report:** jemalloc rows via `epoch::advance()` + `stats::allocated::read()` + `stats::resident::read()` (Pattern 4/5); mimalloc + system rows via `/proc/self/statm` RSS (mimalloc has no equivalent safe ctl crate; be explicit about the per-row method).
- **Benchmark approach:** a custom in-process RSS sampler is recommended over `criterion` — criterion measures *time*, the report's subject is *peak RSS/allocation*. (criterion is optional if timing is also wanted.) Load a small representative model set (reuse existing fixtures: an XGBoost JSON model, a LightGBM model, a sklearn RF), run predict, sample before/after per allocator config.
- **`MEMORY_REPORT.md` contents (following the Phase-7 `GPU_EQUIVALENCE_REPORT.md` precedent):** a header noting it is **observational, not a CI gate** (D-10); a table with columns `Model | Allocator (system / jemalloc / mimalloc) | peak resident (RSS) | bytes allocated | measurement method`; a `size_of::<Model>()` before/after row (sanity that SmallVec didn't bloat — Pitfall 2); a manifest/provenance note (toolchain, host = x86-64 LE ROCm box, date); and an explicit "1e-5 harness + golden byte-compare both green" attestation line (D-11). Regenerated explicitly (`#[ignore]` or a bin) on the dev box like the GPU report.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `jemallocator 0.5` | `tikv-jemallocator 0.7` | ~2022+ | The TiKV fork is the maintained binding; the original is dormant. Manual is stale here. |
| `smallvec` fixed size-list API | `smallvec 1.x` const-generic-capable (`SmallVec<[T; N]>`, `const_generics` feature) / `2.0` new `SmallVec<T, N>` API (alpha) | 2.0 in alpha as of 2026-06 | Stay on stable `1.x`; 2.0 is pre-release (FND-02 forbids). |
| `compact_str 0.8` (manual) | `compact_str 0.9.1` | recent | Same SSO semantics; pin 0.9.1. |

**Deprecated/outdated:**
- The manuals' `jemallocator = "0.5"` (JEMALLOC_MANUAL.md §4) — superseded by `tikv-jemallocator 0.7`. Use the tikv crate.
- `compact_str = "0.8"` (manual) — registry latest is `0.9.1`; pin the latest stable per FND-02.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `smallvec 1.15.1` is the exact latest `1.x` stable (manual cites it; `cargo search` shows only the `2.0-alpha` head, API blocked) | Standard Stack | LOW — planner runs `cargo add smallvec --dry-run` to confirm the exact 1.x patch; any 1.x works (API stable across 1.x). |
| A2 | `tikv-jemalloc-ctl 0.7` exposes `epoch::advance`, `stats::allocated`, `stats::resident` with that path shape | MEM-03 / Pattern 4 | LOW — long-stable API; planner verifies against docs.rs at plan time. If a path differs, the `/proc/self/statm` fallback covers the report regardless. |
| A3 | Changing `Model` metadata field types does not alter `Model`'s `!Send` auto-trait | Project Constraints | LOW — `SmallVec`/`CompactString` are `Send` (don't remove it); `!Send` comes from the `*const T` in `TreeBuf::Borrowed`, untouched. A `static_assert`-style `fn _assert_not_send()` test confirms. |
| A4 | The v5 serializer never serde-serializes `Model` directly (so the `serde` features on smallvec/compact_str are optional, not required) | Installation note | LOW — verified the serializer is hand-framed and `serde_json` is dump-only via getters; planner drops `serde` features if unused. |
| A5 | `mimalloc 0.1.52` default features (v2/v3) build + run on the AMD/ROCm Linux box | Standard Stack / D-09 | LOW — D-09 *requires* this validation as a phase task; it's a build-and-run check, not an assumption to ship on. |

## Open Questions

1. **Should `builder::Metadata` fields also become SmallVec/CompactString, or stay `Vec`/`String` with `.into()` at the `Model` boundary?**
   - What we know: changing `Metadata` too eliminates per-assignment conversions and matches D-04's "ripple accepted"; keeping `Vec`/`String` localizes the change to `model.rs` + conversion sites.
   - Recommendation: change `Metadata` too (consistency, fewer conversion points). Planner decides; both compile.

2. **Does the planner want to ADD Python metadata getters (`num_class`, `base_scores`, …) in this phase, or leave them unexposed?**
   - What we know: they are not exposed today (verified). D-05 only requires that *if/when* exposed they return `list`/`str`. Phase 9's scope (CONTEXT) is hardening, not new API surface.
   - Recommendation: do NOT add new getters (out of scope — "no new functionality", CONTEXT domain). The D-05 list/str gate then applies only to *existing* getters, which are unaffected. If a getter is added incidentally, return `list`/`str`.

3. **One bench target per allocator, or one parameterized binary selecting allocator via feature?**
   - What we know: `#[global_allocator]` is compile-time per-binary; comparing system vs jemalloc vs mimalloc requires either three feature-built runs of one target or three targets.
   - Recommendation: one target, built three times (`--features jemalloc`, `--features mimalloc`, no features = system), each appending its row to `MEMORY_REPORT.md`. Simplest and matches the mutually-exclusive-feature design (D-07).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable, edition 2024 | whole workspace | ✓ | (workspace builds today) | — |
| C toolchain (`cc`) | `tikv-jemalloc-sys`, `libmimalloc-sys` vendored build (D-09) | ✓ | (upstream C++ tree builds locally) | — |
| `cargo` / crates.io | adding the 5 deps | ✓ | — | — |
| maturin | rebuild abi3 wheel after MEM-02 to confirm isolation | ✓ (Phase 8) | — | — |
| jemalloc/mimalloc on Linux | MEM-03 build+run validation (D-09) | ✓ (Linux ROCm box) | tikv-jemallocator 0.7 / mimalloc 0.1.52 | system malloc row always available |
| `/proc/self/statm` | RSS sampling for mimalloc/system rows | ✓ (Linux) | — | jemalloc-ctl for the jemalloc row |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none material — all required tooling is present on the dev box.

## Validation Architecture

> `workflow.nyquist_validation` not found in `.planning/config.json` scope read; treating as enabled. (Planner should confirm the config key; the gates below are the project's existing instruments regardless.)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + pytest (py crate) |
| Config file | none (workspace default) / `crates/treelite-py` pytest conftest |
| Quick run command | `cargo test -p treelite-core` (serializer + tree_buf unit tests) |
| Full suite command | `cargo test --workspace` + `uv run pytest` (py, per MEMORY: `uv run python` not bare) |
| Phase gate | `cargo test --workspace` green + golden byte-compare green + pytest green within 1e-5 (D-11) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MEM-01 | serializer recast keeps bytes identical | integration | `cargo test -p treelite-harness --test golden_v5` | ✅ `tests/golden_v5.rs` (the gate) |
| MEM-01 | `TreeBuf::as_bytes` roundtrip after bound change | unit | `cargo test -p treelite-core tree_buf` | ✅ `tree_buf.rs` tests |
| MEM-01/02 | full equivalence within 1e-5 | integration | `cargo test -p treelite-harness` (equivalence/matrix) | ✅ `equivalence.rs`, `gtil_matrix*.rs` |
| MEM-02 | SmallVec/CompactString deref → byte-identical | integration | `cargo test -p treelite-harness --test golden_v5` | ✅ (same gate) |
| MEM-02 | workspace compiles + tests green after ripple | build+unit | `cargo test --workspace` | ✅ all crate tests |
| MEM-02 | py list/str contract (if getters touched) | integration | `uv run pytest crates/treelite-py` | ✅ pytest suite |
| MEM-02 | `Model` stays `!Send`, size not bloated | unit | new `#[test]` `model_stays_not_send` + `size_of` assertion | ❌ Wave 0 |
| MEM-03 | allocator builds+runs on Linux | smoke | `cargo run/bench -p treelite-harness --features jemalloc` (and mimalloc) | ❌ Wave 0 |
| MEM-03 | wheel stays allocator-free | smoke | `cargo tree -p treelite-py ... | grep -E "jemalloc|mimalloc"` empty | ❌ Wave 0 |
| MEM-03/D-10 | report regenerates | manual/`#[ignore]` | new bench/bin writing `docs/MEMORY_REPORT.md` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <touched-crate>` (e.g. `-p treelite-core` after model.rs edits).
- **Per wave merge:** `cargo test --workspace` + `cargo test -p treelite-harness --test golden_v5`.
- **Phase gate:** full `cargo test --workspace` + pytest green + golden byte-compare green (D-11), then `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/treelite-core/tests/` — a `model_stays_not_send` + `size_of::<Model>()` sanity test (catches Pitfall 2/A3).
- [ ] `crates/treelite-harness/benches/memory_report.rs` (or `src/bin/`) — the allocator-gated RSS sampler + report writer (MEM-03/D-10).
- [ ] `crates/treelite-harness/Cargo.toml` — `jemalloc`/`mimalloc` non-default mutually-exclusive features + optional deps.
- [ ] A wheel-isolation check (`cargo tree -p treelite-py` grep) wired as a test or CI/script step.
- [ ] Dep install: add the 5 crates to `[workspace.dependencies]`; confirm exact pinned versions via `cargo add --dry-run`.

## Security Domain

> `security_enforcement` config not located in the read scope; this is an internal numerical library with no network/auth surface. The only relevant ASVS axis is input validation on the *deserialize* path, which is pre-existing and must not regress.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library, no auth) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | The v5 deserialize `Reader` is bounds-checked (`binary.rs:92-244`): every count is validated against remaining buffer before allocation, no `unsafe` slicing. **MEM-01 must NOT replace the element-wise bounds-checked read with a bulk `cast_slice` over untrusted input** (Pitfall 4) — that would convert a typed error into a panic and reintroduce alignment risk. |
| V6 Cryptography | no | — |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed v5 blob drives huge alloc | Denial of Service | Existing `check_count` pre-allocation bound (`binary.rs:176`) — keep on the read path; MEM-01 only touches the *emit* path. |
| `cast_slice` on misaligned/odd input panic | Denial of Service | Restrict MEM-01 recasts to the safe `&[T]→&[u8]` emit direction only (always aligned/divisible). |
| Allocator vendored C build supply chain | Tampering | Pinned versions, slopcheck-verified, vendored `*-sys` build (no network postinstall); maintained TiKV/Microsoft sources. |

## Sources

### Primary (HIGH confidence)
- Codebase (read directly): `crates/treelite-core/src/{tree_buf.rs, model.rs}`, `crates/treelite-core/src/serialize/{mod.rs, binary.rs, fields.rs}`, `crates/treelite-builder/src/{lib.rs, bulk.rs, concat.rs}`, `crates/treelite-py/src/model.rs`, `crates/treelite-harness/{Cargo.toml, src/lib.rs, src/report.rs, tests/}`, workspace `Cargo.toml` — the ripple set, serializer emit path, byte gate, and isolation pattern are all observed, not assumed.
- ctx7 `/websites/rs_smallvec_smallvec` (High reputation, 2084 snippets) — `SmallVec<[T; N]>` array-syntax API, `serde`/`const_new`/`const_generics` features, `Deref` to slice.
- `slopcheck install` against crates.io — all 6 candidate crates `[OK]`.
- `.planning/{REQUIREMENTS.md, ROADMAP.md, phases/09/09-CONTEXT.md}` — requirements, invariants, locked decisions.
- Optimiser manuals (canonical refs): `SMALLVEC_MANUAL.md`, `COMPACT_STR_OPTIMIZATION_EN.md`, `JEMALLOC_MANUAL.md`, `MIMALLOC_MANUAL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md` — read in full.

### Secondary (MEDIUM confidence)
- `cargo search` (crates.io) version reads: `compact_str 0.9.1`, `tikv-jemallocator 0.7.0`, `tikv-jemalloc-ctl 0.7.0`, `mimalloc 0.1.52`, `criterion 0.8.2`, `jemallocator 0.5.4`.

### Tertiary (LOW confidence)
- crates.io JSON API (blocked by data-access policy) — could not pull exact `smallvec 1.x` patch / publish dates; mitigated by the manual's `1.15.1` citation + `cargo add --dry-run` at plan time (A1).
- `tikv-jemalloc-ctl` epoch/stats API path shape — from training + manual consistency, not ctx7-confirmed (A2); `/proc/self/statm` fallback de-risks it.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — names from CONTEXT/manuals, versions from `cargo search`, all slopcheck-clean; only the exact smallvec 1.x patch is `cargo add`-confirm-at-plan (A1).
- Architecture / ripple map: HIGH — read every consumer site directly; the serializer's deref-transparency and the existing `unsafe le_bytes_of` replace-target are observed in source.
- Pitfalls: HIGH — derived from the actual code (bounds-checked reader, enum/bool emit maps, isolation pattern) and documented crate contracts (epoch, cast_slice panic).
- MEM-03 stats API specifics: MEDIUM — jemalloc-ctl path shape is training-sourced (A2), de-risked by the `/proc` fallback.

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (stable crates; re-confirm exact versions if planning slips a month — verify `smallvec 2.0` is still pre-release before any bump).

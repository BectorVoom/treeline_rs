# Phase 9: Memory-Efficiency Hardening - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 13 modify + 3 new
**Analogs found:** 16 / 16 (every touched site has an in-repo analog; this phase replaces hand-rolled primitives, it does not invent patterns)

> All file paths below are workspace-relative to `/home/user/Documents/workspace/treeline_rs`.
> Every excerpt was read from the live source, not from RESEARCH prose. RESEARCH line
> numbers were re-verified; a few drifted by ±2 and are corrected here.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-core/src/serialize/mod.rs` (MODIFY: `le_bytes_of`) | serializer | transform (`&[T]→&[u8]`) | `crates/treelite-core/src/tree_buf.rs:96-105` (`as_bytes`) | exact — same `bytemuck::cast_slice` call, already in this repo |
| `crates/treelite-core/src/serialize/mod.rs` (MODIFY: deserialize assign ~355-363) | serializer | transform (read-back) | self (lines 355-363) | in-place — wrap `.into()` at 7 assign sites |
| `crates/treelite-core/src/model.rs` (MODIFY: 7 field types + `new()`) | model | CRUD (owns metadata) | self (the `Vec<i32>`/`String` decls 68-84) | in-place type swap |
| `crates/treelite-builder/src/lib.rs` (MODIFY: `Metadata` 59-71 + assign 768-774) | builder | CRUD | self / model.rs | role-match |
| `crates/treelite-builder/src/bulk.rs` (MODIFY: 229-235) | builder | CRUD | builder/lib.rs:768-774 | exact (identical assign block) |
| `crates/treelite-builder/src/concat.rs` (MODIFY: 79-150) | builder | transform (merge) | self | in-place local-type swap |
| `crates/treelite-xgboost/src/lib.rs` (MODIFY: ~202-260) | loader | CRUD (write) | self | `vec![]`/`.to_string()` literals → `.into()`/`smallvec![]` |
| `crates/treelite-lightgbm/src/lib.rs` (MODIFY: 342-364) | loader | CRUD (write) | xgboost/lib.rs | role-match |
| `crates/treelite-sklearn/src/bulk.rs` (MODIFY: 272-371) | loader | CRUD (write) | xgboost/lib.rs | role-match |
| `crates/treelite-gtil/src/shape.rs` (READ-ONLY verify, no edit) | service | read | self (61-77 `.iter()/.first()/.get()`) | NO CHANGE — deref-transparent |
| `crates/treelite-py/src/model.rs` (READ-ONLY verify, no edit) | binding | read | self (getters 49-85) | NO CHANGE — no metadata getter exists |
| `crates/treelite-harness/Cargo.toml` (MODIFY: features + optional deps) | config | — | self (GPU `[features]` rocm/cuda/wgpu) | exact (same optional-dep + feature shape) |
| `Cargo.toml` (MODIFY: `[workspace.dependencies]`) | config | — | self (`bytemuck` entry, line 27) | in-place add |
| **NEW** `crates/treelite-harness/src/bin/memory_report.rs` (or `benches/`) | bin/bench | event-driven (sample→write) | `crates/treelite-harness/src/report.rs` + `tests/gtil_matrix_gpu.rs:330-494` | role-match (Phase-7 report precedent) |
| **NEW** `crates/treelite-harness/src/memory.rs` (optional sampler/writer helper) | utility | transform | `crates/treelite-harness/src/report.rs` (`render_markdown`/`emit`) | exact-structure |
| **NEW** `crates/treelite-core/tests/model_invariants.rs` (`model_stays_not_send` + `size_of`) | test | — | `crates/treelite-core/tests/tree_buf.rs` | role-match (core integration test) |

---

## Shared Patterns

### Zero-copy `&[T]→&[u8]` recast (MEM-01) — the canonical seam ALREADY in this repo
**Source:** `crates/treelite-core/src/tree_buf.rs:96-105`
**Apply to:** the serializer's `le_bytes_of` (the ONE replace target). Copy this exact call form.
```rust
// tree_buf.rs:96-105 — the existing, tested bytemuck seam to copy.
impl<T: Copy + bytemuck::Pod> TreeBuf<T> {
    /// Uses `bytemuck::cast_slice`, which validates size/alignment and is
    /// never a hand-rolled `transmute` (T-06-02).
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
}
```
> The bound `T: Copy + bytemuck::Pod` and the body `bytemuck::cast_slice(...)` are the template.
> The serializer's replacement uses the SAME call; only the bound tightens `Copy` → `Pod`.
> Roundtrip test idiom (`tree_buf.rs:126-140`): `let back: &[T] = bytemuck::cast_slice(buf.as_bytes());`

### Deref-transparency through SmallVec / CompactString (MEM-02) — the zero-edit win
**Source / proof:** `crates/treelite-core/src/serialize/mod.rs:102-114` (emit) reads `&m.num_class` / `&m.postprocessor`.
**Apply to:** the serializer emit path (lines 102-114) needs **NO edit** — `SmallVec<[i32;N]>` derefs to `&[i32]` and `CompactString` derefs to `&str`, feeding the exact same `le_bytes_of(...)` / `b.string(...)` calls.
```rust
// serialize/mod.rs:102-114 — UNCHANGED after MEM-02 (deref carries the new types through).
emit_array(b, m.num_class.len(), le_bytes_of(&m.num_class));        // serializer.cc:113
emit_array(b, m.leaf_vector_shape.len(), le_bytes_of(&m.leaf_vector_shape)); // :114
emit_array(b, m.target_id.len(), le_bytes_of(&m.target_id));        // serializer.cc:115
emit_array(b, m.class_id.len(), le_bytes_of(&m.class_id));          // serializer.cc:116
b.string(&m.postprocessor);                                         // serializer.cc:117
emit_array(b, m.base_scores.len(), le_bytes_of(&m.base_scores));    // serializer.cc:120
b.string(&m.attributes);                                            // serializer.cc:121
```
> This is the single most fidelity-critical consumer and it is touched zero times. The golden
> byte-compare (`tests/golden_v5.rs`) is the gate that proves the deref is byte-identical (D-03/D-05).

### Allocator isolation via optional dep + non-default features (MEM-03)
**Source:** `crates/treelite-harness/Cargo.toml` `[features]` (the GPU rocm/cuda/wgpu shape).
**Apply to:** the harness Cargo.toml — copy the `optional = true` + non-default feature pattern, one tier stricter (the static lives only in a bin/bench, never a `pub fn`).
```toml
# crates/treelite-harness/Cargo.toml — EXISTING GPU isolation pattern to mirror:
[features]
default = []
rocm = ["dep:cubecl", "cubecl/rocm", "treelite-cubecl/rocm"]   # ← shape to copy
# ...
[dependencies]
cubecl = { workspace = true, optional = true }                  # ← optional-dep shape to copy
```
> MEM-03 adds, in the same file: `jemalloc = ["dep:tikv-jemallocator", "dep:tikv-jemalloc-ctl"]`,
> `mimalloc = ["dep:mimalloc"]` (both non-default, mutually exclusive), with the three crates as
> `{ workspace = true, optional = true }`. The `#[global_allocator]` static goes ONLY in the new
> `src/bin/memory_report.rs` (or `benches/`), guarded by `#[cfg(all(feature="jemalloc", not(feature="mimalloc")))]`
> + a `compile_error!` for the both-on case (RESEARCH Pattern 3).

### Workspace dependency declaration
**Source:** `Cargo.toml:27` — `bytemuck = { version = "1", features = ["derive"] }`
**Apply to:** add the 5 new pins (smallvec 1.15.1, compact_str 0.9.1, tikv-jemallocator 0.7.0, tikv-jemalloc-ctl 0.7.0, mimalloc 0.1.52) to the same `[workspace.dependencies]` table (line 20+). Crates reference them with `{ workspace = true }` (model.rs/builder) or `{ workspace = true, optional = true }` (harness allocator deps).

---

## Pattern Assignments

### `crates/treelite-core/src/serialize/mod.rs` — `le_bytes_of` (serializer, transform) [MEM-01]

**Analog:** `crates/treelite-core/src/tree_buf.rs:96-105` (the existing `bytemuck::cast_slice` seam).

**Current REPLACE target** (lines 77-81 — verified, matches RESEARCH):
```rust
/// Reinterpret a `&[T]` of plain-old-data as its little-endian byte image.
fn le_bytes_of<T: Copy>(slice: &[T]) -> &[u8] {
    // SAFETY: `T: Copy` POD scalars have no padding/invalid bit patterns when
    // viewed as bytes; the lifetime is tied to `slice`; length is exact.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}
```

**MEM-01 form** (copy the `tree_buf.rs:102` body; tighten bound; document LE per D-02):
```rust
/// Reinterpret a `&[T]` POD column as its native-LE byte image (D-02: LE-host only).
/// On the x86-64/ROCm manifest host this is byte-identical to the old
/// `from_raw_parts` transmute and to upstream's memcpy image — gated by
/// `fixtures/golden_v5.bin` (D-03). No big-endian byte-swap path (D-02).
fn le_bytes_of<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
    bytemuck::cast_slice(slice)
}
```
> All call sites (mod.rs:102,105-106,108,109,113) pass `i32`/`f64` columns — already `Pod`. The bound
> change from `T: Copy` to `T: bytemuck::Pod` compiles unchanged at every site. The `b.scalar_le(&x.to_le_bytes())`
> calls (lines 90-117) are NOT `le_bytes_of` and stay as-is.

**Excluded — leave untouched (Anti-Pattern):** the enum/bool tree columns emitted via explicit `as u8` /
`bool_bytes` maps in the per-tree walk are NOT Pod-recast candidates; the explicit map defines the wire byte
(RESEARCH §Recast Site Inventory row 3). Restrict MEM-01 to the `&[T]→&[u8]` emit direction only.

**Do NOT touch the read path:** `binary.rs Reader::array` decodes element-wise from bounds-checked offsets
over untrusted input — a bulk `cast_slice` there would panic instead of returning `SerializeError` (RESEARCH Pitfall 4 / V5).

---

### `crates/treelite-core/src/serialize/mod.rs` — deserialize assign (serializer, read-back) [MEM-02]

**Analog:** self, lines 355-363 (the read-back assignment block).

**Current** (verified 303-311 reads return `Vec`; 355-363 assigns into `Model`):
```rust
let num_class = r.array(4, decode_i32)?;     // line 303 → returns Vec<i32>
// ...
let base_scores = r.array(8, decode_f64)?;   // line 310 → returns Vec<f64>
let postprocessor = r.string()?;             // line 307 → returns String
// ...
model.num_class = num_class;                 // line 355
model.base_scores = base_scores;             // line 362
model.postprocessor = postprocessor;         // line 359
model.attributes = attributes;               // line 363
```
**MEM-02 minimal change:** wrap each assignment with `.into()` (the 4 i32 arrays + base_scores + 2 strings).
`SmallVec: From<Vec<T>>` and `CompactString: From<String>` both exist, so:
```rust
model.num_class = num_class.into();
model.base_scores = base_scores.into();
model.postprocessor = postprocessor.into();
model.attributes = attributes.into();
```
> 7 assign sites (355-363, skipping the scalar `num_feature`/`task_type`/`num_target`/`sigmoid_alpha`/`ratio_c`).

---

### `crates/treelite-core/src/model.rs` — field types + `new()` (model, CRUD) [MEM-02]

**Analog:** self. The ONLY type declarations that change.

**Current** (verified 68-84):
```rust
pub num_class: Vec<i32>,          // 68
pub leaf_vector_shape: Vec<i32>,  // 70
pub target_id: Vec<i32>,          // 72
pub class_id: Vec<i32>,           // 74
pub postprocessor: String,        // 76
pub base_scores: Vec<f64>,        // 82
pub attributes: String,           // 84
```
**MEM-02 form** (inline N from RESEARCH §MEM-02 sizing — N for the dominant shape, not the max):
```rust
pub num_class: SmallVec<[i32; 1]>,          // [1] binary clf / regression
pub leaf_vector_shape: SmallVec<[i32; 2]>,  // always a 2-tuple [rows, cols]
pub target_id: SmallVec<[i32; 1]>,          // len==num_tree; small N, spills when big
pub class_id: SmallVec<[i32; 1]>,           // same — scales with num_tree
pub postprocessor: CompactString,           // names ≤24B inline
pub base_scores: SmallVec<[f64; 1]>,        // scalar base_score dominant
pub attributes: CompactString,              // "{}" inline; large JSON spills (harmless, D-06)
```
**`Model::new()` initializers** (verified 118-126): swap constructors:
```rust
// Vec::new()    → SmallVec::new()
// String::new() → CompactString::new("")  (or CompactString::default())
num_class: SmallVec::new(),            // was Vec::new()      (118)
postprocessor: CompactString::new(""), // was String::new()  (122)
attributes: CompactString::new(""),    // was String::new()  (126)
```

---

### `crates/treelite-builder/src/lib.rs` — `Metadata` struct + assign (builder, CRUD) [MEM-02]

**Analog:** model.rs field decls (mirror them). Recommendation (RESEARCH Open-Q1): change `Metadata`'s
field types TOO, so the assignment block needs no per-field `.into()`.

**Current `Metadata`** (verified 59-71) — same `Vec<i32>`/`String` shape as model.rs, plus `attributes: Option<String>`:
```rust
pub num_class: Vec<i32>,            // 59
pub leaf_vector_shape: Vec<i32>,   // 61
pub target_id: Vec<i32>,           // 63
pub class_id: Vec<i32>,            // 65
pub postprocessor: String,         // 67
pub base_scores: Vec<f64>,         // 69
pub attributes: Option<String>,    // 71
```
**Assign block** (verified 768-774):
```rust
model.num_class = metadata.num_class;                       // 768
model.leaf_vector_shape = metadata.leaf_vector_shape;       // 769
model.target_id = metadata.target_id;                       // 770
model.class_id = metadata.class_id;                         // 771
model.postprocessor = metadata.postprocessor;               // 772
model.base_scores = metadata.base_scores;                   // 773
model.attributes = metadata.attributes.unwrap_or_else(|| "{}".to_string()); // 774
```
**MEM-02 change:** if `Metadata` fields become SmallVec/CompactString to match, assigns 768-773 stay
verbatim. Line 774 becomes `.unwrap_or_else(|| CompactString::from("{}"))` (or keep `Option<CompactString>`).
`expected_num_tree = metadata.target_id.len()` (line 212) and `metadata.leaf_vector_shape.iter()...product()`
(line 208) work unchanged (SmallVec shares the slice/iter API).

---

### `crates/treelite-builder/src/bulk.rs` — metadata assign (builder, CRUD) [MEM-02]

**Analog:** builder/lib.rs:768-774 (the assign block is byte-identical).

**Current** (verified 229-235): identical 7-line block to lib.rs:768-774, same line 235 `.unwrap_or_else(|| "{}".to_string())`.
**Change:** same as lib.rs — if `Metadata` is migrated, 229-234 stay; line 235 → `CompactString::from("{}")`.

---

### `crates/treelite-builder/src/concat.rs` — merge (builder, transform) [MEM-02]

**Analog:** self. Builds `Vec<i32>` locals then assigns + does `!=` comparisons.

**Current** (verified):
```rust
out.num_class = first.num_class.clone();          // 79  — .clone() works on SmallVec
out.base_scores = first.base_scores.clone();      // 84
out.attributes = first.attributes.clone();        // 85  — CompactString: Clone
let mut target_id: Vec<i32> = Vec::new();         // 90  — local
let mut class_id: Vec<i32> = Vec::new();          // 91
// ...
if m.num_class != out.num_class { /* HeaderMismatch */ }       // 113 — PartialEq works
if m.leaf_vector_shape != out.leaf_vector_shape { /* ... */ }  // 119
target_id.extend_from_slice(&m.target_id);        // 141 — &SmallVec derefs to &[i32]
class_id.extend_from_slice(&m.class_id);          // 142
out.target_id = target_id;                        // 149 — Vec → field
out.class_id = class_id;                          // 150
```
**MEM-02 change (minimal):** the `.clone()`, `!=`, and `extend_from_slice(&m.target_id)` calls all work
unchanged. Only the two `Vec<i32>` locals (90-91) + their assigns (149-150) need either `.into()` at
assignment OR change the local type to `SmallVec<[i32; 1]>`. The `debug_assert_eq!(out.target_id.len(), num_tree)`
(157-158) is unchanged.

---

### `crates/treelite-xgboost/src/lib.rs` — loader write (loader, CRUD-write) [MEM-02]

**Analog:** the loader builds `vec![]`/`.to_string()` literals into a `BuilderMetadata`. If `Metadata`
fields are migrated, add `.into()` / `smallvec![]` at the literal sites.

**Current literal sites** (verified):
```rust
vec![num_class_param],                  // 207  → SmallVec
vec![0; num_tree],                      // 208  → target_id
booster.tree_info.clone(),              // 209  → class_id (Vec from clone)
vec![1; num_target as usize],           // 222
let postprocessor = ...to_string();     // 193  → CompactString
// BuilderMetadata { num_class, leaf_vector_shape: vec![1, 1], target_id, class_id, postprocessor, base_scores, attributes: None } (248-260)
```
**Change:** if `Metadata` is SmallVec/CompactString, wrap the `Vec`-producing exprs (`vec![...]`,
`tree_info.clone()`, `parse_base_score(...)?`) with `.into()` at the struct literal, and `postprocessor` →
`get_postprocessor(objective)?.into()`. `leaf_vector_shape: vec![1, 1].into()` or `smallvec![1, 1]`.

---

### `crates/treelite-lightgbm/src/lib.rs` — loader write (loader, CRUD-write) [MEM-02]

**Analog:** xgboost/lib.rs (same `BuilderMetadata { ... }` literal shape).

**Current** (verified 342-364): `class_id`/`target_id`/`base_scores` built as `Vec<i32>`/`Vec<f64>` locals
(342-351), then `BuilderMetadata { num_class: vec![num_class], leaf_vector_shape: vec![1, 1],
postprocessor: postproc.postprocessor.to_string(), attributes: None, ... }`.
**Change:** `.into()` on the `Vec` locals/literals at the struct fields; `.to_string()` → `.into()` for postprocessor.

---

### `crates/treelite-sklearn/src/bulk.rs` — loader write (loader, CRUD-write) [MEM-02]

**Analog:** xgboost/lib.rs. Two `BuilderMetadata` literals (regressor 272-277, multiclass 366-371).

**Current** (verified): `num_class: vec![1; n_targets as usize]` / `n_classes.to_vec()`,
`leaf_vector_shape: vec![n_targets, 1]` / `vec![n_targets, max_num_class]`,
`postprocessor: "identity".to_string()` / `"identity_multiclass".to_string()`,
`base_scores: vec![0.0; ...]`.
**Change:** `.into()` on each `Vec`/`vec![]`; `.to_string()` → `.into()` for the two postprocessor literals.

---

### `crates/treelite-gtil/src/shape.rs` — read (service, read) — NO CHANGE, verify only [MEM-02 / D-11]

**Analog / proof:** self. GTIL reads metadata exclusively via slice/iter API that SmallVec shares:
```rust
model.leaf_vector_shape.first().copied().unwrap_or(1)   // 61
model.leaf_vector_shape.get(1).copied().unwrap_or(1)    // 62
model.num_class.iter().copied().max().unwrap_or(1)      // 77
```
> `.first()`, `.get()`, `.iter()` are all `Deref<[T]>` methods → unchanged by the SmallVec swap. No edit.
> Verification task: `cargo test -p treelite-gtil` stays green (D-11).

---

### `crates/treelite-py/src/model.rs` — binding (binding, read) — NO CHANGE, verify only [D-05]

**Analog / proof:** self, getters 49-85. The ONLY getters today are `num_tree`, `num_feature`,
`input_type`, `output_type` (verified) — **none of the 7 migrated fields is exposed**, so the D-05
"keep list/str" gate is forward-looking. Existing getters read `&self.inner.num_feature` directly and are
untouched.
```rust
#[getter]
fn num_feature(&self) -> i32 { self.inner.num_feature }   // 60-62 — unaffected
```
> Do NOT add metadata getters this phase (out of scope — "no new functionality", RESEARCH Open-Q2).
> Verification task: pytest suite + maturin abi3 wheel build stay green (D-05/D-11).

---

### `crates/treelite-harness/Cargo.toml` — allocator features (config) [MEM-03]

**Analog:** the existing GPU `[features]` block (rocm/cuda/wgpu) + `cubecl = { workspace = true, optional = true }`.
See Shared Patterns §Allocator isolation above for the exact excerpt. Add `jemalloc`/`mimalloc` non-default
mutually-exclusive features + the 3 optional allocator deps. Verify wheel isolation (Pitfall 3):
`cargo tree -p treelite-py --no-default-features --features cpu | grep -E "jemalloc|mimalloc"` must be empty.

---

### NEW `crates/treelite-harness/src/bin/memory_report.rs` (+ optional `src/memory.rs`) — report writer [MEM-03 / D-10]

**Analog:** `crates/treelite-harness/src/report.rs` (the Phase-7 `GPU_EQUIVALENCE_REPORT.md` writer) +
its caller `tests/gtil_matrix_gpu.rs:330-494`.

**Report-writer structure to copy** (`report.rs`):
- `render_markdown(rows, manifest, device_name) -> String` — builds a markdown table with `use std::fmt::Write as _;` + `writeln!(s, ...)`. Header carries provenance (device/rustc/os/arch from `Manifest`) and an **"Observational — NOT a CI gate"** banner (D-10, copy verbatim style from report.rs:200-204).
- `emit(rows, manifest, device_name, report_md_path) -> anyhow::Result<()>` (report.rs:281-305) — `std::fs::create_dir_all(parent)` then `std::fs::write(path, md)`; optional JSON sidecar via `render_json` (report.rs:256-275). Use `anyhow` (binary, per CLAUDE.md).

**Report-target / docs path pattern** (`tests/gtil_matrix_gpu.rs`):
```rust
#[ignore = "...run explicitly to regenerate docs/MEMORY_REPORT.md"]  // gtil_matrix_gpu.rs:330 style
// docs path helper (gtil_matrix_gpu.rs:111-115):
fn docs_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs").join(name)
}
// then: emit(&rows, &manifest, &device_name, &docs_path("MEMORY_REPORT.md"))?;  // :492-493
```
> `Manifest` (src/manifest.rs) is the existing provenance struct (os/arch/rustc/treelite) — reuse it for the
> report header. RESEARCH §MEM-03 columns: `Model | Allocator | peak resident (RSS) | bytes allocated |
> measurement method`, plus a `size_of::<Model>()` before/after row and a "1e-5 + golden green" attestation line.

**`#[global_allocator]` static** — the ONLY site (this bin's crate root):
```rust
#[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(all(feature = "jemalloc", feature = "mimalloc"))]
compile_error!("features `jemalloc` and `mimalloc` are mutually exclusive (D-07)");
```
**RSS sampling** (RESEARCH Pattern 4/5): jemalloc rows via `tikv_jemalloc_ctl::{epoch, stats}` —
`epoch::advance().unwrap()` BEFORE each `stats::allocated::read()` / `stats::resident::read()` (Pitfall 5).
mimalloc + system rows via `/proc/self/statm` (field 2 × page size). Be explicit per-row about the method.

---

### NEW `crates/treelite-core/tests/model_invariants.rs` — `!Send` + `size_of` sanity (test) [MEM-02 / A3 / Pitfall 2]

**Analog:** `crates/treelite-core/tests/tree_buf.rs` (a core integration test: `use treelite_core::...; #[test] fn ...`).

**Pattern to write** (no existing `!Send`/`size_of` assertion in the repo — this closes Wave-0 gap):
```rust
use treelite_core::Model;

// A3: Model stays !Send (the *const T in TreeBuf::Borrowed keeps it !Send;
// SmallVec/CompactString are Send-neutral). A compile-fail-style static check:
fn _assert_not_send() {
    fn requires_send<T: Send>() {}
    // requires_send::<Model>();  // ← must NOT compile; left commented as the documented invariant.
}

#[test]
fn model_size_not_bloated_by_smallvec() {
    // Pitfall 2: an over-large inline N bloats Model. Assert an upper bound that
    // the chosen N's stay under (planner sets the concrete byte budget).
    assert!(std::mem::size_of::<Model>() <= /* budget */ 512,
        "size_of::<Model>() = {}", std::mem::size_of::<Model>());
}
```
> The `!Send` half is best expressed as a `trybuild`-style compile-fail OR a documented commented assertion
> (the repo has no trybuild today; planner picks). The `size_of` half is a plain runtime `#[test]` — its
> budget is the guard against Pitfall 2 and feeds the `MEMORY_REPORT.md` before/after row.

---

## No Analog Found

None. Every touched site has an in-repo analog. The three "new" files are new *files*, not new *patterns* —
each copies an existing one (`report.rs`, the harness `[features]` block, `tests/tree_buf.rs`).

| File | Role | Data Flow | Note |
|------|------|-----------|------|
| (none) | — | — | The `!Send` compile-fail assertion is the only idiom with no exact repo precedent; expressed as a documented static check (no `trybuild` dep exists yet). |

---

## Metadata

**Analog search scope:** `crates/treelite-core/src/{tree_buf.rs, model.rs, serialize/mod.rs}`,
`crates/treelite-builder/src/{lib.rs, bulk.rs, concat.rs}`,
`crates/treelite-{xgboost,lightgbm,sklearn}/src/`, `crates/treelite-gtil/src/shape.rs`,
`crates/treelite-py/src/model.rs`,
`crates/treelite-harness/{Cargo.toml, src/report.rs, src/manifest.rs, tests/gtil_matrix_gpu.rs, tests/tree_buf.rs}` (core),
workspace `Cargo.toml`.
**Files scanned:** ~16 source files read directly (RESEARCH line numbers re-verified against live source).
**Pattern extraction date:** 2026-06-11
**Verified deviations from RESEARCH:** serialize/mod.rs deserialize assigns at 355-363 (RESEARCH said ~303-363);
builder assign at 768-774 (matches); concat locals at 90-91 / assigns 149-150 (matches). py getters confirmed:
only `num_tree`/`num_feature`/`input_type`/`output_type` exist (no metadata getter) — D-05 is forward-looking.

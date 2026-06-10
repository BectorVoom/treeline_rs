# Phase 6: cubecl GTIL Kernels (CPU Backend) - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 9 (1 new crate w/ 8 files + 1 new test, 4 modified)
**Analogs found:** 9 / 9 (all anchors verified against the live tree)

This phase is **registration-not-refactor (D-11)**. Every new/modified surface has a
concrete in-repo analog to copy line-for-line; almost nothing is invented. The cubecl
kernels are a *faithful re-expression* of the green scalar GTIL — not a redesign. Treat
`crates/treelite-gtil/src/lib.rs` and `postprocessor.rs` as the line-by-line spec.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-cubecl/Cargo.toml` (NEW) | config | — | `crates/treelite-gtil/Cargo.toml` | role-match |
| `crates/treelite-cubecl/src/lib.rs` (NEW) | service | request-response | `crates/treelite-gtil/src/lib.rs::predict` (lib.rs:892) | role-match |
| `crates/treelite-cubecl/src/upload.rs` (NEW) | service | transform / file-I/O (host→device) | `treelite-core/src/tree_buf.rs::as_slice` (tree_buf.rs:57) + RESEARCH upload example | partial |
| `crates/treelite-cubecl/src/kernels/traversal.rs` (NEW) | utility (#[cube] helper) | transform | `treelite-gtil/src/lib.rs::evaluate_tree` (lib.rs:432) + `next_node` (lib.rs:333) | role-match (verbatim port) |
| `crates/treelite-cubecl/src/kernels/postproc.rs` (NEW) | utility (#[cube] helper) | transform | `treelite-gtil/src/postprocessor.rs` (10 fns) | exact (verbatim port) |
| `crates/treelite-cubecl/src/kernels/{default_raw,leaf_id,score_per_tree}.rs` (NEW) | service (#[cube(launch)]) | batch / streaming | `treelite-gtil/src/lib.rs::predict_preset` (lib.rs:643) | role-match |
| `crates/treelite-cubecl/src/error.rs` (NEW) | model (error enum) | — | `treelite-gtil` `GtilError` (thiserror enum) | role-match |
| `crates/treelite-harness/src/lib.rs` (MOD) | service (registration) | request-response | `scalar_cpu_case()` (lib.rs:107) + `Backend` enum (lib.rs:53) | exact |
| `crates/treelite-harness/src/manifest.rs` (MOD) | model (provenance) | — | `Manifest.backend` field (manifest.rs:59) | exact |
| `crates/treelite-core/src/tree_buf.rs` (MOD) | model (accessor) | transform | `TreeBuf::as_slice` (tree_buf.rs:57) | exact |
| `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` (NEW) | test | batch | `crates/treelite-harness/tests/gtil_matrix.rs` | exact (reuse runner) |
| `Cargo.toml` (MOD) | config | — | existing `[workspace.dependencies]` block (Cargo.toml:18) | exact |

## Pattern Assignments

---

### `crates/treelite-harness/src/lib.rs` (MOD — backend registration)

**Analog:** `scalar_cpu_case()` at `crates/treelite-harness/src/lib.rs:107-134` and the
`Backend` enum at `lib.rs:53-60`. This is the single registration point; copy the slot
shape verbatim and swap the dense bodies to the cubecl entry point.

**Backend enum — add one variant** (`lib.rs:53-60`, verified):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    ScalarCpu,
    // Phase 6/7: CubeclCpu, Cuda, Wgpu, Rocm — added as registrations.
}
```
Delta: add `CubeclCpu,` (the comment at line 59 already names it as the planned variant).

**Four fn-pointer slot types are already declared** (`lib.rs:65-73`, verified) — reuse
unchanged:
```rust
pub type DensePredictF32Fn = fn(&Model, &[f32], usize, &Config) -> anyhow::Result<Vec<f64>>;
pub type DensePredictF64Fn = fn(&Model, &[f64], usize, &Config) -> anyhow::Result<Vec<f64>>;
pub type SparsePredictF32Fn = fn(&Model, SparseCsr<'_, f32>, usize, &Config) -> anyhow::Result<Vec<f64>>;
pub type SparsePredictF64Fn = fn(&Model, SparseCsr<'_, f64>, usize, &Config) -> anyhow::Result<Vec<f64>>;
```

**Core registration pattern** (`scalar_cpu_case`, `lib.rs:107-134`, verified) — the
exact template for `cubecl_cpu_case()`:
```rust
pub fn scalar_cpu_case() -> RunnerCase {
    RunnerCase {
        backend: Backend::ScalarCpu,
        dense_f32: |model, data, num_row, cfg| {
            // PREDICT runs in f32 (no pre-cast); widen the f32 RESULT to f64 after.
            let out = treelite_gtil::predict::<f32>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out.into_iter().map(|v| v as f64).collect())
        },
        dense_f64: |model, data, num_row, cfg| {
            let out = treelite_gtil::predict::<f64>(model, data, num_row, cfg)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(out)
        },
        sparse_f32: |model, csr, num_row, cfg| { /* predict_sparse::<f32> ... */ },
        sparse_f64: |model, csr, num_row, cfg| { /* predict_sparse::<f64> ... */ },
    }
}
```

**Minimal delta for `cubecl_cpu_case()`:**
- `backend: Backend::CubeclCpu`.
- `dense_f32` / `dense_f64`: call `treelite_cubecl::predict_cpu::<f32>` / `::<f64>` instead
  of `treelite_gtil::predict` (D-01, D-05 — both input×preset combos run in-kernel). Keep
  the **"widen the f32 result AFTER predict"** discipline verbatim (lib.rs:111-116) — never
  a pre-cast (Pitfall 6).
- `sparse_f32` / `sparse_f64`: keep them pointing at `treelite_gtil::predict_sparse` (D-02 —
  sparse rides the scalar fallback this phase, recorded as `scalar-fallback` provenance).
- The `map_err(|e| anyhow::anyhow!("{e}"))?` typed→anyhow bridge is the established pattern
  (lib.rs:115) — reuse it for `CubeclError`.

**Note:** The slot signatures and the `RunnerCase` struct (`lib.rs:89-101`) do NOT change.
Adding the case is purely additive — matrix iteration is untouched (D-11).

---

### `crates/treelite-harness/src/manifest.rs` (MOD — per-cell provenance, D-06)

**Analog:** the `backend` field at `crates/treelite-harness/src/manifest.rs:55-60` (verified),
already a per-cell defaulted string with a `default_backend()` helper at `manifest.rs:25-27`.

**Existing field** (`manifest.rs:59-60`):
```rust
#[serde(default = "default_backend")]
pub backend: String,
```
The field is ALREADY per-cell (every `MatrixGolden` carries its own `Manifest`). The Phase-5
default helper (`manifest.rs:25`) returns `"scalar-cpu"`.

**Minimal delta (D-06):** the `backend` field is already provenance-capable; Phase 6's work is
*recording* the right value per cell, not adding a field. Two options, both additive:
- Have the cubecl runner WRITE the executed-path tag (`"cubecl-kernel"` vs `"scalar-fallback"`)
  into the per-cell record/report when a model falls back (categorical/sparse → scalar).
- Optionally extend `check_manifest` (`manifest.rs:110-153`) — the existing backend-drift
  `eprintln!` at `manifest.rs:134-140` is the analog to copy if a `cubecl-cpu` vs
  `scalar-fallback` warning is wanted. **Never fail the gate** — this function only warns
  (manifest.rs:18-19 contract).

Because `backend` is `#[serde(default)]` (manifest.rs:59), no existing frozen manifest breaks.

---

### `crates/treelite-core/src/tree_buf.rs` (MOD — additive `as_bytes()`, SC3/GPU-05)

**Analog:** `TreeBuf::as_slice` at `crates/treelite-core/src/tree_buf.rs:57-65` (verified) —
the existing zero-copy view over a `T: Copy` POD enum.
```rust
pub fn as_slice(&self) -> &[T] {
    match self {
        TreeBuf::Owned(v) => v.as_slice(),
        TreeBuf::Borrowed { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
    }
}
```

**Minimal delta:** add one additive method in a NEW `impl` block (do not touch the `T: Copy`
enum bound — `tree_buf.rs:20`, and respect the module note at `tree_buf.rs:11-12` that the
broad `bytemuck::Pod` seam is deferred to Phase 9):
```rust
impl<T: Copy + bytemuck::Pod> TreeBuf<T> {
    /// Zero-copy byte view of the column for the cubecl SoA upload (GPU-05/SC3).
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
}
```
- Gate `as_bytes` behind `T: Pod` (a *narrower* second impl), keeping the primary `T: Copy`
  API intact. Only the numeric upload columns (`f32`/`f64`/`i32`/`u32`/`u64`) need it.
- `bool`/`TreeNodeType`/`Operator` columns (tree.rs:20,28,34,36) are NOT `Pod`-uploaded — they
  are materialized to a small `u32`/`i32` discriminant column on the host (Pitfall 4). So
  `as_bytes` is never needed on them.
- Requires `bytemuck` as a new dep on `treelite-core` (workspace-pinned).

---

### `crates/treelite-cubecl/src/kernels/traversal.rs` (NEW — `#[cube]` descent)

**Analog:** `evaluate_tree` at `crates/treelite-gtil/src/lib.rs:432-502` and `next_node` at
`lib.rs:333-355` (both verified). The kernel reproduces this control flow break-free.

**Scalar descent loop** (`lib.rs:449-501`, the line-by-line spec):
```rust
let mut nid: usize = 0;
while !tree.is_leaf(nid) {                          // cleft[nid] != -1
    let fi = tree.split_index(nid);                 // bounds-checked
    let fvalue = row[fi as usize];
    let next: i32 = if fvalue.is_nan_val() {
        tree.default_child(nid)                      // NaN → default child (predict.cc:158-159)
    } else if tree.node_type(nid) == TreeNodeType::kCategoricalTestNode {
        /* categorical → scalar fallback this phase (D-02) */
    } else {
        next_node(nid, fvalue.to_compare_f64(), tree.threshold(nid).threshold_to_f64(),
                  tree.comparison_op(nid), tree.left_child(nid), tree.right_child(nid))?
    };
    nid = next as usize;
}
Ok(nid)
```

**Comparison switch** (`next_node`, `lib.rs:341-354`): XGBoost always `kLT` →
`fvalue < threshold ? left : right` (lib.rs:342).

**Minimal delta to the `#[cube]` port (RESEARCH Pattern 1):**
- Mark the helper `#[cube]` (a plain Rust fn fails E0433).
- `while cleft[base+nid] != -1` — break-free, NO `continue` (already structurally true here).
- NaN routing: `F::is_nan(fv)` as an **associated fn**, never `fv.is_nan()` (Pitfall 1).
- The if/else routing must use `let mut next; if … { next = … }` **statements**, never an
  `if`-expression value (Pitfall 2 / RESEARCH anti-pattern).
- Add `base`/`row_off` offset indexing for the ragged-SoA concatenated columns (the scalar
  uses a per-tree `Tree<T>`; the kernel addresses `concat[tree_node_offset[t] + nid]`).
- Detect `node_type == kCategoricalTestNode` and route the WHOLE model to scalar fallback if
  any tree has a categorical split (Open Q1 recommendation) — `Tree::has_categorical_split`
  (tree.rs:70) is the host-side gate.

---

### `crates/treelite-cubecl/src/kernels/postproc.rs` (NEW — `#[cube]` postprocessors, D-03)

**Analog:** `crates/treelite-gtil/src/postprocessor.rs` (608 lines, verified) — all 10
postprocessors plus their `*_f64` twins. The cast order IS the 1e-5 contract; mirror each
exactly. Confirmed line anchors:

| Postprocessor | f32 fn | f64 twin | Cast-order note |
|---------------|--------|----------|-----------------|
| identity | `identity` :50 | — | no-op |
| identity_multiclass | `identity_multiclass` :61 | — | no-op |
| sigmoid | `sigmoid` :74 | `sigmoid_f64` :91 | `alpha` is f32, cast at multiply site |
| exponential | `exponential` :103 | `exponential_f64` :109 | `exp` in element width |
| exponential_standard_ratio | `exponential_standard_ratio` :124 | (twin ~:130) | **`exp2` base-2** (Pitfall 2 / A1) |
| logarithm_one_plus_exp | `logarithm_one_plus_exp` :146 | `_f64` :152 | — |
| softmax | `softmax` :175 | `softmax_f64` :227 | f32 `max_margin`/`t`/divisor + f64 `norm_const` (Pitfall 3) |
| signed_square | `signed_square` :267 | `_f64` :273 | `copysign` |
| hinge | `hinge` :285 | — | — |
| multiclass_ova | `multiclass_ova` :301 | `_f64` :312 | reuses sigmoid per cell |

**The two cast-order excerpts that MUST be reproduced verbatim in-kernel:**

`sigmoid` / `sigmoid_f64` (`postprocessor.rs:74-93`):
```rust
pub fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-sigmoid_alpha * v).exp())
}
pub fn sigmoid_f64(sigmoid_alpha: f32, v: f64) -> f64 {
    1.0_f64 / (1.0_f64 + (-(sigmoid_alpha as f64) * v).exp())   // alpha cast to f64 at multiply
}
```
In-kernel: `F::new(1.0) / (F::new(1.0) + F::exp(F::new(0.0) - alpha_F * v))` — `F::exp`, not
`.exp()` (Pitfall 1). Launch `F=f32` for f32 input, `F=f64` for f64 (D-05).

`softmax_f64` mixed-precision (`postprocessor.rs:227-251`, the CR-01 contract):
```rust
let mut max_margin: f32 = row[0] as f32;
for &x in &row[1..] { if x > max_margin as f64 { max_margin = x as f32; } }
let mut norm_const: f64 = 0.0;
for cell in row.iter_mut() {
    let t: f32 = (*cell - max_margin as f64).exp() as f32;  // f64 - f32 -> f64; exp in f64; narrow to f32
    norm_const += t as f64;
    *cell = t as f64;
}
let divisor: f64 = norm_const as f32 as f64;                // static_cast<float>(norm_const)
for cell in row.iter_mut() { *cell /= divisor; }            // f64 /= float
```
**Minimal delta:** keep f64 cells, introduce explicit `f32`-typed `max_margin`/`t`/divisor
locals at the EXACT sites the scalar twin does (mixed f32/f64 locals in one f64 kernel —
A2, the spike confirms this compiles on CpuRuntime). For `exp2` (Pitfall 2): try `F::exp2(x)`;
if absent in 0.10.0, use `F::exp(x * F::new(LN_2))` or `F::powf(F::new(2.0), x)` in the
element's own width — the spike exercises `exponential_standard_ratio` to lock this (A1).

---

### `crates/treelite-cubecl/src/kernels/{default_raw,leaf_id,score_per_tree}.rs` (NEW — `#[cube(launch)]`)

**Analog:** `predict_preset` at `crates/treelite-gtil/src/lib.rs:643-741` (verified) — the
accumulate→average→base-score assembly order, which IS the 1e-5 contract (lib.rs:622-642).

**Serial-tree accumulation core** (`lib.rs:661-675`, the SC1 shape):
```rust
for r in 0..num_row {
    rows.materialize(r, &mut scratch);
    for (tree_id, tree) in trees.iter().enumerate() {     // SERIAL over trees (GTIL-08)
        let leaf = evaluate_tree(tree, row)?;
        if has_leaf_vector(tree, leaf)? {
            output_leaf_vector(/* leaf-vector broadcast */)?;
        } else {
            output_leaf_value(/* scalar leaf into (target,class) cell */)?;
        }
    }
}
```
Base-score 2D f64 add (`lib.rs:728-738`) and RF averaging (`lib.rs:679-721`) follow.

**Minimal delta (RESEARCH Pattern 2):**
- One unit per row: `let row = ABSOLUTE_POS; if row < num_row { … }` (mandatory bounds check).
- The `for r in 0..num_row` outer loop becomes the launch grid (`CubeCount::Static((num_row+255)/256,1,1)`,
  `CubeDim{x:256,…}`); the `for tree_id` inner loop stays **serial in-kernel** (no reduce/atomic
  over the tree axis — Pitfall 6 / SC1).
- **Separate kernels per predict kind** (host-known from `Config.kind`, config.rs:13/45): branch
  on the host (kernel selection), not in-kernel. `default_raw` fuses traversal+accumulate(+postproc
  for `default`); `leaf_id` writes the leaf node id; `score_per_tree` writes raw per-tree leaf data.
- Output element type follows the INPUT dtype `F`, not the preset (Pitfall 6 / lib.rs:892 is
  `O`-generic). Thresholds/leaves read in the preset's `T`; kernel is generic over both.

---

### `crates/treelite-cubecl/src/upload.rs` (NEW — per-column ragged SoA upload, SC3/GPU-05)

**Analog:** `TreeBuf::as_slice` (tree_buf.rs:57) + the SoA `Tree<T>` columns (tree.rs:17-81,
verified) + the RESEARCH upload example. Each column is concatenated across trees into one
host `Vec`, byte-cast, and uploaded as ONE device handle per column with a parallel
`(offset, len)` index — no per-tree handle explosion (SC3 anti-pattern).

**Columns to upload** (from `Tree<T>`, tree.rs:18-52): `cleft`/`cright`/`split_index` (`i32`),
`threshold`/`leaf_value`/`leaf_vector` (`T`), `leaf_vector_begin`/`leaf_vector_end` (`u64`,
CSR offsets for multiclass broadcast), `default_left` → materialized as `u32` (bool, Pitfall 4),
`node_type` → materialized as `i32` discriminant (enum, Pitfall 4).

**Upload pattern** (per RESEARCH Code Example, derived from the zero-copy manuals):
```rust
let bytes = bytemuck::cast_slice::<F, u8>(&concat).to_vec();
let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(bytes));
```
**Minimal delta:** use the new `TreeBuf::as_bytes()` (above) per column; build a
`tree_node_offset` prefix-sum (`num_nodes` per tree, tree.rs:72) and `tree_leafvec_offset`
index so the kernel addresses tree `t`'s node `n`. Use `bytemuck::cast_slice` — never a
hand-rolled transmute (Don't-Hand-Roll). Validate `num_feature`/`num_row`/buffer lengths up
front (mirror `predict` at lib.rs:902-926) before any `client.create`/launch (V5 input
validation — no OOB device write).

---

### `crates/treelite-cubecl/src/lib.rs` + `error.rs` + `Cargo.toml` (NEW crate scaffold)

**Analog:** `crates/treelite-gtil` crate layout (Cargo.toml + lib.rs `pub mod` exports +
typed `thiserror` error enum). The host entry `predict_cpu::<F>` mirrors `treelite_gtil::predict`'s
signature shape (lib.rs:892): `(&Model, &[F], usize, &Config) -> Result<Vec<F>, CubeclError>`.

**Minimal delta:**
- `Cargo.toml`: `cubecl = { workspace = true }`, `bytemuck = { workspace = true }`,
  `treelite-core = { path }`, `treelite-gtil = { path }` (for the fallback), `thiserror = { workspace = true }`.
- `error.rs`: a `CubeclError` `thiserror` enum (mirror `GtilError` discipline — no panic/OOB
  in a library crate, CLAUDE.md error contract). Bridge to `anyhow` only at the harness slot.
- `lib.rs`: `pub fn predict_cpu::<F>` host launcher (validate → upload → select kernel by
  `Config.kind` → launch::<F, CpuRuntime> → `read` → `bytemuck::cast_slice` read-back); route
  to `treelite_gtil::predict_sparse` / scalar fallback for sparse + categorical (D-02).

---

### `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` (NEW — reuse the matrix runner)

**Analog:** `crates/treelite-harness/tests/gtil_matrix.rs` (verified). Per D-11 the matrix
iteration is reused UNCHANGED; the only delta is the `RunnerCase` it dispatches.

**The dispatch the new test reuses** (`gtil_matrix.rs:300-383`, `run_cell`): branches on
`manifest.input_dtype` (f32/f64) and `manifest.layout` (dense/sparse), calling
`(case.dense_f32)` / `(case.dense_f64)` / `(case.sparse_f32)` / `(case.sparse_f64)`
(gtil_matrix.rs:328/338/357/365) — with NO f32→f64 pre-cast (the input-dtype axis, Pitfall 1).

**Case construction** (`gtil_matrix.rs:425`):
```rust
let case = scalar_cpu_case();
```
**Minimal delta (RESEARCH Wave 0 Gap recommendation):** add a SIBLING test file (or a
`cases: &[RunnerCase]` loop) that constructs `let case = cubecl_cpu_case();` and runs the
identical iteration body against the SAME frozen `fixtures/gtil/` goldens. Do NOT restructure
`gtil_matrix()` — if the existing body would need reshaping, that is a D-11 smell; add the thin
sibling instead. Add the SC2 two-run bit-identity check (`out_a.to_bits() == out_b.to_bits()`)
and assert per-cell provenance (D-06).

**Caution:** the existing test hard-asserts `golden.manifest.backend == "scalar-cpu"`
(gtil_matrix.rs:474) — the cubecl sibling must assert its own backend/provenance, NOT copy that
literal.

---

## Shared Patterns

### Typed-error → anyhow bridge at the harness boundary
**Source:** `crates/treelite-harness/src/lib.rs:115` — `.map_err(|e| anyhow::anyhow!("{e}"))?`
**Apply to:** every `cubecl_cpu_case()` fn-pointer body (CubeclError → anyhow). Library crates
(`treelite-cubecl`) stay `thiserror`; only the harness uses `anyhow` (CLAUDE.md ERR-02).

### "Widen the result AFTER predict, never pre-cast" (Pitfall 1/6)
**Source:** `lib.rs:111-116` — predict runs in the input width; `out.into_iter().map(|v| v as f64)`
widens the *result*.
**Apply to:** `cubecl_cpu_case().dense_f32` (and the kernel output read-back). A pre-cast erases
the input-dtype axis the instrument exists to verify.

### Up-front shape validation before any unsafe/device op (V5 input validation)
**Source:** `crates/treelite-gtil/src/lib.rs:902-926` (`predict`'s `num_feature`/buffer-length
guards returning typed `InvalidInputShape`).
**Apply to:** `treelite-cubecl` host launcher + `upload.rs` — validate lengths BEFORE
`client.create`/`launch`/`ArrayArg::from_raw_parts`, never an OOB device write.

### Verbatim mixed-precision cast order (the 1e-5 contract)
**Source:** `crates/treelite-gtil/src/postprocessor.rs` (the `*_f64` twins + softmax f32/f64
split) and `predict_preset` assembly order (lib.rs:622-642).
**Apply to:** every `#[cube]` postprocessor + the accumulate/base-score kernel. Reproduce, never
"simplify" to all-f32 or all-f64 (CR-01).

### Additive, back-compat-preserving extension (no refactor)
**Source:** `manifest.rs` `#[serde(default)]` discipline (manifest.rs:41-100); `Backend` enum's
reserved-variant comment (lib.rs:59).
**Apply to:** the `Backend::CubeclCpu` variant, the `TreeBuf::as_bytes` second impl, and the
sibling test — all purely additive (D-11).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (cubecl `#[cube(launch)]` launch glue / `CubeCount`/`CubeDim` host setup) | service | batch | No cubecl code exists anywhere in-repo yet; the kernel-launch boilerplate (grid sizing, `ArrayArg::from_raw_parts`, `client.read`) has NO in-repo analog — derive it from the vendored cubecl manual (`cubecl_manual/manual/Cubecl/Cubecl_multi_threading.md`, `Cubecl_generics.md`) and lock it in the mandatory spike. The DATA the glue moves and the MATH it runs both have exact analogs above; only the cubecl plumbing is new. |

## Metadata

**Analog search scope:** `crates/treelite-harness/{src,tests}`, `crates/treelite-gtil/src`,
`crates/treelite-core/src`, root `Cargo.toml`.
**Files scanned:** lib.rs, manifest.rs, tree_buf.rs, tree.rs, postprocessor.rs (gtil),
gtil_matrix.rs, config.rs/shape.rs/accessor.rs (signature confirmation), Cargo.toml.
**All file:line anchors verified against the live tree** (no anchor recorded unread).
**Pattern extraction date:** 2026-06-10

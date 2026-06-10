---
phase: 06-cubecl-gtil-kernels-cpu-backend
reviewed: 2026-06-10T00:00:00Z
depth: standard
files_reviewed: 20
files_reviewed_list:
  - crates/treelite-core/Cargo.toml
  - crates/treelite-core/src/tree_buf.rs
  - crates/treelite-cubecl/Cargo.toml
  - crates/treelite-cubecl/src/error.rs
  - crates/treelite-cubecl/src/kernels/default_raw.rs
  - crates/treelite-cubecl/src/kernels/leaf_id.rs
  - crates/treelite-cubecl/src/kernels/mod.rs
  - crates/treelite-cubecl/src/kernels/postproc.rs
  - crates/treelite-cubecl/src/kernels/score_per_tree.rs
  - crates/treelite-cubecl/src/kernels/traversal.rs
  - crates/treelite-cubecl/src/lib.rs
  - crates/treelite-cubecl/src/upload.rs
  - crates/treelite-cubecl/tests/determinism.rs
  - crates/treelite-cubecl/tests/postproc.rs
  - crates/treelite-cubecl/tests/predict_kinds.rs
  - crates/treelite-cubecl/tests/spike.rs
  - crates/treelite-cubecl/tests/upload.rs
  - crates/treelite-harness/Cargo.toml
  - crates/treelite-harness/src/lib.rs
  - crates/treelite-harness/tests/gtil_matrix_cubecl.rs
findings:
  critical: 3
  warning: 5
  info: 3
  total: 11
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-06-10
**Depth:** standard
**Files Reviewed:** 20
**Status:** issues_found

## Summary

This phase ports the GTIL numerical-dense inference hot path to cubecl CPU kernels
(`descend` traversal + four predict-kind launch kernels + the host launcher
`predict_cpu`), with a categorical/sparse scalar-fallback gate. The project's core
value is 1e-5 numerical equivalence to upstream Treelite, so correctness of the
traversal routing, mixed-precision cast order, and the fallback gate were the focus.

Two correctness defects in the traversal break the 1e-5 contract for whole classes
of model that the cubecl path silently accepts: the `descend` helper hardcodes the
`kLT` comparison operator (so any `kLE`/`kGE`/etc. model — every LightGBM model —
routes ties the wrong way), and it compares the threshold in the *input* width `F`
rather than promoting both operands to `f64` like the scalar reference (so an
`f32`-input / `f64`-preset model can route differently near a boundary). Neither is
caught by the test matrix because the committed fixtures are XGBoost-derived
(all-`kLT`) and the mixed-width fixtures apparently don't land a threshold on an
`f32`-unrepresentable boundary. A third blocker: the leaf-vector accumulation loops
read the device buffer with no bounds check, so a malformed model OOB-reads on the
device (the scalar path returns a typed error here) — undermining the T-06-09
"no OOB device write/read on a malformed model" contract.

The postprocessor `#[cube]` ports (postproc.rs) faithfully reproduce the scalar
mixed-width cast order, but they are NOT wired into the production `predict_cpu`
path (which applies the postprocessor on the host via `F::apply_postprocessor`) —
they are exercised only by `tests/postproc.rs`. That is acceptable but means the
"device postprocessor" claim in several doc comments overstates what ships.

## Critical Issues

### CR-01: `descend` hardcodes `kLT`; non-`kLT` models (all LightGBM) route ties wrong

**File:** `crates/treelite-cubecl/src/kernels/traversal.rs:94-102`, gated by `crates/treelite-cubecl/src/lib.rs:262-279`
**Issue:** The kernel descent unconditionally computes `fv < threshold` (strict
less-than, `kLT`). The scalar reference `evaluate_tree` (`treelite-gtil/src/lib.rs:481-489`)
reads the per-node operator via `tree.comparison_op(nid)` and dispatches across
`kLT`/`kLE`/`kEQ`/`kGT`/`kGE` (erroring on `kNone`). LightGBM models are loaded with
`Operator::kLE` for *every* split (`treelite-lightgbm/src/lib.rs:273`: "LightGBM
always uses <="). `predict_cpu`'s fallback gate (`lib.rs:262-269`) only routes
*categorical* models to the scalar engine — it never inspects the comparison
operator. So a numerical-dense LightGBM (or any `kLE`/`kGE` sklearn) model is sent
to the kernel, where every `fvalue == threshold` boundary case routes RIGHT
(`kLT` false) instead of LEFT (`kLE` true) — a definite wrong prediction, not a
1e-5 rounding drift. The committed fixtures are XGBoost-only (all-`kLT`), so the
matrix sibling never exercises this and the gate passes green while the bug ships.
**Fix:** Either (a) gate on the operator the same way categorical splits are gated —
route the whole model to the scalar fallback unless every node's `cmp` is `kLT`:
```rust
let all_klt = match &model.variant {
    ModelVariant::F32(p) => p.trees.iter().all(|t|
        t.cmp.as_slice().iter().zip(t.cleft.as_slice())
            .all(|(&op, &cl)| cl == -1 || op == Operator::kLT)),
    ModelVariant::F64(p) => /* same */,
};
if has_categorical || !all_klt {
    return treelite_gtil::predict::<F>(model, data, num_row, cfg)
        .map_err(|e| CubeclError::Unsupported(format!("scalar fallback: {e}")));
}
```
or (b) upload the per-node operator discriminant column (it is already materialized
host-side as `cmp`, though not currently uploaded) and branch on it in `descend`.
Option (a) is the smaller, D-02-consistent change. Add a `kLE` fixture (or a
LightGBM model) to `fixtures/gtil/` so the matrix actually covers it.

### CR-02: threshold compared in input width `F`, not promoted to f64 (mixed-width routing divergence)

**File:** `crates/treelite-cubecl/src/kernels/traversal.rs:99`
**Issue:** The kernel compares `fv < F::cast_from(threshold[...])`, i.e. it casts the
threshold (width `T`) *down into the input width `F`* and compares in `F`. The scalar
reference promotes BOTH operands to `f64` before comparing
(`evaluate_tree` → `next_node(.., fvalue.to_compare_f64(), threshold.threshold_to_f64(), ..)`,
`treelite-gtil/src/lib.rs:481-484`, `326-331`). For the mixed-width combination
`F = f32` over a `T = f64` preset (which is a *supported* combination — Pitfall 6 says
the output element equals the input element regardless of preset, and `predict_cpu::<f32>`
over `ModelVariant::F64` is reachable, e.g. the `binary.f32.f64.*` fixtures), the
kernel does `f64 -> f32` on the threshold (LOSSY) and compares in f32, while the
scalar promotes the f32 input up to f64 and compares against the exact f64 threshold.
Near a threshold whose f64 value is not f32-representable these route differently —
a wrong leaf, not a 1e-5 drift. The traversal.rs doc comment (lines 56-59) claims this
reproduces `NextNode<InputT, ThresholdT>` usual-arithmetic promotion, but that
promotion widens to the *wider* type (f64), never narrows to the input — the comment
is incorrect and the code follows the incorrect comment.
**Fix:** Compare in f64 like the scalar reference:
```rust
// Promote BOTH operands to f64 (order-preserving for f32/f64), matching
// next_node's fvalue.to_compare_f64() < threshold.threshold_to_f64().
if f64::cast_from(fv) < f64::cast_from(threshold[(base + nid) as usize]) {
    next = cleft[(base + nid) as usize];
}
```
Add a mixed-width fixture whose f64 threshold falls between two adjacent f32 values
(e.g. `0.1_f64` rounded vs not) to lock this down — the current `binary.f32.f64`
fixtures evidently don't, since the gate is green.

### CR-03: leaf-vector kernels read the device buffer with no bounds check (OOB device read on malformed model)

**File:** `crates/treelite-cubecl/src/kernels/default_raw.rs:115-156`, `crates/treelite-cubecl/src/kernels/score_per_tree.rs:69-87`
**Issue:** The `OutputLeafVector` broadcast loops index
`leaf_vector[(lv_base + li) as usize]` (and `+ t`, `+ c`) with no check that the
index is within the uploaded `leaf_vector` column. The scalar twin `output_leaf_vector`
(`treelite-gtil/src/lib.rs:801-868`) bounds-checks every leaf-vector access and
returns `GtilError::LeafVectorTooShort` on a short vector. `validate_shape`
(`upload.rs:224-262`) only validates `split_index` and the input-buffer length — it
never validates that each leaf's `leaf_vector_begin/end` span lies within the tree's
leaf-vector segment, nor that a `(num_target, max_num_class)` broadcast fits the
leaf vector. A malformed `Model` (short leaf vector, or `begin/end` past the segment)
therefore performs an out-of-bounds *device* read inside the kernel — exactly the
T-06-09/T-06-06 "no OOB device op on a malformed model" contract this phase claims to
uphold (`lib.rs:241-247` doc, `error.rs:6-8`). On the CPU backend this is a read into
adjacent device memory (silent wrong value); on a future GPU backend it is undefined
behavior.
**Fix:** Validate leaf-vector spans host-side in `validate_shape` before upload —
for every leaf node assert `leaf_vector_end[n] <= tree_leaf_vector_len` and, for the
broadcast routes, that the span covers the addressed `(target, class)` cells; return
a typed `CubeclError` (add a `LeafVectorTooShort`/`MalformedLeafVector` variant
mirroring `GtilError::LeafVectorTooShort`). The kernel cannot itself error, so the
guarantee must be established on the host before launch.

## Warnings

### WR-01: RF average divisor cast widens through f64 for f32 cells (cast-order mismatch vs scalar)

**File:** `crates/treelite-cubecl/src/kernels/default_raw.rs:177-183`
**Issue:** The kernel divides `output[cell] /= F::cast_from(factor)` where `factor`
is `f64` (the host builds `average_factor: Vec<f64>`). For `F = f32` this is
`(count as f64) as f32`. The scalar `div_by_count` for f32 is `self / factor as f32`
i.e. `count as usize as f32` — a *direct* `usize -> f32`. For integer tree counts
within the f32 exact-integer range (< 2^24) the two casts produce identical f32
divisors, so this is not currently a 1e-5 break, but it is a needless deviation from
the documented "matches `O::div_by_count`" claim (line 182) and would diverge for a
forest with > 16.7M trees routed to one cell. Prefer building the divisor in the
output width, or document that the f64→f32 round-trip is exact for the supported
tree-count range.
**Fix:** Pass `average_factor` as integer counts and cast `count as F` in-kernel, or
note the exact-integer-range assumption explicitly and add a guard/test.

### WR-02: uploaded `node_type` and `cmp` columns are never consumed by any kernel (dead device traffic + missing operator data)

**File:** `crates/treelite-cubecl/src/upload.rs:154,294`; `crates/treelite-cubecl/src/kernels/*`
**Issue:** `concat_columns`/`upload_forest` materialize and upload the `node_type`
i32-discriminant column (`upload.rs:154,294`) "so a kernel can detect
`kCategoricalTestNode` and route to fallback" (module doc lines 26-27), but no kernel
reads it — the categorical decision is made entirely host-side in `predict_cpu`
(`lib.rs:262-269`). So `node_type` is uploaded on every launch and never used (a
wasted device allocation + copy). Meanwhile the `cmp` (operator) column is NOT
uploaded at all, which is the root enabler of CR-01. Either wire `node_type`/`cmp`
into the kernel or stop uploading `node_type`.
**Fix:** Remove the `node_type` upload from `UploadedForest`/`upload_forest` if the
gate stays host-side (it is cheap to recompute on the host), or — if CR-01 is fixed
via in-kernel operator dispatch — upload `cmp` and drop the unused `node_type`.

### WR-03: `read_one_unchecked` byte length never validated before `cast_slice` to `Vec<F>`

**File:** `crates/treelite-cubecl/src/lib.rs:428-429,482-483,553-554`
**Issue:** Each launch path reads back the output via
`client.read_one_unchecked(h_out)` then `bytemuck::cast_slice::<u8, F>(&bytes).to_vec()`.
If the returned byte buffer length is ever not a multiple of `size_of::<F>()` (a
runtime/driver contract violation), `bytemuck::cast_slice` panics rather than
returning a typed `CubeclError`. This crate's contract (`error.rs:6-8`) is "never a
`panic!`". The output handle was sized by the host, so this is unlikely on the CPU
backend, but the unchecked read + panicking cast is a latent panic path that
contradicts the stated error discipline.
**Fix:** Use `bytemuck::try_cast_slice` and map the error to
`CubeclError::Unsupported`, or assert-with-typed-error that
`bytes.len() == zero_out.len() * size_of::<F>()` before the cast.

### WR-04: `predict_cpu` re-creates a fresh `CpuRuntime` client on every call

**File:** `crates/treelite-cubecl/src/lib.rs:271`
**Issue:** `let client = CpuRuntime::client(&Default::default());` runs on every
`predict_cpu` invocation. For the categorical fallback path the client is created
(line 271) and then never used (the fallback returns at 266-269 *before* line 271 —
actually the client is created after the gate, so this is fine), but for repeated
predictions each call spins up a new client/context. This is a correctness-adjacent
robustness concern (resource churn), out of the v1 performance scope, but worth
noting the client is constructed unconditionally even though some paths could reuse
a cached one. Not a blocker.
**Fix:** Out of v1 perf scope; consider a caller-provided or lazily-cached client in
a later phase. No change required for correctness.

### WR-05: `multiclass_ova` / `softmax` device kernels in `postproc.rs` are not wired into `predict_cpu`

**File:** `crates/treelite-cubecl/src/kernels/postproc.rs` (whole module) vs `crates/treelite-cubecl/src/lib.rs:354-356`
**Issue:** The production postprocessor is applied on the HOST via
`F::apply_postprocessor` (`lib.rs:354-356`, calling the scalar `treelite_gtil::postprocessor::*`).
The `#[cube]` postprocessor ports in `postproc.rs` (and their `*_kernel` launch
wrappers) are exercised ONLY by `tests/postproc.rs`; they never run in `predict_cpu`.
Several doc comments (`default_raw.rs:16-19`, `lib.rs:23-27`, `mod.rs:17-20`) describe
the postprocessor as "a separate device step selected host-side" — but it is in fact
a host CPU step, and the device ports are effectively unused production code. This is
not a correctness bug (the host path matches the scalar reference exactly), but the
comments overstate device coverage and the device postproc kernels are dead weight in
the shipped library.
**Fix:** Either wire the device postproc kernels into `predict_cpu` (and delete the
host `apply_postprocessor` duplication), or relabel them as test-only fixtures and
correct the "separate device step" comments to say "host CPU step".

## Info

### IN-01: `traversal.rs` comment claims `NextNode` promotion that the code contradicts

**File:** `crates/treelite-cubecl/src/kernels/traversal.rs:56-59,95-98`
**Issue:** The doc claims casting the threshold into `F` "reproduces
`NextNode<InputT, ThresholdT>`'s usual-arithmetic-conversion promotion". Usual
arithmetic conversion promotes to the *wider* operand (f64), never narrows to f32 —
the comment is the inverse of C++ semantics and rationalizes CR-02. Correct the
comment when fixing CR-02.
**Fix:** Replace with "both operands promote to f64 (the wider type)".

### IN-02: `node_type` discriminant cast relies on enum repr never asserted in this crate

**File:** `crates/treelite-cubecl/src/upload.rs:154`
**Issue:** `n as i32` on `TreeNodeType` assumes the enum's discriminant values match
what the (unused) consumer would expect (the upload test asserts `kNumericalTestNode=1`,
`kLeafNode=0`). Since the column is unused (WR-02) this is harmless today, but if a
kernel ever reads it the magic discriminant mapping is undocumented at the cast site.
**Fix:** If kept, add an explicit `#[repr(i32)]` reference / comment at the cast.

### IN-03: duplicated `split_tree` / `scalar_model` fixtures across three test files

**File:** `crates/treelite-cubecl/tests/predict_kinds.rs:31-49`, `crates/treelite-cubecl/tests/determinism.rs:24-42`, `crates/treelite-cubecl/tests/spike.rs:181-202`
**Issue:** The `split_tree` and `scalar_model` builders are copy-pasted across the
cubecl test files (and `tests/upload.rs`). Acceptable for test isolation, but a small
shared `tests/common/` module would reduce drift risk (a fixture change must currently
be made in 3-4 places to stay consistent).
**Fix:** Optional: extract to a shared test helper module. Not required.

---

_Reviewed: 2026-06-10_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

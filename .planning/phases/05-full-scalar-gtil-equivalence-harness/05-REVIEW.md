---
phase: 05-full-scalar-gtil-equivalence-harness
reviewed: 2026-06-10T09:28:01Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - crates/treelite-gtil/src/lib.rs
  - crates/treelite-gtil/src/postprocessor.rs
  - crates/treelite-gtil/src/shape.rs
  - crates/treelite-gtil/src/error.rs
  - crates/treelite-gtil/tests/predict.rs
  - crates/treelite-gtil/tests/predict_kinds.rs
  - crates/treelite-harness/tests/gtil_matrix.rs
  - fixtures/capture_gtil_matrix.py
  - fixtures/capture_gtil_models.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-10T09:28:01Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

This is a gap-closure cycle (commits `e366dea..HEAD`, plans 05-06 + 05-07) that
landed the f64-input postprocessor twins (CR-01), the malformed-Model bounds
guards (ERR-01 / WR-01..05), and the exhaustive GTIL equivalence-matrix runner.
The ERR-01 bounds-check surface (`evaluate_tree`, `category_list_safe`,
`has_leaf_vector`, `SparseCsr::validate`, the negative-`num_feature` guards) is
thorough, well-tested, and faithfully translates each upstream unchecked access
into a typed `GtilError` — no panics, no silent mis-prediction on the malformed
inputs the tests construct. The CR-01 non-softmax f64 twins
(`sigmoid_f64`/`exponential_f64`/etc.) are correct and genuinely run in f64
(verified against upstream `postprocessor.cc`).

The review surfaced one **BLOCKER**: the f64 `softmax` arm violates the exact
cast-ordering contract the module claims to uphold — and it does so in the same
class of defect CR-01 was opened for. It narrows the entire output row to f32
*before* the `row[i] - max_margin` subtraction and the `std::exp`, whereas
upstream `softmax<double>` keeps `row[i]` in double for both. This is reachable
today through the committed `leaf_vec_mc.f32.f64.default.*` fixtures and produces
a ~9e-8 deviation from upstream — under the 1e-5 gate purely by margin, not by
contract.

A secondary structural concern (WARNING): the 1e-5 matrix gate does **not**
actually catch the CR-01-class regression for the `large_margin` f64 cells (the
buggy collapsed-f32 path deviates from the f64 golden by only ~6e-8, well inside
1e-5). The only real regression guard is the WR-06 `max_div > 0.0`
bit-inequality assertion, which is directionally correct but fragile. The runner
comments overstate what the 1e-5 gate proves.

## Critical Issues

### CR-01: f64 `softmax` narrows the row to f32 before the subtraction/exp — diverges from upstream `softmax<double>`

**File:** `crates/treelite-gtil/src/lib.rs:1369-1387` (contract docs at `crates/treelite-gtil/src/postprocessor.rs:32-35` and `crates/treelite-gtil/src/lib.rs:126-133`)

**Issue:**
The f64 postprocessor arm handles `softmax` by collapsing the whole output row to
`Vec<f32>` (`output[start..end].iter().map(|&v| v as f32)`), running the f32
`postprocessor::softmax`, and widening back. The module asserts this is correct
because "softmax hardcodes `float` for every `InputT`."

That premise is wrong. Upstream `softmax` (`postprocessor.cc:57-75`) is templated
on `InputT` and operates on the `InputT*` row in place. For the `double`
instantiation (`ApplyPostProcessor<double>`, `predict.cc:307-323`, verified in
the vendored tree) only `max_margin`, the temporary `t`, and the final
`static_cast<float>(norm_const)` divisor are `float`. The cell reads `row[i]`
stay **double**, so:

- `t = std::exp(row[i] - max_margin)` computes `double - float = double`, then
  `std::exp(double)` in **double**, narrowing only the result to `float t`;
- `row[i] /= static_cast<float>(norm_const)` is `double /= float` — a **double**
  divide.

The Rust port narrows `row[i]` to f32 up front, so the subtraction and `exp` run
entirely in f32. This is exactly the "collapse the f64 buffer to f32 before the
postprocessor" defect CR-01 was opened to fix — just for softmax instead of
sigmoid.

Verified numerically against the committed `leaf_vec_mc.f32.f64.raw.dense.s1234`
margins (4-class softprob, the exact rows the f64 softmax fixtures feed): the
narrow-to-f32 path deviates from the upstream `softmax<double>` ordering by up to
**~9.0e-8** across the row set (and ~2.8e-8 on a synthetic near-tie row). It is
under 1e-5 here only by margin; a closer-spaced or larger-magnitude multiclass
row (a deeper softprob booster, a different seed) can push this past the
project's 1e-5 core-value contract. The `leaf_vec_mc.f32.f64.default.*` cells
exercise this path today, so the matrix runner is asserting a non-faithful
computation that merely happens to fit under the tolerance.

**Fix:** Implement an `InputT == double` softmax that keeps the cell in f64 for
the subtraction and `exp`, mirroring the upstream float/double split exactly. Add
a `softmax_f64` to `postprocessor.rs`:

```rust
/// `softmax<double>` (postprocessor.cc:57-75): row cells stay f64 for the
/// `row[i] - max_margin` subtraction and `std::exp`; only max_margin, t, and the
/// final divisor are f32 (matching the upstream template body literally).
pub fn softmax_f64(row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    let mut max_margin: f32 = row[0] as f32;
    for &x in row.iter().skip(1) {
        if (x as f32) > max_margin {
            max_margin = x as f32;
        }
    }
    let mut norm_const: f64 = 0.0;
    for cell in row.iter_mut() {
        // double - float -> double; std::exp in double; narrow to f32 t.
        let t: f32 = (*cell - max_margin as f64).exp() as f32;
        norm_const += t as f64;
        *cell = t as f64;
    }
    let divisor = norm_const as f32 as f64; // static_cast<float>(norm_const)
    for cell in row.iter_mut() {
        *cell /= divisor; // double /= float
    }
}
```

Then call `softmax_f64` directly in the f64 arm instead of the narrow/widen
dance, and correct the contract docs in both files (softmax is **not** uniformly
f32-correct on every `InputT`; only `max_margin`/`t`/divisor are float). Add a
test mirroring `f64_twins_*` that proves `softmax_f64` diverges from
`softmax(narrowed)` on a double-precision near-tie row.

## Warnings

### WR-01: The 1e-5 matrix gate cannot catch the CR-01 regression it claims to measure

**File:** `crates/treelite-harness/tests/gtil_matrix.rs:540-594`

**Issue:**
The runner's comments and the `large_margin_f64_cells` coverage gate present the
1e-5 golden assert as the guard that "would have caught the pre-05-06
collapse-to-f32." It would not. Verified against the committed
`large_margin.f32.f64.default.dense.s1234` golden: the buggy collapsed-f32
sigmoid path deviates from the f64 golden by only **~6.0e-8** — comfortably
inside the 1e-5 epsilon. So both the corrected `sigmoid_f64` path and the old
buggy path pass the 1e-5 gate identically; the gate is real but blind to this
regression class. The actual guard is the separate WR-06 `max_div > 0.0`
assertion. The eprintln at :551-554 and the gate message at :585-594 overstate
what the 1e-5 assert proves.

**Fix:** Reword the CR-01 comments to state plainly that the 1e-5 gate confirms
the f64 path *matches* upstream but does **not** by itself reject the collapsed
path (the divergence is sub-1e-5); the regression guard is WR-06's paired
divergence. Optionally add an explicit dual assert on the `large_margin` f64
*default* cells ("matches the f64 golden to < 1e-7 AND the synthetic collapsed-f32
recompute does not") so the file documents the true mechanism.

### WR-02: WR-06 divergence guard is a strict bit-inequality, not a magnitude floor — fragile against future fixtures

**File:** `crates/treelite-harness/tests/gtil_matrix.rs:613-637`

**Issue:**
The WR-06 paired f32-vs-f64 guard asserts `max_div > 0.0` — any single bit of
difference passes. For the current `large_margin` fixtures this is satisfied on
all 140 rows (verified), and a future f32->f64 input pre-cast would collapse the
two computations and make them bit-identical, failing the assert — so the
*direction* is correct. But the worst genuine divergence between the two paths is
~2.6e-17 absolute on the saturated tails (sigmoid output ~1.9e-9). A guard that
trips on a single ULP is brittle: a future fixture whose margins land where the
f32 and f64 sigmoid round to the same f64 bits on *every* row would silently
pass `wr06_checked` while contributing zero real signal, masking a true collapse
on the cells that matter.

**Fix:** Replace `max_div > 0.0` with a relative-divergence floor anchored to the
expected f32-vs-f64 separation (e.g. require `max_rel > 1e-9` on at least one
row, mirroring the in-crate `sigmoid_f64_diverges_*` test at
`postprocessor.rs:412-420`), and assert that floor is met *per shared-axis pair*,
not just that some non-zero bit differs.

### WR-03: f64 softmax final `static_cast<float>(norm_const)` divide is also collapsed to f32

**File:** `crates/treelite-gtil/src/lib.rs:1380-1384`

**Issue:**
Subordinate to CR-01 but worth flagging independently: even setting aside the
subtraction/exp, the final divide in the f64 arm runs in f32 (the whole `tmp`
vector is f32, so `*cell /= divisor` is an f32 divide), then the f32 quotient is
widened back to f64. Upstream `row[i] /= static_cast<float>(norm_const)` with a
`double* row` is a `double /= float` — the quotient is computed in **double**.
The CR-01 fix above resolves this, but if softmax is left as-is this is a second,
separate ULP-shifting divergence on every f64 softmax cell.

**Fix:** Covered by the CR-01 `softmax_f64` fix (its final divide is
`*cell /= divisor` with `*cell: f64` and `divisor: f64` derived from
`norm_const as f32 as f64`).

### WR-04: `predict_score_by_tree_preset` reports a misleading `needed`/`got` on leaf-vector overflow

**File:** `crates/treelite-gtil/src/lib.rs:1183-1191`

**Issue:**
In the ScorePerTree leaf-vector arm the bounds check is `if i >= lvs` (per
element), but the error payload reports `needed: leafvec.len(), got: lvs`. When a
leaf vector is longer than the model's declared `lvs`, the loop errors on the
first `i == lvs`, yet `needed` is the *full* `leafvec.len()` rather than the
index that actually overflowed. This is a diagnostic inaccuracy, not a
memory-safety bug (the check still prevents the OOB write at line 1190), but the
reported pair is misleading for a malformed-model diagnosis and is inconsistent
with the precise `needed: li + 1` reporting in `output_leaf_vector`
(lines 828-831).

**Fix:** Either hoist a single `if leafvec.len() > lvs` pre-check before the loop
and frame the message as "leaf vector longer than declared lvs", or keep the
per-element check and report `needed: i + 1, got: lvs` so the payload names the
overflow point.

## Info

### IN-01: `GtilError::UnsupportedPredictKind` is now dead

**File:** `crates/treelite-gtil/src/error.rs:93-100`

**Issue:**
The doc comment says `LeafId`/`ScorePerTree` "surface as this typed error" until
Plan 05-04, but both kinds are now fully wired (`predict_rows` dispatches them at
`lib.rs:1020-1023`) and no code path constructs `UnsupportedPredictKind`
anymore. The variant and its `kind: &'static str` field are dead.

**Fix:** Remove the variant (and its stale doc), or if it is retained as a
forward-compat placeholder, update the comment to say so explicitly rather than
referencing a closed plan.

### IN-02: `output_shape` and predict resolve the target dim with differently-spelled clamps

**File:** `crates/treelite-gtil/src/shape.rs:38-46` vs `crates/treelite-gtil/src/lib.rs:1031-1034`

**Issue:**
`output_shape` emits the target dim as `model.num_target as u64` when
`num_target > 1`, else literally `1`. `predict_rows` clamps
`num_target == 0 -> 1` then uses `num_target`. They agree for all `num_target >= 1`
models and even for the degenerate `num_target == 0` case (both yield dim 1), so
there is no live bug — but the two clamps are spelled differently, so a future
edit to one will not be caught by the other. No fixture exercises `num_target == 0`.

**Fix:** Factor the `(num_target, max_num_class)` resolution into one shared
helper used by both `output_shape` and `predict_rows` so the published shape and
the produced buffer can never drift.

### IN-03: D-04 parity probe drops present-NaN cells; asymmetry with kept inf is undocumented

**File:** `crates/treelite-harness/tests/gtil_matrix.rs:252-272`

**Issue:**
`build_csr` reconstructs the parity CSR by treating every non-NaN dense cell as
present. The edge matrix injects a present `np.nan` (`X[5, cat_col]`) and `±inf`.
The present-NaN is dropped (becomes "absent" in the parity CSR) while inf is
kept and the dense path keeps NaN as a present feature. Because NaN routes to the
default child identically whether present or absent, the parity assert still
holds — but this is the same "present NaN == absent" ambiguity WR-01/T-05-19
flagged for the golden path, surviving in the parity probe. It is correctly
scoped to parity-only (the golden gate uses the frozen CSR via `frozen_csr`), so
this is informational; the NaN-dropped / inf-kept asymmetry just deserves a
comment.

**Fix:** Add a one-line note at `build_csr` that it is a NaN-presence
reconstruction used only for the D-04 invariant and deliberately differs from the
frozen capture CSR on present-NaN cells.

---

_Reviewed: 2026-06-10T09:28:01Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

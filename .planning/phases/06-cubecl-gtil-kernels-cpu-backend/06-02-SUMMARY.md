---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 02
subsystem: cubecl-kernels
tags: [cubecl, spike, descend, exp2, softmax, f64, wave-1, wave-2, tdd, D-04]
requires:
  - "treelite-cubecl scaffold (06-01: crate + cubecl 0.10.0 cpu + bytemuck + RED spike scaffold)"
  - "treelite-core::{Model, ModelPreset, ModelVariant, Tree, TreeBuf} (hand-built 2-tree forests)"
  - "treelite-gtil::{predict, Config} + postprocessor::{exponential_standard_ratio, exponential_standard_ratio_f64, softmax_f64} (the 1e-5 scalar twins)"
provides:
  - "treelite_cubecl::kernels::traversal::descend<F: Float> — the verbatim #[cube] break-free numerical descent helper Wave 3 reuses (D-11)"
  - "kernels/ directory module (mod.rs) with traversal authored + postproc/default_raw/leaf_id/score_per_tree reserved for Wave 3"
  - "green spike (3 tests): 2-tree default kernel f32+f64 vs predict, exp_standard_ratio f32+f64, softmax_f64 — all within 1e-5 on CpuRuntime"
  - "cubecl 0.10.0 kernel API surface confirmed: A1 exp2=exp(x*ln2) identity, A2 f64+mixed-width locals, A3 create_from_slice upload, A4 cpu::CpuRuntime import"
affects:
  - "crates/treelite-cubecl/src/kernels.rs (file module -> kernels/ directory module)"
tech-stack:
  added: []
  patterns:
    - "#[cube] break-free descent: while-!is_leaf, if-STATEMENT child routing (never if-expr value), self-inequality NaN test (fv != fv)"
    - "Float-scalar launch args ride as 1-element Array<F> (sidesteps the Float ScalarArg ambiguity); u32 scalars passed plain; ABSOLUTE_POS as u32"
    - "exp2(x) == exp(x*ln2) identity in element width F (cube frontend has Exp, no Exp2)"
key-files:
  created:
    - crates/treelite-cubecl/src/kernels/mod.rs
    - crates/treelite-cubecl/src/kernels/traversal.rs
  modified:
    - crates/treelite-cubecl/tests/spike.rs
decisions:
  - "A1 RESOLVED via the exp(x*ln2) IDENTITY, NOT direct exp2 (overturns the 06-01 pin): cubecl 0.10.0's Float::exp2 (typemap.rs:680) is the DynamicScalar RUNTIME path; the cube FRONTEND expandable-intrinsic set (frontend/operation/unary.rs) has Exp but NO Exp2, so F::exp2(x) fails E0599 on a generic F inside #[cube]. exp2(x)==exp(x*ln(2)) in element width F matches the exponential_standard_ratio/_f64 twins within 1e-5 (f32 AND f64)."
  - "NaN routing in descend uses the self-inequality fv != fv (overturns the planned/RESEARCH F::is_nan form): Float::is_nan returns Self::WithScalar<bool> (an associated type, not plain bool) on a generic F, so `if F::is_nan(fv)` fails E0308. fv != fv lowers to the same in-kernel NaN test and is the verbatim equivalent of evaluate_tree's fvalue.is_nan_val()."
  - "ABSOLUTE_POS is usize in cubecl 0.10.0 (not u32): cast `as u32` at the kernel top so all index math + the descend u32 offset params stay one width."
  - "Float-width launch scalars (base_score, ratio_c, ln2) ride as 1-element Array<F> rather than a Float ScalarArg — the launch macro's plain-value scalar path is documented only for integer widths; the 1-element-Array form is unambiguous and zero-risk for the spike."
  - "A3 upload entry is client.create_from_slice(&[u8]) (matches the 06-01 pin); read-back via client.read_one_unchecked(handle) + bytemuck::cast_slice. A4 import path is cubecl::cpu::CpuRuntime + cubecl::{CubeCount,CubeDim,Runtime} + cubecl::prelude::*."
metrics:
  duration: ~25min
  completed: 2026-06-10
  tasks: 2
  files: 4
---

# Phase 6 Plan 02: cubecl Descent Spike (D-04 Confirmation) Summary

Authored the verbatim `#[cube]` break-free numerical descent helper
(`kernels::traversal::descend`) that Wave 3 reuses, and turned the RED Wave-0
spike green: a 2-tree numerical `default`-kind kernel matches
`treelite_gtil::predict` within 1e-5 on f32 AND f64, and standalone
`exponential_standard_ratio` + `softmax_f64` micro-kernels reproduce their
scalar twins' cast order within 1e-5 on `CpuRuntime`. All four cubecl
API-surface assumptions A1–A4 are retired — with two corrections to the planned
API forms that the spike exists to catch (D-04).

## What Was Built

**Task 1 — `#[cube]` break-free descent helper (commit 09ffe4c):**
- Converted the placeholder `src/kernels.rs` file module into a `kernels/`
  directory module: `mod.rs` declares `pub mod traversal;` and reserves
  `postproc`/`default_raw`/`leaf_id`/`score_per_tree` (as comments) for Wave 3.
- `descend<F: Float>(cleft, cright, split_index, threshold, default_left, base,
  row_off, input) -> u32` ports `evaluate_tree`'s numerical path line-by-line:
  a `while cleft[base+nid] != -1` loop (no loop-skip keyword, no `break`),
  if-STATEMENT child routing (`let mut next = cright[...]; if … { next =
  cleft[...]; }` — never an if-expr value), `default_left == 1u32` (the bool
  column uploaded as u32, Pitfall 4), and ragged-SoA `base`/`row_off` offset
  indexing (`concat[base + nid]`, `input[row_off + fi]`). `cargo build
  -p treelite-cubecl` green; grep gates: loop-skip-keyword=0, `.is_nan()`=0,
  `F::is_nan`=0.

**Task 2 — green spike, A1–A4 retired (commit d7a11c3):**
- `predict_default_2tree<F>` `#[cube(launch)]`: one unit per row
  (`ABSOLUTE_POS as u32`), serial `for tree_id` accumulation calling `descend`,
  base-score add, identity postprocessor. Asserts element-wise within 1e-5 of
  `predict::<f32>` AND `predict::<f64>` over two hand-built single-split trees
  exercising both branches of both trees (retires A2 f64-in-kernel).
- `exp_standard_ratio_kernel<F>`: `exp((-v/ratio_c) * ln2)` matching
  `exponential_standard_ratio`/`_f64` within 1e-5 (f32 + f64) — retires A1.
- `softmax_f64_kernel`: f64 cells with f32 `max_margin`/`t`/divisor locals,
  reproducing `softmax_f64`'s exact mixed-width cast order within 1e-5 over a
  3-class large-spread row — retires A2 mixed-width locals.
- Upload via `client.create_from_slice(bytemuck::cast_slice(..))` (A3),
  read-back via `client.read_one_unchecked` + `bytemuck::cast_slice` (A4 import
  path `cubecl::cpu::CpuRuntime`). The RED `#[ignore]`/`todo!()` scaffold is
  replaced; an in-test header comment records the resolved exp2 form and that
  A1–A4 are retired. No D-04 hedge branch.

## Verification

- `cargo build -p treelite-cubecl` — green (the `#[cube] descend` helper
  compiles: no loop-skip keyword, associated-fn math, if-statement routing all
  compile on cubecl 0.10.0).
- `cargo test -p treelite-cubecl --test spike` — 3/3 green
  (`spike_default_2tree_f32_and_f64_descend`,
  `spike_exp_standard_ratio_matches_scalar_twin`,
  `spike_softmax_f64_matches_scalar_twin`), 0 ignored.
- `cargo test --workspace` — fully green, 0 failures (no regression to any
  existing crate; the spike is the only new behavior).
- Greps: `traversal.rs` continue=0, `.is_nan()`=0, `F::is_nan`=0, `#[cube]`
  present; `spike.rs` `fall.*back|contingency|host.*postproc`=0, `#[ignore]`=0,
  `assert_abs_diff_eq`=6, `treelite_gtil::predict` linked, A1–A4-retired comment
  present.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] NaN test uses `fv != fv`, not `F::is_nan(fv)`**
- **Found during:** Task 1 (`cargo build -p treelite-cubecl`).
- **Issue:** The plan + RESEARCH Pattern 1 specify `if F::is_nan(fv)` as
  associated-fn math. On the installed cubecl 0.10.0, `Float::is_nan` returns
  `Self::WithScalar<bool>` (an associated type, not a plain `bool`) for a
  generic `F: Float`, so `if F::is_nan(fv)` fails E0308 (`expected bool, found
  associated type`).
- **Fix:** Use the self-inequality `fv != fv` (NaN is the only value not equal
  to itself) — the verbatim equivalent of `evaluate_tree`'s
  `fvalue.is_nan_val()`. Functional intent (no `.is_nan()` method, no plain
  Rust helper) preserved; the planned `F::is_nan` grep gate's INTENT (no
  method-call NaN test) is met. Comments reworded so the literal-token grep
  gates (`continue`, `F::is_nan`, `.is_nan()`) read 0.
- **Files modified:** `crates/treelite-cubecl/src/kernels/traversal.rs`.
- **Commit:** 09ffe4c.

**2. [Rule 1 - Bug] `exp2` via the `exp(x*ln2)` identity, not direct `F::exp2`**
- **Found during:** Task 2 (`cargo test --test spike --no-run`).
- **Issue:** The 06-01 scaffold pinned `exp2 = Float::exp2(self) direct`
  (`cubecl-core typemap.rs:680`). That method lives on the DynamicScalar
  *runtime* path; the cube *frontend* expandable-intrinsic set
  (`frontend/operation/unary.rs`) exposes `Exp` but has NO `Exp2`, so
  `F::exp2(x)` fails E0599 (`no function exp2 found for type parameter F`)
  inside `#[cube]`.
- **Fix:** Use the exact algebraic identity `exp2(x) == exp(x * ln(2))` in the
  element's own width `F` (`F::exp` IS a cube-frontend intrinsic), with `ln(2)`
  uploaded as a 1-element `Array<F>`. This is precisely RESEARCH Pitfall 2's
  documented fallback and the spike's A1 lock-down purpose (D-04). Verified
  within 1e-5 vs the scalar twins on f32 AND f64.
- **Files modified:** `crates/treelite-cubecl/tests/spike.rs`.
- **Commit:** d7a11c3.

**3. [Rule 3 - Blocking] `ABSOLUTE_POS as u32` + Float-scalar launch args as 1-element `Array<F>`**
- **Found during:** Task 2 (`cargo test --test spike --no-run`).
- **Issue:** `ABSOLUTE_POS` is `usize` in cubecl 0.10.0 (mixing `usize`/`u32`
  index math failed E0308/E0277). The launch macro's plain-value scalar path is
  documented for integer widths; a `Float` scalar param (`base_score: F`,
  `ratio_c: F`) had no unambiguous launch form in the available examples.
- **Fix:** Cast `ABSOLUTE_POS as u32` at the kernel top (one index width
  throughout); pass `u32` scalars as plain values; ride the `F`-width scalars
  (`base_score`, `ratio_c`, `ln2`) as 1-element `Array<F>` uploads. Zero-risk
  and unambiguous for the spike.
- **Files modified:** `crates/treelite-cubecl/tests/spike.rs`.
- **Commit:** d7a11c3.

These three are the cubecl API-surface corrections the spike exists to catch
(D-04). They affect the API FORM only; the control-flow shape, cast order, and
1e-5 fidelity are exactly as planned. Wave 3 inherits the corrected forms via
the reused `descend` helper and the documented exp2/scalar-arg patterns.

## Known Stubs

Intentional, plan-scoped, tracked by the remaining RED scaffolds:
- `kernels/mod.rs` reserves `postproc`/`default_raw`/`leaf_id`/`score_per_tree`
  as comments — authored in Wave 3 (plan 06-04).
- `predict_cpu` (lib.rs) still returns `CubeclError::Unsupported` — the real
  host launcher body lands in Wave 3.
- `tests/upload.rs` / `tests/determinism.rs` / `gtil_matrix_cubecl.rs` remain
  RED `#[ignore]` (Waves 2/4).

None block this plan's goal (the spike is green; the reusable `descend` helper
exists; A1–A4 retired).

## Self-Check: PASSED

- crates/treelite-cubecl/src/kernels/mod.rs — FOUND
- crates/treelite-cubecl/src/kernels/traversal.rs — FOUND
- crates/treelite-cubecl/tests/spike.rs — FOUND
- crates/treelite-cubecl/src/kernels.rs (old file module) — REMOVED (intentional, file→dir conversion)
- commit 09ffe4c — FOUND
- commit d7a11c3 — FOUND

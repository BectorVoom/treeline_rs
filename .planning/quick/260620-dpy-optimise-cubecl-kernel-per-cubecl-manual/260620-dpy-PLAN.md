---
phase: quick-260620-dpy
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/treelite-cubecl/src/kernels/traversal.rs
  - crates/treelite-cubecl/src/kernels/default_raw.rs
  - crates/treelite-cubecl/src/lib.rs
autonomous: true
requirements: [QUICK-CUBECL-OPT]

must_haves:
  truths:
    - "cargo test -p treelite-cubecl passes (all 9 tests) after the descend CSE refactor"
    - "cargo test -p treelite-harness gtil_matrix_cubecl golden matrix still matches within 1e-5"
    - "The single-cell comptime path of predict_default_raw produces byte-identical output to the general path for cells_per_row == 1 models"
    - "f64-promotion compare, NaN self-inequality, serial tree-sum order, and base-score/RF-average numerics are unchanged (byte-identical)"
  artifacts:
    - path: "crates/treelite-cubecl/src/kernels/traversal.rs"
      provides: "descend with hoisted node-index CSE (single (base+nid) computation per iteration)"
    - path: "crates/treelite-cubecl/src/kernels/default_raw.rs"
      provides: "predict_default_raw with #[comptime] single_cell straight-line specialization"
    - path: "crates/treelite-cubecl/src/lib.rs"
      provides: "host launcher computing single_cell = (cells_per_row == 1) and passing it to launch"
  key_links:
    - from: "crates/treelite-cubecl/src/lib.rs"
      to: "kernels::default_raw::predict_default_raw::launch"
      via: "single_cell comptime argument"
      pattern: "single_cell"
---

<objective>
Optimise the cubecl GTIL inference kernels in `crates/treelite-cubecl/` using two
techniques validated against the CubeCL manual, WITHOUT changing any numerics.

Two changes only, both proven safe against the 1e-5 fidelity contract:
- A. Common-subexpression hoist in the shared `descend` helper (zero numeric change,
  benefits all 3 traversal kernels).
- B. `#[comptime]` single-cell specialization of `predict_default_raw` for the
  overwhelmingly common XGBoost binary/regression output shape (`cells_per_row == 1`),
  producing byte-identical results via the same arithmetic with the per-cell loops
  collapsed to straight-line code (manual: comptime_specialization.md).

Purpose: Reduce per-iteration index arithmetic on the descent hot path and remove
per-row loop/divergence overhead for the typical model shape, while keeping every
prediction byte-identical to the scalar reference (the CLAUDE.md core value).

Output: Updated traversal.rs, default_raw.rs, lib.rs; all cubecl tests + the harness
golden matrix green.

NON-NEGOTIABLE constraints (CLAUDE.md + GTIL-08 + 05-02 promotion contract):
- Serial tree accumulation order preserved EXACTLY — float add is non-associative.
  No plane/atomic/reduction over the tree axis. Do not touch the `for tree_id`
  accumulation loop body's order.
- The f64-promotion compare in `descend`
  (`f64::cast_from(fv) < f64::cast_from(threshold[...])`) stays byte-preserved.
  Do NOT narrow either operand.
- NaN self-inequality test (`fv != fv`) stays verbatim (with its scoped
  `#[allow(clippy::eq_op)]`).
- Zeroing, RF-average divide, and f64 base-score add stay bit-identical — the
  comptime path must be the SAME arithmetic, just unrolled for n==1.

REJECTED options (do NOT attempt — recorded so the executor does not try):
- `#[unroll]` on the tree-descent `while` loop: the loop bound
  (`cleft[(base+nid) as usize] != -1`) is data-dependent, NOT comptime-known, so it
  cannot be unrolled (manual 01_loop_unrolling.md requires a compile-time boundary).
- Autotuning / changing `CubeDim::new_1d(256)` or the cube_count: deferred, out of
  scope for a quick task (no evidence motivating a change).
- Comptime-specializing the leaf-vector broadcast branches: those depend on
  per-tree `(target_id, class_id)` routing, not a single comptime flag — leave as-is.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@/home/user/Documents/workspace/cubecl_manual/manual/cubecl/comptime_specialization.md
@crates/treelite-cubecl/src/kernels/traversal.rs
@crates/treelite-cubecl/src/kernels/default_raw.rs
@crates/treelite-cubecl/src/lib.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Hoist the node-index CSE in the shared descend helper</name>
  <files>crates/treelite-cubecl/src/kernels/traversal.rs</files>
  <action>
In the `while` loop body of `descend` (traversal.rs lines 77-109), the expression
`(base + nid) as usize` is currently recomputed five separate times: in the loop
guard (line 77), and inside the body at lines 78, 81, 93/94, and 104/105. Introduce
a single `let idx = (base + nid) as usize;` as the FIRST statement inside the loop
body, and replace every in-body `(base + nid) as usize` indexing site with `idx`
(lines 78, 81, 93, 94, 104, 105). The loop-guard read `cleft[(base + nid) as usize]`
at line 77 is evaluated before `idx` exists for the next iteration; leave that guard
expression as-is (it is the `!is_leaf` test and `nid` has just been reassigned at the
bottom of the previous iteration). Do NOT reorder, remove, or alter any of the
branches; this is a pure read-site substitution. Keep the f64-promotion compare, the
`fv != fv` NaN test, the `#[allow(clippy::eq_op)]` attribute, and the
default-route/if-statement structure byte-for-byte identical. The generic signature,
the doc comment, and the return value are unchanged.

This is a CSE micro-opt with provably zero numeric effect: `idx` equals the value the
removed expressions computed, the array reads are the same elements in the same order,
and no arithmetic operand changes. All three kernels (default_raw, leaf_id,
score_per_tree) reuse this helper, so all three benefit from the single edit.
  </action>
  <verify>
    <automated>cargo test -p treelite-cubecl 2>&1 | tail -25</automated>
  </verify>
  <done>
`descend` computes `(base + nid) as usize` once per iteration via a hoisted `idx`
binding; all in-body index sites use `idx`. `cargo test -p treelite-cubecl` passes
all 9 tests. `cargo clippy -p treelite-cubecl` is clean (the eq_op allow is retained).
  </done>
</task>

<task type="auto">
  <name>Task 2: Comptime single-cell specialization of predict_default_raw</name>
  <files>crates/treelite-cubecl/src/kernels/default_raw.rs, crates/treelite-cubecl/src/lib.rs</files>
  <action>
Add a `#[comptime] single_cell: bool` parameter to the `predict_default_raw` kernel
(default_raw.rs, after the trailing scalar `num_feature: u32` argument). `single_cell`
is true exactly when `cells_per_row == 1` (i.e. `num_target == 1 && max_num_class == 1`),
the typical XGBoost binary/regression output shape.

Specialize the THREE per-cell loop regions guarded on `single_cell` using a standard
Rust `if single_cell { ... } else { ... }` (the manual's comptime branch — CubeCL
prunes the dead arm at JIT time, no device branch):

1. Zeroing loop (lines 86-90): when `single_cell`, replace the `while z < cells_per_row`
   loop with the straight-line `output[row_base as usize] = F::new(0.0);`
   (row_base + 0, the only cell). Otherwise keep the existing `while` loop verbatim.

2. RF-average loop (lines 171-188): when `single_cell`, the single cell is
   `(t=0, c=0)`, `li = 0`, `cell = row_base`. Read `factor = average_factor[0]`, and
   if `factor != 0.0` do `output[row_base as usize] /= F::cast_from(factor);` — the SAME
   arithmetic as the general path for n==1. Otherwise keep the existing nested `while`.

3. Base-score loop (lines 193-205): when `single_cell`, `li = 0`, `cell = row_base`;
   compute `acc = f64::cast_from(output[row_base as usize]) + base_scores[0]` and write
   `output[row_base as usize] = F::cast_from(acc);` — identical arithmetic for n==1.
   Otherwise keep the existing nested `while`.

CRITICAL: do NOT touch the serial `for tree_id in 0..num_tree` accumulation block
(lines 92-166) — it is shared by both arms and its order is the GTIL-08 contract. The
leaf-vector broadcast / OutputLeafValue branches inside it stay exactly as written. The
specialization is ONLY the three per-cell pre/post loops, and the single-cell arm must
be the same operations the general loop performs when it iterates exactly once. Use
`F::new(0.0)` (zeroing) and `F::cast_from(...)` exactly as the general path does so the
zero and the casts are bit-identical.

In lib.rs `run_default_raw` (the launch call at lines 477-504), compute
`let single_cell = cells_per_row == 1;` (cells_per_row is already in scope at line 460)
and pass it as the final argument to `predict_default_raw::launch::<F, T, R>(...)` after
`num_feature as u32`. `single_cell` is a plain `bool` comptime arg (not wrapped in
`ScalarArg`), matching the manual's `#[comptime] use_plane: bool` launch convention.
No other launcher (leaf_id, score_per_tree) changes.
  </action>
  <verify>
    <automated>cargo test -p treelite-cubecl 2>&1 | tail -25</automated>
  </verify>
  <done>
`predict_default_raw` takes `#[comptime] single_cell: bool` and uses an
`if single_cell { straight-line } else { existing while loops }` branch for the
zeroing, RF-average, and base-score regions; the tree-accumulation block is unchanged.
`run_default_raw` passes `cells_per_row == 1`. `cargo test -p treelite-cubecl` passes
all tests (predict_kinds, determinism, postproc, upload, malformed, device_absent,
spike), proving the single-cell models still match within 1e-5.
  </done>
</task>

<task type="auto">
  <name>Task 3: Prove 1e-5 fidelity via the harness golden matrix</name>
  <files>crates/treelite-cubecl/src/kernels/default_raw.rs</files>
  <action>
No new code — this is the end-to-end fidelity gate that proves Tasks 1 and 2 preserved
byte/1e-5 equivalence against real models. Run the cubecl golden matrix harness test
(`crates/treelite-harness/tests/gtil_matrix_cubecl.rs`), which compares cubecl-cpu
predictions against the frozen scalar golden matrix across the model fixtures
(including single-cell XGBoost binary/regression models that now exercise the comptime
single_cell=true path, and multi-cell models that exercise the single_cell=false arm).
If any case diverges beyond 1e-5, the optimization broke a numeric invariant — STOP and
report which case/model failed rather than loosening tolerance. Also run the full
cubecl crate suite once more so both crates are green together. (No file edit is
expected; the `files` entry is the kernel under test for provenance.)
  </action>
  <verify>
    <automated>cargo test -p treelite-harness --test gtil_matrix_cubecl 2>&1 | tail -30 && cargo test -p treelite-cubecl 2>&1 | tail -15</automated>
  </verify>
  <done>
`cargo test -p treelite-harness --test gtil_matrix_cubecl` passes (all golden-matrix
cases within 1e-5) AND `cargo test -p treelite-cubecl` passes (all 9 tests). The
single-cell comptime path and the descend CSE are confirmed numerically equivalent to
the scalar reference on real model fixtures.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host→device | Host launcher passes the `single_cell` comptime flag and column buffers into the cubecl kernel; flag must equal `cells_per_row == 1`. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-quick-01 | Tampering | `single_cell` comptime flag mismatch (host computes it wrong) | mitigate | Host derives `single_cell` solely from `cells_per_row == 1`; single-cell arm is the same arithmetic as the n==1 iteration of the general loop, and the golden-matrix harness (Task 3) covers both single-cell and multi-cell fixtures so a mismatch fails the 1e-5 gate. |
| T-quick-02 | Tampering | `descend` CSE alters routing | accept | Pure read-site substitution of an identical value (`idx == (base+nid) as usize`); no operand or branch changes. Covered by the full cubecl suite + golden matrix. |
| T-quick-03 | Information disclosure | OOB device read from specialized index | mitigate | `single_cell` arm only reads index `row_base`/`0`, a strict subset of the cells the general loop already validated; `upload_forest` bounds-checks before any device write. No new index space introduced. |
</threat_model>

<verification>
- `cargo test -p treelite-cubecl` — all 9 tests pass (determinism, predict_kinds,
  postproc, upload, malformed, device_absent, spike).
- `cargo test -p treelite-harness --test gtil_matrix_cubecl` — golden matrix within 1e-5.
- `cargo clippy -p treelite-cubecl` — clean (the scoped `eq_op` allow on the NaN test
  is retained; no new lints).
</verification>

<success_criteria>
- `descend` computes the node index once per iteration; all 3 traversal kernels benefit.
- `predict_default_raw` has a `#[comptime] single_cell` straight-line path for
  `cells_per_row == 1` models, byte-identical to the general path.
- Serial tree-sum order, f64-promotion compare, NaN test, RF-average, and base-score
  numerics are unchanged.
- All cubecl tests and the harness golden matrix pass within 1e-5.
</success_criteria>

<output>
Create `.planning/quick/260620-dpy-optimise-cubecl-kernel-per-cubecl-manual/260620-dpy-SUMMARY.md` when done
</output>

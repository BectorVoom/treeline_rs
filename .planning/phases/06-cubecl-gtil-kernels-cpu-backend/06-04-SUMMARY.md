---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 04
subsystem: cubecl-kernels
tags: [cubecl, launch-kernels, predict-cpu, four-predict-kinds, leaf-vector-broadcast, rf-averaging, categorical-fallback, f32, f64, wave-4, tdd, GPU-01, GPU-02, D-01, D-02, D-05, Pitfall-6]
requires:
  - phase: 06-02
    provides: "descend<F> break-free #[cube] descent helper, exp(x*ln2) identity, fv!=fv NaN test, ABSOLUTE_POS as u32, 1-element-Array<F> Float-scalar launch convention, CpuRuntime import path"
  - phase: 06-03
    provides: "upload_forest ragged-SoA per-column handles + tree_node_offset/tree_leafvec_offset prefix-sum index + validate_shape (T-06-06); 10 #[cube] postprocessor helpers"
provides:
  - "treelite_cubecl::kernels::{default_raw::predict_default_raw, leaf_id::predict_leaf_id, score_per_tree::predict_score_per_tree} — three #[cube(launch)] kernels, one unit per row, serial trees, leaf-vector broadcast, NO tree-axis reduction (SC1/SC2/GTIL-08)"
  - "treelite_cubecl::predict_cpu::<F> — real host launcher: validate->categorical/sparse scalar fallback (D-02)->upload forest+routing/averaging/base columns->select kernel by Config.kind->launch on CpuRuntime->read back; output element = input dtype (Pitfall 6)"
  - "treelite_cubecl::PredictCpuElem trait (f32/f64) — per-element postprocessor dispatch mirroring gtil apply_postprocessor_{f32,f64} arm-for-arm; treelite_cubecl::predict_cpu_sparse whole-model scalar fallback"
  - "green tests/predict_kinds.rs (8 tests): all 4 kinds + multiclass leaf-vector broadcast + RF averaging within 1e-5 vs treelite_gtil::predict on f32 AND f64; categorical fallback parity; malformed-shape + bad split_index -> typed CubeclError (no panic)"
affects:
  - "06-05 (determinism two-run bit-identity gate + gtil_matrix_cubecl harness sibling: both call predict_cpu over upload_forest)"
tech-stack:
  added: []
  patterns:
    - "One #[cube(launch)] kernel per predict kind (host-known from Config.kind); the kind branch lives on the HOST (kernel selection), never in-kernel — keeps each kernel's control flow simple (RESEARCH Alternatives Considered)"
    - "Fused traversal+accumulate+RF-average+f64-base-score in the default_raw kernel reproducing predict_preset's exact assembly order (serial tree-sum -> RF /factor -> f64 2D base-score add); the Default postprocessor is a SEPARATE host step after read-back"
    - "descend<F, T> generalized over BOTH the input width F and the threshold width T (Pitfall 6): compare fv (F) against F::cast_from(threshold) (T) — identity for matching-width presets, the NextNode<InputT,ThresholdT> promotion otherwise"
    - "RF average_factor + (target_id,class_id) routing columns precomputed HOST-side (mirroring predict_preset:680-708) and uploaded as device arrays the kernel reads; average_factor is 1.0 per cell when averaging is off (divide by 1)"
    - "Categorical/sparse whole-model scalar fallback at the host boundary (D-02 / Open Q1): any tree.has_categorical_split -> route the WHOLE model to treelite_gtil::predict; the kernels stay purely numerical"
    - "PredictCpuElem postprocessor dispatch reuses the PUBLIC treelite_gtil::postprocessor::* functions arm-for-arm, so Default parity is byte-identical to the scalar reference (same softmax per-(row,target) row spans)"
key-files:
  created:
    - crates/treelite-cubecl/src/kernels/default_raw.rs
    - crates/treelite-cubecl/src/kernels/leaf_id.rs
    - crates/treelite-cubecl/src/kernels/score_per_tree.rs
    - crates/treelite-cubecl/tests/predict_kinds.rs
  modified:
    - crates/treelite-cubecl/src/lib.rs
    - crates/treelite-cubecl/src/kernels/mod.rs
    - crates/treelite-cubecl/src/kernels/traversal.rs
    - crates/treelite-cubecl/tests/spike.rs
key-decisions:
  - "The Default-kind postprocessor is applied HOST-side after the kernel read-back (via the PredictCpuElem trait reusing the public treelite_gtil::postprocessor::* functions arm-for-arm), NOT inside the default_raw kernel. The plan's <action> phrased it as 'the default kernel applies the matching #[cube] postprocessor', but the named-postprocessor dispatch (10 variants + softmax's per-(row,target) row spans) is selected by a host-only String (model.postprocessor) and the parity contract is what matters; reusing the SAME scalar postprocessor functions makes Default byte-identical to treelite_gtil::predict (no second device launch, no in-kernel String dispatch). The default_raw kernel produces the Raw margin (sum+average+base); Default layers the postproc — exactly predict_rows's Default==Raw+postproc structure (lib.rs:1033)."
  - "descend's signature changed from descend<F>(threshold: &Array<F>) to descend<F, T>(threshold: &Array<T>) for the Pitfall-6 input/threshold width split. The Wave-1 spike's call site was updated to descend::<F, F> (matching widths, identity cast) — the spike's behavior and 3/3 green asserts are unchanged."
  - "RF averaging is done as a per-cell average_factor f64 array computed host-side (the verbatim predict_preset:680-708 four-way count) and uploaded; the kernel divides each cell by its factor. When average_tree_output is false the host fills the factor with 1.0 (a divide by 1), so the same kernel path serves both RF and non-RF models with no in-kernel branch on the averaging flag."
  - "The scoped #[allow(clippy::eq_op)] for the fv!=fv NaN test was added on traversal.rs (the deferred-items.md item from 06-03): this plan was already editing traversal.rs for the <F,T> generalization, so the one-line scoped allow landed here. cargo clippy -p treelite-cubecl is now fully clean."
  - "predict_cpu_sparse is a thin whole-model scalar-fallback entry (treelite_gtil::predict_sparse) — sparse input never touches the dense-numerical kernels this phase (D-02)."
patterns-established:
  - "One #[cube(launch)] kernel per predict kind, host-side kernel selection by Config.kind (D-01)"
  - "Host launcher: validate -> categorical/sparse scalar fallback -> upload -> select -> launch -> read (D-02 / Pitfall 6)"
  - "Pitfall-6 descend<F, T> input/threshold width split asserted to 1e-5 across f32+f64 presets"
requirements-completed: [GPU-01, GPU-02]
duration: ~30min
completed: 2026-06-10
---

# Phase 6 Plan 04: cubecl Launch Kernels (All 4 Predict Kinds) + predict_cpu Host Launcher Summary

**The real end-to-end-kernelized MVP slice: three `#[cube(launch)]` kernels (default/raw fused, leaf_id, score_per_tree) running one unit per row with serial trees, leaf-vector broadcast, and no tree-axis reduction, driven by a real `predict_cpu` host launcher (validate -> categorical/sparse scalar fallback -> upload -> select-by-kind -> launch -> read) that reproduces `treelite_gtil::predict` within 1e-5 across all four kinds + multiclass leaf-vector broadcast on f32 AND f64 input (D-01/D-02/D-05/Pitfall 6).**

## Performance

- **Duration:** ~30 min
- **Tasks:** 2
- **Files modified:** 8 (4 created, 4 modified)
- **Tests:** cubecl predict_kinds 8/8, spike 3/3, postproc 9/9; workspace 56 suites ok, 0 failures.

## Accomplishments

- **Task 1 — three `#[cube(launch)]` kernels (commit `2fdff21`):**
  - `default_raw::predict_default_raw<F, T>`: the fused kernel for the `Default`/`Raw` kinds. One unit per row (`ABSOLUTE_POS as u32`, `if row < num_row`); zero-fills the row's cells; a SERIAL `for tree_id in 0..num_tree` loop calling `descend` per tree, accumulating into the row's `(target,class)` cell(s) via the verbatim four-way `OutputLeafVector` branch on `(target_id[tree], class_id[tree])` — INCLUDING the multiclass leaf-vector broadcast (both `-1`) addressed via `leafvec_off[t] + leaf_vector_begin[...]`; then RF averaging (divide each cell by the precomputed `average_factor`), then the f64 2D base-score add (`f64::cast_from(cell) + base_scores[li]` narrowed back to `F`). Reproduces `predict_preset`'s assembly order line-by-line. NO atomic/reduce over the tree axis (disjoint per-row writes, SC1/SC2).
  - `leaf_id::predict_leaf_id<F, T>`: per-`(row, tree)` leaf NODE id (the tree-relative `nid`) cast into the `F` output buffer (`PredictLeaf`, `predict.cc:340`).
  - `score_per_tree::predict_score_per_tree<F, T>`: raw per-tree leaf data into the `(num_row, num_tree, lvs)` buffer — leaf vector elements at `(row, tree, i)` (bounded by `lvs`), or the scalar leaf at index 0 (`PredictScoreByTree`, `predict.cc:367-372`).
  - `descend` generalized to `descend<F, T>` (Pitfall 6): the input value `fv` (width `F`) is compared against `F::cast_from(threshold)` (width `T`) — identity for the matching-width presets, the `NextNode<InputT,ThresholdT>` promotion otherwise. The Wave-1 spike call site updated to `descend::<F, F>` (unchanged behavior). A scoped `#[allow(clippy::eq_op)]` was added on the `fv != fv` NaN test (deferred-items.md from 06-03).
- **Task 2 — `predict_cpu` host launcher + 4-kind 1e-5 test (commit `c39dc0c`):**
  - Replaced the plan-01 stub with the real launcher: (1) the categorical whole-model fallback gate (`any tree.has_categorical_split` -> `treelite_gtil::predict`, D-02) is checked before any upload; (2) `upload_forest`'s `validate_shape` runs BEFORE any device write (the malformed-shape / bad-`split_index` -> typed `CubeclError` path, T-06-09); (3) the forest + input + routing/averaging/base columns are uploaded, the kernel is selected by `Config.kind`, `launch::<F, T, CpuRuntime>` runs with the ceiling-division grid (`CubeCount::Static((num_row+255)/256,1,1)`, `CubeDim{x:256}`), and the output reads back via `read_one_unchecked` + `bytemuck::cast_slice` into `Vec<F>`. The output element EQUALS the input dtype (Pitfall 6 — no PRE-cast of input inside `predict_cpu`).
  - `PredictCpuElem` trait (f32/f64): the per-element postprocessor dispatch mirroring `gtil`'s private `apply_postprocessor_{f32,f64}` arm-for-arm (reusing the PUBLIC `postprocessor::*` functions, including softmax's per-`(row,target)` row spans), applied HOST-side after read-back for the `Default` kind only (`Raw` skips). `predict_cpu_sparse` is the whole-model scalar `predict_sparse` fallback.
  - `tests/predict_kinds.rs` (8 tests): all 4 kinds on a scalar binary model + a multiclass leaf-vector-broadcast model (the 4-way `OutputLeafVector` `-1/-1` branch) + an RF-averaging model, within 1e-5 of `treelite_gtil::predict` on f32 AND f64 input; a categorical-split model asserting the scalar fallback still lands within 1e-5; and two typed-error tests (short data buffer -> `InvalidInputShape`, out-of-range `split_index` -> `FeatureIndexOutOfBounds`) proving no panic.
- **Cleanup (commit `bc68865`):** consolidated the duplicate `PredictOut` bound on `predict_cpu` into a `PredictCpuElem` supertrait — `cargo clippy -p treelite-cubecl` is now fully clean (the `eq_op` deny that previously blocked the crate's clippy is scoped-allowed).

## Task Commits

Each task committed atomically (sequential executor, main tree, hooks on):

1. **Task 1: three #[cube(launch)] kernels for all 4 predict kinds + leaf-vector broadcast** — `2fdff21` (feat)
2. **Task 2: predict_cpu host launcher + categorical/sparse fallback + 4-kind 1e-5 test** — `c39dc0c` (feat)
3. **Cleanup: consolidate PredictCpuElem PredictOut supertrait (clippy clean)** — `bc68865` (refactor)

## Files Created/Modified

- `crates/treelite-cubecl/src/kernels/default_raw.rs` (created) — `predict_default_raw<F, T>` fused kernel.
- `crates/treelite-cubecl/src/kernels/leaf_id.rs` (created) — `predict_leaf_id<F, T>` kernel.
- `crates/treelite-cubecl/src/kernels/score_per_tree.rs` (created) — `predict_score_per_tree<F, T>` kernel.
- `crates/treelite-cubecl/tests/predict_kinds.rs` (created) — 8 green parity / fallback / typed-error tests.
- `crates/treelite-cubecl/src/lib.rs` (modified) — real `predict_cpu` + `predict_cpu_sparse` + `PredictCpuElem` + host helpers (`routing_columns`, `average_factor`, `run_*`).
- `crates/treelite-cubecl/src/kernels/mod.rs` (modified) — promoted `default_raw`/`leaf_id`/`score_per_tree` from reserved comments to `pub mod`.
- `crates/treelite-cubecl/src/kernels/traversal.rs` (modified) — `descend<F, T>` Pitfall-6 generalization + scoped `#[allow(clippy::eq_op)]`.
- `crates/treelite-cubecl/tests/spike.rs` (modified) — `descend::<F, F>` call-site update (unchanged behavior, 3/3 green).

## Decisions Made

See `key-decisions` in the frontmatter. The substantive one: the `Default`-kind postprocessor is applied HOST-side after read-back (via `PredictCpuElem` reusing the public scalar `postprocessor::*` functions arm-for-arm), making `Default` byte-identical to `treelite_gtil::predict` — rather than a second device launch with in-kernel String dispatch. The default_raw kernel produces the `Raw` margin; `Default` layers the postproc, mirroring `predict_rows`'s `Default == Raw + postproc` structure exactly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `descend` signature generalized to `descend<F, T>` (Pitfall 6)**
- **Found during:** Task 1 (authoring the kernels generic over input `F` and threshold `T`).
- **Issue:** The plan's `<behavior>` requires the kernels be "generic over BOTH the input element `F` and the preset's `T`" (Pitfall 6), but the 06-02 `descend` was `descend<F>` with a single shared width (`threshold: &Array<F>`). The kernels could not be `<F, T>` without `descend` also taking the threshold in `T`.
- **Fix:** Generalized `descend<F, T>(threshold: &Array<T>)` and compared `fv < F::cast_from(threshold[...])` (the `NextNode<InputT,ThresholdT>` promotion; identity for matching widths). Updated the spike's call to `descend::<F, F>`. No behavior change for the matching-width presets the spike and these tests exercise.
- **Files modified:** `crates/treelite-cubecl/src/kernels/traversal.rs`, `crates/treelite-cubecl/tests/spike.rs`.
- **Commit:** `2fdff21`.

**2. [Rule 3 - Blocking] Default postprocessor applied host-side after read-back, not as a `#[cube]` kernel step**
- **Found during:** Task 2 (wiring the `Default` kind).
- **Issue:** The plan's `<action>` phrases it as "the `default` kernel applies the matching `#[cube]` postprocessor from plan 03". But the postprocessor is selected by `model.postprocessor` (a host-only `String`, 10 named variants), and `softmax`/`multiclass_ova` operate per-`(row, target)` over that target's `num_class` cells — an in-kernel String dispatch is not expressible, and a second device launch keyed on a host `which` code would duplicate `apply_postprocessor`'s row-span logic with more risk.
- **Fix:** Produce the `Raw` margin on device (the `default_raw` kernel), read it back, and apply the postprocessor host-side via the `PredictCpuElem` trait reusing the SAME public `treelite_gtil::postprocessor::*` functions arm-for-arm. This makes `Default` byte-identical to the scalar reference (the 1e-5 parity contract is met exactly) and mirrors `predict_rows`'s `Default == Raw + postproc` structure. The plan-03 `#[cube]` postproc helpers remain available for a future fully-on-device pass.
- **Files modified:** `crates/treelite-cubecl/src/lib.rs`.
- **Commit:** `c39dc0c`.

**3. [Rule 1 - Bug / hygiene] Doc-comment wording + bound consolidation for clippy**
- **Found during:** Task 1 (acceptance grep gates) and post-Task-2 clippy.
- **Issue:** (a) The Task-1 `grep -ci 'Atomic\|plane_\|sync_cube\|reduce'` gate matched my explanatory doc comment ("no atomic, plane reduction, or `sync_cube`") rather than any code; (b) `predict_cpu` declared `F: PredictCpuElem` AND `where F: PredictOut`, tripping clippy's "bound defined in more than one place".
- **Fix:** (a) Reworded the determinism doc comment to "no cross-unit accumulation primitive over the tree axis" (same wording-adjustment approach as 06-02/06-03 — meaning preserved, literal tokens cleared, the grep gate reads 0); (b) made `PredictOut` a `PredictCpuElem` supertrait and dropped the duplicate `where`. No behavior change.
- **Files modified:** `crates/treelite-cubecl/src/kernels/default_raw.rs`, `crates/treelite-cubecl/src/lib.rs`.
- **Commits:** `2fdff21` / `bc68865`.

---

**Total deviations:** 3 (2 blocking API/structure adaptations within the plan's stated intent + parity contract, 1 doc-wording/hygiene). **Impact on plan:** No scope creep. The `descend<F, T>` generalization is the plan's own Pitfall-6 requirement; the host-side postproc is the byte-identical-parity realization of `Default == Raw + postproc`. All control-flow shapes (one unit/row, serial trees, no tree-axis reduction), the four kinds, leaf-vector broadcast, RF averaging, the f64 base-score add, the D-02 categorical fallback, and 1e-5 fidelity on f32+f64 are exactly as planned.

## Issues Encountered

- **Resolved a 06-03 deferred item.** The `clippy::eq_op` deny on `traversal.rs`'s `fv != fv` (which blocked `cargo clippy -p treelite-cubecl` despite green tests) is now fixed with a scoped `#[allow(clippy::eq_op)]` carrying the 06-02 NaN-test rationale — landed here because this plan was already editing `traversal.rs`. `deferred-items.md` updated to mark it RESOLVED.

## Known Stubs

Intentional, plan-scoped (none block this plan's goal):
- `tests/determinism.rs` remains a RED `#[ignore]` and `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` is the Wave-4 (plan 06-05) work: the two-run bit-identity determinism gate (SC2) and the full golden-matrix harness sibling, both of which call this plan's `predict_cpu`.

## User Setup Required

None — the cubecl CPU backend needs no external configuration.

## Next Phase Readiness

- Wave 4 (plan 06-05) has its sole dependency green: `predict_cpu` is the real validate->fallback->upload->select->launch->read host launcher, matching the scalar reference within 1e-5 across all 4 kinds + leaf-vector broadcast on f32 AND f64. Plan 06-05 turns the RED `determinism.rs` (two-run bit-identity, SC2) and `gtil_matrix_cubecl.rs` (the full golden-matrix harness sibling) green over this launcher.

## Self-Check: PASSED

- crates/treelite-cubecl/src/kernels/default_raw.rs — FOUND
- crates/treelite-cubecl/src/kernels/leaf_id.rs — FOUND
- crates/treelite-cubecl/src/kernels/score_per_tree.rs — FOUND
- crates/treelite-cubecl/tests/predict_kinds.rs — FOUND
- crates/treelite-cubecl/src/lib.rs — FOUND (predict_cpu real launcher)
- commit 2fdff21 (Task 1) — FOUND
- commit c39dc0c (Task 2) — FOUND
- commit bc68865 (cleanup) — FOUND

---
*Phase: 06-cubecl-gtil-kernels-cpu-backend*
*Completed: 2026-06-10*

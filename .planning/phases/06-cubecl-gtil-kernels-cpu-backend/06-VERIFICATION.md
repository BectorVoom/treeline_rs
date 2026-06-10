---
phase: 06-cubecl-gtil-kernels-cpu-backend
verified: 2026-06-11T00:30:00Z
status: passed
score: 6/6
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/6
  gaps_closed:
    - "CR-01: non-kLT operator-coverage fallback gate added — model_routes_to_scalar_fallback routes any model with kLE/kGE/etc to scalar reference before kernel launch"
    - "CR-02: descend() f64-promoted comparison — F::cast_from(threshold) narrowing replaced with f64::cast_from(fv) < f64::cast_from(threshold), matching scalar next_node"
    - "CR-03: validate_leaf_vectors host-side leaf-vector span validation — MalformedLeafVector typed error returned before any client.create_from_slice"
    - "REVIEW CR-01: routing-aware bound in validate_leaf_vectors — four-way branch mirrors default_raw.rs:113-158, no false rejection of valid per-target/per-class models"
    - "WR-01: saturating_sub used for seg_len computation — non-monotonic offset columns yield seg_len=0 instead of panic"
    - "WR-04: model_routes_to_fallback in gtil_matrix_cubecl.rs delegates to exported treelite_cubecl::model_routes_to_scalar_fallback — not a re-derived copy"
    - "lgbm_numerical and mixedwidth upstream goldens captured and committed — matrix gate widened from 96 to 160 cells"
  gaps_remaining: []
  regressions: []
---

# Phase 6: cubecl GTIL Kernels (CPU Backend) — Re-verification Report

**Phase Goal:** Reimplement the GTIL hot path (traversal + postprocessors) as cubecl kernels with the CPU backend as the deterministic default, validated to 1e-5 against the green scalar reference — the project's compute spine widened onto cubecl.
**Verified:** 2026-06-11T00:30:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure plans 06-06 and 06-07

## Summary of Gap Closure

The three BLOCKERs from the prior verification (CR-01 kLT hardcoding, CR-02 f64->f32 narrowing, CR-03 leaf-vector OOB) are all closed. The REVIEW added a second-order CR-01 concern (routing-blind broadcast bound) that was also fixed and locked by a dedicated test. The matrix gate widened from 96 to 160 cells with two new model classes.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Tree traversal and the postprocessor set run as `#[cube(launch)]` kernels generic over `R: Runtime`, with one unit per row looping over trees serially (no `atomicAdd`/reduce over the tree axis, no `continue`) | VERIFIED | `kernels/default_raw.rs`, `leaf_id.rs`, `score_per_tree.rs`: `#[cube(launch)]`, `ABSOLUTE_POS` per-row, inner `for tree_id` serial loop. `cargo test -p treelite-cubecl` passes all 9 predict_kinds + 3 spike + 3 upload + 7 malformed + 2 determinism tests. 0 `continue` in kernel files. |
| 2 | The cubecl CPU backend is the default and the full equivalence harness passes within 1e-5 on it in CI, with output bit-identical across two runs of the same input (determinism check) | VERIFIED | `cargo test -p treelite-harness --test gtil_matrix_cubecl` GREEN: 160 cells (80 f32, 80 f64), 48 cubecl-kernel + 112 scalar-fallback, global max delta 2.907e-6 (< 1e-5). New cells: lgbm_numerical.* (scalar-fallback, kLE, max delta ~5.55e-17) and mixedwidth.* (scalar-fallback, kLE, max delta 0e0). Determinism test: 2/2 bit-identical. D-11 confirmed: `git diff --stat -- crates/treelite-harness/tests/gtil_matrix.rs` = 0 lines. |
| 3 | SoA model buffers upload host->device via `TreeBuf::as_bytes()` + `client.create_from_slice` with per-column ragged-SoA concatenation across the forest (no per-tree handle explosion), and a plain-Rust fallback exists for any unimplemented cubecl op | VERIFIED | Upload confirmed: per-column handles, prefix-sum offset index, `as_bytes()` calls. Fallback gate: `model_routes_to_scalar_fallback` covers categorical AND non-kLT operators — all such models route to `treelite_gtil::predict` before any device op. `cargo test -p treelite-cubecl --test upload` green. |
| 4 | A model containing any non-kLT comparison operator (e.g. LightGBM numerical kLE) is routed to the scalar fallback instead of reaching the kLT-only kernel | VERIFIED | `model_routes_to_scalar_fallback` at `lib.rs:255` scans every tree's `cmp` column across both ModelVariant arms, restricted to internal nodes (`cleft != -1`). Gate at `lib.rs:307` defers to `treelite_gtil::predict`. Matrix gate: 32 lgbm_numerical.* cells all tagged scalar-fallback, all within 1e-5. `has_non_klt_split` confirmed in source. |
| 5 | The dense numerical comparison in `descend()` promotes both operands to f64, matching the scalar reference's f64 promotion | VERIFIED | `f64::cast_from(fv) < f64::cast_from(threshold[...])` at `traversal.rs:104`. `F::cast_from(threshold` removed: `grep -c 'F::cast_from(threshold' crates/treelite-cubecl/src/kernels/traversal.rs` = 0. Self-contained regression `f64_threshold_f32_input_routes_like_scalar` (predict_kinds.rs:185) passes: f64-preset split at 0.1, f32 input straddling, cubecl == scalar routing. |
| 6 | A malformed Model with an out-of-range leaf_vector span returns a typed `CubeclError::MalformedLeafVector` before any device op (T-06-09 no-OOB contract) | VERIFIED | `validate_leaf_vectors` at `upload.rs:304` called from `upload_forest:426` before first `create_from_slice` at `upload.rs:432`. Routing-aware four-way bound mirrors `default_raw.rs:113-158`. 7/7 malformed tests pass, including: end-past-segment, inverted span, broadcast overrun, non-monotonic offset (WR-01 saturating), well-formed pass, per-target multiclass pass (REVIEW CR-01 false-rejection regression), per-class multiclass pass. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-cubecl/src/lib.rs` | predict_cpu + model_routes_to_scalar_fallback (operator gate) | VERIFIED | has_non_klt_split scanning cmp column; gate at line 307; model_routes_to_scalar_fallback pub exported at line 255 |
| `crates/treelite-cubecl/src/kernels/traversal.rs` | f64-promoted comparison in descend() | VERIFIED | `f64::cast_from(fv) < f64::cast_from(threshold[...])` at line 104; F::cast_from(threshold) absent; incorrect doc comment corrected |
| `crates/treelite-cubecl/tests/predict_kinds.rs` | f64_threshold_f32_input_routes_like_scalar CR-02 regression | VERIFIED | Test at line 185; passes in isolation; F64-preset split at 0.1, f32 input straddling |
| `crates/treelite-cubecl/src/upload.rs` | validate_leaf_vectors + routing-aware bound | VERIFIED | Function at line 304; called at line 426 before first create_from_slice; four-way routing branch; saturating_sub for WR-01 |
| `crates/treelite-cubecl/src/error.rs` | MalformedLeafVector typed variant | VERIFIED | At line 69; thiserror message with tree/node/begin/end/segment_len fields |
| `crates/treelite-cubecl/tests/malformed.rs` | 7 malformed tests incl. routing-aware regression | VERIFIED | 7/7 pass; includes validate_leaf_vectors_accepts_per_target_multiclass (REVIEW CR-01) and validate_leaf_vectors_rejects_non_monotonic_offset_without_panic (WR-01) |
| `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` | model_routes_to_fallback delegating to exported function | VERIFIED | Line 339: `treelite_cubecl::model_routes_to_scalar_fallback(model)` — single source of truth, no re-derived copy (WR-04 fixed) |
| `fixtures/gtil/lgbm_numerical.model.bin` | frozen kLE LightGBM model (non-empty) | VERIFIED | 20666 bytes; `test -s` passes |
| `fixtures/gtil/mixedwidth.model.bin` | frozen mixed-width model with 0.1 threshold (non-empty) | VERIFIED | 511 bytes; `test -s` passes |
| `fixtures/gtil/lgbm_numerical.*.golden.json` | 32 golden cells captured from upstream Treelite | VERIFIED | 32 files confirmed; all within 1e-5 (worst ~5.55e-17) |
| `fixtures/gtil/mixedwidth.*.golden.json` | 32 golden cells captured from upstream Treelite | VERIFIED | 32 files confirmed; all within 1e-5 (worst 0e0) |
| `fixtures/capture_gtil_matrix.py` | build_lgbm_numerical_model + build_mixedwidth_model | VERIFIED | grep count >= 4 confirmed |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `lib.rs` | `treelite_gtil::predict` | `model_routes_to_scalar_fallback` gate | WIRED | Lines 255+307; categorical OR non-kLT routes whole model to scalar reference |
| `upload.rs` | `error.rs` | `validate_leaf_vectors` returns `CubeclError::MalformedLeafVector` | WIRED | Lines 304/369/382 return MalformedLeafVector; called before any create_from_slice |
| `gtil_matrix_cubecl.rs` | `treelite_cubecl::model_routes_to_scalar_fallback` | `model_routes_to_fallback` wrapper at line 339 | WIRED | Single delegating call; no re-derived predicate (WR-04 closed) |
| `fixtures/gtil/lgbm_numerical.*.golden.json` | `gtil_matrix_cubecl.rs` | auto-discovered *.golden.json | WIRED | 32 cells loaded, asserted, tagged scalar-fallback in matrix output |
| `fixtures/gtil/mixedwidth.*.golden.json` | `gtil_matrix_cubecl.rs` | auto-discovered *.golden.json | WIRED | 32 cells loaded, asserted, tagged scalar-fallback in matrix output |
| `crates/treelite-harness/tests/gtil_matrix.rs` | (unchanged) | D-11 registration-not-refactor | VERIFIED | `git diff --stat` = 0 lines; confirmed by git log |
| `predict_kinds.rs` | `predict_cpu::<f32>` and `treelite_gtil::predict::<f32>` | `f64_threshold_f32_input_routes_like_scalar` | WIRED | Test asserts cubecl routes same child as scalar on 0.1 boundary |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `gtil_matrix_cubecl.rs` | golden output vectors | `fixtures/gtil/*.golden.json` (160 cells) | Yes — frozen from C++ Treelite via `uv run python fixtures/capture_gtil_matrix.py` | FLOWING (160 cells: binary/leaf_vec_mc/large_margin/lgbm_numerical/mixedwidth) |
| `lgbm_numerical.*.golden.json` | kLE prediction output | LightGBM model via `treelite.frontend.load_lightgbm_model` → scalar reference | Yes — upstream Treelite 1e-5 contract | FLOWING (worst delta ~5.55e-17) |
| `mixedwidth.*.golden.json` | mixed-width prediction output | ModelBuilder <f64,f64> split at 0.1 → scalar reference | Yes — upstream Treelite, 0.1 is f32-unrepresentable | FLOWING (worst delta 0e0) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| treelite-cubecl full test suite (24 tests across 5 binaries) | `cargo test -p treelite-cubecl` | 24/24 pass (predict_kinds 9, spike 3, upload 3, malformed 7, determinism 2) | PASS |
| gtil_matrix_cubecl 160-cell gate (all model classes) | `cargo test -p treelite-harness --test gtil_matrix_cubecl` | 1/1 pass; 160 cells; max delta 2.907e-6 | PASS |
| Full workspace test suite | `cargo test --workspace` | All pass, 0 failures | PASS |
| lgbm_numerical cells tagged scalar-fallback (CR-01 end-to-end) | matrix --nocapture output | 32 lgbm_numerical.* cells, all (scalar-fallback), worst delta ~5.55e-17 | PASS |
| mixedwidth cells within 1e-5 (CR-02 end-to-end) | matrix --nocapture output | 32 mixedwidth.* cells, all (scalar-fallback), worst delta 0e0 | PASS |
| kernel_cells > 0 guard still holds | matrix --nocapture summary line | 48 cubecl-kernel cells (XGBoost dense kLT) | PASS |
| D-11: gtil_matrix.rs untouched | `git diff --stat -- crates/treelite-harness/tests/gtil_matrix.rs` | 0 lines changed | PASS |
| F::cast_from(threshold) narrowing removed | `grep -c 'F::cast_from(threshold' traversal.rs` | 0 matches | PASS |
| CR-02 self-contained test | `cargo test -p treelite-cubecl --test predict_kinds f64_threshold_f32_input_routes_like_scalar` | PASS | PASS |
| Routing-aware false-rejection regression | `cargo test -p treelite-cubecl --test malformed validate_leaf_vectors_accepts_per_target_multiclass` | PASS | PASS |

### Probe Execution

No probe scripts declared or conventional in this phase. Step 7c: SKIPPED (no probe-*.sh present).

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| GPU-01 | 06-01..06-06 | GTIL inference hot path (traversal + postprocessors) implemented as cubecl kernels | SATISFIED | `#[cube(launch)]` kernels: default_raw.rs, leaf_id.rs, score_per_tree.rs. descend() in traversal.rs. 10 postprocessors in postproc.rs. All compile and run on CpuRuntime. kLT models traverse in-kernel; non-kLT ride the proven scalar fallback (D-02, approach A). |
| GPU-02 | 06-04..06-07 | The cubecl CPU backend is the default and is validated to 1e-5 | SATISFIED | 160-cell matrix gate GREEN. Covers: XGBoost kLT (cubecl-kernel, 48 cells), LightGBM kLE (scalar-fallback, 32 cells), mixed-width 0.1 threshold (scalar-fallback, 32 cells), leaf-vector multiclass, large-margin sigmoid. Global max delta 2.907e-6. Determinism: 2/2 bit-identical. |
| GPU-05 | 06-01..06-05 | SoA model buffers upload host->device zero-copy | SATISFIED | TreeBuf::as_bytes() + client.create_from_slice; one handle per column for the whole forest; no per-tree explosion. upload tests 3/3 green. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-cubecl/src/lib.rs` | 432-439, 474, 505-530, 573-604 | `bytemuck::cast_slice` panics if byte length not a multiple of size_of::<T>() (WR-03, carried from prior review — unchanged) | Warning | Latent panic path on mis-aligned input; low trigger probability for host-owned `Vec<T>: Pod`. Not a BLOCKER: the panic path requires a layout invariant violation that host-built vectors do not produce. Documented but not yet resolved. |
| `crates/treelite-cubecl/src/kernels/default_raw.rs` / `src/lib.rs` | Various | Device postprocessor kernels (`postproc.rs`) are tested to 1e-5 but `predict_cpu` uses the host `apply_postprocessor` — doc comments were previously misleading (WR-05). Comments updated; behavior unchanged. | Info | Noted; no correctness impact. Device kernels exist and are regression-tested; host postprocessor runs in production. |

No BLOCKER anti-patterns remain. All three prior BLOCKERs (CR-01, CR-02, CR-03) are closed and regression-locked. WR-03 (bytemuck panic paths) is a pre-existing advisory warning with no feasible trigger on the CPU backend from host-built data; the crate's no-panic discipline is aspirational for this edge case.

### Human Verification Required

None. All verification is mechanical and has been performed via `cargo test`. The phase delivers CPU-backend-only cubecl validation; GPU backend validation is Phase 7.

---

## Gaps Summary

**No gaps.** All six observable truths are VERIFIED.

The three BLOCKERs from the prior `gaps_found` report are closed:
- CR-01: `model_routes_to_scalar_fallback` gates any non-kLT model to the scalar reference before kernel launch. The REVIEW's follow-on concern (routing-blind broadcast bound over-rejecting valid multi-target models) was also fixed: `validate_leaf_vectors` now uses a routing-aware four-way branch matching `default_raw.rs:113-158`, locked by `validate_leaf_vectors_accepts_per_target_multiclass` in `malformed.rs`.
- CR-02: `descend()` compares both operands in f64. `F::cast_from(threshold)` narrowing is completely absent from the kernel files (grep confirms 0 matches). Self-contained regression `f64_threshold_f32_input_routes_like_scalar` locks it inside 06-06, independent of upstream fixtures.
- CR-03: `validate_leaf_vectors` runs host-side before any `client.create_from_slice`. WR-01 saturating subtraction prevents panic on non-monotonic offset columns. 7 malformed tests all pass.

The matrix gate is now 160 cells (up from 96), covering XGBoost (kLT, cubecl-kernel), LightGBM (kLE, scalar-fallback), mixed-width 0.1 threshold (scalar-fallback), multiclass leaf-vector, and large-margin sigmoid. All cells pass within 1e-5 of frozen upstream Treelite goldens. Provenance is honest per cell (D-06): `model_routes_to_fallback` delegates to the single exported `model_routes_to_scalar_fallback` function — no re-derived predicate (WR-04 fixed). `gtil_matrix.rs` is byte-unchanged (D-11). The phase goal is achieved.

---

_Verified: 2026-06-11T00:30:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes (previous: gaps_found 3/6 -> current: passed 6/6)_

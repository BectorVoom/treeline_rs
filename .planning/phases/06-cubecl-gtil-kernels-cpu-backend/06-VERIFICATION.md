---
phase: 06-cubecl-gtil-kernels-cpu-backend
verified: 2026-06-10T12:00:00Z
status: gaps_found
score: 4/6 must-haves verified
overrides_applied: 0
gaps:
  - truth: "The cubecl CPU backend is the default and the full equivalence harness passes within 1e-5 on it in CI, with output bit-identical across two runs of the same input (determinism check)"
    status: partial
    reason: "The gtil_matrix_cubecl harness passes on 96 XGBoost-only (kLT-operator) fixtures. However, the kLT hardcoding in descend() means any model loaded with a non-kLT operator (every LightGBM numerical model uses kLE, confirmed at treelite-lightgbm/src/lib.rs:273) silently reaches the cubecl kernel and produces definitively wrong tie-routing — not a 1e-5 rounding drift. predict_cpu's fallback gate (lib.rs:262-269) only checks has_categorical_split, never the per-node comparison operator. The project's core value ('predictions match upstream Treelite within 1e-5') and the phase goal ('validated to 1e-5 against the green scalar reference') are not satisfied for the full set of model classes this project supports."
    artifacts:
      - path: "crates/treelite-cubecl/src/kernels/traversal.rs:95-101"
        issue: "descend() unconditionally executes fv < F::cast_from(threshold[...]) — the kLT predicate — regardless of the per-node Operator stored in the model. The comment on line 39 and 95 acknowledges 'XGBoost always kLT' but does not gate the kernel to kLT models. The scalar twin (treelite-gtil/src/lib.rs:341-354) dispatches on kLT/kLE/kEQ/kGT/kGE per node."
      - path: "crates/treelite-cubecl/src/lib.rs:262-269"
        issue: "Fallback gate checks only has_categorical_split. A numerical LightGBM model (kLE operators, non-categorical) is NOT intercepted here — it reaches the kernel and every fv==threshold boundary routes right (kLT false) instead of left (kLE true)."
    missing:
      - "Either add an operator-check gate in predict_cpu that routes any model containing non-kLT operators to the scalar fallback (analogous to the categorical gate), OR upload the cmp column and branch on it in descend<F,T>"
      - "Add a kLE fixture (e.g., a LightGBM numerical model cell) to fixtures/gtil/ so the matrix actually covers non-kLT operators"

  - truth: "Dense numerical traversal runs correctly for the supported preset combinations (including f32 input / f64 preset) matching the scalar reference"
    status: failed
    reason: "CR-02: descend() compares fv < F::cast_from(threshold[...]), casting the f64 threshold DOWN to the input width F (e.g., f32) before comparing. The scalar reference (treelite-gtil/src/lib.rs:326-331) promotes BOTH operands to f64. For the f32-input/f64-preset combination (binary.f32.f64.* fixtures exist), this is a lossy f64->f32 narrowing that can route differently when the f64 threshold falls between two adjacent f32 values — a definite wrong leaf, not a drift. The traversal.rs doc comment (lines 56-59) incorrectly describes this as 'usual-arithmetic-conversion promotion' — that promotion widens to f64, never narrows to f32. The current fixtures do not expose a threshold at an f32 boundary, so the gate passes green while the bug ships."
    artifacts:
      - path: "crates/treelite-cubecl/src/kernels/traversal.rs:99"
        issue: "fv < F::cast_from(threshold[(base + nid) as usize]) performs f64->f32 narrowing for F=f32. Scalar reference: next_node(fvalue.to_compare_f64(), threshold.threshold_to_f64(), op, ...) — both operands widened to f64."
    missing:
      - "Change to: if f64::cast_from(fv) < f64::cast_from(threshold[(base + nid) as usize]) — compare both operands in f64 regardless of F and T widths"
      - "Add a mixed-width fixture whose f64 threshold falls between two adjacent f32 representable values (e.g., threshold = 0.1_f64, which is not exactly representable as f32) to lock this down"

  - truth: "predict_cpu validates shapes and ensures no OOB device reads on malformed models (T-06-09)"
    status: failed
    reason: "CR-03: leaf-vector broadcast loops in default_raw.rs and score_per_tree.rs read leaf_vector[(lv_base + li) as usize] (and similar indices) with no host-side validation that lv_base + li lies within the uploaded leaf_vector column. validate_shape() in upload.rs:224-262 checks only split_index bounds and input buffer length — it never validates that leaf_vector_begin/end spans lie within the per-tree leaf_vector segment. A malformed Model with a short leaf vector or out-of-range begin/end offsets performs an out-of-bounds device read in-kernel. The scalar twin (treelite-gtil/src/lib.rs:801-868) bounds-checks every leaf-vector access and returns GtilError::LeafVectorTooShort. The T-06-09 contract ('no OOB device op on a malformed model') claimed in lib.rs:241-247 is not upheld for leaf-vector paths."
    artifacts:
      - path: "crates/treelite-cubecl/src/kernels/default_raw.rs:113-156"
        issue: "leaf_vector[(lv_base + li) as usize] — lv_base + li is not bounded against num_leafvec_total before the kernel is launched"
      - path: "crates/treelite-cubecl/src/kernels/score_per_tree.rs:69-87"
        issue: "leaf_vector[(lv_base + i) as usize] — same unbounded access pattern"
      - path: "crates/treelite-cubecl/src/upload.rs:224-262"
        issue: "validate_shape() does not check leaf_vector_begin/end values against the per-tree leaf-vector segment length"
    missing:
      - "Add host-side validation in validate_shape() (or a new validate_leaf_vectors() called from upload_forest()): for every leaf node assert leaf_vector_end[n] <= tree_leaf_vector_len[t] and that the (num_target, max_num_class) broadcast span fits the leaf vector; return a typed CubeclError (add LeafVectorTooShort / MalformedLeafVector variant)"
      - "Add a test that passes a Model with a short leaf_vector and asserts a typed CubeclError is returned instead of an OOB device read"
---

# Phase 6: cubecl GTIL Kernels (CPU Backend) Verification Report

**Phase Goal:** Reimplement the GTIL hot path (traversal + postprocessors) as cubecl kernels with the CPU backend as the deterministic default, validated to 1e-5 against the green scalar reference — the project's compute spine widened onto cubecl.
**Verified:** 2026-06-10T12:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Tree traversal and the postprocessor set run as `#[cube(launch)]` kernels generic over `R: Runtime`, with one unit per row looping over trees serially (no `atomicAdd`/reduce over the tree axis, no `continue`) | ✓ VERIFIED | `kernels/default_raw.rs`, `leaf_id.rs`, `score_per_tree.rs`: `#[cube(launch)]`, `ABSOLUTE_POS` per-row, inner `for tree_id in 0..num_tree` serial loop; grep confirms 0 `continue` and 0 atomic/sync_cube in the kernel files. `cargo test --workspace` green. |
| 2 | The cubecl CPU backend is the default and the full equivalence harness passes within 1e-5 on it in CI, with output bit-identical across two runs of the same input (determinism check) | ✗ FAILED (partial) | The gtil_matrix_cubecl passes on all 96 XGBoost-only (kLT) fixtures (max delta 2.9e-6); determinism.rs green. BUT: the harness only exercises XGBoost models. descend() hardcodes kLT — LightGBM numerical (kLE) models silently reach the kernel and produce wrong tie-routing. The phase goal says "validated to 1e-5 against the green scalar reference"; that reference covers kLE/kGE operators. See CR-01. |
| 3 | SoA model buffers upload host→device via `TreeBuf::as_bytes()` + `client.create_from_slice` with per-column ragged-SoA concatenation across the forest (no per-tree handle explosion), and a plain-Rust fallback exists for any unimplemented cubecl op | ✗ FAILED (partial) | Upload is verified (one handle per column, prefix-sum offset index, cargo test upload green). TreeBuf::as_bytes() verified. Fallback EXISTS for categorical/sparse. BUT: non-kLT operators (kLE, kGE) are silently accepted by the kernel with wrong routing — they are not "unimplemented cubecl ops" with a fallback; they are mis-implemented ops. SC-3's "fallback exists for any unimplemented cubecl op" is violated. |
| 4 | Per-column ragged-SoA upload round-trips correctly for all column types | ✓ VERIFIED | `cargo test -p treelite-cubecl --test upload` passes. validate_shape() correctly rejects bad split_index before device ops. |
| 5 | All four predict kinds + leaf-vector broadcast run in-kernel on f32 AND f64 input and match scalar reference within 1e-5 (for the XGBoost-kLT model class) | ✓ VERIFIED (narrowed scope) | `cargo test -p treelite-cubecl --test predict_kinds` 8/8 pass. predict_cpu host launcher: validate→upload→select→launch→read confirmed. Categorical fallback confirmed. NOTE: all test fixtures use kLT only. |
| 6 | predict_cpu validates shapes and ensures no OOB device reads on malformed models (T-06-09) | ✗ FAILED | validate_shape() checks split_index and input buffer size, but does NOT validate leaf_vector_begin/end spans. Malformed models with short leaf vectors cause unbounded OOB device reads in default_raw.rs:113-156 and score_per_tree.rs:69-87. The scalar twin returns a typed LeafVectorTooShort error. See CR-03. |

**Score:** 3/6 truths fully verified (1 additional truth partially verified with narrowed scope)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-cubecl/Cargo.toml` | New kernel crate manifest with cubecl(cpu)/bytemuck/treelite-core/treelite-gtil/thiserror deps | ✓ VERIFIED | Exists, substantive, workspace member registered |
| `crates/treelite-cubecl/src/lib.rs` | predict_cpu host-launcher + module exports | ✓ VERIFIED | 626 lines, real implementation; categorical/sparse fallback wired |
| `crates/treelite-cubecl/src/error.rs` | CubeclError thiserror enum | ✓ VERIFIED | Exists, non-stub thiserror enum |
| `crates/treelite-cubecl/src/kernels/traversal.rs` | `#[cube]` break-free descend + next-node helper | ✓ VERIFIED (with known CR-01/CR-02 defects) | 106 lines, real `#[cube]` implementation; defects in operator dispatch and comparison width |
| `crates/treelite-cubecl/src/kernels/default_raw.rs` | `#[cube(launch)]` traversal+accumulate(+postproc for default) kernel | ✓ VERIFIED (with known CR-03 defect) | 207 lines, real kernel; leaf-vector OOB risk |
| `crates/treelite-cubecl/src/kernels/leaf_id.rs` | `#[cube(launch)]` leaf id kernel | ✓ VERIFIED | 63 lines, real kernel |
| `crates/treelite-cubecl/src/kernels/score_per_tree.rs` | `#[cube(launch)]` score per tree kernel | ✓ VERIFIED (with known CR-03 defect) | 90 lines, real kernel; leaf-vector OOB risk |
| `crates/treelite-cubecl/src/kernels/postproc.rs` | 10 `#[cube]` postprocessor ports | ✓ VERIFIED | 284 lines, all 10 postprocessors, tested to 1e-5 vs scalar twins |
| `crates/treelite-cubecl/src/upload.rs` | Per-column ragged-SoA upload + prefix-sum index | ✓ VERIFIED (with CR-03 gap) | 301 lines, real implementation; missing leaf-vector span validation |
| `crates/treelite-core/src/tree_buf.rs` | TreeBuf::as_bytes() additive accessor | ✓ VERIFIED | Exists, gated on T: Pod, round-trip tests pass |
| `crates/treelite-harness/src/lib.rs` | Backend::CubeclCpu + cubecl_cpu_case() | ✓ VERIFIED | CubeclCpu variant exists; cubecl_cpu_case() wires dense to predict_cpu, sparse to fallback |
| `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` | Sibling matrix gate: frozen goldens to 1e-5 on cubecl + per-cell provenance + SC2 determinism | ✓ VERIFIED (narrowed scope — XGBoost only) | 459 lines, real gate; 96 cells pass; per-cell provenance recorded; kernel_cells > 0 guard in place |
| `crates/treelite-cubecl/tests/determinism.rs` | Two-run .to_bits() bit-identity (SC2) | ✓ VERIFIED | 133 lines; f32 + f64, all 4 predict kinds; 2/2 pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `Cargo.toml [workspace.members]` | `crates/treelite-cubecl` | member registration | ✓ WIRED | Confirmed at root Cargo.toml line 11 |
| `crates/treelite-core/src/tree_buf.rs` | `bytemuck::cast_slice` | as_bytes zero-copy view | ✓ WIRED | `as_bytes()` calls `bytemuck::cast_slice(self.as_slice())` |
| `crates/treelite-cubecl/tests/spike.rs` | `treelite_gtil::predict` | 1e-5 assertion against scalar reference | ✓ WIRED | spike.rs imports and asserts against treelite_gtil::predict |
| `crates/treelite-cubecl/src/kernels/traversal.rs` | cubecl Float associated fns | `fv != fv` NaN test, no continue, if-statement routing | ✓ WIRED (with CR-01/CR-02 defects) | NaN via self-inequality; no continue; if-statement child selection |
| `crates/treelite-cubecl/src/upload.rs` | `treelite_core::TreeBuf::as_bytes` | zero-copy per-column byte view | ✓ WIRED | upload.rs uses as_bytes() for numeric columns |
| `crates/treelite-cubecl/src/kernels/postproc.rs` | `treelite_gtil::postprocessor` | verbatim cast-order reproduction asserted to 1e-5 | ✓ WIRED (test-only, not wired in predict_cpu) | postproc.rs tests pass vs scalar twins; NOTE: actual predict_cpu uses host apply_postprocessor, not device kernels |
| `crates/treelite-harness/src/lib.rs` | `treelite_cubecl::predict_cpu` | cubecl_cpu_case dense slots | ✓ WIRED | cubecl_cpu_case() calls predict_cpu::<f32> and predict_cpu::<f64> |
| `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` | `fixtures/gtil/*.golden.json` | 1e-5 assertion against frozen goldens | ✓ WIRED | 96 cells loaded and asserted; XGBoost-only fixture coverage |
| `crates/treelite-cubecl/src/lib.rs` | `treelite_gtil::predict_sparse` | categorical/sparse scalar fallback (D-02) | ✓ WIRED | has_categorical_split gate present; predict_cpu_sparse routes all-sparse to fallback |
| `crates/treelite-cubecl/src/lib.rs` | kLE/kGE operator routing | fallback for non-kLT models | ✗ NOT_WIRED | No gate for non-kLT operators. LightGBM numerical (kLE) reaches kernel with wrong routing. CR-01. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `gtil_matrix_cubecl.rs` | golden output vectors | `fixtures/gtil/*.golden.json` (96 cells, XGBoost) | Yes — frozen from C++ Treelite | ✓ FLOWING (for XGBoost-kLT subset) |
| `predict_cpu` postprocessor | `apply_postprocessor` output | scalar `treelite_gtil::postprocessor::*` (host CPU) | Yes | ✓ FLOWING — note: device postproc kernels in postproc.rs are exercised only by tests/postproc.rs, NOT wired into predict_cpu (WR-05) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds with treelite-cubecl registered | `cargo build --workspace` | 0 errors | ✓ PASS |
| gtil_matrix_cubecl golden gate (96 XGBoost cells) | `cargo test -p treelite-harness --test gtil_matrix_cubecl` | 1/1 pass, max delta 2.9e-6 | ✓ PASS |
| Determinism SC2 two-run bit-identity | `cargo test -p treelite-cubecl --test determinism` | 2/2 pass | ✓ PASS |
| predict_kinds all 4 kinds + f32/f64 + fallback | `cargo test -p treelite-cubecl --test predict_kinds` | 8/8 pass | ✓ PASS (kLT-only fixtures) |
| Full workspace suite | `cargo test --workspace` | All pass, 0 failures | ✓ PASS |

### Probe Execution

No probe scripts declared or conventional in this phase. Step 7c: SKIPPED (no probe-*.sh present).

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| GPU-01 | 06-01, 06-02, 06-03, 06-04 | GTIL inference hot path (traversal + postprocessors) implemented as cubecl kernels | ✓ SATISFIED (narrowed) | Traversal: traversal.rs. Postprocessors: postproc.rs (device kernels exist and are tested). Launch kernels: default_raw.rs, leaf_id.rs, score_per_tree.rs. All compile and run on CpuRuntime. **Narrowing:** kLT-only correct; kLE/kGE operators silently produce wrong routing (CR-01). |
| GPU-02 | 06-04, 06-05 | The cubecl CPU backend is the default and is validated to 1e-5 | ✗ BLOCKED | The gtil_matrix_cubecl gate passes on 96 XGBoost-only cells. The harness does not include any LightGBM (kLE) or other non-kLT models. LightGBM numerical models reach the kernel and produce definitively wrong routing at kLE ties. "Validated to 1e-5" is not achieved for the project's full supported model set. CR-01 + CR-02. |
| GPU-05 | 06-01, 06-03, 06-05 | SoA model buffers upload host→device zero-copy | ✓ SATISFIED | TreeBuf::as_bytes() + client.create_from_slice; per-column ragged-SoA concatenation; no per-tree handle explosion. Upload round-trip tests pass. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-cubecl/src/kernels/traversal.rs` | 95-101 | `fv < F::cast_from(threshold[...])` — kLT hardcoded, kLE/kGE silently wrong | 🛑 Blocker | Any non-kLT model (all LightGBM numerical) reaches kernel and produces wrong routing at ties. Violates phase goal and GPU-02. (CR-01) |
| `crates/treelite-cubecl/src/kernels/traversal.rs` | 99 | `F::cast_from(threshold[...])` — f64→f32 lossy narrowing for f32 input/f64 preset | 🛑 Blocker | Near an f32-unrepresentable f64 threshold, routing diverges from scalar reference. (CR-02) |
| `crates/treelite-cubecl/src/kernels/default_raw.rs` | 113-156 | `leaf_vector[(lv_base + li) as usize]` — no host-side leaf-vector span validation | 🛑 Blocker | OOB device read on malformed Model; scalar twin returns typed error. T-06-09 contract violated. (CR-03) |
| `crates/treelite-cubecl/src/kernels/score_per_tree.rs` | 69-87 | `leaf_vector[(lv_base + i) as usize]` — same OOB risk | 🛑 Blocker | Same root cause as CR-03 in default_raw.rs |
| `crates/treelite-cubecl/src/upload.rs` | 224-262 | `validate_shape()` missing leaf_vector_begin/end validation | 🛑 Blocker | Root cause of CR-03; validation does not cover leaf-vector span correctness |
| `crates/treelite-cubecl/src/kernels/traversal.rs` | 56-59 | Doc comment incorrectly describes the comparison as "usual-arithmetic-conversion promotion" — the code narrows (f64→f32), not widens | ⚠️ Warning | The comment rationalizes CR-02 with incorrect semantics; misleads future readers |
| `crates/treelite-cubecl/src/upload.rs` | 154, 294 | `node_type` i32-discriminant column uploaded on every call but never read by any kernel | ⚠️ Warning | Wasted device allocation + copy per call; meanwhile `cmp` (the operator column that would fix CR-01) is not uploaded (WR-02) |
| `crates/treelite-cubecl/src/lib.rs` | 428-429, 482-483, 553-554 | `bytemuck::cast_slice` panics if byte length not a multiple of size_of::<F>() | ⚠️ Warning | Latent panic path contradicts the crate's "never panic" error discipline (WR-03); unlikely on CPU backend |
| `crates/treelite-cubecl/src/kernels/default_raw.rs` | 16-19; `src/lib.rs` | Doc comments describe postprocessor as "a separate device step" but it runs on the host CPU | ℹ️ Info | Overstates device coverage; device postproc kernels in postproc.rs are test-only fixtures (WR-05) |

### Human Verification Required

None. All remaining verification is mechanical (code correctness of the identified fixes).

---

## Gaps Summary

Three BLOCKERs prevent the phase goal from being achieved:

**CR-01 (kLT hardcoding, most critical):** `descend()` in `traversal.rs:95-101` unconditionally applies `fv < threshold` (strict less-than, kLT). The scalar reference dispatches on the per-node `Operator` field (`next_node()`, `treelite-gtil/src/lib.rs:341-354`). LightGBM — a first-class supported loader in this project — always emits `Operator::kLE` for every numerical split (`treelite-lightgbm/src/lib.rs:273`). A numerical LightGBM model has `has_categorical_split = false` and therefore passes the categorical-fallback gate unchecked, reaching the kernel where every `fv == threshold` tie routes right (kLT false) instead of left (kLE true). This is a definite wrong prediction on a supported model class, not a 1e-5 rounding drift. The golden matrix passes green only because all 96 fixtures are XGBoost-derived (all kLT). Fix: add an operator-coverage gate in `predict_cpu` routing any model with non-kLT nodes to the scalar fallback, or upload and dispatch on the `cmp` column in `descend`.

**CR-02 (comparison width, mixed-preset):** `F::cast_from(threshold[...])` narrows a f64 threshold to f32 before comparison. The scalar reference promotes both operands to f64. For the `f32` input / `f64` preset combination (24 dense cells in the current fixture matrix run through the kernel), a threshold value not representable as f32 can route to the wrong child. The existing `binary.f32.f64` fixtures do not expose an unrepresentable threshold, so the gate passes green while the bug is present. Fix: compare both operands as `f64::cast_from(fv) < f64::cast_from(threshold[...])`.

**CR-03 (leaf-vector OOB):** `leaf_vector[(lv_base + li) as usize]` in `default_raw.rs:113-156` and `score_per_tree.rs:69-87` is unbounded. `validate_shape()` does not validate `leaf_vector_begin/end` spans. A malformed Model causes an OOB device read in-kernel; the scalar reference returns a typed `LeafVectorTooShort` error. The T-06-09 "no OOB device op on a malformed model" security contract is violated. Fix: validate leaf-vector spans host-side in `upload_forest` before any `client.create_from_slice`.

These three gaps share a single root pattern: the kernel validation and routing logic was authored and tested against XGBoost-only models (kLT, no mixed-width boundary cases, well-formed leaf vectors), leaving other supported model classes in a silently-wrong or unsafe state.

**All other claims verified:** crate scaffold, cubecl 0.10.0 pinning, TreeBuf::as_bytes(), SoA upload, all 10 postprocessors to 1e-5 (device kernels tested; host apply_postprocessor wired in predict_cpu), all 4 predict kinds (on kLT models), determinism (SC2), per-cell provenance (D-06), gtil_matrix_cubecl sibling, registration-not-refactor (D-11 — gtil_matrix.rs untouched), full workspace green.

---

_Verified: 2026-06-10T12:00:00Z_
_Verifier: Claude (gsd-verifier)_

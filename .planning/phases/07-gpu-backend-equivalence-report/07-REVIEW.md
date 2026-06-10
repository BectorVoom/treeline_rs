---
phase: 07-gpu-backend-equivalence-report
reviewed: 2026-06-10T22:21:42Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - crates/treelite-cubecl/Cargo.toml
  - crates/treelite-cubecl/src/device.rs
  - crates/treelite-cubecl/src/error.rs
  - crates/treelite-cubecl/src/lib.rs
  - crates/treelite-cubecl/tests/device_absent.rs
  - crates/treelite-harness/Cargo.toml
  - crates/treelite-harness/src/lib.rs
  - crates/treelite-harness/src/report.rs
  - crates/treelite-harness/tests/gpu_crossover.rs
  - crates/treelite-harness/tests/gtil_matrix_gpu.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-06-10T22:21:42Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Phase 07 generalizes the cubecl launcher from `CpuRuntime` to a generic `R: Runtime`, adds
`rocm`/`cuda`/`wgpu` cargo features (default cpu-only), a typed `CubeclError::DeviceUnavailable`
device-absence skip, per-backend `device::client::<R>()` constructors, harness `Backend::Rocm/Cuda/Wgpu`
variants + `*_case()` constructors, and a report-mode GPU equivalence/crossover artifact generator.

The core numeric path is sound: the `predict::<R, F>` generalization is a faithful lift of the proven
CPU launcher (the kernels/upload/postprocessor arms are byte-unchanged from Phase 06, confirmed by diff),
the `model_routes_to_scalar_fallback` gate runs before client construction, and there is no silent
CPU-fallback branch on device absence — `DeviceUnavailable` propagates via `?`. The CPU 1e-5 gate is
untouched. No Critical issues found in the GTIL/predict numeric path.

The findings concentrate in the **error-discrimination and report-provenance** seam — the part of the
phase that decides what counts as a "skip" vs a real failure, and what gets recorded as a GPU
measurement. Because the GPU report is the committed deliverable (GPU-04) and is observational
(never gates), several defects there can let a genuine GPU correctness problem masquerade as a benign
skip or vanish from the artifact entirely. None corrupts the CPU baseline, hence no Critical, but all
four Warnings weaken the very fidelity-provenance the phase exists to establish.

## Warnings

### WR-01: `device::client::<R>` maps EVERY panic to `DeviceUnavailable`, masking real GPU failures

**File:** `crates/treelite-cubecl/src/device.rs:49-52`
**Issue:** The `catch_unwind` wrapper converts *any* panic during client construction into
`CubeclError::DeviceUnavailable { backend }`, discarding the panic payload entirely
(`.map_err(|_| ...)`). The A3 spike only validated that a *missing-device* panic is catchable — but
this code cannot distinguish a missing-device panic from a genuine driver fault, OOM, an internal
cubecl assertion, or any other bug that unwinds through construction. On the report path
(`gtil_matrix_gpu.rs`), a `DeviceUnavailable` error is treated as "not run — no device (skip)" and the
test PASSES. So a real GPU initialization bug on a machine that *does* have a device would be silently
reported as a benign absence-skip rather than surfaced — directly undermining the 1e-5 provenance the
phase is meant to establish. The panic hook still prints to stderr, but the semantic outcome is a
false skip.
**Fix:** Either narrow the mapping (inspect the panic payload string for the known device-load
signature before classifying as `DeviceUnavailable`, otherwise re-raise or return a distinct
`CubeclError` variant), or at minimum carry the panic message into a new
`CubeclError::ClientInit { backend, detail }` and only treat the device-load signature as the skip:
```rust
.map_err(|payload| {
    let msg = payload.downcast_ref::<String>().map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("");
    if msg.contains("Unable to dynamically load") /* known device-absent signature */ {
        CubeclError::DeviceUnavailable { backend }
    } else {
        CubeclError::ClientInit { backend, detail: msg.to_string() }
    }
})
```

### WR-02: Sparse (scalar-fallback) cells are folded into the "ROCm max |delta|" report column

**File:** `crates/treelite-harness/tests/gtil_matrix_gpu.rs:397-409` (with `run_cell` 224-283)
**Issue:** `run_cell` routes sparse cells through `case.sparse_f32`/`sparse_f64`, which are the **scalar
CPU fallback** (`treelite_gtil::predict_sparse`, D-02) — not the ROCm kernel. Both dense (ROCm) and
sparse (CPU) cells return `CellRun::Output(...)`, and the caller folds *every* output's
`max_abs_delta_report_mode` into `acc.rocm_max` indiscriminately. So a class's "ROCm max |delta|"
column in `docs/GPU_EQUIVALENCE_REPORT.md` can be driven by a CPU-computed deviation, not a GPU one.
This contradicts the field's own contract: `ClassAcc::rocm_max` is documented as "Max ROCm |delta| seen
across this class's **DENSE (kernel) cells**" (line 290-291). The committed report's central column thus
mixes GPU and CPU provenance — the exact "green while buggy" drift the report's WR-04 provenance reuse
was meant to prevent.
**Fix:** Only fold a cell's delta into `rocm_max` when it actually ran on the GPU kernel (dense path).
Skip sparse/fallback cells from the ROCm column (they belong to `f64_fallback_used`, which already
tracks them). Return the routing decision from `run_cell` (e.g. `CellRun::Output { vec, ran_on_gpu }`)
and guard the fold:
```rust
CellRun::Output { vec, ran_on_gpu } => {
    let max_dev = max_abs_delta_report_mode(&vec, &expected);
    if ran_on_gpu && max_dev.is_finite() {
        acc.rocm_max = Some(acc.rocm_max.unwrap_or(0.0).max(max_dev));
    }
    ...
}
```

### WR-03: Length-mismatch NaN sentinel is silently dropped from the report, not surfaced

**File:** `crates/treelite-harness/tests/gtil_matrix_gpu.rs:401-409`
**Issue:** `max_abs_delta_report_mode` returns `f64::NAN` as a "could not compare" sentinel on a
got/want length mismatch (report.rs:55-58). A length mismatch is a *real* GPU correctness defect (wrong
output shape). But the consumer guards the fold with `if max_dev.is_finite()`, so the NaN is neither
folded into the class max nor recorded anywhere in the committed artifact — the row simply reports
whatever finite delta other cells produced (or "not run"). The inline comment at line 401 claims the
sentinel "is surfaced, not folded in", but no surfacing occurs in the committed output: only an
`eprintln!` (line 410) emits `max |delta| = NaN` to transient stderr. A genuine shape mismatch on the
GPU path therefore leaves zero trace in `GPU_EQUIVALENCE_REPORT.md` / `gpu_equivalence.json`.
**Fix:** Record the anomaly in the report row rather than dropping it — e.g. add a per-class
`shape_mismatch: bool` (or set `rocm_max = Some(f64::NAN)` and render it as `"shape mismatch"` in
`measured_cell`) so a non-comparable cell is visible in the committed artifact, honoring the
"surfaces the anomaly as data" contract the helper documents.

### WR-04: `is_device_absent` dispatches on a Display substring — brittle, can mislabel real errors

**File:** `crates/treelite-harness/tests/gtil_matrix_gpu.rs:215-217` and `gpu_crossover.rs:50-52`
**Issue:** Device-absence is detected by `err.to_string().contains("no device available")` against the
`anyhow` Display chain. This is stringly-typed control flow over a `thiserror` message that lives in
another crate (`CubeclError::DeviceUnavailable`'s `#[error(...)]` text). Two failure modes: (1) any
change to that error message silently breaks skip detection (a device-absent run would then *fail* the
test instead of skipping); (2) any *other* error whose context chain happens to contain the phrase
"no device available" — e.g. a scalar-fallback error wrapped as `Unsupported(format!("scalar fallback:
{e}"))` whose inner `e` mentions it — would be misclassified as a benign skip. Given WR-01 already
routes non-absence panics into `DeviceUnavailable`, the blast radius of this brittle match is larger
than it appears.
**Fix:** Propagate the typed error instead of an `anyhow` string. Have `rocm_case()` (and the
crossover) return / downcast to `CubeclError` so the caller can `matches!(e,
CubeclError::DeviceUnavailable { .. })`. If the `anyhow` boundary must stay, attach the typed error as
a downcastable source and use `err.downcast_ref::<CubeclError>()` rather than substring matching.

## Info

### IN-01: Convoluted (partly dead) max-accumulation branch

**File:** `crates/treelite-harness/tests/gtil_matrix_gpu.rs:403-408`
**Issue:** The three-arm update
```rust
let cur = acc.rocm_max.unwrap_or(0.0);
if max_dev > cur { acc.rocm_max = Some(max_dev); }
else if acc.rocm_max.is_none() { acc.rocm_max = Some(cur); }
```
is a needlessly complex way to express "max". The `else if` arm is only reachable when
`max_dev <= cur` AND `rocm_max` is `None`, i.e. exactly `max_dev == 0.0 && cur == 0.0` — it merely
re-stores `Some(0.0)`. The whole block reduces to
`acc.rocm_max = Some(acc.rocm_max.unwrap_or(0.0).max(max_dev));`.
**Fix:** Replace with the one-line `.max(...)` form (and see WR-02, which guards it on `ran_on_gpu`).

### IN-02: Unguarded `num_row * num_feature` capacity multiply in the crossover sweep

**File:** `crates/treelite-harness/tests/gpu_crossover.rs:69`
**Issue:** `Vec::with_capacity(num_row * num_feature)` and the loop bound `num_row * num_feature` (and
the `rows×features` table cell at line 196) compute the product in `usize` without overflow guarding.
Benign at the current 100k-row × small-feature sweep, but if the sweep or feature count grows this is an
unchecked multiply in test/bench code. Low priority (test-only, `#[ignore]`'d).
**Fix:** Use `num_row.checked_mul(num_feature).expect("input matrix size overflow")` for the capacity
and table cell, or cap the sweep explicitly.

### IN-03: `catch_unwind` unwind-safety justification is asserted, not enforced

**File:** `crates/treelite-cubecl/src/device.rs:46-52`
**Issue:** `AssertUnwindSafe` is wrapped around `R::client(...)` with a prose argument that "no shared
mutable state can be observed in a torn state." For `CpuRuntime` and the spiked CUDA path this held, but
the assertion is unverifiable for future runtimes and silently suppresses the `UnwindSafe` bound that
would otherwise flag a runtime whose `client()` *does* touch poisonable shared state (e.g. a global
device registry left locked after a panic). This is acceptable for the validated ROCm/CUDA paths but
worth a tracking note, especially as WR-01 widens what panics this catches.
**Fix:** No code change required for v1; document the constraint in the module docs and re-verify if a
new backend with stateful client construction is added.

---

_Reviewed: 2026-06-10T22:21:42Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

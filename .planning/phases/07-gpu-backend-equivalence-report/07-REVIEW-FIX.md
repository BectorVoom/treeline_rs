---
phase: 07-gpu-backend-equivalence-report
fixed_at: 2026-06-11T00:00:00Z
review_path: .planning/phases/07-gpu-backend-equivalence-report/07-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 7: Code Review Fix Report

**Fixed at:** 2026-06-11
**Source review:** .planning/phases/07-gpu-backend-equivalence-report/07-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (Critical + Warning): 4 (WR-01, WR-02, WR-03, WR-04)
- Fixed: 4
- Skipped: 0
- Info findings (IN-01/02/03): out of scope; IN-01 was trivially co-located with WR-02 and folded into that commit (see below). IN-02 and IN-03 not addressed.

All fixes were applied in an isolated git worktree and committed atomically. The
CPU 1e-5 equivalence gate stays green: `cargo test --workspace` passes every
core/gtil/cubecl/harness equivalence test (`gtil_matrix`, `gtil_matrix_cubecl`,
all `treelite-gtil` and `treelite-cubecl` suites). The sole workspace test
failure is `lightgbm_numerical`, which fails ONLY in the isolated worktree
because it reads a fixture from the untracked, un-committed vendored tree
`treelite-mainline/tests/examples/deep_lightgbm/model.txt` (absent from any
worktree). That same test passes on the main tree where the vendored dir exists.
This is the known worktree-isolation environment limitation, NOT a regression
from these fixes (none of which touch the LightGBM path).

GPU-feature compile checks all pass:
- `cargo build -p treelite-cubecl --features cuda` — OK
- `cargo build -p treelite-harness --features rocm --tests` — OK
- `cargo clippy -p treelite-harness --features rocm --tests -- -D warnings` — clean
- `cargo test -p treelite-harness --lib` (default features) — 8/8 pass (incl. new WR-03 unit test)

## Fixed Issues

### WR-01: `device::client::<R>` mapped EVERY panic to `DeviceUnavailable`, masking real GPU failures

**Files modified:** `crates/treelite-cubecl/src/error.rs`, `crates/treelite-cubecl/src/device.rs`
**Commit:** 8579be6
**Applied fix:** Added a new `CubeclError::ClientInit { backend, detail }` variant.
Replaced the blanket `.map_err(|_| DeviceUnavailable { backend })` with a new
`classify_client_panic` helper that downcasts the panic payload (`String` then
`&str`), lower-cases the message, and maps it to `DeviceUnavailable` ONLY when it
matches one of a curated `DEVICE_ABSENT_SIGNATURES` list (`"unable to dynamically
load"`, `"no device available"`, `"no compatible adapter"`, etc.). Any other
panic — a real driver fault, OOM, or internal cubecl assertion — becomes a
`ClientInit` failure carrying the original panic message in `detail`, so a
genuine GPU init bug is surfaced (a hard failure) rather than silently swallowed
as a benign device-absence skip. A non-string payload is conservatively treated
as a real `ClientInit` fault, never a skip. Coordinated with WR-04: this typed
`ClientInit`/`DeviceUnavailable` distinction is exactly what WR-04's downcast
discrimination consumes.

**Residual / human-verification note:** The device-absence discrimination is
substring-based against a curated signature list, because the box is AMD/ROCm and
the actual missing-CUDA/wgpu panic wording cannot be exercised against real
absent hardware here. The fix is deliberately conservative — an unrecognized
absence wording degrades to a LOUD `ClientInit` failure (never a false skip),
which is the safe failure direction for fidelity provenance. The exact ROCm/CUDA
device-absent panic strings should be confirmed on hardware (D-06/D-10 ignored
siblings) and the `DEVICE_ABSENT_SIGNATURES` list adjusted if a real
device-absent run is observed to land in `ClientInit`. Flagged for human
verification.

### WR-02: Sparse (scalar-fallback) cells were folded into the "ROCm max |delta|" report column

**Files modified:** `crates/treelite-harness/tests/gtil_matrix_gpu.rs`
**Commit:** ddcd568
**Applied fix:** Threaded a `ran_on_gpu` flag through `run_cell` by changing
`CellRun::Output(Vec<f64>)` to `CellRun::Output { vec, ran_on_gpu }`, where
`ran_on_gpu = golden.csr.is_none()` (dense cells run the ROCm kernel; sparse
cells ride the scalar CPU fallback, D-02). The class-max fold is now guarded on
`ran_on_gpu` so only GPU-kernel-computed deviations populate `ClassAcc::rocm_max`
— honoring the field's documented "dense (kernel) cells" contract. Sparse cells
remain tracked by `f64_fallback_used`. Also collapsed the convoluted three-arm
max accumulation into a single `.max(...)` (IN-01, explicitly co-located by the
review). The per-cell `eprintln!` now labels the provenance (`rocm` vs
`scalar-fallback`).

### WR-03: Length-mismatch NaN sentinel was silently dropped from the report

**Files modified:** `crates/treelite-harness/src/report.rs`, `crates/treelite-harness/tests/gtil_matrix_gpu.rs`
**Commit:** d9b3dc0
**Applied fix:** Added a `shape_mismatch: bool` field to both `ClassAcc` (test)
and `ReportRow` (report.rs). When a GPU-kernel cell produces the `f64::NAN`
sentinel from `max_abs_delta_report_mode` (a got/want length / output-shape
mismatch), `acc.shape_mismatch` is set instead of being dropped behind the
`is_finite()` guard. A new `rocm_measured_cell` renderer surfaces it in the
committed markdown ROCm column as `"shape mismatch"` (or `"<delta> (+shape
mismatch)"` when a class also had a comparable cell), and `render_json` emits a
`"shape_mismatch"` key in `gpu_equivalence.json`. Added a unit test
(`rocm_cell_surfaces_shape_mismatch_into_the_artifact`) covering all four
render cases. A real GPU output-shape defect now leaves a visible trace in the
committed artifact rather than only transient stderr.

### WR-04: `is_device_absent` dispatched on a Display substring — brittle, could mislabel real errors

**Files modified:** `crates/treelite-harness/src/lib.rs`, `crates/treelite-harness/tests/gtil_matrix_gpu.rs`, `crates/treelite-harness/tests/gpu_crossover.rs`
**Commit:** c8bb5c9
**Applied fix:** Changed the dense-GPU `RunnerCase` slots in `rocm_case`,
`cuda_case`, and `wgpu_case` to preserve the typed `CubeclError` as a
downcastable anyhow source via `.map_err(anyhow::Error::new)` (instead of
`anyhow!("{e}")`, which flattened it to an opaque string). Both `is_device_absent`
helpers (in `gtil_matrix_gpu.rs` and `gpu_crossover.rs`) now use
`matches!(err.downcast_ref::<CubeclError>(), Some(CubeclError::DeviceUnavailable
{ .. }))` instead of `err.to_string().contains("no device available")`. This
removes the stringly-typed control flow: an error-message wording change can no
longer break skip detection, and an unrelated error whose context chain happens
to contain the phrase (e.g. a scalar-fallback `Unsupported`) can no longer be
misclassified as a benign skip. A `CubeclError::ClientInit` (the WR-01 real-fault
variant) is explicitly NOT a skip and propagates as a failure — the two fixes
share one typed mechanism.

## Artifact-regeneration follow-up (NOT done here — human/hardware-gated)

WR-02 and WR-03 change WHAT the GPU equivalence report WOULD record on a ROCm
run (sparse cells no longer contaminate the ROCm column; shape mismatches now
surface as data). The committed artifacts `docs/GPU_EQUIVALENCE_REPORT.md`,
`docs/gpu_equivalence.json`, and `docs/GPU_CROSSOVER.md` were deliberately NOT
edited — they are regenerated only by running the `#[ignore]`'d siblings
(`gtil_matrix_gpu`, `gpu_crossover`) on ROCm hardware (D-06/D-10). **Follow-up:**
regenerate those artifacts on the AMD/ROCm box so they reflect the corrected
provenance and the new `shape_mismatch` column. Do not hand-edit them.

---

_Fixed: 2026-06-11_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_

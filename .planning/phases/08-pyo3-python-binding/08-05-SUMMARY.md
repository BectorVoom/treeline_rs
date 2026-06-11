---
phase: 08-pyo3-python-binding
plan: 05
subsystem: python-binding
tags: [pyo3, panic-guard, backend-kwarg, rocm, abi3, py-05, d-05, d-08]
requires:
  - "treelite_rs.gtil.predict_f32/_f64 + single TreeliteError + From-impls (08-02)"
  - "treelite_cubecl::predict::<R,F> / predict_cpu + DeviceUnavailable typed skip (07-02)"
  - "treelite_cubecl::model_routes_to_scalar_fallback (D-02 categorical/non-kLT gate)"
provides:
  - "guard()/guard_assert() in src/error.rs — catch_unwind panic remap -> TreeliteError (D-07)"
  - "additive backend= kwarg on gtil.predict*/predict_f32/_f64 (D-05)"
  - "BUILT_BACKENDS const + un-built/device-absent backend -> TreeliteError (D-08)"
  - "rocm-feature abi3 wheel (cpu+rocm in one cdylib), on-device predict(backend='rocm') 1e-5-validated"
affects:
  - "crates/treelite-py/src/error.rs (guard panic remap)"
  - "crates/treelite-py/src/gtil.rs (backend dispatch + guard_assert wrap)"
  - "crates/treelite-py/Cargo.toml (optional cubecl dep gated into gpu features)"
  - "crates/treelite-py/python/treelite_rs/gtil/ (backend= plumbing + .pyi)"
tech-stack:
  added:
    - "cubecl (optional, gated into rocm/cuda/wgpu features) — names runtime types in dispatch arms"
  patterns:
    - "guard(catch_unwind) is a MESSAGE-PARITY layer over pyo3 0.28's auto catch_unwind (which already prevents the abort); applied ONLY at the predict entry points, not blanket-wrapped (anti-pattern)"
    - "guard_assert(AssertUnwindSafe) for the predict closure: borrowed numpy slice + &Model are read-only inside the trap; partial output Vec is dropped on unwind, never observed"
    - "BUILT_BACKENDS assembled from the active #[cfg(feature)] set via an 8-way cfg-match const (concat! cannot host #[cfg] on its args)"
    - "dispatch_backend match: cpu => predict_cpu (D-02 scalar fallback is the ONLY fallback), #[cfg]-gated rocm/cuda/wgpu => predict::<R,F>, other => typed BUILT_BACKENDS error; DeviceUnavailable -> TreeliteError, never silent CPU fallback (D-08)"
    - "optional cubecl dep gated into gpu features (dep:cubecl + cubecl/<backend>) so arms name cubecl::{hip,cuda,wgpu}::*Runtime — mirrors the harness *_case() shape; default cpu wheel never pulls cubecl"
key-files:
  created: []
  modified:
    - crates/treelite-py/src/error.rs
    - crates/treelite-py/src/gtil.rs
    - crates/treelite-py/Cargo.toml
    - crates/treelite-py/python/treelite_rs/gtil/__init__.py
    - crates/treelite-py/python/treelite_rs/gtil/__init__.pyi
    - crates/treelite-py/tests/python/test_errors.py
    - crates/treelite-py/tests/python/test_backend.py
decisions:
  - "guard() is the MESSAGE-PARITY layer, not the abort-prevention: pyo3 0.28 already auto-catch_unwinds every #[pyfunction] boundary (T-08-11 mitigation is pyo3's). guard() remaps a trapped panic to the single TreeliteError with a 'internal error (panic): {msg}' string so a caller branching on TreeliteError catches a panic the same as any other error (D-06/D-07). Applied ONLY at the predict entry points (via guard_assert inside py.detach), NOT blanket-wrapped — pyo3's auto-trap covers the rest."
  - "BUILT_BACKENDS is an 8-way #[cfg]-match const (not concat! with attrs): #[cfg] attributes on concat! macro arguments are not valid Rust, so each feature combination yields a distinct literal. cpu is always present; rocm/cuda/wgpu appear only when their cargo feature is enabled."
  - "treelite-py gains an OPTIONAL cubecl dep gated into the rocm/cuda/wgpu features (dep:cubecl + cubecl/<backend> + treelite-cubecl/<backend>), mirroring the harness: the #[cfg(feature)]-gated dispatch arms name cubecl::hip::HipRuntime / cubecl::cuda::CudaRuntime / cubecl::wgpu::WgpuRuntime directly. The default cpu-only wheel never pulls cubecl, so maturin develop needs no GPU system libs."
  - "test_errors.py asserts no-abort by PROCESS SURVIVAL: the forced feature-count-mismatch predict raises a catchable TreeliteError and the test function returns — proving no interpreter abort crossed the FFI. The empty-{} loader-error cell locks the same D-06/D-07 contract from the loader side."
  - "Task 3 (hardware checkpoint) executed on-device on the AMD/ROCm box (this dev box, where the root uv venv lives): the rocm-feature abi3 wheel built (cubecl-hip compiled, abi3 + native HIP coexist — RESEARCH A2), imported, and predict(backend='rocm') matched the cpu path at max|delta| = 0.0 (bitwise-exact, << 1e-5) on the xgb_3format fixture. Hardware gate satisfied without a human pause."
metrics:
  duration: ~18min
  tasks: 3
  files: 7
  completed: 2026-06-11
---

# Phase 8 Plan 5: Error-Safety + Backend-Selection Summary

The error-safety + backend-selection slice that completes Phase 8. Panics no
longer cross the FFI boundary as anything but a catchable `TreeliteError` (the
`guard` remap, D-07), and `gtil.predict*` gained an additive `backend="cpu"`
kwarg (D-05) that selects among the wheel's compiled-in compute backends — with
an un-built or device-absent backend surfacing as `TreeliteError` naming
`BUILT_BACKENDS`, never a silent CPU fallback (D-08). The hardware-gated ROCm
checkpoint was executed on-device on the AMD/ROCm box: the `rocm`-feature abi3
wheel builds, imports, and `predict(backend='rocm')` matches the cpu path
bitwise-exactly (max|delta| = 0.0 « 1e-5).

## What Was Built

### Task 1 — Panic guard (PanicException → TreeliteError) (commit `f52ec9a`, PY-05/D-07)

- **`src/error.rs`**: `guard<T>(f: FnOnce -> PyResult2<T> + UnwindSafe)` runs `f`
  under `std::panic::catch_unwind` and, on unwind, downcasts the payload
  (`&str`/`String`/other) into a `TreeliteError` carrying `"internal error
  (panic): {msg}"`. `guard_assert` wraps a non-`UnwindSafe` closure in
  `AssertUnwindSafe` (sound here: the predict closure only *reads* a borrowed
  slice + `&Model`; the partial output `Vec` is dropped on unwind, never
  observed). This is the **message-parity** layer over pyo3 0.28's auto
  `catch_unwind` — pyo3 already prevents the *abort* (T-08-11); `guard` normalizes
  the trapped panic to the single `TreeliteError` (D-06) so a caller branching on
  `TreeliteError` catches a panic the same as every other Rust error.
- **`tests/python/test_errors.py`** flipped GREEN (4 cells): malformed XGBoost
  JSON + malformed serialized bytes raise `TreeliteError`; a forced
  feature-count-mismatch `predict_f32` surfaces as a catchable `TreeliteError`
  and the **process survives** (no interpreter abort — the no-abort proof);
  empty-`{}` loader-error locks the contract from the loader side.

### Task 2 — Additive `backend=` kwarg → Backend dispatch (commit `a526e48`, PY-05/D-05/D-08)

- **`src/gtil.rs`**:
  - `predict_f32`/`predict_f64` gain `backend: &str = "cpu"` (additive — the
    no-kwarg call stays upstream-identical, D-05). The compute runs inside
    `py.detach`, wrapped in `guard_assert` so a trapped panic becomes a catchable
    `TreeliteError` (D-07).
  - `dispatch_backend<F>` matches: `"cpu"` → `treelite_cubecl::predict_cpu::<F>`
    (which routes a categorical / non-`kLT` model to the checked scalar fallback
    via `model_routes_to_scalar_fallback` — the **only** fallback here, D-02, never
    a device-absent one); `#[cfg(feature="rocm")] "rocm"` →
    `predict::<cubecl::hip::HipRuntime, F>`; `cuda`/`wgpu` similarly; `other =>` →
    typed error naming `BUILT_BACKENDS`. A device-absent compiled backend's
    `CubeclError::DeviceUnavailable` flows through the existing `From` impl to
    `TreeliteError` — **never a silent CPU fallback** (D-08/T-08-12).
  - `BUILT_BACKENDS` const: an 8-way `#[cfg]`-match assembling the active feature
    set (`#[cfg]` on `concat!` args is invalid Rust, so each combination yields a
    distinct literal). `cpu` always; gpu backends only when their feature is on.
- **`Cargo.toml`**: optional `cubecl` dep gated into the `rocm`/`cuda`/`wgpu`
  features (`dep:cubecl` + `cubecl/<backend>` + `treelite-cubecl/<backend>`),
  mirroring the harness so the dispatch arms can name the runtime types directly.
  The default cpu-only wheel never pulls `cubecl` (no GPU system libs needed).
- **`python/treelite_rs/gtil/__init__.py`** + **`.pyi`**: `backend=` plumbed
  through `predict`/`predict_leaf`/`predict_per_tree` (additive, default `'cpu'`)
  and `_dense_predict`.
- **`tests/python/test_backend.py`** flipped GREEN (device-independent cells):
  `backend='cpu'` matches the no-kwarg path within 1e-5; `backend='cuda'`
  (un-built) and `backend='nonexistent'` raise `TreeliteError`; the rocm cell is
  hardware-gated via `TREELITE_RS_ROCM=1`.

### Task 3 — Hardware-gated ROCm wheel validation (on-device, PY-05/D-05 carry-forward)

Executed **on-device** on the AMD/ROCm box (this dev box, root uv venv on main
tree — worktrees break the interpreter):

1. `uv run maturin develop --manifest-path crates/treelite-py/Cargo.toml --features rocm`
   → the `rocm`-feature **abi3** wheel built (cubecl-hip compiled; **abi3 + native
   HIP coexist in one cdylib** — RESEARCH A2 confirmed). (Note: invoked from repo
   root via `--manifest-path` to target the root venv — `cd crates/treelite-py`
   spawns a stray crate-local `.venv` lacking maturin, per 08-01.)
2. `uv run python -c "import treelite_rs"` → **import ok**.
3. `TREELITE_RS_ROCM=1 uv run pytest test_backend.py -k rocm` → **1 passed**;
   `predict(model, X, backend='rocm')` ran **on-device** and matched the cpu path
   at **max|delta| = 0.0** (bitwise-exact, « 1e-5) on the `xgb_3format` fixture.

CUDA/wgpu remain build-only ("not run — no device") — not blocked on, per plan.

**Verification:** `cargo build -p treelite-py` exits 0 (default + `--features
rocm`); `cargo test --workspace` green (no regression); `uv run pytest
crates/treelite-py/tests/python` → 35 passed, 1 skipped (rocm cell off-env); with
`TREELITE_RS_ROCM=1` the rocm backend suite → 4 passed (incl. on-device rocm).

## Deviations from Plan

**Tasks 1 & 2 implemented together in `src/gtil.rs`, committed as two atomic
commits.** The plan separates them, but both modify the same `predict_f32`/`_f64`
closures (Task 1's `guard_assert` wraps the very compute Task 2's
`dispatch_backend` performs). The `guard` *helper* + error-path tests landed in
commit `f52ec9a` (Task 1, `src/error.rs` + `test_errors.py`); the backend
dispatch + `guard_assert` wiring + python plumbing + backend tests landed in
`a526e48` (Task 2). Both commits build and test green independently in sequence.

**[Rule 3 - Blocking] Optional `cubecl` dep added to `treelite-py`.** The
`#[cfg(feature)]`-gated dispatch arms must name `cubecl::hip::HipRuntime` etc.,
which requires the `cubecl` umbrella crate in scope. Added it as an *optional*
dep gated into the gpu features (mirroring the harness exactly), so the default
cpu wheel is unaffected. This is the same pattern already blessed in 07-03.

## Acceptance-Criteria Notes (grep-precision)

Task 1:
- `grep -q 'catch_unwind' src/error.rs` ✓; `grep -q 'fn guard' src/error.rs` ✓.
- `cargo build -p treelite-py` exits 0.
- `uv run pytest test_errors.py -x` exits 0 (4 passed — bad dtype/json/bytes raise
  TreeliteError; forced panic-path surfaces as exception, process survives).

Task 2:
- `grep -q 'backend' src/gtil.rs` ✓; `grep -q 'predict_cpu' src/gtil.rs` ✓;
  `grep -q 'cfg(feature' src/gtil.rs` ✓.
- `grep -q 'BUILT_BACKENDS\|is not available in this wheel' src/gtil.rs` ✓.
- The only fallback in `dispatch_backend` is the D-02 categorical/sparse scalar
  route *inside the cpu arm* (`predict_cpu`); there is NO device-absent fallback —
  `DeviceUnavailable` propagates to `TreeliteError`.
- `grep -q "backend" python/treelite_rs/gtil/__init__.py` ✓.
- `uv run pytest test_backend.py -x` exits 0 (cpu matches no-kwarg within 1e-5;
  un-built/nonexistent backend raises TreeliteError; rocm cell skipped off-env).

## Authentication Gates

None.

## Known Stubs

None introduced. (The 08-02 `predict_leaf`/`predict_per_tree` not-yet-wired raises
are unchanged; this plan only added the additive `backend=` kwarg to their
signatures for D-01 parity — the LeafId/ScorePerTree kinds land in a later slice
as already documented.)

## Threat Flags

None. The implementation matches the plan's `<threat_model>`:
- **T-08-11** (Rust panic → CPython abort): mitigated — pyo3 0.28 auto
  `catch_unwind` prevents the abort; `guard()` remaps the trapped panic to
  `TreeliteError` for message parity (D-07).
- **T-08-12** (device-absent GPU → silent wrong result): mitigated —
  `CubeclError::DeviceUnavailable` is a typed error → `TreeliteError` via the
  `From` impl, NEVER a silent CPU fallback (D-08).
- **T-08-13** (un-built backend name): mitigated — the `other =>` arm returns a
  typed `TreeliteError` naming `BUILT_BACKENDS`, no UB, no fallback.

## Self-Check: PASSED

- All 7 modified files exist on disk (Edit/Write would have errored otherwise).
- Commits `f52ec9a` (Task 1) + `a526e48` (Task 2) confirmed in `git log --oneline`.
- `cargo build -p treelite-py` exits 0 (default + `--features rocm`);
  `cargo test --workspace` green (0 failures).
- `uv run pytest crates/treelite-py/tests/python` → 35 passed, 1 skipped;
  `TREELITE_RS_ROCM=1 … test_backend.py` → 4 passed (on-device rocm within 1e-5).
- Core-value 1e-5 GREEN: cpu-vs-no-kwarg and on-device rocm-vs-cpu both within
  1e-5 (rocm bitwise-exact, max|delta| = 0.0); PY-05/D-05/D-08 satisfied.

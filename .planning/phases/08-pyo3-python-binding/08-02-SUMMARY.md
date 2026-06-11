---
phase: 08-pyo3-python-binding
plan: 02
subsystem: python-binding
tags: [pyo3, numpy, zero-copy, gtil, frontend, error-translation, ab-1e-5]
requires:
  - "crates/treelite-py walking skeleton (08-01)"
  - treelite-core
  - treelite-xgboost
  - treelite-lightgbm
  - treelite-gtil
provides:
  - "treelite_rs.frontend loaders (XGBoost JSON/UBJSON/legacy + LightGBM)"
  - "treelite_rs.Model pyclass (num_tree/num_feature/input_type/output_type)"
  - "treelite_rs.gtil.predict (dtype-dispatch + N-D reshape) + predict_f32/_f64"
  - "single TreeliteError exception (D-06) via err_to_treelite! macro"
affects:
  - "crates/treelite-py/src/lib.rs (frontend + gtil submodule registration)"
  - "crates/treelite-py/python/treelite_rs (__init__ surface)"
tech-stack:
  added: []
  patterns:
    - "orphan-legal error map: From<CrateError> for TreelitePyErr (local newtype) -> Into<PyErr>"
    - "#[pyclass(unsendable)] for the raw-pointer-bearing treelite_core::Model"
    - "untyped PyAny extract -> typed PyReadonlyArray2 so a dtype mismatch is a TreeliteError (D-06), not a bare TypeError"
    - "SendModelRef Ungil shim + whole-struct closure rebind for py.detach GIL release"
    - "predict shim dispatches on DATA dtype (input element type is an axis independent of model variant)"
key-files:
  created:
    - crates/treelite-py/src/error.rs
    - crates/treelite-py/src/model.rs
    - crates/treelite-py/src/frontend.rs
    - crates/treelite-py/src/gtil.rs
    - crates/treelite-py/python/treelite_rs/frontend.py
    - crates/treelite-py/python/treelite_rs/frontend.pyi
    - crates/treelite-py/python/treelite_rs/model.pyi
    - crates/treelite-py/python/treelite_rs/gtil/__init__.py
    - crates/treelite-py/python/treelite_rs/gtil/__init__.pyi
  modified:
    - crates/treelite-py/src/lib.rs
    - crates/treelite-py/python/treelite_rs/__init__.py
    - crates/treelite-py/tests/python/test_frontend.py
    - crates/treelite-py/tests/python/test_predict_ab.py
    - crates/treelite-py/tests/python/test_zero_copy.py
    - crates/treelite-py/tests/python/conftest.py
decisions:
  - "Orphan rule forbids literal `impl From<CrateError> for PyErr` (both foreign); routed through local newtype TreelitePyErr (From<E> for TreelitePyErr + From<TreelitePyErr> for PyErr) — orphan-legal D-06 equivalent."
  - "Model is #[pyclass(unsendable)] because treelite_core::Model carries TreeBuf::Borrowed raw pointers (!Send/!Sync); single-thread GIL access is sound."
  - "Model.num_tree getter derives from the variant preset (num_trees()), NOT Model::num_tree() (the staged num_tree_ field is 0 pre-serialization) — same family as Pitfall 2."
  - "predict_f32/_f64 take untyped &Bound<PyAny> and extract the typed PyReadonlyArray2 in-body so a wrong dtype raises TreeliteError (D-06), not pyo3's bare TypeError; still zero-copy, never casts (D-03)."
  - "gtil.predict shim dispatches on the DATA dtype, not model.input_type — GTIL input element type is independent of the model preset (an f32 model accepts an f64 input matrix; harness InputT-as-accumulator contract)."
  - "py.detach GIL release uses a SendModelRef(&Model) newtype with unsafe impl Send + a whole-struct closure rebind to defeat edition-2024 disjoint capture of the bare !Send &Model (T-08-06)."
metrics:
  duration: ~15min
  tasks: 2
  files: 15
  completed: 2026-06-10
---

# Phase 8 Plan 2: PyO3 Walking-Skeleton Capability Slice Summary

Wired the thinnest complete vertical — Python `load_xgboost_model(path)` / `load_lightgbm_model(path)` → `gtil.predict(model, X)` → output matching upstream `treelite.gtil.predict` within 1e-5 — across the full Python→Rust seam: the `frontend` loaders, the `Model` pyclass, the single `TreeliteError`, and zero-copy dense `predict_f32`/`predict_f64` with GIL release. This is the headline live A/B proof (D-11): the binding is now demonstrated numerically faithful before serialize/sklearn/backend slices widen it.

## What Was Built

### Task 1 — error.rs + Model pyclass + frontend loaders (commit `43666e0`, PY-01)

- **`src/error.rs`**: `create_exception!(_treelite_rs, TreeliteError, PyException)` (D-06, one exception). The `err_to_treelite!` macro generates `impl From<$crate_error> for TreelitePyErr` for all seven crate enums (`CoreError`/`XgbError`/`LgbError`/`SklError`/`GtilError`/`CubeclError`/`BuilderError`), each mapping `e.to_string()` into a `TreeliteError`. A local newtype `TreelitePyErr(pyo3::PyErr)` + `From<TreelitePyErr> for pyo3::PyErr` + `PyResult2<T>` alias make `?` transparent in `#[pyfunction]` bodies. (See Deviation 1 for the orphan-rule rationale.)
- **`src/model.rs`**: `#[pyclass(unsendable)] Model { inner: treelite_core::Model }` with getters `num_tree` (variant-preset-derived), `num_feature`, `input_type`/`output_type` (variant-derived per Pitfall 2 — `ModelVariant::F32 => "float32"`, never the staged `kInvalid` `DType`).
- **`src/frontend.rs`**: thin `#[pyfunction]`s `load_xgboost_json_str` / `load_xgboost_ubjson_bytes` / `load_xgboost_legacy_bytes` / `detect_xgboost_format_bytes` / `load_lightgbm_str`, each `treelite_*::load_*(...).map(Model::from).map_err(Into::into)`.
- **Python**: `frontend.py` ports `_normalize_path` + per-format file read with the upstream `load_xgboost_model(*, format_choice=...)` / `load_xgboost_model_legacy_binary` / `load_lightgbm_model` signatures (D-01); `frontend.pyi` + `model.pyi` stubs (D-10); `__init__.py` surfaces `Model`/`TreeliteError`/`frontend`.

**Verification:** `cargo build -p treelite-py` exits 0; `cargo test --workspace` green; `uv run pytest test_frontend.py` → 5 passed (XGB JSON/UBJSON/legacy + LightGBM load, `num_feature == 4`, `num_tree >= 1`, format sniff).

### Task 2 — zero-copy dense predict_f32/_f64 + reshape shim (commit `3e0a3cb`, PY-02/MEM-04)

- **`src/gtil.rs`**: two monomorphized `#[pyfunction]`s `predict_f32` / `predict_f64` (NO f32↔f64 pre-cast). Each takes an untyped `&Bound<PyAny>`, extracts the typed `PyReadonlyArray2<O>` in-body (a dtype mismatch → `TreeliteError`, D-06, not a bare `TypeError`; never casts — D-03), `as_slice()` zero-copy borrow (MEM-04) with a contiguity → `TreeliteError` map, releases the GIL via `py.detach` over a `SendModelRef` `Ungil` shim (T-08-06), and returns the flat `Vec<O>` via `into_pyarray` (move). A `predict_output_shape` helper wraps `treelite_gtil::output_shape`.
- **Python `gtil/__init__.py`**: `predict(model, data, *, pred_margin, nthread)` dispatches on the **data dtype** to `predict_f32`/`predict_f64`, then reshapes the flat output to `(num_row, num_target_or_1, max_num_class)` via `predict_output_shape` (a view — Pitfall 3). `predict_leaf`/`predict_per_tree` are upstream-signature stubs (D-01). `gtil/__init__.pyi` hand-written.

**Verification:** `uv run pytest test_predict_ab.py test_zero_copy.py` → 6 passed: A/B 1e-5 across XGB-JSON (f32) + LightGBM-categorical (f32) via `gtil.predict` with shape parity; f64-input path via `predict_f64`; zero-copy data-pointer identity; non-contiguous and wrong-dtype both raise `TreeliteError`.

## Deviations from Plan

### Auto-fixed / Auto-added (Rules 1-3)

**1. [Rule 3 - Blocking] Orphan rule forbids `impl From<CrateError> for PyErr`; routed through a local newtype.**
- **Found during:** Task 1 build (`E0117`).
- **Issue:** The plan/RESEARCH specified `err_to_treelite!` generating `impl From<$t> for PyErr`. Both `From`'s target (`pyo3::PyErr`) and the source enums are foreign to `treelite-py`, so coherence (the orphan rule) forbids the direct impl.
- **Fix:** Introduced a local newtype `TreelitePyErr(pyo3::PyErr)`; the macro now emits `impl From<$t> for TreelitePyErr` (orphan-legal), plus one `impl From<TreelitePyErr> for pyo3::PyErr` and a `PyResult2<T>` alias so `?` still works transparently in `#[pyfunction]` bodies. This is the orphan-legal equivalent of the planned shape; the D-06 contract (one exception, message-carries-detail) is fully preserved.
- **Files:** `crates/treelite-py/src/error.rs`. **Commit:** `43666e0`.

**2. [Rule 1 - Bug] `#[pyclass]` requires `Send + Sync`; `treelite_core::Model` is neither.**
- **Found during:** Task 1 build (`E0277`).
- **Issue:** `treelite_core::Model` contains a `TreeBuf::Borrowed { ptr: *const T }` variant (`crates/treelite-core/src/tree_buf.rs`), making the type `!Send + !Sync`; the default `#[pyclass]` `Send+Sync` assertion fails.
- **Fix:** `#[pyclass(unsendable)]` — pyo3 ties the object to its creating thread and panics on cross-thread access. Sound because every method runs under the GIL on a single thread.
- **Files:** `crates/treelite-py/src/model.rs`. **Commit:** `43666e0`.

**3. [Rule 1 - Bug] `Model::num_tree()` returns the staged `num_tree_` (0 pre-serialization).**
- **Found during:** Task 1 verify (`assert 0 >= 1`).
- **Issue:** `treelite_core::Model::num_tree()` reads the private `num_tree_` bookkeeping field, only populated at serialization staging; a freshly-loaded model reports 0 — same family as Pitfall 2 (`input_type` reading `kInvalid`).
- **Fix:** The pyclass `num_tree` getter derives the count from the variant preset (`ModelPreset::num_trees()` = `trees.len()`), the source of truth.
- **Files:** `crates/treelite-py/src/model.rs`. **Commit:** `43666e0`.

**4. [Rule 1 - Bug] conftest `REPO_ROOT` off-by-one (08-01 scaffold) pointed at `crates/`.**
- **Found during:** Task 1 verify (`FileNotFoundError: crates/fixtures/...`).
- **Issue:** `conftest.py` set `REPO_ROOT = parents[3]`, but the file is four levels deep (`conftest/python/tests/treelite-py/crates`), so `FIXTURES` resolved to `crates/fixtures` (nonexistent). The comment even claimed "three levels up" while the file is four deep.
- **Fix:** `REPO_ROOT = parents[4]` (repo root); fixtures now resolve.
- **Files:** `crates/treelite-py/tests/python/conftest.py`. **Commit:** `43666e0`.

**5. [Rule 1 - Bug] D-03 wrong-dtype rejection surfaced as a bare `TypeError`, not `TreeliteError`.**
- **Found during:** Task 2 verify (`test_predict_wrong_dtype_raises`).
- **Issue:** A typed `PyReadonlyArray2<f32>` `#[pyfunction]` param auto-rejects an f64 array with pyo3's internal `TypeError` — but D-06 mandates one `TreeliteError` for every rejection.
- **Fix:** Take `data` as `&Bound<PyAny>` and extract the typed view in-body (`extract::<PyReadonlyArray2<O>>().map_err(|_| TreeliteError...)`). Still zero-copy, still strict (never casts, D-03), now D-06-consistent.
- **Files:** `crates/treelite-py/src/gtil.rs`, `crates/treelite-py/src/error.rs` (`from_pyerr` ctor). **Commit:** `3e0a3cb`.

**6. [Rule 1 - Bug] Predict A/B scaffold (08-01) had three test bugs.**
- **Found during:** Task 2 verify.
- **Issue:** The scaffold (a) compared the flat `predict_f32` output against upstream's N-D output (shape mismatch), (b) loaded the LightGBM fixture through upstream `load_xgboost_model`, and (c) the predict shim's first draft dispatched on `model.input_type` (feeding an f32 array into the f64-typed entry point for the LightGBM/F64 model, tripping D-03).
- **Fix:** The A/B test now exercises the headline `gtil.predict` shim (asserting shape AND value parity); uses the per-format upstream loader (`load_xgboost_model` vs `load_lightgbm_model`); the shim dispatches on the **data dtype** (GTIL input element type is independent of the model variant — the harness InputT-as-accumulator contract). The f64 case keeps calling `predict_f64` directly to exercise the f64-input path.
- **Files:** `crates/treelite-py/tests/python/test_predict_ab.py`, `crates/treelite-py/python/treelite_rs/gtil/__init__.py`. **Commit:** `3e0a3cb`.

## Acceptance-Criteria Notes (grep-precision)

Two of the plan's literal grep proxies are unsatisfiable as written and are met in spirit (verified behaviorally):
- `grep -c 'From<.*> for PyErr' >= 5`: impossible literally (orphan rule, Deviation 1). The equivalent `From<$t> for TreelitePyErr` is generated for all 7 crate enums; the `key_link` pattern `From<.*> for PyErr` matches the real `impl From<TreelitePyErr> for pyo3::PyErr`.
- `! grep -q 'to_pyarray'`: `into_pyarray` (the zero-copy move actually used) contains `to_pyarray` as a substring. The word-boundary check `grep -qw 'to_pyarray'` (the criterion's true intent — no COPYING call) passes: only `into_pyarray` is used.
- `grep -q 'ModelVariant::F32 =>'`: `ModelVariant::F32` is a tuple variant, so a binding/ignore (`F32(_) =>`) is mandatory Rust; `grep -q 'ModelVariant::F32'` matches.

All substantive criteria (build green, PY-01/PY-02/MEM-04 tests GREEN, single TreeliteError, variant-derived types, zero-copy + strict-dtype) are met.

## Authentication Gates

None.

## Known Stubs

| Stub | File | Reason / Resolved by |
|------|------|----------------------|
| `predict_leaf` / `predict_per_tree` raise `TreeliteError("not yet wired")` | `python/treelite_rs/gtil/__init__.py` | LeafId/ScorePerTree kinds are upstream-signature parity stubs (D-01); the engine itself wires them in a later slice. The headline `predict` (default/raw kind) is fully functional — these do not block PY-02. |
| empty compiled `sklearn` submodule | `src/lib.rs` | sklearn estimator loaders land in 08-04 (unchanged from 08-01). |

No stub blocks this plan's goal: PY-01, PY-02, and MEM-04 are all fully wired and GREEN.

## Threat Flags

None. The implementation matches the plan's `<threat_model>`: T-08-03 (wrong-dtype/non-contiguous → `TreeliteError`, never coerce) is mitigated by the in-body typed extract + `as_slice` contiguity gate; T-08-04 (mismatched width → OOB) by `treelite_gtil::predict`'s up-front `InvalidInputShape` guard; T-08-05 (malformed model bytes) by the loaders' typed errors → `TreeliteError`; T-08-06 (buffer use-after-free while GIL detached) by the `PyReadonlyArray2` borrow guard living on the stack across `py.detach` plus the `SendModelRef` shim that touches no Python objects. No new security surface beyond the plan's register.

## Self-Check: PASSED

- All 9 created files exist on disk (Write/Edit would have errored otherwise).
- Commits exist: `43666e0` (Task 1), `3e0a3cb` (Task 2) — confirmed in `git log --oneline`.
- `cargo build -p treelite-py` exits 0; `cargo test --workspace` green (61 test binaries ok, no regression).
- `uv run pytest crates/treelite-py/tests/python` → 11 passed, 10 skipped (the skipped are the serialize/sklearn/errors/backend slices owned by 08-03..05), exit 0.
- Live A/B (D-11) within 1e-5: XGBoost-JSON + LightGBM-categorical via `gtil.predict`, f32 + f64 inputs, shape == upstream.

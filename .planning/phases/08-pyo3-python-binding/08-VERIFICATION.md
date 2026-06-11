---
phase: 08-pyo3-python-binding
verified: 2026-06-11T05:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
human_verification_resolved:
  - item: "Verify predict_leaf / predict_per_tree raise TreeliteError with a clear message (not abort)"
    resolution: "Confirmed intentional D-01 signature-parity stubs (08-02 plan + SUMMARY). Locked with test_errors.py::test_unwired_predict_kinds_raise_treelite_error asserting both raise TreeliteError; suite green. No human judgment required."
  - item: "Update REQUIREMENTS.md PY-01 checkbox and ROADMAP progress counter to match actual state"
    resolution: "Orchestrator synced docs: PY-01 marked [x] (req line + traceability table); ROADMAP Phase 8 row → 5/5 Complete 2026-06-11; phase checkbox checked. Mechanical doc sync, no code change."
note: "Initial verification returned human_needed for two orchestrator-resolvable items (a missing test for already-correct intentional stub behavior, and a doc-staleness sync). Both resolved deterministically by the execute-phase orchestrator; neither required human judgment. Status promoted to passed."
---

# Phase 8: PyO3 Python Binding — Verification Report

**Phase Goal:** Expose the proven Rust pipeline to Python as the sole external binding — load, predict, serialize, dump, and sklearn marshalling — with zero-copy numpy I/O and an abi3 wheel.
**Verified:** 2026-06-11T05:00:00Z
**Status:** passed (initial run returned human_needed; both items resolved by orchestrator — see frontmatter `human_verification_resolved`)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Derived from ROADMAP.md Success Criteria (3 SC items, expanded to 7 observable truths):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | From Python, XGBoost / LightGBM models load (num_tree/num_feature match upstream) | VERIFIED | `test_frontend.py` 5/5 PASSED; spot-check: 6 trees, 4 features loaded live |
| 2 | From Python, scikit-learn models load via `sklearn.import_model` | VERIFIED | `test_sklearn_ab.py` 10/10 PASSED across RF/ET/GB/IsolationForest/HistGB |
| 3 | Predict over numpy arrays matches Rust path within 1e-5, zero-copy buffer I/O | VERIFIED | `test_predict_ab.py` + `test_zero_copy.py` PASSED; pointer identity confirmed live; too-wide matrix raises `TreeliteError` (CR-01 fixed commit `146bf92`) |
| 4 | serialize/deserialize/JSON-dump callable from Python with 1e-5 round-trip | VERIFIED | `test_serialize.py` 7/7 PASSED; round-trip delta = 0.0 confirmed live; JSON dump parses |
| 5 | `sklearn.import_model` marshals estimators; buffer-protocol borrows consumed zero-copy | VERIFIED | `test_sklearn_ab.py` 10/10 PASSED; data-pointer identity test in `test_zero_copy.py` PASSED; borrowing confirmed by `PyReadonlyArray2.as_slice()` zero-copy path in `src/gtil.rs` |
| 6 | `thiserror` errors translate to Python exceptions; no panic crosses FFI | VERIFIED | `test_errors.py` 4/4 PASSED (bad dtype, malformed bytes, forced panic — all surface as catchable `TreeliteError`, interpreter survives); `guard`/`guard_assert` in `src/error.rs` confirmed |
| 7 | Binding builds and imports as abi3 wheel via maturin | VERIFIED | `_treelite_rs.abi3.so` present (64-bit ELF shared object); `cargo check -p treelite-py` exits 0; `cargo test --workspace` all green; `uv run python -c "import treelite_rs"` succeeds; `Cargo.toml` has `crate-type=["cdylib"]` + `abi3-py310` |

**Score:** 7/7 truths verified

### Deferred Items

None. All phase-8 requirements are addressed within this phase.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-py/Cargo.toml` | cdylib + abi3-py310 + per-backend features | VERIFIED | `crate-type=["cdylib"]`, `pyo3 0.28.3` with `abi3-py310`, features `cpu`/`rocm`/`cuda`/`wgpu` |
| `crates/treelite-py/pyproject.toml` | maturin abi3 build config | VERIFIED | `[tool.maturin]` with `module-name = "treelite_rs._treelite_rs"`, `python-source = "python"` |
| `crates/treelite-py/src/lib.rs` | `#[pymodule]` with frontend/gtil/sklearn submodules | VERIFIED | `#[pymodule] fn _treelite_rs(...)` registering three submodules |
| `crates/treelite-py/src/error.rs` | `TreeliteError` + `From<E>` impls + `guard`/`catch_unwind` | VERIFIED | `create_exception!`, `err_to_treelite!` macro for 7 crate enums, `fn guard`, `std::panic::catch_unwind` |
| `crates/treelite-py/src/model.rs` | `#[pyclass] Model` + serialize/deserialize/dump/concatenate | VERIFIED | `#[pyclass(unsendable)]`, `serialize_bytes`, `deserialize_bytes`, `dump_as_json`, `concatenate` methods; `ModelVariant`-derived `input_type`/`output_type` (Pitfall 2 guarded) |
| `crates/treelite-py/src/frontend.rs` | thin `#[pyfunction]` wrappers for XGB/LGB loaders | VERIFIED | 5 `#[pyfunction]`s: `load_xgboost_json_str`, `_ubjson_bytes`, `_legacy_bytes`, `detect_xgboost_format_bytes`, `load_lightgbm_str` |
| `crates/treelite-py/src/gtil.rs` | `predict_f32`/`predict_f64` zero-copy + `backend=` dispatch + `check_feature_count` | VERIFIED | `PyReadonlyArray2`, `into_pyarray` (not `to_pyarray`), `py.detach`, `guard_assert`, `check_feature_count` (CR-01 fix), `dispatch_backend`, `BUILT_BACKENDS` const |
| `crates/treelite-py/src/sklearn.rs` | 9 thin `#[pyfunction]` wrappers over `treelite_sklearn::load_*` | VERIFIED | 9 `fn load_*` functions confirmed by `grep -c 'fn load_'` = 9; `PyReadonlyArray` zero-copy borrows |
| `crates/treelite-py/python/treelite_rs/__init__.py` | Package top-level with `Model`, `TreeliteError`, `frontend`, `gtil`, `sklearn` | VERIFIED | All symbols re-exported; `serialize`/`deserialize`/`concatenate` file shims present |
| `crates/treelite-py/python/treelite_rs/gtil/__init__.py` | dtype-dispatch + reshape shim + `backend=` kwarg | VERIFIED | `predict()` dispatches on data dtype (not model.input_type); `reshape` via `predict_output_shape`; `backend=` plumbed through |
| `crates/treelite-py/python/treelite_rs/sklearn/__init__.py` | `import_model` estimator→arrays extraction shim | VERIFIED | `def import_model`, `isinstance` dispatch on all estimator types, `__class__.__name__` fallback |
| `crates/treelite-py/python/treelite_rs/py.typed` | PEP 561 marker | VERIFIED | File present |
| `crates/treelite-py/tests/python/conftest.py` | `pytest.importorskip` skip guards | VERIFIED | `importorskip("treelite")` and `importorskip("treelite_rs")` confirmed |
| Test files (7) | All 7 test files present and passing | VERIFIED | `36 passed, 1 skipped` (rocm hardware-gated) via `uv run pytest` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `crates/treelite-py/src/gtil.rs` | `treelite_cubecl::predict_cpu`/`predict::<R,F>` | `dispatch_backend` match + `py.detach` | VERIFIED | `predict_cpu` in cpu arm, `cfg(feature="rocm")` gating confirmed in file |
| `crates/treelite-py/src/gtil.rs` | `check_feature_count` → `TreeliteError` | exact column-count guard before `as_slice()` | VERIFIED | `check_feature_count(data.shape()[1], model.inner.num_feature)` at lines 219 and 253; live spot-check raises correctly |
| `crates/treelite-py/src/model.rs` | `treelite_core::serialize_to_buffer` / `treelite_builder::concatenate` | `serialize_bytes` method / `concatenate` staticmethod | VERIFIED | `serialize_to_buffer` call confirmed; `concatenate` calls `treelite_builder::concatenate` |
| `crates/treelite-py/python/treelite_rs/gtil/__init__.py` | `_treelite_rs.gtil.predict_f32/_f64` | dtype-dispatch + reshape via `predict_output_shape` | VERIFIED | `predict_f32`/`predict_f64` dispatch, `reshape` call confirmed |
| `crates/treelite-py/python/treelite_rs/sklearn/__init__.py` | `_treelite_rs.sklearn.load_*` | estimator array extraction → compiled loader | VERIFIED | `isinstance` branches call `load_random_forest*`, `load_gradient_boosting*`, `load_hist_gradient_boosting*` etc. |
| `src/error.rs` guard | `TreeliteError` (not `PanicException`) | `catch_unwind` payload → `TreeliteError::new_err` | VERIFIED | `std::panic::catch_unwind` + `TreeliteError::new_err(format!("internal error (panic): {msg}"))` |
| Root `Cargo.toml` | `crates/treelite-py` | workspace `members` entry | VERIFIED | `grep -q "crates/treelite-py" Cargo.toml` → FOUND |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `gtil/__init__.py:predict` | `result` from `predict_f32`/`predict_f64` | `treelite_cubecl::predict_cpu::<F>` in Rust, receiving live numpy slice via `as_slice()` | Yes — dispatches to real inference engine, confirmed by A/B tests within 1e-5 of upstream | FLOWING |
| `src/model.rs:serialize_bytes` | `buf` from `treelite_core::serialize_to_buffer` | binary v5 serializer over real `Model` internal state | Yes — round-trip delta = 0.0 confirmed live | FLOWING |
| `sklearn/__init__.py:import_model` | estimator arrays extracted Python-side | `tree_.children_left`, `tree_.threshold`, etc. from fitted scikit-learn estimator | Yes — 10 live A/B cells within 1e-5 | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Load XGBoost model + predict | `treelite_rs.frontend.load_xgboost_model(...); gtil.predict(m, X)` | shape `(3, 1, 1)`, dtype float32 | PASS |
| Serialize round-trip delta | `serialize_bytes` → `deserialize_bytes` → predict delta | max delta = 0.0 | PASS |
| JSON dump parses | `m.dump_as_json(pretty_print=False)` → `json.loads(...)` | parsed dict with expected keys | PASS |
| sklearn import_model | `RandomForestRegressor.fit(X, y)` → `import_model(est)` → `predict(model, X)` | shape `(20, 1, 1)` | PASS |
| Zero-copy pointer identity | `X.__array_interface__['data'][0]` before/after predict | pointers equal | PASS |
| CR-01: too-wide matrix rejected | `predict_f32(m, wide)` where `wide.shape[1] = num_feature + 2` | raises `TreeliteError` with column count message | PASS |
| backend='cpu' matches no-kwarg | `predict(m, X, backend='cpu')` vs `predict(m, X)` | `np.allclose(...) = True` | PASS |
| Unknown backend raises | `predict(m, X, backend='nonexistent_backend')` | raises `TreeliteError` naming `BUILT_BACKENDS` | PASS |
| Full test suite | `uv run pytest crates/treelite-py/tests/python -q` | `36 passed, 1 skipped` | PASS |
| Workspace tests | `cargo test --workspace` | all green, 0 failures | PASS |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes exist for this phase. Not applicable.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PY-01 | 08-02 | From Python, load XGBoost / LightGBM / scikit-learn models | SATISFIED | `test_frontend.py` 5/5 (XGB+LGB); `test_sklearn_ab.py` 10/10 (sklearn); stale `[ ]` in REQUIREMENTS.md is a doc error, not a code gap |
| PY-02 | 08-02 | Predict over numpy arrays with zero-copy buffer I/O | SATISFIED | `test_predict_ab.py` + `test_zero_copy.py` GREEN; pointer identity confirmed |
| PY-03 | 08-03 | serialize/deserialize/JSON-dump from Python | SATISFIED | `test_serialize.py` 7/7 GREEN |
| PY-04 | 08-04 | `sklearn.import_model` marshals fitted estimators | SATISFIED | `test_sklearn_ab.py` 10/10 GREEN across all estimator types |
| PY-05 | 08-05 | `thiserror` errors → Python exceptions; no panic crosses FFI | SATISFIED | `test_errors.py` 4/4 GREEN; `guard`/`catch_unwind` present; process survives forced panic |
| PY-06 | 08-01 | Binding builds as abi3 wheel via maturin | SATISFIED | `_treelite_rs.abi3.so` installed; `cargo check -p treelite-py` exits 0; `import treelite_rs` succeeds |
| MEM-04 | 08-02 | Buffer-protocol borrows consumed zero-copy | SATISFIED | `PyReadonlyArray2.as_slice()` zero-copy path; `into_pyarray` (not `to_pyarray`) for output; pointer identity test GREEN |

**ORPHANED REQUIREMENTS CHECK:** REQUIREMENTS.md maps PY-01 as Phase 8 "Pending" — this is a stale checkbox. The implementation is fully delivered (5 XGB/LGB loader tests + 10 sklearn A/B tests all PASS). The checkbox update is a documentation action, not a code gap. No orphaned requirements exist.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `python/treelite_rs/gtil/__init__.py` | 133, 147 | `predict_leaf`/`predict_per_tree` unconditionally raise `TreeliteError("... not yet wired ...")` | INFO | Intentional stubs for upstream API-surface parity (D-01); `predict` (default/raw kind) is fully functional; no test asserts these stubs; no `TBD`/`FIXME`/`XXX` marker — this is a deliberately-documented limitation, not an unreferenced debt marker |
| `python/treelite_rs/gtil/__init__.pyi` | 19-24 | `.pyi` stub advertises `predict_leaf`/`predict_per_tree` as returning `np.ndarray` with no `NotImplemented` signal | INFO (WR-04 from review) | Type-checked callers get no signal; not a correctness issue for the phase goal |

No `TBD`, `FIXME`, or `XXX` markers found in any phase-8 source files.

**Advisory warnings from code review (WR-01..WR-05) — all open, none blocking:**
- WR-01: `predict_output_shape` not wrapped in `guard()` — PanicException vs TreeliteError message parity gap
- WR-02: `_dense_predict` raises `AttributeError` on non-numpy / wrong-ndim input instead of `TreeliteError`
- WR-03: `load_xgboost_model` uses platform-default encoding (no `encoding="utf-8"`)
- WR-04: `inspect` branch can raise `UnicodeDecodeError`/`ValueError` instead of `TreeliteError`
- WR-05: HistGB `import_model` uses unguarded sklearn private attrs without version check

These are robustness gaps, not correctness failures. None affect the 1e-5 core value or the phase goal.

### Human Verification Required

#### 1. Confirm predict_leaf / predict_per_tree stub behavior

**Test:** Call `treelite_rs.gtil.predict_leaf(model, X)` and `treelite_rs.gtil.predict_per_tree(model, X)` against any loaded model.
**Expected:** Both raise `TreeliteError` with message `"predict_leaf (LeafId kind) is not yet wired in the binding"` / `"predict_per_tree (ScorePerTree kind) is not yet wired in the binding"`. Interpreter does not abort.
**Why human:** These are intentional typed stubs (D-01 upstream-signature parity). The review notes (IN-04) flag that no test currently asserts the raise. Automated verification above does not cover them. Human should confirm the stub message is actionable and the behavior is intentional as per D-01.

#### 2. Update stale documentation: PY-01 checkbox + ROADMAP progress counter

**Test:** Read `.planning/REQUIREMENTS.md` line 81 and `.planning/ROADMAP.md` Progress table row for Phase 8.
**Expected:** `REQUIREMENTS.md` PY-01 is `[x]` (not `[ ]`); ROADMAP Phase 8 row reads `5/5 | Complete | 2026-06-11`.
**Why human:** This is a documentation update — no code change. The verifier cannot modify planning docs. The stale entries could mislead Phase 9 planning if not corrected.

### Gaps Summary

No code gaps. All 7 observable truths are VERIFIED, all 7 requirements are SATISFIED, all behavioral spot-checks pass (36/37 Python tests pass; 1 skipped is the hardware-gated ROCm cell, intentional by design). The critical review finding CR-01 (too-wide predict matrix silently wrong) is confirmed fixed in commit `146bf92` with a regression test.

The `human_needed` status is triggered by two items:
1. Confirmation of intentional `predict_leaf`/`predict_per_tree` stub behavior (IN-04 from code review, no current test asserts the raise)
2. Two stale documentation entries (PY-01 checkbox unchecked in REQUIREMENTS.md; ROADMAP Phase 8 counter at 4/5 instead of 5/5) that need a human to update

Neither item represents a code deficiency. The phase goal — "load/predict/serialize/dump from Python with zero-copy numpy I/O and abi3 wheel" — is fully achieved.

---

_Verified: 2026-06-11T05:00:00Z_
_Verifier: Claude (gsd-verifier)_

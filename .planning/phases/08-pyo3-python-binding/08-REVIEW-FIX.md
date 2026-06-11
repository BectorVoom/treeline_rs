---
phase: 08-pyo3-python-binding
fixed_at: 2026-06-11T00:00:00Z
review_path: .planning/phases/08-pyo3-python-binding/08-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 5
skipped: 0
already_resolved: 1
status: all_fixed
---

# Phase 8: Code Review Fix Report

**Fixed at:** 2026-06-11T00:00:00Z
**Source review:** .planning/phases/08-pyo3-python-binding/08-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope (critical_warning): 6 (CR-01 + WR-01..WR-05)
- Already resolved before this run: 1 (CR-01)
- Fixed this run: 5 (WR-01..WR-05)
- Skipped: 0

**Validation:** `cargo check -p treelite-py` and `cargo check -p treelite-py --features rocm`
both clean. Editable wheel rebuilt (`uv run maturin develop --manifest-path
crates/treelite-py/Cargo.toml`); `uv run pytest crates/treelite-py/tests/python -q`
is green at **39 passed, 1 skipped** (up from 37 passed — two new WR-02 regression
tests). The 1e-5 core-value / numeric path is untouched; all fixes are
error-handling and encoding only.

## Already Resolved (not re-applied)

### CR-01: `gtil.predict_f32`/`predict_f64` never validate input column count

**File:** `crates/treelite-py/src/gtil.rs`
**Status:** Already fixed and regression-tested in commit `146bf92` prior to this run.
**Verification:** Confirmed the `check_feature_count` guard is present
(`crates/treelite-py/src/gtil.rs:79-88`) and called in both predict entry points
(`predict_f32` line 219, `predict_f64` line 253). The too-wide regression test
`test_predict_too_wide_raises` (covering f32 and f64) exists in
`crates/treelite-py/tests/python/test_zero_copy.py` and passes. Not re-applied or
duplicated.

## Fixed Issues

### WR-01: `predict_output_shape` not wrapped in `guard()` — panic surfaces as `PanicException`

**Files modified:** `crates/treelite-py/src/gtil.rs`
**Commit:** `8c669e5`
**Applied fix:** Changed `predict_output_shape` to return `PyResult2<Vec<u64>>` and
wrapped its body in the existing `guard_assert(|| Ok(...))` remap. A panic in
`output_shape` on a degenerate model now becomes the single catchable
`TreeliteError` (D-07) instead of a bare pyo3 `PanicException`, restoring the
predict-path message parity the rest of the binding upholds (D-06). The Python
`predict` shim already consumes the returned vector via `tuple(...)`, unchanged.

### WR-02: `_dense_predict` raised wrong exception type for non-numpy / wrong-ndim input

**Files modified:** `crates/treelite-py/python/treelite_rs/gtil/__init__.py`,
`crates/treelite-py/tests/python/test_zero_copy.py`
**Commit:** `31fedaf`
**Applied fix:** Promoted the `numpy` import to runtime (was `TYPE_CHECKING`-only) and
added a rank/type guard at the top of `_dense_predict`: a non-`ndarray` input (a
Python list lacking `.dtype`) and a non-2-D array now raise `TreeliteError` (D-06)
instead of a bare `AttributeError` or a misleading "dtype does not match" message.
Per D-03 the input is rejected, not silently coerced/copied. Added two regression
tests (`test_predict_wrong_ndim_raises_treelite_error`,
`test_predict_non_array_raises_treelite_error`) that lock the behavior on the
high-level `gtil.predict` path.

### WR-04: `inspect`/`detect` path raised `UnicodeDecodeError` / `ValueError` outside D-06

**Files modified:** `crates/treelite-py/python/treelite_rs/frontend.py`
**Commit:** `f4889a8`
**Applied fix:** In the `format_choice='inspect'` branch, wrapped
`data.decode("utf-8")` in a `try/except UnicodeDecodeError` that re-raises a
`TreeliteError`, and changed the undetected-format `raise ValueError(...)` to
`raise _treelite_rs.TreeliteError(...)`. Every rejection on this path is now one
catchable `TreeliteError` (D-06).

### WR-03: path loaders used platform-default `read_text()` encoding

**Files modified:** `crates/treelite-py/python/treelite_rs/frontend.py`
**Commit:** `952c8ba`
**Applied fix:** Added `encoding="utf-8"` to the three `path.read_text()` calls (the
`use_suffix` JSON branch and the explicit `json` branch of `load_xgboost_model`,
and `load_lightgbm_model`). XGBoost JSON and LightGBM dumps are UTF-8; loads are now
platform-independent and a non-ASCII feature name no longer mis-decodes or raises
`UnicodeDecodeError` on a non-UTF-8 locale.

### WR-05: HistGB import reached sklearn private attrs with no version guard

**Files modified:** `crates/treelite-py/python/treelite_rs/sklearn/__init__.py`
**Commit:** `172cec9`
**Applied fix:** Added a `_HISTGB_MIN_SKLEARN = "1.0.0"` floor and an up-front guard
in `_import_hist_gradient_boosting`: before any private-attribute access it checks
the installed `sklearn.__version__` against the floor and verifies the fitted
estimator exposes the `_predictors` / `_baseline_prediction` private attributes,
raising an actionable `TreeliteError` (D-06) on an unsupported version / layout
instead of a bare `AttributeError` deep in the node-extraction loop.

---

_Fixed: 2026-06-11T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_

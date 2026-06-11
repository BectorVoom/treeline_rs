---
phase: 08-pyo3-python-binding
plan: 04
subsystem: python-binding
tags: [pyo3, sklearn, import-model, estimator-marshalling, ab-1e-5, py-04]
requires:
  - "treelite_rs.Model pyclass + frontend loaders + gtil.predict (08-02)"
  - "treelite-sklearn array-signature loaders (Phase 4, D-01)"
  - treelite-core
provides:
  - "treelite_rs.sklearn.import_model(estimator) — fitted-estimator marshaller (PY-04)"
  - "_treelite_rs.sklearn.load_* — 9 thin array-loader pyfunctions over treelite_sklearn::load_*"
  - "treelite_rs.sklearn.load_* re-exports (raw array loaders, A/B-direct callable)"
affects:
  - "crates/treelite-py/src/sklearn.rs (new — array-loader pyfunctions)"
  - "crates/treelite-py/src/lib.rs (sklearn submodule registration)"
  - "crates/treelite-py/python/treelite_rs/sklearn/ (new package — importer.py port)"
  - "crates/treelite-py/python/treelite_rs/__init__.py (sklearn re-export)"
tech-stack:
  added: []
  patterns:
    - "ArrayOfArrays<'py, T>: Vec<PyReadonlyArray1> guards + Vec<&[T]> slices in one struct — zero-copy &[&[T]] borrow with the guards keeping the numpy buffers alive across the loader call"
    - "estimator->arrays extraction is PYTHON-SIDE (port of importer.py); the estimator object never crosses the FFI boundary — only numpy arrays do (D-01)"
    - "HistGB packed node bytes copied into owned Box<[u8]> per tree (acceptable one-time copy; small per-tree node tables, not the zero-copy float matrices)"
    - "pure-Python treelite_rs.sklearn package SHADOWS the compiled _treelite_rs.sklearn submodule so import_model resolves while the compiled loaders stay reachable via _treelite_rs.sklearn"
    - "SklError -> single TreeliteError (D-06); typed PyReadonlyArray1<i64/f64> rejects wrong dtype before the body (T-08-10)"
key-files:
  created:
    - crates/treelite-py/src/sklearn.rs
    - crates/treelite-py/python/treelite_rs/sklearn/__init__.py
    - crates/treelite-py/python/treelite_rs/sklearn/__init__.pyi
  modified:
    - crates/treelite-py/src/lib.rs
    - crates/treelite-py/python/treelite_rs/__init__.py
    - crates/treelite-py/tests/python/test_sklearn_ab.py
decisions:
  - "import_model dispatches on isinstance against the sklearn estimator classes (verbatim importer.py order: HistGB first, then RF/ET reg, RF/ET clf, IsolationForest, GB reg, GB clf), NOT on __class__.__name__ string — the upstream port uses isinstance and it keeps the ExtraTrees-is-a-RandomForest subclass relationship correct."
  - "The pure-Python treelite_rs.sklearn package replaces the 08-02 empty-compiled-submodule re-export guard in __init__.py. `from . import sklearn` now binds the python-source package (which itself re-exports the compiled _treelite_rs.sklearn.load_* loaders); treelite_rs.sklearn.import_model resolves to the port shim."
  - "Zero-copy &[&[T]] is assembled in src/sklearn.rs via an ArrayOfArrays<'py,T> holding both the PyReadonlyArray1 guards and the &[T] slices; a std::mem::transmute extends each slice lifetime to 'py because the backing guard is moved into the same struct (the buffer outlives the slices). Wrong dtype is rejected by the typed PyReadonlyArray1<T> extract; non-contiguous -> typed TreeliteError, never a silent copy."
  - "HistGB `nodes` cross as a Python list of `bytes`; each is copied out into an owned Box<[u8]> (NodeBuffers) — a deliberate one-time copy (the packed node tables are small and not in the predict hot path), unlike the zero-copy float columns. raw_left_cat_bitsets ride the u32 ArrayOfArrays; categories_map crosses as Option<Vec<Vec<i64>>> -> Option<&[Vec<i64>]>."
  - "test_sklearn_ab.py is a LIVE A/B (D-11): for each estimator it fits a small estimator (mirroring fixtures/capture_sklearn.py construction), imports through BOTH treelite_rs.sklearn.import_model and upstream treelite.sklearn.import_model, predicts the same X via each stack's gtil.predict, and asserts assert_allclose(atol=1e-5, rtol=0). 10 cells cover RF/ET reg+clf, GB reg+clf, IsolationForest, HistGB reg (numerical+categorical) + clf."
metrics:
  duration: ~12min
  tasks: 2
  files: 6
  completed: 2026-06-11
---

# Phase 8 Plan 4: sklearn import_model Marshalling Summary

`treelite_rs.sklearn.import_model(fitted_estimator)` now marshals a fitted
scikit-learn estimator into a `treelite_rs.Model` that predicts within 1e-5 of
upstream `treelite.sklearn.import_model` across the full estimator set —
RandomForest / ExtraTrees (Regressor + Classifier), GradientBoosting (Regressor +
Classifier), IsolationForest, and HistGradientBoosting (Regressor + Classifier,
numerical + categorical). The heavy estimator→arrays logic is a branch-for-branch
Python-side port of upstream `importer.py`; the estimator object never crosses
the FFI boundary — only numpy arrays do, borrowed zero-copy into the existing
Phase-4 `treelite_sklearn` array-signature loaders (PY-04 GREEN).

## What Was Built

### Task 1 — sklearn array-loader pyfunctions (commit `6db67c9`, PY-04)

- **`src/sklearn.rs`** (new): one thin `#[pyfunction]` per `treelite_sklearn::load_*`
  entry point — 9 loaders: `load_random_forest_regressor`/`_classifier`,
  `load_extra_trees_regressor`/`_classifier`, `load_gradient_boosting_regressor`/`_classifier`,
  `load_isolation_forest`, `load_hist_gradient_boosting_regressor`/`_classifier`.
  - `ArrayOfArrays<'py, T>` holds a `Vec<PyReadonlyArray1<'py, T>>` (the borrow
    guards) plus a `Vec<&'py [T]>` (the slice views) in one struct: each per-tree
    column is borrowed zero-copy (`as_slice()`), the slice lifetime extended to
    `'py` via `transmute` because the guard is moved into the same struct (the
    numpy buffer outlives the slices), and `.view()` yields the `&[&[T]]` the
    loader signature expects. A wrong-dtype element is rejected by the typed
    `PyReadonlyArray1<T>` extract (T-08-10); a non-contiguous element is a typed
    `TreeliteError`, never a silent copy.
  - HistGB `nodes` cross as a Python list of `bytes`, copied out into owned
    `Box<[u8]>` per tree (`NodeBuffers`) — a deliberate one-time copy (small
    packed node tables, not the hot path); `raw_left_cat_bitsets` ride the u32
    `ArrayOfArrays`; `categories_map` crosses as `Option<Vec<Vec<i64>>>` →
    `Option<&[Vec<i64>]>`.
  - Each loader maps `SklError → TreeliteError` via the existing `?`/`PyResult2`
    seam (D-06) and wraps the result `Model { inner }`.
- **`src/lib.rs`**: the previously-empty `sklearn` submodule now calls
  `sklearn::register(&sklearn)` to add all 9 loaders under `_treelite_rs.sklearn`.

### Task 2 — import_model estimator→arrays shim (commit `71556a9`, PY-04 GREEN)

- **`python/treelite_rs/sklearn/__init__.py`** (new): port of upstream
  `importer.py`'s `import_model(sklearn_model)`:
  - `isinstance` dispatch (verbatim upstream order): HistGB first
    (`_import_hist_gradient_boosting`), then RF/ET regressor, RF/ET classifier,
    IsolationForest, GB regressor, GB classifier.
  - `_extract_forest` runs the per-tree `ArrayOfArrays` extraction
    (children_left/right, feature, threshold, value, n_node_samples,
    weighted_n_node_samples, impurity), with GB leaf-shrink by `learning_rate`
    (importer.py:218-223), IsolationForest isolation depths via the ported
    `_calculate_depths`/`_expected_depth`/`_harmonic` helpers + the feature
    subsample remap (importer.py:208-216), and the GB base-score derivation
    (DummyRegressor.constant_ / `_raw_predict_init`).
  - `_import_hist_gradient_boosting` ports the packed-node + `features_map`
    (feat_remapper) + `categories_map` (embedded-OrdinalEncoder remap) extraction
    and dispatches to the HistGB regressor/classifier loaders.
  - The dtype contract mirrors the loader signatures: children/feature/n_node_samples
    are int64; threshold/value/weighted/impurity are float64; node_count is an
    int64 array; HistGB nodes are raw `bytes`.
- **`python/treelite_rs/sklearn/__init__.pyi`** (new): hand-written stub for
  `import_model` + the 9 raw loaders (D-10).
- **`python/treelite_rs/__init__.py`**: `from . import sklearn` now binds the
  pure-Python package (which re-exports the compiled `_treelite_rs.sklearn.load_*`),
  replacing the 08-02 empty-submodule re-export guard; `sklearn` is in `__all__`.
- **`tests/python/test_sklearn_ab.py`**: flipped GREEN — 10 live A/B cells
  (`import_model` vs upstream, predict within 1e-5) across the full estimator set.

**Verification:** `cargo build -p treelite-py` exits 0; `cargo test --workspace`
green (no regression); `uv run pytest test_sklearn_ab.py` → 10 passed; full python
suite → 28 passed (was 18 in 08-03), 6 skipped (the backend/errors slices owned
by 08-05).

## Deviations from Plan

None of substance — plan executed as written. One naming clarification recorded as
a decision: `import_model` dispatches via `isinstance` against the sklearn estimator
classes (the verbatim `importer.py` mechanism) rather than on
`sklearn_model.__class__.__name__` string-matching as the RESEARCH skeleton sketched.
`isinstance` is the upstream port's actual approach and correctly preserves the
ExtraTrees-subclass-of-RandomForest relationship. The acceptance grep
(`__class__.__name__\|RandomForest\|HistGradientBoosting`) still passes via the
`RandomForest`/`HistGradientBoosting` class references in the dispatch.

## Acceptance-Criteria Notes (grep-precision)

Task 1:
- `grep -v '^//' src/sklearn.rs | grep -c 'fn load_'` = 9 (one wrapper per loader).
- `grep -q 'treelite_sklearn::load' src/sklearn.rs` — present.
- `grep -q 'PyReadonlyArray' src/sklearn.rs` — present (zero-copy array borrow).
- `cargo build -p treelite-py` exits 0.

Task 2:
- `grep -q 'def import_model' sklearn/__init__.py` — present.
- `grep -q 'RandomForest\|HistGradientBoosting\|__class__.__name__' sklearn/__init__.py`
  — present (per-estimator dispatch ported).
- `uv run pytest test_sklearn_ab.py -x` exits 0 (10 passed — PY-04 GREEN).

## Authentication Gates

None.

## Known Stubs

None introduced by this plan. (The 08-02 `predict_leaf`/`predict_per_tree`
not-yet-wired raises are unchanged and unrelated to PY-04.)

## Threat Flags

None. The implementation matches the plan's `<threat_model>`: T-08-09
(malformed/ill-shaped sklearn arrays → OOB in the loader) is mitigated because
the loaders themselves validate dimensions/topology → typed `SklError` →
`TreeliteError` (Phase-4 bounds-checks intact, no transmute on the model data);
T-08-10 (wrong-dtype numpy array) by the typed `PyReadonlyArray1<i64/f64>` extract
rejecting a wrong dtype before the loader body, with the Python extraction casting
to the expected dtype (port of importer.py's `ArrayOfArrays`). No new security
surface beyond the plan's register. (The single internal `transmute` in
`src/sklearn.rs` is a slice-LIFETIME extension only — same type `&[T]`, the
backing buffer is co-located in the struct — not a data reinterpret cast.)

## Self-Check: PASSED

- All 3 created + 3 modified files exist on disk (Edit/Write would have errored otherwise).
- Commits `6db67c9` (Task 1) + `71556a9` (Task 2) confirmed in `git log --oneline`.
- `cargo build -p treelite-py` exits 0; `cargo test --workspace` green (0 failures across all binaries).
- `uv run pytest crates/treelite-py/tests/python` → 28 passed, 6 skipped, exit 0;
  `test_sklearn_ab.py` → 10 passed.
- Core-value 1e-5 GREEN: `import_model` A/B vs upstream `treelite.sklearn.import_model`
  matches within 1e-5 across RF/ET reg+clf, GB reg+clf, IsolationForest, HistGB
  reg (numerical+categorical) + clf — with only numpy arrays crossing the FFI boundary (PY-04).

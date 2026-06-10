# Phase 8: PyO3 Python Binding - Research

**Researched:** 2026-06-11
**Domain:** PyO3 0.28 / rust-numpy 0.28 / maturin abi3 packaging — exposing the proven Rust tree-ensemble pipeline to Python as the sole external binding
**Confidence:** HIGH (versions verified against crates.io + local venv; APIs cross-checked against pyo3 0.28.3 docs; in-repo seam files read directly)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions (D-01 .. D-11 — DO NOT relitigate)
- **D-01:** Drop-in 1:1 mirror of upstream Treelite's public Python API: `frontend.load_xgboost_model` / `load_xgboost_model_legacy_binary` / `load_lightgbm_model` / `from_xgboost*` / `from_lightgbm`; `Model` with `serialize`/`serialize_bytes`/`deserialize`/`deserialize_bytes`/`dump_as_json`/`num_tree`/`num_feature`/`input_type`/`output_type`/`concatenate`/`HeaderAccessor`/`TreeAccessor`; `gtil.predict`/`predict_leaf`/`predict_per_tree`; `sklearn.import_model`; `core.TreeliteError`.
- **D-02:** Importable module name is **`treelite_rs`** (NOT `treelite`) — coexists with pip-installed upstream `treelite` in one venv. Drop-in usage: `import treelite_rs as treelite`.
- **D-03:** Strict zero-copy input or error. Predict requires a C-contiguous numpy array whose dtype exactly matches the model's `input_type` (f32/f64). On mismatch (non-contiguous, wrong dtype) raise `TreeliteError` telling the caller to convert. No hidden allocation.
- **D-04:** Backend fixed at install time — one wheel per backend (`cpu`, `rocm`, etc.). Maturin bakes the chosen Cargo feature into the wheel.
- **D-05:** A GPU wheel bundles CPU too; `predict(..., backend=...)` kwarg (additive, defaults `'cpu'`) picks within the installed wheel. The `cpu` wheel exposes only `'cpu'`. Requesting an un-built backend raises a typed exception.
- **D-06:** Single `TreeliteError(Exception)` — every Rust `thiserror` variant surfaces as one `TreeliteError` with a descriptive message. Callers branch on message text, not type.
- **D-07:** Panics caught and surfaced as `TreeliteError` (or RuntimeError-style), never an interpreter abort.
- **D-08:** Device-absent / un-built-backend selection is a typed failure surfaced as `TreeliteError` (from `CubeclError::DeviceUnavailable`); never a silent CPU fallback.
- **D-09:** abi3-py310 floor. Single abi3 wheel covering CPython 3.10+. PyO3 0.28 abi3 build.
- **D-10:** Ship PEP 561 type info — `py.typed` marker + hand-written `.pyi` stubs for the public surface.
- **D-11:** Live A/B equivalence pytest: imports both `treelite_rs` and pip-installed `treelite`, loads the same fixtures, predicts the same inputs, asserts within 1e-5. Runs in the `uv run`/golden-capture venv.

### Claude's Discretion (research/planner resolves)
- **sklearn marshalling location (PY-04):** port upstream `sklearn/importer.py` as Python-side estimator→arrays extraction calling the existing `treelite-sklearn` array entry points. The estimator stays in Python; only numpy arrays cross into Rust. *(Recommendation below: thin `.py` shim.)*
- **New crate name & layout:** `treelite-py` workspace member + maturin config; whether Python-side shims are thin `.py` wrappers over one compiled `_treelite_rs` extension or direct pyo3 submodules. *(Recommendation below: single compiled `_treelite_rs` + thin `.py` package shims.)*
- **GIL/threading, buffer-protocol mechanics, numpy zero-copy return** — owned by the ROADMAP research flag. *(Resolved below.)*
- **Whether to also commit golden-vector assertions** alongside the live A/B pytest (planner refinement, not mandated).
- **`nthread`/`pred_margin` predict config** — follows upstream `gtil.predict` 1:1.

### Deferred Ideas (OUT OF SCOPE)
- sklearn `export_model` (only `import_model` required).
- Memory-efficiency hardening through the Python path (bytemuck recast beyond buffer-protocol borrow, smallvec/compact_str, custom allocator) — Phase 9.
- CUDA/wgpu wheels validated on real hardware (build-supported, "not run — no device").
- Auto-routing CPU↔GPU crossover inside `predict` (explicitly NOT done; `backend=` is explicit).
- Frozen golden-vector assertions in CI (planner discretion).
- `coerce-with-copy` numpy ergonomics (rejected for v1 in favor of strict zero-copy).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PY-01 | From Python, load XGBoost / LightGBM / sklearn models | `frontend.*` thin pyo3 wrappers over `treelite_xgboost::{load_xgboost_json,load_xgboost_ubjson,load_xgboost_legacy}`, `treelite_lightgbm::load_lightgbm`, `treelite_sklearn::load_*`. File-read happens Python-side; bytes/str cross to Rust (Standard Stack + Architecture Patterns). |
| PY-02 | Predict over numpy arrays with zero-copy buffer I/O | `PyReadonlyArray2<f32/f64>` borrow + `as_slice()` → `treelite_gtil::predict::<O>` / `predict_sparse::<O>`; output via `IntoPyArray` (zero-copy from owned `Vec`). Strict dtype/contiguity gate (D-03). See Buffer Protocol + Zero-Copy Return patterns. |
| PY-03 | Serialize/deserialize/JSON-dump | `Model.serialize_bytes` → `treelite_core::serialize_to_buffer`; `deserialize_bytes` → `treelite_core::deserialize`; `dump_as_json` → `treelite_core::serialize::json::dump_as_json_string`. PyBuffer frame round-trip maps `Frame<'a>` ↔ `_TreelitePyBufferFrame`. See Frame Mapping pattern. |
| PY-04 | `sklearn.import_model` marshals fitted estimators | Port `sklearn/importer.py` as a `.py` shim doing estimator→arrays extraction, calling pyo3-wrapped `treelite_sklearn` array loaders. See sklearn marshalling pattern. |
| PY-05 | thiserror errors → Python exceptions (no panic crosses FFI) | `create_exception!` → `TreeliteError`; blanket `From<E: Display> for PyErr`-style mapping; pyo3 auto-`catch_unwind` converts panics to `PanicException` (D-07 wraps to `TreeliteError`). See Error Mapping pattern. |
| PY-06 | abi3 wheel via maturin | `[tool.maturin]` with `features = ["pyo3/abi3-py310"]`, `bindings = "pyo3"`, per-backend feature wheels. See Packaging pattern. |
| MEM-04 | Python buffer-protocol borrowed buffers consumed zero-copy | `PyReadonlyArray` holds the GIL-scoped borrow; `as_slice()` returns `&[O]` aliasing numpy memory with NO copy. `.as_ptr()` identity assertion proves it. See Validation Architecture. |
</phase_requirements>

## Summary

Phase 8 wraps the already-green Rust pipeline (`treelite-core` serialize, `treelite-{xgboost,lightgbm,sklearn}` loaders, `treelite-gtil` predict, `treelite-cubecl`/`treelite-harness` backend seam) in a single compiled PyO3 extension named `_treelite_rs`, exposed to Python as the `treelite_rs` package whose API is a 1:1 mirror of upstream Treelite 4.7.0's Python surface. All the heavy lifting already exists and is validated to 1e-5; this phase is **a binding layer, not new compute**. The three research-flag gaps — buffer-protocol zero-copy borrow, numpy zero-copy return, and GIL/threading — all resolve cleanly with rust-numpy 0.28's `PyReadonlyArray`/`IntoPyArray` and pyo3 0.28's `Python::detach`.

The verified stack is **pyo3 0.28.3**, **numpy (rust-numpy) 0.28.0**, **maturin 1.13.3**, building an **abi3-py310** wheel. These three crate versions are co-released and ABI-compatible (numpy 0.28 targets pyo3 0.28). The local environment is Python 3.13.13 with numpy 2.4.6 and upstream `treelite` 4.7.0 already pip-installed in the `uv` venv — so the D-11 live A/B pytest has its witness in place today.

**Two notable API shifts from training data, both verified:** (1) pyo3 0.28 renamed `Python::allow_threads` → **`Python::detach`** (the GIL-release call for the predict hot path); (2) pyo3 catches Rust panics at every `#[pyfunction]`/`#[pymethods]` boundary automatically via an internal `catch_unwind` and raises `pyo3_runtime.PanicException` — no manual `catch_unwind` is needed to satisfy D-07's "no panic crosses the boundary," though a small wrapper is still wanted to remap `PanicException` → `TreeliteError` for D-06/D-07 message parity.

**Primary recommendation:** One `treelite-py` workspace crate compiling a single `cdylib` `_treelite_rs` (pyo3 `#[pymodule]` with nested submodules for `frontend`/`gtil`/`sklearn`), wrapped by a thin pure-Python `treelite_rs/` package (`python-source` layout) that ports the upstream `.py` ergonomics (path normalization, sklearn estimator→arrays extraction, scipy CSR unpacking) and re-exports the compiled symbols. Predict borrows numpy zero-copy via `PyReadonlyArray2`, releases the GIL with `py.detach()`, and returns results via `Vec::into_pyarray` (zero-copy move). Map every Rust error to one `create_exception!`-defined `TreeliteError` and wrap entry points so `PanicException` is re-surfaced as `TreeliteError`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Model file reading (open/read bytes) | Python shim (`.py`) | — | Upstream reads files Python-side; keeps Rust loaders pure (str/bytes in). Matches `frontend.py` `_normalize_path`. |
| Format parse (XGB/LGB/sklearn → Model) | Rust (`treelite-{xgboost,lightgbm,sklearn}`) | — | Already validated to 1e-5; pyo3 wrappers are thin. |
| sklearn estimator → numpy arrays extraction | Python shim (`.py`) | — | Estimator object is Python (`tree_.children_left` etc.); only arrays cross FFI (D-01 discretion). Port of `importer.py`. |
| numpy buffer borrow (zero-copy) | Rust boundary (`treelite-py` via rust-numpy) | — | `PyReadonlyArray` holds the GIL-scoped borrow; `as_slice()` aliases numpy memory (MEM-04). |
| Predict compute | Rust (`treelite-gtil` / `treelite-cubecl`) | — | The proven hot path; GIL released around it. |
| Output numpy array construction | Rust boundary (`IntoPyArray`) | — | `Vec<O>` moved into a `PyArray` zero-copy; reshape Python-side. |
| Backend selection (`backend=` kwarg) | Rust boundary (string → `Backend`) | Python shim (kwarg plumbing) | Selection within compiled-in backends (D-05); device-absent → `TreeliteError` (D-08). |
| Serialize / PyBuffer frames | Rust (`treelite-core::serialize`) | Python shim (frame→numpy if accessors exposed) | `Frame<'a>` already zero-copy borrows TreeBuf columns (SER-02); maps to `_TreelitePyBufferFrame`. |
| Error translation | Rust boundary (`From`/`create_exception!`) | — | One `TreeliteError` (D-06); panic catch (D-07). |
| Packaging / wheel build | maturin (`pyproject.toml`) | — | abi3-py310, per-backend feature wheels (D-04/D-06/D-09). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | 0.28.3 | Rust↔Python FFI: `#[pymodule]`, `#[pyclass]`, `#[pyfunction]`, abi3, exceptions, GIL | `[VERIFIED: crates.io]` The de-facto Rust↔CPython binding; CONTEXT D-01..D-10 pin 0.28. |
| `numpy` (rust-numpy) | 0.28.0 | Zero-copy numpy borrow (`PyReadonlyArray`) + return (`IntoPyArray`) | `[VERIFIED: crates.io]` Co-released with pyo3 0.28; the standard numpy bridge. |
| `maturin` | 1.13.3 | Build/package the abi3 wheel; per-feature builds | `[VERIFIED: PyPI]` The standard pyo3 wheel builder; CONTEXT D-06/D-09 pin maturin. |

### Supporting (already in the workspace — wired as path deps)
| Crate | Purpose | When to Use |
|---------|---------|-------------|
| `treelite-core` | `Model`, `serialize_to_buffer`, `deserialize`, `dump_as_json_string`, `Frame`, `DType` | serialize/deserialize/dump/field accessors (PY-03) |
| `treelite-xgboost` | `load_xgboost_json` / `load_xgboost_ubjson` (+ legacy entry — see Open Q) | `frontend.load_xgboost_model*` (PY-01) |
| `treelite-lightgbm` | `load_lightgbm` | `frontend.load_lightgbm_model` (PY-01) |
| `treelite-sklearn` | `load_random_forest_*` / `load_extra_trees_*` / `load_gradient_boosting_*` / `load_isolation_forest` / `load_hist_gradient_boosting_*` | `sklearn.import_model` (PY-04) |
| `treelite-gtil` | `predict::<O>` / `predict_sparse::<O>`, `Config`, `PredictKind`, `SparseCsr` | `gtil.predict/predict_leaf/predict_per_tree` (PY-02) |
| `treelite-cubecl` | `predict::<R>` / `predict_cpu`, `CubeclError` | `backend=` GPU path (D-05/D-08) |
| `thiserror` | 2.0.18 (workspace) | the error enums being mapped | error translation (PY-05) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rust-numpy `PyReadonlyArray` | raw pyo3 `PyBuffer<T>` | `PyBuffer` works (`as_slice` → `&[ReadOnlyCell<T>]`) but is lower-level: you re-implement dtype/contiguity/ndim checks rust-numpy already does. Use `PyBuffer` only if you must accept arbitrary buffer-protocol objects beyond numpy. **Recommend rust-numpy.** |
| single compiled module + `.py` shims | pure-pyo3 nested submodules (no `.py`) | Pure-pyo3 avoids a Python layer but forces the sklearn estimator→arrays extraction (heavy `tree_.*` attribute access) into Rust via `getattr` — ugly and slow. The `.py` shim ports `importer.py` almost verbatim. **Recommend `.py` shims.** |
| maturin | setuptools-rust | maturin is the pyo3-native, abi3-first, per-feature-wheel tool CONTEXT pins (D-06/D-09). No reason to deviate. |

**Installation (Cargo, in the new `crates/treelite-py/Cargo.toml`):**
```toml
[lib]
name = "_treelite_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28.3", features = ["abi3-py310", "extension-module"] }
numpy = "0.28.0"
treelite-core    = { path = "../treelite-core" }
treelite-xgboost = { path = "../treelite-xgboost" }
treelite-lightgbm = { path = "../treelite-lightgbm" }
treelite-sklearn = { path = "../treelite-sklearn" }
treelite-gtil    = { path = "../treelite-gtil" }
treelite-builder = { path = "../treelite-builder" }  # for Model.concatenate (D-01)
treelite-cubecl  = { path = "../treelite-cubecl" }   # default features = cpu

[features]
default = ["cpu"]
cpu  = []                                  # cpu always available
rocm = ["treelite-cubecl/rocm"]            # GPU wheel: bundles cpu + rocm
cuda = ["treelite-cubecl/cuda"]
wgpu = ["treelite-cubecl/wgpu"]
```
**`extension-module`** is required so the linked Python symbols aren't resolved at build time (the wheel loads against the host interpreter). **`abi3-py310`** produces one wheel for CPython ≥3.10 (D-09).

**Version verification (run this session):**
- `cargo search pyo3` → `pyo3 = "0.28.3"` `[VERIFIED: crates.io 2026-06-11]`
- `cargo search numpy` → `numpy = "0.28.0"` `[VERIFIED: crates.io 2026-06-11]`
- `pip index versions maturin` → `1.13.3` (latest) `[VERIFIED: PyPI 2026-06-11]`
- venv: Python 3.13.13, numpy 2.4.6, treelite 4.7.0 `[VERIFIED: uv run python 2026-06-11]`
- toolchain: rustc 1.95.0 `[VERIFIED]`

## Package Legitimacy Audit

> Ran the Package Legitimacy Gate. `slopcheck install pyo3 numpy maturin` → **3 OK** (all clean on crates.io). Registry existence cross-checked via `cargo search` (correct ecosystem). pyo3/numpy versions confirmed authoritative (numpy 0.28 is the co-released pyo3-0.28 bridge — verified against rust-numpy's own version-pairing). maturin confirmed on PyPI.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `pyo3` 0.28.3 | crates.io | ~7 yrs | very high (top crate) | github.com/PyO3/pyo3 | [OK] | Approved `[VERIFIED]` |
| `numpy` 0.28.0 | crates.io | ~7 yrs | high | github.com/PyO3/rust-numpy | [OK] | Approved `[VERIFIED]` |
| `maturin` 1.13.3 | crates.io / PyPI | ~5 yrs | very high | github.com/PyO3/maturin | [OK] | Approved `[VERIFIED]` |

**Packages removed due to slopcheck [SLOP]:** none
**Packages flagged [SUS]:** none
**Postinstall scripts:** N/A (Rust crates; maturin is a build tool, no postinstall hook). All three are PyO3-org maintained — the canonical, authoritative sources for this exact use case.

## Architecture Patterns

### System Architecture Diagram

```
  Python user code:  import treelite_rs as treelite
        │
        ▼
  ┌─────────────────────────────  treelite_rs/  (pure-Python package, python-source) ─────────────────┐
  │  __init__.py        → re-exports Model, frontend, gtil, sklearn, TreeliteError                      │
  │  frontend.py        → load_xgboost_model(path,...)  : open file Python-side, read bytes/str         │
  │  sklearn/__init__.py→ import_model(estimator)       : extract tree_.* arrays (port importer.py)     │
  │  gtil/__init__.py   → predict/predict_leaf/predict_per_tree(model, data, backend=...)               │
  │  *.pyi + py.typed   → PEP 561 stubs (D-10)                                                          │
  └───────────────────────────────────────────┬────────────────────────────────────────────────────────┘
        │ calls compiled symbols                │ hands numpy arrays / str / bytes
        ▼                                       ▼
  ┌─────────────────────────────  _treelite_rs  (compiled cdylib, pyo3 #[pymodule]) ─────────────────────┐
  │  #[pyclass] Model { inner: treelite_core::Model }                                                     │
  │     ├─ serialize_bytes()  → treelite_core::serialize_to_buffer → PyBytes (copy out, small)            │
  │     ├─ deserialize_bytes()→ treelite_core::deserialize                                                 │
  │     ├─ dump_as_json()     → serialize::json::dump_as_json_string                                       │
  │     ├─ num_tree / num_feature / input_type / output_type   (from Model accessors / variant)           │
  │     └─ concatenate()      → treelite_builder::concatenate                                              │
  │  submodule frontend:  load_xgboost_json_str / _ubjson / load_lightgbm_str / load_sklearn_*            │
  │  submodule gtil:      predict(model, &PyReadonlyArray2<O>, backend, kind, nthread)                    │
  │       ── strict dtype+contiguity gate (D-03) ──► as_slice() [ZERO-COPY borrow, MEM-04]                │
  │       ── py.detach(|| treelite_gtil::predict::<O>(...) OR treelite_cubecl::predict::<R>) ──            │
  │       ── Vec<O>.into_pyarray(py)  [ZERO-COPY return] ──► reshape to (num_row,num_target,max_class)    │
  │  TreeliteError = create_exception!(...);  From<XgbError/LgbError/SklError/GtilError/CubeclError>      │
  │  every #[pyfunction] auto catch_unwind → PanicException → remapped to TreeliteError (D-07)            │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │ path deps (already 1e-5-green)
        ▼
  treelite-core · treelite-xgboost · treelite-lightgbm · treelite-sklearn · treelite-gtil · treelite-cubecl
```

### Recommended Project Structure
```
crates/treelite-py/
├── Cargo.toml                # cdylib _treelite_rs, abi3-py310, per-backend features
├── pyproject.toml            # [tool.maturin] — module-name, python-source, features, include stubs
├── src/
│   ├── lib.rs                # #[pymodule] _treelite_rs { frontend, gtil, sklearn submodules }
│   ├── model.rs              # #[pyclass] Model + serialize/deserialize/dump/accessors
│   ├── frontend.rs           # #[pyfunction] loaders (str/bytes in)
│   ├── gtil.rs               # #[pyfunction] predict* with zero-copy numpy + backend + GIL release
│   ├── sklearn.rs            # #[pyfunction] array-signature sklearn loaders
│   └── error.rs              # create_exception! TreeliteError + From impls + panic remap
└── python/                   # python-source root (the importable package)
    └── treelite_rs/
        ├── __init__.py       # re-export from _treelite_rs; __version__
        ├── py.typed          # PEP 561 marker (D-10)
        ├── frontend.py       # path open/read → calls _treelite_rs.frontend.*
        ├── frontend.pyi
        ├── model.pyi
        ├── gtil/
        │   ├── __init__.py   # predict/predict_leaf/predict_per_tree (CSR unpack via scipy)
        │   └── __init__.pyi
        └── sklearn/
            ├── __init__.py   # import_model: estimator → arrays (port of importer.py)
            └── __init__.pyi
```

### Pattern 1: Zero-copy numpy INPUT borrow (D-03, MEM-04, PY-02)
**What:** Borrow a C-contiguous, exact-dtype numpy array as a Rust slice with no copy. The GIL-scoped `PyReadonlyArray` is the borrow guard; dropping it releases the numpy buffer lock.
**When to use:** every `gtil.predict*` dense call.
```rust
// Source: numpy 0.28 docs (PyReadonlyArray2 / as_slice) + pyo3 0.28 detach
use numpy::{PyReadonlyArray2, PyArray1, IntoPyArray};
use pyo3::prelude::*;
use treelite_gtil::{predict, Config, PredictKind};

#[pyfunction]
#[pyo3(signature = (model, data, *, nthread=-1, pred_margin=false))]
fn predict_f32<'py>(
    py: Python<'py>,
    model: &Model,                 // #[pyclass] wrapper
    data: PyReadonlyArray2<'py, f32>,  // strict f32; wrong dtype -> pyo3 TypeError before body
    nthread: i32,
    pred_margin: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    // D-03 strict contiguity gate. as_slice() returns AsSliceError if non-contiguous;
    // surface it as TreeliteError telling the caller to np.ascontiguousarray(...).
    let slice: &[f32] = data.as_slice()                       // ZERO-COPY: aliases numpy memory (MEM-04)
        .map_err(|_| TreeliteError::new_err(
            "input array must be C-contiguous; call np.ascontiguousarray(arr) first"))?;
    let num_row = data.shape()[0];
    let cfg = Config { kind: if pred_margin { PredictKind::Raw } else { PredictKind::Default }, nthread };

    // Release the GIL around the pure-Rust compute (Pattern 3).
    let out: Vec<f32> = py.detach(|| predict::<f32>(&model.inner, slice, num_row, &cfg))
        .map_err(TreeliteError::from)?;                       // GtilError -> TreeliteError (D-06)

    Ok(out.into_pyarray(py))                                  // ZERO-COPY return (Pattern 2)
    // reshape to (num_row,num_target,max_class) happens in the .py gtil shim
}
```
Dispatch f32 vs f64 by **inspecting `model.input_type` Python-side and calling the matching typed `#[pyfunction]`** (two entry points: `predict_f32`, `predict_f64`), or by accepting an untyped `&Bound<PyAny>`, reading its dtype, and downcasting to `PyReadonlyArray2<f32/f64>`. The two-typed-entry-points approach mirrors the harness's four-fn-pointer seam (`treelite-harness/src/lib.rs:99-107`) and keeps each path monomorphized — **recommend it.**

### Pattern 2: Zero-copy numpy RETURN (PY-02)
**What:** Move a Rust-owned `Vec<O>` into a numpy array without copying. `into_pyarray` transfers ownership of the Vec's allocation to numpy (numpy frees it on GC).
```rust
// Source: numpy 0.28 IntoPyArray
use numpy::IntoPyArray;
let v: Vec<f64> = /* predict output */;
let arr: Bound<PyArray1<f64>> = v.into_pyarray(py);   // no element copy; allocation moved
```
- `into_pyarray` (owns) is zero-copy. `to_pyarray` (borrows `&[T]`) **copies** — use `into_pyarray` for predict output.
- predict returns a **1-D flat** buffer `(num_row * num_target * max_num_class)` (`treelite-gtil/src/lib.rs:892` doc). Reshape to the upstream 3-D shape `(num_row, num_target, max(num_class))` **in the `.py` gtil shim** via `arr.reshape(...)` — a view, no copy. `predict_leaf` → `(num_row, num_tree)`; `predict_per_tree` → `(num_row, num_tree, lvshape[0]*lvshape[1])`. Compute these via `treelite_gtil::output_shape` / `Shape`.

### Pattern 3: GIL release around the predict hot path (research-flag — GIL/threading)
**What:** `Python::detach` (pyo3 0.28; **renamed from `allow_threads`**) releases the GIL so other Python threads run while Rust predicts.
```rust
let out = py.detach(|| predict::<f32>(&model.inner, slice, num_row, &cfg));
```
**Constraints / safety:**
- The closure must be `Send` and must **not** touch Python objects (the `Ungil` bound enforces this at compile time). `&model.inner` (a `treelite_core::Model`) and `slice` (`&[f32]` aliasing numpy memory) are plain Rust — fine to move in.
- **Thread-safety of the borrowed buffer:** the `PyReadonlyArray` borrow guard is created *before* `detach` and lives across it. While detached, the GIL is released, so other Python threads could theoretically mutate the array — but numpy's buffer-protocol export with `PyBUF_RECORDS_RO` (read-only) plus the `PyReadonlyArray` borrow flag means a concurrent in-place mutation attempt raises in the other thread, not here. The slice stays valid for the detached region. This is the standard rust-numpy pattern and is sound.
- `treelite-gtil` scalar predict is single-threaded (`Config.nthread` is recorded-not-used — `config.rs:39-43`); `treelite-cubecl` CPU/GPU may be internally parallel. Releasing the GIL benefits multi-threaded Python callers regardless. **Always release the GIL** around predict — there is no Python object access inside the compute.
- `nthread` is accepted and threaded into `Config` for upstream signature parity (D-01) even though the scalar engine ignores it for allocation.

### Pattern 4: Single `TreeliteError` + panic catch (D-06, D-07, PY-05)
**What:** One Python exception; every Rust error maps to it; panics never abort.
```rust
// Source: pyo3 0.28 create_exception! + From<E> for PyErr
use pyo3::create_exception;
use pyo3::exceptions::PyException;
create_exception!(_treelite_rs, TreeliteError, PyException);   // class: treelite_rs.TreeliteError

// One From impl per crate error enum (D-06: all collapse to one exception, message carries detail).
macro_rules! err_to_treelite {
    ($($t:ty),+) => { $(
        impl From<$t> for PyErr {
            fn from(e: $t) -> PyErr { TreeliteError::new_err(e.to_string()) }
        }
    )+ };
}
err_to_treelite!(
    treelite_xgboost::XgbError, treelite_lightgbm::LgbError, treelite_sklearn::SklError,
    treelite_gtil::GtilError, treelite_core::CoreError, treelite_cubecl::CubeclError,
    treelite_builder::BuilderError
);
```
**Panic handling (D-07) — verified behavior:** pyo3 0.28 wraps **every** `#[pyfunction]`/`#[pymethods]` body in an internal `catch_unwind`. A Rust panic does **not** abort the interpreter — it surfaces as `pyo3_runtime.PanicException` (subclass of `BaseException`). To honor D-06 (single `TreeliteError`) and D-07 literally, wrap fallible entry points so a trapped panic is re-raised as `TreeliteError`:
```rust
fn guard<T>(f: impl FnOnce() -> PyResult<T> + std::panic::UnwindSafe) -> PyResult<T> {
    match std::panic::catch_unwind(f) {
        Ok(r) => r,
        Err(p) => {
            let msg = p.downcast_ref::<&str>().map(|s| s.to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "internal panic".into());
            Err(TreeliteError::new_err(format!("internal error (panic): {msg}")))
        }
    }
}
```
This mirrors the in-repo discipline in `treelite-cubecl/src/device.rs` (07-01: `catch_unwind` around GPU client construction → `DeviceUnavailable`/`ClientInit`). **Note:** D-08's `CubeclError::DeviceUnavailable` is already a *typed Result*, not a panic — it flows through the `From` impl above to `TreeliteError` with no special handling. The `guard` is only for genuinely unexpected panics.

### Pattern 5: `backend=` kwarg → `Backend` selection (D-04/D-05/D-08)
**What:** additive kwarg, default `'cpu'`, picks among compiled-in backends.
```rust
#[pyfunction]
#[pyo3(signature = (model, data, *, backend="cpu", nthread=-1, pred_margin=false))]
fn predict_f32(/* ... */, backend: &str, /* ... */) -> PyResult<...> {
    let out = py.detach(|| match backend {
        "cpu" => treelite_cubecl::predict_cpu::<f32>(&model.inner, slice, num_row, &cfg)
                    .or_else(|_| /* scalar fallback for categorical/sparse, D-02 */
                             treelite_gtil::predict::<f32>(&model.inner, slice, num_row, &cfg)),
        #[cfg(feature = "rocm")]
        "rocm" => treelite_cubecl::predict::<cubecl::hip::HipRuntime, f32>(...),
        #[cfg(feature = "cuda")]
        "cuda" => treelite_cubecl::predict::<cubecl::cuda::CudaRuntime, f32>(...),
        #[cfg(feature = "wgpu")]
        "wgpu" => treelite_cubecl::predict::<cubecl::wgpu::WgpuRuntime, f32>(...),
        other => return Err(TreeliteError::new_err(format!(
            "backend '{other}' is not available in this wheel (built with: cpu{})", BUILT_BACKENDS))),
    }).map_err(TreeliteError::from)?;   // DeviceUnavailable -> TreeliteError (D-08), never silent CPU fallback
}
```
A backend requested but not compiled in (`#[cfg]` arm absent) hits the `other =>` branch → typed `TreeliteError`. A compiled-in-but-device-absent backend returns `CubeclError::DeviceUnavailable` → `TreeliteError` (D-08). Mirror the harness `Backend` enum (`treelite-harness/src/lib.rs:55-94`) and its `*_case()` `#[cfg]` gating.

### Pattern 6: PyBuffer frame ↔ `_TreelitePyBufferFrame` mapping (PY-03, SER-02)
**What:** the Rust `Frame<'a>` enum (`treelite-core/src/serialize/pybuffer.rs:25-44`) is the zero-copy borrowed image of upstream's `_TreelitePyBufferFrame` (`model.py:331-339`: `buf, format, itemsize, nitem`).
- The **`serialize_bytes`/`deserialize_bytes` round-trip does NOT need the frame layout** — it uses the *binary* serializer (`serialize_to_buffer` → `Vec<u8>` → `PyBytes`; `deserialize(&[u8])`). This is the simplest faithful path and what `Model.serialize_bytes`/`deserialize_bytes` should call. The byte buffer is small relative to predict I/O, so a copy out to `PyBytes` is acceptable (upstream also copies via `bytes_from_string_and_size`).
- The **`Frame` enum is needed only if you expose `HeaderAccessor`/`TreeAccessor` `get_field`** (D-01 lists them). Each `Frame` variant maps to upstream's format string + itemsize:

| `Frame` variant | upstream `format` | itemsize | numpy dtype |
|-----------------|-------------------|----------|-------------|
| `U8(&[u8])` | `=B` | 1 | uint8 |
| `I8(&[i8])` | `=b` | 1 | int8 |
| `U32` | `=L` | 4 | uint32 |
| `I32` | `=l` | 4 | int32 |
| `U64` | `=Q` | 8 | uint64 |
| `I64` | `=q` | 8 | int64 |
| `F32` | `=f` | 4 | float32 |
| `F64` | `=d` | 8 | float64 |
| `Str(&str)` | `=c` (S1) | 1 | bytes→decode utf-8 |

To expose `get_field` zero-copy, return each `Frame`'s borrowed slice as a numpy array via `PyArray1::borrow_from_array`-style aliasing **bound to the `Model`'s lifetime** (the frame borrows `&Model`, `pybuffer.rs:7` D-05 — the model must outlive the array). **Recommendation:** for v1, implement field accessors by **copying** the small column into a fresh numpy array (`to_pyarray`) to sidestep the lifetime-into-Python hazard; the zero-copy contract that matters for MEM-04 is the *predict* path, not field inspection. Flag the zero-copy-accessor option as a Phase-9 refinement.

### Anti-Patterns to Avoid
- **Copying numpy input before predict.** Defeats MEM-04/D-03. Borrow with `PyReadonlyArray`; if dtype/contiguity is wrong, *error* (D-03), don't silently `ascontiguousarray`.
- **Holding the GIL across predict.** Wrap the compute in `py.detach`. (But never construct/touch `Py*` objects inside `detach`.)
- **`to_pyarray` for predict output.** Copies. Use `into_pyarray` (moves the Vec).
- **Manual `catch_unwind` on every function "to be safe".** pyo3 already does it; only add the `guard` remap where you want `PanicException`→`TreeliteError` parity.
- **Reading model files in Rust.** Keep file I/O Python-side (matches upstream; keeps loaders pure str/bytes).
- **A second exception type.** D-06 mandates exactly one `TreeliteError`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| numpy dtype/contiguity/ndim validation | manual `PyBuffer` format-string parsing | `PyReadonlyArray2<T>` (typed) + `as_slice()` | rust-numpy does the strided/contiguity/alignment checks; typed param rejects wrong dtype before the body runs |
| zero-copy Vec→numpy | `PyArray::new` + manual memcpy | `Vec::into_pyarray` | moves the allocation; numpy owns+frees it |
| panic→exception bridging | custom FFI shims | pyo3's built-in `catch_unwind` (+ thin remap) | pyo3 already guards every boundary; PanicException is automatic |
| abi3 wheel + Python ABI tags | hand-rolled setup.py/cffi | maturin `abi3-py310` | one wheel for CPython ≥3.10, correct tags, per-feature builds |
| sklearn estimator → arrays | re-deriving tree internals in Rust | port `importer.py` Python-side | `tree_.children_left/threshold/value/...` are Python attrs; extraction is pure numpy glue |
| CSR unpacking | custom sparse parser | scipy `csr_matrix.data/indices/indptr` in the `.py` shim | upstream does exactly this (`gtil.py:178-186`); arrays cross as `SparseCsr` |

**Key insight:** This phase has *almost no novel compute*. Every numerically-load-bearing line already exists and is 1e-5-green. The risk surface is entirely in the **binding correctness** (dtype dispatch, zero-copy borrow lifetimes, error/panic mapping, packaging) — so lean on rust-numpy/pyo3/maturin idioms and keep the Rust glue thin.

## Runtime State Inventory

> This is a greenfield additive phase (new `treelite-py` crate + new Python package), not a rename/refactor. No existing runtime state is being renamed or migrated. The one cross-cutting touch is the workspace member list.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys involved. | none |
| Live service config | None. | none |
| OS-registered state | None. | none |
| Secrets/env vars | None. The maturin build reads `VIRTUAL_ENV`/interpreter from the active `uv` venv (untracked, main-tree only — per project memory). | Build/test must run on the main tree via `uv run`, not a worktree. |
| Build artifacts / installed packages | A new `_treelite_rs` extension gets `maturin develop`-installed into the `uv` venv alongside upstream `treelite`. Root `Cargo.toml` `members` must gain `crates/treelite-py`. Root `pyproject.toml` currently declares `name = "treeline-rs"` with `requires-python = ">=3.13"` and lists `treelite>=4.7.0` as a dep — the new package's maturin `pyproject.toml` is a **separate** file in `crates/treelite-py/` (do not overload the root project file). | Add member to workspace; create `crates/treelite-py/pyproject.toml`; `maturin develop` into the existing venv. |

**Verified explicitly:** the upstream `treelite` 4.7.0 is already importable in the venv (`uv run python -c "import treelite"` → 4.7.0), so the D-11 A/B harness has both sides available with no extra install.

## Common Pitfalls

### Pitfall 1: `allow_threads` no longer exists in pyo3 0.28
**What goes wrong:** code copied from pyo3 ≤0.24 tutorials calls `py.allow_threads(...)` and fails to compile.
**Why it happens:** pyo3 0.28 renamed it to `Python::detach` (paired with `Python::attach`).
**How to avoid:** use `py.detach(|| ...)`. `[VERIFIED: pyo3.rs/v0.28.3/parallelism.html]`
**Warning signs:** `no method named allow_threads`.

### Pitfall 2: `input_type`/`output_type` read stale `kInvalid` before serialization staging
**What goes wrong:** `Model.input_type` returns `"invalid"`. The core `Model.threshold_type()`/`leaf_output_type()` are only populated by `stage_serialization_fields()` (`model.rs:151,161-162` set them at serialize time); a freshly loaded model has `DType::kInvalid` (`model.rs:135-136`).
**Why it happens:** the type tags are derived lazily from the variant during serialization.
**How to avoid:** in the `#[pyclass] Model`, compute `input_type`/`output_type` **directly from the variant** (`ModelVariant::F32 => "float32"`, `F64 => "float64"`) rather than from the staged DType field — or call `stage_serialization_fields()` first. The variant is the source of truth. `[VERIFIED: model.rs read this session]`
**Warning signs:** `model.input_type == "invalid"`; A/B dtype dispatch sends f64 data to the f32 path.

### Pitfall 3: `into_pyarray` returns a 1-D array; upstream returns N-D
**What goes wrong:** A/B test fails on `.shape` even though values match — `treelite_gtil::predict` returns a flat `(num_row*num_target*max_class)` Vec (`lib.rs:877-883`), but upstream returns `(num_row, num_target, max(num_class))`.
**How to avoid:** reshape in the `.py` gtil shim using dims from `treelite_gtil::output_shape`/`Shape` (exposed via a small `#[pyfunction] output_shape`). Reshape is a view (no copy). `[VERIFIED: gtil/src/lib.rs + config.rs]`
**Warning signs:** `ValueError: cannot reshape`; assertion on `.shape` mismatch.

### Pitfall 4: wrong-dtype numpy silently coerced
**What goes wrong:** caller passes float64 data to an f32 model; a permissive binding copies/casts, violating D-03's "strict or error."
**How to avoid:** typed `PyReadonlyArray2<f32>` *rejects* float64 with a `TypeError` automatically; for the untyped-dispatch variant, read `input_type` and raise `TreeliteError("dtype mismatch: model expects float32, got float64; convert with arr.astype(...)")`. **Do not auto-cast** (D-03). `[CITED: CONTEXT D-03]`
**Warning signs:** predictions off by quantization error (~1e-7) that shouldn't exist for an exact-dtype path.

### Pitfall 5: maturin builds against the wrong interpreter in a worktree
**What goes wrong:** the venv/pyproject are untracked and live only on the main tree (project memory); building in a git worktree finds no interpreter or the system Python.
**How to avoid:** run `maturin develop`/`build` and the A/B pytest on the **main tree** under `uv run` (mirrors the golden-capture constraint). `[CITED: MEMORY.md python-venv-uv-run, worktree-isolation-unsafe-here]`
**Warning signs:** `Couldn't find a Python interpreter`; import of upstream `treelite` fails in the build venv.

### Pitfall 6: abi3 + a non-abi3 transitive dep
**What goes wrong:** enabling `abi3-py310` but a transitive crate links a Python-version-specific symbol breaks the stable-ABI guarantee.
**How to avoid:** only `pyo3` touches Python; the treelite crates are pure Rust (no Python linkage). `treelite-cubecl` GPU features link HIP/CUDA/wgpu (native, not Python) — orthogonal to abi3. Keep `extension-module` + `abi3-py310` on pyo3 only. `[ASSUMED — low risk; treelite crates verified Python-free this session]`
**Warning signs:** wheel imports on the build interpreter but not on a different 3.1x.

## Code Examples

### `#[pymodule]` with submodules (D-01 layout, D-02 name)
```rust
// Source: pyo3 0.28 #[pymodule] (module name = lib name = _treelite_rs)
use pyo3::prelude::*;

#[pymodule]
fn _treelite_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TreeliteError", m.py().get_type::<TreeliteError>())?;
    m.add_class::<Model>()?;

    let frontend = PyModule::new(m.py(), "frontend")?;
    frontend.add_function(wrap_pyfunction!(load_xgboost_json_str, &frontend)?)?;
    frontend.add_function(wrap_pyfunction!(load_xgboost_ubjson_bytes, &frontend)?)?;
    frontend.add_function(wrap_pyfunction!(load_lightgbm_str, &frontend)?)?;
    m.add_submodule(&frontend)?;

    let gtil = PyModule::new(m.py(), "gtil")?;
    gtil.add_function(wrap_pyfunction!(predict_f32, &gtil)?)?;
    gtil.add_function(wrap_pyfunction!(predict_f64, &gtil)?)?;
    m.add_submodule(&gtil)?;

    let sklearn = PyModule::new(m.py(), "sklearn")?;
    sklearn.add_function(wrap_pyfunction!(load_random_forest_regressor, &sklearn)?)?;
    // ... rf_classifier, extra_trees_*, gradient_boosting_*, isolation_forest, hist_gb_*
    m.add_submodule(&sklearn)?;
    Ok(())
}
```
*(Submodules added via `add_submodule` are not auto-registered in `sys.modules`; the `.py` package shims that `from _treelite_rs.gtil import predict_f32` handle import ergonomics — another reason to use the `.py`-shim layout.)*

### `#[pyclass] Model` serialize/deserialize/dump (PY-03)
```rust
use pyo3::types::PyBytes;
#[pyclass]
pub struct Model { pub inner: treelite_core::Model }

#[pymethods]
impl Model {
    fn serialize_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = treelite_core::serialize_to_buffer(&mut self.inner);   // Vec<u8>
        Ok(PyBytes::new(py, &bytes))
    }
    #[staticmethod]
    fn deserialize_bytes(buf: &[u8]) -> PyResult<Model> {
        Ok(Model { inner: treelite_core::deserialize(buf).map_err(PyErr::from)? })
    }
    fn dump_as_json(&mut self, pretty_print: bool) -> PyResult<String> {
        Ok(treelite_core::serialize::json::dump_as_json_string(&mut self.inner))
        // pretty_print plumbing per upstream dump_as_json(pretty_print=True)
    }
    #[getter] fn num_tree(&self) -> u64 { self.inner.num_tree() }
    #[getter] fn num_feature(&self) -> i32 { self.inner.num_feature }
    #[getter] fn input_type(&self) -> &'static str {
        match self.inner.variant { ModelVariant::F32(_) => "float32", ModelVariant::F64(_) => "float64" }
    }
}
```

### sklearn `import_model` shim (PY-04, port of `importer.py`)
```python
# treelite_rs/sklearn/__init__.py  (Python-side estimator -> arrays, then call compiled loader)
import numpy as np
from .. import _treelite_rs

def import_model(sklearn_model):
    cls = sklearn_model.__class__.__name__
    if cls in ("RandomForestRegressor", "ExtraTreesRegressor"):
        node_count, cl, cr, feat, thr, val, nns, wnns, imp = _extract_forest(sklearn_model)
        h = _treelite_rs.sklearn.load_random_forest_regressor(
            sklearn_model.n_estimators, sklearn_model.n_features_in_,
            sklearn_model.n_outputs_, node_count, cl, cr, feat, thr, val, nns, wnns, imp)
        return h
    # ... classifiers, GB (learning-rate leaf shrink), IsolationForest (depths + ratio_c),
    #     HistGB (packed nodes + features_map/categories_map) — mirror importer.py branch-for-branch
```
`_extract_forest` is the `ArrayOfArrays` logic from `importer.py:174-229`, producing `&[&[i64/f64]]`-shaped lists the Rust array loaders consume (`treelite-sklearn/src/lib.rs:65-114` signatures). Each inner array crosses as a contiguous numpy buffer borrowed zero-copy by the Rust loader.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Python::allow_threads` | `Python::detach` / `Python::attach` | pyo3 0.26→0.28 | GIL-release call renamed; old name removed |
| `Python<'py>` GIL-token-first ergonomics | `Bound<'py, T>` smart pointers | pyo3 0.21+ | use `Bound<'py, PyArray1<T>>` return types, `&Bound<PyModule>` in `#[pymodule]` |
| `IntoPy`/`ToPyObject` | `IntoPyObject` trait | pyo3 0.23+ | conversions; mostly transparent via `?`/return types |
| manual abi3 feature flags per Python ver | `abi3-py310` single floor | stable | one wheel CPython ≥3.10 (D-09) |

**Deprecated/outdated:**
- `Python::allow_threads` — use `Python::detach`.
- GIL-Refs API (`&PyAny` without `Bound`) — fully removed in 0.28; use the Bound API.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A2 | abi3-py310 + GPU-feature native links (HIP/CUDA/wgpu) coexist cleanly in one wheel. | Pitfall 6 | Low — native libs are orthogonal to Python ABI; verify with a `--features rocm` build on ROCm hardware. |
| A3 | `dump_as_json_string` honors a `pretty_print` toggle (upstream `dump_as_json(pretty_print=True)`); current signature is `dump_as_json_string(&mut Model) -> String`. | Code Examples (Model) | Low — may need a `pretty` param added or post-format; cosmetic, not 1e-5. |
| A4 | Field-accessor `get_field` can be v1-implemented by copy (`to_pyarray`) rather than zero-copy, deferring the frame-lifetime-into-Python hazard to Phase 9. | Pattern 6 | Low — D-01 lists accessors but MEM-04 only mandates zero-copy for the *predict* path; copy is faithful, just not zero-copy for inspection. |

## Open Questions

> The legacy-loader name and scipy availability questions were RESOLVED this session — see resolutions below; no longer open.

**RESOLVED — legacy XGBoost binary loader:** `treelite_xgboost::load_xgboost_legacy(bytes: &[u8]) -> Result<Model, XgbError>` is publicly re-exported (`crates/treelite-xgboost/src/lib.rs:25`, defined `legacy.rs:451`). So `frontend.load_xgboost_model_legacy_binary` maps to it: the `.py` shim reads the file as bytes and passes them in. `detect_xgboost_format` is also exported (`lib.rs:24`) for the `format_choice="inspect"` path.

**RESOLVED — scipy availability:** scipy 1.17.1 is installed in the venv (`uv run python -c "import scipy"` → 1.17.1). The sparse-CSR predict A/B cells are in play; the `.py` gtil shim unpacks `csr_matrix.data/indices/indptr` (mirroring upstream `gtil.py:178-186`) into `treelite_gtil::SparseCsr` via `predict_sparse::<O>`.

1. **Whether `concatenate` is reachable from `treelite-py` without a dependency cycle.**
   - What we know: `treelite_builder::concatenate(&[&Model]) -> Result<Option<Model>, BuilderError>` exists (`concat.rs:64`). `Model.concatenate` (D-01) maps to it.
   - What's unclear: whether `treelite-py` should depend on `treelite-builder` (it must, for concatenate). No cycle expected (builder depends on core, not on py).
   - Recommendation: add `treelite-builder` as a path dep of `treelite-py`; map `None` (empty input) to a `TreeliteError`.

3. **abi3 + cubecl GPU-feature wheel buildability on the dev box.**
   - What we know: ROCm is the only hardware-validated backend; CUDA/wgpu are build-only.
   - What's unclear: whether a `--features rocm` abi3 wheel links and imports on the ROCm box (vs. just compiles).
   - Recommendation: a hardware-gated checkpoint task (like 07-04) builds + imports the `rocm` wheel and runs one predict on-device; CUDA/wgpu wheels are "build-only, not run — no device."

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (edition 2024) | compile `treelite-py` | ✓ | rustc 1.95.0 | — |
| pyo3 0.28.3 | binding | ✓ (crates.io) | 0.28.3 | — |
| numpy crate 0.28.0 | zero-copy numpy | ✓ (crates.io) | 0.28.0 | — |
| maturin | wheel build | ✓ (PyPI) | 1.13.3 | install via `uv pip install maturin` |
| Python interpreter | abi3 target + A/B test | ✓ | 3.13.13 (abi3-py310 floor) | — |
| numpy (runtime) | A/B test, predict I/O | ✓ | 2.4.6 | — |
| scipy | CSR predict path (`gtil.predict` sparse) | ? not verified | — | only needed for sparse A/B cells; dense path needs none |
| upstream `treelite` | D-11 live A/B witness | ✓ | 4.7.0 | golden-vector fallback (planner discretion) |
| scikit-learn | sklearn `import_model` A/B + extraction | ✓ (root pyproject dep `scikit-learn>=1.9.0`) | per venv | — |
| ROCm GPU | `rocm` wheel hardware validation | ✓ (AMD dev box) | — | CUDA/wgpu = build-only "not run — no device" |

**Missing dependencies with no fallback:** none blocking. `maturin` must be installed into the venv (`uv pip install maturin`) before building.
**Missing dependencies with fallback:** `scipy` (verify before planning sparse A/B cells; dense predict needs no scipy).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework (Rust) | `cargo test` (workspace) — existing; `treelite-py` adds doc/unit tests where it can without an interpreter |
| Framework (Python) | `pytest` via `uv run pytest` on the main tree (golden-capture venv) |
| Config file | none yet for `treelite-py`; add `crates/treelite-py/pyproject.toml` + a `tests/python/` dir — see Wave 0 |
| Build-into-venv command | `cd crates/treelite-py && uv run maturin develop` (main tree) |
| Quick run command | `uv run pytest crates/treelite-py/tests/python -x -q` |
| Full suite command | `cargo test --workspace && uv run pytest crates/treelite-py/tests/python` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PY-01 | load XGB/LGB/sklearn from Python yields a Model with correct `num_tree`/`num_feature` | integration (pytest) | `uv run pytest .../test_frontend.py -x` | ❌ Wave 0 |
| PY-02 | predict over numpy == upstream within 1e-5 (dense, f32 & f64) | integration A/B (D-11) | `uv run pytest .../test_predict_ab.py -x` | ❌ Wave 0 |
| PY-02/MEM-04 | borrowed numpy consumed zero-copy (no copy) | unit (no-copy proof) | `uv run pytest .../test_zero_copy.py -x` | ❌ Wave 0 |
| PY-03 | serialize_bytes→deserialize_bytes round-trips; dump_as_json parses | integration | `uv run pytest .../test_serialize.py -x` | ❌ Wave 0 |
| PY-04 | `sklearn.import_model(fitted)` predicts == upstream within 1e-5 | integration A/B | `uv run pytest .../test_sklearn_ab.py -x` | ❌ Wave 0 |
| PY-05 | bad dtype / malformed model raise `TreeliteError`; no abort on forced panic | unit | `uv run pytest .../test_errors.py -x` | ❌ Wave 0 |
| PY-06 | wheel builds + imports as abi3 | smoke | `uv run maturin build && uv run python -c "import treelite_rs"` | ❌ Wave 0 |
| D-05/D-08 | `predict(backend='rocm')` works on device / `predict(backend='unavailable')` raises | integration (hw-gated) | `uv run pytest .../test_backend.py -x` | ❌ Wave 0 |

**Zero-copy (no-copy) proof for MEM-04 (D-03):** assert the numpy input array's `__array_interface__['data'][0]` (its buffer pointer) is unchanged across the predict call (the array object isn't copied), and Rust-side a debug-only `as_ptr()` identity check (mirroring `tests/serialize_pybuffer.rs`'s `.as_ptr()` equality proof for `Frame`) confirms `as_slice().as_ptr()` equals the numpy data pointer. A complementary test: pass a non-contiguous (`arr[:, ::2]`) or wrong-dtype array and assert `TreeliteError` is raised (D-03 strict).

**Live A/B equivalence (D-11) — the headline test:**
```python
import numpy as np, treelite, treelite_rs
m_up = treelite.frontend.load_xgboost_model("fixtures/.../model.json")
m_rs = treelite_rs.frontend.load_xgboost_model("fixtures/.../model.json")
X = np.ascontiguousarray(rng.standard_normal((512, m_up.num_feature)).astype(np.float32))
np.testing.assert_allclose(treelite.gtil.predict(m_up, X),
                           treelite_rs.gtil.predict(m_rs, X), atol=1e-5, rtol=0)
```
Run for each model class already captured in `fixtures/` (XGB json/ubjson/legacy, LightGBM numerical/categorical, sklearn RF/ET/GB/IsolationForest/HistGB), both presets, dense + (if scipy) sparse, and all three predict kinds (`predict`/`predict_leaf`/`predict_per_tree`).

### Sampling Rate
- **Per task commit:** `cargo test -p treelite-py` (Rust-side compile/unit) + the one pytest file the task touches.
- **Per wave merge:** `cargo test --workspace && uv run pytest crates/treelite-py/tests/python`.
- **Phase gate:** full Rust workspace green + full pytest green (live A/B within 1e-5 across all fixtures) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/treelite-py/Cargo.toml` + `pyproject.toml` (maturin config, abi3-py310, per-backend features) — and register `crates/treelite-py` in root `Cargo.toml` members.
- [ ] `crates/treelite-py/tests/python/conftest.py` — shared fixtures (fixture paths, rng seed, `treelite`+`treelite_rs` import, skip-if-no-upstream).
- [ ] `test_frontend.py` (PY-01), `test_predict_ab.py` (PY-02), `test_zero_copy.py` (MEM-04/D-03), `test_serialize.py` (PY-03), `test_sklearn_ab.py` (PY-04), `test_errors.py` (PY-05), `test_backend.py` (D-05/D-08, hw-gated).
- [ ] `maturin develop` into the existing `uv` venv (main tree) — the prerequisite for any pytest.
- [ ] Confirm `scipy` availability or descope sparse A/B cells.

## Security Domain

> `security_enforcement` not found in config; treating as enabled. This is an FFI boundary that ingests untrusted numpy buffers and model files, so input-validation and memory-safety controls are the relevant categories.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | local library, no auth surface |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | strict dtype/contiguity/shape gate on numpy input (D-03); typed `PyReadonlyArray2<T>` rejects wrong dtype; `treelite_gtil`/`treelite_cubecl` already bounds-check `num_feature`/CSR (`lib.rs:898-926`, `error.rs`) → typed errors, never OOB |
| V6 Cryptography | no | no crypto in scope |
| V14 Config / Memory safety | **yes** | no panic crosses FFI (D-07, pyo3 catch_unwind); zero-copy borrow lifetime tied to GIL-scoped `PyReadonlyArray` guard; serialize/deserialize is bounds-checked (`deserialize` returns `Result`, 02-03) |

### Known Threat Patterns for the PyO3/numpy boundary
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed model bytes → OOB read in deserialize | Tampering / DoS | `treelite_core::deserialize` is bounds-checked → `SerializeError` → `TreeliteError` (02-03 "bounds-checked deserialize"); no `transmute` |
| Wrong-dtype / non-contiguous numpy buffer → OOB slice | Tampering | typed `PyReadonlyArray2<T>` + `as_slice()` contiguity check; reject (D-03), never coerce |
| Mismatched `num_feature` vs data width → OOB row slice | DoS | `predict::<O>` validates `data.len() >= num_row*num_feature` (saturating) → `InvalidInputShape` (`lib.rs:918-926`) |
| Rust panic unwinding into CPython → UB/abort | DoS | pyo3 auto `catch_unwind` → `PanicException`; `guard` remap → `TreeliteError` (D-07) |
| `nthread` huge value → resource exhaustion | DoS | scalar engine records-not-allocates `nthread` (`config.rs:39-43`); no thread-count-driven allocation |
| Device-absent GPU backend → silent wrong result | Spoofing | `DeviceUnavailable` is a typed error → `TreeliteError`, never silent CPU fallback (D-08) |

## Sources

### Primary (HIGH confidence)
- `cargo search pyo3 / numpy` (crates.io) — pyo3 0.28.3, numpy 0.28.0 — verified this session.
- `pip index versions maturin` (PyPI) — maturin 1.13.3 — verified this session.
- `uv run python` — Python 3.13.13, numpy 2.4.6, treelite 4.7.0 — verified this session.
- https://docs.rs/pyo3/0.28.3/pyo3/buffer/struct.PyBuffer.html — `PyBuffer::as_slice`, `is_c_contiguous`, `format`.
- https://docs.rs/numpy/0.28.0/numpy/ — `PyReadonlyArray1/2`, `as_slice`, `IntoPyArray::into_pyarray`, `Element`, Bound API.
- https://www.maturin.rs/config.html — `[tool.maturin]` bindings/features/manifest-path/python-source/include.
- https://pyo3.rs/v0.28.3/parallelism.html — `Python::detach` (GIL release; renamed from `allow_threads`).
- https://pyo3.rs/v0.28.3/exception.html — `create_exception!`, `#[pymodule_export]`.
- https://pyo3.rs/v0.28.3/function/error-handling.html — `From<MyError> for PyErr`.
- In-repo files read this session: `crates/treelite-core/src/serialize/pybuffer.rs`, `.../model.rs`, `crates/treelite-gtil/src/lib.rs` + `config.rs`, `crates/treelite-harness/src/lib.rs`, `crates/treelite-cubecl/src/error.rs`, `crates/treelite-sklearn/src/lib.rs`, root `Cargo.toml`/`pyproject.toml`, upstream `treelite-mainline/python/treelite/{model,frontend,core,util}.py`, `gtil/gtil.py`, `gtil/__init__.py`, `sklearn/importer.py`, `__init__.py`.

### Secondary (MEDIUM confidence)
- https://github.com/PyO3/pyo3/issues/492, /pull/797, /issues/2880 — pyo3 catch_unwind → `PanicException` at FFI boundary (cross-checked against pyo3 docs).
- WebSearch (pyo3 0.28 panic behavior) — confirms automatic boundary guard + `PanicException`.

### Tertiary (LOW confidence)
- `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_*.md` — present (verified exist); relevant to MEM-04 zero-copy patterns but not directly cited (predict zero-copy is covered by rust-numpy idioms).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all three crate versions verified against the correct registries this session; numpy 0.28/pyo3 0.28 pairing is the canonical co-release; slopcheck clean.
- Architecture / patterns: HIGH — grounded in the actual in-repo seam files (read directly) and pyo3 0.28.3 / numpy 0.28 docs; the predict/serialize/error/backend seams all already exist and are typed.
- GIL/threading/zero-copy (research-flag gaps): HIGH — `Python::detach`, `PyReadonlyArray::as_slice`, `into_pyarray` all verified against 0.28 docs.
- Pitfalls: HIGH for the API-shift ones (detach rename, input_type staging — both verified in source/docs); MEDIUM for abi3+GPU coexistence (A2, build-time verifiable).
- Open Questions: the legacy-loader name (Q1) and scipy availability are the only real gaps; both are quick planner greps, neither blocks the architecture.

**Research date:** 2026-06-11
**Valid until:** ~2026-07-11 (30 days; pyo3/numpy/maturin are stable but fast-moving — re-verify versions if the phase starts later).

## RESEARCH COMPLETE

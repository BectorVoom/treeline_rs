# Phase 8: PyO3 Python Binding - Pattern Map

**Mapped:** 2026-06-11
**Files analyzed:** 22 new (12 Rust/config/Python-package + ~10 pytest/test) + 1 modified
**Analogs found:** 18 / 22 (with strong in-repo analogs) — 4 are pyo3/maturin-idiom files with no Rust analog (planner uses RESEARCH.md patterns)

> **Scope note:** This is a **greenfield additive phase** — a NEW `crates/treelite-py` workspace member (cdylib `_treelite_rs`) + a NEW pure-Python `treelite_rs/` package + NEW pytest suite. There is **no existing pyo3 / maturin / pyproject** anywhere under `crates/` (verified: `find crates -name pyproject.toml` → none). The closest in-repo analogs are the **other workspace crates** (their `Cargo.toml` / `lib.rs` / `error.rs` conventions) and the **upstream Python package** at `treelite-mainline/python/treelite/` (the API shape to mirror 1:1, D-01). The pyo3/numpy/maturin-specific mechanics have NO Rust analog and must come from RESEARCH.md §"Architecture Patterns" / §"Code Examples".

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `Cargo.toml` (root) — add member | config | — | `Cargo.toml` (existing `members` list) | exact (1-line append) |
| `crates/treelite-py/Cargo.toml` | config | — | `crates/treelite-harness/Cargo.toml` (per-backend features) + `crates/treelite-cubecl/Cargo.toml` | role-match (no `[lib] cdylib`/pyo3 analog) |
| `crates/treelite-py/pyproject.toml` | config | — | `treelite-mainline/python/pyproject.toml` (shape only) | none in-repo (maturin idiom — RESEARCH §Packaging) |
| `crates/treelite-py/src/lib.rs` | route (`#[pymodule]`) | request-response | `crates/treelite-harness/src/lib.rs` (registration seam) + RESEARCH §"#[pymodule] with submodules" | role-match |
| `crates/treelite-py/src/error.rs` | utility (error map) | transform | `crates/treelite-gtil/src/error.rs` (thiserror enum) + RESEARCH Pattern 4 | role-match (target is `create_exception!`, not thiserror) |
| `crates/treelite-py/src/model.rs` | model (`#[pyclass]`) | CRUD / request-response | `treelite-mainline/python/treelite/model.py` (API shape) + RESEARCH §"#[pyclass] Model" | role-match |
| `crates/treelite-py/src/frontend.rs` | controller (`#[pyfunction]` loaders) | request-response | `crates/treelite-xgboost/src/lib.rs` + `treelite-lightgbm/src/lib.rs` re-export surface | role-match |
| `crates/treelite-py/src/gtil.rs` | controller (`#[pyfunction]` predict) | streaming / zero-copy buffer-I/O | `crates/treelite-harness/src/lib.rs:99-128` (four-fn-pointer dtype seam) + RESEARCH Patterns 1-3,5 | role-match |
| `crates/treelite-py/src/sklearn.rs` | controller (`#[pyfunction]` array loaders) | request-response | `crates/treelite-sklearn/src/lib.rs` (array signatures) | exact (thin wrap of existing sigs) |
| `crates/treelite-py/python/treelite_rs/__init__.py` | provider (re-export) | — | `treelite-mainline/python/treelite/__init__.py` | exact (mirror) |
| `crates/treelite-py/python/treelite_rs/frontend.py` | controller (path→bytes shim) | file-I/O | `treelite-mainline/python/treelite/frontend.py` | exact (port) |
| `crates/treelite-py/python/treelite_rs/gtil/__init__.py` | controller (CSR unpack + reshape) | streaming | `treelite-mainline/python/treelite/gtil/{__init__,gtil}.py` | exact (port) |
| `crates/treelite-py/python/treelite_rs/sklearn/__init__.py` | controller (estimator→arrays) | transform | `treelite-mainline/python/treelite/sklearn/importer.py` | exact (port `ArrayOfArrays`) |
| `crates/treelite-py/python/treelite_rs/py.typed` | config | — | `treelite-mainline/python/treelite/py.typed` (empty marker) | exact (copy) |
| `*.pyi` stubs (frontend/model/gtil/sklearn) | config (PEP 561) | — | upstream `.py` signatures (no upstream `.pyi` — hand-write) | partial (RESEARCH D-10) |
| `crates/treelite-py/tests/python/conftest.py` | test (fixtures) | — | `xgboost-master/tests/python/conftest.py` (pytest idiom) + `fixtures/` layout | partial |
| `tests/python/test_predict_ab.py` (+ test_frontend/serialize/sklearn_ab/zero_copy/errors/backend) | test (A/B) | — | `crates/treelite-harness/tests/golden_v5.rs` (1e-5 / golden-equivalence ethos) + RESEARCH §Validation | role-match |

---

## Pattern Assignments

### `crates/treelite-py/Cargo.toml` (config)

**Analogs:** `crates/treelite-harness/Cargo.toml` (per-backend feature gating), `crates/treelite-cubecl/Cargo.toml` (feature→cubecl forwarding).

**Package-header pattern to copy** (every crate uses `*.workspace = true`):
```toml
[package]
name = "treelite-py"
version.workspace = true
edition.workspace = true
license.workspace = true
```

**Per-backend feature pattern — copy from `treelite-cubecl/Cargo.toml` lines 12-16 / `treelite-harness/Cargo.toml`:**
```toml
[features]
default = ["cpu"]
cpu  = []
rocm = ["treelite-cubecl/rocm"]   # mirrors treelite-cubecl: rocm = ["cubecl/rocm"]
cuda = ["treelite-cubecl/cuda"]
wgpu = ["treelite-cubecl/wgpu"]
```
The harness `Cargo.toml` proves the **forwarding-feature** discipline: each GPU feature forwards to the matching `treelite-cubecl/<backend>` feature so the default build stays cpu-only and needs no GPU system libs. Copy that exact forwarding shape.

**Path-dep pattern — copy from `treelite-harness/Cargo.toml` `[dependencies]`:** all sibling crates use `{ path = "../<crate>" }`. Add `treelite-core`, `treelite-xgboost`, `treelite-lightgbm`, `treelite-sklearn`, `treelite-gtil`, `treelite-builder`, `treelite-cubecl` as path deps; `thiserror = { workspace = true }`.

**NEW (no analog — RESEARCH §"Installation" lines 106-129):** the `[lib] name = "_treelite_rs" crate-type = ["cdylib"]` block and `pyo3 = { version = "0.28.3", features = ["abi3-py310", "extension-module"] }` + `numpy = "0.28.0"` deps. These are pyo3 idiom; copy verbatim from RESEARCH.

---

### `crates/treelite-py/src/error.rs` (utility, transform)

**Analog:** `crates/treelite-gtil/src/error.rs` (the canonical thiserror enum + `#[error(...)]` doc-comment discipline; every variant documents the upstream path it replaces).

**What to copy from the analog:** the module-doc-comment style and the discipline that **every error carries a descriptive message** (the GtilError variants all have rich `#[error("...")]` strings). At the Python boundary D-06 collapses these to ONE exception, but the *messages* are what callers branch on — so preserve the descriptive `.to_string()` of each source enum.

**Source-error inventory to map (all confirmed exported):**
- `treelite_core::CoreError` (re-exported), `treelite_xgboost::XgbError` (`lib.rs:38`), `treelite_lightgbm::LgbError` (`lib.rs:30`), `treelite_sklearn::SklError` (`lib.rs:55`), `treelite_gtil::GtilError` (`lib.rs:17`), `treelite_cubecl::CubeclError` (`lib.rs:35`), `treelite_builder::BuilderError`.
- Note `GtilError::Core(#[from] treelite_core::CoreError)` (gtil `error.rs:174-175`) — the `#[error(transparent)]` + `#[from]` bridge pattern is already in the codebase; mirror it conceptually for the `From<E> for PyErr` macro.

**NEW (RESEARCH Pattern 4, lines 274-310):** `create_exception!(_treelite_rs, TreeliteError, PyException)` + the `err_to_treelite!` macro generating `impl From<$t> for PyErr` per crate enum, + the `guard()` panic-remap helper. The `guard` mirrors the in-repo `catch_unwind` discipline in `crates/treelite-cubecl/src/device.rs` (07-01: trap panic → typed error) — cite that as the in-repo precedent for the panic-trap shape.

---

### `crates/treelite-py/src/model.rs` (model, `#[pyclass]`, CRUD)

**Analog (API shape):** `treelite-mainline/python/treelite/model.py` — methods `serialize`/`serialize_bytes` (250), `deserialize`/`deserialize_bytes` (270/299), `dump_as_json` (171), `num_tree` (40), `num_feature` (49), `input_type` (58), `output_type` (67), `concatenate` (76); `HeaderAccessor` (433), `TreeAccessor` (503), `_TreelitePyBufferFrame` (331), `_PyBuffer` (342).

**Analog (Rust seam it wires into):** `crates/treelite-core/src/serialize/pybuffer.rs` — the `Frame<'a>` enum (lines 25-44) is the zero-copy borrowed image of upstream's `_TreelitePyBufferFrame{buf, format, itemsize, nitem}`. The frame variant → format-string/itemsize table is in RESEARCH Pattern 6 (lines 340-352).

**Core CRUD pattern (RESEARCH §"#[pyclass] Model" lines 456-481):** `#[pyclass] Model { inner: treelite_core::Model }` with `#[pymethods]` calling `treelite_core::serialize_to_buffer` / `deserialize` / `serialize::json::dump_as_json_string`. `serialize_bytes`/`deserialize_bytes` ride the **binary** serializer (Vec<u8> ↔ PyBytes) — they do NOT need the `Frame` layout (RESEARCH Pattern 6, line 337).

**CRITICAL pitfall — `input_type`/`output_type` (RESEARCH Pitfall 2, lines 397-401):** compute the dtype directly from `ModelVariant::F32 => "float32" / F64 => "float64"` (the variant is the source of truth), NOT from the staged `DType` field which reads `kInvalid` before serialization staging. The `ModelVariant` enum is in `crates/treelite-core/src/model.rs` (imported by `pybuffer.rs:16`).

**`concatenate` (D-01):** maps to `treelite_builder::concatenate(&[&Model]) -> Result<Option<Model>, BuilderError>` (`crates/treelite-builder/src/concat.rs:64`). Map `Ok(None)` (empty input) → `TreeliteError` (RESEARCH Open Q1, line 534).

**Field-accessor v1 simplification (RESEARCH A4 / Pattern 6, line 352):** implement `HeaderAccessor`/`TreeAccessor.get_field` by **copying** the column into a fresh numpy array (`to_pyarray`), deferring zero-copy-accessor lifetime hazards to Phase 9. MEM-04 zero-copy is mandated only for the *predict* path, not field inspection.

---

### `crates/treelite-py/src/frontend.rs` (controller, request-response)

**Analogs (the loader entry points being wrapped — all confirmed `pub`):**
- `treelite_xgboost::load_xgboost_legacy` (`lib.rs:25`), `detect_xgboost_format` (`lib.rs:24`); JSON/UBJSON loaders (in `json.rs`/`ubjson.rs`).
- `treelite_lightgbm::load_lightgbm` (the `load_lightgbm` converge-then-build path, `lib.rs`).

**API shape:** `treelite-mainline/python/treelite/frontend.py` — `load_xgboost_model_legacy_binary` (25), `load_xgboost_model` (60), `load_lightgbm_model` (147), `from_xgboost*` (178/228/275), `from_lightgbm` (314), `_detect_xgboost_format` (348). **File I/O stays Python-side** (`_normalize_path` at line 17) — Rust loaders take str/bytes (RESEARCH Architectural Responsibility Map, line 67).

**Wrapper pattern:** each `#[pyfunction]` is a thin `treelite_xgboost::load_*(bytes/str).map(|m| Model{inner: m}).map_err(PyErr::from)`. RESEARCH §"#[pymodule] with submodules" (lines 435-439) shows the submodule registration (`load_xgboost_json_str` / `load_xgboost_ubjson_bytes` / `load_lightgbm_str`).

---

### `crates/treelite-py/src/gtil.rs` (controller, streaming / zero-copy buffer-I/O)

**Analog (the dtype-dispatch seam):** `crates/treelite-harness/src/lib.rs:99-128` — the **four-fn-pointer seam** (`DensePredictF32Fn` / `DensePredictF64Fn` / `SparsePredictF32Fn` / `SparsePredictF64Fn`). RESEARCH explicitly recommends mirroring this (line 250): two typed entry points (`predict_f32`, `predict_f64`) keep each path monomorphized with NO f32→f64 pre-cast. The harness comment (`lib.rs:120-128`) is the load-bearing rationale: "an f32-input fixture MUST flow through the f32 entry point with NO f32→f64 pre-cast."

**Analog (the predict surface it dispatches into):**
- `treelite_gtil::predict::<O>` (`crates/treelite-gtil/src/lib.rs:892`), `predict_sparse::<O>` (`:949`); `Config`/`PredictKind` (`config.rs`, re-exported `lib.rs:16`); `output_shape`/`Shape` (`lib.rs:18`); `SparseCsr` (`lib.rs:15`).
- `treelite_cubecl::predict::<R, F>` (`cubecl/src/lib.rs:306`), `predict_cpu::<F>` (`:347`), `predict_cpu_sparse` (`:360`), `model_routes_to_scalar_fallback` (`:261`) — the backend dispatch the `backend=` kwarg drives.
- `treelite_cubecl::CubeclError::DeviceUnavailable { backend }` (`cubecl/src/error.rs:90`) — the typed device-absent skip → `TreeliteError` (D-08); NEVER silent CPU fallback.

**Core zero-copy pattern (RESEARCH Pattern 1, lines 219-249):** `PyReadonlyArray2<'py, O>` typed param (rejects wrong dtype as TypeError before body) → `.as_slice()` (zero-copy alias, MEM-04) with `AsSliceError → TreeliteError("...call np.ascontiguousarray...")` (D-03 strict gate).

**GIL release (RESEARCH Pattern 3, lines 263-272):** `py.detach(|| predict::<O>(&model.inner, slice, num_row, &cfg))` — `detach` (renamed from `allow_threads` in pyo3 0.28, RESEARCH Pitfall 1). Closure is `Send` + touches no Python objects.

**Zero-copy return (RESEARCH Pattern 2, lines 252-261):** `out.into_pyarray(py)` (moves the Vec; `to_pyarray` would COPY). Returns a **1-D flat** buffer; reshape to upstream N-D `(num_row, num_target, max_class)` happens in the **`.py` gtil shim** via `output_shape`/`Shape` (RESEARCH Pitfall 3).

**Backend kwarg (RESEARCH Pattern 5, lines 312-333):** additive `backend="cpu"` kwarg → `match backend { "cpu" => ..., #[cfg(feature="rocm")] "rocm" => ..., other => Err(TreeliteError::...) }`. Mirror the harness `Backend` enum `#[cfg(feature=…)]` gating (`harness/src/lib.rs:55-94`).

---

### `crates/treelite-py/src/sklearn.rs` (controller, request-response)

**Analog (exact — thin wrap of existing array signatures):** `crates/treelite-sklearn/src/lib.rs` — `load_random_forest_{regressor,classifier}`, `load_extra_trees_{regressor,classifier}`, `load_gradient_boosting_{regressor,classifier}`, `load_isolation_forest`, `load_hist_gradient_boosting_{regressor,classifier}` (all `pub`, re-exported `lib.rs:55-66`). Signatures already take `&[&[i64]]` / `&[&[f64]]` array-of-arrays (Phase-4 D-01 — `lib.rs` doc lines 5-12 confirm "so the Phase-8 PyO3 layer can hand zero-copy numpy buffers straight through"). The pyo3 wrappers borrow numpy arrays and pass slices straight in — best match quality in the phase.

---

### `crates/treelite-py/python/treelite_rs/sklearn/__init__.py` (controller, transform)

**Analog (exact port):** `treelite-mainline/python/treelite/sklearn/importer.py` — `import_model` (62), `ArrayOfArrays` (15: `add`/`as_c_array`), `_import_hist_gradient_boosting` (355), the per-estimator branches (RF/ET at 175-198, classifiers, GB with learning-rate leaf shrink, IsolationForest with depths + `ratio_c`, HistGB with packed nodes + `features_map`/`categories_map`). Port branch-for-branch; the estimator stays Python (`tree_.children_left` etc.), only numpy arrays cross into `_treelite_rs.sklearn.load_*`. RESEARCH §"sklearn import_model shim" (lines 483-500) gives the shim skeleton.

---

### `crates/treelite-py/python/treelite_rs/gtil/__init__.py` (controller, streaming)

**Analog (exact port):** `treelite-mainline/python/treelite/gtil/{__init__.py,gtil.py}` — `predict` (gtil.py:54), `predict_leaf` (87), `predict_per_tree` (123); CSR unpack via `csr_matrix.data/indices/indptr` (gtil.py:178-186 per RESEARCH "Don't Hand-Roll"). This shim: validates dtype/contiguity Python-side feedback, calls the typed `_treelite_rs.gtil.predict_f32`/`_f64`, then **reshapes** the flat output to upstream N-D (RESEARCH Pitfall 3). The `backend=` kwarg (D-05) is additive over the upstream signature.

---

### `tests/python/test_predict_ab.py` + sibling pytest files (test, A/B)

**Analog (ethos, not code):** `crates/treelite-harness/tests/golden_v5.rs` — the in-repo "1e-5 / byte-fidelity is THE gate" discipline. The Python A/B suite reproduces the same 1e-5 invariant at the Python surface against a **live** upstream `treelite` import (enabled by the `treelite_rs` name, D-11).

**A/B headline pattern (RESEARCH §Validation lines 586-594):**
```python
import numpy as np, treelite, treelite_rs
m_up = treelite.frontend.load_xgboost_model("fixtures/.../model.json")
m_rs = treelite_rs.frontend.load_xgboost_model("fixtures/.../model.json")
X = np.ascontiguousarray(rng.standard_normal((512, m_up.num_feature)).astype(np.float32))
np.testing.assert_allclose(treelite.gtil.predict(m_up, X),
                           treelite_rs.gtil.predict(m_rs, X), atol=1e-5, rtol=0)
```

**Fixtures available (verified `fixtures/`):** `binary_logistic.model.json`, `xgb_3format.{json,ubj,model}`, `lightgbm_{numerical,categorical}.txt`, `sklearn_{rf,gb,iforest,histgb_*}.golden.json`, `golden_v5.bin`. The A/B suite iterates these across both presets, dense + (scipy) sparse, all three predict kinds.

**Zero-copy proof (MEM-04, RESEARCH line 583):** assert the numpy input's `__array_interface__['data'][0]` is unchanged across predict; Rust-side a debug `as_slice().as_ptr()` identity check **mirroring `crates/treelite-core/tests/serialize_pybuffer.rs:80-90`** (the `.as_ptr()` equality proof that `Frame` borrows its `TreeBuf` column with no copy). A complementary test passes `arr[:, ::2]` (non-contiguous) / wrong-dtype and asserts `TreeliteError` (D-03 strict).

**conftest pattern:** shared fixture paths, rng seed, side-by-side `treelite`+`treelite_rs` import with skip-if-no-upstream. Use `xgboost-master/tests/python/conftest.py` as a pytest-idiom reference (the only in-repo pytest analogs live under `xgboost-master/` and `.venv/`).

---

## Shared Patterns

### Error translation → single `TreeliteError` (D-06)
**Source:** `crates/treelite-gtil/src/error.rs` (thiserror discipline) + RESEARCH Pattern 4.
**Apply to:** `src/error.rs`, and every `#[pyfunction]`/`#[pymethods]` in `model.rs`/`frontend.rs`/`gtil.rs`/`sklearn.rs`.
Every Rust crate error enum (`CoreError`/`XgbError`/`LgbError`/`SklError`/`GtilError`/`CubeclError`/`BuilderError`) gets `impl From<$t> for PyErr { TreeliteError::new_err(e.to_string()) }`. The descriptive `#[error("...")]` messages (already rich in `gtil/error.rs`) are preserved as the exception message — callers branch on text (D-06).

### Panic trap → no abort crosses FFI (D-07)
**Source:** in-repo precedent `crates/treelite-cubecl/src/device.rs` (07-01 `catch_unwind` → `DeviceUnavailable`/`ClientInit`); RESEARCH Pattern 4 `guard()`.
**Apply to:** fallible entry points wanting `PanicException → TreeliteError` parity. pyo3 0.28 already auto-`catch_unwind`s every boundary → `PanicException`; the `guard` only remaps it for D-06 message parity. Note `CubeclError::DeviceUnavailable` (`cubecl/src/error.rs:90`) is already a *typed Result*, not a panic — it flows through the `From` impl, no special handling (D-08).

### Per-backend feature gating (D-04/D-05)
**Source:** `crates/treelite-cubecl/Cargo.toml:12-16`, `crates/treelite-harness/Cargo.toml` features, `harness/src/lib.rs:55-94` `#[cfg(feature=…)]`-gated `Backend` variants.
**Apply to:** `treelite-py/Cargo.toml` `[features]` and the `match backend` arms in `gtil.rs`. Default `cpu`-only; `rocm`/`cuda`/`wgpu` forward to `treelite-cubecl/<backend>`; an un-built backend hits the `other =>` arm → `TreeliteError` (D-08).

### Zero-copy borrow proof (MEM-04)
**Source:** `crates/treelite-core/tests/serialize_pybuffer.rs:80-90` (`.as_ptr()` equality proves `Frame` aliases its column).
**Apply to:** `test_zero_copy.py` and any Rust-side debug assertion — the predict-input `as_slice().as_ptr()` must equal the numpy data pointer.

### `*.workspace = true` package headers
**Source:** every `crates/*/Cargo.toml` (`version.workspace`/`edition.workspace`/`license.workspace`).
**Apply to:** `treelite-py/Cargo.toml` package header.

---

## No Analog Found

Files whose core mechanics have no in-repo precedent — the planner must use RESEARCH.md patterns (all are pyo3/numpy/maturin idioms, NOT novel compute):

| File | Role | Data Flow | Reason | Use Instead |
|------|------|-----------|--------|-------------|
| `crates/treelite-py/pyproject.toml` | config | — | No maturin config exists anywhere in-repo (greenfield) | RESEARCH §Packaging + `treelite-mainline/python/pyproject.toml` for shape only |
| `src/lib.rs` `#[pymodule]` body | route | request-response | No pyo3 module in-repo; submodule registration is pyo3-specific | RESEARCH §"#[pymodule] with submodules" (lines 426-453) |
| `*.pyi` stubs | config (PEP 561) | — | Upstream ships `py.typed` but NO `.pyi` (verified: `find … -name "*.pyi"` → none); hand-write from upstream `.py` signatures | RESEARCH D-10 + upstream `.py` signatures |
| `[lib] cdylib` + pyo3/numpy deps in `Cargo.toml` | config | — | No cdylib crate in workspace (all are libs) | RESEARCH §Installation (lines 106-129) |

---

## Metadata

**Analog search scope:** `crates/` (all 8 workspace members — Cargo.toml/lib.rs/error.rs), `crates/treelite-core/src/serialize/pybuffer.rs`, `crates/treelite-core/tests/serialize_pybuffer.rs`, `crates/treelite-harness/{src/lib.rs,tests/}`, `crates/treelite-cubecl/{src/error.rs,src/lib.rs}`, `treelite-mainline/python/treelite/` (`__init__`/`frontend`/`model`/`core`/`gtil`/`sklearn`), root `Cargo.toml`/`pyproject.toml`, `fixtures/`, `tests/`.
**Files scanned:** ~30 (8 crate Cargo.toml, 6 crate lib.rs, 2 error.rs, pybuffer.rs + test, harness lib.rs, cubecl error/lib, 8 upstream Python files, root config, fixtures listing).
**Greenfield confirmation:** no `pyproject.toml`/`conftest.py`/`*.pyi`/cdylib exists under `crates/` — this phase introduces all pyo3/maturin/pytest scaffolding fresh.
**Pattern extraction date:** 2026-06-11

# Phase 8: PyO3 Python Binding - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Expose the already-proven Rust pipeline to Python as the **sole external binding** (no C-API — explicit project constraint), built with **PyO3 0.28** and packaged as an **abi3 wheel via maturin**. The Python surface is a **drop-in 1:1 mirror of upstream Treelite's public API** — `frontend.load_*` / `from_*`, `Model` (serialize / deserialize / `dump_as_json` / field accessors / `concatenate`), `gtil.predict` / `predict_leaf` / `predict_per_tree`, and `sklearn.import_model` — but imported under the module name **`treelite_rs`** so it coexists with the real `treelite` in one venv.

The binding wires Python calls into the existing Rust crates (`treelite-xgboost`, `treelite-lightgbm`, `treelite-sklearn`, `treelite-gtil`, `treelite-core` serialize, `treelite-cubecl`/`treelite-harness` backend seam). Predict consumes borrowed numpy buffers **zero-copy** (MEM-04) and returns numpy output; library `thiserror` errors translate to a single Python `TreeliteError`; **no panic crosses the FFI boundary**.

**Requirements covered:** PY-01 (load XGBoost/LightGBM/sklearn from Python), PY-02 (predict over numpy, zero-copy buffer I/O), PY-03 (serialize/deserialize/JSON-dump), PY-04 (`sklearn.import_model` marshals fitted estimators), PY-05 (`thiserror` → Python exceptions), PY-06 (abi3 wheel via maturin), MEM-04 (Python buffer-protocol borrowed buffers consumed zero-copy).

**In scope (HOW, not WHETHER):**
- The new `treelite-py` PyO3 crate + maturin packaging, importable as `treelite_rs`.
- The drop-in-mirrored Python API surface over the existing Rust crates.
- Zero-copy numpy input (strict) and numpy output for predict.
- Per-backend wheels with a `predict(..., backend=...)` selection kwarg.
- Single `TreeliteError` exception + panic-catching boundary.
- abi3-py310 wheel, `py.typed` + `.pyi` stubs, live A/B equivalence pytest vs upstream.

**Out of scope (this phase):**
- **Memory-efficiency hardening** (bytemuck input recast beyond the buffer-protocol borrow, smallvec/compact_str, custom global allocator) — Phase 9.
- **GPU hardware validation beyond ROCm** — inherited from Phase 7 (CUDA/wgpu build-supported, "not run — no device" locally).
- **Enforced/auto-routing CPU↔GPU crossover** — the crossover stays documented-only (Phase 7 D-09); the Python `backend=` kwarg is the explicit consumer, not an auto-router.
- **sklearn model *export*** (`export_model`) — only `import_model` is required (PY-04); export is not in the requirement set.
- **Legacy serialization formats** (v3.9/v4.0) — v5 only, project-wide.

</domain>

<decisions>
## Implementation Decisions

### API surface & module name (PY-01/PY-02/PY-03/PY-04)
- **D-01: Drop-in 1:1 mirror of upstream Treelite's public API.** Reproduce upstream's module layout and signatures so existing Treelite scripts run unchanged: `frontend.load_xgboost_model` / `load_xgboost_model_legacy_binary` / `load_lightgbm_model` / `from_xgboost*` / `from_lightgbm`; `Model` with `serialize` / `serialize_bytes` / `deserialize` / `deserialize_bytes` / `dump_as_json` / `num_tree` / `num_feature` / `input_type` / `output_type` / `concatenate` / field accessors (`HeaderAccessor` / `TreeAccessor`); `gtil.predict` / `predict_leaf` / `predict_per_tree`; `sklearn.import_model`; `core.TreeliteError`. The shape is the upstream package at `treelite-mainline/python/treelite/`.
- **D-02: Importable module name is `treelite_rs`** (NOT `treelite`). This lets the Rust binding coexist with the pip-installed upstream `treelite` in a single venv — required for the live A/B equivalence pytest (D-11). Drop-in usage is `import treelite_rs as treelite`.

### numpy predict ergonomics & backend exposure (PY-02/MEM-04)
- **D-03: Strict zero-copy input or error.** Predict requires a C-contiguous numpy array whose dtype exactly matches the model's `input_type` (f32/f64). On any mismatch (non-contiguous, wrong dtype), raise `TreeliteError` instructing the caller to convert. No hidden allocation — the zero-copy contract is honest, not best-effort. (Borrowed via the Python buffer protocol → consumed zero-copy per MEM-04.)
- **D-04: Backend is fixed at install time — one wheel per backend.** The user installs a backend-specific build (a `cpu` wheel, a `rocm` wheel, etc.) rather than flipping backends at runtime. This is the Python-facing expression of Phase 7's additive-Cargo-feature model (`rocm`/`cuda`/`wgpu` features, default cpu); maturin bakes the chosen feature into the wheel.
- **D-05: A GPU wheel bundles CPU too; `predict(..., backend=...)` kwarg picks within the installed wheel.** A gpu wheel (e.g. rocm) compiles in CPU **and** that GPU backend. `predict` takes an **additive `backend=` kwarg defaulting to `'cpu'`**, so the documented Phase-7 crossover works from Python (small inputs stay CPU, large go GPU — the caller's choice). The `cpu` wheel exposes only `'cpu'`. Requesting a backend the installed wheel wasn't built with raises a typed exception (D-08/D-09). The `backend=` kwarg is additive over the upstream signature — without it, behavior is upstream-identical.

### Error mapping & FFI safety (PY-05)
- **D-06: Single `TreeliteError(Exception)` — exact upstream parity.** Every Rust `thiserror` variant (loader / gtil / serialize / backend) surfaces as one `TreeliteError` with a descriptive message. Callers branch on message text, not type. Matches upstream's single-exception design; simplest drop-in.
- **D-07: Panics are caught and surfaced as `TreeliteError`, never an interpreter abort.** Wrap Rust entry points (`catch_unwind` / PyO3's mechanism) so an unexpected panic becomes a `TreeliteError` (or RuntimeError-style) carrying the panic message. Honors the project ethos literally: no panic crosses the boundary.
- **D-08: Device-absent / un-built-backend selection is a typed failure surfaced as `TreeliteError`.** The Rust `treelite-cubecl` `CubeclError::DeviceUnavailable` (Phase 7 D-05) and the "backend not compiled into this wheel" case both translate into `TreeliteError` at the Python boundary (consistent with D-06's single-exception rule) — never a silent CPU fallback (Phase 7 D-05/D-09 carry forward).

### Packaging & validation parity (PY-06)
- **D-09: abi3-py310 floor.** Single abi3 wheel covering CPython 3.10+, aligning with numpy 2.x's own runtime floor (the numpy dependency already implies ≥3.10). PyO3 0.28 abi3 build.
- **D-10: Ship PEP 561 type info — `py.typed` marker + hand-written `.pyi` stubs** for the public surface (`Model`, `frontend.*`, `gtil.*`, `sklearn.import_model`, `TreeliteError`). A pure-Rust pyo3 module exposes little to type-checkers without stubs; upstream ships stubs and users expect IDE/mypy support.
- **D-11: Live A/B equivalence pytest vs upstream.** A pytest imports **both** `treelite_rs` and the pip-installed `treelite`, loads the same fixtures, predicts the same inputs, and asserts within **1e-5** directly. This is the headline proof the Python surface is faithful end-to-end, enabled by the `treelite_rs` name (D-02). Runs in the existing `uv run`/golden-capture venv where upstream `treelite` is installed.

### Claude's Discretion (for research/planner)
- **sklearn marshalling location (PY-04):** port upstream's `treelite-mainline/python/treelite/sklearn/importer.py` as a **Python-side estimator→arrays extraction** that calls the existing `treelite-sklearn` crate's array entry points (the crate already takes array signatures, Phase-4 D-01). The estimator object stays in Python; only numpy arrays cross into Rust.
- **New crate name & layout** — a `treelite-py` (or similarly named) workspace member holding the PyO3 module + maturin config; whether the Python-side shims (`frontend`/`sklearn`/`gtil` packages) are thin `.py` wrappers over a single compiled `_treelite_rs` extension or direct pyo3 submodules.
- **GIL/threading pattern, buffer-protocol mechanics, numpy zero-copy *return*** — the ROADMAP research flag explicitly owns these (PyO3 0.28 buffer protocol; numpy zero-copy return; GIL/threading). Resolve in the research phase, not here.
- **Whether to also commit golden-vector assertions** alongside the live A/B pytest for CI runs that lack an upstream install (D-11 chose live A/B as the headline; a frozen-golden complement is a reasonable planner refinement, not mandated).
- **`nthread` / `pred_margin` predict config** — follows upstream `gtil.predict` signature 1:1 (D-01); no separate decision needed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Upstream Python API to mirror 1:1 (the D-01 source of truth)
- `treelite-mainline/python/treelite/__init__.py` — top-level exports (`frontend`, `gtil`, `model_builder`, `sklearn`, `TreeliteError`, `Model`).
- `treelite-mainline/python/treelite/frontend.py` — `load_xgboost_model` / `load_xgboost_model_legacy_binary` / `load_lightgbm_model` / `from_xgboost*` / `from_lightgbm` signatures.
- `treelite-mainline/python/treelite/model.py` — `Model` methods (`serialize`/`serialize_bytes`/`deserialize`/`deserialize_bytes`/`dump_as_json`/`num_tree`/`num_feature`/`input_type`/`output_type`/`concatenate`), `HeaderAccessor`/`TreeAccessor`, and the `_TreelitePyBufferFrame`/`_PyBuffer` ctypes structs (the zero-copy frame layout the Rust `Frame` enum must match).
- `treelite-mainline/python/treelite/gtil/__init__.py` + `gtil/gtil.py` — `predict` / `predict_leaf` / `predict_per_tree` signatures (the `backend=` kwarg of D-05 is additive over these).
- `treelite-mainline/python/treelite/sklearn/importer.py` — the Python-side estimator→arrays extraction to port (Claude's-discretion marshalling).
- `treelite-mainline/python/treelite/core.py` — `class TreeliteError(Exception)` (the single exception of D-06).
- `treelite-mainline/python/treelite/py.typed` — upstream's PEP 561 marker (precedent for D-10).

### In-repo Rust assets the binding wires into (the seam Phase 8 plugs into)
- `crates/treelite-core/src/serialize/pybuffer.rs` — the **zero-copy `Frame<'a>` PyBuffer frames** (SER-02, Phase 2 D-06) borrowing into `TreeBuf` columns; serialize/`serialize_bytes` ride this. Lifetime tied to `&Model` (model must outlive frames).
- `crates/treelite-gtil/src/lib.rs` — `predict::<O>` (~892) / `predict_sparse::<O>` (~949); the predict surface the Python `gtil.*` calls dispatch to.
- `crates/treelite-xgboost`, `crates/treelite-lightgbm`, `crates/treelite-sklearn/src/lib.rs` — loader entry points behind `frontend.*` / `sklearn.import_model` (treelite-sklearn already takes array signatures — Phase 4 D-01).
- `crates/treelite-harness/src/lib.rs` — `Backend` enum (~55) + `DeviceUnavailable` skip semantics (~228); the backend the `predict(backend=...)` kwarg (D-05) selects.
- `crates/treelite-cubecl/src/error.rs` — `CubeclError::DeviceUnavailable` (~90), translated to `TreeliteError` per D-08.

### Prior context (precedent inherited — read before planning)
- `.planning/phases/07-gpu-backend-equivalence-report/07-CONTEXT.md` — D-04 (per-backend additive Cargo features, explicit selection, no auto-detect), D-05 (device-absent = typed skip, no silent CPU fallback), D-09 (crossover documented-only; **Phase 8 is the named consumer**). These directly shape D-04/D-05/D-08.
- `.planning/phases/02-builder-serialization/02-CONTEXT.md` — SER-02 zero-copy PyBuffer frames decision (the serialize path D-01/the `Frame` enum the Python side consumes). *(Read if present; the frame layout also lives in `pybuffer.rs`.)*
- `.planning/ROADMAP.md` §"Phase 8" — SC1/SC2/SC3 (the locked WHAT) + the **research flag**: PyO3 0.28 buffer-protocol; numpy zero-copy return; GIL/threading pattern.
- `.planning/REQUIREMENTS.md` — PY-01..06, MEM-04 (this phase); SER-02, BLD-02 (`concatenate`) as completed dependencies.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONVENTIONS.md`, `.planning/codebase/TESTING.md`, `.planning/codebase/INTEGRATIONS.md` — SoA/variant pattern, thiserror translation discipline, frozen-golden + harness layout.

### External library docs (research phase fetches current versions)
- **PyO3 0.28** — `#[pymodule]`/`#[pyclass]`/`#[pyfunction]`, abi3 feature, buffer protocol (`PyBuffer`), panic handling, exception types. (ROADMAP research flag.)
- **rust-numpy / numpy crate** — `PyReadonlyArray`/`PyArray` zero-copy borrow + zero-copy return.
- **maturin** — abi3-py310 wheel build, per-feature wheel builds (the D-04 per-backend wheels), `pyproject.toml` `[tool.maturin]` config.
- **Optimisor manuals** — `/home/user/Documents/workspace/optimisor/manual/ZERO_COPY_ARROW_CUBECL.md`, `ZERO_COPY_TRANSMUTATION_CUBECL.md` — zero-copy buffer borrow patterns relevant to MEM-04.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Zero-copy `Frame<'a>` PyBuffer frames** (`treelite-core/src/serialize/pybuffer.rs`, SER-02) — already borrow directly into `TreeBuf` columns with `&Model` lifetime; `Model.serialize_bytes`/PyBuffer round-trip rides this unchanged. The Python `_PyBuffer`/`_TreelitePyBufferFrame` ctypes layout it must match is in upstream `model.py`.
- **`predict::<O>` / `predict_sparse::<O>`** (`treelite-gtil`) — the full scalar/cubecl GTIL surface (all 4 predict kinds, postprocessors) the Python `gtil.*` calls dispatch into. `O: PredictOut` already generalizes f32/f64 output.
- **`treelite-sklearn` array entry points** (Phase 4 D-01) — take array signatures, so the Python-side `import_model` extraction (port of upstream `importer.py`) passes numpy arrays straight in.
- **`Backend` enum + `DeviceUnavailable` skip** (`treelite-harness`, `treelite-cubecl`, Phase 7) — the runtime-selectable backend the `predict(backend=...)` kwarg (D-05) drives; the typed device-absent skip becomes a `TreeliteError` (D-08).
- **Loader crates** (`treelite-xgboost`/`treelite-lightgbm`/`treelite-sklearn`) — the `frontend.*` Python functions are thin pyo3 wrappers over these.

### Established Patterns
- **thiserror typed errors at crate boundaries** → translated to a single `TreeliteError` at the FFI edge (D-06); no panic crosses (D-07) — extends the project-wide error discipline.
- **Additive Cargo features, default-minimal** (Phase 7 D-04) — `cpu` default, GPU backends opt-in; Phase 8 expresses this as **per-backend maturin wheels** (D-04).
- **Frozen goldens + 1e-5 equivalence as the measuring stick** — the Python A/B pytest (D-11) asserts the same 1e-5 invariant at the Python surface, now against a *live* upstream import (enabled by the `treelite_rs` name).
- **Drop-in fidelity to upstream** — mirrors the porting ethos (faithful 1:1 reproduction) at the API layer (D-01).

### Integration Points
- **New `treelite-py` workspace crate** — the single pyo3 extension module exporting the mirrored surface; registered in root `Cargo.toml` members + a maturin `pyproject.toml`.
- **Python buffer protocol → `predict`** — borrowed numpy array (C-contiguous, exact dtype) consumed zero-copy (MEM-04, D-03) into `predict::<O>`; output returned as numpy.
- **`backend=` kwarg → `Backend` enum** — Python string → Rust `Backend` selection within the installed wheel's compiled-in backends (D-05).
- **Maturin per-backend builds** — root/`treelite-cubecl` Cargo features (`rocm`/`cuda`/`wgpu`) drive distinct wheels (D-04); abi3-py310 (D-09).
- **A/B pytest in the `uv run` venv** — imports `treelite_rs` + upstream `treelite` side-by-side (D-11); runs on the main tree where the golden-capture venv lives.

</code_context>

<specifics>
## Specific Ideas

- **`import treelite_rs as treelite` is the drop-in incantation.** D-01 mirrors the *shape* exactly; D-02 only changes the import name so the port can sit next to the original. The two together make the binding both faithful and A/B-testable in one venv.
- **Backend is an install-time choice, not a runtime resolver.** The user "installs the library per backend" — a `cpu` wheel vs a `rocm` wheel, etc. (D-04). Within a GPU wheel, CPU is still bundled and `backend=` picks (D-05), so the documented Phase-7 crossover is exercisable from Python while explicit selection stays literal and honest (no silent fallback).
- **Honesty at the boundary mirrors Phase 7's ethos.** Strict zero-copy or error (D-03), single descriptive `TreeliteError` (D-06), caught panics (D-07), typed device-absent surfaced not hidden (D-08) — the Python edge tells the truth about what it did, same as the GPU report did about deviation.
- **The 1e-5 proof gets a live witness.** Rather than only trusting frozen goldens, the A/B pytest (D-11) re-derives the headline invariant against the actual upstream `treelite` at the Python surface — the strongest possible end-to-end fidelity claim for v1's sole binding.

</specifics>

<deferred>
## Deferred Ideas

- **sklearn `export_model`** — only `import_model` is required (PY-04); round-trip export is not in v1 scope. Note for a future phase if Python export is ever wanted.
- **Memory-efficiency hardening through the Python path** (bytemuck input recast beyond the buffer-protocol borrow, smallvec/compact_str, custom allocator) — Phase 9.
- **CUDA/wgpu wheels validated on real hardware** — build-supported per D-04, but locally "not run — no device" (Phase 7). Their wheels fill in wherever such hardware exists (CI/future).
- **Auto-routing CPU↔GPU crossover inside `predict`** — explicitly NOT done; the `backend=` kwarg is explicit (D-05) and the crossover stays documented-only (Phase 7 D-09). A future phase could add an opt-in auto-router on top.
- **Frozen golden-vector assertions in CI** (complement to the live A/B pytest for upstream-less CI) — left to planner discretion (D-11 chose live A/B as the headline).
- **`coerce-with-copy` numpy ergonomics** — considered and rejected for v1 in favor of strict zero-copy (D-03); could be revisited as an opt-in convenience later.

None of the above is scope creep out of Phase 8 — all recorded so they aren't lost.

</deferred>

---

*Phase: 8-pyo3-python-binding*
*Context gathered: 2026-06-11*

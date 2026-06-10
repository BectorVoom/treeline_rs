# Phase 8: PyO3 Python Binding - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 8-pyo3-python-binding
**Areas discussed:** API surface & module name, numpy predict ergonomics, Error / exception mapping, Packaging & validation parity

---

## API surface & module name

### API fidelity

| Option | Description | Selected |
|--------|-------------|----------|
| Drop-in 1:1 mirror | Reproduce upstream's exact module layout & signatures (frontend / Model / gtil / sklearn). Existing Treelite scripts work unchanged. | ✓ |
| Idiomatic treelite-rs API | Cleaner Rust-flavored API not bound to upstream's module tree. | |
| Mirror core, trim deprecated | Mirror live surface 1:1, omit upstream's deprecated/legacy shims. | |

**User's choice:** Drop-in 1:1 mirror

### Module import name

| Option | Description | Selected |
|--------|-------------|----------|
| treelite_rs | Coexists with the real `treelite` in one venv — enables a live A/B 1e-5 pytest. | ✓ |
| treelite (shadow) | True drop-in name, but collides with upstream in the same env. | |
| You decide | Recommend based on the validation approach. | |

**User's choice:** treelite_rs
**Notes:** `import treelite_rs as treelite` gives drop-in usage while keeping the port importable alongside upstream for equivalence testing.

---

## numpy predict ergonomics

### Backend exposure (reformulated after clarification)

**Clarification from user:** "User install this library each backend." → backend is chosen at **install time** (one wheel per backend), not flipped at runtime. Reframed the question around that.

| Option | Description | Selected |
|--------|-------------|----------|
| GPU wheel bundles CPU too; kwarg picks | GPU wheel compiles in CPU + that GPU backend; `predict(backend=)` (default 'cpu') picks, honoring the Phase-7 crossover. Uncompiled backend → typed exception. | ✓ |
| Wheel IS the backend — no kwarg | Each wheel runs exactly its built backend; no runtime choice. | |
| Install-fixed default, kwarg can override down to CPU | Built backend is default; kwarg may only downshift to CPU. | |

**User's choice:** GPU wheel bundles CPU too; kwarg picks

### Input-array contract

| Option | Description | Selected |
|--------|-------------|----------|
| Strict zero-copy or error | Require C-contiguous + exact dtype; raise TreeliteError on mismatch. No hidden allocation. | ✓ |
| Coerce-with-copy fallback | Zero-copy happy path; otherwise copy/cast and warn. | |
| Coerce, no warning | Convert silently and predict. | |

**User's choice:** Strict zero-copy or error
**Notes:** First pass (before a clarify-reset) the user had already leaned strict; confirmed after reframing the backend question around install-time backend selection.

---

## Error / exception mapping

### Exception taxonomy

| Option | Description | Selected |
|--------|-------------|----------|
| Single TreeliteError | One TreeliteError(Exception), exact upstream parity; all Rust errors raise it. | ✓ |
| TreeliteError base + subclasses | Semantic subclasses (ModelLoadError/PredictError/SerializationError/BackendUnavailableError) all catchable as TreeliteError. | |
| You decide | Recommend (leaned base+subclasses). | |

**User's choice:** Single TreeliteError

### Panic policy

| Option | Description | Selected |
|--------|-------------|----------|
| Catch → TreeliteError | catch_unwind / PyO3 mechanism converts panics to TreeliteError; never an abort. | ✓ |
| PyO3 default (PanicException) | Rely on PyO3's built-in panic→PanicException. | |
| You decide | Recommend based on binding structure. | |

**User's choice:** Catch → TreeliteError

---

## Packaging & validation parity

### abi3 floor

| Option | Description | Selected |
|--------|-------------|----------|
| abi3-py38 | Matches upstream's stated 3.8+ support; widest compat. | |
| abi3-py39 | Floor at 3.9. | |
| abi3-py310 | Aligns with numpy 2.x's own runtime floor. | ✓ |

**User's choice:** abi3-py310

### Type information (PEP 561)

| Option | Description | Selected |
|--------|-------------|----------|
| py.typed + .pyi stubs | Ship marker + hand-written stubs for the public surface. Full IDE/mypy support. | ✓ |
| py.typed only (inline) | Marker + inline annotations, no separate .pyi. | |
| No type info (v1) | Skip PEP 561 this phase. | |

**User's choice:** py.typed + .pyi stubs

### Validation parity

| Option | Description | Selected |
|--------|-------------|----------|
| Live A/B pytest vs upstream | Import both treelite_rs and upstream treelite; assert 1e-5 directly. | ✓ |
| Reuse frozen Rust goldens | Assert against committed golden vectors; trust the Rust harness. | |
| Both | Live A/B + golden-vector assertions for CI. | |

**User's choice:** Live A/B pytest vs upstream

---

## Claude's Discretion

- **sklearn marshalling location** — port upstream `sklearn/importer.py` as Python-side estimator→arrays extraction calling the existing `treelite-sklearn` array entry points.
- **New `treelite-py` crate name & layout** (single extension vs submodules).
- **GIL/threading, buffer-protocol mechanics, numpy zero-copy return** — owned by the ROADMAP research flag.
- **Optional frozen-golden assertions to complement the live A/B pytest** for upstream-less CI.
- **`nthread`/`pred_margin` predict config** — follows upstream signature 1:1.

## Deferred Ideas

- sklearn `export_model` (only `import_model` required for v1).
- Memory-efficiency hardening through the Python path — Phase 9.
- CUDA/wgpu wheels validated on real hardware — build-supported, "not run — no device" locally.
- Auto-routing CPU↔GPU crossover inside `predict` — explicitly not done; crossover stays documented-only.
- `coerce-with-copy` numpy ergonomics — rejected for v1 in favor of strict zero-copy; revisitable as opt-in.

---
phase: 08-pyo3-python-binding
plan: 01
subsystem: python-binding
tags: [pyo3, maturin, abi3, pytest, walking-skeleton, nyquist]
requires:
  - treelite-core
  - treelite-xgboost
  - treelite-lightgbm
  - treelite-sklearn
  - treelite-gtil
  - treelite-builder
  - treelite-cubecl
provides:
  - "crates/treelite-py workspace crate"
  - "compiled abi3 cdylib _treelite_rs (empty frontend/gtil/sklearn submodules)"
  - "treelite_rs python-source package (importable, empty surface)"
  - "pytest harness: conftest + 7 RED/skip test files (the phase Nyquist instrument)"
affects:
  - "root Cargo.toml (members)"
tech-stack:
  added:
    - "pyo3 0.28.3 (abi3-py310 + extension-module)"
    - "numpy 0.28.0 (Rust bridge)"
    - "maturin 1.13.3 (build tool, uv venv)"
    - "pytest 9.0.3 (uv venv)"
  patterns:
    - "python-source maturin layout (module-name treelite_rs._treelite_rs)"
    - "per-backend cargo features forwarding to treelite-cubecl (default cpu)"
    - "collectible RED/skip pytest scaffolds (Nyquist MISSING markers)"
key-files:
  created:
    - crates/treelite-py/Cargo.toml
    - crates/treelite-py/pyproject.toml
    - crates/treelite-py/README.md
    - crates/treelite-py/.gitignore
    - crates/treelite-py/src/lib.rs
    - crates/treelite-py/python/treelite_rs/__init__.py
    - crates/treelite-py/python/treelite_rs/py.typed
    - crates/treelite-py/tests/python/conftest.py
    - crates/treelite-py/tests/python/test_frontend.py
    - crates/treelite-py/tests/python/test_predict_ab.py
    - crates/treelite-py/tests/python/test_zero_copy.py
    - crates/treelite-py/tests/python/test_serialize.py
    - crates/treelite-py/tests/python/test_sklearn_ab.py
    - crates/treelite-py/tests/python/test_errors.py
    - crates/treelite-py/tests/python/test_backend.py
  modified:
    - Cargo.toml
    - Cargo.lock
decisions:
  - "module-name treelite_rs._treelite_rs (compiled cdylib installed AS a private submodule of the pure-Python package); python-source = python/"
  - "maturin run from repo root via --manifest-path so it targets the root uv venv; running `uv run` inside the crate dir spawns a stray crate-local .venv (cleaned + .gitignored)"
  - "maturin editable install drops _treelite_rs.abi3.so/__pycache__/uv.lock into python-source — gitignored, never committed"
  - "RED stubs use @pytest.mark.skip (not xfail) so the not-yet-existing symbol references in the bodies never execute at collection or run time"
metrics:
  duration: ~7min
  tasks: 2
  files: 15
  completed: 2026-06-10
---

# Phase 8 Plan 1: PyO3 Walking-Skeleton + pytest Harness Summary

Stood up the `treelite-py` workspace crate with maturin abi3 packaging so an empty `treelite_rs` package builds, installs into the uv venv, and imports — and landed the full Nyquist pytest scaffold (conftest + 7 collectible RED/skip test files) that every downstream slice in Phase 8 verifies against (PY-06). This is the binding's walking skeleton: build → import → test plumbing proven end-to-end before any capability is wired.

## What Was Built

### Task 1 — crate + maturin abi3 packaging + minimal `#[pymodule]` (commit `71545b0`)

- Registered `crates/treelite-py` as a workspace member.
- `Cargo.toml`: `[lib] name = "_treelite_rs"`, `crate-type = ["cdylib"]`; `pyo3 0.28.3` (`abi3-py310` + `extension-module`), `numpy 0.28.0`, `thiserror` (workspace); path deps on all seven treelite crates; per-backend features `cpu`(default)/`rocm`/`cuda`/`wgpu` forwarding to `treelite-cubecl` (same shape that crate uses for `cubecl/*`).
- `pyproject.toml`: `[tool.maturin]` with `module-name = "treelite_rs._treelite_rs"`, `python-source = "python"`, `bindings = "pyo3"`, `features = ["pyo3/abi3-py310"]`.
- `src/lib.rs`: minimal `#[pymodule] fn _treelite_rs(...)` creating three EMPTY submodules (`frontend`/`gtil`/`sklearn`) via `PyModule::new` + `add_submodule`. No functions yet (those land in 08-02 .. 08-05).
- `python/treelite_rs/__init__.py`: imports the compiled `_treelite_rs` and re-exports its submodules (guarded so import succeeds before any symbol exists). `py.typed` PEP 561 marker added.

**Verification:** `cargo build -p treelite-py` exits 0; `maturin develop` builds a `cp310-abi3-linux_x86_64` wheel and installs it editable into the root uv venv; `uv run python -c "import treelite_rs"` prints `ok` with all three submodules present; `cargo test --workspace` green (no regression).

### Task 2 — pytest harness: conftest + 7 RED/skip stubs (commit `01463f1`)

- `conftest.py`: `FIXTURES`/`GTIL_FIXTURES` path constants anchored at the repo root, seeded `rng` fixture (`np.random.default_rng(0)`), session fixtures importing upstream `treelite` and `treelite_rs` via `pytest.importorskip` (D-11 skip-if-no-upstream witness), and a `scipy.sparse` `importorskip` gate.
- 7 collectible RED scaffolds, each with a real assertion body marked `@pytest.mark.skip(reason="MISSING — implemented in 08-0X")`:
  - `test_frontend.py` (PY-01), `test_predict_ab.py` (PY-02/D-11, `assert_allclose atol=1e-5 rtol=0`), `test_zero_copy.py` (MEM-04/D-03, `__array_interface__["data"][0]` + dtype/contiguity rejection), `test_serialize.py` (PY-03), `test_sklearn_ab.py` (PY-04), `test_errors.py` (PY-05, panic→`TreeliteError` not abort), `test_backend.py` (D-05/D-08, `backend=` kwarg; rocm cell hardware-gated `skipif`).

**Verification:** `uv run pytest ... --collect-only -q` collects 21 tests from all 7 files with zero errors; full run is `21 skipped`, exit 0, zero failures/errors.

## Deviations from Plan

### Auto-fixed / Auto-added (Rules 1-3)

**1. [Rule 3 - Blocking] Installed `pytest` alongside `maturin` into the uv venv**
- **Found during:** Task 2 setup (and Task 1 — `maturin` was also absent).
- **Issue:** Neither `maturin` nor `pytest` was present in the root uv venv; the plan's `user_setup` only called for maturin, but Task 2's verify command needs pytest to collect.
- **Fix:** `uv pip install maturin pytest` → `maturin 1.13.3` (matches RESEARCH) + `pytest 9.0.3`. Both are dev tooling, not slop-risk packages (maturin is in the RESEARCH legitimacy audit; pytest is the standard runner already implied by the test-harness task).
- **Files modified:** none (venv state only).

**2. [Rule 3 - Blocking] Ran `maturin develop` from repo root via `--manifest-path` instead of `cd crates/treelite-py`**
- **Found during:** Task 1 verify.
- **Issue:** The plan's verify command was `cd crates/treelite-py && uv run maturin develop`. Because the crate has its own `pyproject.toml`, `uv run` inside that dir treats it as a fresh project and creates a stray crate-local `.venv` (without maturin/treelite) — `maturin` then fails to spawn.
- **Fix:** Removed the stray `.venv`; ran `uv run maturin develop --manifest-path crates/treelite-py/Cargo.toml` from the repo root so it targets the existing root uv venv. Wheel built + installed successfully (`import treelite_rs` → ok). This is the correct invocation under the sequential-on-main-tree constraint (the venv lives at the repo root).
- **Files modified:** none.

**3. [Rule 2 - Critical hygiene] Added `crates/treelite-py/.gitignore` + a `README.md`**
- **Found during:** Task 1 commit staging.
- **Issue:** `maturin develop` drops build artifacts into the python-source tree (`_treelite_rs.abi3.so`, `__pycache__/`, a stray `uv.lock`); maturin also requires the `readme` referenced by `pyproject.toml` to exist.
- **Fix:** Added a `.gitignore` excluding `*.so`/`__pycache__`/`uv.lock`/`.venv` so generated artifacts never leak into git, and a minimal `README.md`.
- **Files modified:** `crates/treelite-py/.gitignore`, `crates/treelite-py/README.md` (both created).

## Authentication Gates

None.

## Known Stubs

The entire `treelite_rs` surface is intentionally empty in Wave 0 — this is the plan's stated purpose (walking skeleton). The compiled `frontend`/`gtil`/`sklearn` submodules contain no functions; the `__init__.py` re-export of `Model`/`TreeliteError` is guarded for symbols that do not yet exist. The 21 pytest cases are RED/skip Nyquist MISSING markers. All of this is by design and is resolved by downstream plans:

| Stub | File | Resolved by |
|------|------|-------------|
| empty `frontend`/`gtil`/`sklearn` submodules | `src/lib.rs` | 08-02 (frontend/serialize), 08-03 (gtil), 08-04 (sklearn), 08-05 (backend) |
| guarded `Model`/`TreeliteError` re-export | `python/treelite_rs/__init__.py` | 08-02 |
| 21 skipped RED tests | `tests/python/test_*.py` | each flips GREEN as its slice lands (08-02 .. 08-05) |

No stub blocks the plan's goal: PY-06 (build/import/test plumbing for an empty surface) is fully proven.

## Threat Flags

None. The implementation matches the plan's `<threat_model>`: cpu-only default wheel (T-08-02 accept), build+test on the main tree under `uv run` (T-08-01 mitigate), and the only new packages (`pyo3`/`numpy`/`maturin`/`pytest`) are PyO3-org canonical / standard tooling (T-08-SC mitigate — no human checkpoint required per the RESEARCH legitimacy audit).

## Self-Check: PASSED

- All 15 created files exist on disk (verified at commit time; Edit/Write would have errored otherwise).
- Commits exist: `71545b0` (Task 1), `01463f1` (Task 2) — confirmed in `git log --oneline`.
- `cargo build -p treelite-py` exits 0; `cargo test --workspace` green.
- `uv run python -c "import treelite_rs"` → ok (frontend/gtil/sklearn present).
- `uv run pytest crates/treelite-py/tests/python` → 21 collected, 21 skipped, exit 0.

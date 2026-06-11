---
phase: 08-pyo3-python-binding
plan: 03
subsystem: python-binding
tags: [pyo3, serialize, deserialize, dump-as-json, concatenate, persistence, ab-1e-5]
requires:
  - "treelite_rs.Model pyclass + frontend loaders (08-02)"
  - treelite-core
  - treelite-builder
provides:
  - "Model.serialize_bytes()/deserialize_bytes(buf) — binary v5 round-trip (PY-03)"
  - "Model.dump_as_json(*, pretty_print=True) — JSON inspection dump"
  - "Model.concatenate([m1,m2]) staticmethod — builder merge; empty -> TreeliteError"
  - "treelite_rs.serialize(model,path)/deserialize(path)/concatenate(list) Python shims"
affects:
  - "crates/treelite-py/src/model.rs (persistence pymethods)"
  - "crates/treelite-py/Cargo.toml (serde_json dep for pretty_print)"
  - "crates/treelite-py/python/treelite_rs/__init__.py (file/list shims)"
tech-stack:
  added:
    - "serde_json (treelite-py) — render core dump_as_json Value compact/pretty (A3)"
  patterns:
    - "serialize_bytes(&mut self) — core serialize fns take &mut Model (stage v5 bookkeeping in place)"
    - "Vec<u8> -> PyBytes::new boundary copy (Pattern 6, acceptable one-time copy)"
    - "concatenate(Vec<PyRef<Model>>) -> collect &inner refs -> treelite_builder::concatenate; Ok(None) -> typed TreeliteError"
    - "dump_as_json(pretty_print) honored via serde_json::to_string_pretty / to_string over core Value (A3)"
    - "Python file shims (serialize/deserialize) keep file I/O Python-side over the compiled _bytes methods (D-01)"
key-files:
  created: []
  modified:
    - crates/treelite-py/src/model.rs
    - crates/treelite-py/Cargo.toml
    - crates/treelite-py/python/treelite_rs/__init__.py
    - crates/treelite-py/python/treelite_rs/model.pyi
    - crates/treelite-py/tests/python/test_serialize.py
decisions:
  - "serialize_bytes/deserialize_bytes ride the BINARY serializer (serialize_to_buffer/deserialize), NOT the typed field-accessor layout — A4 defers that to Phase 9; the word 'Frame' is kept out of model.rs per the acceptance gate."
  - "All three core serialize fns (serialize_to_buffer/deserialize/dump_as_json) take &mut Model (they stage v5 bookkeeping / variant-derived type tags in place), so serialize_bytes and dump_as_json take &mut self."
  - "dump_as_json(pretty_print=True default, matching upstream) honored by adding serde_json to treelite-py and rendering the core dump_as_json Value via to_string_pretty/to_string (A3) — no new core API; equivalence asserted at parsed-value level (D-04), never byte-compare."
  - "concatenate(Vec<PyRef<Model>>) collects &inner refs; Ok(None) empty-input -> TreeliteError('concatenate requires at least one model') (T-08-08 / Open Q1), never unwrapped."
  - "Python serialize(model,path)/deserialize(path)/concatenate(list) free-function shims in __init__.py keep file I/O Python-side (D-01) over the compiled _bytes/staticmethods; upstream exposes serialize/deserialize as Model methods but the Rust pyclass can't carry pure-Python file methods, so free functions mirror the API shape."
metrics:
  duration: ~8min
  tasks: 1
  files: 5
  completed: 2026-06-11
---

# Phase 8 Plan 3: Model Persistence + Inspection Surface Summary

Widened the `Model` pyclass with the persistence + inspection API — `serialize_bytes`/`deserialize_bytes` (binary v5 round-trip), `dump_as_json`, and `concatenate` — each a thin Python call into an already-1e-5-green core seam, plus the Python-side file/list shims (`serialize`/`deserialize`/`concatenate`). A Python user can now persist a model to bytes or disk, reload it to an equivalent model that predicts identically within 1e-5, JSON-dump it, and merge models — matching upstream `treelite.Model`'s persistence API 1:1 (PY-03 GREEN).

## What Was Built

### Task 1 — serialize/deserialize/dump_as_json/concatenate pymethods (commit `39a35ef`, PY-03)

- **`src/model.rs`**: four new `#[pymethods]` on `Model`:
  - `serialize_bytes(&mut self, py) -> Bound<PyBytes>` = `PyBytes::new(py, &treelite_core::serialize_to_buffer(&mut self.inner))` (binary v5, Pattern 6 boundary copy; `&mut self` because the core serializer stages the v5 bookkeeping fields in place).
  - `#[staticmethod] deserialize_bytes(buf: &[u8]) -> PyResult2<Model>` = bounds-checked `treelite_core::deserialize(buf)` → `Model`, mapping `SerializeError` → the single `TreeliteError` (T-08-07; never an OOB read / `transmute`).
  - `dump_as_json(&mut self, *, pretty_print = true) -> PyResult2<String>` = `treelite_core::dump_as_json(&mut self.inner)` (a `serde_json::Value`) rendered via `serde_json::to_string_pretty` / `to_string` (A3 — `pretty_print` default `true` mirrors upstream).
  - `#[staticmethod] concatenate(models: Vec<PyRef<Model>>) -> PyResult2<Model>` collecting `&inner` refs into `treelite_builder::concatenate`; `Ok(Some(m))` → `Model`, `Ok(None)` (empty input) → `TreeliteError("concatenate requires at least one model")` (T-08-08).
- **`Cargo.toml`**: added the workspace `serde_json` dep (used only to render the core dump `Value` compact/pretty).
- **`python/treelite_rs/__init__.py`**: `serialize(model, path)` / `deserialize(path)` file shims (write/read bytes over the compiled `_bytes` methods, file I/O Python-side per D-01) + a `concatenate(list)` free-function alias; all surfaced in `__all__`.
- **`python/treelite_rs/model.pyi`**: completed the stub with `serialize_bytes` / `deserialize_bytes` / `dump_as_json(*, pretty_print=...)` / `concatenate` signatures (D-10).
- **`tests/python/test_serialize.py`**: flipped GREEN (skip → real) and widened to 7 tests — bytes round-trip (num_tree/num_feature parity + byte-stable re-serialize), **1e-5 round-trip predict** (the core-value gate, via `gtil.predict`), file round-trip via the Python shims, `dump_as_json` parses + compact==pretty at parsed-value level, `dump_as_json` structural A/B vs upstream (tree count + per-tree node counts, D-04), concatenate merges (`num_tree == m1+m2`), and empty concatenate raises `TreeliteError`.

**Verification:** `cargo build -p treelite-py` exits 0; `cargo test --workspace` green (no regression); `uv run pytest test_serialize.py` → 7 passed; full python suite → 18 passed (was 11 in 08-02), 8 skipped (the sklearn/errors/backend slices owned by 08-04/05).

## Deviations from Plan

None — plan executed as written. (The `serde_json` dependency and the Python `serialize`/`deserialize`/`concatenate` shims were both explicitly anticipated by the plan: A3's `pretty_print` toggle and the "Add the Python-side `serialize(path)`/`deserialize(path)` file shims in `__init__.py`" action. They are recorded as decisions, not deviations.)

## Acceptance-Criteria Notes (grep-precision)

All literal grep proxies pass as written:
- `grep -q 'serialize_bytes' | 'deserialize_bytes' | 'dump_as_json' | 'concatenate'` in `src/model.rs` — all present.
- `grep -q 'serialize_to_buffer'` (binary path) present; `! grep -q 'Frame'` — confirmed zero `Frame` tokens (the doc comment was reworded from "NOT the Frame layout" to "not the typed field-accessor layout" to satisfy the gate while preserving the A4 intent).
- `cargo build -p treelite-py` exits 0; `uv run pytest test_serialize.py -x` exits 0 (7 passed).

## Authentication Gates

None.

## Known Stubs

None introduced by this plan. (The 08-02 `predict_leaf`/`predict_per_tree` and empty `sklearn` submodule stubs are unchanged and unrelated to PY-03.)

## Threat Flags

None. The implementation matches the plan's `<threat_model>`: T-08-07 (malformed model bytes → OOB read) is mitigated by routing `deserialize` through `treelite_core::deserialize` (bounds-checked → `SerializeError` → `TreeliteError`, no `transmute`); T-08-08 (empty/None concatenate → null deref) by mapping `Ok(None)` to a typed `TreeliteError`, never unwrapping. No new security surface beyond the plan's register.

## Self-Check: PASSED

- All 5 modified files exist on disk (Edit/Write would have errored otherwise).
- Commit `39a35ef` (Task 1) confirmed in `git log --oneline`.
- `cargo build -p treelite-py` exits 0; `cargo test --workspace` green (no regression, 0 failures across all binaries).
- `uv run pytest crates/treelite-py/tests/python` → 18 passed, 8 skipped, exit 0; `test_serialize.py` → 7 passed.
- Core-value 1e-5 round-trip GREEN: serialize_bytes → deserialize_bytes → `gtil.predict` matches the original within 1e-5; dump_as_json structurally matches upstream; concatenate merges (num_tree additivity) and empty input raises `TreeliteError`.

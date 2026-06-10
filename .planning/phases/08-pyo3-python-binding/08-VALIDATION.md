---
phase: 8
slug: pyo3-python-binding
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `cargo test` (Rust workspace, existing) + `pytest` via `uv run pytest` (Python A/B, main tree golden-capture venv) |
| **Config file** | none yet for `treelite-py` — Wave 0 adds `crates/treelite-py/pyproject.toml` + `tests/python/` |
| **Build-into-venv** | `cd crates/treelite-py && uv run maturin develop` (main tree — prerequisite for any pytest) |
| **Quick run command** | `uv run pytest crates/treelite-py/tests/python -x -q` |
| **Full suite command** | `cargo test --workspace && uv run pytest crates/treelite-py/tests/python` |
| **Estimated runtime** | ~60–120 seconds (build + A/B fixtures) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p treelite-py` + the one pytest file the task touches
- **After every plan wave:** Run `cargo test --workspace && uv run pytest crates/treelite-py/tests/python`
- **Before `/gsd-verify-work`:** Full Rust workspace green + full pytest green (live A/B within 1e-5 across all fixtures)
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists | Status |
|--------|----------|-----------|-------------------|-------------|--------|
| PY-01 | load XGB/LGB/sklearn from Python yields Model with correct `num_tree`/`num_feature` | integration | `uv run pytest .../test_frontend.py -x` | ❌ W0 | ⬜ pending |
| PY-02 | predict over numpy == upstream within 1e-5 (dense, f32 & f64) | integration A/B (D-11) | `uv run pytest .../test_predict_ab.py -x` | ❌ W0 | ⬜ pending |
| PY-02 / MEM-04 | borrowed numpy consumed zero-copy (no-copy proof) | unit | `uv run pytest .../test_zero_copy.py -x` | ❌ W0 | ⬜ pending |
| PY-03 | serialize_bytes→deserialize_bytes round-trips; dump_as_json parses | integration | `uv run pytest .../test_serialize.py -x` | ❌ W0 | ⬜ pending |
| PY-04 | `sklearn.import_model(fitted)` predicts == upstream within 1e-5 | integration A/B | `uv run pytest .../test_sklearn_ab.py -x` | ❌ W0 | ⬜ pending |
| PY-05 | bad dtype / malformed model raise `TreeliteError`; no abort on forced panic | unit | `uv run pytest .../test_errors.py -x` | ❌ W0 | ⬜ pending |
| PY-06 | wheel builds + imports as abi3 | smoke | `uv run maturin build && uv run python -c "import treelite_rs"` | ❌ W0 | ⬜ pending |
| D-05 / D-08 | `predict(backend='rocm')` works on device / unavailable backend raises `TreeliteError` | integration (hw-gated) | `uv run pytest .../test_backend.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Zero-copy (no-copy) proof for MEM-04 / D-03:** assert the numpy input array's `__array_interface__['data'][0]` (buffer pointer) is unchanged across the predict call; Rust-side a debug-only `as_slice().as_ptr()` identity check (mirroring `tests/serialize_pybuffer.rs`'s `.as_ptr()` equality proof) confirms the borrow aliases numpy memory. Complement: pass a non-contiguous (`arr[:, ::2]`) or wrong-dtype array and assert `TreeliteError` (D-03 strict).

**Live A/B equivalence (D-11) — headline test:** import both `treelite` (upstream 4.7.0, confirmed installed) and `treelite_rs`, load the same fixtures, predict the same inputs, `np.testing.assert_allclose(..., atol=1e-5, rtol=0)`. Run across every captured model class (XGB json/ubjson/legacy, LightGBM numerical/categorical, sklearn RF/ET/GB/IsolationForest/HistGB), both presets, dense + (if scipy) sparse, all three predict kinds (`predict`/`predict_leaf`/`predict_per_tree`).

---

## Wave 0 Requirements

- [ ] `crates/treelite-py/Cargo.toml` + `pyproject.toml` (maturin config, abi3-py310, per-backend features) — register `crates/treelite-py` in root `Cargo.toml` members
- [ ] `crates/treelite-py/tests/python/conftest.py` — shared fixtures (fixture paths, rng seed, `treelite`+`treelite_rs` import, skip-if-no-upstream)
- [ ] `test_frontend.py` (PY-01), `test_predict_ab.py` (PY-02), `test_zero_copy.py` (MEM-04/D-03), `test_serialize.py` (PY-03), `test_sklearn_ab.py` (PY-04), `test_errors.py` (PY-05), `test_backend.py` (D-05/D-08, hw-gated)
- [ ] `maturin develop` into the existing `uv` venv (main tree) — prerequisite for any pytest
- [ ] Confirm `scipy` availability (1.17.1 present) or descope sparse A/B cells

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `predict(backend='rocm')` runs on real GPU | D-05 | ROCm hardware-gated; CI/no-device runs skip | On the ROCm box: `maturin develop --features rocm` then `uv run pytest .../test_backend.py -k rocm` |
| CUDA / wgpu wheels build & import | D-04 | No NVIDIA device locally — build-only, not hardware-validatable | `maturin build --features cuda` / `--features wgpu` succeeds; import smoke only |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

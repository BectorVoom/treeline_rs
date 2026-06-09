# External Integrations

**Analysis Date:** 2026-06-09

## APIs & External Services

**Rust crate (treeline_rs):**
- None — the crate is at scaffold stage with no external integrations.

**Upstream C++ reference (treelite-mainline) — being ported:**

- XGBoost model format (no runtime dependency — file-based)
  - Legacy binary format: `treelite-mainline/src/model_loader/xgboost_legacy.cc`
  - JSON format: `treelite-mainline/src/model_loader/xgboost_json.cc`
    and `treelite-mainline/src/model_loader/detail/xgboost_json/`
  - UBJSON format: `treelite-mainline/src/model_loader/xgboost_ubjson.cc`
  - C API entry: `TreeliteLoadXGBoostModelLegacyBinary`, `TreeliteLoadXGBoostModelJSON`,
    `TreeliteLoadXGBoostModelUBJSON` in `treelite-mainline/include/treelite/c_api.h`

- LightGBM model format (no runtime dependency — file-based)
  - Text format parser: `treelite-mainline/src/model_loader/lightgbm.cc`
  - Detail header: `treelite-mainline/src/model_loader/detail/lightgbm.h`
  - C API entry: `TreeliteLoadLightGBMModel` in `treelite-mainline/include/treelite/c_api.h`
  - Python integration tests: `treelite-mainline/tests/python/test_lightgbm_integration.py`

- scikit-learn model format (no runtime dependency — Python object protocol)
  - Importer/exporter: `treelite-mainline/python/treelite/sklearn/`
  - Isolation forest support: `treelite-mainline/python/treelite/sklearn/isolation_forest.py`
  - Python integration tests: `treelite-mainline/tests/python/test_sklearn_integration.py`

## Data Storage

**Databases:**
- None — no database used. Models are loaded from files (binary, JSON, UBJSON, text).

**File Storage:**
- Local filesystem only
  - Model files read by the C API via filename parameters
  - Serialization format (v3 and v4) documented in
    `treelite-mainline/docs/serialization/v3.rst` and `treelite-mainline/docs/serialization/v4.rst`
  - Serializer implementation: `treelite-mainline/src/serializer.cc`,
    `treelite-mainline/src/json_serializer.cc`

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- Not applicable — this is a pure ML inference library with no user identity concepts.

## Monitoring & Observability

**Error Tracking:**
- None (no external error tracking service)

**Logs:**
- Custom logging subsystem in the upstream C++ library
  - `treelite-mainline/include/treelite/logging.h`
  - `treelite-mainline/src/logging.cc`
  - Thread-local error storage pattern: `treelite-mainline/include/treelite/thread_local.h`
  - Error propagated to callers via C API return codes (-1 for failure)
    and `TreeliteGetLastError()` (see `treelite-mainline/include/treelite/c_api_error.h`)

## CI/CD & Deployment

**Upstream C++ project CI:**
- Docker-based CI: `treelite-mainline/tests/ci_build/` (Ubuntu 20 amd64 and aarch64 images)
- Build scripts: `treelite-mainline/ops/build-linux.sh`, `build-macos.sh`, `build-windows.bat`
- Python wheel testing: `treelite-mainline/ops/test-linux-python-wheel.sh`,
  `test-macos-python-wheel.sh`, `test-win-python-wheel.bat`
- Serializer compatibility testing: `treelite-mainline/ops/test-serializer-compatibility.sh`
- Pre-commit config: `treelite-mainline/.pre-commit-config.yaml`
- ReadTheDocs docs build: `treelite-mainline/.readthedocs.yaml`

**Rust crate CI:**
- None configured yet

**Hosting:**
- Upstream: Python wheels published to PyPI; docs at https://treelite.readthedocs.io/
- Rust crate: not yet published

## Environment Configuration

**Required env vars:**
- Rust crate: none
- Upstream C++ (optional): `CONDA_PREFIX` — auto-detected by CMake when building in a Conda env

**Secrets location:**
- None — no secrets used

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Python Buffer Protocol Integration

**Upstream C++ — upstream-specific FFI pattern:**
- Implements PEP 3118 (Python buffer protocol) for zero-copy data transfer between
  C++ and Python
- `TreelitePyBufferFrame` struct defined in `treelite-mainline/include/treelite/c_api.h`
- Used to pass numpy arrays without copying through the C API boundary
- This is a key pattern to replicate (or replace with safe Rust FFI) during the port

---

*Integration audit: 2026-06-09*

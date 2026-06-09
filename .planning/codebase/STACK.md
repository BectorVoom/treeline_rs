# Technology Stack

**Analysis Date:** 2026-06-09

## Languages

**Primary (Rust crate — treeline_rs):**
- Rust (edition 2024) — the active port target; currently scaffolded at `src/main.rs`

**Primary (upstream reference — treelite-mainline):**
- C++17 — core library logic; all files under `treelite-mainline/src/` and `treelite-mainline/include/`
- C — public C API surface exposed via `treelite-mainline/include/treelite/c_api.h`

**Secondary (upstream reference):**
- Python 3.8+ — Python bindings and packaging at `treelite-mainline/python/`

## Runtime

**Rust crate:**
- Runtime: native binary / library (no external runtime)
- Toolchain: Rust stable, edition 2024

**Upstream C++ project:**
- Runtime: native shared library (`libtreelite.so` / `treelite.dll`)
- Minimum compiler: GCC 8.1 / AppleClang 11.0 / Clang 9.0 / MSVC VS2022 (1930+)

## Package Manager

**Rust crate:**
- Cargo (Rust standard package manager)
- Lockfile: not yet committed (no `Cargo.lock` present — crate has no dependencies yet)

**Upstream C++ project:**
- CMake 3.16+ with `FetchContent` for vendored dependencies
- Python: hatchling (custom PEP 517 backend at `treelite-mainline/python/packager/pep517.py`)

## Frameworks

**Rust crate:**
- None yet — `src/main.rs` is a bare `fn main()` scaffold with zero dependencies declared in `Cargo.toml`

**Upstream C++ project (reference for porting):**
- CMake — build system (`treelite-mainline/CMakeLists.txt`, `treelite-mainline/src/CMakeLists.txt`)
- OpenMP — optional parallelism (`USE_OPENMP=ON` default, `treelite-mainline/include/treelite/detail/threading_utils.h`)
- Google Test 1.14.0 — C++ unit tests (`treelite-mainline/tests/cpp/`)
- pytest + hypothesis — Python tests (`treelite-mainline/tests/python/`)

## Key Dependencies

**Rust crate:**
- None declared (empty `[dependencies]` section in `Cargo.toml`)

**Upstream C++ project (vendored via CMake FetchContent):**
- RapidJSON (header-only) — JSON parsing for XGBoost model loader; pinned to commit `ab1842a2`
  (`treelite-mainline/src/model_loader/detail/xgboost_json/`)
- nlohmann/json 3.11.3 (header-only) — UBJSON parsing for XGBoost UBJSON format
  (`treelite-mainline/src/model_loader/xgboost_ubjson.cc`)
- mdspan 0.6.0 (header-only, kokkos) — multi-dimensional array views (C++23 backport)
  (`treelite-mainline/src/gtil/predict.cc`)
- Google Test 1.14.0 — test framework (test-only)
- fmtlib 10.1.1 — string formatting in tests (test-only)

**Python binding dependencies (upstream):**
- numpy — array I/O
- scipy — sparse matrix support
- packaging — version handling
- scikit-learn (optional) — sklearn model importer/exporter
  (`treelite-mainline/python/treelite/sklearn/`)

## Configuration

**Rust crate:**
- Configuration: `Cargo.toml` at repo root — currently minimal (name, version, edition only)
- No environment variable configuration

**Upstream C++ project:**
- CMake options: `USE_OPENMP`, `BUILD_CPP_TEST`, `Treelite_BUILD_STATIC_LIBS`,
  `HIDE_CXX_SYMBOLS`, `TEST_COVERAGE`, `USE_SANITIZER`
- Conda environment auto-detection via `DETECT_CONDA_ENV` CMake option
- Version defined in `CMakeLists.txt` (`4.7.0`) and generated into
  `treelite-mainline/cmake/version.h.in` → `include/treelite/version.h`

**Build:**
- Rust: `cargo build` / `cargo run`
- Upstream C++: `treelite-mainline/CMakeLists.txt` produces shared library `libtreelite`
  (and optional `libtreelite_static`)

## Platform Requirements

**Development (Rust crate):**
- Any platform supported by Rust stable toolchain
- Cargo

**Development (upstream C++ reference):**
- Linux (amd64, aarch64), macOS, Windows
- CMake 3.16+
- GCC 8.1+ / AppleClang 11.0+ / Clang 9.0+ / MSVC VS2022+
- Optional: Conda environment for dependency resolution

**Production:**
- Rust crate: not yet defined (scaffold stage)
- Upstream C++: cross-platform shared library deployed via Python wheel or CMake install

---

*Stack analysis: 2026-06-09*

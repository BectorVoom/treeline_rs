<!-- GSD:project-start source:PROJECT.md -->

## Project

**treelite-rs**

A from-scratch Rust rewrite of [Treelite](https://github.com/dmlc/treelite) — the tree-ensemble model library that imports trained gradient-boosted/forest models (XGBoost, LightGBM, scikit-learn), holds them in a compact in-memory representation, runs reference inference (GTIL), and serializes them. The upstream C++ source (v4.7.0) is vendored read-only at `treelite-mainline/` and is the porting source of truth. The Rust version is a Cargo workspace with strict separation of concerns, a PyO3 Python binding, `cubecl`-accelerated inference, and aggressive memory efficiency — validated to match upstream predictions within 1e-5.

**Core Value:** **Predictions match upstream Treelite within 1e-5.** A model loaded and predicted through treelite-rs must produce numerically equivalent output to the C++ original. Everything else (speed, memory, GPU) is secondary to that fidelity.

### Constraints

- **Tech stack**: Rust (edition 2024) Cargo workspace — modular crates, clear separation of responsibilities.
- **Equivalence**: predictions must match upstream Treelite within **1e-5** (the highest precision upstream targets).
- **Python**: PyO3 module — the sole external binding. No C-API.
- **Error handling**: `thiserror` in library crates, `anyhow` in binaries/tests.
- **Compute**: GTIL inference hot path via `cubecl`; CPU backend default, GPU opt-in.
- **Dependencies**: all crates pinned to their latest published versions.
- **Performance/Memory**: high focus on memory efficiency — zero-copy where possible, compact data structures, custom allocator (jemalloc/mimalloc), optional f16 half-precision via cubecl.
- **Serialization**: current (v5) format generation only for v1.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust (edition 2024) — the active port target; currently scaffolded at `src/main.rs`
- C++17 — core library logic; all files under `treelite-mainline/src/` and `treelite-mainline/include/`
- C — public C API surface exposed via `treelite-mainline/include/treelite/c_api.h`
- Python 3.8+ — Python bindings and packaging at `treelite-mainline/python/`

## Runtime

- Runtime: native binary / library (no external runtime)
- Toolchain: Rust stable, edition 2024
- Runtime: native shared library (`libtreelite.so` / `treelite.dll`)
- Minimum compiler: GCC 8.1 / AppleClang 11.0 / Clang 9.0 / MSVC VS2022 (1930+)

## Package Manager

- Cargo (Rust standard package manager)
- Lockfile: not yet committed (no `Cargo.lock` present — crate has no dependencies yet)
- CMake 3.16+ with `FetchContent` for vendored dependencies
- Python: hatchling (custom PEP 517 backend at `treelite-mainline/python/packager/pep517.py`)

## Frameworks

- None yet — `src/main.rs` is a bare `fn main()` scaffold with zero dependencies declared in `Cargo.toml`
- CMake — build system (`treelite-mainline/CMakeLists.txt`, `treelite-mainline/src/CMakeLists.txt`)
- OpenMP — optional parallelism (`USE_OPENMP=ON` default, `treelite-mainline/include/treelite/detail/threading_utils.h`)
- Google Test 1.14.0 — C++ unit tests (`treelite-mainline/tests/cpp/`)
- pytest + hypothesis — Python tests (`treelite-mainline/tests/python/`)

## Key Dependencies

- None declared (empty `[dependencies]` section in `Cargo.toml`)
- RapidJSON (header-only) — JSON parsing for XGBoost model loader; pinned to commit `ab1842a2`
- nlohmann/json 3.11.3 (header-only) — UBJSON parsing for XGBoost UBJSON format
- mdspan 0.6.0 (header-only, kokkos) — multi-dimensional array views (C++23 backport)
- Google Test 1.14.0 — test framework (test-only)
- fmtlib 10.1.1 — string formatting in tests (test-only)
- numpy — array I/O
- scipy — sparse matrix support
- packaging — version handling
- scikit-learn (optional) — sklearn model importer/exporter

## Configuration

- Configuration: `Cargo.toml` at repo root — currently minimal (name, version, edition only)
- No environment variable configuration
- CMake options: `USE_OPENMP`, `BUILD_CPP_TEST`, `Treelite_BUILD_STATIC_LIBS`,
- Conda environment auto-detection via `DETECT_CONDA_ENV` CMake option
- Version defined in `CMakeLists.txt` (`4.7.0`) and generated into
- Rust: `cargo build` / `cargo run`
- Upstream C++: `treelite-mainline/CMakeLists.txt` produces shared library `libtreelite`

## Platform Requirements

- Any platform supported by Rust stable toolchain
- Cargo
- Linux (amd64, aarch64), macOS, Windows
- CMake 3.16+
- GCC 8.1+ / AppleClang 11.0+ / Clang 9.0+ / MSVC VS2022+
- Optional: Conda environment for dependency resolution
- Rust crate: not yet defined (scaffold stage)
- Upstream C++: cross-platform shared library deployed via Python wheel or CMake install

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Overview

## Rust Crate (`src/`)

### Naming Patterns

- Snake\_case for all Rust source files (standard Rust convention, not yet tested at scale)
- `snake_case` for all functions (Rust standard)
- `PascalCase` for all types (Rust standard)
- `SCREAMING_SNAKE_CASE` (Rust standard)
- `snake_case` directory/file names

### Code Style

- `rustfmt` with default settings (no custom config present)
- Run: `cargo fmt`
- `clippy` with default settings (no custom config present)
- Run: `cargo clippy`

### Error Handling

- No error-handling patterns established yet (crate is a stub)
- Port target uses `throw treelite::Error(...)` on fatal conditions — translate to Rust `Result<T, E>` with a custom error enum

### Module Design

- No module structure yet beyond `fn main()` in `src/main.rs`
- Cargo edition 2024 is declared; use `mod` declarations as modules are added

## C++ Reference Codebase (`treelite-mainline/`)

### Naming Patterns

- `snake_case.cc` / `snake_case.h` for implementation and header files
- Header guards: `TREELITE_<PATH>_H_` (all caps, path separators become underscores, trailing underscore)
- Test files: `test_<component>.cc` in `tests/cpp/`
- `PascalCase` — e.g., `ModelBuilder`, `LogMessageFatal`, `ReturnValueEntry`
- `PascalCase` for public API methods — e.g., `StartTree()`, `EndNode()`, `NumericalTest()`
- `snake_case` for private member variables with trailing underscore — e.g., `log_stream_`, `node_type_`
- Descriptive `PascalCase` suffixed with type meaning — e.g., `ThresholdType`, `LeafOutputType`, `ThresholdT`, `LeafOutputT`
- Enum types in `PascalCase` — e.g., `TaskType`, `TypeInfo`, `Operator`
- Enum values prefixed with `k` — e.g., `kBinaryClf`, `kFloat32`, `kLT`
- All lowercase — e.g., `treelite`, `treelite::gtil`, `treelite::model_builder`, `treelite::c_api`
- Nested namespaces use `::` syntax when declaring: `namespace treelite::model_builder {`
- Anonymous namespaces used for file-local helpers in `.cc` files
- `TREELITE_` prefix + `SCREAMING_SNAKE_CASE` — e.g., `TREELITE_CHECK`, `TREELITE_LOG`, `TREELITE_DLL_EXPORT`

### Code Style

- `BasedOnStyle: Google`
- `IndentWidth: 2`, `TabWidth: 2`, `UseTab: Never`
- `ColumnLimit: 100`
- `PointerAlignment: Left` (i.e., `int* ptr`, not `int *ptr`)
- `QualifierAlignment: Right` (i.e., `int const&`, not `const int&`)
- `InsertBraces: true` — braces mandatory even for single-line if/loops
- `AllowShortIfStatementsOnASingleLine: Never`

### Comments

- Doxygen style with `/*!` block openers and `\brief`, `\param`, `\note` tags
- Used for all public API methods in headers
- `// comment text` — always a space after `//`
- `//!<` for trailing member documentation

### Error Handling

- Fatal errors: throw `treelite::Error` (subclasses `std::runtime_error`) via `TREELITE_CHECK*` macros
- `TREELITE_CHECK(condition)` — throws with `"Check failed: <expr>: "` prefix
- `TREELITE_CHECK_EQ/LT/GT/LE/GE/NE(x, y)` — binary comparison checks with value printing
- `TREELITE_LOG(FATAL)` — also throws via `LogMessageFatal` destructor
- `TREELITE_LOG(INFO)` / `TREELITE_LOG(WARNING)` — routes through callback registry
- C API layer: all functions return `int` (0 = success, -1 = error); last error stored via `TreeliteGetLastError()`

### Logging

- `TREELITE_LOG(INFO) << "message";`
- `TREELITE_LOG(WARNING) << "message";`
- `TREELITE_LOG(FATAL) << "message";` — throws `Error`
- Log messages include file/line prefix automatically
- Callbacks are thread-local and can be overridden (used by Python bindings to redirect to `print`/`warnings.warn`)

### Python Conventions (`treelite-mainline/python/`)

- `snake_case` for all functions and variables
- `PascalCase` for classes — e.g., `Metadata`, `TreeAnnotation`, `ModelBuilder`
- Private/internal helpers prefixed with `_` — e.g., `_load_lib()`, `_log_callback`, `_check_call`
- C API call wrappers use `_LIB.<TreeliteFunctionName>(...)`
- `@dataclasses.dataclass` for value objects (e.g., `Metadata`, `TreeAnnotation`, `PostProcessorFunc`)
- Type annotations required on all dataclass fields
- `asdict()` method provided for JSON serialization
- NumPy style, with `Parameters` and `Returns` sections
- All public classes and functions documented
- `TreeliteError` (subclass of `Exception`) raised when C API returns non-zero
- Pattern: `_check_call(lib.TreeliteSomeFunc(...))` — `_check_call` raises `TreeliteError` on failure

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| Rust crate (stub) | Project entry point; porting work lives here | `src/main.rs` |
| `Model` | Central in-memory tree ensemble object; owns variant over typed presets | `include/treelite/tree.h` |
| `ModelPreset<T,L>` | Typed container for a `Vec<Tree<T,L>>`; two concrete variants: `<float,float>` and `<double,double>` | `include/treelite/tree.h` |
| `Tree<T,L>` | Single decision tree; stores all node fields as parallel `ContiguousArray` columns | `include/treelite/tree.h`, `include/treelite/detail/tree.h` |
| `ContiguousArray<T>` | Owned or externally-borrowed flat buffer; primary storage primitive | `include/treelite/contiguous_array.h` |
| `ModelBuilder` | Builder pattern API for constructing `Model` objects node-by-node | `include/treelite/model_builder.h`, `src/model_builder/model_builder.cc` |
| Model loaders | Parse XGBoost (legacy binary, JSON, UBJSON), LightGBM, scikit-learn formats into `Model` | `src/model_loader/`, `include/treelite/model_loader.h` |
| GTIL | Reference prediction engine; supports dense and sparse CSR input, four prediction kinds | `include/treelite/gtil.h`, `src/gtil/` |
| Serializer | Binary and JSON round-trip for `Model`; supports three format generations (v3.9, v4.0, ≥5.0) | `src/serializer.cc`, `src/json_serializer.cc`, `include/treelite/detail/serializer.h` |
| C API | C-language extern wrapper over all subsystems; used by Python bindings | `include/treelite/c_api.h`, `src/c_api/` |
| Enums | `TaskType`, `TreeNodeType`, `Operator`, `TypeInfo` — shared vocabulary across all layers | `include/treelite/enum/` |

## Pattern Overview

- `Tree<T,L>` stores all node fields as separate parallel `ContiguousArray` columns (not a node struct), enabling cache-friendly traversal and zero-copy serialization.
- `Model` holds a `std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` (`ModelPresetVariant`) and dispatches all operations via `std::visit`.
- Template instantiations are explicit (`extern template`) to keep link times manageable.
- Serialization uses a mixin-based Serializer/Deserializer template in `detail/serializer.h` and `detail/serializer_mixins.h`; backward compatibility spans three wire format generations.
- GTIL prediction is multi-threaded via OpenMP utilities in `detail/threading_utils.h`.

## Layers

- Purpose: In-memory tree ensemble representation.
- Location: `include/treelite/tree.h`, `include/treelite/detail/tree.h`, `include/treelite/contiguous_array.h`
- Contains: `Tree<T,L>`, `ModelPreset<T,L>`, `Model`, `ContiguousArray<T>`, `Version`
- Depends on: Enum types in `include/treelite/enum/`
- Used by: All other layers
- Purpose: Shared constants (task kind, node kind, comparison operator, numeric type).
- Location: `include/treelite/enum/`, `src/enum/`
- Contains: `TaskType`, `TreeNodeType`, `Operator`, `TypeInfo`
- Depends on: Nothing
- Used by: Core model, model builder, model loaders, GTIL, serializer
- Purpose: Programmatic construction of `Model` objects via a fluent Begin/End node API.
- Location: `include/treelite/model_builder.h`, `src/model_builder/`
- Contains: `ModelBuilder` interface, `Metadata`, `TreeAnnotation`, `PostProcessorFunc`
- Depends on: Core model layer, enum layer
- Used by: Model loaders, C API, tests
- Purpose: Parsing external model formats (XGBoost, LightGBM, scikit-learn) into `Model`.
- Location: `include/treelite/model_loader.h`, `src/model_loader/`
- Contains: XGBoost (legacy binary, JSON via SAX, UBJSON), LightGBM, sklearn loaders
- Depends on: Model builder layer, core model layer
- Used by: C API, Python bindings
- Purpose: Binary and JSON round-trip persistence of `Model` objects; multi-version support.
- Location: `src/serializer.cc`, `src/json_serializer.cc`, `include/treelite/detail/serializer.h`, `include/treelite/detail/serializer_mixins.h`
- Contains: Mixin-based `Serializer<MixIn>` / `Deserializer<MixIn>` templates; v3/v4/v5 wire format handling
- Depends on: Core model layer
- Used by: C API, `Model::SerializeToPyBuffer`, `Model::SerializeToStream`, `Model::SerializeToBuffer`
- Purpose: Pure-C++ reference prediction implementation over a loaded `Model`.
- Location: `include/treelite/gtil.h`, `src/gtil/`
- Contains: `Predict<T>`, `PredictSparse<T>`, `GetOutputShape`, `Configuration`, postprocessors, output shape calculation
- Depends on: Core model layer, threading utilities
- Used by: C API (`src/c_api/gtil.cc`), Python GTIL wrapper
- Purpose: Stable C ABI for use by Python and other language bindings.
- Location: `include/treelite/c_api.h`, `include/treelite/c_api_error.h`, `src/c_api/`
- Contains: Extern C wrappers for model loading, building, serialization, field access, GTIL prediction, sklearn loading
- Depends on: All layers above
- Used by: Python bindings (`treelite-mainline/python/`)

## Data Flow

### Load External Model and Predict

### Serialize and Deserialize a Model

### Build a Model Programmatically

- `ModelBuilder` is single-threaded; parallel model construction is done by building multiple `Model` objects and concatenating with `ConcatenateModelObjects` (`src/model_concat.cc`).
- `Tree<T,L>` fields are non-copyable (`= delete`); explicit `Clone()` is provided.
- `ContiguousArray<T>` may own its buffer or alias an external one (via `UseForeignBuffer`).

## Key Abstractions

- Purpose: Single handle for any tree ensemble regardless of numeric type
- File: `include/treelite/tree.h`
- Pattern: Holds `ModelPresetVariant` (a `std::variant`); dispatches via `std::visit`
- Purpose: All node fields stored as parallel flat arrays indexed by node ID
- File: `include/treelite/tree.h`, `include/treelite/detail/tree.h`
- Pattern: Struct-of-Arrays; getter/setter methods for each field; no public node struct
- Purpose: Unified array primitive supporting both owned heap allocations and zero-copy foreign buffers (for Python buffer protocol)
- File: `include/treelite/contiguous_array.h`
- Pattern: Move-only; `UseForeignBuffer` for non-owning mode
- Purpose: Decouple parsing logic from model construction
- File: `include/treelite/model_builder.h`
- Pattern: Abstract interface with Begin/End pairing; `CommitModel()` finalizes
- Purpose: Reusable serialization logic with pluggable I/O backends (stream vs PyBuffer vs in-memory buffer)
- File: `include/treelite/detail/serializer.h`, `include/treelite/detail/serializer_mixins.h`
- Pattern: CRTP-like mixin; `Serializer<MixIn>` / `Deserializer<MixIn>`

## Entry Points

- Location: `src/main.rs`
- Current state: Stub (`fn main() { println!("Hello, world!"); }`). This is where Rust port implementation begins.
- Triggers: `cargo run` / `cargo build`
- Location: `include/treelite/c_api.h`, `src/c_api/`
- Triggers: Shared library load by Python or other FFI consumers
- Location: `treelite-mainline/python/treelite/`
- Triggers: `import treelite`; delegates to C API

## Architectural Constraints

- **Threading:** GTIL uses OpenMP (`detail/threading_utils.h`); `ModelBuilder` must be accessed from a single thread.
- **Global state:** Thread-local error storage via `include/treelite/thread_local.h`; logging via `include/treelite/logging.h` and `src/logging.cc`.
- **Type parameterization:** Only two concrete `ModelPreset` specializations exist: `<float, float>` and `<double, double>`. Mixed threshold/leaf types are not supported.
- **Circular imports:** None detected in C++ reference; standard include-guard discipline used throughout.
- **Rust crate:** Zero external dependencies declared in `Cargo.toml` at time of analysis; all porting work is greenfield.
- **Serialization versions:** Wire format compatibility must be maintained across three generations (v3.9, v4.0, ≥5.0); see compatibility matrix in `include/treelite/tree.h` lines 492–500.

## Anti-Patterns

### Bypassing `ModelBuilder` for direct `Tree<T,L>` mutation

### Copying `Model` or `Tree<T,L>`

## Error Handling

- Internal: `TREELITE_CHECK(condition) << "message"` — throws or aborts on failure
- C API boundary: all C API functions catch exceptions, store message in thread-local, return error code
- Thread-local error buffer: `include/treelite/thread_local.h`

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->

# Coding Conventions

**Analysis Date:** 2026-06-09

## Overview

This repo has two codebases: the Rust crate (`src/`) which is a minimal scaffold being
developed, and the upstream C++ reference (`treelite-mainline/`) being ported. Conventions
below cover both, with clear labeling.

---

## Rust Crate (`src/`)

The Rust crate is at earliest-stage scaffolding (`src/main.rs` is 3 lines). Rust
edition 2024 is declared in `Cargo.toml`. No linting config (`.clippy.toml`, `rustfmt.toml`)
is present yet, so standard Rust community defaults apply.

### Naming Patterns

**Files:**
- Snake\_case for all Rust source files (standard Rust convention, not yet tested at scale)

**Functions:**
- `snake_case` for all functions (Rust standard)

**Types/Structs/Enums:**
- `PascalCase` for all types (Rust standard)

**Constants/Statics:**
- `SCREAMING_SNAKE_CASE` (Rust standard)

**Modules:**
- `snake_case` directory/file names

### Code Style

**Formatting:**
- `rustfmt` with default settings (no custom config present)
- Run: `cargo fmt`

**Linting:**
- `clippy` with default settings (no custom config present)
- Run: `cargo clippy`

### Error Handling

- No error-handling patterns established yet (crate is a stub)
- Port target uses `throw treelite::Error(...)` on fatal conditions — translate to Rust `Result<T, E>` with a custom error enum

### Module Design

- No module structure yet beyond `fn main()` in `src/main.rs`
- Cargo edition 2024 is declared; use `mod` declarations as modules are added

---

## C++ Reference Codebase (`treelite-mainline/`)

This is the upstream treelite C++ project. Read these conventions to understand what is
being ported to Rust.

### Naming Patterns

**Files:**
- `snake_case.cc` / `snake_case.h` for implementation and header files
- Header guards: `TREELITE_<PATH>_H_` (all caps, path separators become underscores, trailing underscore)
  - Example: `include/treelite/tree.h` → `#ifndef TREELITE_TREE_H_`
- Test files: `test_<component>.cc` in `tests/cpp/`

**Classes/Structs:**
- `PascalCase` — e.g., `ModelBuilder`, `LogMessageFatal`, `ReturnValueEntry`

**Methods:**
- `PascalCase` for public API methods — e.g., `StartTree()`, `EndNode()`, `NumericalTest()`
- `snake_case` for private member variables with trailing underscore — e.g., `log_stream_`, `node_type_`

**Template Parameters:**
- Descriptive `PascalCase` suffixed with type meaning — e.g., `ThresholdType`, `LeafOutputType`, `ThresholdT`, `LeafOutputT`

**Enums and Enum Values:**
- Enum types in `PascalCase` — e.g., `TaskType`, `TypeInfo`, `Operator`
- Enum values prefixed with `k` — e.g., `kBinaryClf`, `kFloat32`, `kLT`

**Namespaces:**
- All lowercase — e.g., `treelite`, `treelite::gtil`, `treelite::model_builder`, `treelite::c_api`
- Nested namespaces use `::` syntax when declaring: `namespace treelite::model_builder {`
- Anonymous namespaces used for file-local helpers in `.cc` files

**Macros:**
- `TREELITE_` prefix + `SCREAMING_SNAKE_CASE` — e.g., `TREELITE_CHECK`, `TREELITE_LOG`, `TREELITE_DLL_EXPORT`

### Code Style

**Formatter:** `clang-format` (config: `treelite-mainline/.clang-format`)

**Key settings:**
- `BasedOnStyle: Google`
- `IndentWidth: 2`, `TabWidth: 2`, `UseTab: Never`
- `ColumnLimit: 100`
- `PointerAlignment: Left` (i.e., `int* ptr`, not `int *ptr`)
- `QualifierAlignment: Right` (i.e., `int const&`, not `const int&`)
- `InsertBraces: true` — braces mandatory even for single-line if/loops
- `AllowShortIfStatementsOnASingleLine: Never`

**Include ordering** (four groups, separated by blank lines):
1. Standard library headers in `<>` without extension
2. Treelite project headers `<treelite/...>`
3. External library headers `<rapidjson/...>`, `<nlohmann/...>`, `<gtest/...>`, `<fmt/...>`
4. Relative local headers in `"..."`

### Comments

**File headers:**
```cpp
/*!
 * Copyright (c) [year] by Contributors
 * \file [filename]
 * \brief [brief description]
 * \author [author]
 */
```

**Doc comments:**
- Doxygen style with `/*!` block openers and `\brief`, `\param`, `\note` tags
- Used for all public API methods in headers

**Inline comments:**
- `// comment text` — always a space after `//`
- `//!<` for trailing member documentation

### Error Handling

**C++ strategy:**
- Fatal errors: throw `treelite::Error` (subclasses `std::runtime_error`) via `TREELITE_CHECK*` macros
- `TREELITE_CHECK(condition)` — throws with `"Check failed: <expr>: "` prefix
- `TREELITE_CHECK_EQ/LT/GT/LE/GE/NE(x, y)` — binary comparison checks with value printing
- `TREELITE_LOG(FATAL)` — also throws via `LogMessageFatal` destructor
- `TREELITE_LOG(INFO)` / `TREELITE_LOG(WARNING)` — routes through callback registry
- C API layer: all functions return `int` (0 = success, -1 = error); last error stored via `TreeliteGetLastError()`

**C API pattern:**
```cpp
// Every C API function body is wrapped:
API_BEGIN();
  // ... logic
  TREELITE_CHECK_EQ(...);
API_END();
```

### Logging

**C++ framework:** Custom macros wrapping `LogMessage`, `LogMessageWarning`, `LogMessageFatal`

**Patterns:**
- `TREELITE_LOG(INFO) << "message";`
- `TREELITE_LOG(WARNING) << "message";`
- `TREELITE_LOG(FATAL) << "message";` — throws `Error`
- Log messages include file/line prefix automatically
- Callbacks are thread-local and can be overridden (used by Python bindings to redirect to `print`/`warnings.warn`)

### Python Conventions (`treelite-mainline/python/`)

**Formatter:** `black` (via pre-commit; flake8 E501 and W503 deferred to black)
**Linter:** `flake8` (config: `.flake8`); `pylint` (config: `python/.pylintrc`)
**Import sorter:** `isort` (config: `.isort.cfg`)

**Naming:**
- `snake_case` for all functions and variables
- `PascalCase` for classes — e.g., `Metadata`, `TreeAnnotation`, `ModelBuilder`
- Private/internal helpers prefixed with `_` — e.g., `_load_lib()`, `_log_callback`, `_check_call`
- C API call wrappers use `_LIB.<TreeliteFunctionName>(...)`

**Data classes:**
- `@dataclasses.dataclass` for value objects (e.g., `Metadata`, `TreeAnnotation`, `PostProcessorFunc`)
- Type annotations required on all dataclass fields
- `asdict()` method provided for JSON serialization

**Docstrings:**
- NumPy style, with `Parameters` and `Returns` sections
- All public classes and functions documented

**Error handling:**
- `TreeliteError` (subclass of `Exception`) raised when C API returns non-zero
- Pattern: `_check_call(lib.TreeliteSomeFunc(...))` — `_check_call` raises `TreeliteError` on failure

---

*Convention analysis: 2026-06-09*

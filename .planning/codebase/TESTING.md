# Testing Patterns

**Analysis Date:** 2026-06-09

## Overview

The Rust crate (`src/`) has no tests yet — it is a stub. All testing infrastructure exists
in the upstream C++ reference (`treelite-mainline/`). Port target testing patterns are
documented here to guide Rust test design.

---

## Rust Crate (`src/`)

### Test Framework

**Runner:** Rust built-in test harness (no additional crate in `Cargo.toml` yet)

**Run Commands:**
```bash
cargo test              # Run all tests
cargo test -- --nocapture  # Show stdout
cargo test <filter>     # Run matching tests
```

### Current State

No tests exist. `src/main.rs` is a 3-line hello-world stub. As porting progresses,
follow the patterns below (adapted from the C++ reference) and standard Rust idioms:

**Unit tests:** Co-located in the source file in a `#[cfg(test)]` module
```rust
// src/some_module.rs
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}
```

**Integration tests:** Place in `tests/` directory at crate root (to be created)

**Error testing:**
```rust
#[test]
fn test_invalid_input() {
    let result = some_fallible_fn(-1);
    assert!(result.is_err());
}
```

---

## C++ Reference (`treelite-mainline/`)

### Test Framework

**Runner:** Google Test (gtest)
- Config: `tests/cpp/CMakeLists.txt`
- Entry point: `tests/cpp/test_main.cc` — calls `testing::InitGoogleTest` and `RUN_ALL_TESTS()`
- Death test style set to `"threadsafe"`: `testing::FLAGS_gtest_death_test_style = "threadsafe"`

**Assertion Library:** Google Test macros (`EXPECT_*`, `ASSERT_*`)

**External libraries used in tests:**
- `<gtest/gtest.h>` — test framework
- `<rapidjson/document.h>` — for JSON assertion helpers
- `<fmt/format.h>` — string formatting in test output

**Run Commands:**
```bash
# After CMake build:
ctest                           # Run all tests via CTest
./build/treelite_cpp_tests      # Run gtest binary directly
```

### Python Test Framework

**Runner:** pytest
- Test files: `tests/python/test_*.py`
- Hypothesis integration for property-based tests: `tests/python/hypothesis_util.py`

**Run Commands:**
```bash
pytest tests/python/            # Run all Python tests
pytest tests/python/test_model_builder.py  # Run specific file
```

### Test File Organization

**C++ tests:**
- Location: `tests/cpp/` — all in one flat directory
- Naming: `test_<component>.cc`
- Files: `test_gtil.cc`, `test_model_builder.cc`, `test_model_concat.cc`, `test_model_loader.cc`, `test_serializer.cc`, `test_utils.cc`

**Python tests:**
- Location: `tests/python/`
- Naming: `test_<component>.py`
- Shared utilities: `tests/python/util.py`, `tests/python/hypothesis_util.py`, `tests/python/metadata.py`

**Test fixtures/data:**
- Binary model files: `tests/examples/` (e.g., `mushroom.model`, `toy_categorical_model.txt`)

### C++ Test Structure

**Suite and test naming:**
```cpp
// Standard test: TEST(SuiteName, TestName)
TEST(ModelBuilder, OrphanedNodes) { ... }
TEST(FileUtils, StreamIO) { ... }

// Parameterized test: TEST_P(SuiteName, TestName)
class ParametrizedTestSuite : public testing::TestWithParam<std::string> {};
TEST_P(ParametrizedTestSuite, MulticlassClfGrovePerClass) { ... }
INSTANTIATE_TEST_SUITE_P(GTIL, ParametrizedTestSuite, testing::Values("dense", "sparse"));
```

**File-local helpers** placed in anonymous namespace before the `treelite` namespace:
```cpp
namespace {
void AssertDocumentValid(rapidjson::Document const& doc) { ... }
void AssertJSONStringsEqual(std::string const& actual, std::string const& expected) { ... }
}  // anonymous namespace

namespace treelite {
TEST(ModelBuilder, ...) { ... }
}  // namespace treelite
```

**Exception testing:**
```cpp
EXPECT_THROW(builder->EndTree(), treelite::Error);
EXPECT_THROW(builder->StartNode(-1), treelite::Error);
```

**Assertion preference:**
- Use `ASSERT_TRUE(a == b)` over `ASSERT_EQ(a, b)` when values are large (e.g., raw byte buffers) to avoid OOM from gtest's diff printing
- Use `EXPECT_*` (non-fatal) when test can continue; `ASSERT_*` (fatal) when failure makes continuation meaningless

### Mocking

**Framework:** Not used — no gmock or other mock library detected
**Approach:** Tests construct real objects using the `ModelBuilder` API and verify end-to-end behavior
**What to test directly:**
- Builder state machine (invalid states throw `treelite::Error`)
- Serialization round-trips (serialize then deserialize, compare JSON dumps)
- Prediction correctness (build minimal tree, verify output values)

### Python Test Patterns

**Standard test:**
```python
def test_orphaned_nodes():
    """Test for orphaned nodes"""
    builder = ModelBuilder(...)
    builder.start_tree()
    ...
    with pytest.raises(TreeliteError):
        builder.end_tree()
```

**Property-based tests (Hypothesis):**
```python
from hypothesis import given
from tests.python.hypothesis_util import standard_classification_datasets

@given(standard_classification_datasets())
def test_sklearn_roundtrip(dataset):
    ...
```

**Shared fixture helpers:**
- `tests/python/util.py` — array construction utilities, tolerance comparisons
- `tests/python/hypothesis_util.py` — `@composite` strategies for generating classification/regression datasets
- `tests/python/metadata.py` — shared metadata constants

### Serializer Compatibility Tests

**Script:** `tests/serializer/test_serializer.py` and `tests/serializer/compatibility_tester.py`
**Purpose:** Cross-version serialization round-trip compatibility
**Run via:** `ops/test-serializer-compatibility.sh`

### C++ Round-Trip Testing Pattern

```cpp
inline void TestRoundTrip(treelite::Model* model) {
  // In-memory (PyBuffer)
  auto buffer = model->SerializeToPyBuffer();
  auto received = treelite::Model::DeserializeFromPyBuffer(buffer);
  ASSERT_TRUE(model->DumpAsJSON(false) == received->DumpAsJSON(false));

  // In-memory (stream)
  std::ostringstream oss;
  model->SerializeToStream(oss);
  std::istringstream iss(oss.str());
  auto received2 = treelite::Model::DeserializeFromStream(iss);
  ASSERT_TRUE(model->DumpAsJSON(false) == received2->DumpAsJSON(false));

  // File (path)
  // ... similar pattern with temporary filesystem path
}
```

### Coverage

**C++ requirements:** Not enforced via config — coverage run via shell scripts:
- `ops/cpp-python-coverage.sh`
- `ops/macos-python-coverage.sh`

**Python requirements:** Not enforced in config

### CI Test Scripts

Located in `ops/` and `tests/ci_build/`:
- `ops/build-linux.sh`, `ops/build-macos.sh`, `ops/build-windows.bat` — build scripts
- `ops/test-linux-python-wheel.sh` — test Python wheel on Linux
- `ops/test-cmake-import.sh` — verify CMake import works
- `ops/test-sdist.sh` — test source distribution
- `tests/ci_build/ci_build.sh` — CI entry point
- `tests/ci_build/build_via_cmake.sh` — CMake build helper

---

*Testing analysis: 2026-06-09*

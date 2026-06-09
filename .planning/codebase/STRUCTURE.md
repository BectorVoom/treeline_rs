# Codebase Structure

**Analysis Date:** 2026-06-09

## Directory Layout

```
treeline_rs/                        # Repository root
├── Cargo.toml                      # Rust crate manifest (no external deps yet)
├── src/
│   └── main.rs                     # Rust entry point (stub — porting starts here)
├── treelite-mainline/              # Upstream C++ Treelite reference (read-only source of truth)
│   ├── include/
│   │   └── treelite/
│   │       ├── tree.h              # Core: Model, ModelPreset<T,L>, Tree<T,L>
│   │       ├── contiguous_array.h  # Core: ContiguousArray<T> flat buffer
│   │       ├── model_builder.h     # Builder API: ModelBuilder, Metadata, TreeAnnotation
│   │       ├── model_loader.h      # Loader API: XGBoost, LightGBM, sklearn
│   │       ├── gtil.h              # Inference API: Predict, PredictSparse, GetOutputShape
│   │       ├── c_api.h             # C ABI entry points
│   │       ├── c_api_error.h       # Error handling for C API
│   │       ├── pybuffer_frame.h    # Python buffer protocol frame struct
│   │       ├── contiguous_array.h  # Flat buffer primitive
│   │       ├── logging.h           # Logging macros
│   │       ├── error.h             # Error types
│   │       ├── thread_local.h      # Thread-local storage
│   │       ├── enum/
│   │       │   ├── operator.h      # Enum: Operator (comparison ops)
│   │       │   ├── task_type.h     # Enum: TaskType (classifier/regressor/etc)
│   │       │   ├── tree_node_type.h # Enum: TreeNodeType (leaf/numerical/categorical)
│   │       │   └── typeinfo.h      # Enum: TypeInfo (float32/float64)
│   │       └── detail/
│   │           ├── tree.h          # Inline implementations for Tree<T,L>
│   │           ├── serializer.h    # Mixin serializer building blocks
│   │           ├── serializer_mixins.h  # Concrete mixin backends
│   │           ├── contiguous_array.h   # Inline implementations for ContiguousArray
│   │           ├── file_utils.h    # File I/O helpers
│   │           ├── omp_exception.h # OpenMP exception wrapper
│   │           └── threading_utils.h   # Thread pool / config
│   ├── src/
│   │   ├── c_api/
│   │   │   ├── model.cc            # C API: model lifecycle
│   │   │   ├── model_builder.cc    # C API: builder wrappers
│   │   │   ├── model_loader.cc     # C API: loader wrappers
│   │   │   ├── gtil.cc             # C API: prediction wrappers
│   │   │   ├── serializer.cc       # C API: serialization wrappers
│   │   │   ├── field_accessor.cc   # C API: field get/set
│   │   │   ├── sklearn.cc          # C API: sklearn loaders
│   │   │   ├── logging.cc          # C API: logging callback
│   │   │   └── c_api_utils.h       # Internal C API helpers
│   │   ├── enum/
│   │   │   ├── operator.cc         # Enum string conversions
│   │   │   ├── task_type.cc
│   │   │   ├── tree_node_type.cc
│   │   │   └── typeinfo.cc
│   │   ├── gtil/
│   │   │   ├── predict.cc          # Inference engine (dense + sparse CSR)
│   │   │   ├── postprocessor.cc    # Post-processing functions (sigmoid, softmax, etc)
│   │   │   ├── postprocessor.h     # Internal postprocessor header
│   │   │   ├── output_shape.cc     # Output buffer shape calculation
│   │   │   └── config.cc           # Configuration parsing from JSON
│   │   ├── model_builder/
│   │   │   ├── model_builder.cc    # ModelBuilder implementation
│   │   │   ├── metadata.cc         # Metadata / TreeAnnotation / PostProcessorFunc impl
│   │   │   └── detail/
│   │   │       └── json_parsing.h  # JSON parsing helpers for builder
│   │   ├── model_loader/
│   │   │   ├── xgboost_json.cc     # XGBoost JSON format loader
│   │   │   ├── xgboost_legacy.cc   # XGBoost legacy binary format loader
│   │   │   ├── xgboost_ubjson.cc   # XGBoost UBJSON format loader
│   │   │   ├── lightgbm.cc         # LightGBM text format loader
│   │   │   ├── sklearn.cc          # scikit-learn array-based loaders
│   │   │   ├── sklearn_bulk.cc     # Bulk sklearn tree construction (BulkConstructTree)
│   │   │   └── detail/
│   │   │       ├── lightgbm.h      # LightGBM internal parse helpers
│   │   │       ├── string_utils.h  # String utilities for parsers
│   │   │       ├── xgboost.cc/.h   # XGBoost shared parsing logic
│   │   │       └── xgboost_json/
│   │   │           ├── delegated_handler.cc/.h   # SAX handler delegation
│   │   │           └── sax_adapters.cc/.h         # SAX to ModelBuilder adapters
│   │   ├── serializer.cc           # Binary serializer (multi-version)
│   │   ├── json_serializer.cc      # JSON serializer (DumpAsJSON)
│   │   ├── field_accessor.cc       # PyBuffer field accessor for Model/Tree
│   │   ├── model_concat.cc         # ConcatenateModelObjects
│   │   ├── model_query.cc          # GetTreeDepth and other model queries
│   │   └── logging.cc              # Logging implementation
│   ├── python/
│   │   └── treelite/               # Python package (wraps C API)
│   │       ├── gtil/               # Python GTIL prediction wrappers
│   │       └── sklearn/            # Python sklearn integration
│   ├── tests/
│   │   ├── cpp/                    # C++ unit/integration tests
│   │   ├── python/                 # Python tests
│   │   ├── examples/               # Example models (mushroom, lightgbm, etc)
│   │   └── serializer/             # Serializer round-trip tests
│   └── cmake/                      # CMake build configuration
├── .planning/
│   └── codebase/                   # GSD planning documents (this directory)
└── .serena/                        # Serena agent cache/memories
```

## Directory Purposes

**`src/` (Rust):**
- Purpose: Rust port implementation. Currently a stub.
- Contains: `main.rs` only
- Key files: `src/main.rs`

**`treelite-mainline/include/treelite/` (C++ headers):**
- Purpose: Public API surface and core data model definitions for the upstream library. Primary reference for all porting decisions.
- Contains: All public headers, enum definitions, template implementations (in `detail/`)
- Key files: `include/treelite/tree.h`, `include/treelite/model_builder.h`, `include/treelite/gtil.h`

**`treelite-mainline/src/` (C++ implementations):**
- Purpose: Implementation files for the upstream library, organized by subsystem.
- Contains: Model loaders, builder, GTIL inference, serialization, C API, enum string conversions
- Key files: `src/gtil/predict.cc`, `src/serializer.cc`, `src/model_builder/model_builder.cc`

**`treelite-mainline/python/` (Python bindings):**
- Purpose: Python package wrapping the C API; not part of the Rust port but useful for understanding expected external behavior.
- Contains: `treelite/` package, `gtil/` submodule, `sklearn/` submodule, packager tooling

**`treelite-mainline/tests/` (C++ and Python tests):**
- Purpose: Integration and unit tests for the upstream library; serve as reference test cases for the Rust port.
- Contains: C++ tests, Python tests, serializer round-trip tests, example model files

**`.planning/codebase/` (GSD planning):**
- Purpose: Architecture and convention documents consumed by GSD planning and execution commands.
- Generated: Yes (by `/gsd-map-codebase`)
- Committed: Yes (tracks project context)

## Key File Locations

**Entry Points:**
- `src/main.rs`: Rust crate entry point (current stub)
- `treelite-mainline/include/treelite/c_api.h`: C ABI entry point (upstream reference)

**Core Data Model (upstream reference):**
- `treelite-mainline/include/treelite/tree.h`: `Model`, `ModelPreset<T,L>`, `Tree<T,L>`, `Version`
- `treelite-mainline/include/treelite/contiguous_array.h`: `ContiguousArray<T>`
- `treelite-mainline/include/treelite/detail/tree.h`: Inline method bodies for `Tree<T,L>`

**Builder API (upstream reference):**
- `treelite-mainline/include/treelite/model_builder.h`: `ModelBuilder`, `Metadata`, `TreeAnnotation`, `PostProcessorFunc`
- `treelite-mainline/src/model_builder/model_builder.cc`: Implementation

**Inference Engine (upstream reference):**
- `treelite-mainline/include/treelite/gtil.h`: `Predict<T>`, `PredictSparse<T>`, `Configuration`, `PredictKind`
- `treelite-mainline/src/gtil/predict.cc`: Implementation with `DenseMatrixAccessor` and `SparseMatrixAccessor`
- `treelite-mainline/src/gtil/postprocessor.cc`: Postprocessors (sigmoid, softmax, etc.)

**Serialization (upstream reference):**
- `treelite-mainline/src/serializer.cc`: Binary format with multi-version (v3.9/v4.0/v5.0) support
- `treelite-mainline/src/json_serializer.cc`: JSON dump
- `treelite-mainline/include/treelite/detail/serializer.h`: Mixin templates

**Enums (upstream reference):**
- `treelite-mainline/include/treelite/enum/task_type.h`: `TaskType` (5 variants)
- `treelite-mainline/include/treelite/enum/tree_node_type.h`: `TreeNodeType` (leaf, numerical, categorical)
- `treelite-mainline/include/treelite/enum/operator.h`: `Operator` (comparison operators)
- `treelite-mainline/include/treelite/enum/typeinfo.h`: `TypeInfo` (float32 / float64)

**Configuration:**
- `Cargo.toml`: Rust manifest (no deps currently)
- `treelite-mainline/cmake/`: CMake build config for the C++ library

## Naming Conventions

**C++ (upstream reference):**
- Files: `snake_case.cc` / `snake_case.h`
- Classes: `PascalCase` (e.g., `ModelBuilder`, `ContiguousArray`, `TreeAnnotation`)
- Methods: `PascalCase` (e.g., `StartTree`, `LeafScalar`, `CommitModel`)
- Enums: `PascalCase` with `k` prefix for values (e.g., `TaskType::kBinaryClf`, `PredictKind::kPredictDefault`)
- Namespaces: `snake_case` (e.g., `treelite`, `treelite::gtil`, `treelite::model_builder`)
- Template type params: `ThresholdType`, `LeafOutputType` (descriptive)
- Member fields (private): `snake_case_` with trailing underscore

**Rust (crate — to be established):**
- Files: `snake_case.rs` (standard Rust convention)
- Types: `PascalCase`
- Functions/methods: `snake_case`
- Modules: `snake_case`
- Enum variants: `PascalCase` (avoid `k` prefix — use Rust idiom instead of C++ convention)

## Where to Add New Code

**New Rust module implementing a C++ subsystem:**
- Create `src/<subsystem_name>.rs` (e.g., `src/tree.rs`, `src/model_builder.rs`)
- Declare module in `src/main.rs` with `mod <subsystem_name>;`
- Reference the corresponding C++ header in `treelite-mainline/include/treelite/` for the porting spec

**New Rust data type (porting a C++ struct/class):**
- Place in the module file corresponding to its C++ header (e.g., `Tree` from `tree.h` → `src/tree.rs`)

**New Rust inference logic (porting GTIL):**
- Create `src/gtil.rs`; port from `treelite-mainline/src/gtil/predict.cc`

**New Rust serialization:**
- Create `src/serializer.rs`; reference `treelite-mainline/src/serializer.cc` and `treelite-mainline/include/treelite/detail/serializer.h`

**New Rust model loader:**
- Create `src/model_loader/` directory with per-format files (e.g., `src/model_loader/xgboost.rs`)

**External Rust dependencies:**
- Add to `[dependencies]` in `Cargo.toml`

## Special Directories

**`treelite-mainline/`:**
- Purpose: Upstream C++ Treelite project checked in as a reference (not compiled as part of the Rust crate build)
- Generated: No (vendored upstream source)
- Committed: Yes

**`.planning/codebase/`:**
- Purpose: GSD codebase map documents
- Generated: Yes
- Committed: Yes

**`.serena/`:**
- Purpose: Serena agent cache and memory files
- Generated: Yes
- Committed: As needed

---

*Structure analysis: 2026-06-09*

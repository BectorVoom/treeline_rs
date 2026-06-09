<!-- refreshed: 2026-06-09 -->
# Architecture

**Analysis Date:** 2026-06-09

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     treeline_rs (Rust Crate)                            │
│                     src/main.rs  — stub entry point                     │
└────────────────────────────────────────────────────────────────────────-┘

         ┌─────────────────────────────────────────────────────────┐
         │          treelite-mainline (C++ upstream reference)      │
         │          All subsystems below are the porting target.    │
         └─────────────────────────────────────────────────────────┘

┌──────────────────────┬──────────────────────┬────────────────────────┐
│   Model Loaders      │   Model Builder       │   Serialization        │
│ `src/model_loader/`  │ `src/model_builder/`  │ `src/serializer.cc`    │
│ `src/c_api/model_    │ `src/c_api/model_     │ `src/json_serializer.  │
│  loader.cc`          │  builder.cc`          │  cc`                   │
└──────────┬───────────┴──────────┬────────────┴────────────┬───────────┘
           │                      │                          │
           ▼                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Model (Core Data Model)                          │
│  `include/treelite/tree.h`  ·  Model / ModelPreset<T,L> / Tree<T,L>    │
│  `include/treelite/contiguous_array.h`  ·  ContiguousArray<T>           │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   GTIL — General Tree Inference Library                 │
│  `src/gtil/predict.cc`  ·  `src/gtil/postprocessor.cc`                 │
│  `src/gtil/output_shape.cc`  ·  `src/gtil/config.cc`                   │
│  `include/treelite/gtil.h`                                              │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        C API (public FFI surface)                       │
│  `src/c_api/`  ·  `include/treelite/c_api.h`                           │
└─────────────────────────────────────────────────────────────────────────┘
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

**Overall:** Struct-of-Arrays (SoA) tree representation inside a type-parameterized variant, exposed through a type-erased `Model` object and a fluent `ModelBuilder` interface. The C API provides a stable ABI over the C++ internals.

**Key Characteristics:**
- `Tree<T,L>` stores all node fields as separate parallel `ContiguousArray` columns (not a node struct), enabling cache-friendly traversal and zero-copy serialization.
- `Model` holds a `std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` (`ModelPresetVariant`) and dispatches all operations via `std::visit`.
- Template instantiations are explicit (`extern template`) to keep link times manageable.
- Serialization uses a mixin-based Serializer/Deserializer template in `detail/serializer.h` and `detail/serializer_mixins.h`; backward compatibility spans three wire format generations.
- GTIL prediction is multi-threaded via OpenMP utilities in `detail/threading_utils.h`.

## Layers

**Core Model Layer:**
- Purpose: In-memory tree ensemble representation.
- Location: `include/treelite/tree.h`, `include/treelite/detail/tree.h`, `include/treelite/contiguous_array.h`
- Contains: `Tree<T,L>`, `ModelPreset<T,L>`, `Model`, `ContiguousArray<T>`, `Version`
- Depends on: Enum types in `include/treelite/enum/`
- Used by: All other layers

**Enum / Vocabulary Layer:**
- Purpose: Shared constants (task kind, node kind, comparison operator, numeric type).
- Location: `include/treelite/enum/`, `src/enum/`
- Contains: `TaskType`, `TreeNodeType`, `Operator`, `TypeInfo`
- Depends on: Nothing
- Used by: Core model, model builder, model loaders, GTIL, serializer

**Model Builder Layer:**
- Purpose: Programmatic construction of `Model` objects via a fluent Begin/End node API.
- Location: `include/treelite/model_builder.h`, `src/model_builder/`
- Contains: `ModelBuilder` interface, `Metadata`, `TreeAnnotation`, `PostProcessorFunc`
- Depends on: Core model layer, enum layer
- Used by: Model loaders, C API, tests

**Model Loader Layer:**
- Purpose: Parsing external model formats (XGBoost, LightGBM, scikit-learn) into `Model`.
- Location: `include/treelite/model_loader.h`, `src/model_loader/`
- Contains: XGBoost (legacy binary, JSON via SAX, UBJSON), LightGBM, sklearn loaders
- Depends on: Model builder layer, core model layer
- Used by: C API, Python bindings

**Serialization Layer:**
- Purpose: Binary and JSON round-trip persistence of `Model` objects; multi-version support.
- Location: `src/serializer.cc`, `src/json_serializer.cc`, `include/treelite/detail/serializer.h`, `include/treelite/detail/serializer_mixins.h`
- Contains: Mixin-based `Serializer<MixIn>` / `Deserializer<MixIn>` templates; v3/v4/v5 wire format handling
- Depends on: Core model layer
- Used by: C API, `Model::SerializeToPyBuffer`, `Model::SerializeToStream`, `Model::SerializeToBuffer`

**GTIL (Inference) Layer:**
- Purpose: Pure-C++ reference prediction implementation over a loaded `Model`.
- Location: `include/treelite/gtil.h`, `src/gtil/`
- Contains: `Predict<T>`, `PredictSparse<T>`, `GetOutputShape`, `Configuration`, postprocessors, output shape calculation
- Depends on: Core model layer, threading utilities
- Used by: C API (`src/c_api/gtil.cc`), Python GTIL wrapper

**C API Layer:**
- Purpose: Stable C ABI for use by Python and other language bindings.
- Location: `include/treelite/c_api.h`, `include/treelite/c_api_error.h`, `src/c_api/`
- Contains: Extern C wrappers for model loading, building, serialization, field access, GTIL prediction, sklearn loading
- Depends on: All layers above
- Used by: Python bindings (`treelite-mainline/python/`)

## Data Flow

### Load External Model and Predict

1. Caller invokes a model loader (`LoadXGBoostModelJSON`, `LoadLightGBMModel`, etc.) in `src/model_loader/`
2. Loader uses `ModelBuilder` (`src/model_builder/model_builder.cc`) — calls `StartTree`, `StartNode`, `NumericalTest`/`CategoricalTest`/`LeafScalar`, `EndNode`, `EndTree`, `CommitModel`
3. `CommitModel` returns `std::unique_ptr<Model>` with variant set to the appropriate `ModelPreset<T,L>`
4. Caller passes `Model` to `gtil::Predict<T>` (`src/gtil/predict.cc`) with a `Configuration`
5. GTIL traverses each `Tree<T,L>` column-by-column using `DenseMatrixAccessor` or `SparseMatrixAccessor`, dispatches to postprocessor (`src/gtil/postprocessor.cc`)
6. Results written into caller-allocated output buffer

### Serialize and Deserialize a Model

1. `Model::SerializeToPyBuffer()` or `Model::SerializeToStream()` calls the mixin-based `Serializer` in `src/serializer.cc`
2. Serializer iterates over each field of `Model` and all `Tree<T,L>` objects; emits PyBuffer frames or stream bytes
3. `Model::DeserializeFromPyBuffer()` / `Model::DeserializeFromStream()` reconstruct the model, handling version migration from wire format v3.9, v4.0, or ≥5.0 (compatibility matrix documented in `include/treelite/tree.h`)

### Build a Model Programmatically

1. Call `model_builder::GetModelBuilder(threshold_type, leaf_output_type, metadata, ...)` in `src/model_builder/model_builder.cc`
2. Call `StartTree()`, `StartNode(key)`, split/leaf methods, `EndNode()`, `EndTree()` for each tree
3. Call `CommitModel()` to obtain `std::unique_ptr<Model>`

**State Management:**
- `ModelBuilder` is single-threaded; parallel model construction is done by building multiple `Model` objects and concatenating with `ConcatenateModelObjects` (`src/model_concat.cc`).
- `Tree<T,L>` fields are non-copyable (`= delete`); explicit `Clone()` is provided.
- `ContiguousArray<T>` may own its buffer or alias an external one (via `UseForeignBuffer`).

## Key Abstractions

**`Model` (type-erased ensemble):**
- Purpose: Single handle for any tree ensemble regardless of numeric type
- File: `include/treelite/tree.h`
- Pattern: Holds `ModelPresetVariant` (a `std::variant`); dispatches via `std::visit`

**`Tree<ThresholdType, LeafOutputType>` (struct-of-arrays tree):**
- Purpose: All node fields stored as parallel flat arrays indexed by node ID
- File: `include/treelite/tree.h`, `include/treelite/detail/tree.h`
- Pattern: Struct-of-Arrays; getter/setter methods for each field; no public node struct

**`ContiguousArray<T>` (owned/borrowed flat buffer):**
- Purpose: Unified array primitive supporting both owned heap allocations and zero-copy foreign buffers (for Python buffer protocol)
- File: `include/treelite/contiguous_array.h`
- Pattern: Move-only; `UseForeignBuffer` for non-owning mode

**`ModelBuilder` (builder interface):**
- Purpose: Decouple parsing logic from model construction
- File: `include/treelite/model_builder.h`
- Pattern: Abstract interface with Begin/End pairing; `CommitModel()` finalizes

**Mixin Serializer:**
- Purpose: Reusable serialization logic with pluggable I/O backends (stream vs PyBuffer vs in-memory buffer)
- File: `include/treelite/detail/serializer.h`, `include/treelite/detail/serializer_mixins.h`
- Pattern: CRTP-like mixin; `Serializer<MixIn>` / `Deserializer<MixIn>`

## Entry Points

**Rust crate:**
- Location: `src/main.rs`
- Current state: Stub (`fn main() { println!("Hello, world!"); }`). This is where Rust port implementation begins.
- Triggers: `cargo run` / `cargo build`

**C++ C API (upstream reference):**
- Location: `include/treelite/c_api.h`, `src/c_api/`
- Triggers: Shared library load by Python or other FFI consumers

**Python bindings (upstream reference):**
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

**What happens:** Some sklearn bulk-load paths call `BulkConstructTree` as a `friend` function directly on `Tree<T,L>`, bypassing the `ModelBuilder` interface (`src/model_loader/sklearn_bulk.cc`).
**Why it's wrong:** Skips the validation and orphaned-node checks enforced by `ModelBuilder::EndTree`.
**Do this instead:** Use `ModelBuilder` for all tree construction unless you are implementing a high-performance bulk path that explicitly assumes validated input.

### Copying `Model` or `Tree<T,L>`

**What happens:** Copy constructors are `= delete` on both `Model` and `Tree<T,L>`.
**Why it's wrong:** These types contain `ContiguousArray` members that may alias external buffers; naive copying would produce dangling references.
**Do this instead:** Use `Tree<T,L>::Clone()` for explicit deep copies; use `ConcatenateModelObjects` (`src/model_concat.cc`) to merge models.

## Error Handling

**Strategy:** Macro-based `TREELITE_CHECK` / `TREELITE_LOG(FATAL)` assertions throughout C++ code; errors propagate through thread-local storage to the C API layer, which exposes `TreeliteGetLastError()`.

**Patterns:**
- Internal: `TREELITE_CHECK(condition) << "message"` — throws or aborts on failure
- C API boundary: all C API functions catch exceptions, store message in thread-local, return error code
- Thread-local error buffer: `include/treelite/thread_local.h`

## Cross-Cutting Concerns

**Logging:** `TREELITE_LOG(severity)` macro defined in `include/treelite/logging.h`; implementation in `src/logging.cc`
**Validation:** `ModelBuilder` performs orphan-node checking and topology validation on `EndTree`; controlled by `SetValidationFlag("check_orphaned_nodes", bool)`
**Authentication:** Not applicable — library has no network or auth surface

---

*Architecture analysis: 2026-06-09*

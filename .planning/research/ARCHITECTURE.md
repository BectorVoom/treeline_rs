# Architecture Research

**Domain:** Rust port of Treelite — tree-ensemble model library (load → build → in-memory SoA model → GTIL inference → serialize), Cargo workspace, cubecl-accelerated hot path, PyO3 binding
**Researched:** 2026-06-09
**Confidence:** HIGH (crate split, variant/SoA representation, PyO3 placement, build order — grounded in upstream headers + verified cubecl/PyO3 docs); MEDIUM (exact cubecl kernel-launch ergonomics for tree traversal — verified API shape, but tree traversal is an unusual kernel shape vs the matmul/axpy examples in the manual)

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ treelite-py  (PyO3 cdylib — the ONLY crate that links libpython / C-ABI)   │
│   numpy/buffer-protocol IN  ·  bytes/bytearray OUT  ·  module = `treelite`  │
└───────────────┬───────────────────────────────────────────┬───────────────┘
                │ depends on (pure-Rust API, no C-ABI leak)   │
                ▼                                             ▼
┌───────────────────────────┐  ┌───────────────┐  ┌──────────────────────────┐
│ treelite-loader-*          │  │ treelite-     │  │ treelite-serialize        │
│  xgboost / lightgbm /      │  │ builder       │  │  (binary v5 + JSON dump,  │
│  sklearn                   │  │ (ModelBuilder)│  │   trait-based)            │
└─────────────┬──────────────┘  └──────┬────────┘  └────────────┬─────────────┘
              │ build via                │ produces              │ reads/writes
              ▼                          ▼                       ▼
┌──────────────────────────┐  ┌────────────────────────────────────────────┐
│ treelite-gtil             │  │ treelite-core                                │
│  inference hot path       │──│  Model / ModelPreset enum / Tree<T> SoA /    │
│  cubecl kernels +         │  │  TreeBuf storage primitive                   │
│  CPU fallback, postproc   │  │  (depends only on treelite-enum)             │
└──────────────┬────────────┘  └───────────────────┬──────────────────────────┘
               │ generic over R: Runtime            │ depends on
               ▼                                     ▼
┌──────────────────────────┐  ┌────────────────────────────────────────────┐
│ cubecl (cpu/cuda/hip/wgpu │  │ treelite-enum                                │
│  selected by feature)     │  │  TaskType · TreeNodeType · Operator · DType  │
└──────────────────────────┘  │  (zero internal deps — the vocabulary root)  │
                              └────────────────────────────────────────────┘
```

All edges point downward → the dependency graph is a DAG (no cycles). `treelite-enum` is the sink; `treelite-py` is the source.

### Component Responsibilities

| Crate | Responsibility (one line) | Depends on |
|-------|---------------------------|------------|
| `treelite-enum` | Shared vocabulary: `TaskType`, `TreeNodeType`, `Operator`, `DType` (float32/float64) + string/serde conversions | — (leaf) |
| `treelite-core` | In-memory model: `Model`, `ModelPreset` enum over f32/f64, `Tree<T>` struct-of-arrays, the `TreeBuf<T>` storage primitive, typed errors | `treelite-enum` |
| `treelite-builder` | Fluent + validated `ModelBuilder` (orphan/topology checks) → produces a `Model` | `treelite-core`, `treelite-enum` |
| `treelite-loader-xgboost` | Parse XGBoost JSON / UBJSON / legacy binary into a `Model` via the builder | `treelite-builder`, `treelite-core`, `treelite-enum` |
| `treelite-loader-lightgbm` | Parse LightGBM text format into a `Model` | `treelite-builder`, `treelite-core`, `treelite-enum` |
| `treelite-loader-sklearn` | Build models from sklearn array dumps incl. the bulk path | `treelite-builder`, `treelite-core`, `treelite-enum` |
| `treelite-serialize` | Trait-based binary (v5) + JSON serializer/deserializer; round-trip | `treelite-core`, `treelite-enum` |
| `treelite-gtil` | Inference hot path: tree traversal + postprocessors as cubecl kernels; CPU default / GPU opt-in; output-shape calc; config | `treelite-core`, `treelite-enum`, `cubecl` |
| `treelite-py` | PyO3 module — the sole C-ABI / libpython surface; orchestrates load/build/predict/serialize over the pure-Rust crates | all crates above + `pyo3`, `numpy` (optional) |
| `treelite-rs` (facade, optional) | Umbrella crate re-exporting the public API as `treelite_rs::{...}` for Rust consumers and integration tests | core, builder, loaders, gtil, serialize |

**Why these boundaries (justification per split):**

- **`treelite-enum` separate from core** — the enums are the one thing every layer needs and nothing in them depends on the model shape. Isolating them keeps `treelite-core` from becoming a kitchen-sink dependency and lets loaders/serialize reference the vocabulary without pulling in `Tree`/`Model`. Mirrors upstream `include/treelite/enum/` which depends on nothing.
- **Loaders split per-format** (`xgboost` / `lightgbm` / `sklearn`) — each pulls a different parser dependency (JSON/UBJSON/SAX for xgboost, a bespoke text grammar for lightgbm, raw array marshalling for sklearn). Per-crate feature flags let a downstream consumer compile only the formats they need and keep heavy parser deps off the critical path. Matches upstream `src/model_loader/{xgboost_*,lightgbm,sklearn}.cc`.
- **`treelite-builder` separate from loaders** — upstream deliberately decouples *parsing* from *construction* (loaders call the `ModelBuilder` interface). Keeping the builder its own crate enforces that one-way edge (loader → builder → core) and gives the validated construction path a single owner. Note the upstream anti-pattern: sklearn bulk-load bypasses the builder via a `friend BulkConstructTree`. In Rust we replace `friend` with a deliberate `pub(crate)`/feature-gated "unchecked bulk construct" entry exposed by `treelite-core` and consumed only by `treelite-loader-sklearn` — keeping the bypass explicit and auditable rather than a hidden friendship.
- **`treelite-serialize` separate from core** — keeps wire-format concerns (versioning, byte framing, JSON shape) out of the in-memory types. Core stays decoupled from the format (see Serializer pattern below). Mirrors upstream `src/serializer.cc` + `detail/serializer.h` living outside `tree.h`.
- **`treelite-gtil` separate and the only cubecl dependent** — the project constraint is "cubecl applied to the GTIL hot path only." Isolating cubecl (and its backend feature flags) in one crate means loaders/builder/serialize never compile a GPU toolchain, keeps the 1e-5-equivalence risk contained to one crate, and lets the GPU backends be opt-in features that don't leak into the rest of the workspace.
- **`treelite-py` at the top, alone with the C-ABI** — PyO3 links libpython and is the only crate that emits a C-ABI. Every other crate is pure Rust. This satisfies the explicit "no C-API; PyO3 is the only binding" constraint and means the core can be unit-tested and consumed without Python in scope.
- **Optional `treelite-rs` facade** — gives Rust consumers and the equivalence harness one import surface without each having to enumerate member crates. Pure re-exports, no logic, no new edges that create cycles.

**Shared types live in `treelite-enum` (vocabulary) and `treelite-core` (model + errors).** Builder/loader/serialize/gtil all depend *downward* on these two; they never depend on each other (loaders do not depend on serialize, gtil does not depend on the builder). This is what keeps the DAG acyclic.

## Recommended Project Structure

```
treeline_rs/
├── Cargo.toml                      # [workspace] members + shared [workspace.dependencies] pins
├── crates/
│   ├── treelite-enum/              # vocabulary root
│   │   └── src/{lib,task_type,tree_node_type,operator,dtype}.rs
│   ├── treelite-core/              # Model / ModelPreset / Tree<T> / TreeBuf<T> / error
│   │   └── src/{lib,model,preset,tree,buf,error}.rs
│   ├── treelite-builder/           # ModelBuilder (typestate + validation)
│   │   └── src/{lib,builder,metadata,validate}.rs
│   ├── treelite-loader-xgboost/
│   │   └── src/{lib,json,ubjson,legacy,common}.rs
│   ├── treelite-loader-lightgbm/
│   │   └── src/{lib,parse}.rs
│   ├── treelite-loader-sklearn/
│   │   └── src/{lib,arrays,bulk}.rs
│   ├── treelite-serialize/         # trait-based wire format
│   │   └── src/{lib,binary_v5,json_dump,frame,sink_source}.rs
│   ├── treelite-gtil/              # cubecl hot path + CPU fallback
│   │   └── src/{lib,predict,traverse_kernel,postprocessor,output_shape,config,backend}.rs
│   ├── treelite-py/                # PyO3 cdylib — the only C-ABI crate
│   │   └── src/{lib,model,predict,serialize,buffer}.rs
│   └── treelite-rs/                # optional facade re-export
│       └── src/lib.rs
└── tests/                          # equivalence harness (golden vectors vs C++)
    └── equivalence/
```

### Structure Rationale

- **`crates/` flat workspace** — every member is a sibling; Cargo resolves the DAG. Pin all third-party versions once in `[workspace.dependencies]` (satisfies the "pin to latest published versions" constraint in one place).
- **`treelite-core` carries the typed `thiserror` error enum** — library crates return `Result<_, treelite_core::Error>` (or their own `thiserror` enum that `#[from]`-wraps core's); binaries/tests/the harness use `anyhow`. This matches the constraint exactly.
- **`treelite-gtil/backend.rs` owns runtime selection** — a single place that, behind feature flags, picks the cubecl `Runtime` and constructs the `ComputeClient`.

## Architectural Patterns

### Pattern 1: Type-erased `Model` via enum dispatch (the `std::variant` translation)

**What:** Upstream holds `std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` and dispatches with `std::visit`. The only two valid instantiations are `<f32,f32>` and `<f64,f64>` (threshold type == leaf type, enforced by `static_assert`). The faithful Rust translation is a two-variant enum:

```rust
// treelite-core
pub enum ModelPreset {
    F32(Vec<Tree<f32>>),
    F64(Vec<Tree<f64>>),
}

pub struct Model {
    pub variant: ModelPreset,
    pub num_feature: i32,
    pub task_type: TaskType,
    pub average_tree_output: bool,
    pub num_target: i32,
    pub num_class: Vec<i32>,
    // ... header fields (base_scores, postprocessor, etc.)
}
```

Since threshold and leaf type are always equal upstream, parameterize `Tree<T>` over a single `T: TreeFloat` (sealed trait implemented for `f32`/`f64`) rather than carrying two type params. This drops the impossible `<f32,f64>` combinations the C++ `static_assert` rejects anyway.

**When to use:** This is the recommended representation. Choose **enum dispatch** over the alternatives:

| Approach | Perf | Memory | Ergonomics | Verdict |
|----------|------|--------|------------|---------|
| **Enum `ModelPreset::{F32,F64}`** | One branch at the *top* of a predict call, then a monomorphized inner loop over `Tree<T>` — branch cost is amortized across the whole matrix | Tag is one discriminant; no per-tree overhead | `match` once, hand off to generic fn. Public API is a single non-generic `Model` type — matches upstream's type-erased handle | **Recommended** |
| Generics all the way (`Model<T>`) | Fastest in theory | Same | Forces every downstream API (loaders return `Model<?>`, PyO3 must expose two types) to be generic or boxed — leaks the type param everywhere; poor match for a type-erased Python object | Rejected — ergonomics |
| `Box<dyn ModelTrait>` (trait objects) | vtable indirection on every node access; defeats SoA cache locality | Fat pointer + heap | Hides the type but kills the hot path | Rejected — perf |

The enum gives the type-erasure ergonomics of trait objects with the monomorphized inner-loop speed of generics: you `match` once, then call a generic `predict_inner<T>(...)`.

**Trade-offs:** A `match` at every public method (`get_num_tree`, `serialize`, etc.) — trivial cost, and exactly what `std::visit` compiles to anyway. A small amount of duplicated dispatch code, conventionally handled with a `with_preset!` macro that expands the two arms.

### Pattern 2: Struct-of-Arrays `Tree<T>` over a `TreeBuf<T>` storage primitive

**What:** Upstream `Tree<T,L>` stores ~20 parallel `ContiguousArray<T>` columns (`cleft_`, `cright_`, `split_index_`, `threshold_`, `leaf_value_`, `cmp_`, …) indexed by node id — not a `Vec<Node>`. `ContiguousArray<T>` is a move-only POD buffer that can **own** its allocation or **alias a foreign buffer** (`UseForeignBuffer`), which is how upstream achieves zero-copy deserialization from the Python buffer protocol.

**Recommended Rust representation — decision:**

| Storage option | Owned path | Zero-copy-from-Python | Into cubecl | Verdict |
|----------------|-----------|----------------------|-------------|---------|
| **`Vec<T>` columns** | idiomatic, simplest | needs copy on the borrowed path | `bytemuck::cast_slice(&vec) → client.create_from_slice` works directly | Use for **owned** columns (the loader/builder output) |
| **`Cow<'a,[T]>` / enum `Owned(Vec)`/`Borrowed(&[])`** | — | aliases a `PyBuffer` slice with zero copy, matching `UseForeignBuffer` | same cast | Use for the **deserialize-from-buffer** path |
| Arrow `ScalarBuffer<T>` | heavier dep, refcounted | zero-copy, but only buys interop we don't need (no Arrow upstream) | works (the ARROW_CUBECL manual shows it) | **Rejected for storage** — adds a large dependency for a borrowing model we get cheaper with `Cow`/a small enum. Keep the *technique* (bytemuck recast) without the Arrow type |
| `bytemuck`-backed `&[T]` slices | — | the bridge primitive, not a container | yes | Use as the *cast step*, not the owner |

**Recommended `TreeBuf<T>`** — a thin storage primitive that is the faithful, idiomatic replacement for `ContiguousArray<T>`:

```rust
// treelite-core — `T: Pod` (bytemuck) so every column is zero-copy castable
pub enum TreeBuf<'a, T: Pod> {
    Owned(Vec<T>),          // loader/builder output, serialize-to-bytes path
    Borrowed(&'a [T]),      // == UseForeignBuffer: aliases a Python buffer / mmap, zero copy
}
impl<T: Pod> TreeBuf<'_, T> {
    pub fn as_slice(&self) -> &[T] { /* deref either arm */ }
    pub fn as_bytes(&self) -> &[u8] { bytemuck::cast_slice(self.as_slice()) }
}
```

Every numeric column (`threshold`, `leaf_value`, child indices, `split_index`) is a `TreeBuf<T>` / `TreeBuf<i32>`; small enum columns (`node_type`, `cmp`) are `TreeBuf<u8>` over `#[repr(u8)]` enums (`bytemuck`-safe). The boolean columns (`default_left`, `*_present`) become `TreeBuf<u8>` (not Rust `bool`, to keep a defined bit pattern for `Pod` and for buffer-protocol round-trip — upstream uses `ContiguousArray<bool>` but a `u8` is safer for transmutation).

**Why this wins on memory:** SoA keeps each column densely packed (no per-node struct padding), so a 4-byte `split_index` column has no holes, and the hot traversal touches only the 2–3 columns it needs (`cleft`/`cright`/`threshold`/`split_index`) — maximal cache density. The `Borrowed` arm means deserialization is a pointer assignment, not an allocation+copy.

**Zero-copy chain (the load-bytes → predict-on-GPU path):**

```
Python bytes  ──PyBuffer::as_slice──►  &[T]  (no copy)
   └─► TreeBuf::Borrowed(&[T])                       (no copy; aliases the Py buffer)
         └─► TreeBuf::as_bytes() = bytemuck::cast_slice ──► &[u8]   (pointer recast, no copy)
               └─► client.create_from_slice(T::as_bytes(slice)) ──► cubecl handle (one host→device upload)
```

`bytemuck::cast_slice` is the verified host bridge; cubecl ingests `&[u8]` / `T::as_bytes(&[T])` directly into a device handle. (Confirmed against ZERO_COPY_TRANSMUTATION_CUBECL.md and cubecl docs: `client.create_from_slice(f32::as_bytes(&data))`.)

**When to use / trade-offs:** The `Borrowed` lifetime parameter propagates into `Tree<'a,T>` and `Model<'a>` only on the deserialize path. To avoid lifetime infection of the whole public API, the deserializer can either (a) keep the backing `Py_buffer`/`Bytes` alive in an owner held alongside the model (`self_cell`/`ouroboros`-style), or (b) default to `Owned` and offer zero-copy as an explicit `deserialize_borrowed(&bytes)` constructor. **Recommendation: default `Owned` for ergonomics; expose `Borrowed` as an opt-in for the memory-critical Python path**, matching upstream's two `Deserialize*` entry points.

### Pattern 3: cubecl integration boundary (GTIL hot path)

**What:** `treelite-gtil` is the only cubecl-dependent crate. It hosts (1) tree-traversal kernels and (2) postprocessor kernels, generic over `R: Runtime` and the float `F: Float`. Runtime is chosen by Cargo feature; the CPU runtime is the default and is what CI uses.

**Backend selection (CPU default, GPU opt-in, runtime-dispatched):**

```rust
// treelite-gtil/backend.rs
#[derive(Clone, Copy)]
pub enum Backend { Cpu, #[cfg(feature="cuda")] Cuda, #[cfg(feature="wgpu")] Wgpu, ... }

pub fn predict(model: &Model, x: &Matrix, cfg: &Config) -> Result<Vec<f64>, Error> {
    match cfg.backend {
        Backend::Cpu  => predict_on::<cubecl::cpu::CpuRuntime>(model, x, cfg),
        #[cfg(feature="cuda")]
        Backend::Cuda => predict_on::<cubecl::cuda::CudaRuntime>(model, x, cfg),
        // ...
    }
}
fn predict_on<R: Runtime>(model: &Model, x: &Matrix, cfg: &Config) -> Result<Vec<f64>, Error> {
    let client = R::client(&Default::default());
    /* upload SoA columns + input matrix, launch traverse kernel, read back, postprocess */
}
```

Verified facts from cubecl docs: kernels are `#[cube(launch)]`/`#[cube(launch_unchecked)]`, generic over `R: Runtime`; runtimes are gated by features `cpu`/`cuda`/`hip`/`wgpu`; a `ComputeClient` comes from `R::client(device)`; data goes up via `client.create_from_slice(F::as_bytes(&slice))` and comes back via `client.read_one(handle)`; launch takes `CubeCount` and `CubeDim`. Because all backends are generic over `R`, the kernel body and the upload/readback logic are written **once** and the `match` only picks the monomorphization — this is exactly cubecl's intended pattern ("generic over `R: Runtime`, the rest of the code stays identical").

**How the SoA model crosses into a kernel launch (zero-copy on the host, one upload):**

1. The `match` on `ModelPreset` selects `F = f32`/`f64`.
2. For each tree column needed by traversal (`cleft`, `cright`, `split_index`, `threshold`, `default_left`, plus the categorical side-tables), call `TreeBuf::as_bytes()` (= `bytemuck::cast_slice`, pointer recast, no copy) and `client.create_from_slice(...)` → one device handle per column. (For multi-tree ensembles, concatenate columns across trees into a flat column + a per-tree offset array — a classic "ragged SoA" so the whole forest is a handful of handles, not handles-per-tree.)
3. The input matrix (row-major dense, or CSR `data`/`col_ind`/`row_ptr`) uploads the same way.
4. Launch the traverse kernel with `CubeCount`/`CubeDim` sized to (rows × trees); each unit walks one tree for one row using only integer index columns + the threshold column, writing a per-(row,tree) margin.
5. A reduce/postprocess kernel (sigmoid/softmax/exp/identity — the upstream postprocessor set) turns margins into the final output shape.
6. `client.read_one` → `bytemuck::cast_slice` back to `&[F]` → caller buffer.

**Important hot-path nuance:** tree traversal is *data-dependent branching*, which is the opposite of the SIMD-friendly axpy/matmul kernels in the cubecl manual. The kernel must avoid `continue` (the manual flags it as unsupported — use a bounded `for`/`while` with `break`), and divergent traversal across a warp/plane will under-utilize a GPU. This is exactly why the project scopes GPU as **opt-in** and keeps the **CPU cubecl backend as the deterministic default** for the 1e-5 equivalence guarantee. (See Pitfall in PITFALLS.md.)

### Pattern 4: ModelBuilder — typestate vs runtime-validated

**What:** Upstream `ModelBuilder` is a fluent interface with `Begin/End` pairing (`StartTree`/`StartNode`/`NumericalTest`|`CategoricalTest`|`LeafScalar`/`EndNode`/`EndTree`/`CommitModel`) and a runtime validation flag (`check_orphaned_nodes`).

**Recommendation: runtime-validated builder, not full typestate.**

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Runtime-validated** (methods return `Result`, `commit()` runs orphan/topology checks) | Mirrors upstream behavior 1:1 incl. `check_orphaned_nodes`; loaders drive it dynamically (node count/shape unknown until parse time); errors are recoverable `thiserror` values | Misuse caught at runtime, not compile time | **Recommended** |
| Typestate (`Builder<TreeOpen>` → `Builder<NodeOpen>` …) | Compile-time Begin/End pairing | Loaders build trees in *data-driven* order from a parser stream where the state isn't statically known — typestate fights the primary caller; can't express "validate orphans" as a type | Rejected for the public/loader-facing API |

A light typestate wrapper *may* be offered as an ergonomic Rust-only convenience, but the canonical path is runtime-validated because the loaders (the dominant consumer) are inherently dynamic. Keep the typestate bypass concern (sklearn bulk) explicit as noted in Pattern 1.

### Pattern 5: Serializer — trait-based replacement for the C++ mixin

**What:** Upstream uses a CRTP-style `Serializer<MixIn>`/`Deserializer<MixIn>` where the `MixIn` plugs in the I/O backend (stream vs PyBuffer vs in-memory buffer); the field-walking logic is shared (`detail/serializer.h`). The idiomatic Rust translation is **two traits** (a `Sink` and a `Source`) plus generic walk functions — keeping the model types ignorant of the wire format.

```rust
// treelite-serialize
pub trait Sink {                          // == the write-side MixIn
    fn put_scalar<T: Pod>(&mut self, v: &T);
    fn put_array<T: Pod>(&mut self, a: &[T]);
    fn put_str(&mut self, s: &str);
}
pub trait Source {                        // == the read-side MixIn
    fn get_scalar<T: Pod>(&mut self) -> Result<T, Error>;
    fn get_array<T: Pod>(&mut self) -> Result<TreeBuf<'_, T>, Error>;  // can return Borrowed for zero-copy
    fn get_str(&mut self) -> Result<String, Error>;
}
// One generic walker, three impls of Sink/Source (Stream, PyBuffer, Vec<u8>)
pub fn serialize<S: Sink>(model: &Model, sink: &mut S) -> Result<(), Error> { /* walk fields */ }
pub fn deserialize<S: Source>(src: &mut S) -> Result<Model, Error> { /* read fields, version-gate v5 */ }
```

**How core stays decoupled from the wire format:** `treelite-core` defines the data only and exposes plain field accessors. `treelite-serialize` depends on `treelite-core` (one-way edge) and owns the byte framing, the v5 format constants, and the optional-field/extension-slot skipping logic (`SkipOptionalField*`). The model never imports the serializer; it doesn't know whether it's being written to a stream, a `Vec<u8>`, or PyBuffer frames. This is the same separation the C++ achieves via friend-class mixins, but expressed as a one-directional crate edge plus traits. For JSON dump, a `serde`-backed path is fine since JSON dump is one-way (no zero-copy concern). v3.9/v4.0 are explicitly out of scope — implement v5 only and reject older magic with a typed error.

### Pattern 6: PyO3 placement — depend on core without leaking C-ABI

**What:** `treelite-py` is a `cdylib` with `pyo3` (and optionally `numpy`/`pyo3-arrow`) and is the **only** crate that links libpython or emits any C symbol. It calls the pure-Rust crates through their normal Rust APIs.

- **Input zero-copy:** accept numpy/array inputs via PyO3's `PyBuffer<T>` and `as_slice()` (verified: succeeds when the buffer is C-contiguous and the format matches `T`, returning `&[ReadOnlyCell<T>]` with no copy). Feed that slice straight into `treelite-gtil` / `TreeBuf::Borrowed`.
- **Output / serialize zero-copy:** expose model bytes by implementing `__getbuffer__`/`__releasebuffer__` (PyO3 buffer protocol) so Python sees the serialized buffer without a copy, matching upstream `SerializeToPyBuffer`.
- **No C-ABI leak:** because only `treelite-py` depends on `pyo3`, none of `treelite-core/-gtil/-serialize/...` exports `extern "C"` or `#[no_mangle]`. The C-ABI surface is confined to PyO3's generated module init. Use the `abi3` feature for a stable, version-portable wheel.
- **Constraint satisfied:** "No C-API; PyO3 is the only binding" — the workspace has exactly one C-ABI crate and it is the Python module.

## Data Flow

### Load → Build → Model → Predict → Serialize (Rust terms)

```
load_xgboost_json(&bytes)                       [treelite-loader-xgboost]
   → drives ModelBuilder.start_tree()/start_node()/numerical_test()/end_*()   [treelite-builder]
   → builder.commit()  →  Model { variant: ModelPreset::F32(Vec<Tree<f32>>), .. }   [treelite-core]
        │
        ├─ gtil::predict(&model, &x, &cfg)       [treelite-gtil]
        │     match model.variant → F=f32
        │     upload SoA columns (TreeBuf::as_bytes → client.create_from_slice)  [zero-copy host recast]
        │     launch traverse kernel (R: Runtime = Cpu by default)
        │     launch postprocessor kernel
        │     read_one → cast_slice → output Vec<f32>
        │
        └─ serialize::serialize(&model, &mut sink)   [treelite-serialize]
              walk fields → Sink (Stream | Vec<u8> | PyBuffer)
              deserialize::<Source>(&mut src) → Model   (v5 only; Borrowed arm = zero-copy)
```

### Python request flow

```
Python: treelite.load_xgboost("model.json"); treelite.gtil.predict(model, X_numpy)
   → treelite-py: PyBuffer::get(X).as_slice()  [zero-copy &[f32]]
   → calls treelite-gtil::predict over the Rust Model
   → returns numpy array (or a buffer-protocol object) back to Python
```

## Scaling Considerations

| Scale | Architecture adjustments |
|-------|--------------------------|
| Small model / few rows | CPU cubecl backend; single upload; traversal cost dominated by row count. GPU not worth the transfer overhead. |
| Large ensemble (1000s of trees) × large matrix | Concatenate per-tree columns into ragged-SoA flat handles (few uploads, not per-tree); GPU opt-in starts paying off; consider tiling rows to fit device memory. |
| Memory-constrained host | Use `TreeBuf::Borrowed` deserialization (mmap or Python buffer) to avoid duplicating the model in RAM; optional jemalloc/mimalloc allocator; optional f16 thresholds via cubecl half-precision (only if it stays within 1e-5). |

### Scaling priorities

1. **First bottleneck: host→device transfer of inputs/model.** Fix with the ragged-SoA concatenation (one handle per column for the whole forest) and zero-copy `bytemuck` recast so the only real cost is the PCIe upload.
2. **Second bottleneck: warp divergence in tree traversal on GPU.** Mitigate by keeping GPU opt-in, batching rows per cube, and defaulting to the deterministic CPU backend for correctness-critical use.

## Anti-Patterns

### Anti-Pattern 1: Parameterizing the whole model over `<T, L>` two type params

**What people do:** Faithfully copy `Tree<ThresholdType, LeafOutputType>` with two params.
**Why it's wrong:** Upstream's own `static_assert` forbids the mixed combinations; the only valid pairs are `<f32,f32>` and `<f64,f64>`. Two params double the monomorphizations and infect every signature for no expressive gain.
**Do this instead:** Single `Tree<T: TreeFloat>` and the two-variant `ModelPreset` enum.

### Anti-Pattern 2: Adding cubecl (or any GPU dep) to crates other than `treelite-gtil`

**What people do:** Pull cubecl into core to "make Tree GPU-ready."
**Why it's wrong:** Violates the project scope (cubecl is hot-path-only), forces a GPU toolchain onto loaders/serialize, and expands the 1e-5-equivalence risk surface.
**Do this instead:** Keep cubecl confined to `treelite-gtil`. Core exposes plain `bytemuck`-castable `TreeBuf` columns; gtil does the uploading.

### Anti-Pattern 3: Letting `treelite-core` depend on the serializer or on PyO3

**What people do:** Add `serde`/`pyo3` derives onto the model types for convenience.
**Why it's wrong:** Couples the in-memory representation to a wire format / to libpython, creating the exact mixin-tangle the trait split is meant to avoid, and risks a dependency cycle (serialize → core → serialize).
**Do this instead:** Keep edges one-way (serialize/py depend on core, never the reverse). Implement the wire format with the `Sink`/`Source` traits in `treelite-serialize`; keep PyO3 conversions in `treelite-py`.

### Anti-Pattern 4: Using a `Vec<Node>` (array-of-structs) for the tree

**What people do:** Define `struct Node { left, right, threshold, ... }` and `Vec<Node>`.
**Why it's wrong:** Reintroduces per-node padding, kills the cache density of column traversal, and breaks the zero-copy column-wise buffer-protocol/cubecl path that depends on each field being a contiguous typed buffer.
**Do this instead:** Struct-of-Arrays `TreeBuf<T>` columns, faithful to upstream.

## Suggested Build Order (dependency-driven phasing)

The DAG dictates a clear bottom-up order. Each phase delivers a compilable, testable crate before its dependents exist.

1. **`treelite-enum`** — the vocabulary root, zero deps. Must come first; everything references it.
2. **`treelite-core`** — `Model` / `ModelPreset` enum / `Tree<T>` SoA / `TreeBuf<T>` / typed errors. The keystone; lock the variant + SoA representation here because it constrains every later crate.
3. **`treelite-builder`** — validated construction; unblocks all loaders. Build before any loader.
4. **`treelite-serialize`** — v5 binary + JSON, trait-based. Independent of loaders/gtil; can be built in parallel with the builder once core is stable. Enables early round-trip fixtures.
5. **Loaders** (`xgboost`, then `lightgbm`, then `sklearn`) — depend on builder + core. XGBoost first (richest equivalence fixtures: mushroom etc.); sklearn last (needs the explicit bulk-construct path).
6. **`treelite-gtil`** — depends only on core/enum + cubecl. Start with the **CPU cubecl backend** to nail 1e-5 equivalence against golden vectors before adding GPU features. The riskiest crate (data-dependent kernels) — flag for deeper research at phase time.
7. **`treelite-py`** — last; depends on all of the above. Wires the Python module, buffer-protocol I/O, abi3.
8. **`treelite-rs` facade + equivalence harness** — can be stubbed early (re-exports grow as crates land); the harness becomes meaningful once a loader + gtil exist.

**Critical-path note:** core (2) gates everything; gtil (6) is the highest-risk and should follow a stable core + at least one working loader so its outputs can be validated immediately against golden vectors. Serialize (4) and the builder (3) can proceed concurrently after core.

## Sources

- Upstream headers (porting spec, authoritative): `treelite-mainline/include/treelite/tree.h` (Model / ModelPreset / Tree, `std::variant`, `static_assert` on T==L), `contiguous_array.h` (owned/foreign-buffer storage primitive), `detail/serializer.h` (mixin Serializer/Deserializer + PyBuffer framing), `gtil.h` (Predict / PredictSparse / PredictKind / Configuration) — HIGH
- `.planning/PROJECT.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` (constraints, layer map, anti-patterns) — HIGH
- CubeCL docs via Context7 `/tracel-ai/cubecl` (runtime feature flags cpu/cuda/hip/wgpu; generic `R: Runtime`; `R::client`, `create_from_slice(F::as_bytes(..))`, `read_one`, `CubeCount`/`CubeDim`/`#[cube(launch)]`; `continue` unsupported) — HIGH
- PyO3 docs via Context7 `/pyo3/pyo3` (`PyBuffer<T>::as_slice` zero-copy on C-contiguous numpy; `__getbuffer__`/`__releasebuffer__`; `abi3`) — HIGH
- Optimisor manuals `ZERO_COPY_TRANSMUTATION_CUBECL.md`, `ZERO_COPY_ARROW_CUBECL.md` (`bytemuck::cast_slice` host bridge; Arrow buffer path considered and rejected for storage) — HIGH

---
*Architecture research for: Rust workspace port of Treelite (SoA tree ensembles, cubecl GTIL hot path, PyO3 binding)*
*Researched: 2026-06-09*

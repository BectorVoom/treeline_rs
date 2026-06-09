# Technology Stack

**Project:** treelite-rs (from-scratch Rust port of Treelite 4.7.0)
**Researched:** 2026-06-09
**Mode:** Ecosystem (Stack dimension)
**Overall confidence:** HIGH for versions (verified against crates.io sparse index 2026-06-09 + Context7), MEDIUM-HIGH for fit (grounded in upstream loader source + project manuals)

> All versions below were verified live against the crates.io sparse index
> (`https://index.crates.io/...`) on 2026-06-09 and PyO3/maturin against
> Context7 (`/pyo3/pyo3`). They are NOT from training data. Where a crate's
> newest published version is a pre-release, the latest **stable** is pinned
> and the pre-release is called out.

---

## Recommended Stack

### Workspace-wide / Cross-cutting

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rust edition | **2024** | Language edition (per PROJECT constraint) | Required by project; PyO3 0.28 and all crates below build on edition 2024 / current stable. |
| Cargo resolver | **"3"** | Dependency resolution | Resolver 3 is the edition-2024 default and enables `rust-version`-aware (MSRV) resolution. Set once in the workspace root `[workspace] resolver = "3"`. |
| `thiserror` | **2.0.18** | Typed library errors | Per constraint: libraries. v2.x is current; `#[error]` derive, `#[from]`, no_std-friendly. |
| `anyhow` | **1.0.102** | Bin/test/loader-internal errors | Per constraint: binaries + tests + the equivalence harness. |

### Compute (GTIL inference hot path)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `cubecl` | **0.10.0** | Kernel authoring + launch for GTIL traversal & postprocessors | Newest published (22 versions; matches the CubeCL manual at `cubecl_manual/`). Single umbrella crate; backends are Cargo features. |
| `cubecl-runtime` | **0.10.0** | `ComputeClient`, memory mgmt, autotune (transitive via `cubecl`) | Pull in only if you need to name runtime types directly; otherwise `cubecl` re-exports what you need. Keep version locked to `cubecl`. |
| `cubecl-cpu` | **0.10.0** | **Default** CPU backend | Enabled via `cubecl`'s `cpu` feature. Deterministic, GPU-free CI — satisfies the "CPU default, deterministic, runs in CI without a GPU" decision. |
| `cubecl-cuda` | **0.10.0** | Opt-in NVIDIA backend | `cuda` feature. Off by default. |
| `cubecl-wgpu` | **0.10.0** | Opt-in portable GPU (Vulkan/Metal/DX/WebGPU) | `wgpu` feature. The most portable GPU path; good for "GPU opt-in" on machines without CUDA. |
| (cubecl `rocm`/`hip`, `metal`, `vulkan`) | 0.10.0 | Further opt-in backends | Additional feature flags exist (`rocm`, `hip`, `metal`, `vulkan`, `webgpu`). Ship `cpu` + `cuda` + `wgpu`; add others on demand. |

**CRITICAL — cubecl 0.10.0 has NO default backend.** Verified from the 0.10.0
feature table: `default` is empty; `cpu`/`cuda`/`hip`/`rocm`/`metal`/`wgpu`/`vulkan`
are explicit features each pulling the matching `cubecl-*` crate. This is exactly
what we want for "CPU default, GPU opt-in" — see backend-selection section below.

### Model-loader parsing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `serde_json` | **1.0.150** | XGBoost JSON loader (primary) | The ecosystem standard; battle-tested, exact f64 round-tripping, `RawValue`, and a **streaming `StreamDeserializer`/SAX-ish `Deserializer::from_reader`** path. Upstream Treelite uses a *SAX* handler precisely to avoid materializing a huge DOM — see streaming note below. |
| `byteorder` | **1.5.0** | XGBoost legacy binary + UBJSON primitive decode | Little-endian fixed-width reads for the legacy binary format and UBJSON scalars. Tiny, zero-risk. |
| (hand-rolled) | — | LightGBM text loader | Upstream `lightgbm.cc` is a hand-rolled `getline` + `strtod` line parser (`key=value`, space-split arrays). Port it directly in Rust with `str::split`, `str::parse`, and `lexical-core` if you need bit-stable float parsing. No external grammar crate needed. |
| `lexical-core` (optional) | latest 1.x | Exact, fast float parse for LightGBM/JSON edge cases | Only if `str::parse::<f64>()` / `serde_json` rounding ever diverges from C++ `strtod` at the 1e-5 boundary. Defer until the equivalence harness flags it. |

**UBJSON:** Do **not** add a third-party UBJSON crate. The only `ubjson` crate on
crates.io is `0.1.0` (single release, unmaintained, unproven) — **immature/risky**
and disqualified for a 1e-5-equivalence port. Upstream parses UBJSON through the
*same* SAX delegated handler it uses for JSON (`nlohmann::json::sax_parse` with
`input_format_t::ubjson`). UBJSON is a trivially small binary grammar; hand-roll a
reader over `byteorder` that feeds the **same loader state machine** as the JSON
path (mirror upstream's `DelegatedHandler` design). This keeps JSON and UBJSON
numerically identical by construction — the single most important property for
equivalence.

### Memory-efficiency toolkit

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `bytemuck` | **1.25.0** | Zero-copy `&[T] ⇄ &[u8]` transmutation | The host↔GPU bridge in every CubeCL manual (`ZERO_COPY_*`, `HALF_PRECISION_CUBECL`). `Pod`/`Zeroable`, alignment-checked, no UB. Foundational for the SoA buffers. |
| `arrow` (arrow-rs) | **58.3.0** | Zero-copy dense input matrices / interchange | Newest stable (58.3.0; the sparse index also lists 56.2.1/57.3.1 but 58.3.0 is highest semver and matches the project's arrow manuals). `PrimitiveArray::values()` → `&[T]` → `bytemuck::cast_slice` → `cubecl::bytes::Bytes` is the documented zero-copy ingestion path. **Use it for input matrices and Python-buffer interop, NOT for the internal `Tree` SoA storage** (keep those plain `Vec`/`Box<[T]>`). |
| `smallvec` | **1.15.1** | Inline-small node-children / per-tree scratch vectors | Latest **stable** (2.0 is alpha-only — `2.0.0-alpha.12` — and **must not** be used in a fidelity-critical port). Most split nodes have ≤2–4 children/category buckets; SVO removes per-node heap churn. |
| `compact_str` | **0.9.1** | Feature names, model metadata strings | Newer than the project manual's 0.8 (verified 0.9.1 latest). 24-byte SSO replacement for `String`; XGBoost/LightGBM feature-name vectors are mostly short. `serde` feature for de/serialization. |
| `tikv-jemallocator` | **0.7.0** | **Recommended** global allocator | Best fragmentation control for long-lived, multi-format model objects; mature, the de-facto choice for data-heavy Rust (Polars-class). Note: the project manual references the older `jemallocator 0.5`; **prefer `tikv-jemallocator 0.7.0`**, the maintained fork. |
| `mimalloc` | **0.1.52** | Alternative global allocator | Lower-latency, simpler, great for short-lived allocation churn (parsing). Use if jemalloc's build (C toolchain) is undesirable on a target. Pick **one**, behind a workspace feature; do not link both. |

**Allocator tradeoff (pick one, make it a feature):**
- `tikv-jemallocator 0.7.0` — superior fragmentation control + heap profiling
  (`jeprof`) for a memory-critical port; needs a C compiler at build time;
  weaker Windows/MSVC story.
- `mimalloc 0.1.52` — trivially portable drop-in, lowest latency on alloc-heavy
  parsing, smaller metadata; fewer introspection tools.
- **Recommendation:** default to `tikv-jemallocator` on Linux/macOS builds of the
  binary/bench targets, expose a `--features mimalloc` escape hatch. Allocators
  do **not** affect numeric output, so this is purely a perf/footprint knob and
  is equivalence-neutral.

### Half precision

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `half` | **2.7.1** | Host-side `f16`/`bf16` representation | The industry standard and the type CubeCL's `f16`/`bf16` map to. Enable `features = ["bytemuck", "num-traits"]` so `&[half::f16]` casts straight into GPU `Bytes` (per `HALF_PRECISION_CUBECL.md`). |

**half ↔ cubecl interplay:** CubeCL treats `f16`/`bf16` as first-class kernel
element types, mapping to native GPU half types. The bridge is: host `half::f16`
→ `bytemuck::cast_slice` → `Bytes` → kernel generic over `F: Float`. **Gate on
runtime feature detection** (`client.properties().feature_enabled(Feature::Type(...))`
/ `supports_type(FloatKind::F16)`) — not all backends/HW support f16, and the CPU
default backend's f16 is for storage/bandwidth, not a precision win.
**Equivalence caveat:** f16 inference carries ~1e-3 error, which **violates the
1e-5 contract**. Keep f16 as an explicit opt-in fast path for f32/f64 models where
the caller accepts reduced precision; the equivalence harness must run against the
**f32/f64 CPU path**, never f16.

### Python binding

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pyo3` | **0.28.3** | Python extension module over the Rust core | Newest (verified via Context7 `/pyo3/pyo3` + index). Edition-2024 compatible. Use `features = ["extension-module", "abi3-py38"]`. |
| `maturin` | **1.13.3** | Build/package the PyO3 crate into a wheel | The standard PyO3 build backend (Context7 `/pyo3/maturin`). `pyproject.toml` with `[build-system] requires = ["maturin>=1.13,<2"]`, `build-backend = "maturin"`. |

**abi3 recommendation:** enable `abi3-py38` (matches upstream Treelite's "Python
3.8+" floor). abi3 builds one wheel that works across all CPython ≥3.8 — far less
CI matrix. Confirmed via Context7: add `pyo3 = { version = "0.28", features =
["abi3"] }` (use the versioned `abi3-pyXY` to also set the floor). The `treelite-py`
workspace member sets `crate-type = ["cdylib"]`, depends on the internal
`treelite-core`/`treelite-loader`/`treelite-gtil` crates by `path`, and converts
their `thiserror` enums into `PyErr` at the boundary.

### Test / equivalence tooling

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `proptest` | **1.11.0** | Random seeded input-matrix generation | **Preferred over quickcheck** — richer strategies, shrinking, and a persisted-regression file. Drives the "random seeded inputs vs golden vectors" harness. |
| `approx` | **0.5.1** | 1e-5 float comparison (`assert_abs_diff_eq!`) | Latest **stable** (0.6.0 is `-rc2` only — do not pin a release candidate in a fidelity gate). `abs_diff_eq` with `epsilon = 1e-5` is exactly the equivalence assertion. |
| `criterion` | **0.8.2** | Inference/loader benchmarks | Standard statistical benchmarking; gates the "memory/perf" goals without affecting correctness. |
| `insta` | **1.47.2** | Snapshot tests for serializer / model-structure round-trips | Good for asserting *structural* JSON/round-trip stability. **Not** for the float-equivalence vectors — those should be **committed binary golden fixtures** generated once from C++ Treelite (per the PROJECT decision) and compared with `approx`, since insta's text snapshots are ill-suited to large f64 vectors. |
| `quickcheck` | 1.1.0 | (alternative) | Listed for completeness; `proptest` wins on shrinking + regression persistence. Do not add both. |

**Golden-vector approach:** Commit the C++-generated expectation vectors as
fixture files under `tests/fixtures/` (raw `f64`/`f32` `.bin` via `bytemuck`, or
`.npy`). `proptest` generates seeded input matrices; the harness loads the matching
golden output and asserts with `approx` at `1e-5`. CI never compiles C++ (per
decision). Use `insta` only for the serializer's structural snapshots.

---

## Cargo Workspace Layout

```
treeline_rs/                      # repo dir (name stays treeline_rs)
├── Cargo.toml                    # [workspace], resolver="3", shared dep table
├── crates/
│   ├── treelite-core/            # Model / Tree<T,L> SoA, ContiguousArray analog, enums
│   ├── treelite-builder/         # ModelBuilder (fluent, orphan/topology validation)
│   ├── treelite-loader/          # XGBoost (json/ubjson/binary), LightGBM, sklearn
│   ├── treelite-gtil/            # GTIL: cubecl kernels + postprocessors
│   ├── treelite-serialize/       # v5 binary + JSON round-trip
│   └── treelite-py/              # PyO3 cdylib (the only external binding)
└── Cargo.lock                    # committed
```

**Root `Cargo.toml` skeleton:**

```toml
[workspace]
members = ["crates/*"]
resolver = "3"

[workspace.package]
edition = "2024"
rust-version = "1.85"          # edition-2024 floor; bump after verifying toolchain
version = "0.1.0"

[workspace.dependencies]
# pin once, inherit with `dep.workspace = true`
thiserror      = "2.0.18"
anyhow         = "1.0.102"
cubecl         = "0.10.0"
serde          = { version = "1.0.228", features = ["derive"] }
serde_json     = "1.0.150"
byteorder      = "1.5.0"
bytemuck       = { version = "1.25.0", features = ["derive"] }
arrow          = "58.3.0"
smallvec       = { version = "1.15.1", features = ["union"] }
compact_str    = { version = "0.9.1", features = ["serde"] }
half           = { version = "2.7.1", features = ["bytemuck", "num-traits"] }
pyo3           = { version = "0.28.3", features = ["extension-module", "abi3-py38"] }
# dev / opt
proptest       = "1.11.0"
approx         = "0.5.1"
criterion      = "0.8.2"
insta          = "1.47.2"
tikv-jemallocator = "0.7.0"
mimalloc       = { version = "0.1.52", default-features = false }
```

Conventions: `resolver = "3"` (edition-2024 default, MSRV-aware); single
`[workspace.dependencies]` table so every member inherits one pinned version;
commit `Cargo.lock`; each crate one responsibility (mirrors the upstream layer
boundaries in `.planning/codebase/ARCHITECTURE.md`).

---

## cubecl Backend Selection — CPU default, GPU opt-in (concrete)

Because cubecl 0.10.0 ships **no default backend**, the selection lives in
`treelite-gtil`'s Cargo features:

```toml
# crates/treelite-gtil/Cargo.toml
[dependencies]
cubecl = { workspace = true }

[features]
default = ["cpu"]                       # deterministic, GPU-free, CI-safe
cpu  = ["cubecl/cpu"]
cuda = ["cubecl/cuda"]
wgpu = ["cubecl/wgpu"]
rocm = ["cubecl/rocm"]
```

Kernels are written **generic over `R: Runtime`** (every CubeCL manual sample does
this). A thin runtime-dispatch layer picks the concrete runtime at call time:

```rust
pub enum Backend { Cpu, #[cfg(feature="cuda")] Cuda, #[cfg(feature="wgpu")] Wgpu }

pub fn predict(model: &Model, x: &Matrix, backend: Backend) -> Vec<f64> {
    match backend {
        Backend::Cpu => run::<cubecl::cpu::CpuRuntime>(model, x),
        #[cfg(feature="cuda")] Backend::Cuda => run::<cubecl::cuda::CudaRuntime>(model, x),
        #[cfg(feature="wgpu")] Backend::Wgpu => run::<cubecl::wgpu::WgpuRuntime>(model, x),
    }
}
```

- **Default build / CI**: only `cpu` compiled → no GPU drivers, no CUDA toolkit,
  fully deterministic → satisfies the equivalence harness.
- **GPU**: caller opts in with `--features cuda` (or `wgpu`) and selects
  `Backend::Cuda` at runtime. Per PROJECT, GPU bit-exactness is *not* guaranteed;
  the 1e-5 tolerance absorbs reduction-order differences.

Confidence: **HIGH** (feature names verified directly from cubecl 0.10.0 index
metadata; pattern verified against the CubeCL manual samples).

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| JSON parse | `serde_json` (streaming) | `simd-json` 0.17.0 | simd-json is faster but mutates input, has a heavier unsafe surface, and its float parsing path is less obviously bit-faithful to XGBoost's. For a 1e-5 port, predictability > raw speed; loaders aren't the hot path (GTIL is). Revisit only if profiling shows JSON parse dominates. |
| UBJSON | hand-rolled over `byteorder`, shared SAX state | `ubjson` 0.1.0 crate | Single-release, unmaintained, unproven — **disqualified for a fidelity-critical port**. Hand-rolling guarantees JSON/UBJSON numeric parity (mirrors upstream's shared `DelegatedHandler`). |
| Arrow | `arrow` (arrow-rs) 58.3.0 | `arrow2` 0.18.0 | `arrow2` is effectively unmaintained (superseded; the ecosystem consolidated on official `arrow-rs`). Use `arrow`. |
| SmallVec | `smallvec` 1.15.1 | `smallvec` 2.0.0-alpha | 2.0 is alpha-only — **never** in a fidelity/memory-critical port. |
| approx | `approx` 0.5.1 | `approx` 0.6.0-rc2 | Pre-release; do not gate equivalence on an RC. |
| Allocator | `tikv-jemallocator` 0.7.0 | `jemallocator` 0.5 (manual) / `mimalloc` | `jemallocator` is the older unmaintained crate; `tikv-jemallocator` is the maintained fork. mimalloc is the portability fallback. |
| Float compare | `approx` | hand-rolled `(a-b).abs() < eps` | `approx` handles NaN/inf/relative-vs-abs correctly and reads as intent in the harness. |
| LightGBM parse | hand-rolled `getline`/`parse` | `nom`/`winnow` grammar | The LightGBM text format is `key=value` + space-separated arrays — a parser-combinator grammar is overkill and risks diverging from upstream's exact `strtod` line semantics. Port `lightgbm.cc` line-for-line. |
| Tree storage | plain `Vec`/`Box<[T]>` SoA | `arrow` arrays | Arrow's null bitmaps/offsets add overhead and indirection for fixed-width node columns; plain owned slices are leaner and bit-faithful. Reserve Arrow for *input matrices* and Python interchange. |

---

## Installation

```bash
# core libs
cargo add --package treelite-core thiserror serde serde_json byteorder bytemuck smallvec compact_str half
# compute
cargo add --package treelite-gtil cubecl --features cpu
# python binding
cargo add --package treelite-py pyo3 --features extension-module,abi3-py38
# dev (workspace)
cargo add --dev proptest approx criterion insta
# allocator (bin/bench targets)
cargo add --package <bin> tikv-jemallocator
# build python wheel
pip install "maturin>=1.13,<2" && maturin develop -m crates/treelite-py/Cargo.toml
```

---

## Confidence Assessment

| Item | Confidence | Notes |
|------|------------|-------|
| All pinned versions | HIGH | Verified live against crates.io sparse index 2026-06-09; PyO3/maturin via Context7. |
| cubecl 0.10.0 + backend features | HIGH | Feature table read directly from 0.10.0 index metadata; no default backend confirmed. |
| serde_json over simd-json | MEDIUM-HIGH | Reasoned from upstream SAX design + fidelity priority; loaders not the hot path. |
| Hand-rolled UBJSON | HIGH | `ubjson` 0.1.0 immaturity confirmed; upstream shared-handler design confirmed in source. |
| arrow 58.3.0 (not arrow2) | HIGH | Index confirms 58.3.0 latest; arrow2 consolidation is well-established. |
| Allocator choice | MEDIUM | Equivalence-neutral; perf claim is reasoned, not benchmarked here. |
| half/f16 1e-5 caveat | HIGH | f16 ~1e-3 error is intrinsic; harness must use f32/f64 path. |
| PyO3 abi3/edition-2024 | HIGH | Confirmed via Context7. |

## Risk Flags (for roadmap)

- **f16 path is NOT 1e-5-compliant** — keep it an explicit opt-in; never validate equivalence on it. (HIGH importance)
- **smallvec 2.0 / approx 0.6 are pre-release** — pin the stable versions above; a careless `cargo add` may grab the alpha/RC. (HIGH)
- **`ubjson` crate is a trap** — do not adopt it; hand-roll over `byteorder`. (HIGH)
- **arrow major-version churn** — arrow-rs ships ~monthly majors (58.x now); pin exactly and bump deliberately, since it is on the GPU-ingestion path. (MEDIUM)
- **cubecl is pre-1.0 (0.10.0)** — API may shift between minors; pin exactly and isolate all cubecl usage behind the `treelite-gtil` runtime-dispatch layer so an upgrade touches one crate. (MEDIUM)

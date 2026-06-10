# Phase 5: Full Scalar GTIL & Equivalence Harness - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 12 (4 gtil src, 1 harness src, 1 harness test, 1 capture script + new siblings)
**Analogs found:** 12 / 12 (every touched file is widened in place — its closest analog is itself plus a same-crate sibling)

> This phase **widens existing code in place**; it is NOT greenfield. For every file the closest analog is the file being extended. The executor must replicate the *existing local idioms* (the `PredictScalar` trait shape, `Shape`/`idx` SoA dispatch, `category_list_safe`/`has_leaf_vector` bounds-safe accessors, thiserror variant style, the local-serde golden-assert test, the `_manifest()`/`_payload_sha256()`/`_write_golden()` capture-script trio) — NOT invent new conventions. The upstream `treelite-mainline/src/gtil/*.cc` is the verbatim numeric source; the in-repo Rust files are the *style* source.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/treelite-gtil/src/lib.rs` (modify) | service (compute) | request-response (predict) | itself (existing `predict`/`predict_preset`/`evaluate_tree`) | exact (self) |
| `crates/treelite-gtil/src/postprocessor.rs` (modify) | utility (math) | transform | itself (existing 7 verbatim postprocessors) | exact (self) |
| `crates/treelite-gtil/src/error.rs` (modify) | model (error enum) | — | itself (existing `GtilError` variants) | exact (self) |
| `crates/treelite-gtil/src/config.rs` (NEW) | config / model | — | `error.rs` enum style + upstream `gtil.h`/`config.cc` | role-match |
| `crates/treelite-gtil/src/shape.rs` (NEW) | utility (descriptor) | transform | internal `Shape` in `lib.rs` + upstream `output_shape.cc` | role-match |
| `crates/treelite-gtil/src/accessor.rs` (NEW, optional) | utility (input view) | transform / file-I/O-shaped | `evaluate_tree` row-slicing in `lib.rs` + upstream `SparseMatrixAccessor` | role-match |
| `crates/treelite-harness/src/lib.rs` (modify) | service (test instrument) | request-response (golden assert) | itself (`Golden`/`Manifest`/`run_equivalence`/`check_manifest`) | exact (self) |
| `crates/treelite-harness/tests/gtil_matrix.rs` (NEW) | test | request-response (matrix golden assert) | `tests/lightgbm.rs` (local-serde golden-assert) | exact |
| `fixtures/capture_gtil_matrix.py` (NEW) | test (capture) | batch / file-I/O | `fixtures/capture_lightgbm.py` | exact |
| `fixtures/gtil/*.golden.json` (NEW) | config (frozen data) | file-I/O | `fixtures/lightgbm_*.golden.json` | exact |
| `fixtures/gtil/*.manifest.json` (NEW) | config (frozen data) | file-I/O | `fixtures/xgb_3format.manifest.json` / `golden_v5.manifest.json` | exact |

---

## Pattern Assignments

### `crates/treelite-gtil/src/lib.rs` (service, request-response) — WIDEN IN PLACE

**Analog:** itself. The serial-sum / four-way leaf routing / RF averaging / f64 base-score-add spine is the 1e-5-proven core — do **not** rewrite (RESEARCH Anti-Pattern). New surface: `O`-generic output, sparse path, two new kinds, full categorical guard, `Config` dispatch.

**Output-type generic pattern — extend the existing `PredictScalar` trait idiom** (`lib.rs:28-55`). The existing trait reifies the *threshold/leaf* domain `T`. The new orthogonal `O` (input/output element) trait must follow the same shape — concrete `impl`s for `f32` and `f64`, `#[inline]` methods, doc-comment citing the upstream line that justifies each cast:
```rust
// EXISTING idiom to mirror (lib.rs:28-55):
pub trait PredictScalar: Copy {
    fn from_f32(v: f32) -> Self;   // lift feature into threshold domain
    fn to_f32(self) -> f32;        // static_cast<InputT>(LeafValue) at predict.cc:228
}
impl PredictScalar for f64 {
    #[inline] fn from_f32(v: f32) -> Self { v as f64 }
    #[inline] fn to_f32(self) -> f32 { self as f32 }
}
```
The new `O: PredictOut` trait is the *output/accumulator* element (f32 input → `Vec<f32>`, f64 input → `Vec<f64>`), independent of preset (RESEARCH Pattern 1 / Pitfall 1; A1 confirmed by `c_api/gtil.cc:50-55` and `predict.cc:236` `Array3DView<InputT>`). The current body hardcodes `f32` output in `predict_preset` (`lib.rs:291-378`, `output_leaf_value` `:403`, `output_leaf_vector` `:438`, base-score add `:371`); each `to_f32()` accumulation site and the `vec![0.0_f32; ...]` allocation become `O`-generic. **Keep cross-comparison semantics**: `NextNode<InputT,ThresholdT>` promotes to the wider type (C++ usual arithmetic conversions) — preserve via the trait, do not collapse.

**Categorical full-guard pattern — replace the MINIMAL subset** (`lib.rs:94-120`, explicitly marked "full GTIL-06 is Phase 5"). Port the exact upstream formula verbatim (`predict.cc:127-150`):
```rust
// Source: predict.cc:135-138. digits: f32=24, f64=53.
// max_representable_int = min( O(u32::MAX), O(1u64 << O::DIGITS) )
// f32 → min(4294967295.0, 16777216.0) = 2^24 ; f64 → min(4294967295.0, 9.007e15) = 2^32-1
// reject: fvalue < 0 || fabs(fvalue) > max_representable_int  (BEFORE the `as u32` cast)
```
The existing `category_list_safe` bounds-safe slice helper (`lib.rs:131-147`) and the `category_list_right_child` polarity block (`lib.rs:113-119`) are correct — keep them; only the guard arithmetic widens to the `O`-generic representability check (RESEARCH Pitfall 3 / Code Examples).

**Sparse accessor pattern — NEW, mirrors `SparseMatrixAccessor::GetRow`** (`predict.cc:58-97`). Per-row NaN-materialized scratch so dense==sparse parity is *structural* (D-04) and `evaluate_tree` is reused verbatim:
```rust
// Source: predict.cc:80-85 — fill NaN, then write nonzeros at col_ind.
for v in scratch.iter_mut() { *v = O::nan(); }            // absent = NaN (SC1)
let (b, e) = (row_ptr[row] as usize, row_ptr[row+1] as usize);
for i in b..e { scratch[col_ind[i] as usize] = data[i]; }
```
Single scratch row (scalar single-thread); bounds-check `col_ind[i] < num_feature` and `row_ptr` monotone → typed error, never panic (Security Domain table; matches the existing `FeatureIndexOutOfBounds`/`NodeIndexOutOfBounds` posture at `lib.rs:168,204`).

**Per-kind dispatch pattern — NEW `match config.kind`** mirroring `PredictImpl` (`predict.cc:380-396`). The existing `predict` (`lib.rs:519-582`) is the `default` body (PredictRaw + `apply_postprocessor`); the new dispatch wraps it:
- `Default` → existing `predict_preset` + `apply_postprocessor`
- `Raw` → existing `predict_preset`, skip postprocessor
- `LeafId` → NEW `predict_leaf`: `output_view(row,tree) = leaf_id` cast into the `O` buffer, **no** postprocess/average/base-score (`predict.cc:325-345`; A4)
- `ScorePerTree` → NEW `predict_score_by_tree`: `if has_leaf_vector → write each element; else → write scalar at index 0`, **no** postprocess/average/base-score (`predict.cc:347-378`; RESEARCH Pitfall 5). Reuse the existing `has_leaf_vector` helper (`lib.rs:261-268`).

**Postprocessor dispatch — extend the existing `apply_postprocessor` match** (`lib.rs:592-648`). Add `"signed_square"`, `"hinge"`, `"multiclass_ova"` arms (removing them from the `UnsupportedPostprocessor` fallthrough at `:645`). `multiclass_ova` is row-wise per `(row,target)` over `num_class` cells — copy the exact loop structure of the existing `"softmax"` arm (`lib.rs:632-644`).

**Error-handling pattern (existing — preserve):** every malformed-route/index path returns a typed `GtilError` (`lib.rs:168,204,395,434`); never panic (ERR-01 / ASVS V5). New sparse/kind code adds variants in the same style.

---

### `crates/treelite-gtil/src/postprocessor.rs` (utility, transform) — EXTEND VERBATIM

**Analog:** itself. The 7 existing postprocessors are the cast-order template. Each is a free `fn` taking `f32`, with a doc-comment quoting the upstream C++ and stating the cast-ordering contract.

**Existing verbatim idiom to copy** (`postprocessor.rs:48-50`, `75-77`, `112-132`):
```rust
// sigmoid — f32 throughout, no double promotion (postprocessor.cc:33-37)
pub fn sigmoid(sigmoid_alpha: f32, v: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-sigmoid_alpha * v).exp())
}
// exponential_standard_ratio — note exp2 (BASE-2), not exp (postprocessor.cc:44-47)
pub fn exponential_standard_ratio(ratio_c: f32, v: f32) -> f32 { (-v / ratio_c).exp2() }
```

**Three NEW postprocessors to add (port `postprocessor.cc:22-31, 78-82` verbatim):**
```rust
// signed_square: copysign(margin*margin, margin)  (postprocessor.cc:22-26)
pub fn signed_square(v: f32) -> f32 { (v * v).copysign(v) }
// hinge: 1 if >0 else 0  (postprocessor.cc:28-31)
pub fn hinge(v: f32) -> f32 { if v > 0.0 { 1.0 } else { 0.0 } }
// multiclass_ova: per-class sigmoid (NOT softmax), sigmoid_alpha is float (postprocessor.cc:77-82)
pub fn multiclass_ova(sigmoid_alpha: f32, row: &mut [f32]) {
    for c in row.iter_mut() { *c = 1.0_f32 / (1.0_f32 + (-sigmoid_alpha * *c).exp()); }
}
```

**CRITICAL — softmax/sigmoid intermediates stay `f32` even on the f64-input path** (RESEARCH Pitfall 2 / A2). `softmax` (`postprocessor.rs:112-132`) hardcodes `float max_margin`/`float t` with `f64 norm_const` (`postprocessor.cc:59-61`); `sigmoid_alpha`/`ratio_c` are `float` model fields. When the predict body becomes `O`-generic, these postprocessor intermediates must **not** be promoted to `O` — keep the upstream-literal types. The existing `softmax` mixed-precision body is already correct; do not touch its reduction-scalar types.

**Test idiom to extend** (`postprocessor.rs:134-203`): each new postprocessor gets a `#[cfg(test)]` unit test with a hand-computed reference and a `< 1e-6`/`< 1e-7` assert, exactly like `exponential_standard_ratio_uses_base2_exp2` (`:138-150`).

---

### `crates/treelite-gtil/src/error.rs` (model, error enum) — EXTEND

**Analog:** itself. Existing variant style: `#[derive(Debug, Error, PartialEq, Eq)]`, struct variants with named fields, `#[error("...")]` with interpolated fields, doc-comment per field citing the threat (T-03-01 etc.) and the upstream unchecked path.

**Existing variant template** (`error.rs:16-33`):
```rust
/// A node's `split_index` is outside the feature row. ... typed error (T-03-01).
#[error("feature index {feature} at node {node} is out of bounds (num_feature = {num_feature})")]
FeatureIndexOutOfBounds { node: usize, feature: i32, num_feature: i32 },
```

**NEW variants to add (same style):**
- Sparse `col_ind[i] >= num_feature` → OOB scratch write (Security Domain table).
- Non-monotonic / out-of-range `row_ptr`; `row_ptr[num_row] > data.len()`.
- (Optional) unsupported/unknown predict kind if kind can ever arrive untyped — but `PredictKind` being a Rust enum (D-06) likely makes this unrepresentable; prefer the type system over a variant.

Keep `#[error(transparent)] Core(#[from] treelite_core::CoreError)` (`error.rs:91-92`) and the `UnsupportedPostprocessor(String)` variant (`:87-88`) — the latter shrinks by 3 names as the new postprocessors are implemented.

---

### `crates/treelite-gtil/src/config.rs` (config) — NEW

**Analog:** the `error.rs` enum style + upstream `gtil.h:26-52` / `config.cc:25-41`. Idiomatic, JSON-free (D-06).

```rust
// Source: gtil.h:31-46 (PredictKind values 0..3), gtil.h:51-52 (nthread{0}, pred_kind default)
pub enum PredictKind { Default, Raw, LeafId, ScorePerTree }   // kPredictDefault=0 .. kPredictPerTree=3
pub struct Config { pub kind: PredictKind, pub nthread: i32 } // Default kind, nthread 0
```
`nthread <= 0` ⇒ "all threads" upstream (`threading_utils.h:74-80`); the scalar reference is single-threaded so `nthread` is **accepted-and-recorded, not used to allocate** (RESEARCH Pitfall 6; no DoS amplification). Provide a `Default` impl matching `gtil.h:51-52`. **No JSON parsing** in this crate — any compat shim lives at the Phase-8 PyO3 edge.

---

### `crates/treelite-gtil/src/shape.rs` (utility, descriptor) — NEW (public `Shape` + `output_shape`)

**Analog:** the existing *internal* `Shape<'m>` in `lib.rs:217-252` (the predict-internal `(num_row, num_target, max_num_class)` indexer with `idx()`/`cells_per_row()`/`num_class_of()`). RESEARCH Open Q3: **rename the internal one** (e.g. `OutputLayout`) to avoid clashing with the new *public* per-kind `Shape` descriptor returned to callers (D-07, Phase-8 numpy reshape).

**Port `GetOutputShape` verbatim** (`output_shape.cc:17-39`):
```rust
// default/raw → num_target>1 ? (r, num_target, max_num_class) : (r, 1, max_num_class)
// leaf_id     → (r, num_tree)
// per_tree    → (r, num_tree, leaf_vector_shape[0] * leaf_vector_shape[1])
```
Use core accessors: `model.num_tree()` (model.rs:186), `model.num_class` (`:68`), `model.num_target` (`:66`), `model.leaf_vector_shape` (`:70`). `max_num_class = max over num_class[..num_target]` — mirror the clamp-to-1 already in `lib.rs:556`. Note `default`/`raw` collapse `num_target==1` to dim `1` (not omitted).

---

### `crates/treelite-harness/src/lib.rs` (service, golden assert) — EXTEND

**Analog:** itself. The `Golden { input, output, manifest }` struct (`lib.rs:76-85`), `Manifest` (`:92-108`), `run_equivalence` max-`|delta|` loop (`:189-233`), `check_manifest` OS/arch drift warning (`:242-263`), and the `NanF32` + `normalize_nan_tokens` machinery (`:36-173`) for Python's bare-`NaN` token.

**Manifest `backend` field — extend the existing struct** (`lib.rs:92-108`). Add a `backend: String` (or a `Backend` enum) field plus the new provenance keys (rustc, cubecl placeholder, per-framework versions, seed, sha256) in the same `#[serde(default)]`-tolerant style as the existing optional `xgboost`/`python` fields (`:97-107`):
```rust
// EXISTING shape to extend (lib.rs:92-108):
pub struct Manifest {
    pub treelite: String,
    #[serde(default)] pub xgboost: Option<String>,
    pub os: String, pub arch: String,
    pub libc: serde_json::Value,
    #[serde(default)] pub python: Option<String>,
    // NEW: backend ("scalar-cpu" this phase), rustc, cubecl (placeholder), seed, sha256, framework versions
}
```

**Backend-parameterized seam (D-11, design only).** RESEARCH Pattern 4 — the minimal seam, NOT a trait-object hierarchy:
```rust
pub enum Backend { ScalarCpu /* , CubeclCpu, Cuda, Wgpu, Rocm (Phase 6/7) */ }
type PredictFn = fn(&Model, &[f64], usize, &Config) -> anyhow::Result<Vec<f64>>;
struct RunnerCase { backend: Backend, predict: PredictFn, /* + sparse fn */ }
```
Phase 6 registers `(CubeclCpu, cubecl_predict)` without touching the matrix iteration. **No cubecl code this phase.**

**Max-deviation report pattern (existing — generalize across the matrix)** (`lib.rs:220-232`):
```rust
let mut max_dev: f64 = 0.0;
for (i, &got) in rust.iter().enumerate() {
    let delta = (got - golden.output[i]).abs() as f64;
    if delta > max_dev { max_dev = delta; }
    approx::assert_abs_diff_eq!(got, golden.output[i], epsilon = 1e-5); // HARD gate — never loosen
}
```

---

### `crates/treelite-harness/tests/gtil_matrix.rs` (test) — NEW

**Analog:** `tests/lightgbm.rs` (read in full). Copy its exact skeleton:
- `fixture_path(name)` / `workspace_path(rel)` resolvers via `env!("CARGO_MANIFEST_DIR")` joined to `../../fixtures` / `../..` (`lightgbm.rs:58-74`).
- Local `#[derive(Deserialize)]` golden struct (`LgbGolden`/`LgbManifest`, `lightgbm.rs:30-54`) — RESEARCH §State of the Art notes the move to a *uniform* schema; this new test uses the extended shared `Golden`/`Manifest` from harness `lib.rs` where possible, falling back to the local-serde idiom for matrix-specific fields.
- `check_manifest(&manifest)` drift warning call (`lightgbm.rs:79-100`).
- Load model → flatten golden input to a flat buffer → `treelite_gtil::predict` → length assert → max-`|delta|` loop with `approx::assert_abs_diff_eq!(epsilon = 1e-5)` → `eprintln!("... max |delta| = {max_dev:e}")` (`lightgbm.rs:104-160`).
- `.map_err(|e| anyhow::anyhow!("{e}")).context(...)` to bridge typed `GtilError` → `anyhow` (`lightgbm.rs:118-138`).

**NEW matrix-specific additions:** iterate the exhaustive cross-product (model × preset × in-dtype × kind × {dense,sparse} × seed, D-01/D-03); add the **dense==sparse parity assertion** (D-04: run both Rust paths on identical logical data, assert equal); both f32 and f64 input dtypes (D-05).

---

### `fixtures/capture_gtil_matrix.py` (test, batch/file-I/O) — NEW

**Analog:** `fixtures/capture_lightgbm.py` (read in full). Copy its structure verbatim:
- Module docstring stating "run ONCE on main worktree, commit, CI NEVER regenerates" (`capture_lightgbm.py:1-35`; D-08).
- `_manifest(extra=None)` builder pinning `treelite/<framework>/numpy/python/os/arch/seed` (`:56-68`) — **extend with the `backend: "scalar-cpu"` field + rustc/cubecl placeholders** (D-09).
- `_payload_sha256(input_list, output_list)` (`:71-77`) and `_write_golden(name, payload)` (`:80-85`).
- `SEED = 1234`, `np.random.RandomState(SEED)`, `treelite.gtil.predict(model, X)` as the frozen golden (NOT framework predict) (`:106-119`).
- Capture-time signature assert on `treelite.gtil.predict` (`capture_lightgbm.py:191-198`).

**NEW capture additions:**
- Edge-seeded wide matrices (NaN, ±inf, boundary thresholds, `2**24+1` f32 categorical-gap value) — RESEARCH Code Examples / Pitfall 3.
- Both `np.float32` and `np.float64` input matrices (D-05).
- All 4 kinds: `treelite.gtil.predict` (default), `predict(pred_margin=True)` (raw), `predict_leaf` (leaf_id), `predict_per_tree` (score_per_tree).
- **Sparse-with-NaN construction (RESEARCH Open Q1 — resolve here):** build a presence mask, dense matrix with NaN in absent positions, CSR from present positions, then **assert at capture time** `treelite.gtil.predict(dense_nan) == treelite.gtil.predict(csr)` and freeze both.

---

### `fixtures/gtil/*.golden.json` + `*.manifest.json` (config, frozen data) — NEW

**Analogs:** `fixtures/lightgbm_categorical.golden.json` (payload shape: `{model_path, n_features, input, output, output_shape, manifest, sha256}`), `fixtures/xgb_3format.manifest.json` / `fixtures/golden_v5.manifest.json` (manifest shape). Freeze read-only; never hand-edit (D-08). New subdir `fixtures/gtil/` keeps the root tidy (RESEARCH Recommended Structure). Naming: `<model>.<preset>.<indtype>.<kind>.<dense|sparse>.s<seed>.golden.json`.

---

## Shared Patterns

### Verbatim cast-order / 1e-5 contract
**Source:** `crates/treelite-gtil/src/postprocessor.rs` (the whole module) + upstream `postprocessor.cc:19-115`.
**Apply to:** every new postprocessor and every `O`-generic accumulation site.
Doc-comment each ported fn with the upstream C++ snippet and an explicit "runs in fXX — no promotion" note. `softmax`/`sigmoid`/`ova` reduction scalars stay `f32` regardless of the input dtype (Pitfall 2).

### Bounds-safe accessor → typed error (never panic)
**Source:** `crates/treelite-gtil/src/lib.rs` — `category_list_safe` (`:131-147`), `has_leaf_vector` (`:261-268`), the `FeatureIndexOutOfBounds`/`NodeIndexOutOfBounds` guards (`:167-205`).
**Apply to:** the new sparse `col_ind`/`row_ptr` validation and the new `score_per_tree` leaf-vector access. Every upstream unchecked index becomes a `Result<_, GtilError>` (ASVS V5 / ERR-01).

### thiserror typed-error variant style
**Source:** `crates/treelite-gtil/src/error.rs:12-93`.
**Apply to:** all new `GtilError` variants — `#[derive(Debug, Error, PartialEq, Eq)]`, named struct fields, `#[error("...")]` interpolation, per-field doc citing the threat ID.

### Golden-assert test skeleton
**Source:** `crates/treelite-harness/tests/lightgbm.rs:58-160` (resolvers, local-serde golden, `check_manifest`, flatten, predict, max-`|delta|` loop, hard `1e-5` gate).
**Apply to:** `tests/gtil_matrix.rs`.

### Capture-script trio + frozen-fixture discipline
**Source:** `fixtures/capture_lightgbm.py` — `_manifest()` / `_payload_sha256()` / `_write_golden()` (`:56-85`), `SEED`/`RandomState`, `treelite.gtil.predict` golden, run-once docstring.
**Apply to:** `fixtures/capture_gtil_matrix.py`. Run via `uv run python` on the **main tree** (venv untracked, absent from worktrees — per MEMORY.md).

### Max-deviation report (EQV-04)
**Source:** `crates/treelite-harness/src/lib.rs:220-232` and `tests/lightgbm.rs:147-158`.
**Apply to:** every matrix cell; print `max |delta| = {:e}` per cell; the `1e-5` epsilon is a HARD gate, never loosened.

---

## No Analog Found

None. Every file in this phase is either widened in place or has an exact same-crate sibling analog. The two "newest" surfaces — the `O`-generic output trait and the backend-parameterized harness seam — both have a close in-repo style analog (`PredictScalar` trait; the existing `run_equivalence` runner) plus an explicit minimal shape prescribed in RESEARCH (Pattern 1, Pattern 4). The numeric behavior of every new code path has a verbatim upstream source in `treelite-mainline/src/gtil/`.

---

## Metadata

**Analog search scope:** `crates/treelite-gtil/src/`, `crates/treelite-harness/src/`, `crates/treelite-harness/tests/`, `crates/treelite-core/src/model.rs`, `fixtures/`, and the vendored `treelite-mainline/src/gtil/` + `include/treelite/gtil.h` (verbatim numeric source).
**Files scanned:** ~16 (4 gtil src read in full; harness src + lightgbm test read in full; capture_lightgbm.py read in full; upstream output_shape.cc/postprocessor.cc read in full; predict.cc accessor/kind/dispatch sections; gtil.h/config.cc enum+nthread; core model.rs field grep).
**Pattern extraction date:** 2026-06-10

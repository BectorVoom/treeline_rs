# Phase 1: End-to-End Spine - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up the **thinnest possible end-to-end pipeline** through the whole architecture so the 1e-5 core value is proven on day one: a Cargo workspace + the four upstream enums + a minimal SoA `Model`/`Tree` core + a minimal XGBoost-JSON loader + a scalar single-threaded predict (identity/sigmoid only) + an equivalence-harness skeleton asserting the output is within 1e-5 of a committed golden vector.

**In scope:** workspace scaffolding (edition 2024, resolver "3", pinned `[workspace.dependencies]`); `TaskType`/`TreeNodeType`/`Operator`/`DType` with upstream string round-trip; two-variant `Model` enum `<f32,f32>`/`<f64,f64>`; `Tree<T>` SoA `TreeBuf<T>` (owned + borrowed) carrying full header metadata; one minimal XGBoost-JSON load; scalar identity/sigmoid predict; 1e-5 equivalence harness with frozen toolchain/libm manifest; `thiserror` (libs) + `anyhow` (harness/bins).

**Out of scope (later phases):** `ModelBuilder`/serialization (Phase 2); UBJSON + legacy-binary XGBoost + auto-detect (Phase 3); LightGBM/sklearn (Phase 4); full GTIL surface — 4 predict kinds, 10 postprocessors, sparse CSR, categoricals, output shaping (Phase 5); cubecl kernels (Phase 6); GPU (Phase 7); PyO3 (Phase 8); memory hardening — bytemuck/smallvec/compact_str/allocator (Phase 9).

</domain>

<decisions>
## Implementation Decisions

### Crate Layout
- **D-01:** Workspace is **spine-only, grown one layer per phase** — create only the crates Phase 1 exercises, not stubs for all 9 phases. Aligns with the MVP-slice roadmap; no dead empty crates.
- **D-02:** Initial members (names indicative, planner may refine):
  - `treelite-core` — enums (`TaskType`, `TreeNodeType`, `Operator`, `DType`) + `Model` enum + `Tree<T>` + `TreeBuf<T>` SoA columns + header metadata.
  - `treelite-gtil` — scalar single-threaded predict (identity/sigmoid only).
  - `treelite-xgboost` — minimal XGBoost-JSON loader.
  - `treelite-harness` — 1e-5 equivalence harness (dev/test-facing).
- **D-03:** Single root `[workspace.dependencies]` table; every third-party crate pinned to a current stable version, no pre-release on the critical path (FND-01/FND-02).
- **Rejected:** "Scaffold all 9-phase crates now" (premature structure, empty crates); "Single crate, split later" (conflicts with FND-01's multi-crate-from-Phase-1 requirement).

### Fixture Model
- **D-04:** The Phase 1 fixture is a **hand-crafted XGBoost-JSON literal committed to the repo** — no runtime dependency on the `xgboost` package to produce the model itself.
- **D-05 (CONSTRAINT — critical coupling):** Because the golden is captured by loading this fixture into the **upstream Treelite Python wheel** (D-06), the hand-crafted JSON **must be valid enough for upstream Treelite/XGBoost to parse**. The fixture is NOT free-form: it must conform to the XGBoost-JSON schema (learner / gradient_booster / model / trees structure, `objective`, `base_score`, etc.) closely enough that `treelite` can load it. Use `binary:logistic` so the **sigmoid** postprocessor path is genuinely exercised; keep it minimal (1–2 shallow trees).

### Golden Vector Capture
- **D-06:** The first golden is **captured from the upstream Treelite Python wheel's GTIL** (`pip install treelite==<matching 4.x>`, load fixture, run `treelite.gtil.predict`), then committed and frozen. No C++ source compile; CI never regenerates it. This is the authoritative GTIL source and satisfies PROJECT's "golden frozen from upstream Treelite."
- **D-07:** Commit a **toolchain/libm manifest** alongside the golden recording at minimum: treelite wheel version, OS/arch, libm/glibc version, and (if used) xgboost version. The manifest is part of the frozen artifact, per the Phase-5 blocker note ("store actual input matrices + a toolchain/libm/framework manifest, not just seeds").
- **Rejected:** building C++ Treelite from `treelite-mainline/` (heaviest setup, deferred); reusing upstream checked-in expected outputs (**none exist** — `tests/examples/` ships model files only, no golden vectors).

### Predict-Path Altitude
- **D-08:** Phase 1 predict is the **simplest plain scalar single-threaded function — no backend/`Predictor` trait abstraction**. The cubecl seam is deferred to Phase 6, which is research-flagged and will spike kernel shape first; designing the abstraction now risks the wrong boundary. Keep the spine genuinely thin.

### Claude's Discretion
- Exact crate names/granularity within the spine-only constraint (e.g., whether enums get their own crate vs living in `treelite-core` — default: in `treelite-core`).
- Rust representation mechanism for `TreeBuf<T>` owned-vs-borrowed mode (e.g., `Cow`, enum, custom) — implementation detail, must support zero-copy borrow (CORE-03).
- Error-enum granularity (per-crate `thiserror` enums vs shared) — default: per-crate, idiomatic.
- `DType` numeric-type coverage for Phase 1 (must match upstream `TypeInfo` string values; ENUM-01).
- Whether CI (GitHub Actions) is wired in Phase 1 or deferred — success criterion only requires `cargo build`/`cargo test` to pass.
- Exact manifest file format (TOML/JSON/markdown) and location.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project-level (this milestone)
- `.planning/PROJECT.md` — core value (1e-5 equivalence), constraints, Key Decisions table, Out of Scope (no C-API, v5-only serialization, golden frozen from upstream).
- `.planning/REQUIREMENTS.md` — Phase 1 IDs: FND-01, FND-02, ENUM-01, CORE-01..04, ERR-01, ERR-02 (full text + traceability).
- `.planning/ROADMAP.md` § "Phase 1: End-to-End Spine" — goal + 5 success criteria (the authoritative acceptance bar).

### Upstream porting source of truth (`treelite-mainline/`, C++ v4.7.0)
- `treelite-mainline/include/treelite/enum/` + `treelite-mainline/src/enum/` — `TaskType`, `TreeNodeType`, `Operator`, `TypeInfo` — **source of the exact string values `DType`/enums must round-trip to** (ENUM-01 asserts against these).
- `treelite-mainline/include/treelite/tree.h` + `treelite-mainline/include/treelite/detail/tree.h` — `Tree<T,L>`, `ModelPreset<T,L>`, `Model` variant, SoA node columns + the ~20 node fields and full header metadata (CORE-01..04).
- `treelite-mainline/include/treelite/contiguous_array.h` — owned/borrowed buffer semantics to port into `TreeBuf<T>` (CORE-03 zero-copy borrowed mode).
- `treelite-mainline/src/model_loader/` (XGBoost JSON path) + `treelite-mainline/include/treelite/model_loader.h` — XGBoost-JSON parsing reference for the minimal loader; objective→postprocessor mapping (sigmoid).
- `treelite-mainline/src/gtil/predict.cc` + `treelite-mainline/src/gtil/postprocessor.cc` + `treelite-mainline/include/treelite/gtil.h` — scalar traversal + identity/sigmoid postprocessor reference (port serial scalar slice only).

### Codebase maps
- `.planning/codebase/ARCHITECTURE.md` — SoA + variant pattern, layer dependencies, anti-patterns (no `Model`/`Tree` copy; `ModelBuilder` validation — relevant from Phase 2 on).
- `.planning/codebase/STACK.md` — confirms greenfield Rust crate, zero deps yet; upstream dependency map.
- `.planning/codebase/CONVENTIONS.md` — naming + error-handling translation notes.

### Reference manuals (for later phases; noted now, not Phase 1 critical)
- `/home/user/Documents/workspace/optimisor/manual/` — memory-efficiency playbook (Phase 9).
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — kernel authoring (Phase 6+).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Upstream C++ is the spec, not reusable code** — `treelite-mainline/` is vendored read-only and is the porting source of truth. No Rust assets exist yet (greenfield; `src/main.rs` is a `fn main()` stub to be replaced/removed by the workspace).
- **Vendored test corpus** `treelite-mainline/tests/examples/` — `mushroom` (XGBoost **legacy binary** — Phase 3, NOT usable for Phase 1 JSON), `deep_lightgbm`, `sparse_categorical`, `toy_categorical` (LightGBM text — Phase 4). **No XGBoost-JSON model and no golden vectors ship here** — confirmed; this is why D-04/D-06 produce both.

### Established Patterns (to preserve from upstream)
- Struct-of-Arrays tree storage (parallel columns, not a node struct) → `TreeBuf<T>`.
- Type-erased `Model` over `<f32,f32>`/`<f64,f64>` — C++ `std::variant` → Rust two-variant `enum` (no mixed threshold/leaf types).
- `thiserror` typed errors at library API boundaries; `anyhow` for context in harness/bins (ERR-01/ERR-02).

### Integration Points
- This phase creates the workspace root from scratch — `src/main.rs` stub gives way to `crates/*`. All later phases build on these crate seams.

</code_context>

<specifics>
## Specific Ideas

- Fixture should use `binary:logistic` objective specifically so the **sigmoid** postprocessor path is exercised end-to-end (identity is the trivial baseline; sigmoid is the real fidelity check for Phase 1).
- Golden artifact = output vector **+ input matrix + manifest**, committed together as a frozen unit (mirrors the Phase 5 reproducibility blocker — don't store seeds alone).

</specifics>

<deferred>
## Deferred Ideas

- **Backend/`Predictor` trait abstraction** — explicitly deferred to Phase 6 (cubecl), where kernel shape will be spiked first (per D-08 and the Phase 5/6 cubecl blocker note).
- **Real-world example fixtures** (mushroom legacy binary, LightGBM text examples) — used from Phase 3/4 once their loaders exist.
- **GitHub Actions CI** — not required by Phase 1 success criteria; can be added here or later at Claude's discretion.
- **serde_json NaN/Inf handling** — known Phase 3 XGBoost-JSON blocker; the Phase 1 hand-crafted fixture should avoid NaN/Inf literals so the minimal loader needn't solve this yet.

None of the above is scope creep into Phase 1 — all are recorded so they aren't lost.

</deferred>

---

*Phase: 1-End-to-End Spine*
*Context gathered: 2026-06-09*

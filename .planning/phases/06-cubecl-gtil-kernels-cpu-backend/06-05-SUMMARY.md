---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 05
subsystem: harness-backend-registration
tags: [cubecl, backend-registration, gtil-matrix, provenance, determinism, sc2, d-06, d-11, wave-5, GPU-01, GPU-02, GPU-05]
requires:
  - phase: 06-04
    provides: "treelite_cubecl::predict_cpu::<f32>/::<f64> real host launcher (validate->categorical/sparse scalar fallback->upload->select-by-kind->launch->read) matching treelite_gtil::predict within 1e-5 across all 4 kinds on f32 AND f64 input"
provides:
  - "treelite_harness::Backend::CubeclCpu + cubecl_cpu_case() — the D-11 registration into the RunnerCase seam: dense f32/f64 -> treelite_cubecl::predict_cpu (D-01/D-05, result widened after predict per Pitfall 6); sparse f32/f64 -> treelite_gtil::predict_sparse scalar fallback (D-02). RunnerCase struct + slot aliases + gtil_matrix.rs untouched."
  - "tests/gtil_matrix_cubecl.rs — the GPU-02 sibling gate: drives the SAME 96 frozen fixtures/gtil/ goldens through cubecl to 1e-5 (48 f32-input + 48 f64-input; global max |delta| = 2.9e-6) with per-cell EXECUTED-path provenance (48 cubecl-kernel dense + 48 scalar-fallback sparse, D-06/T-06-12) and a >=1-kernel-cell guard; never asserts the scalar-cpu manifest literal."
  - "crates/treelite-cubecl/tests/determinism.rs — SC2 two-run element-wise .to_bits() bit-identity on f32 AND f64 across all 4 predict kinds (T-06-13)."
affects:
  - "Phase 6 GPU-02 success criterion met and auditable cell-by-cell; future Phase-7 backends (Cuda/Wgpu/Rocm) register the same way (new Backend variant + a *_case() constructor)."
tech-stack:
  added: []
  patterns:
    - "Backend registration is purely additive (D-11): a new Backend enum variant + a *_case() RunnerCase constructor; the matrix iteration shape (gtil_matrix.rs) is never reshaped. The cubecl gate is a thin SIBLING file, not a refactor."
    - "Per-cell provenance (D-06) is recorded from the EXECUTED path at assertion time (dense numerical -> cubecl-kernel; sparse OR categorical-fallback -> scalar-fallback), never from the trusted manifest backend literal; a >=1-kernel-cell ensure prevents 1e-5-on-cubecl silently meaning validated-on-fallback (T-06-12)."
    - "SC2 determinism asserted via .to_bits() (not ==) so +0.0/-0.0 and NaN-payload differences are catchable; holds structurally because the kernels write disjoint per-row cells with no tree-axis reduction (06-04 grep gate)."
    - "treelite-cubecl is a regular [dependency] of treelite-harness (not a dev-dep) because cubecl_cpu_case() is a pub fn in src/lib.rs and must resolve at library-build time — mirroring treelite-gtil/treelite-xgboost."
key-files:
  created:
    - .planning/phases/06-cubecl-gtil-kernels-cpu-backend/06-05-SUMMARY.md
  modified:
    - crates/treelite-harness/Cargo.toml
    - crates/treelite-harness/src/lib.rs
    - crates/treelite-harness/tests/gtil_matrix_cubecl.rs
    - crates/treelite-cubecl/tests/determinism.rs
key-decisions:
  - "treelite-cubecl is registered as a regular [dependency] of treelite-harness, NOT a dev-dependency as the plan's <action> text literally said. cubecl_cpu_case() is a `pub fn` in src/lib.rs, so treelite-cubecl must be available at library-build time; a dev-dependency is only available to the test/example/bench targets, not the lib. The plan also said to mirror 'the existing treelite-gtil/treelite-xgboost dev-deps' — but those are in fact regular [dependencies], so a regular dependency IS the mirror. cargo build -p treelite-harness --tests confirms the lib compiles."
  - "manifest.rs was left unchanged. The plan listed it in <files> and described an optional warn-only drift hook, but the backend manifest field is already per-cell and #[serde(default)] (provenance-capable), and D-06 provenance is RECORDED at assertion time in the matrix sibling, not from the manifest. Adding a manifest mutation would have been net-zero behavior; the sibling carries its own check_manifest warn-only analog (copied from gtil_matrix.rs) for cross-backend drift visibility, never a gate."
  - "The small decode helpers (cell_to_f64/flatten_output/decode_input_f64/frozen_csr/assert_within/kind_of) were DUPLICATED into gtil_matrix_cubecl.rs rather than #[path]-included from gtil_matrix.rs. #[path]-including would re-run gtil_matrix.rs's #[test] fn gtil_matrix() inside this binary; duplication keeps gtil_matrix.rs byte-identical (D-11 git diff --stat == 0) and the sibling self-contained. Only the dtype/layout DISPATCH body is duplicated — the cross-product iteration shape is unchanged."
patterns-established:
  - "D-11 backend registration = new Backend variant + a *_case() RunnerCase constructor + a thin sibling matrix test; the frozen matrix runner is never reshaped."
  - "D-06 provenance recorded from the executed path with a >=1-true-kernel-cell guard (T-06-12)."
requirements-completed: [GPU-01, GPU-02, GPU-05]
duration: ~5min
completed: 2026-06-10
---

# Phase 6 Plan 05: cubecl-cpu Backend Registration + GPU-02 Matrix Gate + SC2 Determinism Summary

**The registration-not-refactor capstone (D-11): `Backend::CubeclCpu` + `cubecl_cpu_case()` register the cubecl CPU kernels (dense f32/f64) and the scalar fallback (sparse f32/f64) into the existing `RunnerCase` seam with NO change to the matrix iteration; the `gtil_matrix_cubecl.rs` sibling drives the SAME 96 frozen `fixtures/gtil/` goldens through cubecl to 1e-5 (global max |delta| = 2.9e-6) with per-cell `cubecl-kernel` vs `scalar-fallback` provenance (D-06/T-06-12), and `determinism.rs` proves two-run `.to_bits()` bit-identity (SC2). GPU-02 is met and auditable cell-by-cell.**

## Performance

- **Duration:** ~5 min
- **Tasks:** 2
- **Files modified:** 4 (1 created summary; Cargo.toml + lib.rs + 2 tests touched)
- **Tests:** gtil_matrix_cubecl 1/1 (96 cells), determinism 2/2, full workspace all suites ok, 0 failures.

## Accomplishments

- **Task 1 — `Backend::CubeclCpu` + `cubecl_cpu_case()` (commit `5e8b8ff`):**
  - Added the `CubeclCpu` variant to the `Backend` enum (replacing the reserved comment) and `pub fn cubecl_cpu_case() -> RunnerCase` mirroring `scalar_cpu_case()` EXACTLY except: `backend: Backend::CubeclCpu`; `dense_f32` calls `treelite_cubecl::predict_cpu::<f32>` then widens the f32 RESULT to f64 with `map(|v| v as f64)` (D-01/D-05; NEVER a pre-cast — Pitfall 6); `dense_f64` calls `treelite_cubecl::predict_cpu::<f64>`; `sparse_f32`/`sparse_f64` keep `treelite_gtil::predict_sparse` (D-02 scalar fallback).
  - Added `treelite-cubecl` as a regular `[dependency]` (see Deviations). The `RunnerCase` struct (lib.rs:89-101) and the four slot type aliases (lib.rs:65-73) are unchanged — `git diff` shows additions only (54 insertions, 1 deletion = the reserved comment line).
- **Task 2 — `gtil_matrix_cubecl` gate + SC2 determinism (commit `1a1edb0`):**
  - `tests/gtil_matrix_cubecl.rs`: replaced the Wave-0 RED stub with the real sibling gate. It constructs `cubecl_cpu_case()`, iterates the SAME frozen `fixtures/gtil/*.golden.json` cells, and asserts each OWN-layout result within `1e-5` of its golden via `approx::assert_abs_diff_eq!(epsilon = 1e-5)`. **96 cells** (48 f32-input + 48 f64-input), global max |delta| = **2.9e-6**. Per-cell provenance (D-06) is recorded from the EXECUTED path: 48 dense numerical cells tag `"cubecl-kernel"`, 48 sparse cells tag `"scalar-fallback"`; a categorical model `predict_cpu` itself routes to the fallback is also tagged `"scalar-fallback"` (the tag reflects what ACTUALLY ran, T-06-12). A `kernel_cells > 0` ensure guards against the gate crediting cubecl on fallback-only validation. It deliberately does NOT assert the `golden.manifest.backend == "scalar-cpu"` literal (gtil_matrix.rs:474).
  - `crates/treelite-cubecl/tests/determinism.rs`: replaced the RED stub with two-run element-wise `.to_bits()` bit-identity across all 4 predict kinds on f32 AND f64 input (SC2, T-06-13). `.to_bits()` (not `==`) so signed-zero / NaN-payload divergence is catchable.
  - `gtil_matrix.rs` is byte-identical (`git diff --stat` == 0, D-11); the full workspace suite is green (no regression to scalar/loader/serializer gates).

## Task Commits

Each task committed atomically (sequential executor, main tree, hooks on):

1. **Task 1: register Backend::CubeclCpu + cubecl_cpu_case()** — `5e8b8ff` (feat)
2. **Task 2: gtil_matrix_cubecl 1e-5 + provenance gate + SC2 determinism** — `1a1edb0` (feat)

## Files Created/Modified

- `crates/treelite-harness/Cargo.toml` (modified) — added `treelite-cubecl` as a regular `[dependency]`.
- `crates/treelite-harness/src/lib.rs` (modified) — `Backend::CubeclCpu` variant + `cubecl_cpu_case()` constructor (additive; struct/slots untouched).
- `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` (modified) — the real GPU-02 sibling gate (1e-5 + per-cell provenance).
- `crates/treelite-cubecl/tests/determinism.rs` (modified) — SC2 two-run `.to_bits()` bit-identity (f32 + f64, all 4 kinds).

## Decisions Made

See `key-decisions` in the frontmatter. The substantive one: `treelite-cubecl` is a regular `[dependency]` of `treelite-harness`, not the dev-dependency the plan's `<action>` text literally specified — because `cubecl_cpu_case()` is a `pub fn` in `src/lib.rs` and must resolve at library-build time. The plan's "mirroring the existing treelite-gtil/treelite-xgboost dev-deps" phrasing is itself imprecise (those are regular `[dependencies]`), so a regular dependency is exactly the mirror intended.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `treelite-cubecl` registered as a regular `[dependency]`, not a dev-dependency**
- **Found during:** Task 1 (wiring the Cargo.toml + `cubecl_cpu_case()`).
- **Issue:** The plan's `<action>` said add `treelite-cubecl` "as a dev-dependency." But `cubecl_cpu_case()` is a `pub fn` in `src/lib.rs` (not a test), so the crate must be available at library-build time. A dev-dependency is only visible to the `tests/`/`examples/`/`benches/` targets — the lib would not compile. (`cargo build -p treelite-harness --tests` would fail with an unresolved-extern-crate error.)
- **Fix:** Added `treelite-cubecl = { path = "../treelite-cubecl" }` under `[dependencies]` (mirroring the regular `treelite-gtil`/`treelite-xgboost` deps, which is what the plan's own "mirroring ... dev-deps" phrasing referenced — those entries are themselves regular `[dependencies]`). `cargo build -p treelite-harness --tests` and `cargo test --workspace --no-run` both compile.
- **Files modified:** `crates/treelite-harness/Cargo.toml`.
- **Commit:** `5e8b8ff`.

**2. [Rule 3 - Blocking] `manifest.rs` left unchanged (no mutation needed)**
- **Found during:** Task 1 (reviewing the manifest provenance contract).
- **Issue:** The plan listed `crates/treelite-harness/src/manifest.rs` in Task 1's `<files>`, but its `backend` field is ALREADY per-cell and `#[serde(default)]` (provenance-capable), and D-06 provenance is RECORDED at assertion time in the matrix sibling, not via the manifest. The plan's own `<action>` flags the manifest hook as optional ("If a check_manifest-style drift warning is wanted ... NEVER fail the gate").
- **Fix:** No change to `manifest.rs` — it would have been net-zero behavior. The sibling test carries its own warn-only `check_manifest` analog (copied from `gtil_matrix.rs`) for cross-backend drift visibility, never a gate. This keeps the manifest module exactly as the scalar reference left it.
- **Files modified:** none (intentional no-op on `manifest.rs`).
- **Commit:** n/a.

**3. [Rule 3 - Blocking] Decode helpers DUPLICATED into the sibling rather than `mod`-included**
- **Found during:** Task 2 (authoring `gtil_matrix_cubecl.rs`).
- **Issue:** The `run_cell`/golden-load helpers are private items in `gtil_matrix.rs`. `#[path = "gtil_matrix.rs"] mod ...`-including that file would re-run its `#[test] fn gtil_matrix()` inside this binary too (duplicating the scalar gate under the cubecl binary), and would risk an accidental coupling that violates the D-11 "no reshape" smell guard.
- **Fix:** Duplicated the small decode helpers (`cell_to_f64`, `flatten_output`, `decode_input_f64`, `frozen_csr`, `assert_within`, `kind_of`, the `FromF64` trait) into the sibling. This keeps `gtil_matrix.rs` byte-identical (`git diff --stat` == 0) and the sibling self-contained. The plan's `<action>` explicitly permits this ("share helpers by mod-including or duplicating the small run_cell/golden-load helpers as needed"). Only the dtype/layout DISPATCH body is duplicated; the cross-product iteration shape is unchanged.
- **Files modified:** `crates/treelite-harness/tests/gtil_matrix_cubecl.rs`.
- **Commit:** `1a1edb0`.

---

**Total deviations:** 3 (all Rule 3 blocking-issue adaptations within the plan's stated intent — a dependency-class correction, an intentional manifest no-op the plan flagged optional, and the plan-sanctioned helper duplication). **Impact on plan:** No scope creep. Every success criterion is met exactly: `Backend::CubeclCpu` + `cubecl_cpu_case()` register additively (dense -> cubecl, sparse -> scalar fallback) with the result-widen-after-predict discipline; the cubecl backend passes the IDENTICAL frozen goldens to 1e-5 across all 96 cells with per-cell `cubecl-kernel`/`scalar-fallback` provenance; SC2 two-run bit-identity is green; `gtil_matrix.rs` is untouched.

## Issues Encountered

- A doc-comment on `cubecl_cpu_case()` initially tripped `clippy::doc_lazy_continuation` (a `+`-prefixed continuation line was parsed as a Markdown list bullet). Reworded ("together with" instead of a leading `+`) — `cargo clippy -p treelite-harness --tests` is clean. No behavior change.
- Pre-existing `clippy::type_complexity` warning on `crates/treelite-cubecl/tests/spike.rs` (`soa_columns` 7-tuple, from 06-02) is OUT OF SCOPE (already logged as cosmetic in `deferred-items.md`); not touched.

## Known Stubs

None. The two Wave-0 RED stubs (`gtil_matrix_cubecl.rs`, `determinism.rs`) are now both green real gates.

## User Setup Required

None — the cubecl CPU backend needs no external configuration.

## Next Phase Readiness

- Phase 6 is complete: GPU-01 (kernels), GPU-02 (CubeclCpu default backend; frozen matrix 1e-5 + SC2 determinism), and GPU-05 (SoA upload) are all green and auditable. The D-11 registration pattern (new `Backend` variant + a `*_case()` constructor + a thin sibling matrix test) is the template Phase-7 GPU backends (Cuda/Wgpu/Rocm) follow.

## Self-Check: PASSED

- crates/treelite-harness/src/lib.rs (cubecl_cpu_case + Backend::CubeclCpu) — FOUND
- crates/treelite-harness/Cargo.toml (treelite-cubecl dep) — FOUND
- crates/treelite-harness/tests/gtil_matrix_cubecl.rs — FOUND
- crates/treelite-cubecl/tests/determinism.rs — FOUND
- commit 5e8b8ff (Task 1) — FOUND
- commit 1a1edb0 (Task 2) — FOUND

---
*Phase: 06-cubecl-gtil-kernels-cpu-backend*
*Completed: 2026-06-10*

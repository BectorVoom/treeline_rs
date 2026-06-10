---
phase: 05-full-scalar-gtil-equivalence-harness
plan: 04
subsystem: gtil
tags: [gtil, sparse-csr, predict-kind, leaf-id, score-per-tree, nan-materialization, dense-sparse-parity, gtil-02, gtil-03, verbatim-port, f32, f64, bounds-check]

# Dependency graph
requires:
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 02)
    provides: "typed Config/PredictKind surface (D-06), O-generic PredictOut trait (f32/f64), O-generic evaluate_tree + has_leaf_vector + output_shape — the dispatch and traversal this plan extends; LeafId/ScorePerTree were typed UnsupportedPredictKind stubs this plan replaces"
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 03)
    provides: "completed 10/10 postprocessors + full O-generic categorical guard; the O-generic evaluate_tree the new sparse row path and the two new kinds reuse verbatim"
  - phase: 05-full-scalar-gtil-equivalence-harness (Plan 01)
    provides: "frozen sparse-CSR goldens with dense==CSR parity asserted at capture time (D-04); the 16 leaf_id + 16 score_per_tree goldens Plan 05-05's matrix runner will assert against"
provides:
  - "accessor.rs: borrowed SparseCsr<'a,O> view (data/col_ind/row_ptr) + validate() (col_ind/row_ptr bounds → typed errors) + get_row() per-row NaN materialization (absent = NaN, predict.cc:80-85)"
  - "predict_sparse(model, csr, num_row, &config) entry point — sparse CSR predict sharing evaluate_tree with the dense path (GTIL-02, D-04 structural parity)"
  - "RowSource enum (Dense/Sparse) + extracted predict_rows shared body: dense and sparse funnel through ONE per-kind dispatch + traversal, so dense==sparse parity is structural not coincidental"
  - "PredictKind::LeafId dispatch (predict_leaf): (num_row, num_tree) integer leaf node ids cast into the O buffer (A4), no postprocess/average/base-score (predict.cc:325-345)"
  - "PredictKind::ScorePerTree dispatch (predict_score_by_tree): (num_row, num_tree, lvs) raw per-tree leaf scalar/vector, lvs = leaf_vector_shape[0]*[1] ≥ 1, no postprocess/average/base-score (predict.cc:347-378, Pitfall 5)"
  - "GtilError::SparseColumnOutOfBounds + SparseRowPtrInvalid variants (T-05-09 / T-05-10)"
affects: [05-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RowSource accessor split (Dense/Sparse) mirrors upstream DenseMatrixAccessor/SparseMatrixAccessor: both materialize a single reusable scratch &mut [O] row that evaluate_tree walks verbatim → structural D-04 parity"
    - "CSR validated ONCE up front (SparseCsr::validate) before any row is materialized, so get_row is index-safe for every row (no per-row bounds branches in the hot loop)"
    - "Per-kind bodies that skip the sum-over-trees assembly (LeafId/ScorePerTree) return directly from predict_rows before the OutputLayout/postprocessor path"

key-files:
  created:
    - crates/treelite-gtil/src/accessor.rs
    - crates/treelite-gtil/tests/sparse.rs
    - crates/treelite-gtil/tests/predict_kinds.rs
  modified:
    - crates/treelite-gtil/src/lib.rs
    - crates/treelite-gtil/src/error.rs

key-decisions:
  - "RowSource enum (not a closure / trait object): a 2-variant enum keeps predict_preset monomorphic and the dense copy / sparse NaN-fill both write the same owned scratch row, making the dense and sparse traversal byte-identical. The dense path now copies its row slice into scratch (a small cost) so BOTH paths feed evaluate_tree the same contiguous buffer — D-04 parity is structural, not a re-implementation"
  - "Extracted predict_rows shared body: predict and predict_sparse each only validate their own input view (dense buffer length / CSR structure) then funnel into predict_rows, so the 4-way kind dispatch + traversal + assembly is written exactly once"
  - "num_tree for LeafId/ScorePerTree uses the ACTUAL per-variant trees.len() (Model::GetNumTree ⇒ trees.size(), tree.h:478), NOT the staged model.num_tree() header field (only set during serialization staging)"
  - "lvs = (leaf_vector_shape[0] * leaf_vector_shape[1]).max(1): read defensively (unwrap_or(1)) and clamped ≥1 so a scalar-leaf model (shape [1,1]) writes at index 0 and a degenerate/malformed shape never produces a zero-width third dim (Pitfall 5 / T-05-11), matching the shape.rs clamp"

patterns-established:
  - "Borrowed CSR view + validate-once-then-get_row accessor (SparseCsr)"
  - "RowSource accessor abstraction giving structural dense==sparse parity"

requirements-completed: [GTIL-02, GTIL-03]

# Metrics
duration: ~6min
completed: 2026-06-10
---

# Phase 5 Plan 04: Sparse CSR Predict + LeafId/ScorePerTree Kinds Summary

**The sparse CSR predict path (absent entries materialized as `O::nan()`, NOT 0; GTIL-02) sharing `evaluate_tree` with the dense path via a `RowSource` accessor so dense==sparse parity is structural (D-04), plus the two remaining predict kinds — `LeafId` (integer node ids) and `ScorePerTree` (raw per-tree leaf scalar/vector, no postprocess/average/base-score; GTIL-03) — completing the full GTIL surface: dense+sparse input × all 4 kinds, all bounds-checked to typed `GtilError`.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-10
- **Completed:** 2026-06-10
- **Tasks:** 2 (both TDD: tests authored alongside the verbatim port → GREEN)
- **Files modified:** 5 (2 source modified, 1 source created, 2 test files created)

## Accomplishments

### Task 1 — Sparse CSR accessor + predict_sparse (GTIL-02, D-04)
- `accessor.rs` (new): borrowed `SparseCsr<'a, O> { data, col_ind, row_ptr }` mirroring the upstream `PredictSparse` pointer-triple (`gtil.h:85-88`). `validate(num_row, num_feature)` checks — once, up front — `row_ptr.len() == num_row+1`, monotone non-decreasing `row_ptr`, trailing fence `<= data/col_ind` length, and every `col_ind[k] < num_feature`, each → a typed error. `get_row(r, scratch)` fills the whole scratch with `O::nan()` (absent = NaN, `predict.cc:81`) then overwrites present columns with their `data` values (`predict.cc:83-85`) — the fill-then-overwrite order ported verbatim.
- `lib.rs`: introduced a `RowSource` enum (`Dense { data, num_feature }` / `Sparse(SparseCsr)`) that materializes row `r` into a single reusable scratch `&mut [O]`. `predict_preset` now allocates ONE scratch row and calls `rows.materialize(r, &mut scratch)` per row, then runs the existing serial-tree accumulation verbatim — so the dense and sparse paths feed `evaluate_tree` the SAME contiguous buffer (structural D-04 parity).
- Extracted the shared `predict_rows` body (kind dispatch + traversal + assembly); `predict` (dense) and the new `predict_sparse` each validate only their own input view then funnel into it.
- `error.rs`: `SparseColumnOutOfBounds { col, num_feature }` (T-05-09) and `SparseRowPtrInvalid { index, value, limit }` (T-05-10), in the existing thiserror struct-variant style with per-field threat-citing docs.
- `tests/sparse.rs` (7 tests): absent-cell-is-NaN-not-0 (cross-checked with both `default_left` polarities so a wrong 0-fill is caught), dense==sparse parity (f32 + f64), and the four malformed-CSR typed-error cases (OOB `col_ind`, non-monotone `row_ptr`, trailing fence past `data.len()`, wrong `row_ptr` length) — all return typed errors, none panic.

### Task 2 — LeafId + ScorePerTree dispatch (GTIL-03)
- `predict_leaf` (`PredictLeaf`, `predict.cc:325-345`): allocates `(num_row, num_tree)`, per `(row, tree)` writes `O::from_leaf_f64(leaf_node_id as f64)` — the integer leaf node id cast into the `O` output buffer (A4) — with NO postprocessor, averaging, or base-score.
- `predict_score_by_tree` (`PredictScoreByTree`, `predict.cc:347-378`): third-dim `lvs = (leaf_vector_shape[0]*[1]).max(1)`, allocates `(num_row, num_tree, lvs)` zero-filled; per `(row, tree)`: if `has_leaf_vector` write each `leaf_vector(leaf)[i]` (bounds-checked → `LeafVectorTooShort`), else write the scalar `leaf_value(leaf)` at index 0 (Pitfall 5). No postprocess/average/base-score.
- Both reuse the `RowSource` NaN-materialized scratch + `evaluate_tree`, so the dense and sparse leaf-id / score-per-tree paths are identical (D-04). `predict_rows` now dispatches all 4 `PredictKind` variants; no `UnsupportedPredictKind` arm remains for `LeafId`/`ScorePerTree`.
- `tests/predict_kinds.rs` (7 tests): leaf-id `(num_row, num_tree)` known-node-id values, score-per-tree scalar (raw leaf at index 0) and leaf-vector (each element), raw-vs-`Default` postprocessor isolation (`ScorePerTree` raw leaf `!=` sigmoid'd `Default`), base/average ignored by `LeafId`, all-4-kinds dispatch sanity, and f64-input leaf-id.

### Verification
- `cargo test -p treelite-gtil` green: 14 lib + 6+6+5+13+7 (config/generic/output_shaping/predict/postprocessor) + 7 sparse + 7 predict_kinds.
- `cargo test --workspace` green — no regression on dense default/raw, the binary path, or the loader/serializer suites. The single `1 ignored` is `gtil_matrix.rs` (Plan 05-05's scope, still RED-gated).
- `treelite-gtil` is `cargo fmt`- and `cargo clippy`-clean.

## Task Commits

Each task committed atomically:

1. **Task 1: sparse CSR predict with per-row NaN materialization (GTIL-02, D-04)** — `f099e82` (feat)
2. **Task 2: PredictKind::LeafId + ScorePerTree dispatch (GTIL-03)** — `95077dd` (feat)

**Plan metadata:** (final docs commit) — see git log.

## Files Created/Modified
- `crates/treelite-gtil/src/accessor.rs` (new) — `SparseCsr` borrowed view, `validate` (CSR bounds checks → typed errors), `get_row` (per-row NaN materialization, `predict.cc:80-85`).
- `crates/treelite-gtil/src/lib.rs` (modified) — `pub mod accessor` + `SparseCsr` re-export; `RowSource` enum; `predict_preset` takes a `RowSource` + reusable scratch; extracted `predict_rows`; new `predict_sparse`, `predict_leaf`/`predict_leaf_preset`, `predict_score_by_tree`/`predict_score_by_tree_preset`; 4-way kind dispatch.
- `crates/treelite-gtil/src/error.rs` (modified) — `SparseColumnOutOfBounds` + `SparseRowPtrInvalid` variants.
- `crates/treelite-gtil/tests/sparse.rs` (new) — 7 sparse tests (NaN-fill, dense==sparse f32/f64, 4 malformed-CSR typed-error cases).
- `crates/treelite-gtil/tests/predict_kinds.rs` (new) — 7 LeafId/ScorePerTree tests.

## Decisions Made
- **`RowSource` enum for structural parity.** Rather than re-deriving the sparse traversal, both the dense and sparse paths materialize each row into a single reusable scratch `&[O]` that `evaluate_tree` walks verbatim. The dense path now copies its row slice into scratch (a small, intentional cost) so the two paths are byte-identical — `predict_sparse(csr) == predict(dense_with_nan)` is a structural guarantee (D-04), proven by the f32 and f64 parity tests.
- **`predict_rows` shared body.** `predict` and `predict_sparse` validate only their own input view (dense buffer length / CSR structure), then funnel into one `predict_rows` that owns the 4-way kind dispatch, the `OutputLayout`/assembly, and the postprocessor gate. The kind dispatch is written exactly once.
- **Actual tree count, not staged field.** `LeafId`/`ScorePerTree` size their output on the per-variant `trees.len()` (matching `Model::GetNumTree ⇒ trees.size()`, `tree.h:478`), not the staged `model.num_tree()` header field which is only populated during serialization staging.
- **`lvs` clamp ≥ 1.** `score_per_tree`'s third dim `leaf_vector_shape[0]*[1]` is read defensively and clamped to ≥1, so a scalar-leaf model (shape `[1,1]`) writes at index 0 and a degenerate/malformed shape never yields a zero-width dim (Pitfall 5 / T-05-11), consistent with the `shape.rs` `output_shape` clamp.

## Deviations from Plan

None — plan executed exactly as written. Two tasks, each a verbatim port of the upstream `SparseMatrixAccessor::GetRow` / `PredictLeaf` / `PredictScoreByTree` with tests authored alongside. The plan suggested `accessor.rs` as "NEW, optional"; it was created (the clean home for `SparseCsr` + the NaN-materialization helper).

## Issues Encountered
- `cargo fmt` reflowed the new `error.rs` variant `#[error(...)]` strings and a couple of `tests/sparse.rs` literals onto single lines; applied with `cargo fmt -p treelite-gtil`. The pre-existing `treelite-builder` fmt drift noted in 05-02/05-03 remains out of scope and untouched. `treelite-gtil` is fmt- and clippy-clean.

## Known Stubs
- None introduced by this plan. The two `UnsupportedPredictKind` stubs carried from Plan 05-02 (`LeafId`/`ScorePerTree`) are now REPLACED with real implementations — that error variant is retained only for any genuinely-future/unhandled kind. The `apply_postprocessor` f32-boundary buffer (carried from 05-02, untouched here) does NOT affect `LeafId`/`ScorePerTree` (which skip postprocessing) nor the sparse path (which shares the same Default/Raw assembly).

## Threat Flags
None — no new security-relevant surface beyond what the plan's `<threat_model>` already enumerated. The `mitigate` dispositions were all applied: T-05-09 (`col_ind` bounds → `SparseColumnOutOfBounds` before the scratch write), T-05-10 (`row_ptr` length/monotonicity/trailing-fence → `SparseRowPtrInvalid` before any slice), and T-05-11 (`score_per_tree` leaf-vector access bounds-checked via the `lvs` clamp + `LeafVectorTooShort`). T-05-SC (no package installs) holds — nothing was installed.

## User Setup Required
None — all work is Rust source; verified via `cargo test --workspace` on the main tree.

## Next Phase Readiness
- The GTIL surface is now complete: dense + sparse input, all 4 predict kinds (default/raw/leaf_id/score_per_tree), 10/10 postprocessors, the full categorical guard, f32/f64 input. Plan 05-05 can now drop the `#[ignore]` on `crates/treelite-harness/tests/gtil_matrix.rs` and wire the real calls — `treelite_gtil::predict(&model, &flat, num_row, &cfg)` for `dense` cells and `treelite_gtil::predict_sparse(&model, csr, num_row, &cfg)` for `sparse` cells, keyed on each fixture's `manifest.{kind, layout, input_dtype}` — against the 64 frozen goldens (incl. the 16 `leaf_id` + 16 `score_per_tree` cells), plus the backend-parameterized seam (D-11).
- No blockers.

## Self-Check: PASSED

- All created/modified files present: `crates/treelite-gtil/src/accessor.rs`, `crates/treelite-gtil/src/lib.rs`, `crates/treelite-gtil/src/error.rs`, `crates/treelite-gtil/tests/sparse.rs`, `crates/treelite-gtil/tests/predict_kinds.rs`.
- Both task commits present in git history: `f099e82` (Task 1), `95077dd` (Task 2).

---
*Phase: 05-full-scalar-gtil-equivalence-harness*
*Completed: 2026-06-10*

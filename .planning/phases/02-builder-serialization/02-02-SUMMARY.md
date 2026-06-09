---
phase: 02-builder-serialization
plan: 02
subsystem: model-builder
tags: [builder, concat, bulk, validation, BLD-01, BLD-02, BLD-03, D-07, D-08, D-09, D-10]
requires:
  - treelite_core::{Model, ModelPreset, ModelVariant, Tree, TreeBuf, CoreError}
  - treelite_core::enums::{Operator, TaskType, TreeNodeType}
provides:
  - treelite_builder::ModelBuilder
  - treelite_builder::BuilderMetadata
  - treelite_builder::BuilderError
  - treelite_builder::concatenate
  - treelite_builder::bulk_construct_tree
affects:
  - Cargo.toml (workspace members gains treelite-builder)
tech-stack:
  added: []
  patterns:
    - "5-state builder machine with forward-reference child-key resolution at EndTree (RESEARCH Pattern 1, Pitfall 6)"
    - "every upstream TREELITE_CHECK becomes a located thiserror BuilderError variant (D-07)"
    - "always-strict orphan check; validation toggle NOT ported (D-08)"
    - "BulkConstructTree validation bypass by construction (D-09)"
key-files:
  created:
    - crates/treelite-builder/Cargo.toml
    - crates/treelite-builder/src/lib.rs
    - crates/treelite-builder/src/error.rs
    - crates/treelite-builder/src/concat.rs
    - crates/treelite-builder/src/bulk.rs
    - crates/treelite-builder/tests/validation.rs
    - crates/treelite-builder/tests/concat.rs
    - crates/treelite-builder/tests/bulk.rs
  modified:
    - Cargo.toml
decisions:
  - "[02-02] ModelBuilder produces only the <f32,f32> preset in Phase 2 (the XGBoost variant); bulk_construct_tree produces Tree<f64> to match sklearn's double element types."
  - "[02-02] node_id_map is a BTreeMap<i32,i32> mirroring upstream std::map for deterministic orphan-error key selection (RESEARCH Pattern 1)."
  - "[02-02] leaf-vs-test mutual exclusivity is enforced structurally by the state machine: after a leaf/test detail the node is NodeComplete, so a second detail call is a WrongState transition (not a dedicated LeafTestConflict path in the happy flow)."
  - "[02-02] concatenate adds NO postprocessor/base_scores cross-input equality checks — upstream model_concat.cc lacks them (BLD-02 fidelity)."
metrics:
  duration: ~7 min
  completed: 2026-06-10
  tasks: 2
  files: 9
---

# Phase 2 Plan 2: Builder, Concat & Bulk Summary

The new `treelite-builder` crate: a fluent always-strict `ModelBuilder` 5-state machine (BLD-01) that validates node well-formedness eagerly and tree topology (orphans, dangling child keys via forward-reference resolution) at `EndTree`, plus a verbatim `ConcatenateModelObjects` port (BLD-02) and a `BulkConstructTree` validation-bypass fast path (BLD-03) — all ported faithfully from vendored upstream C++.

## What Was Built

- **`treelite-builder` crate** added to the workspace `members`; path-deps on `treelite-core` + workspace-pinned `thiserror` only (no new third-party packages — T-02-SC).
- **`BuilderError`** (`error.rs`): a `thiserror` enum where every upstream `TREELITE_CHECK` / `TREELITE_LOG(FATAL)` fatal path becomes a located variant carrying the offending key/index (D-07): `NegativeNodeKey`, `DuplicateNodeKey`, `SelfOrEqualChildKey`, `SplitIndexOutOfRange`, `DanglingChildKey`, `OrphanedNode`, `LeafVectorSizeMismatch`, `WrongState`, `CommitTreeCountMismatch`, `EmptyTree`, `VariantMismatch`, `HeaderMismatch`, `MetadataNotInitialized`, plus `#[from] treelite_core::CoreError`.
- **`ModelBuilder`** (`lib.rs`): the 5-state machine (`ExpectTree → ExpectNode → ExpectDetail → NodeComplete`, plus `ModelComplete`) ported from `model_builder.cc:50-389`. Children are stored RAW at `numerical_test`/`categorical_test` and resolved to internal indices at `end_tree` (RESEARCH Pitfall 6 — forward references resolve at tree close, never at the detail call). The orphan mark-and-sweep is ALWAYS on; the upstream validation toggle is intentionally not ported (D-08, verified `grep -c` == 0). `end_tree` finalizes a `Tree<f32>` via the column-fill → `TreeBuf::from_owned` pattern.
- **`concatenate`** (`concat.rs`): ports `model_concat.cc:19-71` verbatim — header copy from `objs[0]`, same-variant discriminant check, `num_target`/`num_class`/`leaf_vector_shape` match, deep-clone every tree column-by-column, `Extend` `target_id`/`class_id`, empty → `Ok(None)`. No upstream-absent equality checks added.
- **`bulk_construct_tree`** (`bulk.rs`): ports `sklearn_bulk.cc:36-211` — a sklearn-shaped single-pass bulk constructor producing `Tree<f64>`, bypassing all per-node validation by construction (D-09). Leaves get `cmp=kNone`; internals get `cmp=kLE, default_left=true` and the sklearn impurity-reduction gain.

## Tasks & Commits

| Task | Name | Commit | Key Files |
| ---- | ---- | ------ | --------- |
| 1 | Scaffold crate + BuilderError + ModelBuilder state machine (BLD-01) | `a16bccc` | Cargo.toml, crates/treelite-builder/{Cargo.toml, src/lib.rs, src/error.rs}, tests/validation.rs |
| 2 | ConcatenateModelObjects (BLD-02) + BulkConstructTree (BLD-03, D-09) | `ef8285a` | crates/treelite-builder/src/{concat.rs, bulk.rs, lib.rs}, tests/{concat.rs, bulk.rs} |

## Verification

- `cargo build --workspace` — green (new member compiles).
- `cargo test -p treelite-builder` — 17 tests pass (validation 11, concat 4, bulk 2).
- `cargo test --workspace` — green, no regression (all prior core/gtil/xgboost/harness tests pass).
- `cargo clippy --workspace` — clean (no warnings).
- `grep -c 'flag_check_orphaned\|SetValidationFlag' crates/treelite-builder/src/lib.rs` → `0` (D-08 toggle not ported).
- `grep -c 'D-09\|sklearn_bulk' crates/treelite-builder/src/bulk.rs` → `12` (bypass documented, ≥ 1).
- `concat.rs` `postprocessor`/`base_scores` usages are only the copy-from-`objs[0]` assignments + doc-comment — no upstream-absent equality checks (BLD-02 fidelity, reviewer-auditable).

### Acceptance criteria

- BLD-01: distinct `#[test]` for negative key, duplicate key, leaf+test conflict, equal/self child key, dangling child key, orphaned node, forward-reference ACCEPTED, commit tree-count mismatch — each asserts the specific located `BuilderError` variant.
- BLD-02: merging a 2-tree and a 3-tree model yields 5 trees with `target_id`/`class_id` length 5; variant mismatch and `num_target` mismatch rejected with typed errors; empty slice → `None`.
- BLD-03: bulk tree columns (`node_type`/`cleft`/`cright`/`threshold`/`cmp`/`leaf_value`) match an equivalent per-node build; leaves `kNone`, internals `kLE`+`default_left=true`; node stats present, gain only on internals with the exact impurity-reduction value.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `Model` is not `Debug`, breaking `.unwrap_err()` on `commit_model()`**
- **Found during:** Task 1 (first test compile).
- **Issue:** `commit_model` returns `Result<Model, BuilderError>`; `unwrap_err` requires `T: Debug`, which `treelite_core::Model` does not implement.
- **Fix:** The two `commit_model` error-path tests use a `match` on the result instead of `unwrap_err`.
- **Files modified:** crates/treelite-builder/tests/validation.rs, tests/concat.rs.
- **Commit:** `a16bccc`, `ef8285a`.

**2. [Rule 1 - Cleanliness] Removed unused `detail_set` staging field**
- **Found during:** Task 2 review.
- **Issue:** A `detail_set` bool was written in every detail method but never read — the state machine already enforces the leaf-vs-test guard, making it dead.
- **Fix:** Removed the field and its assignments; behavior unchanged (state machine is the sole guard).
- **Files modified:** crates/treelite-builder/src/lib.rs.
- **Commit:** `ef8285a`.

### Plan-discretion choices honored

- `BTreeMap<i32,i32>` for `node_id_map` (planner discretion to mirror upstream `std::map`).
- Leaf-vs-test conflict surfaces as `WrongState` (the state machine makes the second detail call illegal) rather than a dedicated `LeafTestConflict` arm — the plan's `<behavior>` specifies "the state machine makes the second call illegal", which this satisfies. The `LeafTestConflict` variant exists in `BuilderError` for the documented intent but the structural guard is the active path.

## Known Stubs

None. `categorical_test` and `leaf_vector` are implemented for interface completeness (they validate and stage correctly); the category-list / leaf-vector columns are not yet wired into the built `Tree` (XGBoost rewiring in Plan 05 uses `leaf_scalar` + `numerical_test`). This is intentional Phase-2 scope, not a data stub blocking the plan goal.

## Threat Flags

None. No new network endpoints, auth paths, or trust-boundary surface beyond the two boundaries already enumerated in the plan's `<threat_model>` (caller → ModelBuilder, caller → bulk_construct_tree). Mitigations T-02-B01 (reject negative key / out-of-range split_index / equal-self child keys before any `usize` cast) and T-02-B02 (bounded iterative orphan sweep, no recursion) are implemented.

## Self-Check: PASSED

- All 8 created files present on disk.
- Both task commits (`a16bccc`, `ef8285a`) present in git history.

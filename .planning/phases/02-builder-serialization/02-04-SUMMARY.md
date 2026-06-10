---
phase: 02-builder-serialization
plan: 04
subsystem: serialization
tags: [json-dump, field-accessors, ser-03, ser-04, d-04]
requires:
  - "02-03: v5 serializer + Model/Tree columns + enum as_str() spellings"
provides:
  - "treelite_core::dump_as_json / dump_as_json_string (SER-03, D-04)"
  - "Model read-only header accessors (num_feature/num_tree/version/type-tags)"
  - "Tree per-node inspection accessors (node_type/default_left/category_list/leaf_vector/has_*/gain/sum_hess/data_count)"
affects:
  - "Phase 8 (Python binding): JSON dump + field inspection surface"
tech-stack:
  added:
    - "serde_json (workspace-pinned) on treelite-core"
  patterns:
    - "RESEARCH Pattern 6: structural JSON dump mirroring json_serializer.cc"
    - "RESEARCH Pattern 7: typed read accessors now, string-dispatch deferred to Phase 8"
key-files:
  created:
    - crates/treelite-core/src/serialize/json.rs
    - crates/treelite-core/src/serialize/fields.rs
    - crates/treelite-core/tests/dump_json.rs
    - crates/treelite-core/tests/fields.rs
  modified:
    - crates/treelite-core/Cargo.toml
    - crates/treelite-core/src/serialize/mod.rs
    - crates/treelite-core/src/lib.rs
    - crates/treelite-core/src/tree.rs
    - crates/treelite-core/src/model.rs
decisions:
  - "[02-04]: DumpAsJSON reuses the existing enum as_str() spellings verbatim (D-04); no new strings invented."
  - "[02-04]: dump_as_json takes &mut Model to stage the variant-derived type tags, mirroring upstream GetThresholdType()/GetLeafOutputType()."
  - "[02-04]: D-04 equivalence is asserted at the PARSED-value level, never by byte-comparing serialized JSON (RapidJSON vs serde_json float formatting may differ, A4/Q3)."
  - "[02-04]: The Model v5 bookkeeping readers were promoted from pub(crate) to pub (read-only, no setter) to serve as the SER-04 inspection surface, preserving field_accessor.cc Set-rejection fidelity (T-02-J02)."
metrics:
  duration: "~6 min"
  completed: "2026-06-10"
  tasks: 2
  files: 9
---

# Phase 2 Plan 04: DumpAsJSON + Field Accessors Summary

Completes Phase 2's introspection slice: a structural `DumpAsJSON` (SER-03, D-04) whose key names, nesting, and value types mirror upstream `json_serializer.cc` so a Rust dump is value-diffable against a C++ dump, plus typed read-only field accessors (SER-04) exposing the model header and per-tree node fields — with upstream read-only fields (version triple / `num_tree` / type tags) deliberately offering no setter.

## What Was Built

### Task 1 — DumpAsJSON (SER-03, D-04) — commit `8a0597c`

- Added `serde_json.workspace = true` to `treelite-core`.
- `serialize/json.rs`: `dump_as_json(&mut Model) -> serde_json::Value` and a `dump_as_json_string` convenience, porting `json_serializer.cc:135-229` structurally:
  - Model object keys in upstream order: `threshold_type`, `leaf_output_type`, `num_feature`, `task_type`, `average_tree_output`, `num_target`, `num_class`, `leaf_vector_shape`, `target_id`, `class_id`, `postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores`, `attributes`, `trees`.
  - Per-tree object: `num_nodes`, `has_categorical_split`, `nodes[]`.
  - Per-node (`WriteNode`, json_serializer.cc:81-133): always `node_id`; leaf → `leaf_value` (scalar or array via `has_leaf_vector`); internal → `split_feature_id`, `default_left`, `node_type`, then numerical `comparison_op`+`threshold` / categorical `category_list_right_child`+`category_list`, then `left_child`, `right_child`; conditional tail `data_count`/`sum_hess`/`gain` gated on the present-flag columns.
  - Enum strings come from the existing `as_str()` spellings (`kBinaryClf`, `float32`, `numerical_test_node`, `<`) — not re-spelled (11 `as_str` references).
- Added the `Tree` inspection accessors the dump needs (`node_type`, `default_left`, `category_list_right_child`, `leaf_vector`, `category_list`, `has_data_count`/`data_count`, `has_sum_hess`/`sum_hess`, `has_gain`/`gain`), each doc-commented with its `tree.h:NNN` source line.
- Wired `pub mod json` + re-exported `dump_as_json`/`dump_as_json_string` from `lib.rs`.
- `tests/dump_json.rs` (3 tests): deserializes the frozen `golden_v5.bin` and asserts the model-object key set, `task_type == "kBinaryClf"`, `threshold_type == "float32"`, `trees` length == tree count, a numerical internal node with `comparison_op == "<"` + a numeric `threshold`, and a leaf with `leaf_value` — all at the parsed-value level.

### Task 2 — Field accessors (SER-04) — commit `bd1c747`

- `serialize/fields.rs`: documents the full SER-04 read surface (header `pub` fields, the Model bookkeeping readers, and the per-tree node getters) and adds the one missing typed header reader, `num_feature()`.
- Promoted the Model v5 bookkeeping readers (`major_ver`, `minor_ver`, `patch_ver`, `num_tree`, `threshold_type`, `leaf_output_type`, `num_opt_field_per_model`) from `pub(crate)` to `pub`, read-only, each doc-commented with its `field_accessor.cc:NNN` line. No setter exists for any of them (field_accessor.cc Set-rejection fidelity, T-02-J02).
- Wired `pub mod fields` into `serialize/mod.rs`.
- `tests/fields.rs` (2 tests): asserts `num_feature()`, post-staging `num_tree()` == tree count, `threshold_type()`/`leaf_output_type()` == `DType::kFloat32`, the read-only `4.7.0` version triple, and per-node `is_leaf`/`leaf_value`/`threshold`/`node_type`/`comparison_op` for an internal and a leaf node.

## Verification

- `cargo test -p treelite-core --test dump_json` — 3 passed.
- `cargo test -p treelite-core --test fields` — 2 passed.
- `cargo test --workspace` — green, no regressions (all prior suites still pass).
- `cargo clippy --workspace --all-targets` — clean (no warnings).
- `cargo fmt --all` — applied.
- Acceptance greps: `as_str` in json.rs == 11 (≥3); no `set_major_ver`/`set_num_tree`/`set_threshold_type` in fields.rs (== 0).

## Deviations from Plan

None — plan executed as written. Two minor design notes within the plan's stated latitude:

- `dump_as_json` takes `&mut Model` (not `&Model`) so it can call `stage_serialization_fields()` to derive the variant type tags, exactly as upstream reads them from `GetThresholdType()`/`GetLeafOutputType()`. The plan's signature note allowed `-> serde_json::Value` or `-> String`; both are provided.
- The SER-04 read accessors for the Model bookkeeping fields were promoted in place in `model.rs` (rather than redefined in `fields.rs`) to avoid duplicate inherent methods; `fields.rs` documents the surface and adds `num_feature()`. This matches the plan's instruction to "add accessor methods only where a private field needs a read path" without a setter.

## Known Stubs

None. The categorical-node branch in `WriteNode` and the leaf-vector array branch are fully implemented (the binary:logistic golden fixture exercises only numerical splits + scalar leaves, but the code paths are real, not placeholders).

## Self-Check: PASSED

- crates/treelite-core/src/serialize/json.rs — FOUND
- crates/treelite-core/src/serialize/fields.rs — FOUND
- crates/treelite-core/tests/dump_json.rs — FOUND
- crates/treelite-core/tests/fields.rs — FOUND
- Commit 8a0597c — FOUND
- Commit bd1c747 — FOUND

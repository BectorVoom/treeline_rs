---
phase: 02-builder-serialization
verified: 2026-06-10T01:15:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/4
  gaps_closed:
    - "CR-01: end_tree now populates category_list_right_child/leaf_vector_begin/leaf_vector_end/category_list_begin/category_list_end at length num_nodes (AllocNode invariant) — fixed in commit 08f1402"
    - "CR-02: end_tree now gates stat columns (data_count/sum_hess/gain + _present pairs) empty-unless-set per-column-independently via any_* flags — fixed in commit 08f1402"
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "XGBoost loader produces byte-identical output to upstream golden (leaf split_index=-1, attributes={}, sum_hess/gain columns, CSR-offset columns)"
    addressed_in: "Phase 3"
    evidence: "Explicitly deferred as DEF-02-01; loader_path_divergence_diagnostic in golden_v5.rs is a non-fatal diagnostic exactly to keep this visible. The serializer golden round-trip test proves serializer correctness independently."
---

# Phase 02: Builder & Serialization Verification Report

**Phase Goal:** "Widen the construction and persistence layers along the spine — a fluent validated ModelBuilder (plus concatenate and a bulk fast path) and full v5 serialization — so loaders have a construction target and models round-trip."
**Verified:** 2026-06-10T01:15:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (02-06, commits 08f1402 + df7ac1c)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Fluent ModelBuilder constructs node-by-node, rejects ill-formed topologies with typed errors; XGBoost-JSON loader rewired through it; 1e-5 equivalence still holds | VERIFIED | State machine in `lib.rs`; 11 validation tests pass; `treelite-xgboost/src/lib.rs` drives builder API; `equivalence_within_1e5` passes |
| 2 | ConcatenateModelObjects merges built models; BulkConstructTree fast path exists | VERIFIED | `concat.rs` mirrors `model_concat.cc`; 4 concat tests pass; `bulk.rs` single-pass; 2 bulk tests pass |
| 3 | Model round-trips through v5 binary format (serialize → deserialize → identical model) and builder-produced models match upstream column-length invariants | VERIFIED | Self-consistent round-trip: `round_trip_is_byte_identical` + `golden_v5_round_trips_to_itself` pass. CR-01 closed (commit 08f1402): five AllocNode per-node columns now length num_nodes. CR-02 closed (commit 08f1402): stat columns empty-unless-set, per-column independent. Regression guard: `builder_empty_unless_set_and_allocnode_lengths` + `builder_stat_column_emitted_when_set` both pass (commit df7ac1c). `serializer_reproduces_golden_v5_byte_for_byte` stays green. |
| 4 | DumpAsJSON emits model as JSON; field accessors expose model/tree fields | VERIFIED | `json.rs` implements `dump_as_json/dump_as_json_string`; 3 JSON tests + 2 field accessor tests pass |

**Score:** 4/4 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | XGBoost loader byte-fidelity gap (DEF-02-01): sum_hess/gain/CSR-offset columns, attributes, leaf split_index=-1 | Phase 3 | `deferred-items.md` entry DEF-02-01; `loader_path_divergence_diagnostic` test is non-fatal diagnostic |

### CR-01 / CR-02 Gap Closure — Detailed Evidence

**CR-01 fix (`end_tree`, lines 463-492 of `crates/treelite-builder/src/lib.rs`, commit 08f1402):**

`end_tree` now constructs five per-node columns with length `num_nodes` using `TreeBuf::from_owned(vec![...;num_nodes])`:
- `category_list_right_child` = `vec![false; num_nodes]` — mirrors `tree.h:79` (`category_list_right_child_.PushBack(false)`)
- `leaf_vector_begin` = `vec![0u64; num_nodes]` — mirrors `tree.h:81` (`leaf_vector_begin_.PushBack(leaf_vector_.Size())` where builder's `leaf_vector_.Size() == 0`)
- `leaf_vector_end` = `vec![0u64; num_nodes]` — mirrors `tree.h:82`
- `category_list_begin` = `vec![0u64; num_nodes]` — mirrors `tree.h:83`
- `category_list_end` = `vec![0u64; num_nodes]` — mirrors `tree.h:84`

Upstream `AllocNode` (confirmed at `detail/tree.h:79-84`): all five columns get `PushBack` on every node call. The Rust fix correctly sets their lengths to `num_nodes` with the upstream default values for no-category, no-leaf-vector trees (builder never populates the value buffers, so `leaf_vector_.Size()` and `category_list_.Size()` are 0 for every node — begin/end offsets are all 0).

**CR-02 fix (`end_tree`, lines 474-518 of `crates/treelite-builder/src/lib.rs`, commit 08f1402):**

Three `any_*` flags computed from `NodeStaging.{data_count,sum_hess,gain}_present` gate each stat pair independently:
- `any_data_count` false → `tree.data_count = TreeBuf::empty(); tree.data_count_present = TreeBuf::empty()`
- `any_sum_hess` false → `tree.sum_hess = TreeBuf::empty(); tree.sum_hess_present = TreeBuf::empty()`
- `any_gain` false → `tree.gain = TreeBuf::empty(); tree.gain_present = TreeBuf::empty()`
- When any flag is true → stat column and present column both emitted at length `num_nodes`

Upstream `AllocNode` (confirmed at `detail/tree.h:87-98`): `if (!data_count_present_.Empty())` / `if (!sum_hess_present_.Empty())` / `if (!gain_present_.Empty())` guards each pair independently. The Rust fix is a direct port of this invariant. The previous code emitted all six stat columns unconditionally at length `num_nodes`; that is now gone.

**Fidelity cross-check against `detail/tree.h:70-101`:**

The upstream `AllocNode` function (lines 70-101, read directly) confirms:
1. Lines 71-79: eight always-pushed columns (`node_type_`, `cleft_`, `cright_`, `split_index_`, `default_left_`, `leaf_value_`, `threshold_`, `cmp_`, `category_list_right_child_`) — all handled by the existing per-node fill loop plus the five new CR-01 columns.
2. Lines 81-84: four CSR-offset columns pushed unconditionally at `leaf_vector_.Size()` / `category_list_.Size()` — matched by the five CR-01 `vec![0u64; num_nodes]` columns.
3. Lines 87-98: three `if (!*_present_.Empty())` guards, each independently gating a (value, present) pair — matched exactly by the three `any_*` flag checks in CR-02.

The implementation is correct and complete relative to the upstream source of truth.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-builder/src/lib.rs` | ModelBuilder state machine + CR-01/CR-02 fixes | VERIFIED | CR-01: five AllocNode columns populated; CR-02: any_* flags gate stat emission |
| `crates/treelite-builder/tests/column_fidelity.rs` | Column-fidelity regression tests | VERIFIED | 2 tests pass: `builder_empty_unless_set_and_allocnode_lengths`, `builder_stat_column_emitted_when_set` |
| `crates/treelite-builder/src/concat.rs` | ConcatenateModelObjects | VERIFIED | 4 tests pass |
| `crates/treelite-builder/src/bulk.rs` | BulkConstructTree fast path | VERIFIED | 2 tests pass |
| `crates/treelite-core/src/serialize/binary.rs` | v5 binary serializer/deserializer | VERIFIED | `serializer_reproduces_golden_v5_byte_for_byte` still passes |
| `crates/treelite-core/src/serialize/mod.rs` | Header + tree walk, deserialize | VERIFIED | `round_trip_is_byte_identical` + `golden_v5_round_trips_to_itself` still pass |
| `crates/treelite-core/src/serialize/pybuffer.rs` | Zero-copy PyBuffer frames | VERIFIED | 2 pybuffer tests pass |
| `crates/treelite-core/src/serialize/json.rs` | DumpAsJSON | VERIFIED | 3 JSON tests pass |
| `crates/treelite-core/src/serialize/fields.rs` | Typed field accessors | VERIFIED | 2 field tests pass |
| `crates/treelite-xgboost/src/lib.rs` | XGBoost loader rewired through builder | VERIFIED | `equivalence_within_1e5` passes |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/treelite-xgboost/src/lib.rs` | `treelite_builder::ModelBuilder` | loader emits builder calls | WIRED | All builder calls present; equivalence test green |
| `crates/treelite-harness/tests/golden_v5.rs` | golden fixture bytes | `serializer_reproduces_golden_v5_byte_for_byte` | WIRED | Passes — serializer byte-exact against upstream |
| `crates/treelite-builder/tests/column_fidelity.rs` | `ModelVariant::F32(preset).trees[0]` column fields | `.len()` assertions | WIRED | Both tests pass; assert the right columns |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CR-01 + CR-02 regression guard | `cargo test -p treelite-builder --test column_fidelity` | 2/2 pass | PASS |
| Self-consistent binary round-trip | `cargo test -p treelite-core --test serialize_roundtrip` | 5/5 pass | PASS |
| Golden v5 byte-exact serializer | `cargo test -p treelite-harness --test golden_v5` | 2/2 pass | PASS |
| 1e-5 equivalence after builder rewiring | `cargo test -p treelite-harness --test equivalence` | 1/1 pass | PASS |
| Full workspace no-regression gate | `cargo test --workspace` | 0 failures across all crates | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BLD-01 | 02-02, 02-05 | Fluent ModelBuilder with orphan/topology validation | SATISFIED | 11 validation tests; end-to-end via rewired XGBoost loader |
| BLD-02 | 02-02 | ConcatenateModelObjects | SATISFIED | `concat.rs`; 4 tests |
| BLD-03 | 02-02 | BulkConstructTree fast path | SATISFIED | `bulk.rs`; 2 tests |
| SER-01 | 02-03, 02-06 | Model round-trips v5 binary; builder-produced model column-length-faithful to upstream | SATISFIED | Self-consistent round-trip verified; CR-01/CR-02 fixes make builder output column-length-faithful to upstream AllocNode invariant; column_fidelity tests lock the invariant |
| SER-02 | 02-04 | Model serializes to v5 PyBuffer | SATISFIED | 25-frame zero-copy vec; 2 pybuffer tests |
| SER-03 | 02-04 | DumpAsJSON | SATISFIED | 3 tests pass |
| SER-04 | 02-04 | Field accessors expose model/tree fields | SATISFIED | 2 field tests pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-builder/src/concat.rs` | 157-158 | WR-03: post-condition uses `debug_assert_eq!` where upstream uses throwing check | WARNING | Post-condition compiled out in release builds — carry-forward from initial verification |
| `crates/treelite-builder/src/lib.rs` | 173-185 | WR-01: `initialize_metadata` omits target_id/class_id range checks and base_scores length check | WARNING | Malformed metadata accepted silently — carry-forward from initial verification |

No blockers. The two CR-01/CR-02 BLOCKERs from the initial verification are resolved.

### Human Verification Required

None — all phase-2 truths are programmatically verifiable. Visual/UX checks are not applicable.

### Gaps Summary

No gaps remain. The two blockers identified in the initial verification (status: gaps_found, score 3/4) are closed:

- **CR-01 resolved:** `end_tree` now emits all five AllocNode per-node columns at length `num_nodes`, matching upstream `detail/tree.h:79-84`. Evidence: `builder_empty_unless_set_and_allocnode_lengths` asserts `category_list_right_child.len() == 3`, `leaf_vector_begin.len() == 3`, etc., all pass. Commit 08f1402.
- **CR-02 resolved:** `end_tree` now gates stat columns via three independent `any_*` flags, matching upstream `detail/tree.h:87-98`. Evidence: `builder_empty_unless_set_and_allocnode_lengths` asserts all six stat column lengths are 0 when no stat was set; `builder_stat_column_emitted_when_set` asserts per-column independence (`sum_hess` set → `sum_hess.len()==3`; `data_count`/`gain` still 0). Commit 08f1402.
- **No regression:** `cargo test --workspace` exits 0 across all crates. `serializer_reproduces_golden_v5_byte_for_byte`, `round_trip_is_byte_identical`, `golden_v5_round_trips_to_itself`, and `equivalence_within_1e5` all pass.

DEF-02-01 (XGBoost loader byte-fidelity: `sum_hess`/`gain`/CSR columns, `attributes`, leaf `split_index=-1`) remains correctly deferred to Phase 3 and is NOT a Phase 2 gap.

---

_Verified: 2026-06-10T01:15:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: after gap closure plan 02-06 (commits 08f1402, df7ac1c)_

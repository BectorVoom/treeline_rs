---
phase: 02-builder-serialization
verified: 2026-06-10T00:23:49Z
status: gaps_found
score: 3/4 must-haves verified
overrides_applied: 0
gaps:
  - truth: "A model round-trips through the v5 binary format (serialize → deserialize → identical model) AND the builder-produced model is byte-faithful to upstream"
    status: partial
    reason: |
      The self-consistency round-trip (serialize→deserialize→serialize == original bytes) is
      verified — any model built or loaded, serialized, deserialized, and re-serialized produces
      the same bytes as the first serialization. But criterion 3 as stated in the phase goal
      ("serialize → deserialize → identical model") admits two interpretations:
        (a) Self-consistent identity: model→bytes→model is lossless. VERIFIED.
        (b) Upstream byte-fidelity: builder-produced model→bytes equals the upstream golden.
            FAILED for the builder path.
      CR-02 is a newly discovered gap not covered by DEF-02-01: the builder unconditionally
      emits data_count, data_count_present, sum_hess, sum_hess_present, gain, gain_present at
      length num_nodes (all-false) for every built tree, even when no stat was ever set. Upstream
      maintains "empty-unless-set" — the golden shows data_count_present at length 0. This means
      the serialized byte image of ANY builder-produced model differs from upstream for those
      six columns.
    artifacts:
      - path: "crates/treelite-builder/src/lib.rs"
        issue: "end_tree lines 445-483 unconditionally push per-node data_count/sum_hess/gain values and present-flags regardless of whether any node set the stat. Upstream invariant: column is empty unless at least one node set it. Divergence confirmed by CR-02 in 02-REVIEW.md."
      - path: "crates/treelite-builder/src/lib.rs"
        issue: "end_tree does not populate category_list_right_child, leaf_vector_begin, leaf_vector_end, category_list_begin, category_list_end columns. Upstream AllocNode pushes one entry per node into each of these (all-zero/false for no-category no-leaf-vector trees). Confirmed by CR-01 in 02-REVIEW.md."
    missing:
      - "In end_tree: gate data_count/sum_hess/gain stat columns on 'any node set this stat' — only emit non-empty columns when at least one node called data_count()/sum_hess()/gain(). Leave all six (value + present) as TreeBuf::empty() when none did."
      - "In end_tree: populate per-node CSR-offset and category columns (category_list_right_child=false, leaf_vector_begin=0, leaf_vector_end=0, category_list_begin=0, category_list_end=0 for every node) as AllocNode does."
      - "Add a test that serializes a builder-committed model and asserts the absent-stat columns have length 0 (not num_nodes)."
deferred:
  - truth: "XGBoost loader produces byte-identical output to upstream golden (leaf split_index=-1, attributes={}, sum_hess/gain columns, CSR-offset columns)"
    addressed_in: "Phase 3"
    evidence: "Explicitly deferred as DEF-02-01; loader_path_divergence_diagnostic in golden_v5.rs is a non-fatal diagnostic exactly to keep this visible. The serializer golden round-trip test proves serializer correctness independently."
---

# Phase 02: Builder & Serialization Verification Report

**Phase Goal:** "Widen the construction and persistence layers along the spine — a fluent validated ModelBuilder (plus concatenate and a bulk fast path) and full v5 serialization — so loaders have a construction target and models round-trip."
**Verified:** 2026-06-10T00:23:49Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Fluent ModelBuilder constructs node-by-node, rejects ill-formed topologies with typed errors; XGBoost-JSON loader rewired through it; 1e-5 equivalence still holds | VERIFIED | `crates/treelite-builder/src/lib.rs` implements the 5-state machine; 11 validation tests pass; `treelite-xgboost/src/lib.rs` drives `ModelBuilder::start_tree / start_node / numerical_test / leaf_scalar / end_node / end_tree / commit_model`; `cargo test -p treelite-harness --test equivalence` passes with max delta 0 |
| 2 | ConcatenateModelObjects merges models; BulkConstructTree fast path exists | VERIFIED | `concat.rs` mirrors `model_concat.cc`; 4 concat tests pass (merge, empty, variant-mismatch, header-mismatch); `bulk.rs` implements single-pass sklearn-shaped build; 2 bulk tests pass |
| 3 | Model round-trips through v5 binary (self-consistent); builder-produced model is byte-faithful to upstream | PARTIAL — see gap | Self-consistent round-trip: `serialize(deserialize(bytes)) == bytes` is byte-exact for any model (5 round-trip tests pass). But builder-produced models diverge from the upstream golden in 6 columns (CR-01: CSR-offset/category columns absent; CR-02: stat columns non-empty-all-false instead of empty). These are undetected by existing tests. |
| 4 | DumpAsJSON emits model as JSON; field accessors expose model/tree fields | VERIFIED | `json.rs` implements `dump_as_json` / `dump_as_json_string` matching upstream `WriteNode`; `fields.rs` adds `num_feature()` typed accessor; 3 JSON dump tests + 2 field accessor tests pass |

**Score:** 3/4 truths verified (truth 3 is PARTIAL due to CR-01 and CR-02)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | XGBoost loader byte-fidelity gap (DEF-02-01): sum_hess/gain/CSR-offset columns, attributes, leaf split_index=-1 | Phase 3 | `deferred-items.md` entry DEF-02-01; `loader_path_divergence_diagnostic` test is non-fatal diagnostic |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/treelite-builder/src/lib.rs` | ModelBuilder state machine | VERIFIED | 582 lines; 5-state machine; all validation paths implemented |
| `crates/treelite-builder/src/concat.rs` | ConcatenateModelObjects | VERIFIED | 161 lines; mirrors `model_concat.cc`; header copy + per-input checks + deep-clone |
| `crates/treelite-builder/src/bulk.rs` | BulkConstructTree fast path | VERIFIED | 191 lines; single-pass sklearn-shaped build; D-09 bypass documented |
| `crates/treelite-builder/src/error.rs` | BuilderError typed enum | VERIFIED | thiserror-based; locality fields (key, index, field) |
| `crates/treelite-core/src/serialize/binary.rs` | v5 binary serializer/deserializer | VERIFIED | 244 lines; serialize + Reader cursor; all fields annotated with serializer.cc line refs |
| `crates/treelite-core/src/serialize/mod.rs` | Header + tree walk, deserialize | VERIFIED | 515 lines; exact field order; 25-column tree walk; bounds-checked Reader |
| `crates/treelite-core/src/serialize/pybuffer.rs` | Zero-copy PyBuffer frames | VERIFIED | 163 lines; 25-frame tree walk; zero-copy slice borrows confirmed by test |
| `crates/treelite-core/src/serialize/json.rs` | DumpAsJSON | VERIFIED | 183 lines; upstream WriteNode key set and order |
| `crates/treelite-core/src/serialize/fields.rs` | Typed field accessors | VERIFIED | `num_feature()` typed reader; tree node accessors in `tree.rs` |
| `crates/treelite-xgboost/src/lib.rs` | XGBoost loader rewired through builder | VERIFIED | `ModelBuilder` imported and all builder calls present; `TreeBuf::from_owned` removed from build path |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/treelite-xgboost/src/lib.rs` | `treelite_builder::ModelBuilder` | loader emits builder calls | WIRED | `use treelite_builder::{BuilderMetadata, ModelBuilder}` at line 21; `start_tree`, `start_node`, `numerical_test`, `leaf_scalar`, `end_node`, `end_tree`, `commit_model` all present |
| `crates/treelite-xgboost/Cargo.toml` | `treelite-builder` path dep | `[dependencies]` | WIRED | Dependency present; loader builds against it |
| `crates/treelite-harness/tests/equivalence.rs` | `load_xgboost_json` | 1e-5 regression gate | WIRED | Test loads JSON fixture through rewired loader and asserts max delta < 1e-5; PASSES |
| `crates/treelite-core/src/serialize/mod.rs` | `crates/treelite-core/src/serialize/binary.rs` | `serialize_header` + `serialize_trees` | WIRED | `serialize_to_buffer` calls `serialize_header` and `serialize_trees` |
| `crates/treelite-core/src/serialize/pybuffer.rs` | `Model` column data | zero-copy `as_slice()` borrows | WIRED | `push_tree_frames` borrows `TreeBuf::as_slice()` directly; confirmed by pointer-equality test |

### Data-Flow Trace (Level 4)

Not applicable — this phase produces a library (no rendering component). The critical data flow is model→bytes→model round-trip, verified by `serialize_roundtrip.rs` and `golden_v5.rs`.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Self-consistent binary round-trip | `cargo test -p treelite-core --test serialize_roundtrip` | 5/5 pass including `round_trip_is_byte_identical` and `golden_v5_round_trips_to_itself` | PASS |
| 1e-5 equivalence after builder rewiring | `cargo test -p treelite-harness --test equivalence` | passes; max delta = 0e0 | PASS |
| Builder topology validation (11 cases) | `cargo test -p treelite-builder --test validation` | 11/11 pass | PASS |
| Concat + bulk tests | `cargo test -p treelite-builder` | 4 concat + 2 bulk pass | PASS |
| Loader-path byte divergence diagnostic | `cargo test -p treelite-harness --test golden_v5 -- --nocapture` | Diagnostic prints: "produced 805 B, golden 951 B, first divergence at offset 131" | DIAGNOSTIC (known, non-fatal, tracked as DEF-02-01 + CR-01/CR-02) |
| Full workspace | `cargo test --workspace` | 0 failures / 0 panics across all crates | PASS |

### Probe Execution

No probe scripts declared. Step 7c: SKIPPED (no probe-*.sh files present).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BLD-01 | 02-02, 02-05 | Fluent ModelBuilder with orphan/topology validation | SATISFIED | State machine implemented; 11 validation tests; end-to-end via rewired XGBoost loader |
| BLD-02 | 02-02 | ConcatenateModelObjects | SATISFIED | concat.rs mirrors model_concat.cc; 4 tests pass |
| BLD-03 | 02-02 | BulkConstructTree fast path | SATISFIED | bulk.rs single-pass implementation; 2 tests pass |
| SER-01 | 02-03 | Model round-trips v5 binary | SATISFIED (self-consistent) / PARTIAL (upstream byte fidelity) | `serialize(deserialize(model)) == model` holds; upstream byte match holds only for the `deserialize(golden)` path, NOT for the builder-produced path (CR-01, CR-02) |
| SER-02 | 02-04 | Model serializes to v5 PyBuffer | SATISFIED | serialize_to_pybuffer produces 25-frame zero-copy vec; pointer-equality test confirms zero-copy |
| SER-03 | 02-04 | DumpAsJSON | SATISFIED | dump_as_json / dump_as_json_string match upstream WriteNode; 3 tests pass |
| SER-04 | 02-04 | Field accessors expose model/tree fields | SATISFIED | num_feature() + tree node accessors; 2 field tests pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/treelite-builder/src/lib.rs` | 445-483 | CR-02: unconditional stat column emission — data_count, sum_hess, gain and their present-flag columns always pushed length num_nodes even when no node called the setter | BLOCKER | Builder-produced models diverge from upstream byte image; fails upstream byte-fidelity (the project's core value for the builder path) |
| `crates/treelite-builder/src/lib.rs` | 469-488 | CR-01 (partial overlap with DEF-02-01): category_list_right_child, leaf_vector_begin, leaf_vector_end, category_list_begin, category_list_end not set — stay TreeBuf::empty() instead of length-num_nodes all-zero | BLOCKER | Same byte-fidelity divergence; mechanism now in builder (affects all loaders, not just XGBoost) |
| `crates/treelite-builder/src/concat.rs` | 157-158 | WR-03: post-condition uses debug_assert_eq! where upstream uses throwing TREELITE_CHECK_EQ | WARNING | Post-condition compiled out in release builds |
| `crates/treelite-builder/src/lib.rs` | 173-185 | WR-01: initialize_metadata omits target_id/class_id range checks and base_scores length check | WARNING | Malformed metadata accepted silently |

### Human Verification Required

None — all phase-2 truths are programmatically verifiable. Visual/UX checks are not applicable.

### Gaps Summary

The phase achieves its structural goal: all seven requirements (BLD-01 through SER-04) have substantive, wired, tested implementations; the 1e-5 equivalence gate passes; the serializer is byte-perfect against the upstream golden for the `deserialize→serialize` path; and the XGBoost loader is correctly rewired through the builder.

The single gap blocking the "passed" verdict is the byte-fidelity of the BUILDER PATH itself, which the phase's success criterion 3 ("models round-trip") implicates. The gap has two components, both in `end_tree`:

**CR-01 (builder mechanism, overlaps DEF-02-01 scope):** The five CSR-offset and category columns (`category_list_right_child`, `leaf_vector_begin`, `leaf_vector_end`, `category_list_begin`, `category_list_end`) are left empty. Upstream `AllocNode` pushes one per-node entry into each of these for every node (all-zero/false for trees with no categories/leaf-vectors). The responsibility now lives in the builder's `end_tree`, not in any individual loader.

**CR-02 (new, not covered by DEF-02-01):** The three stat columns (`data_count`, `sum_hess`, `gain` and their `_present` companions) are unconditionally emitted at length `num_nodes` (all-false). Upstream's invariant is "empty unless at least one node called the stat setter". The golden shows `data_count_present` length 0. DEF-02-01 describes these columns as empty in the prior Rust path; the builder actually makes them non-empty all-false — a new, distinct divergence.

**Disposition of CR-01 / CR-02 against criterion 3:**

- The self-consistency interpretation of criterion 3 — `serialize(deserialize(model)) == serialize(model)` — HOLDS. Any model (whether built or loaded) round-trips to itself. This is verified by the `round_trip_is_byte_identical` test.
- The upstream-byte-fidelity interpretation — `serialize(build_model()) == upstream_golden` for a model built through the builder/loader path — FAILS for both CR-01 and CR-02.

The project's core value statement is "predictions match upstream Treelite within 1e-5" (primary), with byte-fidelity as the serialization contract. The 1e-5 prediction gate passes. However, the serialization byte-fidelity gap means a treelite-rs-built model will not be readable by upstream C++ Treelite as the same model (the loaded-back model would differ in stat/CSR-offset columns). This is a real functional gap, not merely cosmetic.

CR-02 must be treated as a new tracked gap (not covered by DEF-02-01). CR-01's builder mechanism was newly created in this phase even though the symptom was previously noted in DEF-02-01 under a different attribution.

---

_Verified: 2026-06-10T00:23:49Z_
_Verifier: Claude (gsd-verifier)_

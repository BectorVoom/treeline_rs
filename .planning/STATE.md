---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 03-01-PLAN.md
last_updated: "2026-06-10T10:52:00.000Z"
last_activity: 2026-06-10 -- Completed Phase 03 Plan 01 (3-format fixtures + RED scaffold)
progress:
  total_phases: 9
  completed_phases: 2
  total_plans: 14
  completed_plans: 11
  percent: 24
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-09)

**Core value:** Predictions match upstream Treelite within 1e-5.
**Current focus:** Phase 03 — full-xgboost-loaders

## Current Position

Phase: 03 (full-xgboost-loaders) — EXECUTING
Plan: 2 of 4
Status: Executing Phase 03 (Plan 01 complete)
Last activity: 2026-06-10 -- Completed Phase 03 Plan 01 (3-format fixtures + RED scaffold)

Progress: [██████░░░░] 58%

## Performance Metrics

**Velocity:**

- Total plans completed: 11
- Average duration: ~5 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4 | - | - |
| 02 | 6 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 01 P02 | 4min | 2 tasks | 5 files |
| Phase 01 P03 | 4min | 2 tasks | 5 files |
| Phase 01 P04 | 3min | 2 tasks | 4 files |
| Phase 02 P01 | 10min | 3 tasks | 5 files |
| Phase 02 P02 | 7min | 2 tasks | 9 files |
| Phase 02 P03 | 75min | 2 tasks | 10 files |
| Phase 02 P04 | 6min | 2 tasks | 9 files |
| Phase 02 P05 | 10min | 2 tasks | 3 files |
| Phase 02 P06 | 6min | 2 tasks | 2 files |
| Phase 03 P01 | 12min | 3 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Vertical MVP slices laid along the upstream dependency DAG — Phase 1 is the thinnest load→predict→verify spine; later phases widen one layer each, ending runnable + 1e-5-tested.
- [Roadmap]: HistGradientBoosting confirmed in v1 scope (Phase 4) — the most complex sklearn loader path.
- [Roadmap]: CPU cubecl backend validated to 1e-5 (Phase 6) before any GPU backend is attempted (Phase 7).
- [01-01]: Enum variant names mirror upstream `kXxx` verbatim; `non_camel_case_types` suppressed at module level for porting fidelity.
- [01-01]: Inherent `from_str` (not `std::str::FromStr`) mirrors upstream `FromString` fallible-parse API; `clippy::should_implement_trait` suppressed.
- [01-01]: `TreeBuf<T>` is a two-mode enum `Owned(Vec<T>)`/`Borrowed{ptr,len}` with `T: Copy` POD bound; `bytemuck` deferred to Phase 9.
- [01-01]: Confirmed `num_class`/`leaf_vector_shape`/`target_id`/`class_id` are `Vec<i32>` (array-typed per tree.h:543-547), not scalars as ROADMAP wording implied.
- [Phase ?]: [01-02]: load_xgboost_json builds the F32 variant unconditionally — XGBoost-JSON only ever yields <f32,f32>.
- [Phase ?]: [01-02]: base_score margin transform stays in f64 throughout (sigmoid -ln(1/p-1)); objective.rs has zero f32 tokens.
- [Phase ?]: [01-02]: Per-tree parallel arrays validated against tree_param.num_nodes before building -> DimensionMismatch, never OOB (ERR-01).
- [Phase ?]: Harness: NaN in golden.json normalized to JSON null on read (serde_json rejects bare NaN); NanF32 maps null->f32::NAN — committed golden.json never edited
- [Phase ?]: Spine test passes with max |delta| = 0e0 — Rust pipeline bitwise-exact vs upstream Treelite 4.7.0 on binary:logistic fixture
- [02-01]: v5 header version constants are (4,7,0) NOT (5,x,x) — empirically confirmed by golden_v5.bin first 12 bytes (RESEARCH Pitfall 1 / Assumption A1 settled).
- [02-01]: Model owns 7 private v5 bookkeeping scalars staged at serialize time via stage_serialization_fields; pub(crate) accessors are the Pattern 5 borrow source for the in-crate serializer.
- [02-02]: treelite-builder ModelBuilder builds only the <f32,f32> preset in Phase 2; bulk_construct_tree yields Tree<f64> (sklearn doubles). node_id_map is a BTreeMap to mirror upstream std::map for deterministic orphan-error keying.
- [02-02]: leaf-vs-test mutual exclusivity is enforced structurally by the state machine (second detail call → WrongState), not a dedicated runtime conflict check. Orphan check always-on; D-08 validation toggle NOT ported.
- [02-02]: concatenate adds NO postprocessor/base_scores cross-input equality checks — upstream model_concat.cc lacks them (BLD-02 fidelity).
- [Phase ?]: 02-03: golden byte-fidelity proven via serialize(deserialize(golden_v5.bin))==blob, making the serializer gate loader-independent; XGBoost loader fidelity gap deferred (DEF-02-01)
- [02-04]: DumpAsJSON reuses the existing enum as_str() spellings verbatim (D-04); no new strings invented; dump_as_json takes &mut Model to stage variant-derived type tags (mirrors upstream GetThresholdType()/GetLeafOutputType()).
- [02-04]: D-04 equivalence asserted at the PARSED-value level, never by byte-comparing serialized JSON (RapidJSON vs serde_json float formatting differs, A4/Q3).
- [02-04]: Model v5 bookkeeping readers promoted pub(crate)→pub (read-only, NO setter) as the SER-04 inspection surface, preserving field_accessor.cc Set-rejection fidelity (T-02-J02).
- [02-05]: load_xgboost_json rewired through treelite_builder::ModelBuilder (D-11) — 11 builder calls, 0 TreeBuf::from_owned in build path; loader validators (require_non_negative/check_dim) run BEFORE builder emission; builder errors propagate as XgbError::Builder (thiserror transparent, no panic, no anyhow).
- [02-05]: 1e-5 regression gate proves the rewiring is bit-identical — equivalence max |delta| = 0e0 < 1e-5 (Phase 2 success criterion 1, second half); objective→postprocessor map, f64 base_score margin transform, and F32-only variant all unchanged.
- [02-06]: end_tree ports upstream AllocNode (detail/tree.h:70-101) verbatim — the five per-node CSR/category columns (category_list_right_child, leaf_vector_begin/end, category_list_begin/end) are length num_nodes (CR-01); begin/end default to 0 because the builder leaves leaf_vector_/category_list_ value buffers empty.
- [02-06]: stat columns (data_count/sum_hess/gain + _present) are empty-unless-set per-column (CR-02), gating TreeBuf::empty() vs from_owned on any_* flags — mirrors upstream's if(!_present_.Empty()) guards; deserializer reads by column length so serialize→deserialize stays self-consistent (no regression to golden round-trip or 1e-5 equivalence).
- [03-01]: A1 settled empirically — xgboost 1.7.6 writes genuine legacy binary (a 4-byte `binf` magic prefix + 136-byte LearnerModelParam; base_score=0.5 @0, num_feature=4 @4). Spike resolved autonomously, no human checkpoint; 1.6.2/0.90 fallbacks not needed.
- [03-01]: base_score=0.5 (A2) makes the version-gated sigmoid margin transform a no-op, so all three XGBoost formats serialize to ONE identical v5 blob (golden_v5_3format.bin, 7775 bytes, sha256 ae53fbf8…) — proven at generation time by the A2 cross-format same-blob assert.
- [03-01]: Legacy binary uses treelite's separate load_xgboost_model_legacy_binary entry point (not load_xgboost_model, which only handles JSON/UBJSON and mis-sniffs the binf-prefixed legacy file) — mirrors upstream's D-09 API split; the Rust load_xgboost_legacy must handle the binf magic per D-07.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3] serde_json rejects NaN/Inf by default; XGBoost JSON uses them — resolve in the XGBoost loader research-phase.
- [Phase 5/6] cubecl control-flow constraints (`continue` unsupported, helpers must be `#[cube]`) and CPU-backend op gaps — spike a minimal kernel before the full port.
- [Phase 5] Golden-vector reproducibility — store actual input matrices + a toolchain/libm/framework manifest, not just seeds.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Loader fidelity | DEF-02-01: XGBoost loader→serialize byte-fidelity gap (serializer gate proven loader-independent via golden round-trip) | Deferred to Phase 3 | 02-03 |

## Session Continuity

Last session: 2026-06-10T10:52:00.000Z
Stopped at: Completed 03-01-PLAN.md
Resume file: None

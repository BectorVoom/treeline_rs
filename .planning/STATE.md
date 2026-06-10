---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: 02-05 complete — Phase 2 plans 5/5 done; next is phase verification
last_updated: "2026-06-10T00:30:00.000Z"
last_activity: 2026-06-10 -- 02-05 complete; load_xgboost_json rewired through ModelBuilder (D-11), equivalence max |delta| = 0e0 < 1e-5, workspace green
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 9
  completed_plans: 9
  percent: 11
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-09)

**Core value:** Predictions match upstream Treelite within 1e-5.
**Current focus:** Phase 02 — builder-serialization

## Current Position

Phase: 02 (builder-serialization) — 5/5 PLANS COMPLETE (ready for phase verification)
Plan: 5 of 5 — COMPLETE
Status: Plan 02-05 COMPLETE — load_xgboost_json rewired through treelite_builder::ModelBuilder (D-11) at 5cfa84e; validators preserved ahead of builder emission, builder errors propagate as XgbError::Builder; 1e-5 regression gate green (max |delta| = 0e0); workspace green; next: Phase 2 verification then Phase 3 (Full XGBoost Loaders)
Last activity: 2026-06-10 -- 02-05 complete; treelite-xgboost gains treelite-builder dep, loader emits 11 ModelBuilder calls / 0 TreeBuf::from_owned in build path, workspace green

Progress: [██████░░░░] 56%

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: ~5 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4 | - | - |

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

Last session: 2026-06-10T00:30:00.000Z
Stopped at: 02-05 complete — Phase 2 plans 5/5 done; next is Phase 2 verification
Resume file: .planning/phases/02-builder-serialization/ (phase verification)

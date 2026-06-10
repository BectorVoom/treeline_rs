---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: gap_closure
stopped_at: Phase 05 plans built; CR-01 blocker found in code review — routing to gap closure
last_updated: "2026-06-10T07:41:18.592Z"
last_activity: 2026-06-10 -- Phase 05 code review found CR-01 (f64 postprocessor precision); gap closure pending
progress:
  total_phases: 9
  completed_phases: 4
  total_plans: 27
  completed_plans: 27
  percent: 44
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-09)

**Core value:** Predictions match upstream Treelite within 1e-5.
**Current focus:** Phase 05 — full-scalar-gtil-equivalence-harness

## Current Position

Phase: 05 (full-scalar-gtil-equivalence-harness) — GAP CLOSURE
Plan: 5 of 5 built (not yet verified)
Status: Pending CR-01 gap closure — f64-input postprocessors run at f32 (upstream runs in f64); see 05-REVIEW.md
Last activity: 2026-06-10 -- Phase 05 code review found CR-01; gap closure planned

Progress: [████████] 100% (Phase 04 plans: 8/8)

## Performance Metrics

**Velocity:**

- Total plans completed: 24
- Average duration: ~5 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4 | - | - |
| 02 | 6 | - | - |
| 03 | 4 | - | - |
| 04 | 8 | - | - |

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
| Phase 03 P02 | 18min | 2 tasks | 7 files |
| Phase 03 P03 | 22min | 2 tasks | 6 files |
| Phase 03 P04 | 30min | 2 tasks | 6 files |
| Phase 04 P01 | 12min | 2 tasks | 10 files |
| Phase 04 P02 | 5min | 2 tasks | 5 files |
| Phase 04 P03 | 6min | 2 tasks | 10 files |
| Phase 04 P04 | 6min | 2 tasks | 7 files |
| Phase 04 P06 | 13min | 2 tasks | 11 files |
| Phase Phase 04 PP05 | 5min | 2 tasks tasks | 8 files files |
| Phase 04 P07 | ~2min | 1 tasks | 3 files |
| Phase 04 P08 | ~8min | 2 tasks | 4 files |
| Phase 05 P01 | 12min | 2 tasks | 4 files |
| Phase 05 P02 | 22min | 2 tasks | 7 files |
| Phase 05 P03 | ~8min | 2 tasks | 3 files |
| Phase 05 P04 | ~6min | 2 tasks | 5 files |
| Phase 05 P05 | 30min | 2 tasks | 6 files |

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
- [03-02]: D-02 NaN/Inf resolved via a string-safe value-position pre-lexer (replace_nonfinite) that rewrites bare NaN/Infinity/-Infinity to sentinel STRINGS recovered by de_f32 — never a numeric literal (serde_json rejects out-of-range); string contents are byte-unchanged. Closes the Phase-3 NaN/Inf blocker.
- [03-02]: Shared build_model_from_parsed(XgbModelJson) is the single convergence path (D-01) for all three formats; load_xgboost_json = replace_nonfinite → from_str → build_model_from_parsed. 03-03 (UBJSON via from_value) and 03-04 (legacy) reuse it.
- [03-02]: DEF-02-01 closed for the JSON path — serialize(load_xgboost_json(xgb_3format.json)) == golden_v5_3format.bin byte-for-byte, achieved by emitting sum_hess on every node + gain on internal nodes + attributes:None (→ "{}").
- [03-02]: parse_base_score handles scalar AND vector base_score forms; xgb_3format.json uses the vector form ("[5E-1]"), so the vector path is exercised by the real byte-fidelity/predict tests. expand_to = num_target * max(num_class,1); cast f32→f64 BEFORE the element-wise version-gated transform (Pitfall 3).
- [03-02]: treelite-harness is NOT a dev-dep of treelite-xgboost (cycle: harness depends on xgboost); the JSON predict test parses the golden locally instead.
- [03-03]: UBJSON is BIG-ENDIAN (network byte order) — all multi-byte ints/floats decode via from_be_bytes; confirmed empirically against xgb_3format.ubj (key length `L 00…07`=7, `[$d#L 00…0F`=15). The initial little-endian draft was caught by the byte-identical-to-golden test (Rule 1 fix).
- [03-03]: UBJSON shares the JSON numeric path (D-01/D-03) — decode_ubjson emits serde_json::Value (with @NaN@/@Inf@/@-Inf@ sentinel STRINGS for non-finite d/D floats, Pitfall 5) → from_value into the SAME XgbModelJson + de_f32 adapter → build_model_from_parsed. UBJSON load == JSON load == golden_v5_3format.bin byte-for-byte (D-10).
- [03-03]: $/# strongly-typed optimized-container fast path (Pitfall 4) is mandatory — XGBoost emits [$<type>#<count> everywhere (split_conditions etc.); per-element tags omitted. $/# counts validated against remaining bytes before pre-alloc (T-03-U01); fallible cursor for truncation (T-03-U02) → typed XgbError::Ubjson, never OOB/OOM.
- [03-03]: DetectXGBoostFormat ported verbatim (D-09) returns json/ubjson/unknown only — legacy is NOT auto-detected (reached via explicit load_xgboost_legacy in 03-04, matching upstream's API split).
- [03-04]: Legacy binary decodes field-by-field via a from_le_bytes Cursor (D-07/D-08) — no native-endian struct reinterpret (grep gate). Converges at build_model_from_parsed via XgbModelJson::from_legacy_fields, producing a Model byte-identical to JSON/UBJSON; sindex bit-unpack (&0x7FFFFFFF / >>31), cleft==-1 leaf, info-union f32 reinterpret ported exactly (Pitfall 6).
- [03-04]: GBTreeModelParam is 160 BYTES, not 168 (RESEARCH transcription error) — upstream struct is 4×i32+i64+2×i32+i32[32]=160, no trailing padding; confirmed against mushroom.model (header@173 + 160 + 1168 == 1501). Rule-1 fix caught by the mushroom smoke test.
- [03-04]: Version gate mapped by setting version=[major_version] so the shared path's version[0]>=1 gate reproduces the legacy major_version>=1 gate (XGB-05); mushroom (major_version 0) is the negative case, verified ±.
- [04-01]: f64 ModelBuilder mode is parallel-staging (RESEARCH Open Q2 option B), NOT generic-over-T — NodeStaging carries both f32 and f64 value fields, end_tree branches on a latched is_f64 flag; the f32 XGBoost path stays byte-identical. leaf_scalar_f64/leaf_vector_f64/numerical_test_f64 store f64 WITHOUT downcast (D-05); commit_model wraps ModelVariant::F64(ModelPreset::new(trees)). A shared fill_common! macro keeps the 25-column Tree shape (CR-01/CR-02) identical across both modes.
- [04-01]: MixedNumericMode error rejects mixing f32/f64 entry points in one builder (Rule 2) — protects the 1e-5 fidelity gate from silent downcast/discard.
- [04-01]: bulk_to_model (bulk.rs) wraps Vec<Tree<f64>> + BuilderMetadata into a ModelVariant::F64 Model, hand-setting all 10 header fields (sklearn_bulk.cc:244-330); sigmoid_alpha/ratio_c keep Model::new 1.0 defaults, attributes defaults to {}; no topology re-check (D-09, T-04-02 accepted).
- [04-01]: treelite-lightgbm/treelite-sklearn registered in root Cargo.toml as PLACEHOLDER crates (Rule 3) — registering non-existent members breaks the whole workspace, so minimal doc-only lib stubs keep cargo valid; real loaders land in Plans 04-04/04-06.
- [03-04]: DEF-02-01/D-10 CLOSED across all three formats — serialize(load_json)==serialize(load_ubjson)==serialize(load_legacy)==golden_v5_3format.bin; three_format_equivalence + golden_v5 loader assertion promoted to fatal; cargo test --workspace fully green.
- [04-02]: gtil::predict widened to a FLAT row-major (num_row, num_target, max_num_class) buffer (the Array3DView storage) — binary stays length num_row, byte-identical to Phase 1; no new public entry-point. Four-way OutputLeafValue/OutputLeafVector branch on (target_id[tree]==-1, class_id[tree]==-1) ports predict.cc:174-229; RF averaging (predict.cc:259-293) + f64 2D base-score add (:294-304); serial tree-sum (GTIL-08) and T::to_f32 cast preserved.
- [04-02]: four postprocessors ported verbatim cast-order — exponential_standard_ratio uses exp2 (BASE-2, not exp) with f32 ratio_c; softmax = f32 max-subtraction + f64 norm_const accumulate + f32 divide (the 1e-5 contract). signed_square/hinge/identity_multiclass/multiclass_ova deferred to Phase 5.
- [04-02]: bounds-checked output routing → typed GtilError::OutputRouteOutOfBounds / LeafVectorTooShort (T-04-03 mitigated, never OOB write/panic); has_leaf_vector made bounds-safe in the gtil layer (absent/short CSR offsets → scalar path) so malformed/hand-crafted trees never panic (ERR-01).
- [Phase ?]: [04-03]: Estimator goldens captured from treelite.gtil.predict default (post-processed) kind, NOT framework predict (D-07); each capture script asserts the GTIL default kind so an API default change is caught.
- [Phase ?]: [04-03]: IsolationForest golden == -clf.score_samples(X) cross-checked at capture time (max delta 6.9e-9 < 1e-5) — the canonical Treelite-not-framework case (D-07).
- [Phase ?]: [04-03]: HistGB packed node itemsize is 56 (64-bit feature-index variant) on this env; nodes frozen as base64; numerical fixture has identity features_map, categorical carries categories_map (Pitfall 4 split).
- [Phase ?]: [04-03]: LightGBM numerical golden references vendored deep_lightgbm/model.txt (no dup); categorical model fit with max_cat_to_onehot=1 to force bitset splits (num_cat=3) for LGB-02.
- [04-04]: treelite-lightgbm fleshed out from the Plan-01 placeholder, mirroring treelite-xgboost (parse.rs/objective.rs/error.rs/lib.rs converge-then-build). LightGBM loads into ModelVariant::F64 unconditionally — leaf_value/threshold are f64 emitted via numerical_test_f64/leaf_scalar_f64, NO f32 downcast (D-02/D-05).
- [04-04]: Negative-index leaf re-numbering ported verbatim (lightgbm.cc:533-601): BFS deque seeded (-1,1) single-leaf / (0,1) otherwise; dfs_index starts at 1, +2 per internal node; leaf value = leaf_value[!old_node_id]; children push_front. Missing-type default_left override = (0.0 <= threshold) when missing_type != kNaN (Pitfall 3); operator always kLE.
- [04-04]: CanonicalObjective alias-collapse runs BEFORE the objective→postprocessor map; sigmoid:<a> parsed with strict >0 check (T-04-09); class_id[i]=i%num_class; average_tree_output from average_output key presence; base_scores = num_class zeros; sigmoid_alpha stamped post-commit.
- [04-04]: Categorical splits rejected with a typed LgbError in this numerical slice (LGB-01) rather than mis-predicting — cat bitset decode (LGB-02) is Plan 04-05; cat_boundaries(u64)/cat_threshold(u32) are already parsed and stored on LGBTree so 04-05 only adds emission.
- [04-04]: lightgbm_numerical golden gate green at max |delta| = 0e0 (bitwise-exact vs upstream treelite.gtil.predict); harness gained a treelite-lightgbm dev-dep; cargo test --workspace fully green (no XGBoost regression).
- [Phase ?]: 04-06: treelite-sklearn RF/ET via bulk path, GB via f64 ModelBuilder MixIn; SKL-01/02 within 1e-5 (worst 5.96e-8).
- [Phase ?]: 04-06: GB base_scores derived capture-side per importer.py, added to golden additively (frozen input,output sha256 unchanged); gtil gained identity_multiclass no-op.
- [Phase ?]: [04-05]: LightGBM categorical bitset decoded via BitsetToList ported verbatim (lightgbm.cc:210-221, word=bits[i/32] bit=i%32 LSB-first); decoder takes &[u32] so word index is structurally in-bounds (T-04-11). Categorical node's threshold field is repurposed as cat_idx; slice cat_threshold via cat_boundaries, bounds-checked (T-04-10).
- [Phase ?]: [04-05]: Categorical splits ignore missing_type (default_left=false, category_list_right_child=false, NaN->right, lightgbm.cc:569-573). treelite-builder categorical_test extended to carry the category list + polarity (SetCategoricalTest) and made mode-agnostic (no guard_f32) so the f64 LightGBM path uses it; end_tree flattens per-node staging into the CSR category_list columns.
- [Phase ?]: [04-05]: Minimal NextNodeCategorical GTIL branch (D-03): integer membership + polarity (predict.cc:128-150); load-bearing subset of the float-representability guard applied, exhaustive matrix (GTIL-06) deferred to Phase 5. category_list_safe wrapper returns empty on OOB CSR slice (T-04-12). lightgbm_categorical golden max |delta|=9.54e-7 < 1e-5; workspace green.
- [Phase ?]: 04-07: IsolationForest ratio_c assigned post-commit (Model field, not BuilderMetadata), mirroring upstream PostProcessorFunc config
- [Phase ?]: 04-07: IsolationForest leaf isolation depths consumed AS-IS via shared GB build_tree (no loader-side recomputation, D-07); zero ratio_c rejected (T-04-17)
- [04-08]: HistGB packed HistGradientBoostingNode decoded field-by-field via from_le_bytes at a NodeLayout offset table parameterized by itemsize (52 = i32 feature_idx, 56 = i64); NO transmute/bytemuck (Phase-3 D-08, grep-clean). itemsize ∉ {52,56}, short nodes buffer, OOB feature_idx, and OOB categorical bitset row all rejected with typed SklError::HistGbDecode BEFORE any field read (T-04-18/19/20/21).
- [04-08]: leaf detection is left==0 (HistGB missing-child marker, NOT the sklearn-tree ==-1 rule); split_index = features_map[feature_idx] ALWAYS applied (Pitfall 4); num_threshold read DIRECTLY (Pitfall 5, no _bin_mapper recon); known_cat_bitsets UNUSED in v4.7.0 (A3) so omitted from the Rust signatures.
- [04-08]: HistGB categorical check(bitmap,val,row)=(bitmap[8*row+val/32]>>(val%32))&1 ported verbatim — the 8*row (8 uint32 = one 256-bit row) stride is a SEPARATE function from LightGBM's bitset_to_list (different layout, RESEARCH No-Analog). cat_transform = categories_map[fid][cat] when present else identity.
- [04-08]: sklearn_histgb_numerical max |delta| = 0e0; sklearn_histgb_categorical max |delta| = 1.19e-7 (f32-quant floor) — both « 1e-5. SKL-04 closed; Phase 4 complete. Harness uses a self-contained base64 decoder (no new dependency).
- [Phase ?]: [05-01]: 64 frozen GTIL goldens (binary + unconditional multiclass leaf_vec) capture the exhaustive matrix; dense==CSR parity asserted at capture time (D-04); non-finite cells encoded as null/inf/-inf tokens (D-08 contract)
- [Phase ?]: [05-01]: RED Wave-0 scaffolds (gtil_matrix runner + 3 postprocessor stubs + categorical_full_guard) compile and are ignored with reason strings as Nyquist MISSING markers; existing workspace suite stays green
- [Phase ?]: Plan 05-02: GTIL next_node compares in f64 (exact f32->f64 widening is order-preserving) — bit-faithful routing across all 4 input×preset combos
- [Phase ?]: Plan 05-02: f64 element-wise postprocessor arithmetic deferred to Plan 05-03; apply_postprocessor uses an f32 boundary so postprocessor.rs intermediates stay f32 (Pitfall 2)
- [Phase ?]: GTIL PredictOut representability const MANTISSA_BITS (not DIGITS: inherent f32::DIGITS shadows Self::DIGITS to decimal-6)
- [Phase ?]: RowSource enum gives structural dense==sparse parity (D-04): both paths materialize one reusable scratch row that evaluate_tree walks verbatim
- [Phase ?]: LeafId/ScorePerTree size output on actual trees.len() (GetNumTree), not the staged num_tree() header field
- [Phase ?]: Plan 05-05: committed treelite v5 model bytes loaded via treelite_core::deserialize (the exact model the goldens were captured from); frozen goldens untouched — Rule 3 fix for Plan-01's discarded in-script models.
- [Phase ?]: Plan 05-05: minimal fn-pointer Backend/RunnerCase seam (four input-dtype slots, f64 output) — Phase 6 registers a cubecl runtime by adding a variant + constructor with no matrix-iteration change (D-11).

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 3] ~~serde_json rejects NaN/Inf by default; XGBoost JSON uses them~~ — RESOLVED in 03-02 via the string-safe replace_nonfinite pre-lexer + de_f32 sentinel adapter (D-02).
- [Phase 5/6] cubecl control-flow constraints (`continue` unsupported, helpers must be `#[cube]`) and CPU-backend op gaps — spike a minimal kernel before the full port.
- [Phase 5] Golden-vector reproducibility — store actual input matrices + a toolchain/libm/framework manifest, not just seeds.
- [Phase 5] **CR-01 (BLOCKER, OPEN)** — `apply_postprocessor` (gtil/src/lib.rs:1138) narrows every f64 output cell to f32, runs the postprocessor in f32, then widens back. Upstream `postprocessor.cc` templates sigmoid/exponential/exponential_standard_ratio/logarithm_one_plus_exp/signed_square/multiclass_ova/hinge on `InputT`, so they run in **f64** for f64 input (only `softmax` hardcodes f32). The f64×non-identity/non-softmax postprocessor surface therefore runs at wrong precision — a latent 1e-5 violation masked only because current f64 sigmoid goldens sit ~1e-7 inside band. Known deferral from 05-02 that 05-03 claimed to close but did NOT wire. Fix = make postprocessors O-generic (softmax stays f32) + capture an f64 fixture that actually stresses the path. Also open: WR-01 (sparse harness re-derives CSR from NaN-presence, never asserts a real captured CSR), WR-03 (evaluate_tree no node-0 bounds check → panic on 0-node tree, violates ERR-01), WR-02 (predict_score_by_tree lvs.max(1) vs output_shape disagree), WR-04/WR-05 (silent fallbacks mask wrong predictions). See 05-REVIEW.md.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Loader fidelity | DEF-02-01: XGBoost loader→serialize byte-fidelity gap | FULLY CLOSED in 03-04 — JSON (03-02) + UBJSON (03-03) + legacy (03-04) all serialize to golden_v5_3format.bin byte-for-byte; cross-format single-golden assertion is fatal and green | 02-03 |

## Session Continuity

Last session: 2026-06-10T07:40:37.101Z
Stopped at: Completed 05-04-PLAN.md
Resume file: None

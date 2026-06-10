---
phase: 04
slug: lightgbm-scikit-learn-loaders
status: verified
threats_open: 0
asvs_level: 2
created: 2026-06-10
---

# Phase 04 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Verify-mitigations mode: the register is plan-time-complete (extracted from all
> 8 `04-0N-PLAN.md` `<threat_model>` blocks). Each `mitigate` threat is verified
> by locating its actual guard in the implemented code (file:line); each `accept`
> threat is verified as legitimately accepted and logged below.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Untrusted model file → LightGBM loader | `load_lightgbm(&str)` parses an attacker-controllable `model.txt` | `key=value` text: counts, arrays, categorical bitsets |
| Untrusted model arrays → sklearn MixIn loader | per-node `children/feature/threshold/value/...` arrays from a pickled estimator dump | `i64`/`f64` parallel arrays indexed by node id |
| Untrusted packed buffer → HistGB decoder | `expected_sizeof_node_struct` + packed `nodes` byte buffer | raw little-endian bytes, `feature_idx`, `bitset_idx`, 256-bit category bitmap |
| Loader output → GTIL prediction | a loaded `Model`'s `target_id`/`class_id`/`num_class` route output cells | routing indices into the `(num_row, num_target, max_num_class)` buffer |
| Loader → ModelBuilder | per-node Begin/End calls with raw child keys | node keys, child keys, leaf/threshold values |
| Bulk fast path (RF/ET) | `bulk_construct_tree`/`bulk_to_model` deliberately bypass per-node validation | pre-validated arrays — caller (sklearn bulk loader) owns the bounds-check |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation / Evidence | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-04-01 | DoS | builder f64 `end_tree` column-fill | mitigate | `crates/treelite-builder/src/lib.rs:521-627` — every column built at exactly `num_nodes` length; child keys resolved via map → `DanglingChildKey`, orphans → `OrphanedNode`, empty → `EmptyTree`; `fill_common` macro keeps f32/f64 column shape identical. No OOB/panic. | closed |
| T-04-02 | Tampering | `bulk_to_model` trusts un-validated tree input | accept | Validation BYPASS by design (D-09): `crates/treelite-builder/src/bulk.rs:8-36`. Trust boundary pushed to the caller, which DOES bounds-check: `crates/treelite-sklearn/src/bulk.rs:107-127`. Legitimately accepted (see Accepted Risks R-04-02). | closed |
| T-04-03 | DoS | GTIL output-buffer routing by `class_id`/`target_id` | mitigate | `crates/treelite-gtil/src/lib.rs:393-403` (scalar), `:443-495` (leaf-vector) — bounds-check vs `num_target`/`max_num_class` before `idx()` → `GtilError::OutputRouteOutOfBounds` (`error.rs:55-71`). | closed |
| T-04-04 | DoS | `softmax` norm_const division | accept | `crates/treelite-gtil/src/postprocessor.rs:112-132` — verbatim upstream port; degenerate all-`-inf` row yields NaN matching upstream, not adversarially reachable from a well-formed model. (See Accepted Risks R-04-04.) | closed |
| T-04-05 | Tampering | sklearn/lightgbm capture-only pip installs | mitigate | Capture-only, never in the Rust build graph. Confirmed: root + crate `Cargo.toml` declare only `thiserror` + internal `path` crates — no Python/third-party crate in the build graph. | closed |
| T-04-06 | Repudiation | golden drift on sklearn version change | mitigate | Manifest records versions+seed; capture is write-once (D-06). No code in the Rust build graph; versions pinned in fixtures manifests. | closed |
| T-04-07 | DoS | LightGBM text parser count→OOB slice | mitigate | `crates/treelite-lightgbm/src/parse.rs:93-124` (`parse_array` rejects short token counts), `:216-225` (`num_leaves < 0` guard + `saturating_sub`), `:249-285` (exact cat array lengths) → `LgbError::Parse`/`Bitset`. | closed |
| T-04-08 | DoS | negative-index leaf decode OOB | mitigate | `crates/treelite-lightgbm/src/lib.rs:138-145` — `leaf_idx` bounds-checked vs `n_leaf` before `leaf_value[..]` → `LgbError::LeafIndexOutOfRange`; internal node index checked `:161-172`. | closed |
| T-04-09 | Tampering | `sigmoid_alpha <= 0` | mitigate | `crates/treelite-lightgbm/src/objective.rs:205-217` (`require_positive_alpha`, `a > 0.0`), invoked `:105,:113` for binary/multiclassova → `LgbError`. | closed |
| T-04-10 | DoS | `cat_boundaries` slicing OOB | mitigate | `crates/treelite-lightgbm/src/parse.rs:249-285` — monotone-boundary + exact `cat_threshold.len() == back()` checks BEFORE slicing → `LgbError::Bitset`; re-checked at use `lib.rs:199-220`. | closed |
| T-04-11 | DoS | `BitsetToList` indexes past bitset words | mitigate | `crates/treelite-lightgbm/src/bitset.rs:40-53` — takes `bits: &[u32]`, derives `nslots = bits.len()`; word index `i/32` structurally `< nslots`. Cannot read past the slice. | closed |
| T-04-12 | DoS | GTIL `category_list` begin/end slice OOB | mitigate | `crates/treelite-gtil/src/lib.rs:131-147` (`category_list_safe`) — reads offsets via `.get(nid)`, returns `&[]` on inverted/out-of-range slice; called at `:185`. | closed |
| T-04-13 | DoS | negative/OOB `children_left/right` | mitigate | MixIn path `crates/treelite-sklearn/src/mixin.rs:123-137`; bulk path `crates/treelite-sklearn/src/bulk.rs:107-127` — `child < 0 || child >= n_nodes` → `SklError::ChildIndexOutOfRange` before any gain deref. | closed |
| T-04-14 | DoS | `node_count` overflow on int cast | mitigate | `crates/treelite-sklearn/src/mixin.rs:90-104`, `bulk.rs:77-91`, `histgb.rs:342-355` — `node_count < 0` and `> i32::MAX` → `SklError::InvalidScalar`. | closed |
| T-04-15 | Spoofing | GB leaf learning_rate re-shrink | mitigate | `crates/treelite-sklearn/src/mixin.rs:18-20,150-151` — leaves consumed AS-PROVIDED; grep-clean: no `* learning_rate` anywhere in `treelite-sklearn/src`; asserted by `gb_regressor_uses_leaf_values_as_provided_no_reshrink` (`mixin.rs:615`). | closed |
| T-04-16 | DoS | IsolationForest OOB child indices | mitigate | IsolationForest routes through `build_model`→`build_tree` (`crates/treelite-sklearn/src/mixin.rs:309`,`:230`) reusing the T-04-13 bounds-check (`:123-137`); test `iforest_consumes_leaf_depth_as_is_no_recomputation` (`:594`). | closed |
| T-04-17 | Tampering | `ratio_c == 0`/non-finite div-by-zero | mitigate | `crates/treelite-sklearn/src/mixin.rs:288-294` — `ratio_c == 0.0 || !ratio_c.is_finite()` → `SklError::InvalidScalar` before commit; test `iforest_rejects_zero_ratio_c` (`:583`). | closed |
| T-04-18 | Tampering/DoS | bad HistGB itemsize / short buffer | mitigate | `crates/treelite-sklearn/src/histgb.rs:97-108` (itemsize ∈ {52,56} → `HistGbDecode`), `:357-375` (`nodes_bytes.len() < node_count*itemsize` checked BEFORE decode), plus per-field `rec.get(off..off+N)` guards `:154-196`. | closed |
| T-04-19 | DoS | `feature_idx` OOB into features/categories map | mitigate | `crates/treelite-sklearn/src/histgb.rs:394-406` (`usize::try_from` + `features_map.get(fid)`), `:280-289` (`categories_map.get(fid)`) → `HistGbDecode`. | closed |
| T-04-20 | DoS | `bitset_idx` OOB into category bitmap | mitigate | `crates/treelite-sklearn/src/histgb.rs:257-275` — full 256-bit row `[8*bitset_idx, 8*bitset_idx+8)` range-checked (checked_mul overflow guard) before scan; `check_bit` `:234-238` uses `bitmap.get(word)`. | closed |
| T-04-21 | Tampering | native-endian transmute UB | mitigate | `crates/treelite-sklearn/src/histgb.rs:154-218` — field-by-field `from_le_bytes`; grep gate clean: no `transmute`/`bytemuck` in any audited `src/` (only the doc comment naming the ban). | closed |
| T-04-SC | Tampering | npm/pip/cargo installs (recurs per plan) | mitigate | Supply-chain gate PASSED. Root `Cargo.toml` + every audited crate `Cargo.toml` declare only `thiserror` (pre-existing workspace dep) and internal `path` crates. No new third-party crate entered the build graph this phase. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| R-04-02 | T-04-02 | `bulk_to_model`/`bulk_construct_tree` are an explicit validation-BYPASS fast path (D-09). The trust boundary is pushed to the caller; the only in-tree caller (`treelite-sklearn/src/bulk.rs`) bounds-checks every child index and array length (`bulk.rs:107-127`) before invoking the bypass. No untrusted bytes reach the bypass un-validated. | gsd-security-auditor | 2026-06-10 |
| R-04-04 | T-04-04 | `softmax` is a verbatim upstream port (`postprocessor.cc:57-75`). A degenerate all-`-inf` row divides by `norm_const == 0` → NaN, identical to upstream behavior. Not adversarially reachable from a well-formed loaded model (leaf margins are finite); diverging from upstream here would break the 1e-5 fidelity contract. | gsd-security-auditor | 2026-06-10 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-10 | 22 | 22 | 0 | gsd-security-auditor |

Notes:
- 22 = 21 distinct threats (T-04-01..T-04-21) + T-04-SC (supply-chain, declared once per plan, verified once).
- Disposition split: 20 `mitigate` (all guards located in code), 2 `accept` (T-04-02, T-04-04 — both logged above).
- Implementation files were not modified. Verification was read-only + a build of the four audited crates (clean).
- No `## Threat Flags` section appears in any `04-0N-SUMMARY.md`; the summaries document only the declared register. No unregistered attack-surface flags found.
- Panic-safety grep gate: no `unwrap()`/`expect()`/`panic!`/`unreachable!`/`transmute`/`bytemuck`/`unsafe` in non-test paths of the audited `src/`. The two non-test `expect`/`panic` matches (`parse.rs:170` `in_tree` invariant; `lib.rs` doc) are provably unreachable from untrusted input.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-10

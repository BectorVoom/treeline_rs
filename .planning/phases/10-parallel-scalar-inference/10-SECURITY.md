---
phase: 10
slug: parallel-scalar-inference
status: verified
threats_open: 0
asvs_level: high
created: 2026-06-11
---

# Phase 10 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

This phase introduced the project's first parallelism (rayon row-parallel scalar
GTIL) and its first `unsafe` concurrency surfaces (`unsafe impl Sync` on `Model`
and `TreeBuf<T>`, raw-pointer foreign buffers shared across workers, a scoped
thread pool sized by an untrusted `Config.nthread`). The register below was
authored at plan time across `10-00-PLAN.md` and `10-01-PLAN.md` and verified
against the implementation by `gsd-security-auditor` (ASVS high).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| caller / Python → `Config.nthread` | untrusted integer sizes a scoped rayon thread pool | `i32` worker count |
| `&Model` shared across rayon workers | concurrent read-only access to `TreeBuf::Borrowed { *const T }` foreign buffers | immutable model node arrays (`*const T`) |
| CSR input → parallel row body | untrusted `col_ind` / `row_ptr` validated once, then read per row in parallel | sparse-matrix index arrays |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-10-01 | Denial of Service | `nthread`-sized scoped pool (predict_preset/leaf/score) | mitigate | `run_with_nthread` (lib.rs:649-663): `nthread<=0`→core-capped global pool; `nthread>0`→`ThreadPoolBuilder::num_threads(n).build().map_err(GtilError::ThreadPool)?` then `pool.install` — typed error, no panic. `build_global` never called (forbidden in doc only, lib.rs:646). Per-worker scratch is `num_feature`-sized (not `nthread`-driven). `GtilError::ThreadPool` at error.rs:178-179. | closed |
| T-10-02 | Denial of Service | CSR validation under parallelism | mitigate | Dense buffer-length check up front (lib.rs:996-1004, `saturating_mul`); CSR validated once (lib.rs:1047 `csr.validate(num_row, num_feature)?`) before `predict_rows`. No validation inside any `par_chunks_mut`/`map_init` closure (:723, :1178, :1281). | closed |
| T-10-03 | Tampering / Info Disclosure | concurrent reads of the shared `Model` foreign buffer | mitigate | `unsafe impl Sync for Model` (model.rs:130) with read-only-predict SAFETY argument (:119-129); `unsafe impl<T: Copy + Sync> Sync for TreeBuf<T>` (tree_buf.rs:56, tightened per WR-02). Disjoint writes via `par_chunks_mut` (no manual unsafe indexing); inner tree loop serial; contract pinned by `requires_sync::<Model>()` (model_invariants.rs:34). Determinism tests prove no run-to-run reordering. | closed |
| T-10-04 | Denial of Service | panic crossing the rayon boundary | mitigate | Closures return `Result<(), GtilError>` with `.collect::<Result<_,_>>()?` short-circuit (lib.rs:745, 1196, 1318). Residual worker panic trapped by treelite-py `guard_assert` → `catch_unwind` (gtil.rs:270,302 → error.rs:110-127) remapping to `TreeliteError`. | closed |
| T-10-SC | Supply chain (Tampering) | crates.io install of `rayon` | mitigate | `Cargo.toml:39` `rayon = "1.12.0"` (legitimacy-audited in 10-RESEARCH: crates.io, mature, github.com/rayon-rs/rayon, Approved); `crates/treelite-gtil/Cargo.toml:10` `rayon = { workspace = true }`. Sole new runtime dependency. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|

No accepted risks. (One documented latent item — see Notes — is tracked for milestone-level awareness, not accepted as a phase risk.)

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-11 | 5 | 5 | 0 | gsd-security-auditor (ASVS high) |

---

## Notes

- **WR-01 residual (latent, non-blocking):** `unsafe impl Send for SendModelRef` (treelite-py, gtil.rs:41-56) relies on the caller convention that loader-produced models hold only `Owned` columns (no `TreeBuf::Borrowed`). This precondition is documented in the SAFETY comment but not type-enforced; the sealing newtype was deferred as out-of-scope by the code-review fix pass. T-10-03's declared mitigation (read-only `&Model`, `par_chunks_mut` disjoint writes, documented `Sync` soundness) is fully present — this is a documented latent risk, not a declared-but-absent mitigation. Tracked for milestone-level awareness.
- Code review (10-REVIEW / 10-REVIEW-FIX): 0 critical, 3 warnings + 2 info, all 5 resolved. WR-02 (`T: Copy + Sync`), WR-03 (`checked_mul` allocation guards at lib.rs:696-703/1170-1177/1271-1279), and WR-01 (caller-contract doc) verified present in code.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-11

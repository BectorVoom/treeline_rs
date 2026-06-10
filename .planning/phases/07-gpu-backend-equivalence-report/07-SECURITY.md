---
phase: 07
slug: gpu-backend-equivalence-report
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-11
---

# Phase 07 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

This phase generalized the cubecl launcher to a runtime-generic `predict::<R, F>`, added
`rocm`/`cuda`/`wgpu` cargo features (default = CPU-only), per-backend device-client
constructors with a typed `DeviceUnavailable` skip, harness backend registration, and an
observational GPU equivalence/crossover report. There is **no network, auth, or
untrusted-input surface** — all compute runs over frozen, already-validated in-process
golden fixtures. The only untrusted-control-flow surface is the GPU driver FFI on device
construction. Register authored at plan time across all 4 PLANs; verified against the
implementation by `gsd-security-auditor` (ASVS-1, block_on=high).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| GPU driver FFI (HIP/CUDA/Vulkan) ← `device::client::<R>()` | Missing-device driver-init failure crosses from the C driver into the Rust process — the only untrusted-control-flow surface. | Driver init status (catchable panic → typed error) |
| host launcher → GPU device buffers (`upload_forest`) | Model columns uploaded to device buffers; host-side `validate_shape`/`validate_leaf_vectors` run before any `create_from_slice`. | Trusted in-process model tensors |
| caller → `Backend` selection | Caller names the backend explicitly; no auto-detect/"best available" resolver (D-04). | Enum selection (no external input) |
| committed report/doc files ← harness emission | `docs/GPU_EQUIVALENCE_REPORT.md` + crossover doc written by the harness from the executed path, never hand-edited (D-06/D-10). | Observational deviation/timing numbers |
| cargo dependency graph | Feature forwarding to the official `cubecl` family; no new top-level crates this phase. | Build-time dependency resolution |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-07-01 | Denial of Service | `device::*_client()` on a missing GPU device | mitigate | `catch_unwind(AssertUnwindSafe(\|\| R::client(...)))` → typed `CubeclError::DeviceUnavailable`; host never crashes. A3 spike (`tests/device_absent.rs`) confirms catchable `Err`, not FFI abort. (`device.rs:49-52`) | closed |
| T-07-02 | Tampering | Workspace default build accidentally pulling a GPU runtime | mitigate | Features forward via crate-local `[features]` only; `default=[]`; root `Cargo.toml` has zero rocm/cuda/wgpu/hip references (cpu-only). (`treelite-cubecl/Cargo.toml:14-18`, root `Cargo.toml:25`) | closed |
| T-07-03 | Spoofing / Info Disclosure | "which backend ran" provenance | mitigate | No silent CPU fallback in `predict::<R>` — `device::client::<R>(...)?` propagates `DeviceUnavailable`; selected backend == backend that ran. (`lib.rs:332`) | closed |
| T-07-04 | Tampering / DoS | OOB device read/write from malformed model on the generalized path | accept (mitigated upstream) | Host-side `validate_shape`/`validate_leaf_vectors` run before any device op in `upload_forest`; unchanged from Phase 06, not weakened by this phase. (`upload.rs:423,426` before `:432`) | closed |
| T-07-05 | Denial of Service | `cuda_case()`/`wgpu_case()` constructing a client on absent hardware | mitigate | Honors A3 finding (catchable `Err`) → skip-not-fail; `*_case()` are `#[cfg(feature)]`-gated; never auto-invokes an unavailable backend. (`harness lib.rs:274,317`) | closed |
| T-07-06 | Spoofing | selected == ran provenance in `*_case()` | mitigate | Each `*_case()` sets `backend: Backend::<X>` literally and routes only to that runtime; no fallback re-routing. (`harness lib.rs:236,276,319`) | closed |
| T-07-07 | Repudiation / Tampering | Hand-edited report/crossover numbers drifting from reality | mitigate | Artifacts harness-emitted from the executed ROCm path, regenerated on hardware (D-06/D-10); JSON sidecar enables drift check; `emit()` is the sole writer. (`report.rs:255-276`, `gtil_matrix_gpu.rs:452`) — advisory residual: WR-02/WR-03 (see log) | closed |
| T-07-08 | Info Disclosure (honesty) | A green-looking report masking real GPU divergence | mitigate | Observational, never a CI gate (D-01); records `\|delta\|` even >1e-5; carries predicted band per postprocessor so out-of-band is visible. (`report.rs:54-72,119,182`) — advisory residual: WR-03 (see log) | closed |
| T-07-09 | Denial of Service | cuda/wgpu rows aborting the report run on absent hardware | mitigate | Skip-not-fail: `Err(DeviceUnavailable)` → "not run — no device" + CONTINUE; column recorded `None` without constructing a client. (`gtil_matrix_gpu.rs:280,416-419,427-429`) | closed |
| T-07-SC | Tampering (supply chain) | cargo installs / `zenforks-cubecl-hip` typosquat | mitigate | No new top-level crates this phase (feature forwarding only); `cubecl-hip` 0.10.0 + `cubecl-hip-sys` 7.2.5321100 both from the official `registry+crates.io` source matching umbrella 0.10.0; typosquat absent from `Cargo.lock`. (`Cargo.lock:562-588`) | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-07-1 | T-07-07 (WR-02) | Sparse/scalar-fallback cell deltas are folded into the "ROCm max \|delta\|" report column (`run_cell` returns no `ran_on_gpu` flag), so a CPU-computed deviation can populate the GPU column. Degrades fidelity of an **observational** artifact only; never gates CI, never touches the CPU 1e-5 spine. Tracked as hardening follow-up before the report is relied on as authoritative. | appservice27 (via /gsd-secure-phase) | 2026-06-11 |
| AR-07-2 | T-07-08 (WR-03) | A length-mismatch `NaN` sentinel (real GPU shape defect) is guarded out by `if max_dev.is_finite()` and leaves no trace in the committed artifact (only transient `eprintln!`); the inline "is surfaced" comment overstates. Advisory honesty gap in an observational report; no ASVS-1 control weakened. | appservice27 (via /gsd-secure-phase) | 2026-06-11 |
| AR-07-3 | T-07-03/05/09 (WR-04) | Device-absence skip detection uses `err.to_string().contains("no device available")` — stringly-typed control flow over a cross-crate `thiserror` message; brittle to message changes. Robustness/maintainability defect in `#[ignore]`'d test-harness code; declared mitigations (typed `DeviceUnavailable`, literal `Backend::<X>`) are unaffected. Should accompany a T-07-01 panic-classification narrowing (WR-01/IN-03). | appservice27 (via /gsd-secure-phase) | 2026-06-11 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-11 | 10 | 10 | 0 | gsd-security-auditor (opus) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-11

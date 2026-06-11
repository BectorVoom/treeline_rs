---
phase: 09-memory-efficiency-hardening
plan: 04
subsystem: harness-infra
tags: [mem-03, jemalloc, mimalloc, global-allocator, rss-sampling, tikv-jemalloc-ctl, statm, memory-report, observational, wheel-isolation]

# Dependency graph
requires:
  - phase: 09-memory-efficiency-hardening
    plan: 01
    provides: "harness non-default mutually-exclusive jemalloc/mimalloc features + 3 optional allocator deps + the memory_report bin skeleton (3 cfg #[global_allocator] arms + both-on compile_error)"
  - phase: 09-memory-efficiency-hardening
    plan: 03
    provides: "MEM-01 le_bytes_of recast closed; serializer + Model in their final v1 memory-hardened shape (byte-identical wire, no struct-size cost)"
  - phase: 07-gpu-backends
    provides: "report.rs render_markdown/emit + Manifest provenance precedent (the GPU_EQUIVALENCE_REPORT.md observational-report shape)"
provides:
  - "crates/treelite-harness/src/memory.rs: sample_rss (jemalloc epoch-then-read / statm) + render_markdown/emit observational report writer (MemRow/SizeOfRow/MemSample)"
  - "fleshed-out memory_report bin: loads a benchmark set, drives a predict pass, samples RSS/allocated under the active allocator, merges per-allocator blocks into docs/MEMORY_REPORT.md"
  - "committed observational docs/MEMORY_REPORT.md with system + jemalloc + mimalloc rows, a size_of::<Model>() row, and the 1e-5/golden attestation"
  - "tikv-jemallocator stats + tikv-jemalloc-ctl stats/use_std features enabled so the ctl readers return live numbers"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Observational memory report (no CI threshold) modeled on the Phase-7 report.rs render_markdown/emit + Manifest-provenance header + NOT-a-CI-gate banner"
    - "Allocator-abstracted RSS sampler: jemalloc rows via tikv_jemalloc_ctl epoch::advance()-then-stats::{allocated,resident}::read() (Pitfall 5), mimalloc/system rows via /proc/self/statm field 2 x page size; jemalloc-ctl calls cfg-gated behind feature=jemalloc so the other builds compile without the crate"
    - "Per-allocator block merge: the bin re-parses the prior committed report and replaces only the current allocator's rows, so running once under each allocator assembles the before/after narrative in one committed file"

key-files:
  created:
    - crates/treelite-harness/src/memory.rs
    - docs/MEMORY_REPORT.md
  modified:
    - crates/treelite-harness/src/bin/memory_report.rs
    - crates/treelite-harness/src/lib.rs
    - Cargo.toml

key-decisions:
  - "Benchmark set (Claude's discretion per D-06/D-10): XGBoost-JSON via load_xgboost_json + frozen v5 binary/large_margin/lgbm_numerical via treelite_core::deserialize. The bin is a binary target and CANNOT use the dev-only treelite-lightgbm/treelite-sklearn crates, so a v5-serialized LightGBM-derived model (lgbm_numerical.model.bin) supplies the LightGBM coverage without a new dependency."
  - "tikv-jemallocator gains the `stats` feature and tikv-jemalloc-ctl gains `stats`+`use_std` (both off by default in 0.7.0) — without them the jemalloc build fails (stats module gated) and the readers return zero. Rule 3 blocking fix; allocator deps stay optional+harness-only (D-08 preserved)."
  - "page_size_bytes() returns the conventional 4 KiB rather than taking a libc dep on the harness (libc is not in the graph) — documented as a single override seam for a divergent-page-size host."
  - "Report MERGES per-allocator blocks (re-parse prior file, replace only the current allocator's rows) so one committed docs/MEMORY_REPORT.md shows system vs jemalloc vs mimalloc after three runs. Parse is best-effort: an unreadable prior report degrades to just-new-rows, never fails (observational, D-10)."

requirements-completed: [MEM-03]

# Metrics
duration: 7min
completed: 2026-06-11
---

# Phase 9 Plan 04: MEM-03 Allocator RSS Report + docs/MEMORY_REPORT.md Summary

**Wired jemalloc and mimalloc as runtime-selectable `#[global_allocator]`s in the `memory_report` bin ONLY (D-07/D-08), fleshed out an allocator-abstracted RSS/allocated sampler + observational report writer (`memory.rs`, modeled on the Phase-7 `report.rs`), and generated a committed `docs/MEMORY_REPORT.md` showing the before/after (system 9.71 MiB / jemalloc 5.43 MiB / mimalloc 10.40 MiB) RSS narrative with a `size_of::<Model>()` row and the 1e-5/golden attestation. Both allocators build+run on the Linux/ROCm box (D-09); the abi3 wheel stays allocator-free (D-08); both hard invariants green. MEM-03 closed, Phase 9 complete.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-06-11T03:07:25Z
- **Completed:** 2026-06-11
- **Tasks:** 3
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- **Task 1 — `memory.rs` sampler + report writer:** Created `crates/treelite-harness/src/memory.rs` mirroring `report.rs`'s structure: `MemRow`/`SizeOfRow`/`MemSample` types, `sample_rss(SampleMethod)` abstracted over allocator (jemalloc via `tikv_jemalloc_ctl::{epoch, stats}` — `epoch::advance()` BEFORE each `stats::{allocated,resident}::read()` per Pitfall 5; mimalloc/system via `/proc/self/statm` field 2 × page size), `render_markdown` with Manifest provenance + the verbatim "Observational — NOT a CI gate" banner + the MEM-03 columns + a `size_of::<Model>()` row + the 1e-5/golden attestation, and `emit` (`create_dir_all` then `write`). jemalloc-ctl calls cfg-gated behind `feature = "jemalloc"`. Exported `pub mod memory` from `lib.rs`. Builds under default, `--features jemalloc`, AND `--features mimalloc`; no `#[global_allocator]` in any pub-fn crate. 2 module unit tests green.
- **Task 2 — fleshed `memory_report` bin + committed report:** `main()` loads the benchmark set (XGBoost-JSON + 3 frozen v5 models), drives a 256-row predict pass per model to exercise allocation, samples RSS/allocated under the active allocator, and merges the per-allocator rows into `docs/MEMORY_REPORT.md`. Ran once under each allocator on Linux (D-09): system (9.71 MiB RSS), jemalloc (5.43 MiB RSS, 600.27 KiB allocated via jemalloc-ctl), mimalloc (10.40 MiB RSS). The committed report carries all three blocks + the size_of row + the attestation; both-on stays a `compile_error`.
- **Task 3 — final invariant + wheel-isolation gate:** `cargo tree -p treelite-py | grep -E "jemalloc|mimalloc"` prints nothing (D-08 — allocator deps reachable from the harness-with-features but ZERO from the wheel). golden_v5 both fixtures byte-identical (HARD INVARIANT 1); harness 1e-5 equivalence/matrix all green; `cargo test --workspace` 0 failures; `model_invariants` (248 B ≤ 512, !Send) green; `uv run pytest crates/treelite-py` 39 passed / 1 skipped (HARD INVARIANT 2). MEM-03 closed; Phase 9 complete.

## Task Commits

Each task was committed atomically:

1. **Task 1: RSS/allocated sampler + report writer (memory.rs)** — `f9c1b3b` (feat)
2. **Task 2: fill memory_report bin + generate committed docs/MEMORY_REPORT.md** — `fb63a63` (feat)
3. **Task 3: final invariant + wheel-isolation gate** — gate task, no production edit (the committed `docs/MEMORY_REPORT.md` from Task 2 is the deliverable); all gates green against `fb63a63`, no commit of its own.

## Files Created/Modified

- `crates/treelite-harness/src/memory.rs` (new) — `SampleMethod`/`MemSample`/`MemRow`/`SizeOfRow` + `sample_rss` (jemalloc epoch-then-read / statm) + `render_markdown`/`emit`. jemalloc path cfg-gated; no `#[global_allocator]`.
- `crates/treelite-harness/src/bin/memory_report.rs` — fleshed from the Plan-01 skeleton: benchmark load → drive predict → sample → merge → emit. The 3 cfg `#[global_allocator]` arms + both-on `compile_error` are unchanged.
- `crates/treelite-harness/src/lib.rs` — added `pub mod memory;`.
- `Cargo.toml` — enabled `stats` on `tikv-jemallocator` and `stats`+`use_std` on `tikv-jemalloc-ctl` (both off by default in 0.7.0).
- `docs/MEMORY_REPORT.md` (new) — committed observational report: system + jemalloc + mimalloc blocks, `size_of::<Model>()` = 248 B row, 1e-5/golden attestation, wheel-isolation note.

## Decisions Made

- **Benchmark set is one XGBoost-JSON + three frozen v5 models.** The `memory_report` bin is a `[[bin]]` target, so it can only use the harness's non-dev dependencies (`treelite-core`, `-xgboost`, `-gtil`, `-cubecl`). `treelite-lightgbm`/`treelite-sklearn` are `dev-dependencies` and are unavailable to the bin. The frozen v5 `lgbm_numerical.model.bin` (loaded via `treelite_core::deserialize`) supplies the LightGBM coverage without adding a dependency or breaking the wheel-isolation invariant.
- **jemalloc/jemalloc-ctl `stats` features enabled (Rule 3 blocking).** With the default features, `cargo build -p treelite-harness --features jemalloc` failed (`tikv_jemalloc_ctl::stats` is gated behind the crate's own `stats` feature, E0432) and the readers would return zero. Enabling `stats` on `tikv-jemallocator` (builds jemalloc with statistics collection) and `stats`+`use_std` on `tikv-jemalloc-ctl` (exposes the readers + the std error path for `?`) is the minimal fix; the deps remain `optional = true` on the harness only, so D-08 wheel isolation is preserved (verified in Task 3).
- **Observational, no RSS threshold (D-10).** The report records; it never asserts an RSS bound. The Phase-9 gate is the 1e-5 equivalence harness + the byte-identical golden compare (D-11), which are allocator-independent. This mirrors the Phase-7 `GPU_EQUIVALENCE_REPORT.md` precedent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Enabled `stats` features on the jemalloc crates**
- **Found during:** Task 1 (`cargo build -p treelite-harness --features jemalloc`)
- **Issue:** the jemalloc build failed with `E0432: unresolved import tikv_jemalloc_ctl::stats` — the `stats` module is gated behind `tikv-jemalloc-ctl`'s own (default-off) `stats` feature, and the `?`/`anyhow` error path needs `use_std`. Additionally, `tikv-jemallocator` must build jemalloc with `stats` for `resident`/`allocated` to be non-zero.
- **Fix:** in the workspace `Cargo.toml`, set `tikv-jemallocator = { version = "0.7.0", features = ["stats"] }` and `tikv-jemalloc-ctl = { version = "0.7.0", features = ["stats", "use_std"] }`. Deps stay `optional = true` on the harness only — D-08 wheel isolation unchanged (re-verified in Task 3: 0 allocator nodes in `treelite-py`).
- **Files modified:** `Cargo.toml`
- **Commit:** `f9c1b3b`

**2. [Rule 1 - Bug] Removed an unnecessary `unsafe` block in `page_size_bytes`**
- **Found during:** Task 1 (first default build emitted `warning: unnecessary unsafe block`)
- **Issue:** the initial `page_size_bytes` wrapped a non-FFI shim in `unsafe`, producing an `unused_unsafe` warning (the harness takes no `libc` dep, so there is no actual FFI call).
- **Fix:** simplified `page_size_bytes()` to return the conventional 4 KiB page directly, documented as a single override seam for a divergent-page-size host. No behavior change (4096 is the page size on the x86-64 target).
- **Files modified:** `crates/treelite-harness/src/memory.rs`
- **Commit:** `f9c1b3b`

## Issues Encountered

None blocking. One pre-existing environment note (carried from 09-02/09-03): `uv run maturin develop` warns it cannot set rpath (`patchelf` not installed) but still builds + installs the abi3 wheel successfully and pytest passes — not introduced by this plan. The report header renders "unknown rustc" because `RUSTC_VERSION_AT_BUILD` is not set in this build env (an `option_env!` provenance field, same behavior the `manifest.rs` module documents) — acceptable for an observational artifact.

## Known Stubs

None. The Plan-01 `memory_report` skeleton is fully fleshed out: `main()` loads a real benchmark set, drives prediction, samples RSS/allocated, and writes the committed report. No placeholder/empty-value/TODO surface introduced.

## Threat Flags

None. No new network/auth surface. The `*-sys` allocator crates do a local `cc` build with no network postinstall (T-09-14, pinned/Approved sources). The allocator deps are confined to the harness (T-09-13 mitigated — verified by the `cargo tree -p treelite-py` empty grep in Task 3); the both-on misconfiguration is a `compile_error` (T-09-15 mitigated); the committed `MEMORY_REPORT.md` contains only os/arch/rustc provenance + RSS numbers, no secrets (T-09-16 N/A). File reads are restricted to known fixture paths + `/proc/self/statm` (the process's own RSS).

## Next Phase Readiness

- MEM-03 closed: jemalloc + mimalloc runtime-selectable as `#[global_allocator]` in the `memory_report` bin only (D-07/D-08), both build+run on Linux (D-09); committed observational `docs/MEMORY_REPORT.md` with RSS/allocated rows + `size_of::<Model>()` + the NOT-a-CI-gate banner (D-10); the abi3 wheel has ZERO allocator deps (D-08).
- HARD INVARIANTS held: `golden_v5.bin` AND `golden_v5_3format.bin` byte-identical, harness 1e-5 equivalence/matrix green, `cargo test --workspace` 0 failures, `model_invariants` 248 B / !Send, `uv run pytest` 39 passed / 1 skipped.
- All three MEM requirements (MEM-01/02/03) are now closed; Phase 9 (memory-efficiency hardening) is complete. No deferred items, no blockers from this plan.

## Self-Check: PASSED

- `crates/treelite-harness/src/memory.rs` exists; `docs/MEMORY_REPORT.md` exists with the Observational banner + 3 allocator blocks + size_of row.
- Commits `f9c1b3b` and `fb63a63` present in git history.
- golden_v5 (both fixtures byte-identical), harness 1e-5 (all green), workspace (0 failures), model_invariants (248 ≤ 512, !Send), pytest (39/1) all green; `cargo tree -p treelite-py` has 0 allocator nodes.

---
*Phase: 09-memory-efficiency-hardening*
*Completed: 2026-06-11*

---
phase: 06-cubecl-gtil-kernels-cpu-backend
plan: 01
subsystem: cubecl-kernels
tags: [cubecl, bytemuck, scaffold, tdd, zero-copy, wave-0]
requires:
  - "treelite-core::TreeBuf (T: Copy enum, as_slice)"
  - "treelite-gtil::{Config, predict} (host-launcher shape + fallback)"
  - "treelite-harness RunnerCase seam (05-05; CubeclCpu variant reserved)"
provides:
  - "crate treelite-cubecl (workspace member) building with cubecl 0.10.0 (cpu) + bytemuck"
  - "treelite_cubecl::CubeclError thiserror enum"
  - "treelite_cubecl::predict_cpu host-launcher stub (Unsupported until Wave 3)"
  - "treelite_core::TreeBuf::<T: Pod>::as_bytes() additive zero-copy byte view (GPU-05 enabler)"
  - "4 RED #[ignore] test scaffolds: spike, upload, determinism, gtil_matrix_cubecl sibling"
  - "cubecl 0.10.0 API names pinned: upload=create_from_slice, exp2=Float::exp2 (retires A1/A3)"
affects:
  - "Cargo.toml [workspace.members]/[workspace.dependencies]"
  - "crates/treelite-core/Cargo.toml (bytemuck dep)"
tech-stack:
  added:
    - "cubecl 0.10.0 (features=[\"cpu\"])"
    - "bytemuck 1 (features=[\"derive\"])"
  patterns:
    - "Additive narrower-bound impl block (T: Copy + Pod) — primary T: Copy API untouched"
    - "Nyquist RED #[ignore] scaffolds with MISSING reason strings"
key-files:
  created:
    - crates/treelite-cubecl/Cargo.toml
    - crates/treelite-cubecl/src/lib.rs
    - crates/treelite-cubecl/src/error.rs
    - crates/treelite-cubecl/src/upload.rs
    - crates/treelite-cubecl/src/kernels.rs
    - crates/treelite-cubecl/tests/spike.rs
    - crates/treelite-cubecl/tests/upload.rs
    - crates/treelite-cubecl/tests/determinism.rs
    - crates/treelite-harness/tests/gtil_matrix_cubecl.rs
  modified:
    - Cargo.toml
    - crates/treelite-core/Cargo.toml
    - crates/treelite-core/src/tree_buf.rs
decisions:
  - "cubecl 0.10.0 dependency tree resolves + compiles (34.9s clean) — the strongest legitimacy proof per RESEARCH; the slopcheck [SLOP] verdicts are confirmed wrong-registry (PyPI) false positives, packages APPROVED"
  - "cubecl 0.10.0 API name-pinned at source: upload=ComputeClient::create_from_slice(&[u8]) (cubecl-runtime client.rs:287; create(Bytes) at :345 is the owned variant), exp2=Float::exp2(self) direct (cubecl-core typemap.rs:680, no exp(x*ln2)/powf identity needed) — retires RESEARCH A1/A3 before any kernel is authored"
  - "as_bytes() is a NARROWER second impl<T: Copy + bytemuck::Pod> block; the enum's primary T: Copy bound is NOT widened (the broad Pod seam stays Phase 9, tree_buf.rs:11-12)"
  - "predict_cpu<F: Copy> stub returns CubeclError::Unsupported — F bound minimal so the crate compiles without pulling any cubecl symbol into the stub body (kernel surface referenced only in Wave 3)"
metrics:
  duration: ~6min
  completed: 2026-06-10
  tasks: 2
  files: 12
---

# Phase 6 Plan 01: cubecl Kernel Crate Scaffold + Wave 0 Foundation Summary

Scaffolded the `treelite-cubecl` crate with cubecl 0.10.0 (cpu) + bytemuck pinned, added the additive zero-copy `TreeBuf::as_bytes()` accessor (GPU-05/SC3 enabler), and created the four RED Nyquist test scaffolds (spike, upload, determinism, `gtil_matrix_cubecl` sibling) that Waves 1-4 turn green — every contract the kernel work plugs into now exists before any kernel does.

## What Was Built

**Task 1 — crate scaffold + dependency pinning (commit a0ec287):**
- Registered `crates/treelite-cubecl` in `[workspace.members]`.
- Pinned `cubecl = { version = "0.10.0", features = ["cpu"] }` and `bytemuck = { version = "1", features = ["derive"] }` in `[workspace.dependencies]`; added `bytemuck` to `treelite-core`.
- `CubeclError` `thiserror` enum (`InvalidInputShape`, `FeatureIndexOutOfBounds`, `NodeIndexOutOfBounds`, `Unsupported` catch-all) mirroring the `GtilError` discipline — no `anyhow`, no `panic!`.
- `predict_cpu<F: Copy>(&Model, &[F], usize, &Config) -> Result<Vec<F>, CubeclError>` stub returning `Unsupported` (filled in Wave 3), plus placeholder `upload`/`kernels` modules.
- Forced the cubecl dependency tree to resolve via `cargo build -p treelite-cubecl` (compiled clean in 34.9s — the spike legitimacy gate). Confirmed and recorded the cubecl 0.10.0 API names in `upload.rs`.

**Task 2 — `TreeBuf::as_bytes()` + RED scaffolds (commit 0625d8e):**
- Added a second `impl<T: Copy + bytemuck::Pod> TreeBuf<T>` block exposing `as_bytes(&self) -> &[u8]` via `bytemuck::cast_slice` (validated alignment/size, never a `transmute`, T-06-02). Primary `T: Copy` API and the enum bound untouched.
- 3 unit tests: f32 round-trip (len 8 + exact cast-back), i32 (len 4), empty.
- Four RED `#[ignore]` scaffolds with MISSING reason strings: `spike.rs` (Wave 1), `upload.rs` (Wave 2), `determinism.rs` (Wave 4), `crates/treelite-harness/tests/gtil_matrix_cubecl.rs` (Wave 4 sibling). The sibling is a thin new file (D-11), recording in a comment that it must use its OWN per-cell provenance, never copying `gtil_matrix.rs`'s `"scalar-cpu"` backend literal.

## Verification

- `cargo build -p treelite-cubecl` — clean (cubecl 0.10.0 + bytemuck resolve and compile).
- `cargo build --workspace` — clean (new member, no regression).
- `cargo test -p treelite-core` `as_bytes` tests — 3 passed.
- `cargo test --workspace --no-run` — all test targets compile, including the four RED scaffolds.
- `cargo test --workspace` — fully green; the four scaffolds report `ignored` with their MISSING reason strings; no existing test (including the untouched `gtil_matrix.rs`) regressed.
- Greps: `treelite-cubecl` member + `cubecl = ... features=["cpu"]` present; `anyhow` count 0 in `error.rs`; API pin line present; `as_bytes` present; `scalar-cpu` in the sibling appears only in a comment.

## Deviations from Plan

None — plan executed exactly as written. The `anyhow`-count-0 grep initially matched the word "anyhow" inside a doc comment ("never `anyhow`"); reworded the comment to "no error-aggregation dependency" so the literal-token grep gate reads 0 cleanly. This is a wording adjustment to satisfy the acceptance grep, not a behavioral change.

## Known Stubs

Intentional, plan-scoped, and tracked by the RED scaffolds:
- `predict_cpu` returns `CubeclError::Unsupported` — kernel body lands in Wave 3 (plan 06-04).
- `upload.rs` / `kernels.rs` are doc-only placeholder modules — Waves 2-3 (plans 06-03/06-04).
- The four `#[ignore]` test bodies are `todo!()` Nyquist markers — turned green by Waves 1-4 (plans 06-02..06-05).

All are documented in the plan's wave layout; none block this plan's goal (a compiling workspace with the crate registered, `as_bytes()`, and the RED scaffolds).

## Self-Check: PASSED

- crates/treelite-cubecl/{Cargo.toml,src/lib.rs,src/error.rs,src/upload.rs,src/kernels.rs} — FOUND
- crates/treelite-cubecl/tests/{spike,upload,determinism}.rs — FOUND
- crates/treelite-harness/tests/gtil_matrix_cubecl.rs — FOUND
- crates/treelite-core/src/tree_buf.rs (as_bytes) — FOUND
- commit a0ec287 — FOUND
- commit 0625d8e — FOUND

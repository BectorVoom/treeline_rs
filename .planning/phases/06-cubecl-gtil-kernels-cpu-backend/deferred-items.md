# Phase 06 — Deferred Items

Out-of-scope discoveries logged during execution (not fixed in the discovering plan
per the executor scope boundary).

## From plan 06-03 (Wave 3)

- **`cargo clippy` fails on `crates/treelite-cubecl/src/kernels/traversal.rs`
  (`clippy::eq_op` deny on `fv != fv`).** The `fv != fv` self-inequality NaN test
  was authored by plan 06-02 (commit `09ffe4c`) and is intentional (it is the
  cube-frontend NaN check; `F::is_nan` returns `WithScalar<bool>`, not `bool`).
  Clippy's `eq_op` lint is `deny`-by-default, so `cargo clippy -p treelite-cubecl`
  fails to compile the lib — even though `cargo test --workspace` is fully green.
  - **Scope:** `traversal.rs` is NOT in plan 06-03's `files_modified`; the lint
    predates this plan. Plan 06-03's own files (`upload.rs`, `postproc.rs`, their
    tests) are clippy-clean (verified with `RUSTFLAGS="-A clippy::eq_op"`).
  - **Suggested fix (future wave that touches traversal.rs, e.g. 06-04):** add a
    scoped `#[allow(clippy::eq_op)]` on the `if fv != fv {` line with a comment
    pointing at the 06-02 NaN-test decision, or use `F`'s native NaN intrinsic if
    Wave 3 finds a working associated-fn form. One line, no behavior change.
  - **RESOLVED (plan 06-04, commit `2fdff21`):** plan 06-04 generalized `descend`
    (the same `traversal.rs` it was already touching for the `<F, T>` Pitfall-6
    change), so the scoped `#[allow(clippy::eq_op)]` was added on the `if fv != fv`
    line with the 06-02 NaN-test comment. `cargo clippy -p treelite-cubecl` is now
    clean (no warnings/errors). No behavior change.
  - **Also pre-existing:** `tests/spike.rs` emits a `clippy::type_complexity`
    warning on `soa_columns`'s 7-tuple return (06-02). Cosmetic.

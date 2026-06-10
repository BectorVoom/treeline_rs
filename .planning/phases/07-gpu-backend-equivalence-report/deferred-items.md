# Phase 07 — Deferred Items

Out-of-scope discoveries logged during execution (not fixed; not caused by the current task).

| Discovered In | Item | File | Status |
|---------------|------|------|--------|
| 07-01 Task 2 | Pre-existing clippy `-D warnings` failure under `--tests`: "very complex type used. Consider factoring parts into `type` definitions" | `crates/treelite-cubecl/tests/spike.rs:228` | Deferred — committed in Phase 06 (`2fdff21`), unmodified by this plan. The plan's own verify command (`cargo clippy -p treelite-cubecl -- -D warnings`, without `--tests`) is clean; this only trips when test targets are clippy-linted. Out of scope per the executor SCOPE BOUNDARY (pre-existing, unrelated file). |

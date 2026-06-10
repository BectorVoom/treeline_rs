# treelite_rs

PyO3 Python binding for [treelite-rs](../../README.md) — a from-scratch Rust
rewrite of [Treelite](https://github.com/dmlc/treelite).

This is the sole external language binding for the project (no C-API). It exposes
load → predict → serialize directly over the Rust core, with predictions
validated to match upstream Treelite within `1e-5`.

Build into the active environment:

```bash
maturin develop
```

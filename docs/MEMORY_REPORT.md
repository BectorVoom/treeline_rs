# Memory Efficiency Report (MEM-03)

Regenerated on: treelite 4.7.0 / unknown rustc / captured-on linux x86_64 (provenance from the run Manifest, D-10)
Reference floor: the `1e-5` equivalence harness + the byte-identical `golden_v5` compare (D-11). **Observational — NOT a CI gate (D-10).**

| Model | Allocator | peak resident (RSS) | bytes allocated | measurement method |
|-------|-----------|---------------------|-----------------|--------------------|
| binary (v5) | jemalloc (tikv-jemallocator) | 5689344 (5.43 MiB) | 614680 (600.27 KiB) | tikv_jemalloc_ctl epoch+stats (resident/allocated) |
| large_margin (v5) | jemalloc (tikv-jemallocator) | 5689344 (5.43 MiB) | 614680 (600.27 KiB) | tikv_jemalloc_ctl epoch+stats (resident/allocated) |
| lgbm_numerical (v5, LightGBM-derived) | jemalloc (tikv-jemallocator) | 5689344 (5.43 MiB) | 614680 (600.27 KiB) | tikv_jemalloc_ctl epoch+stats (resident/allocated) |
| xgb_3format (XGBoost JSON) | jemalloc (tikv-jemallocator) | 5689344 (5.43 MiB) | 614680 (600.27 KiB) | tikv_jemalloc_ctl epoch+stats (resident/allocated) |
| binary (v5) | mimalloc | 10903552 (10.40 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| large_margin (v5) | mimalloc | 10903552 (10.40 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| lgbm_numerical (v5, LightGBM-derived) | mimalloc | 10903552 (10.40 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| xgb_3format (XGBoost JSON) | mimalloc | 10903552 (10.40 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| binary (v5) | system (default malloc) | 10178560 (9.71 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| large_margin (v5) | system (default malloc) | 10178560 (9.71 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| lgbm_numerical (v5, LightGBM-derived) | system (default malloc) | 10178560 (9.71 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |
| xgb_3format (XGBoost JSON) | system (default malloc) | 10178560 (9.71 MiB) | n/a (statm path) | /proc/self/statm field 2 × page size (RSS only) |

## `size_of::<Model>()` (MEM-02 structural-size attestation)

| Metric | Bytes | Budget (Wave-0 Pitfall-2 guard) | Within budget? |
|--------|-------|---------------------------------|----------------|
| `size_of::<Model>()` | 248 | 512 | yes |

> MEM-02 migrated the `Model`/`Metadata` metadata fields to `SmallVec`/`CompactString` with the inline `N` chosen for the dominant shape — `size_of::<Model>()` stays byte-identical to the prior `Vec`/`String` layout (zero struct-size cost). MEM-01 routed the serializer's `le_bytes_of` through `bytemuck::cast_slice`, removing an `unsafe` block while emitting byte-identical bytes.

## Attestation (D-11 green floor)

The two hard invariants are green for this Phase-9 state: the v5 serializer emits byte-identical `golden_v5.bin` / `golden_v5_3format.bin` (HARD INVARIANT 1), and the full equivalence/matrix harness is within `1e-5` with `cargo test --workspace` + `uv run pytest crates/treelite-py` passing (HARD INVARIANT 2). The allocator choice above changes only the RSS narrative, never the predictions — the `1e-5`/golden floor is the real gate (D-11).

## Wheel isolation (D-08)

The allocator crates (`tikv-jemallocator`, `tikv-jemalloc-ctl`, `mimalloc`) are `optional = true` on the `treelite-harness` crate ONLY, behind the non-default mutually-exclusive `jemalloc`/`mimalloc` features; the `#[global_allocator]` static lives ONLY in this `memory_report` bin. `cargo tree -p treelite-py | grep -E "jemalloc|mimalloc"` prints nothing — the abi3 wheel pulls ZERO allocator deps.

---
status: complete
phase: 10-parallel-scalar-inference
source: [10-VERIFICATION.md]
started: 2026-06-11
updated: 2026-06-11
---

## Current Test

[testing complete]

## Tests

### 1. Scalar-path wall-clock throughput speedup
expected: |
  Measure wall-clock predict() time on a large-row-batch LightGBM/categorical
  model at nthread=1 vs nthread=-1 (all cores). Expect a meaningful speedup
  (scaling with core count, allowing for Amdahl overhead) with no change in
  predicted values beyond 1e-5. This is the performance goal of the phase —
  correctness of parallelization is already fully automated/green.
result: pass
measured: |
  Categorical LightGBM model, 4,000,000 rows, 16 cores available.
  nthread=1 (serial):    0.708s  (5.6 Mrow/s)
  nthread=0 (all cores): 0.192s  (20.8 Mrow/s)
  SPEEDUP = 3.68x; output bit-identical (max |serial-parallel| = 0.000e0, within 1e-5).
  Sub-linear vs 16 cores is expected: the model is tiny (10 trees, 3 features),
  so per-row traversal is memory-bandwidth-bound. Clear, meaningful speedup on
  the previously single-core scalar path with zero numerical drift.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

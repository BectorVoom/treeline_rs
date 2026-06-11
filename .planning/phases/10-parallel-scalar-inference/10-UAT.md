---
status: testing
phase: 10-parallel-scalar-inference
source: [10-VERIFICATION.md]
started: 2026-06-11
updated: 2026-06-11
---

## Current Test

number: 1
name: Scalar-path wall-clock throughput speedup
expected: |
  On a LightGBM (kLE) or categorical model with a large row batch, predict()
  through the scalar GTIL path runs measurably faster with multiple cores than
  with nthread=1, while producing output identical to the serial path within
  1e-5. Parallelism is structurally proven active (parallel_uses_more_than_one_core
  passes); this item confirms the throughput *magnitude* meets expectations.
awaiting: user response

## Tests

### 1. Scalar-path wall-clock throughput speedup
expected: |
  Measure wall-clock predict() time on a large-row-batch LightGBM/categorical
  model at nthread=1 vs nthread=-1 (all cores). Expect a meaningful speedup
  (scaling with core count, allowing for Amdahl overhead) with no change in
  predicted values beyond 1e-5. This is the performance goal of the phase —
  correctness of parallelization is already fully automated/green.
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps

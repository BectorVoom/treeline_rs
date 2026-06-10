# CPU/GPU Crossover (SC3, documented-only)

Measured wall-clock per dense `predict` call (median of 5 reps, incl. upload + launch + readback) on the developer's AMD/ROCm hardware. CPU baseline = `cubecl_cpu_case()` (same kernels on `CpuRuntime`); GPU = `rocm_case()` (`predict::<HipRuntime, _>`).

**DOCUMENTED-ONLY (D-09): `predict()` does NOT auto-route.** CPU stays the DEFAULT backend; the engine runs exactly the backend it is told (explicit-selection, D-04). This crossover informs a caller's explicit choice (the Phase-8 PyO3 consumer) — it is never a routing rule baked into the engine.

## Forest `binary` (4 features)

| num_row | rows×features | CPU (ns) | ROCm (ns) | ROCm faster? |
|---------|---------------|----------|-----------|--------------|
| 1 | 4 | 439094 | 436188 | yes |
| 10 | 40 | 505469 | 452629 | yes |
| 100 | 400 | 571382 | 457288 | yes |
| 1000 | 4000 | 467187 | 394190 | yes |
| 10000 | 40000 | 822453 | 511780 | yes |
| 100000 | 400000 | 3764754 | 2730714 | yes |

## Forest `leaf_vec_mc` (5 features)

| num_row | rows×features | CPU (ns) | ROCm (ns) | ROCm faster? |
|---------|---------------|----------|-----------|--------------|
| 1 | 5 | 508965 | 468689 | yes |
| 10 | 50 | 473288 | 507903 | no |
| 100 | 500 | 462087 | 535505 | no |
| 1000 | 5000 | 727645 | 507983 | yes |
| 10000 | 50000 | 1918560 | 1678510 | yes |
| 100000 | 500000 | 22225954 | 11984708 | yes |

## Empirical crossover + dominant metric

- On `binary` (4 features), ROCm wall-clock first beats `cubecl_cpu_case` at ~1 rows.
- On `leaf_vec_mc` (5 features), ROCm wall-clock first beats `cubecl_cpu_case` at ~1 rows.

**Dominant metric (let the data decide, D-10):** the GPU pays a fixed upload+launch+readback cost per call, amortized over rows — so the crossover keys on **row count** (or `rows × features` for the input transfer when the forest is small). Compare the two forests' crossover rows above: if they crossover at a similar `rows × features` product rather than a similar row count, the input transfer dominates; if they crossover at a similar row count, the per-row traversal dominates. No formula is pre-committed (D-10) — the table is the evidence.

Consumer: the Phase-8 PyO3 caller (D-09) is the documented consumer of this heuristic. CPU remains the default backend.

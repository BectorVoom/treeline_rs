# CPU/GPU Crossover (SC3, documented-only)

Measured wall-clock per dense `predict` call (median of 5 reps, incl. upload + launch + readback) on the developer's AMD/ROCm hardware. CPU baseline = `cubecl_cpu_case()` (same kernels on `CpuRuntime`); GPU = `rocm_case()` (`predict::<HipRuntime, _>`).

**DOCUMENTED-ONLY (D-09): `predict()` does NOT auto-route.** CPU stays the DEFAULT backend; the engine runs exactly the backend it is told (explicit-selection, D-04). This crossover informs a caller's explicit choice (the Phase-8 PyO3 consumer) — it is never a routing rule baked into the engine.

## Forest `binary` (4 features)

| num_row | rows×features | CPU (ns) | ROCm (ns) | ROCm faster? |
|---------|---------------|----------|-----------|--------------|
| 1 | 4 | 426708 | 733111 | no |
| 10 | 40 | 455772 | 615189 | no |
| 100 | 400 | 471301 | 556721 | no |
| 1000 | 4000 | 474176 | 488403 | no |
| 10000 | 40000 | 823308 | 880235 | no |
| 100000 | 400000 | 3710805 | 3148655 | yes |

## Forest `leaf_vec_mc` (5 features)

| num_row | rows×features | CPU (ns) | ROCm (ns) | ROCm faster? |
|---------|---------------|----------|-----------|--------------|
| 1 | 5 | 484415 | 604369 | no |
| 10 | 50 | 474998 | 630609 | no |
| 100 | 500 | 495095 | 677757 | no |
| 1000 | 5000 | 611253 | 838127 | no |
| 10000 | 50000 | 1926741 | 2037809 | no |
| 100000 | 500000 | 20642951 | 18939317 | yes |

## Empirical crossover + dominant metric

- On `binary` (4 features), ROCm wall-clock first beats `cubecl_cpu_case` at ~100000 rows.
- On `leaf_vec_mc` (5 features), ROCm wall-clock first beats `cubecl_cpu_case` at ~100000 rows.

**Dominant metric (let the data decide, D-10):** the GPU pays a fixed upload+launch+readback cost per call, amortized over rows — so the crossover keys on **row count** (or `rows × features` for the input transfer when the forest is small). Compare the two forests' crossover rows above: if they crossover at a similar `rows × features` product rather than a similar row count, the input transfer dominates; if they crossover at a similar row count, the per-row traversal dominates. No formula is pre-committed (D-10) — the table is the evidence.

Consumer: the Phase-8 PyO3 caller (D-09) is the documented consumer of this heuristic. CPU remains the default backend.

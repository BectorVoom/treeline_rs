# GPU Equivalence Report

Regenerated on: AMD ROCm device (set TREELITE_GPU_DEVICE to label) / ROCm n/a / rustc 1.95.0 (59807616e 2026-04-14) / captured-on Linux-6.17.0-35-generic-x86_64-with-glibc2.39 x86_64 (provenance from the run manifest, D-06)
Reference: f64 scalar GTIL (the 1e-5 CPU spine). **Observational — NOT a CI gate (D-01).**

| Model class | Postprocessor | ROCm max \|delta\| | f64 fallback used? | CUDA | wgpu | Predicted band (D-03) |
|-------------|---------------|--------------------|--------------------|------|------|-----------------------|
| binary | sigmoid | 2.2268493049537597e-7 | yes | not run — no device | not run — no device | ~1e-6..5e-6 |
| large_margin | sigmoid | 2.907300050480899e-6 | yes | not run — no device | not run — no device | ~1e-6..5e-6 |
| leaf_vec_mc | softmax | 2.3096799850463867e-7 | yes | not run — no device | not run — no device | ~1e-6..5e-6 |
| lgbm_numerical | identity | 2.8322729006546865e-7 | yes | not run — no device | not run — no device | ~0e0..1e-6 |
| mixedwidth | identity | 0e0 | yes | not run — no device | not run — no device | ~0e0..1e-6 |

Determinism: observed run-to-run stable on AMD ROCm device (set TREELITE_GPU_DEVICE to label); bit-identity NOT guaranteed on GPU (per the OpenCL spec — transcendental rounding and float-reduction order are implementation-defined). The Phase-6 SC2 bit-identical claim is a CPU-backend property.

A measured ROCm |delta| materially above its predicted band is itself a finding worth recording (e.g. a `native_exp` transcendental mapping, Pitfall 2) — not a CI failure (D-01).

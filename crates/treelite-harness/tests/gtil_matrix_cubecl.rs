//! RED scaffold — Wave 4 cubecl-backend matrix sibling (plan 06-05).
//!
//! This is a SIBLING of `gtil_matrix.rs`, NOT a refactor of it (D-11 smell
//! guard): a thin new file that, once Wave 4 lands, registers
//! `treelite_harness::cubecl_cpu_case()` and drives the SAME frozen golden
//! matrix through the cubecl CPU backend, asserting 1e-5 against the identical
//! goldens the scalar reference uses.
//!
//! IMPORTANT (provenance contract): when this sibling is filled in, it MUST
//! assert its OWN backend/provenance — the per-cell `cubecl-kernel` vs
//! `scalar-fallback` manifest field (D-06) — and MUST NOT copy the hard
//! `golden.manifest.backend == "scalar-cpu"` assert at `gtil_matrix.rs:474`.
//! The scalar literal belongs to the scalar reference test only.

#[test]
#[ignore = "MISSING — Wave 4: register cubecl_cpu_case() and assert the frozen matrix to 1e-5 + provenance"]
fn gtil_matrix_cubecl() {
    todo!("Wave 4 (plan 06-05): cubecl_cpu_case() over the frozen matrix, 1e-5 + per-cell provenance");
}

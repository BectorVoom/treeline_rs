//! RED scaffold — Wave 4 bit-identical determinism (SC2, plan 06-05).
//!
//! Nyquist MISSING marker: two runs of the same cubecl CPU-backend prediction
//! must produce bit-identical output (`f64::to_bits()` equality), the SC2
//! determinism contract for the CPU reference backend.

#[test]
#[ignore = "MISSING — Wave 4: two-run .to_bits() bit-identity (SC2)"]
fn determinism_two_run_bit_identity() {
    todo!("Wave 4 (plan 06-05): two predict_cpu runs, assert .to_bits() equality (SC2)");
}

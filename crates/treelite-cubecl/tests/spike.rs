//! RED scaffold — Wave 1 cubecl descent spike (plan 06-02).
//!
//! Nyquist MISSING marker: this test exists FIRST (before any kernel) so the
//! Wave 1 spike — a break-free bounded `while !is_leaf` descend, in-kernel f64,
//! and the `exp2`/`softmax_f64` cast order — has a pre-existing automated check
//! it must turn green. Per D-04 the spike is a CONFIRMATION step, not a
//! go/no-go gate; cubecl f64/mixed precision is already validated to 1e-5.

#[test]
#[ignore = "MISSING — Wave 1 spike: break-free descend + f64 in-kernel + exp2/softmax_f64 cast order"]
fn spike_descend_f64_exp2() {
    todo!("Wave 1 (plan 06-02): minimal #[cube(launch)] break-free descend + one postprocessor cast-order spike");
}

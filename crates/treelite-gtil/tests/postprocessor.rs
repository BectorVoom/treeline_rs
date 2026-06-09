//! Tests for the `identity` and `sigmoid` postprocessors (Plan 01-03 Task 1).
//!
//! Verifies the verbatim port of `postprocessor.cc:19-37`: identity
//! passthrough, `sigmoid(_, 0.0) == 0.5`, the inverse-of-margin sanity check
//! (`sigmoid(1.0, -ln(3)) ≈ 0.25`, the transform that turns base_score 0.25
//! into its margin), and monotonicity.

use treelite_gtil::postprocessor::{identity, sigmoid};

#[test]
fn identity_is_passthrough() {
    assert_eq!(identity(1.0, 3.5_f32), 3.5_f32);
    assert_eq!(identity(0.0, -2.0_f32), -2.0_f32);
    assert_eq!(identity(42.0, 0.0_f32), 0.0_f32);
}

#[test]
fn sigmoid_at_zero_is_half() {
    assert_eq!(sigmoid(1.0, 0.0_f32), 0.5_f32);
}

#[test]
fn sigmoid_inverts_base_score_margin() {
    // The XGBoost loader transforms base_score 0.25 into the margin
    // -ln(1/0.25 - 1) = -ln(3) ≈ -1.0986123. Applying sigmoid to that margin
    // must recover ~0.25 (this is the round-trip the golden depends on).
    let margin = -(3.0_f32.ln());
    let p = sigmoid(1.0, margin);
    assert!(
        (p - 0.25_f32).abs() < 1e-6,
        "sigmoid(1.0, -ln(3)) = {p}, expected ~0.25"
    );
}

#[test]
fn sigmoid_is_monotonically_increasing() {
    let a = sigmoid(1.0, -2.0_f32);
    let b = sigmoid(1.0, 0.0_f32);
    let c = sigmoid(1.0, 2.0_f32);
    assert!(a < b, "sigmoid(-2) {a} should be < sigmoid(0) {b}");
    assert!(b < c, "sigmoid(0) {b} should be < sigmoid(2) {c}");
}

#[test]
fn sigmoid_alpha_scales_the_margin() {
    // A larger alpha pushes the same positive margin closer to 1.
    let small = sigmoid(0.5, 1.0_f32);
    let large = sigmoid(2.0, 1.0_f32);
    assert!(large > small);
    // All outputs stay strictly inside (0, 1).
    for &v in &[small, large] {
        assert!(v > 0.0 && v < 1.0);
    }
}

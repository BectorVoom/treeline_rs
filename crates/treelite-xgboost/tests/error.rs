//! Task 1 tests: objective→postprocessor map and the f64 base_score→margin
//! transform, plus typed-error discipline on an unrecognized objective (ERR-01).

use treelite_xgboost::error::XgbError;
use treelite_xgboost::objective::{get_postprocessor, transform_base_score_to_margin};

#[test]
fn objective_map_known_values() {
    // Verbatim groupings from xgboost.cc:28-50.
    assert_eq!(get_postprocessor("binary:logistic").unwrap(), "sigmoid");
    assert_eq!(get_postprocessor("reg:logistic").unwrap(), "sigmoid");
    assert_eq!(get_postprocessor("multi:softprob").unwrap(), "softmax");
    assert_eq!(get_postprocessor("multi:softmax").unwrap(), "softmax");
    assert_eq!(get_postprocessor("reg:squarederror").unwrap(), "identity");
    assert_eq!(get_postprocessor("binary:logitraw").unwrap(), "identity");
    assert_eq!(get_postprocessor("count:poisson").unwrap(), "exponential");
    assert_eq!(get_postprocessor("survival:aft").unwrap(), "exponential");
    assert_eq!(get_postprocessor("binary:hinge").unwrap(), "hinge");
}

#[test]
fn unrecognized_objective_returns_typed_err_not_panic() {
    // ERR-01: upstream TREELITE_LOG(FATAL) (xgboost.cc:47) becomes a typed Err.
    let err = get_postprocessor("totally:bogus").unwrap_err();
    match err {
        XgbError::UnrecognizedObjective(name) => assert_eq!(name, "totally:bogus"),
        other => panic!("expected UnrecognizedObjective, got {other:?}"),
    }
}

#[test]
fn sigmoid_margin_transform_is_exact_f64() {
    // -((1.0/0.25) - 1.0).ln() == -(3.0).ln() ≈ -1.0986122886681098
    let got = transform_base_score_to_margin("sigmoid", 0.25);
    let expected = -(3.0_f64).ln();
    assert_eq!(got, expected);
    // Sanity: the documented numeric value (the live CORE-04 base_scores value).
    assert!((got - (-1.0986122886681098_f64)).abs() < 1e-15);
}

#[test]
fn sigmoid_at_half_is_zero_masking_case() {
    // At base_score == 0.5 the margin is exactly 0 — this is why the fixture
    // deliberately uses 0.25, so the transform is genuinely exercised.
    assert_eq!(transform_base_score_to_margin("sigmoid", 0.5), 0.0);
}

#[test]
fn identity_postprocessor_passes_through() {
    assert_eq!(transform_base_score_to_margin("identity", 1.7), 1.7);
}

#[test]
fn exponential_margin_transform_is_ln() {
    let got = transform_base_score_to_margin("exponential", 2.0);
    assert_eq!(got, (2.0_f64).ln());
}

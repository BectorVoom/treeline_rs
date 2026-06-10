//! D-02 NaN/Inf mechanism tests (Phase 3, Plan 03-02, Task 1).
//!
//! Bare `NaN`/`Infinity`/`-Infinity` literals MUST round-trip into f32
//! thresholds/leaf values **value-position only, never inside string contents**
//! (D-02). The pre-lexer rewrites them to sentinel STRINGS (`"@NaN@"`/`"@Inf@"`/
//! `"@-Inf@"`) — never numeric literals (RESEARCH Pitfall 1) — and the `de_f32`
//! adapter recovers them. The pre-lexer tracks in-string state so attribute
//! strings are byte-unchanged (RESEARCH Pitfall 2).
//!
//! Test names use the `nan_inf_` prefix for the VALIDATION test map.

use treelite_xgboost::test_support::{de_vec_f32_value, replace_nonfinite};

#[test]
fn nan_inf_value_position_rewritten_to_sentinel_strings() {
    // Bare NaN / Infinity / -Infinity in VALUE position become sentinel strings.
    let out = replace_nonfinite("[NaN, Infinity, -Infinity]");
    assert_eq!(out, r#"["@NaN@", "@Inf@", "@-Inf@"]"#);
}

#[test]
fn nan_inf_string_contents_are_byte_unchanged() {
    // A `NaN`/`Infinity` INSIDE a string must be left byte-for-byte unchanged
    // (RESEARCH Pitfall 2 — string-safety). The scanner tracks in-string state.
    let input = r#"{"name":"NaN_count","s":"has Infinity inside"}"#;
    let out = replace_nonfinite(input);
    assert_eq!(out, input, "string contents must be byte-unchanged");
}

#[test]
fn nan_inf_sentinels_deserialize_into_f32_nonfinite() {
    // The three sentinel strings round-trip into f32::NAN / INFINITY / NEG_INF
    // via the de_vec_f32 adapter (the shared D-02 recovery point).
    let prelexed = replace_nonfinite("[NaN, Infinity, -Infinity]");
    let v = de_vec_f32_value(&prelexed).expect("sentinels should deserialize");
    assert_eq!(v.len(), 3);
    assert!(v[0].is_nan(), "first element must be NaN");
    assert_eq!(v[1], f32::INFINITY);
    assert_eq!(v[2], f32::NEG_INFINITY);
}

#[test]
fn nan_inf_infinity_never_becomes_a_numeric_literal() {
    // RESEARCH Pitfall 1: `Infinity` must NOT be substituted with an
    // out-of-range numeric literal (e.g. `1e400`), which serde_json rejects with
    // "number out of range". The sentinel-string path means a lone Infinity
    // parses cleanly into f32::INFINITY rather than failing to parse.
    let prelexed = replace_nonfinite("[Infinity]");
    // The rewritten text contains the sentinel STRING, not a number.
    assert!(prelexed.contains("\"@Inf@\""));
    assert!(!prelexed.contains("1e4"), "must not emit a numeric literal");
    let v = de_vec_f32_value(&prelexed).expect("Infinity must parse, not error");
    assert_eq!(v, vec![f32::INFINITY]);
}

#[test]
fn nan_inf_finite_numbers_pass_through_unchanged() {
    // Sanity: ordinary finite numbers are untouched and still deserialize.
    let prelexed = replace_nonfinite("[1.5, -2.0, 0.0]");
    assert_eq!(prelexed, "[1.5, -2.0, 0.0]");
    let v = de_vec_f32_value(&prelexed).unwrap();
    assert_eq!(v, vec![1.5_f32, -2.0_f32, 0.0_f32]);
}

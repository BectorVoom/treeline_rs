//! `DetectXGBoostFormat` heuristic tests (Phase 3, Plan 03-03, Task 1).
//!
//! Verifies the ported first/second-byte JSON-vs-UBJSON-vs-unknown classifier
//! (`detail/xgboost.cc:83-115`, D-09). Legacy binary is NOT auto-detected here —
//! it is reached via an explicit entry point in 03-04, matching upstream's API
//! split. Test names use the `detect_` prefix for the VALIDATION test map.

use treelite_xgboost::detect_xgboost_format;

#[test]
fn detect_json_brace_then_quote() {
    // `{"` — first byte `{`, second byte `"` → JSON.
    assert_eq!(detect_xgboost_format(b"{\""), "json");
}

#[test]
fn detect_json_brace_then_space() {
    // `{` then whitespace → JSON.
    assert_eq!(detect_xgboost_format(b"{ "), "json");
    assert_eq!(detect_xgboost_format(b"{\n"), "json");
    assert_eq!(detect_xgboost_format(b"{\r"), "json");
    assert_eq!(detect_xgboost_format(b"{\t"), "json");
}

#[test]
fn detect_json_leading_whitespace() {
    // A leading whitespace byte → JSON (whitespace is only present in JSON).
    assert_eq!(detect_xgboost_format(b" {"), "json");
    assert_eq!(detect_xgboost_format(b"\n{"), "json");
}

#[test]
fn detect_ubjson_first_byte_noop() {
    // First byte `N` (UBJSON no-op) → UBJSON.
    assert_eq!(detect_xgboost_format(b"N{"), "ubjson");
}

#[test]
fn detect_ubjson_brace_then_type_marker() {
    // `{` then any UBJSON type/no-op marker → UBJSON.
    for second in [b'N', b'$', b'#', b'i', b'U', b'I', b'l', b'L'] {
        let buf = [b'{', second];
        assert_eq!(
            detect_xgboost_format(&buf),
            "ubjson",
            "expected ubjson for {{ + {:?}",
            second as char
        );
    }
    // The validated case from the plan: `{L` (0x7B, 0x4C) → ubjson.
    assert_eq!(detect_xgboost_format(&[0x7B, 0x4C]), "ubjson");
}

#[test]
fn detect_unknown_non_brace_non_n_non_space_first_byte() {
    // First byte not `{`, not `N`, not whitespace → unknown (e.g. a legacy
    // binary first byte 0x00).
    assert_eq!(detect_xgboost_format(&[0x00]), "unknown");
    assert_eq!(detect_xgboost_format(&[0x00, 0x00]), "unknown");
}

#[test]
fn detect_unknown_brace_then_unrecognized_second_byte() {
    // `{` then an unrecognized second byte → unknown.
    assert_eq!(detect_xgboost_format(b"{x"), "unknown");
    assert_eq!(detect_xgboost_format(&[0x7B, 0x00]), "unknown");
}

#[test]
fn detect_short_slices_do_not_panic() {
    // A <2-byte slice must be handled safely (default missing byte = 0).
    assert_eq!(detect_xgboost_format(&[]), "unknown");
    assert_eq!(detect_xgboost_format(b"{"), "unknown"); // second byte defaults to 0
    assert_eq!(detect_xgboost_format(b"N"), "ubjson"); // first byte N is decisive
}

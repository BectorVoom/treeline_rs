//! XGBoost format auto-detection (XGB-04 / D-09).
//!
//! Ports `DetectXGBoostFormat` (`treelite-mainline/src/model_loader/detail/xgboost.cc:83-115`)
//! verbatim: inspect the first two bytes and classify the stream as `"json"`,
//! `"ubjson"`, or `"unknown"`.
//!
//! ## Legacy binary is NOT auto-detected here (D-09 criterion-2)
//!
//! Upstream's loader API is split into three distinct entry points
//! (`LoadXGBoostModelJSON` / `...UBJSON` / `...LegacyBinary`), and
//! `DetectXGBoostFormat` only ever disambiguates JSON vs UBJSON — a legacy
//! binary file's first byte (typically `0x00` from the `LearnerModelParam`
//! header) is not `{`, not `N`, and not whitespace, so it classifies as
//! `"unknown"` here. Legacy is reached via the explicit `load_xgboost_legacy`
//! entry point (03-04), mirroring that upstream API split. This function
//! therefore returns ONLY the three literals `"json"` / `"ubjson"` / `"unknown"`.

/// Whitespace bytes — only present in JSON, never in UBJSON
/// (`xgboost.cc:89`'s `is_space` lambda).
fn is_space(c: u8) -> bool {
    c == b' ' || c == b'\n' || c == b'\r' || c == b'\t'
}

/// Classify an XGBoost model byte stream as `"json"`, `"ubjson"`, or `"unknown"`
/// from its first two bytes (D-09).
///
/// Ported verbatim from `DetectXGBoostFormat` (`detail/xgboost.cc:83-115`). A
/// slice shorter than two bytes is handled safely: a missing byte defaults to
/// `0`, which falls through to the `"unknown"` branch (except a lone leading
/// `N`, which is decisive for UBJSON). Returns one of three `'static` string
/// literals — never a legacy verdict.
pub fn detect_xgboost_format(first_two: &[u8]) -> &'static str {
    let b0 = first_two.first().copied().unwrap_or(0);
    let b1 = first_two.get(1).copied().unwrap_or(0);

    // First look at the first byte.
    if b0 == b'N' {
        // The no-op code is only used in UBJSON.
        return "ubjson";
    } else if is_space(b0) {
        // White-spaces are only present in JSON.
        return "json";
    } else if b0 != b'{' {
        // Otherwise, should have '{' if the file is JSON or UBJSON.
        return "unknown";
    }

    // First byte is '{'. Now look at the second byte.
    if is_space(b1) || b1 == b'"' {
        // White-spaces and double quotation marks are only present in JSON.
        "json"
    } else if b1 == b'N'
        || b1 == b'$'
        || b1 == b'#'
        || b1 == b'i'
        || b1 == b'U'
        || b1 == b'I'
        || b1 == b'l'
        || b1 == b'L'
    {
        // The no-op code and type markers are only present in UBJSON.
        "ubjson"
    } else {
        "unknown"
    }
}

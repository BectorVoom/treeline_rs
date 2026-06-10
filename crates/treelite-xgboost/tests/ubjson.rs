//! UBJSON loader tests (Phase 3, Plan 03-03, Task 2 — XGB-02 / D-03).
//!
//! The hand-rolled UBJSON tag decoder must:
//!  1. handle the `[$<type>#<count>` strongly-typed-container fast path
//!     (RESEARCH Pitfall 4) — XGBoost emits it everywhere;
//!  2. emit the SAME `"@NaN@"`/`"@Inf@"`/`"@-Inf@"` sentinel STRINGS as the JSON
//!     path for non-finite `d`/`D` floats, NOT `Value::Null` (Pitfall 5), so the
//!     shared `de_f32` adapter recovers them identically (criterion-2 parity);
//!  3. converge at the SAME `XgbModelJson` structs + `build_model_from_parsed`
//!     so `load_xgboost_ubjson(xgb_3format.ubj)` produces the IDENTICAL Model as
//!     `load_xgboost_json(xgb_3format.json)` — byte-faithful to the single golden
//!     blob (D-01/D-10) and predicting within 1e-5 (D-05);
//!  4. reject an oversized `#`count with a typed `XgbError::Ubjson`, never a
//!     panic/OOM (ASVS V5, T-03-U01).
//!
//! Test names use the `ubjson_` prefix for the VALIDATION test map.

use std::path::Path;

use treelite_xgboost::error::XgbError;
use treelite_xgboost::test_support::decode_ubjson;
use treelite_xgboost::{load_xgboost_json, load_xgboost_ubjson};

fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

fn read_str(name: &str) -> String {
    std::fs::read_to_string(fixture_path(name)).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

fn read_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture_path(name)).unwrap_or_else(|e| panic!("reading {name}: {e}"))
}

/// Build a tag-prefixed UBJSON integer (`U`=uint8 for small counts).
fn ubj_u8(v: u8) -> Vec<u8> {
    vec![b'U', v]
}

#[test]
fn ubjson_typed_float32_array_fast_path() {
    // `[$d#U<count>` then `count` raw f32 LE values, no per-element tags
    // (the $/# fast path — RESEARCH Pitfall 4).
    let mut bytes = vec![b'[', b'$', b'd', b'#'];
    bytes.extend(ubj_u8(3));
    // UBJSON is big-endian (network byte order).
    for v in [1.5_f32, -2.25_f32, 0.0_f32] {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    let value = decode_ubjson(&bytes).expect("typed float32 array decodes");
    let arr = value.as_array().expect("decoded to an array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_f64().unwrap() as f32, 1.5_f32);
    assert_eq!(arr[1].as_f64().unwrap() as f32, -2.25_f32);
    assert_eq!(arr[2].as_f64().unwrap() as f32, 0.0_f32);
}

#[test]
fn ubjson_non_finite_float_becomes_sentinel_string() {
    // A `d`-tagged f32::NAN / INFINITY / -INFINITY decodes to the sentinel
    // STRING, not Value::Null (Pitfall 5) — so the shared de_f32 adapter
    // recovers it identically to the JSON path.
    for (val, sentinel) in [
        (f32::NAN, "@NaN@"),
        (f32::INFINITY, "@Inf@"),
        (f32::NEG_INFINITY, "@-Inf@"),
    ] {
        let mut bytes = vec![b'd'];
        bytes.extend_from_slice(&val.to_be_bytes());
        let value = decode_ubjson(&bytes).expect("non-finite float decodes");
        assert_eq!(
            value.as_str(),
            Some(sentinel),
            "non-finite f32 must become sentinel string {sentinel}, got {value:?}"
        );
    }
}

#[test]
fn ubjson_loads_identical_model_as_json_byte_for_byte() {
    // D-01/D-10: serialize(load_ubjson(.ubj)) == serialize(load_json(.json))
    // == golden_v5_3format.bin, byte-for-byte.
    let mut from_ubj =
        load_xgboost_ubjson(&read_bytes("xgb_3format.ubj")).expect("ubjson 3format loads");
    let mut from_json =
        load_xgboost_json(&read_str("xgb_3format.json")).expect("json 3format loads");

    let ubj_bytes = treelite_core::serialize_to_buffer(&mut from_ubj);
    let json_bytes = treelite_core::serialize_to_buffer(&mut from_json);
    let golden = read_bytes("golden_v5_3format.bin");

    assert_eq!(
        ubj_bytes, json_bytes,
        "UBJSON and JSON loads must serialize to identical v5 bytes (D-01)"
    );
    assert_eq!(
        ubj_bytes, golden,
        "UBJSON v5 serialization must equal golden_v5_3format.bin (D-10)"
    );
}

#[test]
fn ubjson_predicts_within_1e5_of_golden() {
    // The UBJSON-loaded model predicts within 1e-5 of xgb_3format.golden.json.
    let model = load_xgboost_ubjson(&read_bytes("xgb_3format.ubj")).expect("ubjson loads");

    let raw = read_str("xgb_3format.golden.json");
    let golden: serde_json::Value = serde_json::from_str(&raw).expect("golden parses");
    let input = golden["input"].as_array().expect("input array");
    let expected = golden["output"].as_array().expect("output array");

    let num_row = input.len();
    let num_feature = input[0].as_array().unwrap().len();
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * num_feature);
    for row in input {
        for cell in row.as_array().unwrap() {
            // A null cell (normalized missing value) → NaN.
            let v = cell.as_f64().map(|x| x as f32).unwrap_or(f32::NAN);
            flat.push(v);
        }
    }

    let got = treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default()).expect("predict");
    assert_eq!(got.len(), expected.len());
    let mut max_dev = 0.0_f64;
    for (i, &g) in got.iter().enumerate() {
        let e = expected[i].as_f64().unwrap() as f32;
        let d = (g - e).abs() as f64;
        if d > max_dev {
            max_dev = d;
        }
        assert!(
            (g - e).abs() <= 1e-5,
            "row {i}: ubjson predict {g} vs golden {e} exceeds 1e-5"
        );
    }
    assert!(max_dev <= 1e-5, "max deviation {max_dev} exceeds 1e-5");
}

#[test]
fn ubjson_oversized_count_returns_typed_err_not_panic() {
    // A declared `#`count larger than the remaining stream must return a typed
    // XgbError::Ubjson, never a panic/OOM (ASVS V5, T-03-U01). Declare a typed
    // float32 array of 1_000_000 elements but supply no element bytes.
    let mut bytes = vec![b'[', b'$', b'd', b'#'];
    // `l` = int32 count = 1_000_000 (big-endian, UBJSON byte order).
    bytes.push(b'l');
    bytes.extend_from_slice(&1_000_000_i32.to_be_bytes());
    // (No element bytes follow — the count vastly exceeds remaining bytes.)
    match decode_ubjson(&bytes) {
        Err(XgbError::Ubjson { .. }) => {}
        Err(other) => panic!("expected XgbError::Ubjson, got {other:?}"),
        Ok(v) => panic!("expected an error, got Ok({v:?})"),
    }
}

#[test]
fn ubjson_deeply_nested_input_returns_typed_err_not_stack_overflow() {
    // CR-04 regression: a stream of many `[` openers recurses once per level.
    // Without a depth cap this overflows the native stack and aborts the process
    // (SIGSEGV) — uncatchable. With the cap it must return a typed
    // XgbError::Ubjson. 100_000 openers is far past any legitimate model and
    // well past the MAX_DEPTH cap.
    let bytes = vec![b'['; 100_000];
    match decode_ubjson(&bytes) {
        Err(XgbError::Ubjson { .. }) => {}
        Err(other) => panic!("expected XgbError::Ubjson, got {other:?}"),
        Ok(v) => panic!("expected a depth error, got Ok({v:?})"),
    }
}

#[test]
fn ubjson_modestly_nested_input_still_decodes() {
    // A nesting depth comfortably under the cap must still decode successfully,
    // so the CR-04 guard does not reject legitimate (shallow) nested arrays.
    // 10 nested arrays, innermost terminated by matching `]` closers.
    let mut bytes = vec![b'['; 10];
    bytes.extend(std::iter::repeat_n(b']', 10));
    let value = decode_ubjson(&bytes).expect("10-deep nesting must decode");
    assert!(value.is_array(), "outermost value is an array");
}

#[test]
fn ubjson_truncated_mid_tag_returns_typed_err_not_panic() {
    // A truncated stream (a `d` tag with fewer than 4 trailing bytes) must
    // return a typed error, never an out-of-bounds panic (T-03-U02).
    let bytes = vec![b'd', 0x00, 0x00]; // only 2 of 4 float bytes
    match decode_ubjson(&bytes) {
        Err(XgbError::Ubjson { .. }) => {}
        Err(other) => panic!("expected XgbError::Ubjson, got {other:?}"),
        Ok(v) => panic!("expected an error, got Ok({v:?})"),
    }
}

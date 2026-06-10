//! JSON vertical-slice tests (Phase 3, Plan 03-02, Task 2).
//!
//! These close DEF-02-01 for the JSON path: the widened loader emits
//! `sum_hess`/`gain`/`attributes:None` so its serialized v5 bytes equal the
//! single upstream golden blob (`fixtures/golden_v5_3format.bin`), and the same
//! model predicts within `1e-5` of the shared golden
//! (`fixtures/xgb_3format.golden.json`). Plus the XGB-05 `parse_base_score`
//! scalar/vector/version-gate cases.
//!
//! Test names use the `json_` prefix for the VALIDATION test map.

use std::path::Path;

use treelite_xgboost::{load_xgboost_json, parse_base_score, transform_base_score_to_margin};

/// Resolve a path under the workspace `fixtures/` dir. `CARGO_MANIFEST_DIR` is
/// `crates/treelite-xgboost`, so go up two levels.
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

/// First offset at which two byte slices differ (for a precise divergence
/// report), or a length-difference point.
fn first_diff(got: &[u8], want: &[u8]) -> Option<usize> {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return Some(i);
        }
    }
    if got.len() != want.len() {
        Some(n)
    } else {
        None
    }
}

#[test]
fn json_widened_keys_load_into_f32_model() {
    // The full recognized-key xgb_3format.json (6 trees, 4 features, vector
    // base_score, categorical/stat keys present) loads into the F32 variant.
    let model = load_xgboost_json(&read_str("xgb_3format.json")).expect("3format json loads");
    match &model.variant {
        treelite_core::ModelVariant::F32(p) => assert_eq!(p.num_trees(), 6),
        treelite_core::ModelVariant::F64(_) => panic!("expected F32 variant"),
    }
    assert_eq!(model.task_type, treelite_core::TaskType::kBinaryClf);
    assert_eq!(model.postprocessor, "sigmoid");
    assert_eq!(model.num_feature, 4);
    assert_eq!(model.num_target, 1);
}

#[test]
fn json_serialize_equals_golden_v5_byte_for_byte() {
    // DEF-02-01 (JSON path): serialize(load_json(xgb_3format.json)) must equal
    // golden_v5_3format.bin byte-for-byte. This requires sum_hess on every node,
    // gain on internal nodes, and attributes:None (→ "{}") — the D-10 close.
    let mut model = load_xgboost_json(&read_str("xgb_3format.json")).expect("loads");
    let produced = treelite_core::serialize_to_buffer(&mut model);
    let golden = std::fs::read(fixture_path("golden_v5_3format.bin")).expect("golden blob");

    if let Some(off) = first_diff(&produced, &golden) {
        panic!(
            "JSON v5 serialization diverges from golden_v5_3format.bin at offset {off}: \
             produced={:?} golden={:?} (produced len {}, golden len {})",
            produced.get(off),
            golden.get(off),
            produced.len(),
            golden.len()
        );
    }
    assert_eq!(
        produced, golden,
        "JSON path must equal the golden blob (D-10)"
    );
}

/// The shared prediction golden shape (`{input, output, manifest}`). Parsed
/// locally (not via `treelite_harness`, which depends on this crate — that would
/// be a dependency cycle). Bare `NaN` input cells are normalized to JSON `null`
/// then mapped to `f32::NAN`.
#[derive(serde::Deserialize)]
struct Golden {
    input: Vec<Vec<Option<f32>>>,
    output: Vec<f32>,
}

/// Replace standalone bare `NaN` tokens with `null` so `serde_json` (which
/// rejects the non-standard literal) parses the golden's missing-value rows.
/// Only tokens bounded by non-identifier characters are replaced.
fn normalize_nan(raw: &str) -> String {
    let b = raw.as_bytes();
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'N'
            && raw[i..].starts_with("NaN")
            && (i == 0 || !is_ident(b[i - 1]))
            && (i + 3 >= b.len() || !is_ident(b[i + 3]))
        {
            out.extend_from_slice(b"null");
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

#[test]
fn json_predicts_within_1e5_of_shared_golden() {
    // The same model predicts within 1e-5 of the shared prediction golden.
    let model = load_xgboost_json(&read_str("xgb_3format.json")).expect("loads");

    let raw = read_str("xgb_3format.golden.json");
    let golden: Golden = serde_json::from_str(&normalize_nan(&raw)).expect("parse golden");

    let num_row = golden.input.len();
    let num_feature = golden.input[0].len();
    let mut flat: Vec<f32> = Vec::with_capacity(num_row * num_feature);
    for row in &golden.input {
        flat.extend(row.iter().map(|c| c.unwrap_or(f32::NAN)));
    }

    let rust = treelite_gtil::predict(&model, &flat, num_row, &treelite_gtil::Config::default()).expect("predict");
    assert_eq!(rust.len(), golden.output.len());
    let mut max_dev: f64 = 0.0;
    for (i, &got) in rust.iter().enumerate() {
        let expected = golden.output[i];
        let delta = (got - expected).abs() as f64;
        if delta > max_dev {
            max_dev = delta;
        }
        approx::assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
    }
    println!("json 1e-5 max|delta| = {max_dev:e}");
}

#[test]
fn json_parse_base_score_scalar_form() {
    // Scalar string "5E-1" with expand_to=1 yields one element. With the sigmoid
    // postprocessor + transform applied, sigmoid(0.5) → 0 margin (a no-op).
    let v = parse_base_score("5E-1", 1, "sigmoid", true).expect("scalar parses");
    assert_eq!(v.len(), 1);
    let expected = transform_base_score_to_margin("sigmoid", 0.5);
    assert!((v[0] - expected).abs() < 1e-15);
    // sigmoid(0.5) margin is exactly 0.0.
    assert!(v[0].abs() < 1e-15);

    // Scalar fills across expand_to entries.
    let v3 = parse_base_score("0.25", 3, "identity", false).expect("scalar fill");
    assert_eq!(v3, vec![0.25_f64; 3]);
}

#[test]
fn json_parse_base_score_vector_form() {
    // Vector string "[0.1, 0.2]" with expand_to=2 yields two f64 elements,
    // transformed element-wise. Use "identity" so the values pass through.
    let v = parse_base_score("[0.1, 0.2]", 2, "identity", true).expect("vector parses");
    assert_eq!(v.len(), 2);
    assert!((v[0] - 0.1_f64).abs() < 1e-6);
    assert!((v[1] - 0.2_f64).abs() < 1e-6);

    // The verify-narrow fixture's actual form: "[5E-1]" with expand_to=1.
    let v1 = parse_base_score("[5E-1]", 1, "sigmoid", true).expect("fixture vector");
    assert_eq!(v1.len(), 1);
    assert!(v1[0].abs() < 1e-15); // sigmoid(0.5) → 0 margin.
}

#[test]
fn json_parse_base_score_version_gate_negative() {
    // With apply_transform=false (version[0]==0 / major_version 0), the transform
    // does NOT fire: the raw probability passes through unchanged.
    let v = parse_base_score("0.25", 1, "sigmoid", false).expect("no transform");
    assert_eq!(v, vec![0.25_f64]);
    assert_ne!(v[0], transform_base_score_to_margin("sigmoid", 0.25));
}

#[test]
fn json_parse_base_score_vector_wrong_length_is_typed_error() {
    // A vector base_score whose length != expand_to is a typed BaseScoreShape
    // error, never a silent truncation (T-03-V04).
    match parse_base_score("[0.1, 0.2, 0.3]", 2, "identity", true) {
        Err(treelite_xgboost::XgbError::BaseScoreShape { expected, got }) => {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        }
        Err(other) => panic!("expected BaseScoreShape, got {other:?}"),
        Ok(_) => panic!("expected BaseScoreShape error, got Ok"),
    }
}

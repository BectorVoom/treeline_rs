//! XGBoost legacy-binary loader tests (Phase 3, Plan 03-04 — XGB-03 / D-07 / D-08).
//!
//! The hand-rolled little-endian cursor must:
//!  1. parse the vendored `mushroom.model` legacy fixture end-to-end (1501 bytes
//!     consumed exactly): 2 trees (node counts 13 / 11), objective
//!     `"binary:logistic"`, 127 features — an independent smoke test;
//!  2. unpack `sindex` correctly (`split_index = sindex & 0x7FFFFFFF`,
//!     `default_left = sindex >> 31`), detect leaves via `cleft == -1`, and read
//!     the leaf value / split condition from the `info` union reinterpreted as f32
//!     (Pitfall 6);
//!  3. respect the version gate (`major_version >= 1`): mushroom's `major_version`
//!     is 0, so the base_score → margin transform does NOT fire;
//!  4. return a typed `XgbError::Legacy` (never a panic / OOB) on a buffer
//!     truncated mid-`LearnerModelParam` (T-03-L03);
//!  5. reject a `bs64` (base64) magic with a typed error and consume a `binf`
//!     magic (T-03-L06).
//!
//! Test names use the `legacy_` prefix for the VALIDATION test map.

use std::path::Path;

use treelite_core::ModelVariant;
use treelite_xgboost::error::XgbError;
use treelite_xgboost::load_xgboost_legacy;

/// Resolve the vendored mushroom fixture (a real legacy-binary model).
fn mushroom_path() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../treelite-mainline/tests/examples/mushroom/mushroom.model")
        .to_string_lossy()
        .into_owned()
}

fn read_mushroom() -> Vec<u8> {
    std::fs::read(mushroom_path()).expect("reading vendored mushroom.model")
}

/// (1) Smoke: the vendored mushroom legacy model parses end-to-end.
#[test]
fn legacy_mushroom_smoke_parses_end_to_end() {
    let bytes = read_mushroom();
    assert_eq!(
        bytes.len(),
        1501,
        "mushroom.model is the expected 1501 bytes"
    );

    let model = load_xgboost_legacy(&bytes).expect("mushroom.model loads");

    // 2 trees, node counts 13 / 11.
    let preset = match &model.variant {
        ModelVariant::F32(p) => p,
        ModelVariant::F64(_) => panic!("XGBoost legacy must yield the F32 variant"),
    };
    assert_eq!(preset.trees.len(), 2, "mushroom has 2 trees");
    assert_eq!(preset.trees[0].num_nodes, 13, "tree 0 has 13 nodes");
    assert_eq!(preset.trees[1].num_nodes, 11, "tree 1 has 11 nodes");

    // objective binary:logistic → 127 features → binary classifier.
    assert_eq!(model.num_feature, 127, "mushroom has 127 features");
    assert_eq!(model.postprocessor, "sigmoid", "binary:logistic → sigmoid");
}

/// (2) sindex bit-unpacking + leaf detection + info-union reinterpretation.
///
/// Builds a minimal one-tree legacy buffer in-memory with a single internal node
/// whose `sindex = 0x80000005` (split_index 5, default_left set) pointing at two
/// leaves, then loads it and asserts the decoded topology. This exercises the
/// `& 0x7FFFFFFF` / `>> 31` ops and `cleft == -1` leaf detection directly.
#[test]
fn legacy_sindex_unpacking_and_leaf_detection() {
    let buf = build_minimal_legacy(
        /* major_version */ 0,
        /* num_feature */ 6,
        /* num_class */ 0,
        /* num_target */ 0,
        /* base_score */ 0.5,
        "reg:squarederror",
        // one tree: node0 internal (sindex=0x80000005, split_cond=1.5,
        // cleft=1, cright=2), node1 leaf (value 10.0), node2 leaf (value 20.0).
        &[
            LegacyNode::internal(0x8000_0005, 1.5, 1, 2),
            LegacyNode::leaf(10.0),
            LegacyNode::leaf(20.0),
        ],
    );

    let model = load_xgboost_legacy(&buf).expect("minimal legacy buffer loads");
    let preset = match &model.variant {
        ModelVariant::F32(p) => p,
        ModelVariant::F64(_) => panic!("expected F32"),
    };
    let tree = &preset.trees[0];
    assert_eq!(tree.num_nodes, 3, "one internal + two leaves");

    // Node 0 is the internal split: split_index 5, default_left true, kLT @ 1.5.
    assert_eq!(
        tree.split_index(0),
        5,
        "split_index = 0x80000005 & 0x7FFFFFFF = 5"
    );
    assert!(
        tree.default_left(0),
        "default_left = 0x80000005 >> 31 = 1 (true)"
    );
    // Children 1 and 2 are leaves (cleft == -1 in legacy => is_leaf in Model).
    assert!(tree.is_leaf(1), "node 1 is a leaf");
    assert!(tree.is_leaf(2), "node 2 is a leaf");
    assert!(!tree.is_leaf(0), "node 0 is internal");
}

/// (3) Version gate (negative case): major_version 0 → NO base_score transform.
///
/// With `binary:logistic` (sigmoid postprocessor) and `base_score = 0.5`, the
/// margin transform `-ln(1/0.5 - 1)` would be `0.0` either way — so to make the
/// gate observable we use `base_score = 0.25`: transformed it would be
/// `-ln(1/0.25 - 1) = -ln(3) ≈ -1.0986`; un-transformed it stays `0.25`. With
/// major_version 0 the gate must NOT fire, so the stored base_score is `0.25`.
#[test]
fn legacy_version_gate_zero_does_not_transform_base_score() {
    let buf = build_minimal_legacy(
        /* major_version */ 0,
        /* num_feature */ 1,
        /* num_class */ 0,
        /* num_target */ 0,
        /* base_score */ 0.25,
        "binary:logistic",
        &[LegacyNode::leaf(0.0)],
    );
    let model = load_xgboost_legacy(&buf).expect("loads");
    assert_eq!(model.base_scores.len(), 1, "scalar base score, 1 entry");
    let got = model.base_scores[0];
    assert!(
        (got - 0.25).abs() < 1e-9,
        "major_version 0 must NOT transform base_score: got {got}, want 0.25"
    );
}

/// (3b) Version gate (positive case): major_version 1 → base_score IS transformed.
#[test]
fn legacy_version_gate_one_transforms_base_score() {
    let buf = build_minimal_legacy(
        /* major_version */ 1,
        /* num_feature */ 1,
        /* num_class */ 0,
        /* num_target */ 0,
        /* base_score */ 0.25,
        "binary:logistic",
        &[LegacyNode::leaf(0.0)],
    );
    let model = load_xgboost_legacy(&buf).expect("loads");
    let got = model.base_scores[0];
    let want = -((1.0_f64 / 0.25) - 1.0).ln(); // -ln(3) ≈ -1.0986
    assert!(
        (got - want).abs() < 1e-9,
        "major_version 1 must transform base_score: got {got}, want {want}"
    );
}

/// (4) Truncation: a buffer cut mid-`LearnerModelParam` returns `XgbError::Legacy`
/// (never a panic / OOB).
#[test]
fn legacy_truncated_mid_header_returns_typed_err_not_panic() {
    let bytes = read_mushroom();
    // 70 bytes is inside the 136-byte LearnerModelParam.
    let truncated = &bytes[..70];
    match load_xgboost_legacy(truncated) {
        Err(XgbError::Legacy { .. }) => {}
        Err(other) => panic!("expected XgbError::Legacy, got {other:?}"),
        Ok(_) => panic!("expected an error on a truncated header, got Ok(model)"),
    }
}

/// (4b) An empty buffer also returns a typed error, not a panic.
#[test]
fn legacy_empty_buffer_returns_typed_err() {
    match load_xgboost_legacy(&[]) {
        Err(XgbError::Legacy { .. }) => {}
        Err(other) => panic!("expected XgbError::Legacy, got {other:?}"),
        Ok(_) => panic!("expected an error on an empty buffer, got Ok(model)"),
    }
}

/// (5) Magic: a `bs64` (base64) prefix is rejected with a typed error.
#[test]
fn legacy_bs64_magic_is_rejected() {
    let mut bytes = b"bs64".to_vec();
    bytes.extend_from_slice(&[0u8; 200]);
    match load_xgboost_legacy(&bytes) {
        Err(XgbError::Legacy { detail, .. }) => {
            assert!(
                detail.contains("bs64") || detail.to_lowercase().contains("base64"),
                "error should mention base64/bs64: {detail}"
            );
        }
        Err(other) => panic!("expected XgbError::Legacy, got {other:?}"),
        Ok(_) => panic!("expected bs64 to be rejected, got Ok(model)"),
    }
}

/// (5b) A `binf` magic prefix is consumed: a `binf`-prefixed mushroom parses
/// identically to the bare mushroom.
#[test]
fn legacy_binf_magic_is_consumed() {
    let bare = read_mushroom();
    let mut prefixed = b"binf".to_vec();
    prefixed.extend_from_slice(&bare);

    let m_bare = load_xgboost_legacy(&bare).expect("bare mushroom loads");
    let m_prefixed = load_xgboost_legacy(&prefixed).expect("binf-prefixed mushroom loads");

    // Both serialize to identical v5 bytes (binf magic was consumed, not parsed
    // as data).
    let mut a = m_bare;
    let mut b = m_prefixed;
    let ba = treelite_core::serialize_to_buffer(&mut a);
    let bb = treelite_core::serialize_to_buffer(&mut b);
    assert_eq!(
        ba, bb,
        "binf prefix must be consumed, yielding the same model"
    );
}

// ---------------------------------------------------------------------------
// In-memory legacy-buffer builder (little-endian), for the unit-style tests.
// ---------------------------------------------------------------------------

/// A node to encode into a legacy buffer.
struct LegacyNode {
    cleft: i32,
    cright: i32,
    sindex: u32,
    info: f32,
}

impl LegacyNode {
    fn internal(sindex: u32, split_cond: f32, cleft: i32, cright: i32) -> Self {
        LegacyNode {
            cleft,
            cright,
            sindex,
            info: split_cond,
        }
    }
    fn leaf(value: f32) -> Self {
        LegacyNode {
            cleft: -1,
            cright: -1,
            sindex: 0,
            info: value,
        }
    }
}

/// Encode a minimal single-tree legacy-binary buffer (no magic prefix).
fn build_minimal_legacy(
    major_version: u32,
    num_feature: u32,
    num_class: i32,
    num_target: u32,
    base_score: f32,
    objective: &str,
    nodes: &[LegacyNode],
) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::new();

    // --- LearnerModelParam (136 bytes) ---
    b.extend_from_slice(&base_score.to_le_bytes());
    b.extend_from_slice(&num_feature.to_le_bytes());
    b.extend_from_slice(&num_class.to_le_bytes());
    b.extend_from_slice(&0i32.to_le_bytes()); // contain_extra_attrs
    b.extend_from_slice(&0i32.to_le_bytes()); // contain_eval_metrics
    b.extend_from_slice(&major_version.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // minor_version
    b.extend_from_slice(&num_target.to_le_bytes());
    b.extend_from_slice(&[0u8; 26 * 4]); // pad2[26]
    assert_eq!(b.len(), 136);

    // --- Objective name (u64 len + bytes) ---
    b.extend_from_slice(&(objective.len() as u64).to_le_bytes());
    b.extend_from_slice(objective.as_bytes());

    // --- Booster name "gbtree" ---
    let gbm_name = "gbtree";
    b.extend_from_slice(&(gbm_name.len() as u64).to_le_bytes());
    b.extend_from_slice(gbm_name.as_bytes());

    // --- GBTreeModelParam (168 bytes) ---
    let gbm_start = b.len();
    b.extend_from_slice(&1i32.to_le_bytes()); // num_trees
    b.extend_from_slice(&1i32.to_le_bytes()); // num_roots
    b.extend_from_slice(&(num_feature as i32).to_le_bytes()); // num_feature
    b.extend_from_slice(&0i32.to_le_bytes()); // pad1
    b.extend_from_slice(&0i64.to_le_bytes()); // pad2
    b.extend_from_slice(&1i32.to_le_bytes()); // num_output_group
    b.extend_from_slice(&0i32.to_le_bytes()); // size_leaf_vector
    b.extend_from_slice(&[0u8; 32 * 4]); // pad3[32]
    // GBTreeModelParam is 160 bytes (see legacy.rs SIZE_GBTREE_MODEL_PARAM note).
    assert_eq!(b.len() - gbm_start, 160);

    // --- One tree: TreeParam (148 bytes) ---
    let tp_start = b.len();
    b.extend_from_slice(&1i32.to_le_bytes()); // num_roots
    b.extend_from_slice(&(nodes.len() as i32).to_le_bytes()); // num_nodes
    b.extend_from_slice(&0i32.to_le_bytes()); // num_deleted
    b.extend_from_slice(&0i32.to_le_bytes()); // max_depth
    b.extend_from_slice(&(num_feature as i32).to_le_bytes()); // num_feature
    b.extend_from_slice(&0i32.to_le_bytes()); // size_leaf_vector
    b.extend_from_slice(&[0u8; 31 * 4]); // reserved[31]
    assert_eq!(b.len() - tp_start, 148);

    // --- Nodes (20 bytes each) ---
    for n in nodes {
        b.extend_from_slice(&0i32.to_le_bytes()); // parent
        b.extend_from_slice(&n.cleft.to_le_bytes());
        b.extend_from_slice(&n.cright.to_le_bytes());
        b.extend_from_slice(&n.sindex.to_le_bytes());
        b.extend_from_slice(&n.info.to_le_bytes());
    }

    // --- NodeStats (16 bytes each) ---
    for _ in nodes {
        b.extend_from_slice(&0f32.to_le_bytes()); // loss_chg
        b.extend_from_slice(&1f32.to_le_bytes()); // sum_hess
        b.extend_from_slice(&0f32.to_le_bytes()); // base_weight
        b.extend_from_slice(&0i32.to_le_bytes()); // leaf_child_cnt
    }

    // --- tree_info (num_trees × i32) ---
    b.extend_from_slice(&0i32.to_le_bytes());

    b
}

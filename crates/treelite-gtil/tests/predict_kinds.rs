//! `PredictKind::LeafId` + `PredictKind::ScorePerTree` dispatch tests
//! (Plan 05-04 Task 2, GTIL-03).
//!
//! Asserts:
//! - `LeafId` output is `(num_row, num_tree)` integer leaf NODE ids cast into
//!   the `O` buffer (A4), no postprocess/average/base-score;
//! - `ScorePerTree` on a scalar-leaf model writes the raw leaf value at index 0
//!   per `(row, tree)` (third-dim size 1, Pitfall 5);
//! - `ScorePerTree` on a leaf-vector model writes each leaf-vector element;
//! - neither kind applies postprocess / averaging / base-score (raw leaf data).

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, PredictKind, predict};

/// Single-split `Tree<f32>`: node 0 numerical test (`kLT`, default-left), node 1
/// = left leaf, node 2 = right leaf. So the reached leaf NODE id is 1 (left) or
/// 2 (right).
fn split_tree(feature: i32, threshold: f32, left_leaf: f32, right_leaf: f32) -> Tree<f32> {
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, 0.0, 0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0, left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    // leaf_vector CSR offsets sized to num_nodes, begin == end ⇒ scalar leaves.
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0, 0, 0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0, 0, 0]);
    t
}

/// A single-node leaf-VECTOR tree whose only node (id 0) returns `vec`.
fn leaf_vector_tree(vec: Vec<f32>) -> Tree<f32> {
    let mut t = Tree::<f32>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    let n = vec.len() as u64;
    t.leaf_vector = TreeBuf::from_owned(vec);
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![n]);
    t
}

/// Binary scalar `(num_row, 1, 1)` F32 model over `trees` with `num_feature`
/// features and the given postprocessor / base score.
fn scalar_model(
    trees: Vec<Tree<f32>>,
    num_feature: i32,
    postprocessor: &str,
    base_score: f64,
) -> Model {
    let num_tree = trees.len();
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = num_feature;
    m.num_target = 1;
    m.num_class = vec![1];
    m.leaf_vector_shape = vec![1, 1];
    m.target_id = vec![0; num_tree];
    m.class_id = vec![0; num_tree];
    m.postprocessor = postprocessor.to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![base_score];
    m
}

#[test]
fn leaf_id_writes_node_ids_shape_num_row_times_num_tree() {
    // Two trees. Tree A splits on feature 0, Tree B on feature 1; node 1 = left
    // leaf, node 2 = right leaf. We choose feature values so each tree reaches a
    // KNOWN leaf node id.
    let m = scalar_model(
        vec![split_tree(0, 0.5, 1.0, -1.0), split_tree(1, 0.5, 0.5, -0.5)],
        2,
        // a non-identity postprocessor + non-zero base, to prove LeafId ignores both
        "sigmoid",
        10.0,
    );
    let cfg = Config {
        kind: PredictKind::LeafId,
        nthread: 0,
    };
    // Row 0: f0 = 0.0 (<0.5 → tree A LEFT = node 1), f1 = 9.0 (≥0.5 → tree B RIGHT = node 2).
    // Row 1: f0 = 9.0 (≥0.5 → tree A RIGHT = node 2), f1 = 0.0 (<0.5 → tree B LEFT = node 1).
    let data = [0.0_f32, 9.0, 9.0, 0.0];
    let out = predict(&m, &data, 2, &cfg).unwrap();
    // Shape (num_row=2, num_tree=2) row-major.
    assert_eq!(out.len(), 4, "LeafId output length is num_row * num_tree");
    assert_eq!(out[0], 1.0_f32, "row0 tree0 → node 1");
    assert_eq!(out[1], 2.0_f32, "row0 tree1 → node 2");
    assert_eq!(out[2], 2.0_f32, "row1 tree0 → node 2");
    assert_eq!(out[3], 1.0_f32, "row1 tree1 → node 1");
}

#[test]
fn score_per_tree_scalar_writes_raw_leaf_at_index_0() {
    // Two scalar trees; ScorePerTree writes the raw per-tree leaf value at index
    // 0 (third-dim size lvs = leaf_vector_shape[0]*[1] = 1*1 = 1). A non-identity
    // postprocessor + base must be IGNORED (raw leaf data only).
    let m = scalar_model(
        vec![split_tree(0, 0.5, 3.0, -3.0), split_tree(1, 0.5, 7.0, -7.0)],
        2,
        "sigmoid",
        100.0,
    );
    let cfg = Config {
        kind: PredictKind::ScorePerTree,
        nthread: 0,
    };
    // One row: f0 = 0.0 (<0.5 → tree A left leaf 3.0), f1 = 9.0 (≥0.5 → tree B right leaf -7.0).
    let data = [0.0_f32, 9.0];
    let out = predict(&m, &data, 1, &cfg).unwrap();
    // Shape (num_row=1, num_tree=2, lvs=1) row-major → length 2.
    assert_eq!(out.len(), 2, "length is num_row * num_tree * 1");
    assert_eq!(out[0], 3.0_f32, "tree0 raw leaf, no sigmoid/base");
    assert_eq!(out[1], -7.0_f32, "tree1 raw leaf, no sigmoid/base");
}

#[test]
fn score_per_tree_leaf_vector_writes_each_element() {
    // A single leaf-vector tree of width 3. ScorePerTree third-dim size lvs =
    // leaf_vector_shape[0]*[1] = 1*3 = 3, and each element is written.
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(vec![leaf_vector_tree(
        vec![0.1, 0.2, 0.7],
    )])));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![3];
    m.leaf_vector_shape = vec![1, 3];
    m.target_id = vec![-1];
    m.class_id = vec![-1];
    m.postprocessor = "softmax".to_string(); // must be ignored
    m.base_scores = vec![0.0, 0.0, 0.0];

    let cfg = Config {
        kind: PredictKind::ScorePerTree,
        nthread: 0,
    };
    let data = [0.0_f32];
    let out = predict(&m, &data, 1, &cfg).unwrap();
    // Shape (1, num_tree=1, lvs=3) → length 3.
    assert_eq!(out.len(), 3, "length is num_row * num_tree * 3");
    assert_eq!(out[0], 0.1_f32);
    assert_eq!(out[1], 0.2_f32);
    assert_eq!(out[2], 0.7_f32);
}

#[test]
fn score_per_tree_ignores_postprocessor_unlike_default() {
    // Same model, two kinds. With a sigmoid postprocessor + base score, the
    // Default output is sigmoid(leaf_sum + base); ScorePerTree is the RAW leaf
    // value. They must differ — proving ScorePerTree skips postprocess/base.
    let m = scalar_model(vec![split_tree(0, 0.5, 2.0, -2.0)], 2, "sigmoid", 0.0);
    let data = [0.0_f32, 0.0]; // f0 = 0.0 (<0.5 → left leaf 2.0)

    let default_out = predict(&m, &data, 1, &Config::default()).unwrap();
    let score_out = predict(
        &m,
        &data,
        1,
        &Config {
            kind: PredictKind::ScorePerTree,
            nthread: 0,
        },
    )
    .unwrap();

    // Default applies sigmoid(2.0) ≈ 0.8808; ScorePerTree is the raw 2.0.
    assert_eq!(score_out[0], 2.0_f32, "ScorePerTree is the raw leaf");
    assert!(
        (default_out[0] - 1.0_f32 / (1.0 + (-2.0_f32).exp())).abs() < 1e-6,
        "Default applies sigmoid"
    );
    assert!(
        (score_out[0] - default_out[0]).abs() > 1e-3,
        "ScorePerTree (raw) must differ from Default (postprocessed)"
    );
}

#[test]
fn leaf_id_ignores_base_score_and_averaging() {
    // A single-tree RF-style model with averaging on + non-zero base; LeafId
    // must STILL return the bare integer node id (no /n, no +base).
    let mut t = split_tree(0, 0.5, 1.0, -1.0);
    // make it a leaf-vector-free scalar tree (already is)
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0, 0, 0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0, 0, 0]);
    let mut m = scalar_model(vec![t], 2, "identity", 5.0);
    m.average_tree_output = true;

    let cfg = Config {
        kind: PredictKind::LeafId,
        nthread: 0,
    };
    let data = [9.0_f32, 0.0]; // f0 = 9.0 (≥0.5 → right leaf, node 2)
    let out = predict(&m, &data, 1, &cfg).unwrap();
    assert_eq!(out.len(), 1, "(1 row, 1 tree)");
    assert_eq!(out[0], 2.0_f32, "bare node id, no base/average applied");
}

#[test]
fn all_four_predict_kinds_dispatch_without_error() {
    // Sanity: Default, Raw, LeafId, ScorePerTree all return Ok on a valid model
    // (no remaining UnsupportedPredictKind for LeafId/ScorePerTree).
    let m = scalar_model(vec![split_tree(0, 0.5, 1.0, -1.0)], 2, "identity", 0.0);
    let data = [0.0_f32, 0.0];
    for kind in [
        PredictKind::Default,
        PredictKind::Raw,
        PredictKind::LeafId,
        PredictKind::ScorePerTree,
    ] {
        let cfg = Config { kind, nthread: 0 };
        let out = predict(&m, &data, 1, &cfg);
        assert!(out.is_ok(), "kind {kind:?} must dispatch, got {out:?}");
    }
}

/// f64-input LeafId returns the node ids as f64 (A4 cast into the O buffer).
#[test]
fn leaf_id_f64_input() {
    let m = scalar_model(vec![split_tree(0, 0.5, 1.0, -1.0)], 2, "identity", 0.0);
    let data = [0.0_f64, 0.0]; // f0 = 0.0 (<0.5 → left, node 1)
    let cfg = Config {
        kind: PredictKind::LeafId,
        nthread: 0,
    };
    let out = predict(&m, &data, 1, &cfg).unwrap();
    assert_eq!(out, vec![1.0_f64], "node id 1 cast into f64 buffer");
}

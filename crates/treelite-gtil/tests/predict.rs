//! Tests for the scalar predict engine (Plan 01-03 Task 2).
//!
//! Builds hand-crafted F32 models so the leaf values, threshold, and base
//! score are known, then asserts left/right routing, NaN→default-child
//! routing, serial two-tree summation, sigmoid output in (0,1), and that an
//! out-of-bounds feature index returns a typed `GtilError` (ERR-01).

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf};
use treelite_gtil::{Config, GtilError, predict};

/// Build a single-split `Tree<f32>`:
/// - node 0: numerical test on `feature`, `kLT` threshold, default-left,
/// - node 1: leaf with `left_leaf`,
/// - node 2: leaf with `right_leaf`.
fn split_tree(feature: i32, threshold: f32, left_leaf: f32, right_leaf: f32) -> Tree<f32> {
    use treelite_core::TreeNodeType;
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    // cleft == -1 marks a leaf; node 0 has real children, nodes 1/2 are leaves.
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
    t
}

/// Wrap trees in an F32 `Model` with the given postprocessor and base score.
///
/// The binary scalar shape `(num_row, 1, 1)`: one target, one class, every tree
/// routed to `(target_id=0, class_id=0)`. `target_id`/`class_id` are sized to
/// the tree count so a multi-tree model routes every tree into cell 0 (the
/// serial-sum path), not the leaf-vector broadcast default.
fn model_of(trees: Vec<Tree<f32>>, postprocessor: &str, base_score: f64) -> Model {
    let num_tree = trees.len();
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 2;
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
fn routes_left_and_right_on_numerical_split() {
    // threshold 0.5; left leaf 1.0, right leaf -1.0; identity postprocessor.
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], "identity", 0.0);
    // feature[0] = 0.0 < 0.5 → left leaf (1.0)
    // feature[0] = 1.0 >= 0.5 → right leaf (-1.0)
    let data = [0.0_f32, 9.0, 1.0, 9.0];
    let out = predict(&m, &data, 2, &Config::default()).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], 1.0_f32);
    assert_eq!(out[1], -1.0_f32);
}

/// Build a single-split categorical `Tree<f32>` on `feature` with category list
/// `categories` and the given `category_list_right_child` polarity.
/// - node 0: categorical test, default_left == false,
/// - node 1: leaf with `left_leaf`,
/// - node 2: leaf with `right_leaf`.
fn categorical_tree(
    feature: i32,
    categories: Vec<u32>,
    category_list_right_child: bool,
    left_leaf: f32,
    right_leaf: f32,
) -> Tree<f32> {
    use treelite_core::TreeNodeType;
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![false, false, false]);
    t.threshold = TreeBuf::from_owned(vec![0.0, 0.0, 0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0, left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kCategoricalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    // CSR category list: node 0 has [0, categories.len()); nodes 1/2 empty.
    let n = categories.len() as u64;
    t.category_list = TreeBuf::from_owned(categories);
    t.category_list_begin = TreeBuf::from_owned(vec![0u64, n, n]);
    t.category_list_end = TreeBuf::from_owned(vec![n, n, n]);
    t.category_list_right_child =
        TreeBuf::from_owned(vec![category_list_right_child, false, false]);
    t.has_categorical_split = true;
    t
}

#[test]
fn categorical_match_routes_by_polarity() {
    // category_list_right_child == false ⇒ a category MATCH routes LEFT.
    // categories = {1, 2}; left leaf 10.0, right leaf 20.0.
    let m = model_of(
        vec![categorical_tree(0, vec![1, 2], false, 10.0, 20.0)],
        "identity",
        0.0,
    );
    // feature[0] = 2.0 IS in {1,2} → match → LEFT (10.0).
    // feature[0] = 3.0 NOT in {1,2} → non-match → RIGHT (20.0).
    let data = [2.0_f32, 9.0, 3.0, 9.0];
    let out = predict(&m, &data, 2, &Config::default()).unwrap();
    assert_eq!(out[0], 10.0_f32, "category 2 in list → left");
    assert_eq!(out[1], 20.0_f32, "category 3 not in list → right");
}

#[test]
fn categorical_right_child_polarity_inverts_routing() {
    // category_list_right_child == true ⇒ a MATCH routes RIGHT.
    let m = model_of(
        vec![categorical_tree(0, vec![1, 2], true, 10.0, 20.0)],
        "identity",
        0.0,
    );
    // feature[0] = 1.0 IS in {1,2} → match → RIGHT (20.0).
    // feature[0] = 5.0 NOT in {1,2} → non-match → LEFT (10.0).
    let data = [1.0_f32, 9.0, 5.0, 9.0];
    let out = predict(&m, &data, 2, &Config::default()).unwrap();
    assert_eq!(
        out[0], 20.0_f32,
        "category 1 in list, right-child polarity → right"
    );
    assert_eq!(
        out[1], 10.0_f32,
        "category 5 not in list, right-child polarity → left"
    );
}

#[test]
fn nan_routes_to_default_child() {
    // default_left == true → NaN routes to the LEFT leaf (1.0), regardless of
    // the threshold comparison.
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], "identity", 0.0);
    let data = [f32::NAN, 9.0];
    let out = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(out[0], 1.0_f32);
}

#[test]
fn two_trees_sum_serially_before_base_and_sigmoid() {
    // Tree A: feature 0 < 0.5 → 1.0 else -1.0
    // Tree B: feature 1 < 0.5 → 0.5 else -0.5
    // base_score 0.0, sigmoid postprocessor.
    let m = model_of(
        vec![split_tree(0, 0.5, 1.0, -1.0), split_tree(1, 0.5, 0.5, -0.5)],
        "sigmoid",
        0.0,
    );
    // row: feature0 = 0.0 (<0.5 → +1.0), feature1 = 0.0 (<0.5 → +0.5) ⇒ margin 1.5
    let data = [0.0_f32, 0.0];
    let out = predict(&m, &data, 1, &Config::default()).unwrap();
    let expected = 1.0_f32 / (1.0_f32 + (-1.5_f32).exp());
    assert!(
        (out[0] - expected).abs() < 1e-7,
        "got {}, want {expected}",
        out[0]
    );
    assert!(out[0] > 0.0 && out[0] < 1.0);
}

#[test]
fn base_score_is_added_before_postprocessor() {
    // Single tree always lands on leaf 0.0; base_score 0.25 → margin 0.25;
    // identity postprocessor returns the margin unchanged.
    let m = model_of(vec![split_tree(0, 0.5, 0.0, 0.0)], "identity", 0.25);
    let data = [0.0_f32, 0.0];
    let out = predict(&m, &data, 1, &Config::default()).unwrap();
    assert!((out[0] - 0.25_f32).abs() < 1e-7);
}

#[test]
fn sigmoid_output_in_open_unit_interval() {
    let m = model_of(vec![split_tree(0, 0.5, 5.0, -5.0)], "sigmoid", 0.0);
    let data = [0.0_f32, 0.0, 1.0, 0.0]; // two rows: left then right
    let out = predict(&m, &data, 2, &Config::default()).unwrap();
    for &v in &out {
        assert!(v > 0.0 && v < 1.0, "sigmoid output {v} not in (0,1)");
    }
}

#[test]
fn out_of_bounds_feature_index_is_typed_error() {
    // split on feature 5, but num_feature is only 2 → OOB, must not panic.
    let m = model_of(vec![split_tree(5, 0.5, 1.0, -1.0)], "identity", 0.0);
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    match err {
        GtilError::FeatureIndexOutOfBounds {
            node,
            feature,
            num_feature,
        } => {
            assert_eq!(node, 0);
            assert_eq!(feature, 5);
            assert_eq!(num_feature, 2);
        }
        other => panic!("expected FeatureIndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn input_buffer_too_small_is_typed_error() {
    // model.num_feature == 2 (set by model_of), so 2 rows need 4 elements; only
    // 3 are supplied. predict must return InvalidInputShape, not panic on the
    // row slice (WR-01 / T-03-01).
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], "identity", 0.0);
    let data = [0.0_f32, 0.0, 0.0]; // 3 elements, need 4
    let err = predict(&m, &data, 2, &Config::default()).unwrap_err();
    match err {
        GtilError::InvalidInputShape {
            num_row,
            num_feature,
            required,
            got,
        } => {
            assert_eq!(num_row, 2);
            assert_eq!(num_feature, 2);
            assert_eq!(required, 4);
            assert_eq!(got, 3);
        }
        other => panic!("expected InvalidInputShape, got {other:?}"),
    }
}

#[test]
fn negative_num_feature_is_typed_error() {
    // A malformed model with num_feature < 0 must not abort via a usize cast
    // (WR-02 gtil-side guard); predict returns InvalidInputShape.
    let mut m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], "identity", 0.0);
    m.num_feature = -1;
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert!(matches!(err, GtilError::InvalidInputShape { .. }));
}

#[test]
fn out_of_bounds_child_node_id_is_typed_error() {
    // node 0 routes left to node 99, which does not exist in a 3-node tree.
    // Traversal must return NodeIndexOutOfBounds, not panic (CR-01 / T-03-01).
    let mut t = split_tree(0, 0.5, 1.0, -1.0);
    t.cleft = TreeBuf::from_owned(vec![99, -1, -1]);
    let m = model_of(vec![t], "identity", 0.0);
    // feature[0] = 0.0 < 0.5 → follow cleft[0] == 99 (out of range).
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    match err {
        GtilError::NodeIndexOutOfBounds { node } => assert_eq!(node, 99),
        other => panic!("expected NodeIndexOutOfBounds, got {other:?}"),
    }
}

#[test]
fn negative_child_node_id_other_than_leaf_sentinel_is_typed_error() {
    // node 0 routes RIGHT to node -2 (only -1 is the leaf sentinel). The `-2`
    // must become a typed error rather than `(-2 as usize) == usize::MAX`
    // indexing out of bounds (CR-01 / T-03-01).
    let mut t = split_tree(0, 0.5, 1.0, -1.0);
    t.cright = TreeBuf::from_owned(vec![-2, -1, -1]);
    let m = model_of(vec![t], "identity", 0.0);
    // feature[0] = 1.0 >= 0.5 → follow cright[0] == -2.
    let data = [1.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert!(matches!(err, GtilError::NodeIndexOutOfBounds { .. }));
}

#[test]
fn unsupported_postprocessor_is_typed_error() {
    // `hinge` is a real upstream postprocessor not yet ported (deferred to
    // Phase 5); it must surface as a typed error, not panic. (Plan 04-02 added
    // softmax/exponential/exp_standard_ratio/log1p_exp, so those are now
    // supported and no longer valid "unsupported" probes.)
    let m = model_of(vec![split_tree(0, 0.5, 1.0, -1.0)], "hinge", 0.0);
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(
        err,
        GtilError::UnsupportedPostprocessor("hinge".to_string())
    );
}

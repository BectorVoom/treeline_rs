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
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0; num_tree].into();
    m.class_id = vec![0; num_tree].into();
    m.postprocessor = postprocessor.to_string().into();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![base_score].into();
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
    // A genuinely-unknown postprocessor name must surface as a typed error, not
    // panic. (All 10 real upstream postprocessors are now ported — Plan 05-03
    // added the last three: signed_square/hinge/multiclass_ova — so only a name
    // that is not in the upstream set is a valid "unsupported" probe.)
    let m = model_of(
        vec![split_tree(0, 0.5, 1.0, -1.0)],
        "not_a_real_postprocessor",
        0.0,
    );
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(
        err,
        GtilError::UnsupportedPostprocessor("not_a_real_postprocessor".to_string())
    );
}

// ------------------------------------------------------------------------- //
// WR-05 (Plan 05-06): kNone on a NUMERICAL test node returns a typed
// UnrecognizedOperator, matching upstream's TREELITE_CHECK(false) fatal path,
// instead of silently routing right (a definite wrong prediction).
// ------------------------------------------------------------------------- //

#[test]
fn knone_operator_on_numerical_node_is_typed_error() {
    use treelite_core::TreeNodeType;
    // A numerical-test node whose comparison operator is kNone (never emitted by
    // a well-formed loader). Before WR-05 this routed right silently; now it is
    // GtilError::UnrecognizedOperator { node: 0, op: kNone }.
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![0, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![0.5, 0.0, 0.0]);
    // kNone on the NUMERICAL test node 0 (the malformed case).
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0, 1.0, -1.0]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    let m = model_of(vec![t], "identity", 0.0);
    // A non-NaN feature value reaches the numerical-test branch → next_node(kNone).
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(
        err,
        GtilError::UnrecognizedOperator {
            node: 0,
            op: Operator::kNone,
        },
        "kNone on a numerical node must be a typed error, not a silent route-right"
    );
}

// ------------------------------------------------------------------------- //
// WR-04 (Plan 05-06): malformed category-list / leaf-vector CSR offsets return
// a typed error instead of a silent &[] / scalar fallthrough that changes the
// prediction. Legitimately-empty lists and absent (scalar-leaf) offsets keep
// their correct non-error behavior.
// ------------------------------------------------------------------------- //

/// A categorical tree whose node-0 category-list offsets are explicitly set, so
/// a test can inject an inverted (`begin > end`) or out-of-range (`end > len`)
/// slice. Node 0 categorical, nodes 1/2 leaves.
fn categorical_tree_with_offsets(
    categories: Vec<u32>,
    begin0: u64,
    end0: u64,
    left_leaf: f32,
    right_leaf: f32,
) -> Tree<f32> {
    use treelite_core::TreeNodeType;
    let mut t = Tree::<f32>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![0, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![false, false, false]);
    t.threshold = TreeBuf::from_owned(vec![0.0, 0.0, 0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![0.0, left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kCategoricalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t.category_list = TreeBuf::from_owned(categories);
    // Nodes 1/2 carry legitimately-empty in-bounds offsets (begin==end==0).
    t.category_list_begin = TreeBuf::from_owned(vec![begin0, 0, 0]);
    t.category_list_end = TreeBuf::from_owned(vec![end0, 0, 0]);
    t.category_list_right_child = TreeBuf::from_owned(vec![false, false, false]);
    t.has_categorical_split = true;
    t
}

#[test]
fn malformed_category_list_inverted_offsets_is_typed_error() {
    // begin (2) > end (1) on node 0 → MalformedCategoryList, not a silent &[].
    let m = model_of(
        vec![categorical_tree_with_offsets(
            vec![1, 2, 3],
            2,
            1,
            10.0,
            20.0,
        )],
        "identity",
        0.0,
    );
    let data = [1.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(err, GtilError::MalformedCategoryList { node: 0 });
}

#[test]
fn malformed_category_list_out_of_range_end_is_typed_error() {
    // end (9) > category_list.len() (3) on node 0 → MalformedCategoryList.
    let m = model_of(
        vec![categorical_tree_with_offsets(
            vec![1, 2, 3],
            0,
            9,
            10.0,
            20.0,
        )],
        "identity",
        0.0,
    );
    let data = [1.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(err, GtilError::MalformedCategoryList { node: 0 });
}

#[test]
fn legitimately_empty_category_list_is_not_an_error() {
    // begin == end == 0 (in bounds) on node 0 → Ok(&[]): every category is a
    // non-match, so routing falls to the non-match side (category_list_right_child
    // == false ⇒ non-match routes RIGHT). NOT an error.
    let m = model_of(
        vec![categorical_tree_with_offsets(
            vec![1, 2, 3],
            0,
            0,
            10.0,
            20.0,
        )],
        "identity",
        0.0,
    );
    let data = [1.0_f32, 0.0]; // 1 would match if list were [1..], but list is empty → non-match → right
    let out = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(
        out[0], 20.0_f32,
        "empty list → non-match → right leaf, no error"
    );
}

#[test]
fn malformed_leaf_vector_inverted_offsets_is_typed_error() {
    use treelite_core::TreeNodeType;
    // A single leaf node whose leaf-vector offsets are inverted (begin 1 > end 0)
    // → MalformedLeafVector, NOT a silent scalar fallthrough.
    let mut t = Tree::<f32>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![5.0]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    t.leaf_vector = TreeBuf::from_owned(vec![0.1_f32]); // len 1
    t.leaf_vector_begin = TreeBuf::from_owned(vec![1u64]); // begin > end
    t.leaf_vector_end = TreeBuf::from_owned(vec![0u64]);
    let m = model_of(vec![t], "identity", 0.0);
    let data = [0.0_f32, 0.0];
    let err = predict(&m, &data, 1, &Config::default()).unwrap_err();
    assert_eq!(err, GtilError::MalformedLeafVector { node: 0 });
}

#[test]
fn absent_leaf_vector_offsets_stay_scalar_no_error() {
    use treelite_core::TreeNodeType;
    // A scalar leaf with EMPTY leaf-vector CSR columns (absent offsets) must take
    // the legitimate scalar path → Ok(false), NOT an error. (The split_tree
    // helper leaves leaf_vector_begin/end empty by default.)
    let mut t = Tree::<f32>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![7.0]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    // leaf_vector_begin / leaf_vector_end left EMPTY (absent offsets).
    let m = model_of(vec![t], "identity", 0.0);
    let data = [0.0_f32, 0.0];
    let out = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(
        out[0], 7.0_f32,
        "absent leaf-vector offsets → scalar leaf, no error"
    );
}

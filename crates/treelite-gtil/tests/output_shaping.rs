//! Tests for the widened multiclass / leaf-vector output shaping (Plan 04-02
//! Task 2).
//!
//! Builds hand-crafted models whose `target_id`/`class_id`/`leaf_vector_shape`
//! are known, then asserts:
//! - round-robin `class_id[tree] = tree % n_class` routes each tree's scalar
//!   leaf into the correct class column of a `(num_row, 1, n_class)` output
//!   (no collapse to column 0);
//! - a leaf-vector model with `target_id == -1, class_id == -1` broadcasts the
//!   leaf vector across the `(num_row, num_target, max_num_class)` buffer (RF);
//! - `average_tree_output == true` returns the mean over trees, not the sum;
//! - f64 `base_scores[target, class]` is added per cell with the
//!   `(acc as f64 + base) as f32` promotion;
//! - the scalar binary `(num_row, 1, 1)` path is unchanged (degenerate shape).

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};

/// A single-node leaf tree whose only node (id 0) returns `leaf`.
fn leaf_tree(leaf: f32) -> Tree<f32> {
    let mut t = Tree::<f32>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![0.0]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![leaf]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    // CSR offset columns sized to num_nodes, begin == end ⇒ no leaf vector.
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0]);
    t
}

/// A single-node leaf tree whose only node (id 0) returns a leaf VECTOR `vec`.
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

#[test]
fn round_robin_class_routing_places_trees_in_distinct_columns() {
    // 2-class model, scalar leaves, class_id[tree] = tree % 2.
    // Tree 0 → class 0 leaf 1.0; Tree 1 → class 1 leaf 2.0;
    // Tree 2 → class 0 leaf 0.5; Tree 3 → class 1 leaf 0.25.
    let trees = vec![
        leaf_tree(1.0),
        leaf_tree(2.0),
        leaf_tree(0.5),
        leaf_tree(0.25),
    ];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![2].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0, 0, 0, 0].into();
    m.class_id = vec![0, 1, 0, 1].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0, 0.0].into();

    let data = [0.0_f32]; // one row, one feature (unused; all leaves at node 0)
    let out = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap();
    assert_eq!(out.len(), 2, "shape is (1, 1, 2)");
    // class 0 = 1.0 + 0.5; class 1 = 2.0 + 0.25. No collapse to column 0.
    assert!((out[0] - 1.5).abs() < 1e-6, "class 0 = {}", out[0]);
    assert!((out[1] - 2.25).abs() < 1e-6, "class 1 = {}", out[1]);
}

#[test]
fn leaf_vector_broadcast_fills_full_shape() {
    // RF leaf-vector model: target_id == -1, class_id == -1, leaf vector of
    // shape (num_target=1, max_num_class=3) broadcast across all cells.
    let trees = vec![leaf_vector_tree(vec![0.1, 0.2, 0.7])];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![3].into();
    m.leaf_vector_shape = vec![1, 3].into();
    m.target_id = vec![-1].into();
    m.class_id = vec![-1].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0, 0.0, 0.0].into();

    let data = [0.0_f32];
    let out = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap();
    assert_eq!(out.len(), 3, "shape is (1, 1, 3)");
    assert!((out[0] - 0.1).abs() < 1e-6);
    assert!((out[1] - 0.2).abs() < 1e-6);
    assert!((out[2] - 0.7).abs() < 1e-6);
}

#[test]
fn average_tree_output_returns_mean_not_sum() {
    // 3 RF trees, all routing to a single class via class_id == -1 broadcast;
    // scalar leaves 1.0, 2.0, 3.0. With averaging, cell = (1+2+3)/3 = 2.0,
    // NOT the sum 6.0 (off-by-n_estimators factor must be absent).
    let trees = vec![
        leaf_vector_tree(vec![1.0]),
        leaf_vector_tree(vec![2.0]),
        leaf_vector_tree(vec![3.0]),
    ];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![-1, -1, -1].into();
    m.class_id = vec![-1, -1, -1].into();
    m.average_tree_output = true;
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0].into();

    let data = [0.0_f32];
    let out = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    assert!(
        (out[0] - 2.0).abs() < 1e-6,
        "mean expected 2.0, got {}",
        out[0]
    );
}

#[test]
fn base_scores_added_per_cell_2d() {
    // 2-class scalar model, base_scores[class0] = 0.25, base_scores[class1] = -0.5.
    let trees = vec![leaf_tree(1.0), leaf_tree(2.0)];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![2].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0, 0].into();
    m.class_id = vec![0, 1].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.25, -0.5].into();

    let data = [0.0_f32];
    let out = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap();
    // class 0 = 1.0 + 0.25; class 1 = 2.0 - 0.5.
    assert!((out[0] - 1.25).abs() < 1e-6, "class 0 = {}", out[0]);
    assert!((out[1] - 1.5).abs() < 1e-6, "class 1 = {}", out[1]);
}

#[test]
fn scalar_binary_path_shape_unchanged() {
    // Degenerate (num_row, 1, 1) path: one scalar tree, num_class=[1],
    // target_id=[0], class_id=[0]. Output length == num_row, value == leaf + base.
    let trees = vec![leaf_tree(0.75)];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0].into();
    m.class_id = vec![0].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.25].into();

    let data = [0.0_f32, 0.0]; // two rows
    let out = treelite_gtil::predict(&m, &data, 2, &treelite_gtil::Config::default()).unwrap();
    assert_eq!(out.len(), 2, "binary scalar shape is (num_row, 1, 1)");
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 1.0).abs() < 1e-6);
}

#[test]
fn softmax_normalizes_multiclass_row() {
    // 3-class scalar model, margins [1.0, 2.0, 3.0] via round-robin; softmax
    // postprocessor over the class axis ⇒ probabilities summing to ~1.0.
    let trees = vec![leaf_tree(1.0), leaf_tree(2.0), leaf_tree(3.0)];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![3].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0, 0, 0].into();
    m.class_id = vec![0, 1, 2].into();
    m.postprocessor = "softmax".to_string().into();
    m.base_scores = vec![0.0, 0.0, 0.0].into();

    let data = [0.0_f32];
    let out = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap();
    assert_eq!(out.len(), 3);
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax row sums to {sum}");
    // Monotonic: class 2 (margin 3) > class 1 (margin 2) > class 0 (margin 1).
    assert!(out[2] > out[1] && out[1] > out[0]);
}

#[test]
fn out_of_range_class_route_is_typed_error() {
    // class_id[0] = 5 but num_class = [2] (max_num_class = 2). A scalar leaf
    // routed out of range must surface as OutputRouteOutOfBounds, never panic
    // (T-04-03).
    let trees = vec![leaf_tree(1.0)];
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![2].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0].into();
    m.class_id = vec![5].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0, 0.0].into();

    let data = [0.0_f32];
    let err = treelite_gtil::predict(&m, &data, 1, &treelite_gtil::Config::default()).unwrap_err();
    match err {
        treelite_gtil::GtilError::OutputRouteOutOfBounds {
            target_id,
            class_id,
            num_target,
            max_num_class,
        } => {
            assert_eq!(target_id, 0);
            assert_eq!(class_id, 5);
            assert_eq!(num_target, 1);
            assert_eq!(max_num_class, 2);
        }
        other => panic!("expected OutputRouteOutOfBounds, got {other:?}"),
    }
}

//! `Tree<T>` SoA getter tests (CORE-02) and two-variant `Model` + header
//! metadata tests (CORE-01, CORE-04).

use treelite_core::{
    Model, ModelPreset, ModelVariant, Operator, TaskType, Tree, TreeBuf, TreeNodeType,
};

#[test]
fn tree_leaf_detection() {
    // A single leaf node: cleft == -1.
    let mut tree: Tree<f32> = Tree::new();
    tree.cleft = TreeBuf::from_owned(vec![-1]);
    tree.cright = TreeBuf::from_owned(vec![-1]);
    tree.leaf_value = TreeBuf::from_owned(vec![0.5]);
    tree.num_nodes = 1;
    assert!(tree.is_leaf(0));
    assert_eq!(tree.leaf_value(0), 0.5);
}

#[test]
fn tree_default_child_left() {
    // Internal node: cleft=1, cright=2, default_left=true => default child is 1.
    let mut tree: Tree<f32> = Tree::new();
    tree.cleft = TreeBuf::from_owned(vec![1]);
    tree.cright = TreeBuf::from_owned(vec![2]);
    tree.default_left = TreeBuf::from_owned(vec![true]);
    tree.split_index = TreeBuf::from_owned(vec![0]);
    tree.threshold = TreeBuf::from_owned(vec![1.5]);
    tree.cmp = TreeBuf::from_owned(vec![Operator::kLT]);
    tree.node_type = TreeBuf::from_owned(vec![TreeNodeType::kNumericalTestNode]);
    assert!(!tree.is_leaf(0));
    assert_eq!(tree.default_child(0), 1);
    assert_eq!(tree.left_child(0), 1);
    assert_eq!(tree.right_child(0), 2);
    assert_eq!(tree.split_index(0), 0);
    assert_eq!(tree.threshold(0), 1.5);
    assert_eq!(tree.comparison_op(0), Operator::kLT);
}

#[test]
fn tree_default_child_right() {
    let mut tree: Tree<f32> = Tree::new();
    tree.cleft = TreeBuf::from_owned(vec![1]);
    tree.cright = TreeBuf::from_owned(vec![2]);
    tree.default_left = TreeBuf::from_owned(vec![false]);
    assert_eq!(tree.default_child(0), 2);
}

#[test]
fn tree_has_leaf_vector_false_when_begin_equals_end() {
    let mut tree: Tree<f32> = Tree::new();
    tree.leaf_vector_begin = TreeBuf::from_owned(vec![0u64]);
    tree.leaf_vector_end = TreeBuf::from_owned(vec![0u64]);
    assert!(!tree.has_leaf_vector(0));
}

#[test]
fn tree_has_leaf_vector_true_when_begin_differs_from_end() {
    let mut tree: Tree<f32> = Tree::new();
    tree.leaf_vector_begin = TreeBuf::from_owned(vec![0u64]);
    tree.leaf_vector_end = TreeBuf::from_owned(vec![2u64]);
    assert!(tree.has_leaf_vector(0));
}

#[test]
fn model_two_variants_both_constructible() {
    let f32_preset = ModelPreset::<f32>::new(vec![Tree::new()]);
    let f64_preset = ModelPreset::<f64>::new(vec![Tree::new()]);

    let m32 = Model::new(ModelVariant::F32(f32_preset));
    let m64 = Model::new(ModelVariant::F64(f64_preset));

    match m32.variant {
        ModelVariant::F32(p) => assert_eq!(p.num_trees(), 1),
        ModelVariant::F64(_) => panic!("expected F32 variant"),
    }
    match m64.variant {
        ModelVariant::F64(p) => assert_eq!(p.num_trees(), 1),
        ModelVariant::F32(_) => panic!("expected F64 variant"),
    }
}

#[test]
fn model_header_metadata_array_typed_round_trip() {
    let mut model = Model::new(ModelVariant::F32(ModelPreset::<f32>::new(
        vec![Tree::new()],
    )));
    model.num_feature = 2;
    model.task_type = TaskType::kBinaryClf;
    model.average_tree_output = false;
    model.num_target = 1;
    model.num_class = vec![1].into();
    model.leaf_vector_shape = vec![1, 1].into();
    model.target_id = vec![0].into();
    model.class_id = vec![0].into();
    model.postprocessor = "sigmoid".into();
    model.base_scores = vec![0.0f64].into();

    assert_eq!(model.num_feature, 2);
    assert_eq!(model.task_type, TaskType::kBinaryClf);
    assert!(!model.average_tree_output);
    assert_eq!(model.num_target, 1);
    // Array-typed (SmallVec<[i32; N]>), not scalar — compare via the slice deref.
    assert_eq!(model.num_class.as_slice(), &[1]);
    assert_eq!(model.leaf_vector_shape.as_slice(), &[1, 1]);
    assert_eq!(model.target_id.as_slice(), &[0]);
    assert_eq!(model.class_id.as_slice(), &[0]);
    assert_eq!(model.postprocessor, "sigmoid");
    assert_eq!(model.base_scores.as_slice(), &[0.0f64]);
    // Defaults.
    assert_eq!(model.sigmoid_alpha, 1.0);
    assert_eq!(model.ratio_c, 1.0);
}

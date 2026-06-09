//! Zero-copy PyBuffer frame walk (SER-02, D-06).
//!
//! Proves (1) the frame list is in binary field order with the expected count,
//! and (2) each ARRAY frame borrows DIRECTLY into its owning `TreeBuf` column —
//! the frame's `.as_ptr()` equals the column's `.as_ptr()` (no copy), reusing
//! the `tree_buf.rs:22-31` zero-copy assertion pattern.

use treelite_core::{
    Frame, Model, ModelPreset, ModelVariant, Operator, TaskType, Tree, TreeBuf, TreeNodeType,
    serialize_to_pybuffer,
};

fn sample_model() -> Model {
    let mut tree = Tree::<f32>::new();
    tree.num_nodes = 3;
    tree.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    tree.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    tree.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    tree.split_index = TreeBuf::from_owned(vec![0, -1, -1]);
    tree.default_left = TreeBuf::from_owned(vec![true, false, false]);
    tree.leaf_value = TreeBuf::from_owned(vec![0.0f32, 1.0, -1.5]);
    tree.threshold = TreeBuf::from_owned(vec![0.5f32, 0.0, 0.0]);
    tree.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    tree.category_list_right_child = TreeBuf::from_owned(vec![false, false, false]);
    tree.leaf_vector_begin = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.leaf_vector_end = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.category_list_begin = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.category_list_end = TreeBuf::from_owned(vec![0u64, 0, 0]);
    tree.sum_hess = TreeBuf::from_owned(vec![6.0f64, 3.0, 3.0]);
    tree.sum_hess_present = TreeBuf::from_owned(vec![true, true, true]);
    tree.gain = TreeBuf::from_owned(vec![10.0f64, 0.0, 0.0]);
    tree.gain_present = TreeBuf::from_owned(vec![true, false, false]);

    let mut model = Model::new(ModelVariant::F32(ModelPreset::new(vec![tree])));
    model.num_feature = 2;
    model.task_type = TaskType::kBinaryClf;
    model.num_target = 1;
    model.num_class = vec![1];
    model.leaf_vector_shape = vec![1, 1];
    model.target_id = vec![0];
    model.class_id = vec![0];
    model.postprocessor = "sigmoid".to_string();
    model.base_scores = vec![-1.0986122886681098];
    model.attributes = "{}".to_string();
    model
}

#[test]
fn frame_count_and_order_match_the_binary_walk() {
    let mut model = sample_model();
    let frames = serialize_to_pybuffer(&mut model);

    // 20 header frames + 25 per-tree frames × 1 tree = 45.
    assert_eq!(
        frames.len(),
        20 + 25,
        "frame count must match the field walk"
    );

    // Spot-check the leading header frames are in order (version triple first).
    assert!(matches!(frames[0], Frame::I32(s) if s == [4]));
    assert!(matches!(frames[1], Frame::I32(s) if s == [7]));
    assert!(matches!(frames[2], Frame::I32(s) if s == [0]));
    // Type tags are 1-byte U8 frames (=2 ⇒ float32).
    assert!(matches!(frames[3], Frame::U8(s) if s == [2]));
    assert!(matches!(frames[4], Frame::U8(s) if s == [2]));
    // num_tree is a single-element U64 frame.
    assert!(matches!(frames[5], Frame::U64(s) if s == [1]));
    // postprocessor string frame at header index 14.
    assert!(matches!(frames[14], Frame::Str(s) if s == "sigmoid"));
    // attributes string frame at header index 18.
    assert!(matches!(frames[18], Frame::Str(s) if s == "{}"));
}

#[test]
fn array_frames_are_zero_copy_views_of_their_columns() {
    let mut model = sample_model();

    // Capture the owning columns' pointers BEFORE borrowing the frames.
    let (cleft_ptr, leaf_ptr, sum_hess_ptr, gain_ptr) = match &model.variant {
        ModelVariant::F32(p) => {
            let t = &p.trees[0];
            (
                t.cleft.as_slice().as_ptr(),
                t.leaf_value.as_slice().as_ptr(),
                t.sum_hess.as_slice().as_ptr(),
                t.gain.as_slice().as_ptr(),
            )
        }
        ModelVariant::F64(_) => unreachable!(),
    };

    let frames = serialize_to_pybuffer(&mut model);

    // Tree frames start at index 20. Per-tree order (0-based within the tree):
    //   0 num_nodes, 1 has_cat, 2 node_type, 3 cleft, ...,
    //   7 leaf_value, ..., 19 sum_hess, ..., 21 gain.
    let base = 20;
    // cleft (tree field #4, index 3) — zero-copy i32 column.
    match frames[base + 3] {
        Frame::I32(s) => assert_eq!(
            s.as_ptr(),
            cleft_ptr,
            "cleft frame must alias the owning column (zero-copy)"
        ),
        ref other => panic!("expected I32 cleft frame, got {other:?}"),
    }
    // leaf_value (tree field #8, index 7) — zero-copy f32 column.
    match frames[base + 7] {
        Frame::F32(s) => assert_eq!(s.as_ptr(), leaf_ptr, "leaf_value frame must be zero-copy"),
        ref other => panic!("expected F32 leaf_value frame, got {other:?}"),
    }
    // sum_hess (tree field #20, index 19) — zero-copy f64 column.
    match frames[base + 19] {
        Frame::F64(s) => assert_eq!(s.as_ptr(), sum_hess_ptr, "sum_hess frame must be zero-copy"),
        ref other => panic!("expected F64 sum_hess frame, got {other:?}"),
    }
    // gain (tree field #22, index 21) — zero-copy f64 column.
    match frames[base + 21] {
        Frame::F64(s) => assert_eq!(s.as_ptr(), gain_ptr, "gain frame must be zero-copy"),
        ref other => panic!("expected F64 gain frame, got {other:?}"),
    }
}

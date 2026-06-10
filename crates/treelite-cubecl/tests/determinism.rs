//! Wave 4 (plan 06-05) — SC2 two-run bit-identical determinism.
//!
//! Two runs of the SAME cubecl CPU-backend prediction over the SAME input must
//! produce bit-identical output (`f64::to_bits()`/`f32::to_bits()` equality).
//! This is the SC2 contract for the CPU reference backend: the kernels write
//! disjoint per-row cells with NO cross-unit accumulation over the tree axis
//! (plan 06-04's grep gate), so there is no run-to-run reordering of a
//! floating-point reduction — determinism holds structurally, and this test
//! proves it observationally.
//!
//! `.to_bits()` (not `==`) is the assertion: it distinguishes `+0.0`/`-0.0` and
//! makes a `NaN` payload difference a failure, so a non-deterministic
//! accumulation that happened to land on an equal-but-different bit pattern
//! cannot masquerade as a pass (T-06-13).

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::predict_cpu;
use treelite_gtil::{Config, PredictKind};

/// Single-split numerical `Tree<T>` (node 0 `kLT` default-left; node 1/2 leaves).
/// Mirrors the `split_tree` fixture in `tests/predict_kinds.rs` so the
/// determinism run exercises the same dense-numerical kernel path the parity
/// tests cover.
fn split_tree<T: Copy + Default>(feature: i32, threshold: T, left: T, right: T) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![feature, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, T::default(), T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default(), left, right]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0, 0, 0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0, 0, 0]);
    t
}

/// Binary scalar `(num_row, 1, 1)` model over `trees` (sigmoid postprocessor).
fn scalar_model<T, W>(trees: Vec<Tree<T>>, wrap: W, num_feature: i32) -> Model
where
    T: Copy,
    W: Fn(ModelPreset<T>) -> ModelVariant,
{
    let num_tree = trees.len();
    let mut m = Model::new(wrap(ModelPreset::new(trees)));
    m.num_feature = num_feature;
    m.num_target = 1;
    m.num_class = vec![1];
    m.leaf_vector_shape = vec![1, 1];
    m.target_id = vec![0; num_tree];
    m.class_id = vec![0; num_tree];
    m.postprocessor = "sigmoid".to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![0.5];
    m
}

/// SC2 (f64): two `predict_cpu::<f64>` runs over identical input are
/// element-wise bit-identical.
#[test]
fn determinism_two_run_bit_identity_f64() {
    let trees = vec![
        split_tree::<f64>(0, 0.5, 1.0, -1.0),
        split_tree::<f64>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F64, 2);
    // Several rows exercising both branches of both trees.
    let data: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let num_row = 4;

    for kind in [
        PredictKind::Default,
        PredictKind::Raw,
        PredictKind::LeafId,
        PredictKind::ScorePerTree,
    ] {
        let cfg = Config { kind, nthread: 0 };
        let a = predict_cpu::<f64>(&model, &data, num_row, &cfg).unwrap();
        let b = predict_cpu::<f64>(&model, &data, num_row, &cfg).unwrap();
        assert_eq!(a.len(), b.len(), "{kind:?}: f64 length differs across runs");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{kind:?}: f64 cell {i} not bit-identical across two runs (SC2): \
                 {x} (bits {:#x}) vs {y} (bits {:#x})",
                x.to_bits(),
                y.to_bits(),
            );
        }
    }
}

/// SC2 (f32): two `predict_cpu::<f32>` runs over identical input are
/// element-wise bit-identical (the input-dtype axis is also deterministic).
#[test]
fn determinism_two_run_bit_identity_f32() {
    let trees = vec![
        split_tree::<f32>(0, 0.5, 1.0, -1.0),
        split_tree::<f32>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F32, 2);
    let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let num_row = 4;

    for kind in [
        PredictKind::Default,
        PredictKind::Raw,
        PredictKind::LeafId,
        PredictKind::ScorePerTree,
    ] {
        let cfg = Config { kind, nthread: 0 };
        let a = predict_cpu::<f32>(&model, &data, num_row, &cfg).unwrap();
        let b = predict_cpu::<f32>(&model, &data, num_row, &cfg).unwrap();
        assert_eq!(a.len(), b.len(), "{kind:?}: f32 length differs across runs");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{kind:?}: f32 cell {i} not bit-identical across two runs (SC2): \
                 {x} (bits {:#x}) vs {y} (bits {:#x})",
                x.to_bits(),
                y.to_bits(),
            );
        }
    }
}

//! Wave 3 (plan 06-04) — `predict_cpu` host launcher parity across all four
//! predict kinds + multiclass leaf-vector broadcast, f32 AND f64 input, vs the
//! scalar `treelite_gtil::predict` reference within 1e-5.
//!
//! What this asserts:
//! - `Default` / `Raw` / `LeafId` / `ScorePerTree` each run end-to-end on the
//!   cubecl CPU kernels and match `treelite_gtil::predict` element-wise to 1e-5
//!   on a numerical-dense model, including a multiclass leaf-vector-broadcast
//!   model (target_id == class_id == -1, the 4-way OutputLeafVector branch);
//! - the parity holds on f32 input (F32 preset) AND f64 input (F64 preset)
//!   (Pitfall 6: the output element equals the INPUT dtype);
//! - a model with a categorical split routes the WHOLE model to the scalar
//!   fallback and still lands within 1e-5 (D-02);
//! - a malformed input shape returns a typed `CubeclError` BEFORE any device op
//!   (no panic, T-06-09).

use approx::assert_abs_diff_eq;

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_cubecl::{CubeclError, predict_cpu};
use treelite_gtil::{Config, PredictKind, predict};

// --------------------------------------------------------------------------- //
// Fixtures (mirror crates/treelite-gtil/tests/predict_kinds.rs so the cubecl
// path is checked against the SAME models the scalar reference is built on).
// --------------------------------------------------------------------------- //

/// Single-split `Tree<T>`: node 0 numerical `kLT` (default-left), node 1 = left
/// leaf, node 2 = right leaf. leaf_vector CSR offsets sized to num_nodes,
/// begin == end ⇒ scalar leaves.
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

/// A single-node leaf-VECTOR tree (id 0) returning `vec` (broadcast across the
/// `(num_target, max_num_class)` cells).
fn leaf_vector_tree<T: Copy + Default>(vec: Vec<T>) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 1;
    t.cleft = TreeBuf::from_owned(vec![-1]);
    t.cright = TreeBuf::from_owned(vec![-1]);
    t.split_index = TreeBuf::from_owned(vec![-1]);
    t.default_left = TreeBuf::from_owned(vec![false]);
    t.threshold = TreeBuf::from_owned(vec![T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default()]);
    t.node_type = TreeBuf::from_owned(vec![TreeNodeType::kLeafNode]);
    let n = vec.len() as u64;
    t.leaf_vector = TreeBuf::from_owned(vec);
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![n]);
    t
}

/// Binary scalar `(num_row, 1, 1)` model over `trees`.
fn scalar_model<T, W>(
    trees: Vec<Tree<T>>,
    wrap: W,
    num_feature: i32,
    postprocessor: &str,
    base_score: f64,
) -> Model
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
    m.postprocessor = postprocessor.to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![base_score];
    m
}

/// A `(num_row, 1, K)` multiclass leaf-vector-broadcast model: a forest of
/// single-node leaf-vector trees of width K, each routed with
/// `target_id == class_id == -1` (broadcast). softmax postprocessor.
fn multiclass_model<T, W>(trees: Vec<Tree<T>>, wrap: W, k: i32, postprocessor: &str) -> Model
where
    T: Copy,
    W: Fn(ModelPreset<T>) -> ModelVariant,
{
    let num_tree = trees.len();
    let mut m = Model::new(wrap(ModelPreset::new(trees)));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![k];
    m.leaf_vector_shape = vec![1, k];
    m.target_id = vec![-1; num_tree];
    m.class_id = vec![-1; num_tree];
    m.postprocessor = postprocessor.to_string();
    m.sigmoid_alpha = 1.0;
    m.base_scores = vec![0.0; k as usize];
    m
}

fn assert_parity_f32(model: &Model, data: &[f32], num_row: usize, kind: PredictKind) {
    let cfg = Config { kind, nthread: 0 };
    let expected = predict::<f32>(model, data, num_row, &cfg).unwrap();
    let got = predict_cpu::<f32>(model, data, num_row, &cfg).unwrap();
    assert_eq!(got.len(), expected.len(), "{kind:?} f32 length");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }
}

fn assert_parity_f64(model: &Model, data: &[f64], num_row: usize, kind: PredictKind) {
    let cfg = Config { kind, nthread: 0 };
    let expected = predict::<f64>(model, data, num_row, &cfg).unwrap();
    let got = predict_cpu::<f64>(model, data, num_row, &cfg).unwrap();
    assert_eq!(got.len(), expected.len(), "{kind:?} f64 length");
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*g, *e, epsilon = 1e-5);
    }
}

const KINDS: [PredictKind; 4] = [
    PredictKind::Default,
    PredictKind::Raw,
    PredictKind::LeafId,
    PredictKind::ScorePerTree,
];

// --------------------------------------------------------------------------- //
// All four kinds, scalar binary model, f32 + f64.
// --------------------------------------------------------------------------- //

#[test]
fn all_kinds_scalar_binary_f32() {
    let trees = vec![
        split_tree::<f32>(0, 0.5, 1.0, -1.0),
        split_tree::<f32>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F32, 2, "sigmoid", 0.5);
    // Rows exercise both branches of both trees.
    let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    for kind in KINDS {
        assert_parity_f32(&model, &data, 3, kind);
    }
}

#[test]
fn all_kinds_scalar_binary_f64() {
    let trees = vec![
        split_tree::<f64>(0, 0.5, 1.0, -1.0),
        split_tree::<f64>(1, 0.5, 2.0, -3.0),
    ];
    let model = scalar_model(trees, ModelVariant::F64, 2, "sigmoid", 0.5);
    let data: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    for kind in KINDS {
        assert_parity_f64(&model, &data, 3, kind);
    }
}

// --------------------------------------------------------------------------- //
// Multiclass leaf-vector broadcast (the 4-way OutputLeafVector branch), f32+f64.
// --------------------------------------------------------------------------- //

#[test]
fn multiclass_leaf_vector_broadcast_f32() {
    // Two leaf-vector trees of width 3; broadcast (target==class==-1). softmax.
    let trees = vec![
        leaf_vector_tree::<f32>(vec![0.5, 1.5, -0.5]),
        leaf_vector_tree::<f32>(vec![1.0, -1.0, 2.0]),
    ];
    let model = multiclass_model(trees, ModelVariant::F32, 3, "softmax");
    let data: Vec<f32> = vec![0.0, 0.0]; // 2 rows, 1 feature each
    for kind in KINDS {
        assert_parity_f32(&model, &data, 2, kind);
    }
}

#[test]
fn multiclass_leaf_vector_broadcast_f64() {
    let trees = vec![
        leaf_vector_tree::<f64>(vec![0.5, 1.5, -0.5]),
        leaf_vector_tree::<f64>(vec![1.0, -1.0, 2.0]),
    ];
    let model = multiclass_model(trees, ModelVariant::F64, 3, "softmax");
    let data: Vec<f64> = vec![0.0, 0.0];
    for kind in KINDS {
        assert_parity_f64(&model, &data, 2, kind);
    }
}

// --------------------------------------------------------------------------- //
// RF averaging path (average_tree_output) — the per-cell divide.
// --------------------------------------------------------------------------- //

#[test]
fn rf_averaging_default_f32() {
    let trees = vec![
        split_tree::<f32>(0, 0.5, 4.0, -4.0),
        split_tree::<f32>(0, 0.5, 2.0, -2.0),
    ];
    let mut model = scalar_model(trees, ModelVariant::F32, 1, "identity", 0.0);
    model.average_tree_output = true;
    let data: Vec<f32> = vec![0.0, 1.0]; // 2 rows
    assert_parity_f32(&model, &data, 2, PredictKind::Default);
    assert_parity_f32(&model, &data, 2, PredictKind::Raw);
}

// --------------------------------------------------------------------------- //
// Categorical-split whole-model scalar fallback (D-02) — still within 1e-5.
// --------------------------------------------------------------------------- //

#[test]
fn categorical_split_routes_to_scalar_fallback() {
    // A numerical model, but with `has_categorical_split` set on a tree → the
    // WHOLE model must route to the scalar fallback and still match predict.
    let mut tree = split_tree::<f32>(0, 0.5, 1.0, -1.0);
    tree.has_categorical_split = true; // force the fallback gate (D-02)
    let model = scalar_model(vec![tree], ModelVariant::F32, 2, "sigmoid", 0.25);
    let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
    // The fallback path returns treelite_gtil::predict exactly, so parity is
    // bit-exact (well within 1e-5).
    assert_parity_f32(&model, &data, 2, PredictKind::Default);
}

// --------------------------------------------------------------------------- //
// Malformed shape → typed CubeclError BEFORE any device op (no panic, T-06-09).
// --------------------------------------------------------------------------- //

#[test]
fn malformed_shape_returns_typed_error_not_panic() {
    let trees = vec![split_tree::<f32>(0, 0.5, 1.0, -1.0)];
    let model = scalar_model(trees, ModelVariant::F32, 2, "identity", 0.0);
    // num_row * num_feature = 2 * 2 = 4, but only 2 elements supplied.
    let data: Vec<f32> = vec![0.0, 0.0];
    let err = predict_cpu::<f32>(&model, &data, 2, &Config::default())
        .expect_err("a short data buffer must error, not panic");
    match err {
        CubeclError::InvalidInputShape { .. } => {}
        other => panic!("expected InvalidInputShape, got {other:?}"),
    }
}

#[test]
fn bad_split_index_returns_typed_error() {
    // split_index 9 >= num_feature 2 → FeatureIndexOutOfBounds before any launch.
    let mut tree = split_tree::<f32>(9, 0.5, 1.0, -1.0);
    tree.split_index = TreeBuf::from_owned(vec![9, -1, -1]);
    let model = scalar_model(vec![tree], ModelVariant::F32, 2, "identity", 0.0);
    let data: Vec<f32> = vec![0.0, 0.0];
    let err = predict_cpu::<f32>(&model, &data, 1, &Config::default())
        .expect_err("an out-of-range split_index must error, not panic");
    match err {
        CubeclError::FeatureIndexOutOfBounds { .. } => {}
        other => panic!("expected FeatureIndexOutOfBounds, got {other:?}"),
    }
}

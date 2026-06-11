//! Tests for the `O`-generic input/output element path (D-05, GTIL-01,
//! GTIL-08), Plan 05-02 Task 2.
//!
//! Upstream instantiates `Predict<float>` and `Predict<double>`, and the output
//! buffer element type == the INPUT element type, independent of the model
//! preset (`predict.cc:236` `Array3DView<InputT>`; `c_api/gtil.cc:50-55`). So
//! all 4 (input dtype × preset) combinations are valid:
//! - f32 input over `<f32,f32>` preset → `Vec<f32>` (the existing path, byte-identical)
//! - f64 input over `<f64,f64>` preset → `Vec<f64>`
//! - f32 input over `<f64,f64>` preset → `Vec<f32>`
//! - f64 input over `<f32,f32>` preset → `Vec<f64>`
//!
//! NextNode promotes the input value into the wider of (input, threshold) per
//! C++ usual arithmetic conversions; an exact f32→f64 widening is order
//! preserving, so cross-domain routing is bit-faithful.

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, predict};

/// A single-split tree over `Tree<T>`: node 0 numerical test (`kLT`,
/// default-left) on feature 0, nodes 1/2 leaves.
fn split_tree<T: Copy + Default>(threshold: T, left_leaf: T, right_leaf: T) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.num_nodes = 3;
    t.cleft = TreeBuf::from_owned(vec![1, -1, -1]);
    t.cright = TreeBuf::from_owned(vec![2, -1, -1]);
    t.split_index = TreeBuf::from_owned(vec![0, -1, -1]);
    t.default_left = TreeBuf::from_owned(vec![true, false, false]);
    t.threshold = TreeBuf::from_owned(vec![threshold, T::default(), T::default()]);
    t.cmp = TreeBuf::from_owned(vec![Operator::kLT, Operator::kNone, Operator::kNone]);
    t.leaf_value = TreeBuf::from_owned(vec![T::default(), left_leaf, right_leaf]);
    t.node_type = TreeBuf::from_owned(vec![
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kLeafNode,
        TreeNodeType::kLeafNode,
    ]);
    t
}

fn binary_metadata(m: &mut Model) {
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![1].into();
    m.leaf_vector_shape = vec![1, 1].into();
    m.target_id = vec![0].into();
    m.class_id = vec![0].into();
    m.postprocessor = "identity".to_string().into();
    m.base_scores = vec![0.0].into();
}

fn f32_model() -> Model {
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(vec![
        split_tree::<f32>(0.5, 1.25, -1.5),
    ])));
    binary_metadata(&mut m);
    m
}

fn f64_model() -> Model {
    let mut m = Model::new(ModelVariant::F64(ModelPreset::new(vec![
        split_tree::<f64>(0.5, 1.25, -1.5),
    ])));
    binary_metadata(&mut m);
    m
}

#[test]
fn f32_input_f32_preset_unchanged() {
    let m = f32_model();
    let data = [0.0_f32]; // < 0.5 → left leaf 1.25
    let out: Vec<f32> = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 1.25_f32).abs() < 1e-6, "got {}", out[0]);
}

#[test]
fn f64_input_f64_preset_returns_f64() {
    let m = f64_model();
    let data = [0.0_f64]; // < 0.5 → left leaf 1.25
    let out: Vec<f64> = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    // f64 golden reference: leaf 1.25 + base 0.0, identity postprocessor.
    approx::assert_abs_diff_eq!(out[0], 1.25_f64, epsilon = 1e-5);
}

#[test]
fn f64_input_over_f32_preset_compiles_and_runs() {
    // f64 input over the <f32,f32> preset (input dtype NOT constrained to preset).
    let m = f32_model();
    let data = [9.0_f64]; // >= 0.5 → right leaf -1.5
    let out: Vec<f64> = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    approx::assert_abs_diff_eq!(out[0], -1.5_f64, epsilon = 1e-5);
}

#[test]
fn f32_input_over_f64_preset_compiles_and_runs() {
    // f32 input over the <f64,f64> preset.
    let m = f64_model();
    let data = [9.0_f32]; // >= 0.5 → right leaf -1.5
    let out: Vec<f32> = predict(&m, &data, 1, &Config::default()).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - (-1.5_f32)).abs() < 1e-6, "got {}", out[0]);
}

#[test]
fn cross_domain_comparison_is_order_preserving() {
    // Threshold 0.5 on the f64 preset; f32 input 0.4999999 (< 0.5) must route
    // left, and 0.5 (>= 0.5) must route right — comparison promoted to f64.
    let m = f64_model();
    let left = predict(&m, &[0.4999_f32], 1, &Config::default()).unwrap();
    let right = predict(&m, &[0.5_f32], 1, &Config::default()).unwrap();
    assert!((left[0] - 1.25_f32).abs() < 1e-6, "left route: {}", left[0]);
    assert!(
        (right[0] - (-1.5_f32)).abs() < 1e-6,
        "right route: {}",
        right[0]
    );
}

#[test]
fn f64_two_tree_serial_sum() {
    // Two f64 trees summed serially: leaves 0.1 + 0.2 = 0.3 (the classic
    // non-associative case; serial order preserved, GTIL-08).
    let mut m = Model::new(ModelVariant::F64(ModelPreset::new(vec![
        split_tree::<f64>(0.5, 0.1, -9.0),
        split_tree::<f64>(0.5, 0.2, -9.0),
    ])));
    binary_metadata(&mut m);
    m.target_id = vec![0, 0].into();
    m.class_id = vec![0, 0].into();
    let out: Vec<f64> = predict(&m, &[0.0_f64], 1, &Config::default()).unwrap();
    approx::assert_abs_diff_eq!(out[0], 0.3_f64, epsilon = 1e-5);
}

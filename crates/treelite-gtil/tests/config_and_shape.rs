//! Tests for the typed `Config`/`PredictKind` entry surface (D-06) and the
//! public `Shape`/`output_shape` descriptor (D-07), Plan 05-02 Task 1.
//!
//! `Config::default()` must mirror `gtil.h:51-52` (`pred_kind kPredictDefault`,
//! `nthread 0`). `output_shape` ports `output_shape.cc:17-39` verbatim per kind:
//! - default/raw → `num_target>1 ? [r, num_target, max_num_class] : [r, 1, max_num_class]`
//! - leaf_id     → `[r, num_tree]`
//! - per_tree    → `[r, num_tree, leaf_vector_shape[0] * leaf_vector_shape[1]]`

use treelite_core::{Model, ModelPreset, ModelVariant, Operator, Tree, TreeBuf, TreeNodeType};
use treelite_gtil::{Config, PredictKind, output_shape};

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
    t.leaf_vector_begin = TreeBuf::from_owned(vec![0]);
    t.leaf_vector_end = TreeBuf::from_owned(vec![0]);
    t
}

/// A binary scalar model: one tree, num_target=1, num_class=[1].
fn binary_model() -> Model {
    let mut m = Model::new(ModelVariant::F32(ModelPreset::new(vec![
        leaf_tree(0.5),
        leaf_tree(0.25),
        leaf_tree(0.1),
    ])));
    m.num_feature = 1;
    m.num_target = 1;
    m.num_class = vec![1];
    m.leaf_vector_shape = vec![1, 1];
    m.target_id = vec![0, 0, 0];
    m.class_id = vec![0, 0, 0];
    m.postprocessor = "identity".to_string();
    m.base_scores = vec![0.0];
    m
}

#[test]
fn config_default_matches_upstream() {
    let cfg = Config::default();
    assert_eq!(
        cfg.kind,
        PredictKind::Default,
        "default kind is kPredictDefault"
    );
    assert_eq!(cfg.nthread, 0, "default nthread is 0 (gtil.h:51)");
}

#[test]
fn output_shape_default_collapses_binary_to_one() {
    // num_target == 1 ⇒ default/raw collapses to dim 1 (NOT omitted): [r, 1, 1].
    let m = binary_model();
    let cfg = Config {
        kind: PredictKind::Default,
        nthread: 0,
    };
    let shape = output_shape(&m, 10, &cfg);
    assert_eq!(
        shape.dims,
        vec![10, 1, 1],
        "binary default → [r, 1, max_num_class]"
    );
}

#[test]
fn output_shape_raw_same_as_default() {
    let m = binary_model();
    let cfg = Config {
        kind: PredictKind::Raw,
        nthread: 0,
    };
    let shape = output_shape(&m, 7, &cfg);
    assert_eq!(shape.dims, vec![7, 1, 1]);
}

#[test]
fn output_shape_leaf_id_is_row_by_num_tree() {
    let m = binary_model();
    let num_tree = m.num_tree();
    let cfg = Config {
        kind: PredictKind::LeafId,
        nthread: 0,
    };
    let shape = output_shape(&m, 10, &cfg);
    assert_eq!(shape.dims, vec![10, num_tree], "leaf_id → [r, num_tree]");
}

#[test]
fn output_shape_score_per_tree_uses_leaf_vector_shape_product() {
    let mut m = binary_model();
    // leaf_vector_shape [2, 3] ⇒ product 6.
    m.leaf_vector_shape = vec![2, 3];
    let num_tree = m.num_tree();
    let cfg = Config {
        kind: PredictKind::ScorePerTree,
        nthread: 0,
    };
    let shape = output_shape(&m, 4, &cfg);
    assert_eq!(
        shape.dims,
        vec![4, num_tree, 6],
        "per_tree → [r, num_tree, leaf_vector_shape[0]*leaf_vector_shape[1]]"
    );
}

#[test]
fn predict_takes_config_and_dispatches_default_vs_raw() {
    // Default applies the sigmoid postprocessor; Raw skips it. A single
    // scalar tree with leaf 0.0 ⇒ raw margin 0.0; sigmoid(0) = 0.5.
    let mut m = binary_model();
    m.variant = ModelVariant::F32(ModelPreset::new(vec![leaf_tree(0.0)]));
    m.target_id = vec![0];
    m.class_id = vec![0];
    m.postprocessor = "sigmoid".to_string();
    m.sigmoid_alpha = 1.0;

    let data = [0.0_f32];
    let raw = treelite_gtil::predict(
        &m,
        &data,
        1,
        &Config {
            kind: PredictKind::Raw,
            nthread: 0,
        },
    )
    .unwrap();
    assert!(
        (raw[0] - 0.0).abs() < 1e-6,
        "raw skips postprocessor: {}",
        raw[0]
    );

    let default = treelite_gtil::predict(
        &m,
        &data,
        1,
        &Config {
            kind: PredictKind::Default,
            nthread: 0,
        },
    )
    .unwrap();
    assert!(
        (default[0] - 0.5).abs() < 1e-6,
        "default applies sigmoid(0)=0.5: {}",
        default[0]
    );
}

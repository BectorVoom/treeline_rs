//! Typed field accessors (SER-04) — read-only model/tree inspection.
//!
//! Loads the frozen `golden_v5.bin` (`binary:logistic`) and asserts the SER-04
//! read surface: model header readers (`num_feature()`, and after staging
//! `num_tree()` / `threshold_type()`), and per-tree node accessors
//! (`is_leaf`/`leaf_value`/`threshold`/`node_type`/`comparison_op`) return the
//! expected node values for an internal node and a leaf node.
//!
//! Read-only fidelity (T-02-J02 / field_accessor.cc): the version triple,
//! `num_tree`, and the type tags expose NO setter — this is enforced
//! structurally (no `set_*` method exists) and is asserted at the source level
//! by the plan's grep check, not at runtime.

use std::path::Path;

use treelite_core::{DType, ModelVariant, Operator, TreeNodeType, deserialize};

fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

fn load_golden() -> treelite_core::Model {
    let blob = std::fs::read(fixture_path("golden_v5.bin")).expect("read golden_v5.bin");
    deserialize(&blob).expect("deserialize golden_v5.bin")
}

#[test]
fn model_header_accessors_expose_num_feature_num_tree_and_type_tag() {
    let mut model = load_golden();

    // `num_feature()` is readable without staging.
    assert!(
        model.num_feature() >= 1,
        "binary:logistic fixture has at least one feature"
    );

    // The staged readers reflect the variant after staging.
    let tree_count = match &model.variant {
        ModelVariant::F32(p) => p.trees.len(),
        ModelVariant::F64(p) => p.trees.len(),
    };

    // Before staging, type tags are kInvalid (upstream TypeInfo::kInvalid).
    assert_eq!(model.threshold_type(), DType::kInvalid);

    model.stage_serialization_fields();

    assert_eq!(
        model.num_tree(),
        tree_count as u64,
        "num_tree() must equal the variant's tree count"
    );
    assert_eq!(
        model.threshold_type(),
        DType::kFloat32,
        "binary:logistic golden model is the <f32,f32> preset"
    );
    assert_eq!(model.leaf_output_type(), DType::kFloat32);
    // The version triple is the producing-Treelite 4.7.0 (read-only).
    assert_eq!(
        (model.major_ver(), model.minor_ver(), model.patch_ver()),
        (4, 7, 0)
    );
    assert_eq!(model.num_opt_field_per_model(), 0);
}

#[test]
fn tree_node_accessors_return_expected_internal_and_leaf_values() {
    let model = load_golden();
    let ModelVariant::F32(preset) = &model.variant else {
        panic!("golden model must be the F32 preset");
    };
    let tree = &preset.trees[0];

    // Find one internal node and one leaf node, then cross-check the accessors.
    let mut internal: Option<usize> = None;
    let mut leaf: Option<usize> = None;
    for nid in 0..tree.num_nodes as usize {
        if tree.is_leaf(nid) {
            leaf.get_or_insert(nid);
        } else {
            internal.get_or_insert(nid);
        }
    }
    let internal = internal.expect("tree 0 must have an internal node");
    let leaf = leaf.expect("tree 0 must have a leaf node");

    // Internal node: not a leaf, a numerical test with `<` (kLT) and a valid
    // child wiring (`is_leaf` ⇔ `left_child == -1`).
    assert!(!tree.is_leaf(internal));
    assert_eq!(tree.node_type(internal), TreeNodeType::kNumericalTestNode);
    assert_eq!(
        tree.comparison_op(internal),
        Operator::kLT,
        "binary:logistic numerical splits use `<` (kLT)"
    );
    assert!(tree.left_child(internal) >= 0 && tree.right_child(internal) >= 0);
    // `threshold(nid)` returns the split threshold (finite for a real split).
    assert!(
        tree.threshold(internal).is_finite(),
        "an internal numerical node has a finite threshold"
    );

    // Leaf node: `is_leaf`, `cleft == -1`, and a finite `leaf_value`.
    assert!(tree.is_leaf(leaf));
    assert_eq!(tree.left_child(leaf), -1, "a leaf has cleft == -1");
    assert!(
        tree.leaf_value(leaf).is_finite(),
        "binary:logistic leaf outputs are finite"
    );
}

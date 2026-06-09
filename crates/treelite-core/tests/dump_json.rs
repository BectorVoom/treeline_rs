//! `DumpAsJSON` structural fidelity (SER-03, D-04).
//!
//! Loads the frozen upstream `golden_v5.bin` (a real `binary:logistic`
//! XGBoost model with internal + leaf nodes), dumps it via [`dump_as_json`],
//! and asserts the JSON STRUCTURE at the value level — the model-object key set
//! mirrors upstream `json_serializer.cc`, the enum strings use the upstream
//! `as_str()` spellings (`task_type == "kBinaryClf"`, `threshold_type ==
//! "float32"`), a numerical internal node carries `comparison_op == "<"` + a
//! `threshold`, and a leaf carries a `leaf_value`.
//!
//! Per D-04/Q3 the assertions compare PARSED values, never raw byte strings:
//! JSON float formatting may differ between RapidJSON and `serde_json`, so a
//! whole-string `assert_eq!` is intentionally avoided.

use std::path::Path;

use treelite_core::{deserialize, dump_as_json};

fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// Load the frozen golden v5 model from disk.
fn load_golden() -> treelite_core::Model {
    let blob = std::fs::read(fixture_path("golden_v5.bin")).expect("read golden_v5.bin");
    deserialize(&blob).expect("deserialize golden_v5.bin")
}

#[test]
fn dump_model_object_has_upstream_key_set_and_enum_strings() {
    let mut model = load_golden();
    let v = dump_as_json(&mut model);

    let obj = v.as_object().expect("dump must be a JSON object");

    // Model-object key set in the upstream order (json_serializer.cc:182-217).
    for key in [
        "threshold_type",
        "leaf_output_type",
        "num_feature",
        "task_type",
        "average_tree_output",
        "num_target",
        "num_class",
        "leaf_vector_shape",
        "target_id",
        "class_id",
        "postprocessor",
        "sigmoid_alpha",
        "ratio_c",
        "base_scores",
        "attributes",
        "trees",
    ] {
        assert!(obj.contains_key(key), "model object missing key `{key}`");
    }

    // Enum strings reuse the upstream as_str() spellings (D-04).
    assert_eq!(
        obj["threshold_type"], *"float32",
        "threshold_type must dump as the upstream `float32` spelling"
    );
    assert_eq!(obj["leaf_output_type"], *"float32");
    assert_eq!(
        obj["task_type"], *"kBinaryClf",
        "task_type must dump as the upstream `kBinaryClf` spelling"
    );

    // Type-level checks on a few scalar/array fields.
    assert!(
        obj["num_feature"].is_i64(),
        "num_feature must be an integer"
    );
    assert!(obj["average_tree_output"].is_boolean());
    assert!(obj["num_class"].is_array());
    assert!(obj["base_scores"].is_array());
    assert!(obj["attributes"].is_string());
}

#[test]
fn trees_array_length_matches_tree_count_and_each_tree_has_nodes() {
    let mut model = load_golden();

    // Tree count from the variant (the golden model is the F32 preset).
    let expected_num_tree = match &model.variant {
        treelite_core::ModelVariant::F32(p) => p.trees.len(),
        treelite_core::ModelVariant::F64(p) => p.trees.len(),
    };

    let v = dump_as_json(&mut model);
    let trees = v["trees"].as_array().expect("`trees` must be an array");
    assert_eq!(
        trees.len(),
        expected_num_tree,
        "trees array length must equal num_tree"
    );
    assert!(
        !trees.is_empty(),
        "golden model must have at least one tree"
    );

    for tree in trees {
        let t = tree.as_object().expect("each tree is a JSON object");
        assert!(t.contains_key("num_nodes"));
        assert!(t.contains_key("has_categorical_split"));
        assert!(
            t["nodes"].is_array(),
            "each tree must carry a `nodes` array"
        );
    }
}

#[test]
fn internal_node_and_leaf_node_have_upstream_per_node_keys() {
    let mut model = load_golden();
    let v = dump_as_json(&mut model);

    // Collect all nodes across every tree.
    let mut found_numerical_internal = false;
    let mut found_leaf = false;

    for tree in v["trees"].as_array().unwrap() {
        for node in tree["nodes"].as_array().unwrap() {
            let n = node.as_object().unwrap();
            assert!(n.contains_key("node_id"), "every node carries node_id");

            if n.contains_key("leaf_value") {
                // A leaf node — must NOT carry split keys.
                assert!(!n.contains_key("split_feature_id"));
                found_leaf = true;
            } else {
                // An internal node — must carry the split key set.
                assert!(n.contains_key("split_feature_id"));
                assert!(n.contains_key("default_left"));
                assert!(n.contains_key("node_type"));
                assert!(n.contains_key("left_child"));
                assert!(n.contains_key("right_child"));

                if n["node_type"] == *"numerical_test_node" {
                    // Numerical test ⇒ comparison_op + threshold (D-04).
                    assert_eq!(
                        n["comparison_op"], *"<",
                        "binary:logistic numerical splits use `<` (kLT)"
                    );
                    assert!(
                        n["threshold"].is_number(),
                        "a numerical node must carry a numeric threshold"
                    );
                    found_numerical_internal = true;
                }
            }
        }
    }

    assert!(
        found_numerical_internal,
        "golden model must contain a numerical internal node with `comparison_op == \"<\"`"
    );
    assert!(
        found_leaf,
        "golden model must contain a leaf node with a `leaf_value`"
    );
}

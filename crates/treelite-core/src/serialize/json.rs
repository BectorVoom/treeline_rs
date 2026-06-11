//! `DumpAsJSON` (SER-03, D-04) — emit a [`Model`] as structured JSON.
//!
//! Ports `treelite-mainline/src/json_serializer.cc` (`DumpModelAsJSON`,
//! `DumpTreeAsJSON`, `WriteNode`, and the `SerializeTaskParametersToJSON` /
//! `SerializeModelParametersToJSON` helpers). The produced object's key names,
//! nesting, and value types mirror upstream EXACTLY so a Rust dump is
//! value-diffable against a C++ dump (D-04).
//!
//! Fidelity rule (D-04): the enum string forms come from the already-correct
//! [`crate::enums`] `as_str()` spellings — `task_type` via
//! [`TaskType::as_str`](crate::enums::TaskType::as_str), the type tags via
//! [`DType::as_str`](crate::enums::DType::as_str) (`"float32"`/`"float64"`),
//! `node_type` via [`TreeNodeType::as_str`](crate::enums::TreeNodeType::as_str),
//! and `comparison_op` via [`Operator::as_str`](crate::enums::Operator::as_str).
//! These are NOT re-spelled here.
//!
//! Float caveat (A4/Q3): JSON float *formatting* may differ between RapidJSON
//! and `serde_json`, so equivalence is checked at the PARSED-VALUE level, never
//! by byte-comparing the serialized strings.

use serde_json::{Map, Value, json};

use crate::model::{Model, ModelVariant};
use crate::tree::Tree;

/// Dump `m` as a [`serde_json::Value`] whose structure mirrors upstream
/// `json_serializer.cc::DumpModelAsJSON` (json_serializer.cc:182-217).
///
/// The model object's keys are emitted in upstream order:
/// `threshold_type`, `leaf_output_type`, `num_feature`, `task_type`,
/// `average_tree_output`, then the task parameters (`num_target`, `num_class`,
/// `leaf_vector_shape`), `target_id`, `class_id`, the model parameters
/// (`postprocessor`, `sigmoid_alpha`, `ratio_c`, `base_scores`, `attributes`),
/// and finally `trees`.
///
/// `m` is taken `&mut` so the v5 type tags can be staged via
/// [`Model::stage_serialization_fields`]; upstream reads them from
/// `GetThresholdType()`/`GetLeafOutputType()` which are likewise derived from
/// the active variant.
pub fn dump_as_json(m: &mut Model) -> Value {
    // Stage so `threshold_type`/`leaf_output_type` reflect the active variant
    // (upstream `model.GetThresholdType()`/`GetLeafOutputType()`).
    m.stage_serialization_fields();

    let mut obj = Map::new();

    // json_serializer.cc:185-188 — type tags via DType::as_str (reuse, D-04).
    obj.insert(
        "threshold_type".into(),
        Value::from(m.threshold_type().as_str()),
    );
    obj.insert(
        "leaf_output_type".into(),
        Value::from(m.leaf_output_type().as_str()),
    );
    // json_serializer.cc:189-194 — num_feature, task_type, average_tree_output.
    obj.insert("num_feature".into(), Value::from(m.num_feature));
    obj.insert("task_type".into(), Value::from(m.task_type.as_str()));
    obj.insert(
        "average_tree_output".into(),
        Value::from(m.average_tree_output),
    );

    // SerializeTaskParametersToJSON (json_serializer.cc:135-143).
    obj.insert("num_target".into(), Value::from(m.num_target));
    // MEM-02: deref the migrated `SmallVec` fields to `&[i32]`/`&[f64]` and the
    // `CompactString` fields to `&str` so `json!`/`Value::from` operate on the
    // slice/str (no `serde` feature on SmallVec/CompactString, A4) — identical JSON.
    obj.insert("num_class".into(), json!(&m.num_class[..]));
    obj.insert("leaf_vector_shape".into(), json!(&m.leaf_vector_shape[..]));

    // json_serializer.cc:198-201 — target_id, class_id.
    obj.insert("target_id".into(), json!(&m.target_id[..]));
    obj.insert("class_id".into(), json!(&m.class_id[..]));

    // SerializeModelParametersToJSON (json_serializer.cc:145-157).
    obj.insert("postprocessor".into(), Value::from(m.postprocessor.as_str()));
    obj.insert("sigmoid_alpha".into(), json!(m.sigmoid_alpha as f64));
    obj.insert("ratio_c".into(), json!(m.ratio_c as f64));
    obj.insert("base_scores".into(), json!(&m.base_scores[..]));
    obj.insert("attributes".into(), Value::from(m.attributes.as_str()));

    // json_serializer.cc:205-214 — trees array (std::visit over the variant).
    let trees: Vec<Value> = match &m.variant {
        ModelVariant::F32(p) => p.trees.iter().map(dump_tree_as_json).collect(),
        ModelVariant::F64(p) => p.trees.iter().map(dump_tree_as_json).collect(),
    };
    obj.insert("trees".into(), Value::Array(trees));

    Value::Object(obj)
}

/// Convenience wrapper returning the compact JSON string form.
///
/// Mirrors `Model::DumpAsJSON(fo, pretty_print=false)`
/// (json_serializer.cc:219-229). Compact (non-pretty) output is emitted; the
/// caller diffs parsed values, not raw bytes (A4/Q3).
pub fn dump_as_json_string(m: &mut Model) -> String {
    dump_as_json(m).to_string()
}

/// Dump one tree as a JSON object (`DumpTreeAsJSON`, json_serializer.cc:163-179):
/// `{ num_nodes, has_categorical_split, nodes: [...] }`.
fn dump_tree_as_json<T: Copy + Into<f64>>(tree: &Tree<T>) -> Value {
    let nodes: Vec<Value> = (0..tree.num_nodes as usize)
        .map(|nid| write_node(tree, nid))
        .collect();
    json!({
        "num_nodes": tree.num_nodes,
        "has_categorical_split": tree.has_categorical_split,
        "nodes": Value::Array(nodes),
    })
}

/// Emit one node object (`WriteNode`, json_serializer.cc:81-133).
///
/// Key set: always `node_id`; leaf → `leaf_value` (scalar, or array when the
/// node carries a leaf vector); internal → `split_feature_id`, `default_left`,
/// `node_type`, then for a numerical test `comparison_op` + `threshold`, for a
/// categorical test `category_list_right_child` + `category_list`, then
/// `left_child`, `right_child`. Conditional tail (`data_count`, `sum_hess`,
/// `gain`) is keyed off the per-column present flags.
fn write_node<T: Copy + Into<f64>>(tree: &Tree<T>, nid: usize) -> Value {
    let mut node = Map::new();

    // json_serializer.cc:86-87 — always node_id.
    node.insert("node_id".into(), Value::from(nid as i64));

    if tree.is_leaf(nid) {
        // json_serializer.cc:88-94 — leaf_value scalar or array.
        if tree.has_leaf_vector(nid) {
            let v: Vec<f64> = tree.leaf_vector(nid).iter().map(|&x| x.into()).collect();
            node.insert("leaf_value".into(), json!(v));
        } else {
            node.insert("leaf_value".into(), json!(tree.leaf_value(nid).into()));
        }
    } else {
        // json_serializer.cc:95-102 — split_feature_id, default_left, node_type.
        node.insert(
            "split_feature_id".into(),
            Value::from(tree.split_index(nid) as u64),
        );
        node.insert("default_left".into(), Value::from(tree.default_left(nid)));
        let node_type = tree.node_type(nid);
        node.insert("node_type".into(), Value::from(node_type.as_str()));

        match node_type {
            crate::enums::TreeNodeType::kNumericalTestNode => {
                // json_serializer.cc:103-107 — comparison_op + threshold.
                node.insert(
                    "comparison_op".into(),
                    Value::from(tree.comparison_op(nid).as_str()),
                );
                node.insert("threshold".into(), json!(tree.threshold(nid).into()));
            }
            crate::enums::TreeNodeType::kCategoricalTestNode => {
                // json_serializer.cc:108-112 — category_list_right_child + list.
                node.insert(
                    "category_list_right_child".into(),
                    Value::from(tree.category_list_right_child(nid)),
                );
                node.insert("category_list".into(), json!(tree.category_list(nid)));
            }
            crate::enums::TreeNodeType::kLeafNode => {
                // Unreachable: an internal node is never kLeafNode here.
            }
        }

        // json_serializer.cc:114-117 — left_child, right_child.
        node.insert("left_child".into(), Value::from(tree.left_child(nid)));
        node.insert("right_child".into(), Value::from(tree.right_child(nid)));
    }

    // json_serializer.cc:119-130 — conditional statistics tail.
    if tree.has_data_count(nid) {
        node.insert("data_count".into(), Value::from(tree.data_count(nid)));
    }
    if tree.has_sum_hess(nid) {
        node.insert("sum_hess".into(), json!(tree.sum_hess(nid)));
    }
    if tree.has_gain(nid) {
        node.insert("gain".into(), json!(tree.gain(nid)));
    }

    Value::Object(node)
}

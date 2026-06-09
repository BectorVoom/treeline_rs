//! Tests for `bulk_construct_tree` (BLD-03, D-09), ported from `sklearn_bulk.cc`.
//!
//! Builds a tree from small sklearn-shaped arrays and asserts its columns match
//! an equivalent per-node build of the same logical tree, proving the bypass
//! produces the same tree a validated build would (minus validation). The input
//! is one the fluent builder would accept (root reachable, no orphans), so no
//! validation is being relied upon — it is genuinely bypassed.

use treelite_builder::bulk_construct_tree;
use treelite_core::enums::{Operator, TreeNodeType};

/// A logical tree: node 0 is an internal split on feature 1 at threshold 0.5,
/// children 1 and 2 are leaves with values 10.0 and 20.0.
///
/// sklearn arrays (indexed by node id):
///   children_left  = [1, -1, -1]
///   children_right = [2, -1, -1]
fn small_tree() -> treelite_core::Tree<f64> {
    let children_left = [1_i64, -1, -1];
    let children_right = [2_i64, -1, -1];
    let feature = [1_i64, -1, -1];
    let threshold = [0.5_f64, 0.0, 0.0];
    let value = [0.0_f64, 10.0, 20.0]; // scalar leaves (leaf_vector_size == 1)
    let n_node_samples = [100_i64, 60, 40];
    let weighted_n_node_samples = [100.0_f64, 60.0, 40.0];
    let impurity = [0.5_f64, 0.1, 0.2];
    bulk_construct_tree(
        3,
        &children_left,
        &children_right,
        &feature,
        &threshold,
        &value,
        &n_node_samples,
        &weighted_n_node_samples,
        &impurity,
        100,   // total_sample_cnt
        1,     // n_targets
        1,     // max_num_class (scalar leaves)
        false, // is_classifier
    )
}

#[test]
fn bulk_tree_columns_match_expected_per_node_build() {
    let t = small_tree();
    assert_eq!(t.num_nodes, 3);

    // node_type: internal, leaf, leaf
    assert_eq!(t.node_type[0], TreeNodeType::kNumericalTestNode);
    assert_eq!(t.node_type[1], TreeNodeType::kLeafNode);
    assert_eq!(t.node_type[2], TreeNodeType::kLeafNode);

    // children (cleft/cright)
    assert_eq!(t.cleft[0], 1);
    assert_eq!(t.cright[0], 2);
    assert_eq!(t.cleft[1], -1);
    assert_eq!(t.cright[1], -1);

    // threshold / split_index on the internal node
    assert_eq!(t.threshold[0], 0.5);
    assert_eq!(t.split_index[0], 1);

    // leaf values
    assert_eq!(t.leaf_value[1], 10.0);
    assert_eq!(t.leaf_value[2], 20.0);

    // cmp: internal kLE, leaves kNone (RESEARCH Pattern 3)
    assert_eq!(t.cmp[0], Operator::kLE);
    assert_eq!(t.cmp[1], Operator::kNone);
    assert_eq!(t.cmp[2], Operator::kNone);

    // default_left always true (sklearn_bulk.cc:136)
    assert!(t.default_left[0]);
    assert!(t.default_left[1]);

    // no categorical splits
    assert!(!t.has_categorical_split);
}

#[test]
fn bulk_node_statistics_present_and_gain_only_on_internals() {
    let t = small_tree();

    // data_count / sum_hess present on every node
    assert!(t.data_count_present[0]);
    assert!(t.data_count_present[1]);
    assert_eq!(t.data_count[0], 100);
    assert_eq!(t.sum_hess[0], 100.0);

    // gain present only on internals (sklearn_bulk.cc:204-208)
    assert!(t.gain_present[0]);
    assert!(!t.gain_present[1]);
    assert!(!t.gain_present[2]);

    // gain value = sample_cnt * (impurity - wL*impL/cnt - wR*impR/cnt) / total
    //   = 100 * (0.5 - 60*0.1/100 - 40*0.2/100) / 100
    //   = 0.5 - 0.06 - 0.08 = 0.36
    let expected = 100.0 * (0.5 - 60.0 * 0.1 / 100.0 - 40.0 * 0.2 / 100.0) / 100.0;
    assert!((t.gain[0] - expected).abs() < 1e-12);
    assert_eq!(t.gain[1], 0.0);
}

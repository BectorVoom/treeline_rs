//! `BulkConstructTree` validation-bypass fast path (BLD-03, D-09).
//!
//! Ports the ONLY upstream `BulkConstructTree`
//! (`treelite-mainline/src/model_loader/sklearn_bulk.cc:36-211`): a sklearn-shaped
//! bulk constructor that directly fills a `Tree`'s columns in a single pass,
//! bypassing the node-by-node `ModelBuilder` validation entirely.
//!
//! D-09: this is a validation BYPASS by construction. The bulk arrays are the
//! caller's pre-validated contract (the sklearn loader in Phase 4 owns producing
//! well-formed arrays); no orphan / dangling / duplicate checks run here. The
//! threat register (T-02-B03) accepts this: no external/untrusted bytes reach
//! this path in Phase 2.
//!
//! Upstream uses `Tree<double,double>` for sklearn; this port produces a
//! `Tree<f64>` to match the element types. Leaf detection is `children_left[i]
//! == -1`; internal nodes use `cmp = kLE`, `default_left = true`; the gain is the
//! sklearn impurity-reduction formula (`sklearn_bulk.cc:193-205`).

use treelite_core::enums::{Operator, TreeNodeType};
use treelite_core::{Tree, TreeBuf};

/// Bulk-construct a `Tree<f64>` from pre-validated sklearn-shaped arrays
/// (`sklearn_bulk.cc:36-211`), bypassing per-node validation (D-09).
///
/// All slices are indexed by node id and must be length `n_nodes` (the caller's
/// contract). `value` holds either one scalar per node (`leaf_vector_size <= 1`)
/// or `n_targets * max_num_class` entries per node for vector leaves.
///
/// # Panics
/// Per D-09 this trusts its input; out-of-range indices in `children_left/right`
/// (used to read `n_node_samples`/`impurity` for the gain) would panic. The
/// sklearn loader (Phase 4) guarantees well-formed arrays.
#[allow(clippy::too_many_arguments)]
pub fn bulk_construct_tree(
    n_nodes: usize,
    children_left: &[i64],
    children_right: &[i64],
    feature: &[i64],
    threshold: &[f64],
    value: &[f64],
    n_node_samples: &[i64],
    weighted_n_node_samples: &[f64],
    impurity: &[f64],
    total_sample_cnt: i64,
    n_targets: i32,
    max_num_class: i32,
    is_classifier: bool,
) -> Tree<f64> {
    let mut node_type = Vec::with_capacity(n_nodes);
    let mut cleft = Vec::with_capacity(n_nodes);
    let mut cright = Vec::with_capacity(n_nodes);
    let mut split_index = Vec::with_capacity(n_nodes);
    let mut default_left = Vec::with_capacity(n_nodes);
    let mut leaf_value = Vec::with_capacity(n_nodes);
    let mut threshold_col = Vec::with_capacity(n_nodes);
    let mut cmp = Vec::with_capacity(n_nodes);
    let mut category_list_right_child = Vec::with_capacity(n_nodes);
    let mut leaf_vector: Vec<f64> = Vec::new();
    let mut leaf_vector_begin = Vec::with_capacity(n_nodes);
    let mut leaf_vector_end = Vec::with_capacity(n_nodes);
    let mut category_list_begin = Vec::with_capacity(n_nodes);
    let mut category_list_end = Vec::with_capacity(n_nodes);
    let mut data_count = Vec::with_capacity(n_nodes);
    let mut data_count_present = Vec::with_capacity(n_nodes);
    let mut sum_hess = Vec::with_capacity(n_nodes);
    let mut sum_hess_present = Vec::with_capacity(n_nodes);
    let mut gain = Vec::with_capacity(n_nodes);
    let mut gain_present = Vec::with_capacity(n_nodes);

    // Leaf-vector sizing (`sklearn_bulk.cc:90-91`).
    let leaf_vector_size = (n_targets * max_num_class) as usize;
    let has_leaf_vector = leaf_vector_size > 1;

    // Single-pass fill (`sklearn_bulk.cc:109-210`).
    for node_id in 0..n_nodes {
        let left_child = children_left[node_id] as i32;
        let right_child = children_right[node_id] as i32;
        let is_leaf = left_child == -1;

        if is_leaf {
            node_type.push(TreeNodeType::kLeafNode);
        } else {
            node_type.push(TreeNodeType::kNumericalTestNode);
        }
        cleft.push(left_child);
        cright.push(right_child);

        if is_leaf {
            split_index.push(-1);
            threshold_col.push(0.0_f64);
            cmp.push(Operator::kNone);
        } else {
            split_index.push(feature[node_id] as i32);
            threshold_col.push(threshold[node_id]);
            cmp.push(Operator::kLE);
        }

        default_left.push(true);
        category_list_right_child.push(false);

        // Leaf values (`sklearn_bulk.cc:140-179`).
        let current_leaf_vec_size = leaf_vector.len() as u64;
        if is_leaf && has_leaf_vector {
            let base = node_id * leaf_vector_size;
            if is_classifier {
                for target in 0..n_targets as usize {
                    let mut norm_factor = 0.0_f64;
                    for class in 0..max_num_class as usize {
                        norm_factor += value[base + target * max_num_class as usize + class];
                    }
                    for class in 0..max_num_class as usize {
                        let v = value[base + target * max_num_class as usize + class];
                        let normalized = if norm_factor > 0.0 {
                            v / norm_factor
                        } else {
                            0.0
                        };
                        leaf_vector.push(normalized);
                    }
                }
            } else {
                for k in 0..leaf_vector_size {
                    leaf_vector.push(value[base + k]);
                }
            }
            leaf_vector_begin.push(current_leaf_vec_size);
            leaf_vector_end.push(leaf_vector.len() as u64);
            leaf_value.push(0.0);
        } else if is_leaf {
            leaf_value.push(value[node_id]);
            leaf_vector_begin.push(current_leaf_vec_size);
            leaf_vector_end.push(current_leaf_vec_size);
        } else {
            leaf_value.push(0.0);
            leaf_vector_begin.push(current_leaf_vec_size);
            leaf_vector_end.push(current_leaf_vec_size);
        }

        // Empty category list for all nodes (`sklearn_bulk.cc:181-184`).
        category_list_begin.push(0_u64);
        category_list_end.push(0_u64);

        // Node statistics (`sklearn_bulk.cc:186-190`): always present.
        data_count.push(n_node_samples[node_id] as u64);
        data_count_present.push(true);
        sum_hess.push(weighted_n_node_samples[node_id]);
        sum_hess_present.push(true);

        // Gain (`sklearn_bulk.cc:192-209`): impurity reduction for internals only.
        if !is_leaf {
            let sample_cnt = n_node_samples[node_id] as f64;
            let left_sample_cnt = n_node_samples[left_child as usize] as f64;
            let right_sample_cnt = n_node_samples[right_child as usize] as f64;
            let g = sample_cnt
                * (impurity[node_id]
                    - left_sample_cnt * impurity[left_child as usize] / sample_cnt
                    - right_sample_cnt * impurity[right_child as usize] / sample_cnt)
                / total_sample_cnt as f64;
            gain.push(g);
            gain_present.push(true);
        } else {
            gain.push(0.0);
            gain_present.push(false);
        }
    }

    let mut tree = Tree::<f64>::new();
    tree.node_type = TreeBuf::from_owned(node_type);
    tree.cleft = TreeBuf::from_owned(cleft);
    tree.cright = TreeBuf::from_owned(cright);
    tree.split_index = TreeBuf::from_owned(split_index);
    tree.default_left = TreeBuf::from_owned(default_left);
    tree.leaf_value = TreeBuf::from_owned(leaf_value);
    tree.threshold = TreeBuf::from_owned(threshold_col);
    tree.cmp = TreeBuf::from_owned(cmp);
    tree.category_list_right_child = TreeBuf::from_owned(category_list_right_child);
    tree.leaf_vector = TreeBuf::from_owned(leaf_vector);
    tree.leaf_vector_begin = TreeBuf::from_owned(leaf_vector_begin);
    tree.leaf_vector_end = TreeBuf::from_owned(leaf_vector_end);
    tree.category_list_begin = TreeBuf::from_owned(category_list_begin);
    tree.category_list_end = TreeBuf::from_owned(category_list_end);
    tree.data_count = TreeBuf::from_owned(data_count);
    tree.data_count_present = TreeBuf::from_owned(data_count_present);
    tree.sum_hess = TreeBuf::from_owned(sum_hess);
    tree.sum_hess_present = TreeBuf::from_owned(sum_hess_present);
    tree.gain = TreeBuf::from_owned(gain);
    tree.gain_present = TreeBuf::from_owned(gain_present);
    tree.has_categorical_split = false;
    tree.num_nodes = n_nodes as i32;
    tree
}

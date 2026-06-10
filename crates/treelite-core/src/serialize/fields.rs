//! Typed field accessors (SER-04) — read-only model/tree inspection.
//!
//! Ports the read surface of `treelite-mainline/src/field_accessor.cc`
//! (`GetHeaderField` / `GetTreeField`) as idiomatic typed Rust methods, per
//! RESEARCH Pattern 7: expose `model.num_feature()`, `model.num_tree()`,
//! `tree.threshold(nid)`, … now; defer the string-dispatch
//! `GetHeaderField(name) -> Frame` shape to Phase 8 (the Python binding).
//!
//! The SER-04 inspection surface is spread across three homes, all read-only:
//!
//! - **Model header (already `pub` fields):** `num_feature`, `task_type`,
//!   `average_tree_output`, `num_target`, `num_class`, `leaf_vector_shape`,
//!   `target_id`, `class_id`, `postprocessor`, `sigmoid_alpha`, `ratio_c`,
//!   `base_scores`, `attributes` — read directly.
//! - **Model bookkeeping (read-only methods on [`Model`], `model.rs`):**
//!   `major_ver()`, `minor_ver()`, `patch_ver()`, `num_tree()`,
//!   `threshold_type()`, `leaf_output_type()`, `num_opt_field_per_model()`.
//!   These are READ-ONLY upstream (`SetHeaderField` rejects them,
//!   field_accessor.cc:208-249) and DELIBERATELY carry no setter (T-02-J02):
//!   corrupting the version triple / `num_tree` / type tags would break
//!   serialize fidelity.
//! - **Per-tree node fields (methods on [`crate::tree::Tree`], `tree.rs`):**
//!   `left_child(nid)`, `right_child(nid)`, `split_index(nid)`, `is_leaf(nid)`,
//!   `leaf_value(nid)`, `threshold(nid)`, `comparison_op(nid)`,
//!   `node_type(nid)`, `default_left(nid)`, `has_leaf_vector(nid)`,
//!   `leaf_vector(nid)`, `category_list_right_child(nid)`,
//!   `category_list(nid)`, and the gated statistics
//!   `has_data_count`/`data_count`, `has_sum_hess`/`sum_hess`,
//!   `has_gain`/`gain`.
//!
//! This module adds the single missing typed header reader, `num_feature()`,
//! so the model-level read surface reads uniformly as method calls.

use crate::model::Model;

impl Model {
    /// Number of input features (`field_accessor.cc:38`; tree.h:535).
    ///
    /// A typed reader over the already-`pub` `num_feature` field, so the SER-04
    /// inspection surface reads uniformly as `model.num_feature()`.
    pub fn num_feature(&self) -> i32 {
        self.num_feature
    }
}

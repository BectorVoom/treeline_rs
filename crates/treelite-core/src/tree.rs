//! `Tree<T>` — struct-of-arrays decision tree (CORE-02).
//!
//! Ports `treelite-mainline/include/treelite/tree.h:78-335`. Every node field
//! is a separate parallel [`TreeBuf`] column indexed by node id — there is NO
//! `Node` struct (an explicit anti-pattern; a struct would break zero-copy
//! serialization in Phase 2). Move-only by intent (mirrors the upstream
//! deleted copy ctor); use [`Tree::deep_copy`] for an explicit deep copy.
//!
//! In Phase 1, `ThresholdType == LeafOutputType == T` (the upstream
//! `static_assert` at `tree.h:81-86` forbids mixed types). XGBoost-JSON always
//! yields `Tree<f32>`.

use crate::enums::{Operator, TreeNodeType};
use crate::tree_buf::TreeBuf;

/// A single decision tree stored as parallel SoA columns.
pub struct Tree<T: Copy> {
    // --- core node columns (tree.h:97-119) ---
    /// Node kind per node.
    pub node_type: TreeBuf<TreeNodeType>,
    /// Left child id; `-1` ⇒ leaf (`IsLeaf`, tree.h:204).
    pub cleft: TreeBuf<i32>,
    /// Right child id.
    pub cright: TreeBuf<i32>,
    /// Split feature index.
    pub split_index: TreeBuf<i32>,
    /// Missing-value direction (true ⇒ go left).
    pub default_left: TreeBuf<bool>,
    /// Scalar leaf output value.
    pub leaf_value: TreeBuf<T>,
    /// Numerical split threshold.
    pub threshold: TreeBuf<T>,
    /// Comparison operator (XGBoost always `kLT`).
    pub cmp: TreeBuf<Operator>,
    /// Categorical polarity (unused in Phase 1 fixture).
    pub category_list_right_child: TreeBuf<bool>,

    // --- leaf-vector columns (CSR-style; empty for binary:logistic) ---
    /// Flattened leaf vectors.
    pub leaf_vector: TreeBuf<T>,
    /// CSR begin offsets into `leaf_vector`.
    pub leaf_vector_begin: TreeBuf<u64>,
    /// CSR end offsets into `leaf_vector`.
    pub leaf_vector_end: TreeBuf<u64>,

    // --- category-list columns (CSR-style; empty for binary:logistic) ---
    /// Flattened category lists.
    pub category_list: TreeBuf<u32>,
    /// CSR begin offsets into `category_list`.
    pub category_list_begin: TreeBuf<u64>,
    /// CSR end offsets into `category_list`.
    pub category_list_end: TreeBuf<u64>,

    // --- node-statistic columns (optional; present-but-empty allowed) ---
    /// Per-node data count.
    pub data_count: TreeBuf<u64>,
    /// Per-node sum of hessians.
    pub sum_hess: TreeBuf<f64>,
    /// Per-node split gain.
    pub gain: TreeBuf<f64>,
    /// Presence flag for `data_count`.
    pub data_count_present: TreeBuf<bool>,
    /// Presence flag for `sum_hess`.
    pub sum_hess_present: TreeBuf<bool>,
    /// Presence flag for `gain`.
    pub gain_present: TreeBuf<bool>,

    // --- scalar tree fields ---
    /// Whether any node has a categorical split (tree.h:126).
    pub has_categorical_split: bool,
    /// Number of nodes in this tree (tree.h:158).
    pub num_nodes: i32,

    // --- serialization bookkeeping (tree.h:131-132) ---
    /// Optional-field count in the per-tree extension slot (tree.h:131);
    /// always serialized as `0` (RESEARCH § Per-tree #24).
    pub num_opt_field_per_tree: i32,
    /// Optional-field count in the per-node extension slot (tree.h:132);
    /// always serialized as `0` (RESEARCH § Per-tree #25).
    pub num_opt_field_per_node: i32,
}

impl<T: Copy> Tree<T> {
    /// Construct an empty tree (all columns empty, `num_nodes == 0`).
    pub fn new() -> Self {
        Tree {
            node_type: TreeBuf::empty(),
            cleft: TreeBuf::empty(),
            cright: TreeBuf::empty(),
            split_index: TreeBuf::empty(),
            default_left: TreeBuf::empty(),
            leaf_value: TreeBuf::empty(),
            threshold: TreeBuf::empty(),
            cmp: TreeBuf::empty(),
            category_list_right_child: TreeBuf::empty(),
            leaf_vector: TreeBuf::empty(),
            leaf_vector_begin: TreeBuf::empty(),
            leaf_vector_end: TreeBuf::empty(),
            category_list: TreeBuf::empty(),
            category_list_begin: TreeBuf::empty(),
            category_list_end: TreeBuf::empty(),
            data_count: TreeBuf::empty(),
            sum_hess: TreeBuf::empty(),
            gain: TreeBuf::empty(),
            data_count_present: TreeBuf::empty(),
            sum_hess_present: TreeBuf::empty(),
            gain_present: TreeBuf::empty(),
            has_categorical_split: false,
            num_nodes: 0,
            num_opt_field_per_tree: 0,
            num_opt_field_per_node: 0,
        }
    }

    // --- traversal getters (the contract, tree.h:169-235) ---

    /// Left child id of `nid` (`cleft[nid]`, tree.h:169).
    pub fn left_child(&self, nid: usize) -> i32 {
        self.cleft[nid]
    }

    /// Right child id of `nid` (`cright[nid]`, tree.h:176).
    pub fn right_child(&self, nid: usize) -> i32 {
        self.cright[nid]
    }

    /// Default (missing-value) child of `nid` (tree.h:183):
    /// `default_left[nid] ? cleft[nid] : cright[nid]`.
    pub fn default_child(&self, nid: usize) -> i32 {
        if self.default_left[nid] {
            self.cleft[nid]
        } else {
            self.cright[nid]
        }
    }

    /// Split feature index of `nid` (tree.h:190).
    pub fn split_index(&self, nid: usize) -> i32 {
        self.split_index[nid]
    }

    /// Whether `nid` is a leaf: `cleft[nid] == -1` (tree.h:204).
    pub fn is_leaf(&self, nid: usize) -> bool {
        self.cleft[nid] == -1
    }

    /// Scalar leaf output value of `nid` (tree.h:211).
    pub fn leaf_value(&self, nid: usize) -> T {
        self.leaf_value[nid]
    }

    /// Numerical split threshold of `nid`.
    pub fn threshold(&self, nid: usize) -> T {
        self.threshold[nid]
    }

    /// Comparison operator of `nid`.
    pub fn comparison_op(&self, nid: usize) -> Operator {
        self.cmp[nid]
    }

    /// Whether `nid` carries a leaf vector:
    /// `leaf_vector_begin[nid] != leaf_vector_end[nid]` (tree.h:233).
    pub fn has_leaf_vector(&self, nid: usize) -> bool {
        self.leaf_vector_begin[nid] != self.leaf_vector_end[nid]
    }
}

impl<T: Copy> Default for Tree<T> {
    fn default() -> Self {
        Self::new()
    }
}

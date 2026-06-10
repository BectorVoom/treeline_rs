//! Typed errors for the scikit-learn estimator loaders (ERR-01).
//!
//! Every upstream fatal path in the sklearn loaders (`TREELITE_CHECK*` /
//! `TREELITE_LOG(FATAL)`) becomes a returned `Err` here rather than a panic or
//! an out-of-bounds index. See:
//! - `treelite-mainline/src/model_loader/sklearn.cc:207-219`
//!   (`n_trees`/`n_features` positivity, `node_count <= INT_MAX`)
//! - `treelite-mainline/src/model_loader/sklearn_bulk.cc:240-241,322-323`
//!   (RF/ET `n_estimators`/`n_features` positivity)
//! - the bulk path bounds-checks `children_left`/`children_right` against
//!   `node_count` before indexing the per-node gain arrays (T-04-13).

use thiserror::Error;

/// Errors raised by the `treelite-sklearn` loaders.
#[derive(Debug, Error)]
pub enum SklError {
    /// A per-tree parallel array's length disagreed with that tree's
    /// `node_count`. Mirrors the implicit length contract of the upstream
    /// `double const**` / `std::int64_t const**` array-of-arrays: return a typed
    /// error instead of indexing out of bounds (ERR-01, ASVS V5).
    #[error(
        "field {field:?} has an incorrect dimension (tree {tree}): expected {expected}, got {got}"
    )]
    DimensionMismatch {
        /// The tree index whose array was malformed.
        tree: usize,
        /// The parallel-array field name (e.g. `"children_left"`).
        field: &'static str,
        /// Expected length (the tree's `node_count`).
        expected: usize,
        /// Actual length found in the supplied slice.
        got: usize,
    },

    /// A model scalar that must be positive/non-negative (`n_estimators`,
    /// `n_features`, `n_targets`, `n_classes`, `node_count`) was out of range.
    /// Casting such a value to `usize` would yield a huge size (e.g.
    /// `vec![-1; n as usize]` aborts with capacity overflow), so it becomes a
    /// typed `Err` here rather than an abort/panic (WR-02, ERR-01).
    ///
    /// Ports the upstream `TREELITE_CHECK_GT(n_trees, 0)` /
    /// `TREELITE_CHECK_GT(n_features, 0)` and the
    /// `node_count <= INT_MAX` overflow guard (`sklearn.cc:207-219`, T-04-14).
    #[error("model scalar {field:?} is out of range: {value} ({reason})")]
    InvalidScalar {
        /// The field whose value was invalid (e.g. `"n_estimators"`).
        field: &'static str,
        /// The offending value (as `i64` to carry `node_count`).
        value: i64,
        /// Why it was rejected (e.g. `"must be at least 1"`,
        /// `"exceeds i32::MAX"`).
        reason: &'static str,
    },

    /// A `children_left`/`children_right` index pointed outside the tree's node
    /// range (`< -1` or `>= node_count`). The bulk gain formula reads
    /// `n_node_samples[left_child]`/`impurity[left_child]`; an out-of-range child
    /// index would index out of bounds. Surface a typed error rather than an OOB
    /// access (T-04-13, Security Domain, ASVS V5).
    #[error(
        "child index out of range in tree {tree} node {node}: {child} (node_count {node_count})"
    )]
    ChildIndexOutOfRange {
        /// The tree index whose child pointer was malformed.
        tree: usize,
        /// The node id whose child pointer was malformed.
        node: usize,
        /// The offending child index.
        child: i64,
        /// The tree's `node_count` (valid children are `0..node_count`).
        node_count: usize,
    },

    /// The supplied `value` flat buffer was too short for the per-node leaf
    /// payload (`node_count * n_targets * max_num_class` for a vector-leaf model,
    /// or `node_count` for a scalar-leaf model). Surface a typed error rather
    /// than indexing out of bounds (ERR-01).
    #[error("value buffer too short in tree {tree}: expected at least {expected}, got {got}")]
    ValueBufferTooShort {
        /// The tree index whose value buffer was too short.
        tree: usize,
        /// The minimum required element count.
        expected: usize,
        /// The actual element count supplied.
        got: usize,
    },

    /// The number of `per-tree` outer slices disagreed with `n_estimators` /
    /// `n_iter` (the declared tree count). Surface a typed error rather than
    /// indexing past the supplied outer slice (ERR-01).
    #[error("expected {expected} trees ({field:?}), got {got} outer slices")]
    TreeCountMismatch {
        /// The field whose outer length was wrong (e.g. `"children_left"`).
        field: &'static str,
        /// Expected outer length (`n_estimators` / `n_iter`).
        expected: usize,
        /// Actual outer length supplied.
        got: usize,
    },

    /// A failure decoding the HistGradientBoosting packed-node byte buffer
    /// (SKL-04). Covers: an `expected_sizeof_node_struct` not in {52, 56}
    /// (T-04-18), a `nodes` buffer shorter than `node_count × itemsize`
    /// (T-04-18), a `feature_idx` out of range for `features_map`/`categories_map`
    /// (T-04-19), a `bitset_idx` out of range for the categorical bitmap
    /// (T-04-20), and a short field read mid-decode. The packed buffer is
    /// untrusted; every itemsize, length, and index is validated and a typed
    /// error returned rather than an out-of-bounds read (Security Domain, D-08,
    /// ASVS V5).
    #[error("histgb packed-node decode failed at offset {offset}: {detail}")]
    HistGbDecode {
        /// The byte offset (within a node record, or `0` for whole-buffer/scalar
        /// guards) at which the failure was detected.
        offset: usize,
        /// What went wrong (itemsize, buffer length, or index detail).
        detail: String,
    },

    /// An error bubbled up from `treelite-core` (e.g. an unknown enum string).
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),

    /// An error bubbled up from the `treelite-builder` `ModelBuilder` while the
    /// GradientBoosting MixIn path emitted node/tree calls (e.g. a dangling
    /// child key, an orphaned node, or a split index out of range). The
    /// builder's strict validation surfaces here as a typed loader error rather
    /// than a panic crossing the loader boundary (ERR-01, T-04-13).
    #[error(transparent)]
    Builder(#[from] treelite_builder::BuilderError),
}

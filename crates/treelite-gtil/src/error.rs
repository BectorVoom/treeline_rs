//! Typed errors for the GTIL predict crate (ERR-01).
//!
//! Every upstream fatal/abort path during traversal (an out-of-bounds
//! feature/node index, an unrecognized comparison operator, an unsupported
//! postprocessor name) becomes a returned [`GtilError`] here rather than a
//! panic. This is the ASVS V5 / T-03-01 mitigation: a malformed `Model` must
//! never index out of bounds.

use thiserror::Error;

/// Errors raised by `treelite-gtil` during prediction.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum GtilError {
    /// A node's `split_index` is outside the feature row. Upstream this would
    /// be an unchecked `row(split_index)` access; here it is a typed error
    /// (T-03-01).
    #[error("feature index {feature} at node {node} is out of bounds (num_feature = {num_feature})")]
    FeatureIndexOutOfBounds {
        /// The node id whose split referenced the bad feature.
        node: usize,
        /// The offending feature index read from `split_index`.
        feature: i32,
        /// The number of features available in the row.
        num_feature: i32,
    },

    /// A child id during traversal is outside the tree's node range
    /// (a malformed `cleft`/`cright`).
    #[error("node index {node} is out of bounds")]
    NodeIndexOutOfBounds {
        /// The offending node id.
        node: usize,
    },

    /// The `data` buffer is too small for the declared `num_row × num_feature`
    /// shape (or the product overflows `usize`). Upstream would slice the row
    /// matrix unchecked; here it is a typed error so a malformed `num_feature`
    /// (WR-01) never reaches an out-of-bounds slice and panics (T-03-01).
    #[error(
        "input buffer too small for shape: num_row = {num_row}, num_feature = {num_feature} \
         requires {required} elements, got {got}"
    )]
    InvalidInputShape {
        /// Declared number of rows.
        num_row: usize,
        /// Declared number of features per row (from `model.num_feature`).
        num_feature: usize,
        /// Number of elements the shape requires (`num_row * num_feature`),
        /// or `usize::MAX` if that product overflowed.
        required: usize,
        /// Number of elements actually present in `data`.
        got: usize,
    },

    /// The model's `postprocessor` name is not supported in Phase 1
    /// (only `identity` and `sigmoid`). Upstream silently maps unknown names;
    /// here it is a typed error (T-03-02).
    #[error("unsupported postprocessor: {0:?}")]
    UnsupportedPostprocessor(String),

    /// An error bubbled up from `treelite-core`.
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),
}

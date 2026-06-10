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
    #[error(
        "feature index {feature} at node {node} is out of bounds (num_feature = {num_feature})"
    )]
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

    /// A tree's `target_id`/`class_id` routes an output cell outside the
    /// `(num_target, max_num_class)` buffer (T-04-03). Upstream indexes the
    /// `Array3DView` unchecked; here a malformed route is a typed error so it
    /// never produces an out-of-bounds write.
    #[error(
        "output route (target_id = {target_id}, class_id = {class_id}) is out of bounds \
         (num_target = {num_target}, max_num_class = {max_num_class})"
    )]
    OutputRouteOutOfBounds {
        /// The routing `target_id` read from `model.target_id[tree]`.
        target_id: i32,
        /// The routing `class_id` read from `model.class_id[tree]`.
        class_id: i32,
        /// The model's `num_target`.
        num_target: i32,
        /// The model's `max_num_class` (max over `num_class`).
        max_num_class: i32,
    },

    /// A leaf vector is shorter than the `(target, class)` slots its routing
    /// requires (a malformed `leaf_vector_shape`). Upstream reads the leaf view
    /// unchecked; here it is a typed error rather than an OOB read.
    #[error("leaf vector too short: needed {needed} elements, got {got}")]
    LeafVectorTooShort {
        /// Minimum leaf-vector length the routing requires.
        needed: usize,
        /// Actual leaf-vector length.
        got: usize,
    },

    /// The model's `postprocessor` name is not supported by this GTIL surface.
    /// Upstream silently maps unknown names; here it is a typed error (T-03-02).
    #[error("unsupported postprocessor: {0:?}")]
    UnsupportedPostprocessor(String),

    /// The requested [`PredictKind`](crate::PredictKind) is not yet wired on this
    /// GTIL surface. `LeafId` and `ScorePerTree` are added in Plan 05-04; until
    /// then they surface as this typed error rather than producing wrong output.
    #[error("unsupported predict kind: {kind}")]
    UnsupportedPredictKind {
        /// The name of the unsupported kind (e.g. `"LeafId"`).
        kind: &'static str,
    },

    /// An error bubbled up from `treelite-core`.
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),
}

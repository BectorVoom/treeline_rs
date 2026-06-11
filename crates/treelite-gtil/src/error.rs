//! Typed errors for the GTIL predict crate (ERR-01).
//!
//! Every upstream fatal/abort path during traversal (an out-of-bounds
//! feature/node index, an unrecognized comparison operator, an unsupported
//! postprocessor name) becomes a returned [`GtilError`] here rather than a
//! panic. This is the ASVS V5 / T-03-01 mitigation: a malformed `Model` must
//! never index out of bounds.

use thiserror::Error;
use treelite_core::Operator;

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

    /// A sparse-CSR `col_ind[k]` names a feature column outside
    /// `[0, num_feature)`. Upstream would write `scratch[col_ind[k]]` unchecked
    /// (`SparseMatrixAccessor::GetRow`, `predict.cc:84`); here the column is
    /// validated before the scratch write so a corrupt `col_ind` can never
    /// produce an out-of-bounds write (T-05-09).
    #[error("sparse column index {col} is out of bounds (num_feature = {num_feature})")]
    SparseColumnOutOfBounds {
        /// The offending column index read from `col_ind`.
        col: u64,
        /// The number of feature columns (`model.num_feature`).
        num_feature: u64,
    },

    /// A sparse-CSR `row_ptr` is malformed: wrong length (`!= num_row + 1`),
    /// non-monotone, or its trailing fence exceeds the `data`/`col_ind` backing
    /// length. Upstream slices `data[row_ptr[r]..row_ptr[r+1]]` unchecked
    /// (`predict.cc:76-79`); here `row_ptr` is validated up front so a corrupt
    /// offset array can never produce an out-of-bounds / inverted slice
    /// (T-05-10).
    #[error("sparse row_ptr is invalid at index {index}: value {value} violates limit {limit}")]
    SparseRowPtrInvalid {
        /// The `row_ptr` index where the violation was detected (the offending
        /// fence, or `num_row` for a length / trailing-fence violation).
        index: usize,
        /// The offending value (the fence, the `row_ptr` length, or the
        /// trailing total).
        value: u64,
        /// The limit it violated (the previous fence for monotonicity, the
        /// required length, or the backing-array length).
        limit: u64,
    },

    /// A numerical-test node carries an unrecognized comparison operator
    /// (notably [`Operator::kNone`], which is never emitted by a well-formed
    /// numerical test node). Upstream `NextNode` (`predict.cc:120-122`) hits a
    /// fatal `TREELITE_CHECK(false)` and returns `-1`; here the malformed
    /// operator is a typed error rather than a silent route-right wrong
    /// prediction (WR-05, ERR-01).
    #[error("unrecognized comparison operator {op:?} at node {node}")]
    UnrecognizedOperator {
        /// The node id whose numerical test carried the bad operator.
        node: usize,
        /// The offending operator read from `tree.comparison_op(node)`.
        op: Operator,
    },

    /// A node's category-list CSR offsets are malformed: an inverted slice
    /// (`begin > end`), an end past the `category_list` value buffer, or a
    /// missing begin/end offset for an in-range node. Upstream slices
    /// `category_list[begin..end]` unchecked because the loader guarantees
    /// well-formed offsets (`predict.cc:128-150`); a hand-crafted / corrupt
    /// `Model` must surface a typed error instead of silently treating the node
    /// as a non-match and changing the prediction (WR-04, ERR-01).
    #[error("malformed category-list offsets at node {node}")]
    MalformedCategoryList {
        /// The node id whose category-list CSR offsets are malformed.
        node: usize,
    },

    /// A leaf node's leaf-vector CSR offsets are present but malformed: an
    /// inverted slice (`begin > end`) or an end past the `leaf_vector` value
    /// buffer. An ABSENT offset (a scalar-leaf tree with empty CSR columns) is
    /// NOT malformed — it is the legitimate scalar path. A present-but-inverted
    /// offset must surface a typed error instead of silently being treated as a
    /// scalar leaf and changing the prediction (WR-04, ERR-01).
    #[error("malformed leaf-vector offsets at node {node}")]
    MalformedLeafVector {
        /// The leaf node id whose leaf-vector CSR offsets are malformed.
        node: usize,
    },

    /// A bounded scoped thread pool (sized by `Config.nthread > 0`) failed to
    /// build. T-10-01: the Wave-1 scoped-pool builder maps
    /// `rayon::ThreadPoolBuilder::build()`'s `Err` to this typed variant so a
    /// pool-construction failure surfaces as a `GtilError`, never a panic
    /// (ERR-01). Carries the underlying builder error message.
    #[error("failed to build thread pool: {0}")]
    ThreadPool(String),

    /// An error bubbled up from `treelite-core`.
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),
}

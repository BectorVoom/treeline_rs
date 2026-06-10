//! Typed errors for the LightGBM text-format loader (ERR-01).
//!
//! Every upstream fatal path in the LightGBM loader (`TREELITE_LOG(FATAL)` /
//! `TREELITE_CHECK*`) becomes a returned `Err` here rather than a panic or an
//! out-of-bounds index. See:
//! - `treelite-mainline/src/model_loader/detail/lightgbm.h:54`
//!   (unknown objective name)
//! - `treelite-mainline/src/model_loader/lightgbm.cc:282-414`
//!   (missing keys / malformed counts)
//! - `treelite-mainline/src/model_loader/lightgbm.cc:491-492`
//!   (`sigmoid_alpha > 0` check)
//!
//! Mirrors the `treelite-xgboost` [`XgbError`](treelite_xgboost) enum shape: a
//! `DimensionMismatch`, an `UnrecognizedObjective`, transparent `Core`/`Builder`
//! bridges, plus a LightGBM `Parse { line, detail }` variant (the analog of the
//! XGBoost `Legacy { pos, detail }` positional parse error).

use thiserror::Error;

/// Errors raised by the `treelite-lightgbm` loader.
#[derive(Debug, Error)]
pub enum LgbError {
    /// A required key was missing, a numeric token failed to parse, or a
    /// space-delimited array was shorter than its declared count. Upstream this
    /// is a `TREELITE_CHECK` / `TextToNumber` `TREELITE_LOG(FATAL)`; here it is a
    /// typed, recoverable error carrying the offending logical line / key.
    ///
    /// `line` identifies WHERE parsing failed (a `key=value` line, the tree
    /// index, or the global section); `detail` is the human-readable cause.
    #[error("malformed LightGBM model ({line}): {detail}")]
    Parse {
        /// Where parsing failed (e.g. `"Tree 0 key 'leaf_value'"`, `"global"`).
        line: String,
        /// Human-readable cause (missing key, bad number, short array).
        detail: String,
    },

    /// A per-tree parallel array's length disagreed with the expected count
    /// derived from `num_leaves`. Mirrors the XGBoost `DimensionMismatch`: return
    /// a typed error instead of indexing out of bounds (ERR-01, ASVS V5, T-04-07).
    #[error(
        "field {field:?} has an incorrect dimension (tree {tree}): expected {expected}, got {got}"
    )]
    DimensionMismatch {
        /// The tree index whose array was malformed.
        tree: usize,
        /// The parallel-array field name (e.g. `"left_child"`).
        field: &'static str,
        /// Expected length (derived from `num_leaves`).
        expected: usize,
        /// Actual length found in the model text.
        got: usize,
    },

    /// A decoded negative-index leaf (`leaf_value[!old_node_id]`) fell outside
    /// the `leaf_value` array. Upstream this would be an out-of-bounds vector
    /// access; here it is a typed error (ERR-01, T-04-08).
    #[error(
        "leaf index {index} out of range (tree {tree} has {num_leaves} leaf values)"
    )]
    LeafIndexOutOfRange {
        /// The tree index whose leaf decode went out of range.
        tree: usize,
        /// The decoded leaf index that was out of range.
        index: usize,
        /// The number of leaf values available (`leaf_value.len()`).
        num_leaves: usize,
    },

    /// A node's `split_feature` index fell outside the parsed arrays, or a
    /// child index referenced a non-existent node. Upstream this would be an
    /// out-of-bounds access during the BFS re-numbering; here it is typed
    /// (ERR-01, T-04-07).
    #[error("node index {index} out of range in tree {tree}: {detail}")]
    NodeIndexOutOfRange {
        /// The tree index whose node decode went out of range.
        tree: usize,
        /// The offending index.
        index: i64,
        /// Human-readable cause.
        detail: &'static str,
    },

    /// The `sigmoid:<alpha>` parameter parsed to a non-positive value (or was
    /// absent for an objective that requires it). Ports the upstream
    /// `sigmoid_alpha > 0` check (`lightgbm.cc:491-492`, T-04-09): reject a
    /// degenerate sigmoid with a typed error rather than silently using it.
    #[error("invalid sigmoid_alpha for objective {objective:?}: must be > 0, got {alpha}")]
    InvalidSigmoidAlpha {
        /// The objective that required a valid `sigmoid_alpha`.
        objective: String,
        /// The offending parsed alpha (or a sentinel when absent).
        alpha: f64,
    },

    /// A categorical split's bitset slicing was malformed: the `cat_threshold`
    /// length disagreed with `cat_boundaries.back()`, a node's categorical index
    /// (`threshold[node]` cast to an index) fell outside `cat_boundaries`, or a
    /// boundary pair was non-monotone. Upstream this would be an out-of-bounds
    /// access into `cat_threshold` / `cat_boundaries` during `BitsetToList`
    /// slicing (`lightgbm.cc:563-573`); here it is a typed, recoverable error so
    /// a crafted model can never trigger an OOB slice (T-04-10, T-04-11, ASVS V5).
    #[error("malformed categorical split (tree {tree}): {detail}")]
    Bitset {
        /// The tree index whose categorical split was malformed.
        tree: usize,
        /// Human-readable cause (length mismatch, out-of-range cat index, ...).
        detail: String,
    },

    /// The objective name is not one of the recognized LightGBM objectives.
    /// Upstream this is a `TREELITE_LOG(FATAL)` (`lightgbm.h:54`); here it is a
    /// typed `Err` per ERR-01.
    #[error("unrecognized LightGBM objective: {0:?}")]
    UnrecognizedObjective(String),

    /// An error bubbled up from `treelite-core` (e.g. an unknown enum string).
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),

    /// An error bubbled up from the `treelite-builder` `ModelBuilder` while the
    /// loader emitted node/tree calls (e.g. a dangling child key, an orphaned
    /// node, or a split index out of range). The builder's strict validation
    /// runs after the loader's own count checks as defense-in-depth; its failures
    /// surface here as a typed loader error rather than a panic crossing the
    /// loader boundary (ERR-01, D-11).
    #[error(transparent)]
    Builder(#[from] treelite_builder::BuilderError),
}

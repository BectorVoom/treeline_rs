//! Typed errors for the XGBoost-JSON loader (ERR-01).
//!
//! Every upstream fatal path in the XGBoost loader (`TREELITE_LOG(FATAL)` /
//! `TREELITE_LOG(ERROR)` + `return false`) becomes a returned `Err` here
//! rather than a panic or an out-of-bounds index. See:
//! - `treelite-mainline/src/model_loader/detail/xgboost.cc:47`
//!   (unrecognized objective)
//! - `treelite-mainline/src/model_loader/detail/xgboost_json/delegated_handler.cc:411-432`
//!   (per-tree array-length mismatch)

use thiserror::Error;

/// Errors raised by the `treelite-xgboost` loader.
#[derive(Debug, Error)]
pub enum XgbError {
    /// `serde_json` failed to parse the model JSON, or a JSON scalar could not
    /// be decoded into its expected shape. Upstream the RapidJSON SAX parser
    /// aborts; here it is a typed, recoverable error.
    #[error("malformed XGBoost-JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// A numeric scalar param stored as a JSON string failed to parse
    /// (e.g. `num_feature`, `base_score`). XGBoost-JSON stores these as strings.
    #[error("could not parse field {field:?} value {value:?}: {source}")]
    ParseScalar {
        /// The field whose string value failed to parse.
        field: &'static str,
        /// The offending raw string value.
        value: String,
        /// The underlying parse error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A model scalar that must be non-negative (e.g. `num_target`,
    /// `num_feature`, `num_class`) parsed to a negative value. Casting such a
    /// value to `usize` would yield a huge size (e.g. `vec![1; num_target as
    /// usize]` aborts with capacity overflow), so it becomes a typed `Err` here
    /// rather than an abort/panic (WR-02, ERR-01).
    #[error("model scalar {field:?} must be non-negative, got {value}")]
    InvalidScalar {
        /// The field whose value was negative (e.g. `"num_target"`).
        field: &'static str,
        /// The offending parsed value.
        value: i32,
    },

    /// A per-tree parallel array's length disagreed with `tree_param.num_nodes`.
    /// Mirrors `delegated_handler.cc:411-432`: return an error instead of
    /// indexing out of bounds (ERR-01, ASVS V5 input validation).
    #[error(
        "field {field:?} has an incorrect dimension (tree {tree}): expected {expected}, got {got}"
    )]
    DimensionMismatch {
        /// The tree index whose array was malformed.
        tree: usize,
        /// The parallel-array field name (e.g. `"left_children"`).
        field: &'static str,
        /// Expected length (`tree_param.num_nodes`).
        expected: usize,
        /// Actual length found in the JSON.
        got: usize,
    },

    /// The objective name is not one of the recognized XGBoost objectives.
    /// Upstream this is a `TREELITE_LOG(FATAL)` (`xgboost.cc:47`); here it is a
    /// typed `Err` per ERR-01.
    #[error("unrecognized XGBoost objective: {0:?}")]
    UnrecognizedObjective(String),

    /// An error bubbled up from `treelite-core` (e.g. an unknown enum string).
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),

    /// An error bubbled up from the `treelite-builder` `ModelBuilder` while the
    /// loader emitted node/tree calls (e.g. a dangling child key, an orphaned
    /// node, or a split index out of range). The builder's strict validation
    /// runs after the loader's own `require_non_negative`/`check_dim` checks as
    /// defense-in-depth; its failures surface here as a typed loader error
    /// rather than a panic crossing the loader boundary (ERR-01, D-11).
    #[error(transparent)]
    Builder(#[from] treelite_builder::BuilderError),
}

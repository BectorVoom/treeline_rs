//! Typed errors for the `treelite-builder` crate (ERR-01, D-07).
//!
//! Every upstream fatal path in `treelite-mainline/src/model_builder/model_builder.cc`
//! (a `TREELITE_CHECK*` / `TREELITE_LOG(FATAL)`) becomes a returned [`BuilderError`]
//! here rather than an abort/panic. Per D-07 ("errors at the offending call site
//! with good locality"), each variant carries the offending node key / index /
//! split index so the caller can pinpoint the malformed input. See:
//! - `model_builder.cc:157` (negative node key)
//! - `model_builder.cc:162` (duplicate node key)
//! - `model_builder.cc:176-179` (negative / self / equal child keys)
//! - `model_builder.cc:181-182` (split_index out of range)
//! - `model_builder.cc:121-126` (dangling child key at EndTree)
//! - `model_builder.cc:139-144` (orphaned node at EndTree)
//! - `model_builder.cc:217-229` (leaf-output length mismatch)
//! - `model_builder.cc:280-285` (CommitModel state / tree-count mismatch)
//! - `model_builder.cc:308-333` (illegal state transition)

use thiserror::Error;

/// Errors raised by the `treelite-builder` `ModelBuilder`, `concatenate`, and
/// `bulk_construct_tree` paths.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BuilderError {
    /// A non-leaf node's child key references a node that was never declared in
    /// the tree. Detected at `EndTree` after forward-reference resolution
    /// (`model_builder.cc:121-126`).
    #[error("node with key {key} not found (dangling child reference)")]
    DanglingChildKey {
        /// The unresolved child key.
        key: i32,
    },

    /// A declared node is unreachable from the root (index 0). Detected at
    /// `EndTree` by the orphan mark-and-sweep (`model_builder.cc:139-144`).
    #[error("node with key {key} is orphaned -- it cannot be reached from the root node")]
    OrphanedNode {
        /// The orphaned node's user-defined key.
        key: i32,
    },

    /// The same node key was used twice within one tree
    /// (`model_builder.cc:162`).
    #[error("node key {key} is duplicated")]
    DuplicateNodeKey {
        /// The duplicated key.
        key: i32,
    },

    /// A node key (or child key) is negative (`model_builder.cc:157,176`).
    #[error("node key {key} cannot be negative")]
    NegativeNodeKey {
        /// The offending negative key.
        key: i32,
    },

    /// The node was declared both a leaf and a test, or two test/leaf details
    /// were supplied for one node. The state machine makes the second call
    /// illegal (`model_builder.cc:308-333`).
    #[error("node with key {key} cannot be both a leaf and a test")]
    LeafTestConflict {
        /// The conflicting node's key.
        key: i32,
    },

    /// A test node's two child keys are equal, or a child key equals the node's
    /// own key (`model_builder.cc:177-179`).
    #[error("node {node} uses a child key equal to itself or to its sibling")]
    SelfOrEqualChildKey {
        /// The current node's key.
        node: i32,
    },

    /// A test node's `split_index` is `>= num_feature` (`model_builder.cc:181-182`).
    #[error("split_index {split_index} must be less than num_feature ({num_feature})")]
    SplitIndexOutOfRange {
        /// The offending split index.
        split_index: i32,
        /// The model's declared feature count.
        num_feature: i32,
    },

    /// A `leaf_vector` length disagreed with the metadata's expected leaf size
    /// (`model_builder.cc:229`).
    #[error("expected leaf output of length {expected}, got {got}")]
    LeafVectorSizeMismatch {
        /// The expected leaf-vector length (product of `leaf_vector_shape`).
        expected: usize,
        /// The actual length supplied.
        got: usize,
    },

    /// A method was called in a state where it is illegal (the state machine
    /// rejected the transition) (`model_builder.cc:308-333`).
    #[error("unexpected builder call; expected {expected}")]
    WrongState {
        /// A human-readable description of what call(s) were expected here.
        expected: &'static str,
    },

    /// `CommitModel` was called before exactly `expected_num_tree` trees had been
    /// produced (`model_builder.cc:283-285`).
    #[error("expected {expected} trees but got {got} instead")]
    CommitTreeCountMismatch {
        /// The number of trees declared in the metadata.
        expected: usize,
        /// The number of trees actually built.
        got: usize,
    },

    /// `CommitModel` was called before metadata was initialized
    /// (`model_builder.cc:281-282`).
    #[error("the model does not yet have valid metadata; call initialize_metadata() first")]
    MetadataNotInitialized,

    /// A tree was ended with zero nodes (`model_builder.cc:107-108`).
    #[error("cannot have an empty tree; supply at least one node")]
    EmptyTree,

    /// `concatenate` was given an input whose `Model` variant discriminant
    /// differs from `objs[0]` (`model_concat.cc:47-49`).
    #[error("model object at index {index} has a different variant type than the first model")]
    VariantMismatch {
        /// The index of the mismatching input model.
        index: usize,
    },

    /// `concatenate` was given an input whose `num_target` / `num_class` /
    /// `leaf_vector_shape` differs from `objs[0]` (`model_concat.cc:50-58`).
    #[error("model object at index {index} has a different {field} than the first model")]
    HeaderMismatch {
        /// The index of the mismatching input model.
        index: usize,
        /// Which header field disagreed.
        field: &'static str,
    },

    /// f32 and f64 entry points were mixed within one builder. The numeric mode
    /// is latched on first f64 use; a builder produces exactly one of
    /// `ModelVariant::F32` / `ModelVariant::F64`. Mixing would silently downcast
    /// or discard one type's values, breaking the 1e-5 fidelity gate, so it is
    /// rejected (Plan 04-01, RESEARCH Open Q2 / Pitfall 1).
    #[error(
        "cannot mix f32 and f64 entry points in one builder (this builder is already in {existing} mode)"
    )]
    MixedNumericMode {
        /// The numeric mode already latched on this builder (`"f32"` or `"f64"`).
        existing: &'static str,
    },

    /// An error bubbled up from `treelite-core`.
    #[error(transparent)]
    Core(#[from] treelite_core::CoreError),
}

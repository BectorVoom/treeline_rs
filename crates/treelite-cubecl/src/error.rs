//! Typed errors for the `treelite-cubecl` kernel crate (ERR-01).
//!
//! Mirrors the [`treelite_gtil::GtilError`] discipline: every fatal/abort path
//! a cubecl host launcher might hit (an out-of-bounds feature/node index, a
//! malformed input shape, an unsupported model construct routed away from the
//! kernels) becomes a returned [`CubeclError`] rather than a panic. This is a
//! library crate: `thiserror` only (no error-aggregation dependency), never a
//! `panic!` on a malformed `Model` (the C++/CLAUDE.md error-handling contract).

use thiserror::Error;

/// Errors raised by `treelite-cubecl` host launchers and uploaders.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CubeclError {
    /// The `data` buffer does not match the declared `num_row × num_feature`
    /// shape (or the product overflows `usize`). Mirrors
    /// [`treelite_gtil::GtilError::InvalidInputShape`] at the kernel boundary so
    /// a malformed shape is a typed error, never an out-of-bounds device slice.
    #[error(
        "input buffer too small for shape: num_row = {num_row}, num_feature = {num_feature} \
         requires {required} elements, got {got}"
    )]
    InvalidInputShape {
        /// Declared number of rows.
        num_row: usize,
        /// Declared number of features per row.
        num_feature: usize,
        /// Number of elements the shape requires (`num_row * num_feature`), or
        /// `usize::MAX` if that product overflowed.
        required: usize,
        /// Number of elements actually present in `data`.
        got: usize,
    },

    /// A node's `split_index` is outside the feature row (T-03-01 at the
    /// kernel boundary).
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

    /// A child id during traversal is outside the tree's node range (a
    /// malformed `cleft`/`cright`).
    #[error("node index {node} is out of bounds")]
    NodeIndexOutOfBounds {
        /// The offending node id.
        node: usize,
    },

    /// A leaf node's `[leaf_vector_begin, leaf_vector_end)` span does not lie
    /// within its tree's leaf-vector segment (an inverted span, or an end past
    /// the segment length, or a broadcast span the kernel would read past the
    /// segment). Caught HOST-side in
    /// [`crate::upload::validate_leaf_vectors`] BEFORE any device op, so a
    /// malformed model returns a typed error instead of an out-of-bounds device
    /// read (T-06-09). Mirrors the scalar twin's
    /// `GtilError::LeafVectorTooShort`/`MalformedLeafVector` discipline.
    #[error(
        "malformed leaf vector at tree {tree} node {node}: span [{begin}, {end}) does not fit \
         the tree's leaf-vector segment length {segment_len}"
    )]
    MalformedLeafVector {
        /// The tree whose leaf node carries the bad span.
        tree: usize,
        /// The leaf node id (relative to the tree) with the bad span.
        node: usize,
        /// The offending `leaf_vector_begin` value.
        begin: u32,
        /// The offending `leaf_vector_end` (or broadcast end) value.
        end: u32,
        /// The tree's leaf-vector segment length the span must fit within.
        segment_len: u32,
    },

    /// A backend was compiled in (its cargo feature is enabled) but no device
    /// is present at runtime. A typed SKIP the caller branches on (D-05): the
    /// harness/report marks the row "not run — no device", NEVER a silent CPU
    /// fallback (that would hide which backend actually ran). The `&'static str`
    /// backend tag keeps the enum's `PartialEq, Eq` derives intact. Produced by
    /// the per-backend constructors in [`crate::device`] when client
    /// construction fails on a missing device.
    #[error("no device available for the {backend} backend (skip, not a failure)")]
    DeviceUnavailable {
        /// The selected backend's feature name (`"rocm"` / `"cuda"` / `"wgpu"`).
        backend: &'static str,
    },

    /// A model construct is not yet supported on the cubecl path (sparse CSR,
    /// categorical splits, or — until Wave 3 — the launcher itself). Routed to
    /// the scalar fallback by the host, never a panic. Catch-all so the
    /// host launcher can degrade gracefully (D-02).
    #[error("unsupported on the cubecl backend: {0}")]
    Unsupported(String),
}

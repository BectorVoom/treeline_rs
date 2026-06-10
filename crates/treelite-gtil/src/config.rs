//! Typed GTIL prediction configuration (D-06).
//!
//! Idiomatic Rust mirror of upstream `gtil::Configuration` (`gtil.h:49-55`) and
//! `PredictKind` (`gtil.h:26-47`). Unlike upstream, this crate does NOT parse a
//! JSON config string — the entry points take this typed [`Config`] directly. Any
//! JSON-config compatibility shim belongs at the Phase-8 PyO3 edge, never in the
//! compute crate (D-06).

/// Which prediction the GTIL engine produces (`enum class PredictKind`,
/// `gtil.h:26-47`). The discriminant ordering mirrors the upstream values
/// `kPredictDefault = 0 .. kPredictPerTree = 3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictKind {
    /// Sum over trees and apply post-processing (`kPredictDefault = 0`). Output
    /// dims `(num_row, num_target, max_num_class)`.
    Default,
    /// Sum over trees but skip post-processing — raw margin scores
    /// (`kPredictRaw = 1`). Output dims `(num_row, num_target, max_num_class)`.
    Raw,
    /// One integer leaf ID per tree (`kPredictLeafID = 2`). Output dims
    /// `(num_row, num_tree)`. (Wired in Plan 05-04.)
    LeafId,
    /// One or more margin scores per tree (`kPredictPerTree = 3`). Output dims
    /// `(num_row, num_tree, leaf_vector_shape[0] * leaf_vector_shape[1])`.
    /// (Wired in Plan 05-04.)
    ScorePerTree,
}

impl Default for PredictKind {
    #[inline]
    fn default() -> Self {
        // gtil.h:52 — `PredictKind pred_kind{kPredictDefault}`.
        PredictKind::Default
    }
}

/// GTIL predictor configuration (`struct Configuration`, `gtil.h:49-55`).
///
/// `nthread <= 0` means "use all threads" upstream
/// (`detail/threading_utils.h:74-80`). The scalar reference engine is
/// single-threaded, so `nthread` is **accepted and recorded but never used to
/// allocate** — there is no thread-count-driven allocation and therefore no DoS
/// amplification surface (RESEARCH Pitfall 6 / threat T-05-05).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Config {
    /// Which prediction to produce. Defaults to [`PredictKind::Default`].
    pub kind: PredictKind,
    /// Requested thread count. `<= 0` means "all threads" upstream; the scalar
    /// reference ignores it for allocation (recorded only).
    pub nthread: i32,
}

impl Default for Config {
    #[inline]
    fn default() -> Self {
        // gtil.h:51-52 — `int nthread{0}` and default `kPredictDefault`.
        Config {
            kind: PredictKind::Default,
            nthread: 0,
        }
    }
}

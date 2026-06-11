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
/// `nthread` sizes the rayon worker pool the scalar engine row-parallelizes over
/// (Phase 10, PAR-04):
/// - `nthread <= 0` → use ALL cores (the global rayon pool, upstream
///   `MaxNumThread` "use all threads" semantics, `detail/threading_utils.h:74-80`);
/// - `nthread > 0` → a per-call SCOPED pool bounded to exactly `nthread` workers
///   (`rayon::ThreadPoolBuilder::num_threads(n)`).
///
/// The per-worker scratch is `num_feature`-sized (NOT `nthread`-driven heap
/// growth), so a large `nthread` bounds the worker count but creates no DoS
/// amplification surface (T-10-01). A pool-build failure surfaces as the typed
/// `GtilError::ThreadPool`, never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Config {
    /// Which prediction to produce. Defaults to [`PredictKind::Default`].
    pub kind: PredictKind,
    /// Requested worker-thread count for the row-parallel scalar engine.
    /// `<= 0` means "use all cores" (global pool); `N > 0` bounds a scoped pool
    /// to exactly `N` workers (PAR-04).
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

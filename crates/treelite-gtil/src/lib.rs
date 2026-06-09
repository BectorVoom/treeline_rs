//! GTIL reference inference engine — scalar, single-threaded predict.
//!
//! Runs reference prediction over a [`treelite_core::Model`], porting the
//! upstream GTIL traversal and assembly order VERBATIM
//! (`treelite-mainline/src/gtil/predict.cc`). Per D-08, predict is a plain
//! function — there is NO `Predictor`/backend trait in Phase 1 (deferred to
//! Phase 6).

pub mod error;
pub mod postprocessor;

pub use error::GtilError;

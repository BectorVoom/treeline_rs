//! XGBoost-JSON model loader (Wave 2 — stub for Wave 1).
//!
//! This crate parses XGBoost-JSON models into a `treelite_core::Model`.
//! It is wired into the workspace in Wave 1 so `cargo build --workspace`
//! succeeds; the loader itself is implemented in a later wave.

/// Placeholder marker proving the crate compiles and links `treelite-core`.
pub fn crate_name() -> &'static str {
    "treelite-xgboost"
}

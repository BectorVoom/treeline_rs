//! GTIL reference inference engine (Wave 3 — stub for Wave 1).
//!
//! This crate runs reference prediction over a `treelite_core::Model`.
//! It is wired into the workspace in Wave 1 so `cargo build --workspace`
//! succeeds; the predict engine itself is implemented in a later wave.

/// Placeholder marker proving the crate compiles and links `treelite-core`.
pub fn crate_name() -> &'static str {
    "treelite-gtil"
}

//! Equivalence harness (Wave 3 — stub for Wave 1).
//!
//! This crate drives the load -> predict -> compare-to-golden 1e-5
//! equivalence check. It is wired into the workspace in Wave 1 so
//! `cargo build --workspace` succeeds; the harness itself is implemented
//! in a later wave.

/// Placeholder marker proving the crate compiles and links its deps.
pub fn crate_name() -> &'static str {
    "treelite-harness"
}

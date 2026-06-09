//! THE spine test (Success Criterion 4): load → represent → predict →
//! verify-against-golden, end-to-end, within `1e-5`.
//!
//! Loads the committed `fixtures/binary_logistic.model.json` via the Rust
//! pipeline (`treelite_xgboost::load_xgboost_json` → `treelite_gtil::predict`)
//! over the golden's committed input matrix and asserts every output element is
//! within `1e-5` of the frozen upstream `fixtures/golden.json`. This closes the
//! walking skeleton — the entire core fidelity value, proven end-to-end.
//!
//! Uses `anyhow::Result` as the test return type (ERR-02) so every step
//! propagates with `?` and a context chain. The hard `1e-5` gate lives inside
//! `run_equivalence`'s `assert_abs_diff_eq!`; this test additionally asserts
//! `max_dev < 1e-5` to make the gate explicit in the test body.

use std::path::Path;

/// Resolve a path under `fixtures/` relative to the workspace root.
///
/// `CARGO_MANIFEST_DIR` is `crates/treelite-harness`; the workspace root is two
/// levels up, where `fixtures/` lives.
fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

#[test]
fn equivalence_within_1e5() -> anyhow::Result<()> {
    let model_path = fixture_path("binary_logistic.model.json");
    let golden_path = fixture_path("golden.json");

    let golden = treelite_harness::load_golden(&golden_path)?;

    // Diagnose (not fail) on environment drift from the capture environment.
    treelite_harness::check_manifest(&golden.manifest);

    let max_dev = treelite_harness::run_equivalence(&model_path, &golden)?;
    println!("max observed |delta| = {max_dev:e}");

    // Make the 1e-5 gate explicit in the test body (the hard assertion is also
    // enforced element-wise inside run_equivalence).
    assert!(
        max_dev < 1e-5,
        "max observed |delta| ({max_dev:e}) exceeds the 1e-5 equivalence gate"
    );

    Ok(())
}

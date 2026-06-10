//! THE byte-fidelity test (D-01 / D-02): the Rust v5 serializer reproduces the
//! frozen upstream `fixtures/golden_v5.bin` byte-for-byte.
//!
//! ## What this proves
//!
//! The load-bearing D-01/D-02 assertion is that the Rust serializer emits the v5
//! wire format in EXACT `serializer.cc` field order, width, and framing, so that
//! a model serializes to bytes identical to the upstream `treelite==4.7.0` wheel.
//! The authoritative, model-source-independent proof of that is the
//! **golden round-trip**: `serialize(deserialize(golden_v5.bin)) == golden_v5.bin`
//! byte-for-byte. This exercises the full header + 25-column per-tree walk over a
//! real upstream blob and fails on any transposed/mis-framed/mis-width field.
//!
//! ## The loader path is now also byte-exact (DEF-02-01 closed, Plan 03-02)
//!
//! The golden was captured from upstream `treelite.frontend.load_xgboost_model`,
//! whose loader populates per-node statistics (`sum_hess`, `gain`) and the
//! present-but-empty CSR-offset / `category_list_right_child` columns, sets leaf
//! `split_index = -1`, and stamps `attributes = "{}"`. Plan 03-02 closed the
//! Phase-1 loader-fidelity gap: the Rust XGBoost loader now emits `sum_hess` on
//! every node and `gain` on internal nodes, and passes `attributes = None`
//! (→ `"{}"`). So `serialize(load_xgboost_json(...))` is now byte-identical to
//! the upstream golden too — `loader_path_reproduces_golden_v5_byte_for_byte`
//! below asserts that as a HARD gate (no longer a non-fatal diagnostic), and the
//! cross-format single-golden close lives in `three_format_equivalence.rs`.
//!
//! Uses `anyhow::Result` (ERR-02) so each step propagates with a context chain.

use std::path::Path;

use anyhow::Context;

/// Resolve a path under `fixtures/` relative to the workspace root
/// (mirrors `equivalence.rs:21-27`).
fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// First offset at which `got` and `want` differ (or a length-difference point).
fn first_diff(got: &[u8], want: &[u8]) -> Option<usize> {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return Some(i);
        }
    }
    if got.len() != want.len() {
        return Some(n);
    }
    None
}

/// D-01/D-02: `serialize(deserialize(golden)) == golden` byte-for-byte.
///
/// The serializer's field order/width/framing is proven against a real upstream
/// v5 blob, independent of any model-construction path.
#[test]
fn serializer_reproduces_golden_v5_byte_for_byte() -> anyhow::Result<()> {
    let golden_path = fixture_path("golden_v5.bin");
    let golden = std::fs::read(&golden_path).with_context(|| format!("reading {golden_path}"))?;

    let mut model = treelite_core::deserialize(&golden)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("deserializing golden_v5.bin")?;
    let produced = treelite_core::serialize_to_buffer(&mut model);

    if let Some(off) = first_diff(&produced, &golden) {
        panic!(
            "v5 serialization diverges from golden_v5.bin at offset {off}: \
             produced={:?} golden={:?} (produced len {}, golden len {})",
            produced.get(off),
            golden.get(off),
            produced.len(),
            golden.len()
        );
    }
    assert_eq!(
        produced, golden,
        "serialize(deserialize(golden_v5.bin)) must equal the blob byte-for-byte (D-01/D-02)"
    );
    Ok(())
}

/// DEF-02-01 (closed, Plan 03-02): the JSON loader path is byte-identical to the
/// upstream golden.
///
/// `serialize(load_xgboost_json(binary_logistic.model.json)) == golden_v5.bin`
/// byte-for-byte. This was a NON-fatal diagnostic in Phase 1/2 because the loader
/// then left the per-node stat / CSR-offset columns empty and stamped empty
/// `attributes`; Plan 03-02 closed that gap (sum_hess/gain on the right nodes,
/// `attributes = None` → `"{}"`), so this is now a HARD assertion. A failure here
/// means a real loader-fidelity regression — it must be fixed in the loader, not
/// masked. `first_diff` reports the precise divergence offset.
#[test]
fn loader_path_reproduces_golden_v5_byte_for_byte() -> anyhow::Result<()> {
    let model_path = fixture_path("binary_logistic.model.json");
    let golden_path = fixture_path("golden_v5.bin");

    let model_json =
        std::fs::read_to_string(&model_path).with_context(|| format!("reading {model_path}"))?;
    let mut model = treelite_xgboost::load_xgboost_json(&model_json)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading fixture model")?;
    let produced = treelite_core::serialize_to_buffer(&mut model);
    let golden = std::fs::read(&golden_path).with_context(|| format!("reading {golden_path}"))?;

    if let Some(off) = first_diff(&produced, &golden) {
        panic!(
            "loader-path serialization diverges from golden_v5.bin at offset {off}: \
             produced={:?} golden={:?} (produced len {}, golden len {}). DEF-02-01 \
             regression — the XGBoost loader no longer reproduces the upstream golden.",
            produced.get(off),
            golden.get(off),
            produced.len(),
            golden.len()
        );
    }
    assert_eq!(
        produced, golden,
        "serialize(load_xgboost_json(binary_logistic)) must equal golden_v5.bin \
         byte-for-byte (DEF-02-01 closed, D-10)"
    );
    Ok(())
}

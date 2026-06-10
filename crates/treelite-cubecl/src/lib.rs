//! `treelite-cubecl` — the cubecl-accelerated GTIL kernel crate (GPU-01).
//!
//! This crate reimplements the GTIL inference hot path (numerical-dense
//! traversal + the 10 postprocessors) as `cubecl` `#[cube(launch)]` kernels,
//! defaulting to the cubecl **CPU** runtime (`features = ["cpu"]`) and
//! validated to 1e-5 against the frozen scalar golden matrix. Sparse CSR and
//! categorical splits ride the scalar [`treelite_gtil`] fallback this phase
//! (D-02).
//!
//! Wave layout (this crate is filled in across plans 06-01..06-05):
//! - Wave 0 (this plan): crate scaffold, [`CubeclError`], the [`predict_cpu`]
//!   host-launcher stub, and the placeholder [`upload`]/[`kernels`] modules.
//! - Wave 1: the descend spike (`tests/spike.rs`).
//! - Wave 2: per-column ragged-SoA upload ([`upload`]).
//! - Wave 3: the `#[cube(launch)]` kernels ([`kernels`]) + a real
//!   [`predict_cpu`].
//! - Wave 4: determinism + the `gtil_matrix_cubecl` harness sibling.

pub mod error;

/// Per-column ragged-SoA host→device upload (Wave 2). Placeholder for now; the
/// cubecl 0.10.0 API name-pinning lives in `upload.rs` as a header comment so
/// Waves 2-3 author against the confirmed method names, not an assumption.
pub mod upload;

/// The `#[cube(launch)]` traversal + postprocessor kernels (Wave 3).
/// Placeholder module — kernels land once the spike (Wave 1) and upload
/// (Wave 2) contracts are green.
pub mod kernels;

pub use error::CubeclError;

use treelite_core::Model;
use treelite_gtil::Config;

/// Host launcher for cubecl CPU-backend prediction.
///
/// Mirrors the shape of [`treelite_gtil::predict`]:
/// `(&Model, &[F], num_row, &Config) -> Result<Vec<F>, _>`. The kernel body is
/// authored in Wave 3; until then this is a typed stub that reports the path is
/// not yet wired (the host caller routes to the scalar fallback). `F` is bound
/// minimally so the crate compiles without pulling any `cubecl` symbol into the
/// stub body yet — the dependency is declared (and resolved by the Task 1
/// `cargo build`), but the kernel surface is not referenced until Wave 3.
pub fn predict_cpu<F: Copy>(
    _model: &Model,
    _data: &[F],
    _num_row: usize,
    _cfg: &Config,
) -> Result<Vec<F>, CubeclError> {
    Err(CubeclError::Unsupported(
        "predict_cpu kernel not yet implemented (lands in Wave 3 / plan 06-04)".to_string(),
    ))
}

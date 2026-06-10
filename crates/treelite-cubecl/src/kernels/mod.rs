//! cubecl `#[cube]` / `#[cube(launch)]` traversal + postprocessor kernels.
//!
//! Wave 2 (this plan, 06-02) authors the break-free [`traversal::descend`]
//! helper — the VERBATIM `#[cube]` descent shape that the full per-predict-kind
//! launch kernels in Wave 3 reuse unchanged (registration-not-refactor, D-11).
//! The remaining kernel surface (the 10 postprocessor ports, `predict_default`
//! / `predict_raw` / `predict_leaf_id` / `predict_score_per_tree` launch
//! kernels, and the real [`crate::predict_cpu`] body) lands in plan 06-04 once
//! the Wave 1 spike (`tests/spike.rs`) and Wave 2 upload contracts are green.
//! Authored against the cubecl 0.10.0 API names pinned in `upload.rs`.

/// Break-free numerical tree descent (`#[cube]` helper, ported from
/// `treelite_gtil::evaluate_tree`). The launch kernels in Wave 3 call this
/// verbatim.
pub mod traversal;

/// The ten GTIL postprocessors as `#[cube]` helpers (D-03), porting
/// `treelite_gtil::postprocessor` verbatim with respect to cast order (the 1e-5
/// contract, CR-01). Authored in Wave 3 / plan 06-03.
pub mod postproc;

/// `#[cube(launch)]` kernels for the `Default` / `Raw` predict kinds (fused
/// traversal + accumulate + RF-average + f64 base-score; the `Default`
/// postprocessor is a separate device step selected host-side). Authored in plan
/// 06-04.
pub mod default_raw;

/// `#[cube(launch)]` kernel for the `LeafId` predict kind (one leaf node id per
/// `(row, tree)`). Authored in plan 06-04.
pub mod leaf_id;

/// `#[cube(launch)]` kernel for the `ScorePerTree` predict kind (raw per-tree
/// leaf data into a `(num_row, num_tree, lvs)` buffer). Authored in plan 06-04.
pub mod score_per_tree;

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

// Reserved for later waves (kept as empty module declarations so the public
// module path is stable and Wave 3 only fills bodies, never re-wires the tree):
//   pub mod postproc;        // the 10 postprocessor `#[cube]` ports (Wave 3)
//   pub mod default_raw;     // predict_default / predict_raw launch kernels
//   pub mod leaf_id;         // predict_leaf_id launch kernel
//   pub mod score_per_tree;  // predict_score_per_tree launch kernel

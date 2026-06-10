//! `#[cube(launch)]` kernel for the `ScorePerTree` predict kind
//! (`PredictScoreByTree`, `predict.cc:347-378`), one unit per row, serial trees.
//!
//! Writes the RAW per-tree leaf data into a row-major `(num_row, num_tree, lvs)`
//! buffer where `lvs = leaf_vector_shape[0] * leaf_vector_shape[1]` (>= 1). Per
//! `(row, tree)`:
//!
//! - if the reached leaf carries a leaf VECTOR (`begin != end`), write each
//!   element `tree.leaf_vector(leaf)[i]` at `(row, tree, i)`
//!   (`predict.cc:367-370`);
//! - else write the scalar `tree.leaf_value(leaf)` at `(row, tree, 0)`
//!   (`predict.cc:372`; scalar-leaf models have third-dim size 1).
//!
//! There is NO postprocessor, NO RF averaging, and NO base-score add for this
//! kind. The output buffer is pre-zeroed by the host, so a scalar leaf leaves the
//! padding columns at 0 (`std::fill_n(.., InputT{})`, `predict.cc:355`).
//!
//! Determinism (SC1/SC2): disjoint per-row writes; no atomic / plane reduction /
//! `sync_cube` over the tree axis. `descend` reused verbatim (D-11). Generic over
//! the input element `F` and the threshold/leaf element `T` (Pitfall 6).

use cubecl::prelude::*;

use crate::kernels::traversal::descend;

/// Per-`(row, tree)` raw leaf data (scalar or leaf-vector), one unit per row,
/// serial trees, into the `(num_row, num_tree, lvs)` output.
///
/// The per-tree leaf vector lives at `leaf_vector[leafvec_off[t] + (begin..end)]`
/// where `begin`/`end` are the per-node CSR offsets RELATIVE to the tree's
/// leaf-vector base (mirroring `upload::concat_columns`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn predict_score_per_tree<F: Float, T: Float>(
    cleft: &Array<i32>,
    cright: &Array<i32>,
    split_index: &Array<i32>,
    threshold: &Array<T>,
    leaf_value: &Array<T>,
    leaf_vector: &Array<T>,
    leaf_vector_begin: &Array<u32>,
    leaf_vector_end: &Array<u32>,
    default_left: &Array<u32>,
    node_off: &Array<u32>,
    leafvec_off: &Array<u32>,
    input: &Array<F>,
    output: &mut Array<F>,
    num_row: u32,
    num_tree: u32,
    lvs: u32,
    num_feature: u32,
) {
    let row = ABSOLUTE_POS as u32;
    if row < num_row {
        let row_off = row * num_feature;
        for tree_id in 0..num_tree {
            let base = node_off[tree_id as usize];
            let leaf = descend::<F, T>(
                cleft,
                cright,
                split_index,
                threshold,
                default_left,
                base,
                row_off,
                input,
            );
            let slot = (row * num_tree + tree_id) * lvs;
            let lvb = leaf_vector_begin[(base + leaf) as usize];
            let lve = leaf_vector_end[(base + leaf) as usize];
            if lvb != lve {
                // Leaf vector: write each element at (row, tree, i), bounded by
                // lvs (the per-tree output slot width).
                let lv_base = leafvec_off[tree_id as usize] + lvb;
                let count = lve - lvb;
                let mut i: u32 = 0;
                while i < count {
                    if i < lvs {
                        output[(slot + i) as usize] =
                            F::cast_from(leaf_vector[(lv_base + i) as usize]);
                    }
                    i += 1;
                }
            } else {
                // Scalar leaf → index 0 (predict.cc:372).
                output[slot as usize] = F::cast_from(leaf_value[(base + leaf) as usize]);
            }
        }
    }
}

//! `#[cube(launch)]` kernel for the `LeafId` predict kind (`PredictLeaf`,
//! `predict.cc:325-345`), one unit per row, serial trees.
//!
//! Writes one integer leaf NODE id per `(row, tree)` into a row-major
//! `(num_row, num_tree)` output buffer: `output[row * num_tree + tree_id] =
//! leaf_id`. Upstream stores the node id in the SAME `Array2DView<InputT>` output
//! buffer (`predict.cc:329,340`), so the id is the float-typed node id in the
//! output element width `F`. There is NO postprocessor, NO RF averaging, and NO
//! base-score add for this kind.
//!
//! Determinism (SC1/SC2): each unit writes only its own row's disjoint cells; no
//! atomic / plane reduction / `sync_cube` over the tree axis. The `descend`
//! helper is reused verbatim from plan 06-02 (D-11).
//!
//! Generic over BOTH the input element `F` (the feature matrix + output) and the
//! preset's threshold element `T` (Pitfall 6).

use cubecl::prelude::*;

use crate::kernels::traversal::descend;

/// Per-`(row, tree)` leaf NODE id, one unit per row, serial trees.
///
/// The reached leaf is the raw `nid` RELATIVE to the tree's `node_off[t]` base —
/// exactly the `leaf_id` upstream stores (`predict.cc:340`), not the
/// concatenated global index. It is cast into the output element `F`
/// (`O::from_leaf_f64(leaf as f64)`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn predict_leaf_id<F: Float, T: Float>(
    cleft: &Array<i32>,
    cright: &Array<i32>,
    split_index: &Array<i32>,
    threshold: &Array<T>,
    default_left: &Array<u32>,
    node_off: &Array<u32>,
    input: &Array<F>,
    output: &mut Array<F>,
    num_row: u32,
    num_tree: u32,
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
            // output_view(row, tree) = leaf_id (the tree-relative node id),
            // cast into the output element F (predict.cc:340).
            output[(row * num_tree + tree_id) as usize] = F::cast_from(leaf);
        }
    }
}

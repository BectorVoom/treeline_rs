//! `#[cube(launch)]` kernels for the `Default` and `Raw` predict kinds
//! (numerical-dense path), one unit per row, serial trees (D-01 / GTIL-08).
//!
//! These reproduce `treelite_gtil::predict_preset` (`lib.rs:643-741`) line by
//! line for the general `(num_row, num_target, max_num_class)` output shape:
//!
//! 1. zero-filled per-row output cells (`std::fill_n(.., InputT{})`);
//! 2. SERIAL `for tree_id in 0..num_tree` accumulation into the row's
//!    `(target,class)` cell(s) via the four-way `OutputLeafValue` /
//!    `OutputLeafVector` branch on `(target_id[tree], class_id[tree])`, including
//!    the multiclass leaf-vector broadcast (`predict.cc:174-229`);
//! 3. RF averaging when `average_tree_output` (`predict.cc:259-293`);
//! 4. the f64 2D base-score add per `(target,class)` cell (`predict.cc:294-304`).
//!
//! The postprocessor for the `Default` kind is applied by a SEPARATE device step
//! (the plan-03 `#[cube]` postproc helpers, selected host-side by the
//! postprocessor name); the `Raw` kind skips it. The kind branch therefore lives
//! on the HOST (kernel selection), never in-kernel (RESEARCH "Alternatives
//! Considered: separate kernels per kind").
//!
//! Determinism (SC1/SC2): each unit writes ONLY its own row's disjoint output
//! cells, with no cross-unit accumulation primitive over the tree axis — float
//! add is non-associative (GTIL-08), so the serial tree-sum order is preserved
//! exactly as the scalar reference.
//!
//! Generic over BOTH the input element `F` (the feature matrix + output) and the
//! preset's threshold/leaf element `T` (Pitfall 6, mirroring
//! `predict_preset<T, O>`). The spike used matching widths; the general kernel
//! reads the threshold/leaf columns in `T` and the input in `F`, casting `T`
//! leaves into `F` at the accumulate site (`static_cast<InputT>`,
//! `predict.cc:228`).

use cubecl::prelude::*;

use crate::kernels::traversal::descend;

/// Fused traversal + accumulate + RF-average + f64 base-score kernel for the
/// `Default` / `Raw` kinds. One unit per row; serial trees.
///
/// Column layout (the ragged-SoA concatenation built by `upload::concat_columns`):
/// tree `t`'s node `n` lives at `concat[node_off[t] + n]`; the per-tree leaf
/// vector lives at `leaf_vector[leafvec_off[t] + (begin..end)]`. The per-node CSR
/// offsets `leaf_vector_begin`/`leaf_vector_end` are RELATIVE to the tree's
/// leaf-vector base, so element `i` of node `nid`'s leaf vector is
/// `leaf_vector[leafvec_off[t] + leaf_vector_begin[node_off[t] + nid] + i]`.
///
/// `target_id`/`class_id` are the per-tree routing columns (length `num_tree`);
/// `num_class` is the per-target class count (length `num_target`). `base_scores`
/// is the `(num_target, max_num_class)` f64 base-score plane. `average_factor` is
/// the precomputed per-cell RF divisor (length `num_target * max_num_class`); it
/// is `1` for every cell when `average_tree_output` is false (a divide by 1).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn predict_default_raw<F: Float, T: Float>(
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
    target_id: &Array<i32>,
    class_id: &Array<i32>,
    num_class: &Array<i32>,
    base_scores: &Array<f64>,
    average_factor: &Array<f64>,
    input: &Array<F>,
    output: &mut Array<F>,
    num_row: u32,
    num_tree: u32,
    num_target: u32,
    max_num_class: u32,
    num_feature: u32,
) {
    let row = ABSOLUTE_POS as u32;
    if row < num_row {
        let cells_per_row = num_target * max_num_class;
        let row_base = row * cells_per_row;
        let row_off = row * num_feature;

        // 1. Zero this row's output cells (std::fill_n InputT{}).
        let mut z: u32 = 0;
        while z < cells_per_row {
            output[(row_base + z) as usize] = F::new(0.0);
            z += 1;
        }

        // 2. Serial tree accumulation in tree_id order (GTIL-08 — no reorder).
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
            let tid = target_id[tree_id as usize];
            let cid = class_id[tree_id as usize];

            // HasLeafVector: begin != end for the reached leaf (tree.h:233).
            let lvb = leaf_vector_begin[(base + leaf) as usize];
            let lve = leaf_vector_end[(base + leaf) as usize];
            let lv_base = leafvec_off[tree_id as usize] + lvb;

            if lvb != lve {
                // ---- OutputLeafVector (predict.cc:174-216) ----
                if tid == -1i32 && cid == -1i32 {
                    // leaf is (num_target, max_num_class): broadcast across cells.
                    let mut t: u32 = 0;
                    while t < num_target {
                        let nc = num_class[t as usize] as u32;
                        let mut c: u32 = 0;
                        while c < nc {
                            let li = t * max_num_class + c;
                            let cell = row_base + t * max_num_class + c;
                            let v = F::cast_from(leaf_vector[(lv_base + li) as usize]);
                            output[cell as usize] += v;
                            c += 1;
                        }
                        t += 1;
                    }
                } else {
                    if tid == -1i32 {
                        // leaf is (num_target, 1); route into class_id.
                        let mut t: u32 = 0;
                        while t < num_target {
                            let cell = row_base + t * max_num_class + cid as u32;
                            let v = F::cast_from(leaf_vector[(lv_base + t) as usize]);
                            output[cell as usize] += v;
                            t += 1;
                        }
                    } else {
                        if cid == -1i32 {
                            // leaf is (1, max_num_class); route into target_id.
                            let nc = num_class[tid as usize] as u32;
                            let mut c: u32 = 0;
                            while c < nc {
                                let cell = row_base + tid as u32 * max_num_class + c;
                                let v = F::cast_from(leaf_vector[(lv_base + c) as usize]);
                                output[cell as usize] += v;
                                c += 1;
                            }
                        } else {
                            // leaf is (1, 1); single cell.
                            let cell = row_base + tid as u32 * max_num_class + cid as u32;
                            let v = F::cast_from(leaf_vector[lv_base as usize]);
                            output[cell as usize] += v;
                        }
                    }
                }
            } else {
                // ---- OutputLeafValue (predict.cc:218-229) ----
                // Scalar leaf routes into the single (target_id, class_id) cell.
                let cell = row_base + tid as u32 * max_num_class + cid as u32;
                let v = F::cast_from(leaf_value[(base + leaf) as usize]);
                output[cell as usize] += v;
            }
        }

        // 3. RF averaging: divide each cell by its precomputed factor
        //    (predict.cc:259-293). When average_tree_output is false the host
        //    fills average_factor with 1.0, so this is a divide by 1 (no-op).
        let mut t: u32 = 0;
        while t < num_target {
            let nc = num_class[t as usize] as u32;
            let mut c: u32 = 0;
            while c < nc {
                let li = t * max_num_class + c;
                let factor = average_factor[li as usize];
                if factor != 0.0 {
                    let cell = row_base + li;
                    // output[cell] /= factor — in the output element width F
                    // (float /= float / double /= double). factor rides as f64;
                    // cast to F at the divide (matches O::div_by_count).
                    output[cell as usize] /= F::cast_from(factor);
                }
                c += 1;
            }
            t += 1;
        }

        // 4. f64 2D base-score add per (target, class) cell (predict.cc:294-304).
        //    InputT_view += double_view: promote F->f64, add, narrow back to F
        //    (matches O::add_base_score).
        let mut t2: u32 = 0;
        while t2 < num_target {
            let nc = num_class[t2 as usize] as u32;
            let mut c: u32 = 0;
            while c < nc {
                let li = t2 * max_num_class + c;
                let cell = row_base + li;
                let acc = f64::cast_from(output[cell as usize]) + base_scores[li as usize];
                output[cell as usize] = F::cast_from(acc);
                c += 1;
            }
            t2 += 1;
        }
    }
}

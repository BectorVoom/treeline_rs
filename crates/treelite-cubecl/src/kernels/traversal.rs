//! Break-free `#[cube]` numerical tree descent.
//!
//! This is the cubecl re-expression of the scalar
//! `treelite_gtil::evaluate_tree` (`crates/treelite-gtil/src/lib.rs:432-502`)
//! restricted to the numerical-dense path (categorical / sparse ride the scalar
//! fallback this phase, D-02). It is authored ONCE here and reused verbatim by
//! the Wave-3 launch kernels (`predict_default` etc., D-11
//! registration-not-refactor).
//!
//! cubecl 0.10.0 control-flow / math constraints that shape this code
//! (RESEARCH "Anti-Patterns to Avoid" + the CubeCL manual):
//! - the loop-skip keyword is unsupported in 0.10.0; the descent is naturally
//!   skip-free — every iteration computes and assigns the next node. We use a
//!   `while !is_leaf` loop with no early-exit/loop-skip keyword at all.
//! - if-STATEMENTS that assign a `let mut next`, never an `if`-EXPRESSION value
//!   (an `if`-expr inside `#[cube]` fails E0308 `ExpandElementTyped vs {float}`).
//! - NaN detection is the self-inequality `fv != fv` (NaN is the only value not
//!   equal to itself). The `Float`-trait NaN associated fn returns
//!   `Self::WithScalar<bool>` (an associated type, not a plain `bool`) for a
//!   generic `F: Float`, so the associated-fn form fails E0308 (`expected bool,
//!   found associated type`). `fv != fv` lowers to the same in-kernel NaN test
//!   and is the verbatim equivalent of `evaluate_tree`'s `fvalue.is_nan_val()`.
//!   Math intrinsics elsewhere still use the associated-fn form `F::exp(x)` /
//!   `F::exp2(x)` (NOT the equivalent method call, which fails E0599
//!   `__expand_exp_method`).
//! - `u32`/`i32` for indices/counters; cast to `usize` only at the array-index
//!   site (`arr[idx as usize]`).
//! - `default_left` is uploaded as a `u32` 0/1 column (cubecl `Array` has no
//!   natural `bool` element — RESEARCH Pitfall 4); compared `== 1u32` in-kernel.

use cubecl::prelude::*;

/// Descend one numerical tree from its root to a leaf, break-free.
///
/// Ports `evaluate_tree`'s numerical path line-by-line:
/// - leaf test: `cleft[node] == -1` (the leaf sentinel);
/// - NaN feature → the default child (`default_left == 1` ? left : right),
///   matching `predict.cc:158-159`;
/// - otherwise XGBoost's always-`kLT` route: `fvalue < threshold ? left : right`
///   (`predict.cc`, `next_node` lib.rs:333-355).
///
/// The columns are the ragged-SoA concatenation across the whole forest: tree
/// `t`'s node `n` lives at `concat[base + n]` where `base == tree_node_offset[t]`
/// (the prefix-sum index built host-side in Wave 2 upload). `row_off` is the
/// row's base offset into the flat `(num_row, num_feature)` `input` matrix, so
/// feature `fi` of the current row is `input[row_off + fi]`.
///
/// Returns the leaf node id (relative to this tree's `base`, i.e. the raw `nid`
/// the caller adds `base` to when reading `leaf_value[base + nid]`).
///
/// `F` is the element width of BOTH the input matrix and the threshold column
/// (the spike uses matching widths: `<f32,f32>`+f32 input and `<f64,f64>`+f64
/// input). The Wave-3 generalization to distinct input/threshold widths
/// (`predict_preset<T, O>`) is layered on top of this same control-flow shape.
#[cube]
pub fn descend<F: Float>(
    cleft: &Array<i32>,
    cright: &Array<i32>,
    split_index: &Array<i32>,
    threshold: &Array<F>,
    default_left: &Array<u32>,
    base: u32,
    row_off: u32,
    input: &Array<F>,
) -> u32 {
    let mut nid: u32 = 0;
    // Break-free bounded descent. `cleft[base + nid] != -1` is the
    // `!tree.is_leaf(nid)` loop guard. No early-exit/loop-skip keyword — every
    // iteration assigns `nid` exactly once.
    while cleft[(base + nid) as usize] != -1i32 {
        let fi = split_index[(base + nid) as usize];
        let fv = input[(row_off + fi as u32) as usize];
        // Default route is RIGHT; if-statements (never an if-expr value) flip it.
        let mut next: i32 = cright[(base + nid) as usize];
        // NaN test via self-inequality (`fv != fv`): NaN is the only value not
        // equal to itself. Verbatim equivalent of `evaluate_tree`'s
        // `fvalue.is_nan_val()`; avoids the `Float`-trait NaN associated fn whose
        // `WithScalar<bool>` return is an associated type (not plain `bool`) on
        // generic `F` in cubecl 0.10.0.
        if fv != fv {
            // Missing value → default child (predict.cc:158-159). default_left is
            // the u32 0/1 column (Pitfall 4): 1 ⇒ left, else the right default.
            if default_left[(base + nid) as usize] == 1u32 {
                next = cleft[(base + nid) as usize];
            }
        } else {
            // XGBoost always kLT: fvalue < threshold ? left : right.
            if fv < threshold[(base + nid) as usize] {
                next = cleft[(base + nid) as usize];
            }
        }
        nid = next as u32;
    }
    nid
}

//! GTIL reference inference engine — scalar, single-threaded predict.
//!
//! Runs reference prediction over a [`treelite_core::Model`], porting the
//! upstream GTIL traversal and assembly order VERBATIM
//! (`treelite-mainline/src/gtil/predict.cc`). Per D-08, predict is a plain
//! function — there is NO `Predictor`/backend trait in Phase 1 (deferred to
//! Phase 6).

pub mod error;
pub mod postprocessor;

pub use error::GtilError;

use treelite_core::{Model, ModelVariant, Operator, Tree};

/// Conversion of an `f32` feature value into a tree's threshold/leaf domain `T`,
/// plus a cast of `T` back to `f32` for accumulation. This reifies the upstream
/// template instantiation choices so the f32/f64 cast ordering is explicit:
///
/// - For `Tree<f32>`, `NextNode<InputT=float, ThresholdT=float>` compares
///   `float < float` — `from_f32` is the identity.
/// - For `Tree<f64>`, `NextNode<InputT=float, ThresholdT=double>` compares
///   `float < double`, where C++ promotes the `float` operand to `double`;
///   `from_f32` therefore widens `f32 → f64`.
///
/// `to_f32` mirrors `static_cast<InputT>(tree.LeafValue(leaf_id))` at
/// `predict.cc:228` (leaf value cast to the f32 accumulator type).
pub trait PredictScalar: Copy {
    /// Lift an `f32` feature value into this threshold domain for comparison.
    fn from_f32(v: f32) -> Self;
    /// Narrow this leaf value to the `f32` output accumulator type.
    fn to_f32(self) -> f32;
}

impl PredictScalar for f32 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

impl PredictScalar for f64 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// The `NextNode` comparison switch (`predict.cc:99-124`).
///
/// `fvalue` (already lifted into the threshold domain `T`) is compared against
/// `threshold` with `op`; returns `left` if the condition holds, else `right`.
/// XGBoost always emits `kLT`. The categorical branch (`NextNodeCategorical`)
/// is intentionally NOT ported in Phase 1 (deferred to Phase 5).
#[inline]
fn next_node<T: PartialOrd>(fvalue: T, threshold: T, op: Operator, left: i32, right: i32) -> i32 {
    let cond = match op {
        Operator::kLT => fvalue < threshold,
        Operator::kLE => fvalue <= threshold,
        Operator::kEQ => fvalue == threshold,
        Operator::kGT => fvalue > threshold,
        Operator::kGE => fvalue >= threshold,
        // kNone is never emitted by a numerical test node; treat as no-match
        // (mirrors the upstream default arm returning right_child after the
        // fatal check — Phase 1 fixtures never reach here).
        Operator::kNone => false,
    };
    if cond { left } else { right }
}

/// Scalar node-0 traversal (`EvaluateTree`, `predict.cc:152-172`).
///
/// Walks from node 0 until a leaf is reached, routing NaN features to the
/// default child and otherwise dispatching through [`next_node`]. Returns the
/// leaf node id. Out-of-bounds `split_index` returns a typed
/// [`GtilError::FeatureIndexOutOfBounds`] rather than panicking (T-03-01).
fn evaluate_tree<T: PredictScalar + PartialOrd>(
    tree: &Tree<T>,
    row: &[f32],
) -> Result<usize, GtilError> {
    // Upper bound on valid node ids; `num_nodes` is the loader-produced count
    // (`tree.h:158`). A child id outside `[0, num_nodes)` is a malformed
    // `cleft`/`cright` and must become a typed error, not an OOB slice panic
    // (T-03-01: a malformed `Model` must never index out of bounds).
    let num_nodes = tree.num_nodes.max(0) as usize;
    let mut nid: usize = 0;
    while !tree.is_leaf(nid) {
        let fi = tree.split_index(nid);
        if fi < 0 || (fi as usize) >= row.len() {
            return Err(GtilError::FeatureIndexOutOfBounds {
                node: nid,
                feature: fi,
                num_feature: row.len() as i32,
            });
        }
        let fvalue = row[fi as usize];
        let next: i32 = if fvalue.is_nan() {
            // Missing value → default direction (predict.cc:159).
            tree.default_child(nid)
        } else {
            next_node(
                T::from_f32(fvalue),
                tree.threshold(nid),
                tree.comparison_op(nid),
                tree.left_child(nid),
                tree.right_child(nid),
            )
        };
        // Validate the child id before re-entering the loop. `is_leaf` only
        // treats `-1` as the leaf sentinel, so any other negative id (e.g. `-2`)
        // or any id `>= num_nodes` would otherwise index `cleft[nid]` out of
        // bounds and panic. Return the declared typed error instead (CR-01).
        if next < 0 || (next as usize) >= num_nodes {
            return Err(GtilError::NodeIndexOutOfBounds { node: next as usize });
        }
        nid = next as usize;
    }
    Ok(nid)
}

/// Per-preset prediction body, generic over the leaf/threshold type `T`.
///
/// Implements the `PredictRaw` assembly order EXACTLY (`predict.cc:231-305`) —
/// this ordering IS the 1e-5 contract:
///
/// 1. zero-filled `output` of length `num_row` (`std::fill_n(.., InputT{})`);
/// 2. per row, sum leaf values **serial in tree_id order** into the f32
///    accumulator, casting each leaf to f32 first (`static_cast<InputT>` at
///    `:228`) — no tree-axis parallelism (float add is non-associative, GTIL-08);
/// 3. averaging is skipped (`average_tree_output == false` for XGBoost);
/// 4. add the f64 `base_scores[0]` into the f32 accumulator with the exact
///    `float += double` promotion semantics (`:294-304`).
///
/// The postprocessor is applied by the caller after this returns.
fn predict_preset<T: PredictScalar + PartialOrd>(
    trees: &[Tree<T>],
    base_score: f64,
    data: &[f32],
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<f32>, GtilError> {
    let mut output = vec![0.0_f32; num_row]; // (num_row, 1, 1) for binary:logistic
    for r in 0..num_row {
        let row = &data[r * num_feature..(r + 1) * num_feature];
        // Serial tree summation in tree_id order — do NOT parallelize or reorder.
        for tree in trees {
            let leaf = evaluate_tree(tree, row)?;
            // static_cast<InputT>(tree.LeafValue(leaf)) then f32 += f32.
            output[r] += tree.leaf_value(leaf).to_f32();
        }
        // base_scores is f64; mirror upstream `float_view += double_view`,
        // i.e. promote the f32 accumulator to f64, add, narrow back to f32.
        output[r] = (output[r] as f64 + base_score) as f32;
    }
    Ok(output)
}

/// Scalar single-threaded dense predict (GTIL-01 subset).
///
/// Runs the traversal + serial tree-sum + f64 base-score add + postprocessor
/// over a loaded [`Model`], returning one `f32` per row (the
/// `binary:logistic` output shape `(num_row, 1, 1)`). The `data` slice is the
/// row-major `num_row × num_feature` feature matrix.
///
/// Errors (never panics, ERR-01):
/// - [`GtilError::FeatureIndexOutOfBounds`] if a node's `split_index` exceeds
///   `num_feature` (T-03-01);
/// - [`GtilError::UnsupportedPostprocessor`] for any postprocessor name other
///   than `identity`/`sigmoid` (T-03-02).
pub fn predict(model: &Model, data: &[f32], num_row: usize) -> Result<Vec<f32>, GtilError> {
    // `model.num_feature` is loader-produced/untrusted. A negative value casts
    // to a huge `usize` and overflows the row-slice math; treat it as a (0-sized,
    // impossible) shape so the buffer-length check below rejects it instead of
    // panicking (WR-02 gtil-side guard).
    if model.num_feature < 0 {
        return Err(GtilError::InvalidInputShape {
            num_row,
            num_feature: 0,
            required: usize::MAX,
            got: data.len(),
        });
    }
    let num_feature = model.num_feature as usize;

    // Validate the input buffer up front: `predict_preset` slices
    // `data[r * num_feature..(r + 1) * num_feature]` per row, which would panic
    // on a malformed model whose `num_feature` exceeds the data actually
    // supplied (WR-01 / T-03-01). Saturate the product so an overflow can never
    // wrap into a too-small `required` (it pins to usize::MAX, rejecting the
    // input as intended).
    let required = num_row.saturating_mul(num_feature);
    if data.len() < required {
        return Err(GtilError::InvalidInputShape {
            num_row,
            num_feature,
            required,
            got: data.len(),
        });
    }

    // base_scores[0] is the single binary:logistic margin (num_target=1,
    // num_class=[1]); default to 0.0 if absent (zero-tree edge case).
    let base_score = model.base_scores.first().copied().unwrap_or(0.0);

    let mut output = match &model.variant {
        ModelVariant::F32(preset) => {
            predict_preset(&preset.trees, base_score, data, num_row, num_feature)?
        }
        ModelVariant::F64(preset) => {
            predict_preset(&preset.trees, base_score, data, num_row, num_feature)?
        }
    };

    // Apply the postprocessor selected by name (ApplyPostProcessor,
    // predict.cc:307-323). Phase 1 supports identity + sigmoid only.
    match model.postprocessor.as_str() {
        "identity" => {
            for v in output.iter_mut() {
                *v = postprocessor::identity(1.0, *v);
            }
        }
        "sigmoid" => {
            for v in output.iter_mut() {
                *v = postprocessor::sigmoid(model.sigmoid_alpha, *v);
            }
        }
        other => return Err(GtilError::UnsupportedPostprocessor(other.to_string())),
    }

    Ok(output)
}

//! GTIL reference inference engine — scalar, single-threaded predict.
//!
//! Runs reference prediction over a [`treelite_core::Model`], porting the
//! upstream GTIL traversal and assembly order VERBATIM
//! (`treelite-mainline/src/gtil/predict.cc`). Per D-08, predict is a plain
//! function — there is NO `Predictor`/backend trait in Phase 1 (deferred to
//! Phase 6).

pub mod accessor;
pub mod config;
pub mod error;
pub mod postprocessor;
pub mod shape;

pub use accessor::SparseCsr;
pub use config::{Config, PredictKind};
pub use error::GtilError;
pub use shape::{Shape, output_shape};

use treelite_core::{Model, ModelVariant, Operator, Tree, TreeNodeType};

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
    /// Widen this threshold to `f64` for the cross-domain comparison in
    /// [`next_node`]. f32→f64 and f64→f64 are both exact (lossless), so the
    /// comparison is order-preserving with respect to the original domain
    /// (`NextNode<InputT,ThresholdT>`, `predict.cc:99-124`).
    fn threshold_to_f64(self) -> f64;
    /// Cast this leaf value to `f64` (`static_cast<double>(LeafValue)`), the
    /// widening used when the output element `O` is `f64` (`predict.cc:228`).
    fn to_f64(self) -> f64;
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
    #[inline]
    fn threshold_to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
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
    #[inline]
    fn threshold_to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// The input/output/accumulator element `O` of a prediction (D-05).
///
/// Orthogonal to [`PredictScalar`] (the model's threshold/leaf domain `T`): the
/// output buffer and accumulator element equals the INPUT element type, NOT the
/// leaf type. Upstream instantiates `Predict<float>` and `Predict<double>` and
/// the output pointer is the same type as the input pointer (`predict.cc:236`
/// `Array3DView<InputT>`; `c_api/gtil.cc:50-55`). All four `(input × preset)`
/// combinations are therefore valid: f32 input always yields `Vec<f32>`, f64
/// input always yields `Vec<f64>`, independent of the model preset.
pub trait PredictOut: Copy + PartialOrd {
    /// The quiet-NaN of this element (`std::numeric_limits<InputT>::quiet_NaN()`,
    /// `predict.cc:81`) — the "absent feature" sentinel for the sparse path.
    fn nan() -> Self;
    /// The additive identity (`InputT{}`, `predict.cc:238` `std::fill_n`).
    fn zero() -> Self;
    /// Widen this input value to `f64` for the cross-domain comparison in
    /// [`next_node`] (C++ usual arithmetic conversions promote the narrower
    /// operand to the wider). f32→f64 is exact, so routing is bit-faithful.
    fn to_compare_f64(self) -> f64;
    /// `true` iff this input value is NaN (routes to the default child,
    /// `predict.cc:158-159`).
    fn is_nan_val(self) -> bool;
    /// Cast an `f32` leaf value into this output element
    /// (`static_cast<InputT>(tree.LeafValue(leaf))`, `predict.cc:228`).
    fn from_leaf_f32(v: f32) -> Self;
    /// Cast an `f64` leaf value into this output element
    /// (`static_cast<InputT>(tree.LeafValue(leaf))`, `predict.cc:228`).
    fn from_leaf_f64(v: f64) -> Self;
    /// Add a leaf value (already cast to `Self`) into an accumulator cell
    /// (`output += static_cast<InputT>(LeafValue)`, `predict.cc:228`).
    fn add_assign_leaf(&mut self, leaf: Self);
    /// Divide a cell by an integer tree count for RF averaging
    /// (`output[...] /= factor`, `predict.cc:285`). The factor is cast into the
    /// `O` domain, matching the upstream `float /= float` / `double /= double`.
    fn div_by_count(self, factor: usize) -> Self;
    /// Add the `f64` base score into this cell with the exact upstream promotion
    /// (`InputT_view += double_view` ⇒ `(self as f64 + base) as Self`,
    /// `predict.cc:294-304`).
    fn add_base_score(self, base: f64) -> Self;
    /// Apply the named postprocessor over the `(num_row, num_target,
    /// max_num_class)` `output` buffer in THIS element's own precision
    /// (`ApplyPostProcessor<InputT>`, `predict.cc:307-323`). For `O = f32` this
    /// runs the `f32` postprocessor bodies (`ApplyPostProcessor<float>`); for
    /// `O = f64` it runs the `*_f64` twins (`ApplyPostProcessor<double>`),
    /// including [`postprocessor::softmax_f64`] — softmax<double> keeps its row
    /// cells in f64 for the subtraction/exp/divide (only `max_margin`/`t`/divisor
    /// are f32), so it is NOT narrowed to f32 (CR-01 / WR-03). `sigmoid_alpha` /
    /// `ratio_c` are `f32` model fields on both paths (cast into the element type
    /// at the operation site). Unknown names are a typed
    /// [`GtilError::UnsupportedPostprocessor`].
    fn apply_named_postprocessor(
        model: &Model,
        shape: &OutputLayout<'_>,
        output: &mut [Self],
        num_row: usize,
    ) -> Result<(), GtilError>;

    /// `std::numeric_limits<InputT>::digits` — the IEEE-754 significand bit count
    /// **including** the implicit leading bit: 24 for `f32`, 53 for `f64`
    /// (`predict.cc:137`). Drives the categorical representability bound
    /// `2^MANTISSA_BITS` (the largest consecutive integer exactly representable
    /// in the input type).
    ///
    /// NOTE: deliberately NOT named `DIGITS` — the inherent `f32::DIGITS` /
    /// `f64::DIGITS` consts are the *decimal* digit count (6 / 15) and would
    /// SHADOW a trait const named `DIGITS` at `Self::DIGITS`, silently yielding
    /// the wrong bound. `MANTISSA_BITS` mirrors `f32::MANTISSA_DIGITS` (24).
    const MANTISSA_BITS: u32;

    /// The full categorical float-representability guard + membership test
    /// (`NextNodeCategorical`, `predict.cc:135-143`), evaluated in the input
    /// element's own width so the per-dtype boundary is correct. Returns whether
    /// `self` names a category present in `category_list`:
    ///
    /// ```cpp
    /// max_representable_int = min(InputT(u32::MAX), InputT(1u64 << digits));
    /// if (fvalue < 0 || fabs(fvalue) > max_representable_int) matched = false;
    /// else matched = category_list.contains(static_cast<u32>(fvalue));
    /// ```
    ///
    /// For `f32`: `max = min(4294967295, 2^24) = 2^24 = 16_777_216` — large
    /// `u32`-fitting floats past the f32 mantissa limit are rejected (the
    /// representability gap, RESEARCH Pitfall 3). For `f64`: `max =
    /// min(4294967295, 2^53) = 2^32 - 1`. The reject happens BEFORE the `as u32`
    /// truncation so an out-of-range float is never UB-cast (T-05-06).
    fn category_match(self, category_list: &[u32]) -> bool;
}

impl PredictOut for f32 {
    #[inline]
    fn nan() -> Self {
        f32::NAN
    }
    #[inline]
    fn zero() -> Self {
        0.0_f32
    }
    #[inline]
    fn to_compare_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn is_nan_val(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn from_leaf_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn from_leaf_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn add_assign_leaf(&mut self, leaf: Self) {
        *self += leaf;
    }
    #[inline]
    fn div_by_count(self, factor: usize) -> Self {
        self / factor as f32
    }
    #[inline]
    fn add_base_score(self, base: f64) -> Self {
        // float += double: promote f32→f64, add, narrow back to f32.
        (self as f64 + base) as f32
    }

    #[inline]
    fn apply_named_postprocessor(
        model: &Model,
        shape: &OutputLayout<'_>,
        output: &mut [f32],
        num_row: usize,
    ) -> Result<(), GtilError> {
        // f32 input → `ApplyPostProcessor<float>`: the byte-identical Phase-1
        // path operating directly on the f32 buffer.
        apply_postprocessor_f32(model, shape, output, num_row)
    }

    const MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS; // 24

    #[inline]
    fn category_match(self, category_list: &[u32]) -> bool {
        // max_representable_int = min(InputT(u32::MAX), InputT(1u64 << digits))
        // For f32: min(4294967295.0, 2^24 = 16_777_216.0) = 16_777_216.0.
        let max_representable_int: f32 =
            (u32::MAX as f32).min((1u64 << Self::MANTISSA_BITS) as f32);
        if self < 0.0 || self.abs() > max_representable_int {
            false
        } else {
            let category_value = self as u32; // truncates toward zero (static_cast<u32>).
            category_list.contains(&category_value)
        }
    }
}

impl PredictOut for f64 {
    #[inline]
    fn nan() -> Self {
        f64::NAN
    }
    #[inline]
    fn zero() -> Self {
        0.0_f64
    }
    #[inline]
    fn to_compare_f64(self) -> f64 {
        self
    }
    #[inline]
    fn is_nan_val(self) -> bool {
        self.is_nan()
    }
    #[inline]
    fn from_leaf_f32(v: f32) -> Self {
        v as f64
    }
    #[inline]
    fn from_leaf_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn add_assign_leaf(&mut self, leaf: Self) {
        *self += leaf;
    }
    #[inline]
    fn div_by_count(self, factor: usize) -> Self {
        self / factor as f64
    }
    #[inline]
    fn add_base_score(self, base: f64) -> Self {
        // double += double.
        self + base
    }

    #[inline]
    fn apply_named_postprocessor(
        model: &Model,
        shape: &OutputLayout<'_>,
        output: &mut [f64],
        num_row: usize,
    ) -> Result<(), GtilError> {
        // f64 input → `ApplyPostProcessor<double>`: the non-softmax/non-identity
        // postprocessors run in f64 (CR-01), softmax stays f32 (narrowed per row).
        apply_postprocessor_f64(model, shape, output, num_row)
    }

    const MANTISSA_BITS: u32 = f64::MANTISSA_DIGITS; // 53

    #[inline]
    fn category_match(self, category_list: &[u32]) -> bool {
        // max_representable_int = min(InputT(u32::MAX), InputT(1u64 << digits))
        // For f64: min(4294967295.0, 2^53) = 4294967295.0 = 2^32 - 1.
        let max_representable_int: f64 =
            (u32::MAX as f64).min((1u64 << Self::MANTISSA_BITS) as f64);
        if self < 0.0 || self.abs() > max_representable_int {
            false
        } else {
            let category_value = self as u32; // truncates toward zero (static_cast<u32>).
            category_list.contains(&category_value)
        }
    }
}

/// Cast a tree leaf value (domain `T`) into the output element `O`, matching
/// `static_cast<InputT>(tree.LeafValue(leaf))` (`predict.cc:228`). The cast goes
/// through the `T`'s natural f32/f64 view so no precision is lost beyond the
/// upstream single `static_cast`.
#[inline]
fn leaf_as_out<T: PredictScalar, O: PredictOut>(v: T) -> O {
    // Route f32-domain leaves through `from_leaf_f32` and f64-domain leaves
    // through `from_leaf_f64`, so e.g. an f64 leaf into an f32 output narrows
    // exactly once (double→float), and an f32 leaf into an f64 output widens
    // exactly once (float→double).
    O::from_leaf_f64(v.to_f64())
}

/// The `NextNode` comparison switch (`predict.cc:99-124`).
///
/// Both `fvalue` (the input value, widened from `O`) and `threshold` (widened
/// from the threshold domain `T`) are compared in `f64`. Per C++ usual
/// arithmetic conversions, `NextNode<InputT,ThresholdT>` promotes the narrower
/// operand to the wider; f32→f64 and f64→f64 are both exact (lossless) and
/// order-preserving, so comparing in `f64` yields the bit-identical routing of
/// every `(InputT, ThresholdT)` combination. XGBoost always emits `kLT`.
#[inline]
fn next_node(
    node: usize,
    fvalue: f64,
    threshold: f64,
    op: Operator,
    left: i32,
    right: i32,
) -> Result<i32, GtilError> {
    let cond = match op {
        Operator::kLT => fvalue < threshold,
        Operator::kLE => fvalue <= threshold,
        Operator::kEQ => fvalue == threshold,
        Operator::kGT => fvalue > threshold,
        Operator::kGE => fvalue >= threshold,
        // kNone (and any other unrecognized operator) is never emitted by a
        // well-formed numerical test node. Upstream `NextNode` hits a fatal
        // `TREELITE_CHECK(false)` and returns -1 (`predict.cc:120-122`); the
        // port surfaces a typed error instead of silently routing right and
        // producing a definite wrong prediction (WR-05, ERR-01).
        Operator::kNone => return Err(GtilError::UnrecognizedOperator { node, op }),
    };
    Ok(if cond { left } else { right })
}

/// The `NextNodeCategorical` membership switch (`predict.cc:128-150`).
///
/// Tests whether `fvalue` (a non-NaN feature value in the input element domain
/// `O`) names a category present in `category_list`, then routes per the
/// `category_list_right_child` polarity: a MATCH goes right when
/// `category_list_right_child` is true, else left (and a non-match goes the
/// other way).
///
/// FULL GTIL-06 guard (Plan 05-03): membership is decided by
/// [`PredictOut::category_match`], which applies the exact upstream
/// representability formula (`predict.cc:135-143`) in the input element's own
/// width — `max_representable_int = min(u32::MAX, 2^digits)`, with
/// `digits = 24` for `f32` and `53` for `f64`. This rejects `fvalue < 0 ||
/// fabs(fvalue) > max_representable_int` BEFORE the `as u32` truncation, so a
/// large-but-u32-fitting float past the f32 mantissa limit (e.g. `2^24 + 1`) is
/// a non-match (the representability gap, RESEARCH Pitfall 3 / T-05-06). The
/// per-dtype `digits` is exercised by the edge-seeded categorical fixtures.
#[inline]
fn next_node_categorical<O: PredictOut>(
    fvalue: O,
    category_list: &[u32],
    category_list_right_child: bool,
    left: i32,
    right: i32,
) -> i32 {
    let category_matched = fvalue.category_match(category_list);
    if category_list_right_child {
        if category_matched { right } else { left }
    } else if category_matched {
        left
    } else {
        right
    }
}

/// Checked category-list slice for `nid` (T-04-12 / WR-04).
///
/// The core `Tree::category_list(nid)` indexes the CSR `category_list_begin/end`
/// columns and slices the `category_list` value buffer directly. For builder- or
/// serializer-produced trees that is always in-bounds, but a hand-crafted /
/// malformed `Model` (short offset columns, or begin/end past the value buffer)
/// would panic.
///
/// A LEGITIMATELY-EMPTY in-bounds list (`begin == end` AND `end <=
/// values.len()`) returns `Ok(&[])` — every category is then a non-match, the
/// correct behavior for a node with no categories. A MALFORMED slice (`begin >
/// end`, `end > values.len()`, or a missing begin/end offset for an in-range
/// node) returns `Err(GtilError::MalformedCategoryList { node })` instead of a
/// silent `&[]` that would change the prediction (WR-04, ERR-01).
fn category_list_safe<T: Copy>(tree: &Tree<T>, nid: usize) -> Result<&[u32], GtilError> {
    let begin = tree.category_list_begin.as_slice();
    let end = tree.category_list_end.as_slice();
    let values = tree.category_list.as_slice();
    match (begin.get(nid), end.get(nid)) {
        (Some(&b), Some(&e)) => {
            let b = b as usize;
            let e = e as usize;
            if b <= e && e <= values.len() {
                Ok(&values[b..e])
            } else {
                Err(GtilError::MalformedCategoryList { node: nid })
            }
        }
        // A missing begin/end offset for an in-range node is malformed (the
        // loader sizes both columns to num_nodes). Surface a typed error.
        _ => Err(GtilError::MalformedCategoryList { node: nid }),
    }
}

/// Scalar node-0 traversal (`EvaluateTree`, `predict.cc:152-172`).
///
/// Walks from node 0 until a leaf is reached, routing NaN features to the
/// default child and otherwise dispatching through [`next_node`]. Returns the
/// leaf node id. Out-of-bounds `split_index` returns a typed
/// [`GtilError::FeatureIndexOutOfBounds`] rather than panicking (T-03-01).
fn evaluate_tree<T: PredictScalar + PartialOrd, O: PredictOut>(
    tree: &Tree<T>,
    row: &[O],
) -> Result<usize, GtilError> {
    // Upper bound on valid node ids; `num_nodes` is the loader-produced count
    // (`tree.h:158`). A child id outside `[0, num_nodes)` is a malformed
    // `cleft`/`cright` and must become a typed error, not an OOB slice panic
    // (T-03-01: a malformed `Model` must never index out of bounds).
    let num_nodes = tree.num_nodes.max(0) as usize;
    // A 0-node tree (empty `cleft`/`split_index` columns) has no node 0 to read;
    // the first `tree.is_leaf(0)` accessor below would slice out of bounds and
    // panic. Reject it up front with the declared typed error (WR-03 / ERR-01:
    // a malformed `Model` must never OOB-panic). Reuse `NodeIndexOutOfBounds`
    // with `node: 0` (node 0 does not exist).
    if num_nodes == 0 {
        return Err(GtilError::NodeIndexOutOfBounds { node: 0 });
    }
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
        let next: i32 = if fvalue.is_nan_val() {
            // Missing value → default direction (predict.cc:158-159).
            tree.default_child(nid)
        } else if tree.node_type(nid) == TreeNodeType::kCategoricalTestNode {
            // Categorical split (predict.cc:161-165): membership in the node's
            // category list, routed by the category_list_right_child polarity.
            // The full O-generic GTIL-06 representability guard lives in
            // `next_node_categorical` / `PredictOut::category_match`; the input
            // value is passed in its own `O` domain so the per-dtype boundary
            // (2^24 for f32, 2^32-1 for f64) is applied correctly.
            next_node_categorical(
                fvalue,
                category_list_safe(tree, nid)?,
                tree.category_list_right_child(nid),
                tree.left_child(nid),
                tree.right_child(nid),
            )
        } else {
            // Compare in f64 (order-preserving across all (O, T) combinations).
            // `next_node` returns a typed error on an unrecognized operator
            // (kNone on a numerical node — WR-05) instead of routing right.
            next_node(
                nid,
                fvalue.to_compare_f64(),
                tree.threshold(nid).threshold_to_f64(),
                tree.comparison_op(nid),
                tree.left_child(nid),
                tree.right_child(nid),
            )?
        };
        // Validate the child id before re-entering the loop. `is_leaf` only
        // treats `-1` as the leaf sentinel, so any other negative id (e.g. `-2`)
        // or any id `>= num_nodes` would otherwise index `cleft[nid]` out of
        // bounds and panic. Return the declared typed error instead (CR-01).
        if next < 0 || (next as usize) >= num_nodes {
            return Err(GtilError::NodeIndexOutOfBounds {
                node: next as usize,
            });
        }
        nid = next as usize;
    }
    Ok(nid)
}

/// Internal output-buffer layout extracted from the [`Model`] for indexing.
///
/// Named `OutputLayout` to disambiguate from the public per-kind [`Shape`]
/// descriptor returned by [`output_shape`] (RESEARCH Open Q3).
///
/// `num_target`, `num_class[target]`, `target_id[tree]`, and `class_id[tree]`
/// are loader-produced (untrusted). The output buffer is
/// `(num_row, num_target, max_num_class)` row-major; `cell(row, t, c)` lives at
/// `row * (num_target * max_num_class) + t * max_num_class + c`.
pub struct OutputLayout<'m> {
    num_target: i32,
    max_num_class: i32,
    num_class: &'m [i32],
    target_id: &'m [i32],
    class_id: &'m [i32],
    average_tree_output: bool,
    base_scores: &'m [f64],
}

impl OutputLayout<'_> {
    #[inline]
    fn cells_per_row(&self) -> usize {
        self.num_target as usize * self.max_num_class as usize
    }

    /// Linear index of `(row_id, target_id, class_id)` in the flat 3D buffer.
    #[inline]
    fn idx(&self, row_id: usize, target_id: i32, class_id: i32) -> usize {
        row_id * self.cells_per_row()
            + target_id as usize * self.max_num_class as usize
            + class_id as usize
    }

    /// `num_class` for a target, bounds-safe (T-04-03). Out-of-range
    /// `target_id` yields 0 so the accumulation loops are empty rather than
    /// indexing OOB.
    #[inline]
    fn num_class_of(&self, target_id: i32) -> i32 {
        if target_id < 0 || target_id as usize >= self.num_class.len() {
            0
        } else {
            self.num_class[target_id as usize]
        }
    }
}

/// Checked `HasLeafVector` test (`predict.cc:248`, `tree.h:233`; WR-04).
///
/// Upstream sizes `leaf_vector_begin/end` to `num_nodes` for every tree, so the
/// raw `Tree::has_leaf_vector(leaf)` indexes safely there.
///
/// An ABSENT offset (a scalar-leaf tree whose CSR offset columns are empty /
/// shorter than `leaf`) is the LEGITIMATE scalar-leaf path → `Ok(false)`, never
/// an error. A PRESENT-but-malformed offset (`begin > end`, or `end` past the
/// `leaf_vector` value buffer) must surface a typed
/// `GtilError::MalformedLeafVector { node }` instead of being silently treated
/// as a scalar leaf and changing the prediction (WR-04, ERR-01). A present,
/// in-bounds, non-empty range (`begin < end <= len`) is a leaf vector →
/// `Ok(true)`; a present, in-bounds, empty range (`begin == end <= len`) is a
/// scalar leaf → `Ok(false)`.
fn has_leaf_vector<T: Copy>(tree: &Tree<T>, leaf: usize) -> Result<bool, GtilError> {
    let begin = tree.leaf_vector_begin.as_slice();
    let end = tree.leaf_vector_end.as_slice();
    let values_len = tree.leaf_vector.as_slice().len();
    match (begin.get(leaf), end.get(leaf)) {
        (Some(&b), Some(&e)) => {
            let b = b as usize;
            let e = e as usize;
            if b > e || e > values_len {
                Err(GtilError::MalformedLeafVector { node: leaf })
            } else {
                // b == e ⇒ scalar leaf (false); b < e ⇒ leaf vector (true).
                Ok(b != e)
            }
        }
        // Absent offset(s): legitimate scalar-leaf path (e.g. an XGBoost f32
        // scalar tree with empty leaf-vector CSR columns).
        _ => Ok(false),
    }
}

/// A per-row feature-view producer shared by the dense and sparse predict
/// paths (the upstream `DenseMatrixAccessor` / `SparseMatrixAccessor` split,
/// `predict.cc:38-97`).
///
/// Both variants materialize row `r` into a caller-owned scratch `&mut [O]` of
/// length `num_feature`, which `predict_preset` then hands to `evaluate_tree`
/// VERBATIM. Because the traversal sees an identical contiguous row regardless
/// of source, dense==sparse parity on identical logical data is structural
/// (D-04): the only difference is how the row is filled.
enum RowSource<'a, O> {
    /// Dense row-major `num_row × num_feature` buffer; row `r` is the slice
    /// `data[r * num_feature .. (r + 1) * num_feature]` (`DenseMatrixAccessor`,
    /// `predict.cc:38-56`). The caller has already validated `data.len() >=
    /// num_row * num_feature`.
    Dense { data: &'a [O], num_feature: usize },
    /// Borrowed CSR view; row `r` is NaN-materialized via
    /// [`SparseCsr::get_row`] (absent = NaN, `predict.cc:80-85`). The caller has
    /// already `validate`d the CSR, so `get_row` is in bounds.
    Sparse(SparseCsr<'a, O>),
}

impl<O: PredictOut> RowSource<'_, O> {
    /// Fill `scratch` (length `num_feature`) with row `r`'s feature values.
    #[inline]
    fn materialize(&self, r: usize, scratch: &mut [O]) {
        match self {
            RowSource::Dense { data, num_feature } => {
                // Copy the row slice into scratch so both paths traverse the
                // same owned contiguous buffer (structural D-04 parity).
                let begin = r * num_feature;
                scratch.copy_from_slice(&data[begin..begin + num_feature]);
            }
            RowSource::Sparse(csr) => csr.get_row(r, scratch),
        }
    }
}

/// Per-preset prediction body, generic over the leaf/threshold type `T`.
///
/// Implements the `PredictRaw` assembly order EXACTLY (`predict.cc:231-305`) —
/// this ordering IS the 1e-5 contract:
///
/// 1. zero-filled `output` of length `num_row * num_target * max_num_class`
///    (`std::fill_n(.., InputT{})`);
/// 2. per row, accumulate leaf values **serial in tree_id order** into the f32
///    output cells via the four-way `OutputLeafValue`/`OutputLeafVector` branch
///    on `(target_id[tree], class_id[tree])` (`predict.cc:174-229`), casting
///    each leaf to f32 first (`static_cast<InputT>` at `:228`) — no tree-axis
///    parallelism (float add is non-associative, GTIL-08);
/// 3. if `average_tree_output`, divide each cell by its per-`(target,class)`
///    tree count (`predict.cc:259-293`);
/// 4. add the f64 `base_scores[target,class]` into each f32 cell with the exact
///    `float += double` promotion semantics (`:294-304`).
///
/// The postprocessor is applied by the caller after this returns. The scalar
/// binary `(num_row, 1, 1)` case is a degenerate path of this same code: with
/// `num_target == 1`, `max_num_class == 1`, every tree `target_id == 0`,
/// `class_id == 0`, it reduces to the Phase-1 serial sum into cell 0.
fn predict_preset<T: PredictScalar + PartialOrd, O: PredictOut>(
    trees: &[Tree<T>],
    shape: &OutputLayout<'_>,
    rows: &RowSource<'_, O>,
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<O>, GtilError> {
    let cells_per_row = shape.cells_per_row();
    let mut output = vec![O::zero(); num_row * cells_per_row];
    let num_tree = trees.len();

    // A single reusable scratch row, exactly the upstream `dense_row_` (one row
    // per thread; scalar single-thread ⇒ one row). The sparse accessor
    // NaN-materializes into it per row; the dense accessor copies the row slice
    // into it. Either way `evaluate_tree` walks the SAME contiguous `&[O]`, so
    // the dense and sparse paths are structurally identical (D-04).
    let mut scratch = vec![O::nan(); num_feature];

    for r in 0..num_row {
        rows.materialize(r, &mut scratch);
        let row: &[O] = &scratch;
        // Serial tree accumulation in tree_id order — do NOT parallelize/reorder.
        for (tree_id, tree) in trees.iter().enumerate() {
            let leaf = evaluate_tree(tree, row)?;
            let target_id = shape.target_id.get(tree_id).copied().unwrap_or(-1);
            let class_id = shape.class_id.get(tree_id).copied().unwrap_or(-1);
            if has_leaf_vector(tree, leaf)? {
                output_leaf_vector(&mut output, shape, tree, leaf, r, target_id, class_id)?;
            } else {
                output_leaf_value(&mut output, shape, tree, leaf, r, target_id, class_id)?;
            }
        }
    }

    // Tree averaging (RF): divide each cell by the number of trees routed to it
    // (predict.cc:259-293). Built once, applied to every row.
    if shape.average_tree_output {
        let mut average_factor = vec![0_usize; cells_per_row];
        for tree_id in 0..num_tree {
            let target_id = shape.target_id.get(tree_id).copied().unwrap_or(-1);
            let class_id = shape.class_id.get(tree_id).copied().unwrap_or(-1);
            if target_id < 0 && class_id < 0 {
                for t in 0..shape.num_target {
                    for c in 0..shape.num_class_of(t) {
                        average_factor[t as usize * shape.max_num_class as usize + c as usize] += 1;
                    }
                }
            } else if target_id < 0 {
                if class_id >= 0 && class_id < shape.max_num_class {
                    for t in 0..shape.num_target {
                        average_factor
                            [t as usize * shape.max_num_class as usize + class_id as usize] += 1;
                    }
                }
            } else if class_id < 0 {
                if target_id < shape.num_target {
                    for c in 0..shape.num_class_of(target_id) {
                        average_factor
                            [target_id as usize * shape.max_num_class as usize + c as usize] += 1;
                    }
                }
            } else if target_id < shape.num_target && class_id < shape.max_num_class {
                average_factor
                    [target_id as usize * shape.max_num_class as usize + class_id as usize] += 1;
            }
        }
        for r in 0..num_row {
            for t in 0..shape.num_target {
                for c in 0..shape.num_class_of(t) {
                    let factor =
                        average_factor[t as usize * shape.max_num_class as usize + c as usize];
                    if factor != 0 {
                        let cell = shape.idx(r, t, c);
                        output[cell] = output[cell].div_by_count(factor);
                    }
                }
            }
        }
    }

    // Base scores: 2D f64 add per (target, class) cell, broadcast over rows
    // (predict.cc:294-304). base_scores is f64; mirror upstream
    // `InputT_view += double_view` — for f32 output this promotes f32→f64, adds,
    // and narrows back to f32; for f64 output it is a plain f64 add (PredictOut::
    // add_base_score encodes each).
    for r in 0..num_row {
        for t in 0..shape.num_target {
            for c in 0..shape.num_class_of(t) {
                let bi = t as usize * shape.max_num_class as usize + c as usize;
                if let Some(&base) = shape.base_scores.get(bi) {
                    let cell = shape.idx(r, t, c);
                    output[cell] = output[cell].add_base_score(base);
                }
            }
        }
    }

    Ok(output)
}

/// Accumulate a scalar-leaf tree into its `(target_id, class_id)` output cell
/// (`OutputLeafValue`, `predict.cc:218-229`). Both ids must be `>= 0` for a
/// scalar leaf; bounds-checked against `num_target`/`max_num_class` so an
/// out-of-range route surfaces as a typed error, never an OOB write (T-04-03).
fn output_leaf_value<T: PredictScalar + PartialOrd, O: PredictOut>(
    output: &mut [O],
    shape: &OutputLayout<'_>,
    tree: &Tree<T>,
    leaf: usize,
    row_id: usize,
    target_id: i32,
    class_id: i32,
) -> Result<(), GtilError> {
    if target_id < 0
        || class_id < 0
        || target_id >= shape.num_target
        || class_id >= shape.max_num_class
    {
        return Err(GtilError::OutputRouteOutOfBounds {
            target_id,
            class_id,
            num_target: shape.num_target,
            max_num_class: shape.max_num_class,
        });
    }
    // static_cast<InputT>(tree.LeafValue(leaf)) then O += O.
    let v: O = leaf_as_out(tree.leaf_value(leaf));
    output[shape.idx(row_id, target_id, class_id)].add_assign_leaf(v);
    Ok(())
}

/// Accumulate a leaf-vector tree across its `(target, class)` output cells via
/// the four-way `(target_id == -1, class_id == -1)` branch
/// (`OutputLeafVector`, `predict.cc:174-216`):
///
/// - both `-1` ⇒ broadcast the `(num_target, max_num_class)` leaf vector across
///   all cells (RF leaf-vector shape);
/// - `target_id == -1` only ⇒ leaf is `(num_target, 1)`, routed into `class_id`;
/// - `class_id == -1` only ⇒ leaf is `(1, max_num_class)`, routed into `target_id`;
/// - both `>= 0` ⇒ leaf is `(1, 1)`, routed into the single `(target, class)` cell.
///
/// Leaf-vector index access is bounds-checked; an out-of-range route or a
/// short leaf vector surfaces as a typed error, never an OOB read/write.
fn output_leaf_vector<T: PredictScalar + PartialOrd, O: PredictOut>(
    output: &mut [O],
    shape: &OutputLayout<'_>,
    tree: &Tree<T>,
    leaf: usize,
    row_id: usize,
    target_id: i32,
    class_id: i32,
) -> Result<(), GtilError> {
    let leaf_out = tree.leaf_vector(leaf);
    if target_id == -1 && class_id == -1 {
        // leaf_view is (num_target, max_num_class).
        for t in 0..shape.num_target {
            for c in 0..shape.num_class_of(t) {
                let li = t as usize * shape.max_num_class as usize + c as usize;
                let v = leaf_out
                    .get(li)
                    .copied()
                    .ok_or(GtilError::LeafVectorTooShort {
                        needed: li + 1,
                        got: leaf_out.len(),
                    })?;
                output[shape.idx(row_id, t, c)].add_assign_leaf(leaf_as_out(v));
            }
        }
    } else if target_id == -1 {
        // leaf_view is (num_target, 1); route into class_id.
        if class_id < 0 || class_id >= shape.max_num_class {
            return Err(GtilError::OutputRouteOutOfBounds {
                target_id,
                class_id,
                num_target: shape.num_target,
                max_num_class: shape.max_num_class,
            });
        }
        for t in 0..shape.num_target {
            let v = leaf_out
                .get(t as usize)
                .copied()
                .ok_or(GtilError::LeafVectorTooShort {
                    needed: t as usize + 1,
                    got: leaf_out.len(),
                })?;
            output[shape.idx(row_id, t, class_id)].add_assign_leaf(leaf_as_out(v));
        }
    } else if class_id == -1 {
        // leaf_view is (1, max_num_class); route into target_id.
        if target_id < 0 || target_id >= shape.num_target {
            return Err(GtilError::OutputRouteOutOfBounds {
                target_id,
                class_id,
                num_target: shape.num_target,
                max_num_class: shape.max_num_class,
            });
        }
        for c in 0..shape.num_class_of(target_id) {
            let v = leaf_out
                .get(c as usize)
                .copied()
                .ok_or(GtilError::LeafVectorTooShort {
                    needed: c as usize + 1,
                    got: leaf_out.len(),
                })?;
            output[shape.idx(row_id, target_id, c)].add_assign_leaf(leaf_as_out(v));
        }
    } else {
        // leaf_view is (1, 1); route into the single cell.
        if target_id >= shape.num_target || class_id >= shape.max_num_class {
            return Err(GtilError::OutputRouteOutOfBounds {
                target_id,
                class_id,
                num_target: shape.num_target,
                max_num_class: shape.max_num_class,
            });
        }
        let v = leaf_out
            .first()
            .copied()
            .ok_or(GtilError::LeafVectorTooShort {
                needed: 1,
                got: leaf_out.len(),
            })?;
        output[shape.idx(row_id, target_id, class_id)].add_assign_leaf(leaf_as_out(v));
    }
    Ok(())
}

/// Single-threaded dense predict (GTIL-01 subset, widened in Plan 04-02).
///
/// Runs the traversal + serial tree accumulation + RF averaging + f64
/// per-`(target,class)` base-score add + postprocessor over a loaded [`Model`],
/// returning a flat row-major `(num_row, num_target, max_num_class)` buffer
/// (length `num_row * num_target * max_num_class`). The `data` slice is the
/// row-major `num_row × num_feature` feature matrix.
///
/// The scalar binary `(num_row, 1, 1)` case is unchanged: with `num_target == 1`
/// and `max_num_class == 1` the buffer length is exactly `num_row`, byte-for-byte
/// the Phase-1 output (GTIL-08 serial sum preserved).
///
/// Errors (never panics, ERR-01):
/// - [`GtilError::FeatureIndexOutOfBounds`] if a node's `split_index` exceeds
///   `num_feature` (T-03-01);
/// - [`GtilError::OutputRouteOutOfBounds`] if a tree's `target_id`/`class_id`
///   routes outside the output buffer (T-04-03);
/// - [`GtilError::UnsupportedPostprocessor`] for any postprocessor name not in
///   the supported set.
pub fn predict<O: PredictOut>(
    model: &Model,
    data: &[O],
    num_row: usize,
    config: &Config,
) -> Result<Vec<O>, GtilError> {
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

    // Validate the input buffer up front: the dense accessor slices
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

    let rows = RowSource::Dense { data, num_feature };
    predict_rows(model, &rows, num_row, num_feature, config)
}

/// Single-threaded SPARSE-CSR predict (GTIL-02, D-04).
///
/// The sparse analogue of [`predict`] (`PredictSparse`, `predict.cc:406-414`):
/// the `csr` view's present `(col_ind, data)` pairs name the non-absent feature
/// cells; every ABSENT cell is materialized as `O::nan()` per row
/// (`SparseMatrixAccessor::GetRow`, `predict.cc:80-85`) — NOT `0`. The
/// NaN-materialized row is then fed through the exact same
/// [`evaluate_tree`]/per-kind body the dense path uses, so on identical logical
/// data `predict_sparse(csr)` equals `predict(dense_with_nan)` structurally
/// (D-04 parity).
///
/// `num_row` is the number of rows; `csr.row_ptr` must have `num_row + 1`
/// entries. The CSR is validated once up front (`SparseCsr::validate`): a
/// malformed `col_ind` (≥ `num_feature`) or `row_ptr` (non-monotone / wrong
/// length / past the backing arrays) surfaces as a typed
/// [`GtilError::SparseColumnOutOfBounds`] / [`GtilError::SparseRowPtrInvalid`]
/// rather than an out-of-bounds access (T-05-09 / T-05-10).
pub fn predict_sparse<O: PredictOut>(
    model: &Model,
    csr: SparseCsr<'_, O>,
    num_row: usize,
    config: &Config,
) -> Result<Vec<O>, GtilError> {
    // Same negative-num_feature guard as the dense path.
    if model.num_feature < 0 {
        return Err(GtilError::InvalidInputShape {
            num_row,
            num_feature: 0,
            required: usize::MAX,
            got: csr.data.len(),
        });
    }
    let num_feature = model.num_feature as usize;

    // Validate the entire CSR structure ONCE before any row is materialized, so
    // `get_row` is in-bounds for every row (never an OOB scratch write / data
    // slice). This subsumes the dense path's input-shape check.
    csr.validate(num_row, num_feature)?;

    let rows = RowSource::Sparse(csr);
    predict_rows(model, &rows, num_row, num_feature, config)
}

/// Shared per-kind predict body over an already-validated [`RowSource`]
/// (`PredictImpl`, `predict.cc:380-396`). Both [`predict`] and
/// [`predict_sparse`] funnel here after validating their input view, so the
/// kind dispatch + traversal + assembly is written exactly once and the dense
/// and sparse paths are guaranteed identical (D-04).
fn predict_rows<O: PredictOut>(
    model: &Model,
    rows: &RowSource<'_, O>,
    num_row: usize,
    num_feature: usize,
    config: &Config,
) -> Result<Vec<O>, GtilError> {
    // Per-kind dispatch (PredictImpl, predict.cc:380-396). `LeafId` and
    // `ScorePerTree` write raw leaf data with NO postprocess/average/base-score
    // and have their own output shapes, so they branch out before the
    // sum-over-trees body.
    match config.kind {
        PredictKind::Default | PredictKind::Raw => {}
        // LeafId/ScorePerTree write RAW leaf data with NO postprocess/average/
        // base-score and have their own output shapes (predict.cc:325-378), so
        // they return directly from their own bodies.
        PredictKind::LeafId => return predict_leaf(model, rows, num_row, num_feature),
        PredictKind::ScorePerTree => {
            return predict_score_by_tree(model, rows, num_row, num_feature);
        }
    }

    // Output shape = (num_row, num_target, max_num_class). `max_num_class` is
    // max over num_class[target] (predict.cc:234-235); default to 1 for a
    // degenerate (no-target / empty) model so the binary scalar path stays
    // (num_row, 1, 1). num_target/num_class are loader-produced (untrusted);
    // clamp num_target to >= 0 and max_num_class to >= 1.
    let num_target = model.num_target.max(0);
    let max_num_class = model.num_class.iter().copied().max().unwrap_or(1).max(1);
    let shape = OutputLayout {
        num_target: if num_target == 0 { 1 } else { num_target },
        max_num_class,
        num_class: &model.num_class,
        target_id: &model.target_id,
        class_id: &model.class_id,
        average_tree_output: model.average_tree_output,
        base_scores: &model.base_scores,
    };

    let mut output = match &model.variant {
        ModelVariant::F32(preset) => {
            predict_preset(&preset.trees, &shape, rows, num_row, num_feature)?
        }
        ModelVariant::F64(preset) => {
            predict_preset(&preset.trees, &shape, rows, num_row, num_feature)?
        }
    };

    // Apply the postprocessor selected by name (ApplyPostProcessor,
    // predict.cc:307-323) ONLY for the Default kind. `Raw` returns the raw margin
    // scores with no post-processing (gtil.h:33-36). Scalar postprocessors run
    // per cell; `softmax` runs per `(row, target)` over that target's `num_class`
    // cells (predict.cc:318).
    if config.kind == PredictKind::Default {
        apply_postprocessor(model, &shape, &mut output, num_row)?;
    }

    Ok(output)
}

/// `PredictKind::LeafId` body (`PredictLeaf`, `predict.cc:325-345`).
///
/// Writes one integer leaf NODE ID per `(row, tree)` into a row-major
/// `(num_row, num_tree)` buffer: `output[row * num_tree + tree_id] =
/// leaf_id`. The leaf id is cast into the `O` output element via
/// `O::from_leaf_f64(leaf_id as f64)` — upstream stores it in the SAME
/// `Array2DView<InputT>` output buffer (`predict.cc:329,340`; A4), so for
/// `O = f32`/`f64` it is the float-typed node id. There is NO
/// postprocessor, NO RF averaging, and NO base-score add for this kind
/// (`PredictImpl` routes `kPredictLeafID` straight to `PredictLeaf`,
/// `predict.cc:389-390`).
///
/// `num_tree` is the ACTUAL per-variant tree count (`Model::GetNumTree`,
/// `tree.h:478` ⇒ `trees.size()`), not the staged header field. Reuses the same
/// NaN-materialized scratch row + [`evaluate_tree`] as every other kind, so the
/// dense and sparse leaf-id paths are identical (D-04).
fn predict_leaf<O: PredictOut>(
    model: &Model,
    rows: &RowSource<'_, O>,
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<O>, GtilError> {
    let mut scratch = vec![O::nan(); num_feature];
    match &model.variant {
        ModelVariant::F32(preset) => {
            predict_leaf_preset(&preset.trees, rows, num_row, &mut scratch)
        }
        ModelVariant::F64(preset) => {
            predict_leaf_preset(&preset.trees, rows, num_row, &mut scratch)
        }
    }
}

/// `T`-monomorphic body of [`predict_leaf`].
fn predict_leaf_preset<T: PredictScalar + PartialOrd, O: PredictOut>(
    trees: &[Tree<T>],
    rows: &RowSource<'_, O>,
    num_row: usize,
    scratch: &mut [O],
) -> Result<Vec<O>, GtilError> {
    let num_tree = trees.len();
    let mut output = vec![O::zero(); num_row * num_tree];
    for r in 0..num_row {
        rows.materialize(r, scratch);
        let row: &[O] = scratch;
        for (tree_id, tree) in trees.iter().enumerate() {
            let leaf = evaluate_tree(tree, row)?;
            // output_view(row_id, tree_id) = leaf_id (predict.cc:340). The leaf
            // node id is cast into the InputT output buffer (A4).
            output[r * num_tree + tree_id] = O::from_leaf_f64(leaf as f64);
        }
    }
    Ok(output)
}

/// `PredictKind::ScorePerTree` body (`PredictScoreByTree`,
/// `predict.cc:347-378`).
///
/// Writes the RAW per-tree leaf data into a row-major `(num_row, num_tree,
/// lvs)` buffer where `lvs = leaf_vector_shape[0] * leaf_vector_shape[1]` (≥ 1;
/// read defensively so a short/malformed shape vector yields 1 rather than
/// panicking, matching the `shape.rs` clamp). Per `(row, tree)`:
///
/// - if the reached leaf carries a leaf VECTOR
///   ([`has_leaf_vector`]), write each element `tree.leaf_vector(leaf)[i]` at
///   `(row, tree, i)` (`predict.cc:367-370`);
/// - else write the scalar `tree.leaf_value(leaf)` at `(row, tree, 0)`
///   (`predict.cc:372`; Pitfall 5 — scalar-leaf models have third-dim size 1).
///
/// There is NO postprocessor, NO RF averaging, and NO base-score add for this
/// kind (`PredictImpl` routes `kPredictPerTree` straight to
/// `PredictScoreByTree`, `predict.cc:391-392`). Leaf-vector element access is
/// bounds-checked (`LeafVectorTooShort`), never an OOB read.
fn predict_score_by_tree<O: PredictOut>(
    model: &Model,
    rows: &RowSource<'_, O>,
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<O>, GtilError> {
    // lvs = leaf_vector_shape[0] * leaf_vector_shape[1]; clamp to >= 1 so a
    // scalar-leaf model (shape [1,1]) writes at index 0 and a degenerate /
    // malformed shape never produces a zero-width third dim (Pitfall 5 / T-05-11).
    let a = model.leaf_vector_shape.first().copied().unwrap_or(1).max(0) as usize;
    let b = model.leaf_vector_shape.get(1).copied().unwrap_or(1).max(0) as usize;
    let lvs = (a * b).max(1);
    let mut scratch = vec![O::nan(); num_feature];
    match &model.variant {
        ModelVariant::F32(preset) => {
            predict_score_by_tree_preset(&preset.trees, rows, num_row, lvs, &mut scratch)
        }
        ModelVariant::F64(preset) => {
            predict_score_by_tree_preset(&preset.trees, rows, num_row, lvs, &mut scratch)
        }
    }
}

/// `T`-monomorphic body of [`predict_score_by_tree`].
fn predict_score_by_tree_preset<T: PredictScalar + PartialOrd, O: PredictOut>(
    trees: &[Tree<T>],
    rows: &RowSource<'_, O>,
    num_row: usize,
    lvs: usize,
    scratch: &mut [O],
) -> Result<Vec<O>, GtilError> {
    let num_tree = trees.len();
    // Filled with 0's (predict.cc:355 `std::fill_n(.., InputT{})`); the scalar
    // path only writes index 0 of each (row, tree) slot, leaving any padding
    // columns at 0.
    let mut output = vec![O::zero(); num_row * num_tree * lvs];
    for r in 0..num_row {
        rows.materialize(r, scratch);
        let row: &[O] = scratch;
        for (tree_id, tree) in trees.iter().enumerate() {
            let leaf = evaluate_tree(tree, row)?;
            let base = (r * num_tree + tree_id) * lvs;
            if has_leaf_vector(tree, leaf)? {
                // Write each leaf-vector element at (row, tree, i)
                // (predict.cc:367-370). Bounds-checked against lvs.
                let leafvec = tree.leaf_vector(leaf);
                for (i, &v) in leafvec.iter().enumerate() {
                    if i >= lvs {
                        return Err(GtilError::LeafVectorTooShort {
                            needed: leafvec.len(),
                            got: lvs,
                        });
                    }
                    output[base + i] = leaf_as_out(v);
                }
            } else {
                // Scalar leaf → write at index 0 (predict.cc:372; Pitfall 5).
                output[base] = leaf_as_out(tree.leaf_value(leaf));
            }
        }
    }
    Ok(output)
}

/// Apply the named postprocessor over the `(num_row, num_target, max_num_class)`
/// buffer (`ApplyPostProcessor`, `predict.cc:307-323`).
///
/// Scalar postprocessors (`identity`, `sigmoid`, `exponential`,
/// `exponential_standard_ratio`, `logarithm_one_plus_exp`) are applied cell by
/// cell. `softmax` is row-wise: for each `(row, target)` it operates over that
/// target's `num_class` contiguous cells (upstream `submdspan(..full_extent)`
/// with `model.num_class[target_id]`).
fn apply_postprocessor<O: PredictOut>(
    model: &Model,
    shape: &OutputLayout<'_>,
    output: &mut [O],
    num_row: usize,
) -> Result<(), GtilError> {
    // Dispatch on the output element type O so an f64-input model runs its
    // postprocessor in f64 (`ApplyPostProcessor<double>`), NOT through an f32
    // intermediate (CR-01). For O = f32 this is the byte-identical Phase-1 path;
    // for O = f64 the non-softmax/non-identity bodies run in f64 (softmax stays
    // f32, narrowed per row — `postprocessor.cc:59-73`). `sigmoid_alpha` /
    // `ratio_c` are f32 model fields cast at the operation site on both paths.
    O::apply_named_postprocessor(model, shape, output, num_row)
}

/// f32-buffer postprocessor application (the upstream-literal precision body).
/// Kept as a monomorphic `f32` function so the postprocessor float intermediates
/// (`postprocessor.rs`) are untouched by the O-generic widening (Pitfall 2).
fn apply_postprocessor_f32(
    model: &Model,
    shape: &OutputLayout<'_>,
    output: &mut [f32],
    num_row: usize,
) -> Result<(), GtilError> {
    match model.postprocessor.as_str() {
        "identity" => {
            for v in output.iter_mut() {
                *v = postprocessor::identity(1.0, *v);
            }
        }
        "identity_multiclass" => {
            // No-op (upstream `identity_multiclass` body is empty). The sklearn
            // RF/ET classifier averaged leaf-vectors are already normalized
            // class probabilities at load time (A4).
            for v in output.iter_mut() {
                *v = postprocessor::identity_multiclass(1.0, *v);
            }
        }
        "sigmoid" => {
            for v in output.iter_mut() {
                *v = postprocessor::sigmoid(model.sigmoid_alpha, *v);
            }
        }
        "signed_square" => {
            for v in output.iter_mut() {
                *v = postprocessor::signed_square(*v);
            }
        }
        "hinge" => {
            for v in output.iter_mut() {
                *v = postprocessor::hinge(*v);
            }
        }
        "exponential" => {
            for v in output.iter_mut() {
                *v = postprocessor::exponential(*v);
            }
        }
        "exponential_standard_ratio" => {
            for v in output.iter_mut() {
                *v = postprocessor::exponential_standard_ratio(model.ratio_c, *v);
            }
        }
        "logarithm_one_plus_exp" => {
            for v in output.iter_mut() {
                *v = postprocessor::logarithm_one_plus_exp(*v);
            }
        }
        "softmax" => {
            for r in 0..num_row {
                for t in 0..shape.num_target {
                    let n = shape.num_class_of(t);
                    if n <= 0 {
                        continue;
                    }
                    let start = shape.idx(r, t, 0);
                    let end = start + n as usize;
                    postprocessor::softmax(&mut output[start..end]);
                }
            }
        }
        "multiclass_ova" => {
            // One-vs-all: per (row, target) over that target's num_class cells,
            // an independent per-class sigmoid (NOT a normalizing softmax). Same
            // row-wise loop structure as `softmax` (predict.cc:318); sigmoid_alpha
            // is a float model field and stays f32 (Pitfall 2).
            for r in 0..num_row {
                for t in 0..shape.num_target {
                    let n = shape.num_class_of(t);
                    if n <= 0 {
                        continue;
                    }
                    let start = shape.idx(r, t, 0);
                    let end = start + n as usize;
                    postprocessor::multiclass_ova(model.sigmoid_alpha, &mut output[start..end]);
                }
            }
        }
        other => return Err(GtilError::UnsupportedPostprocessor(other.to_string())),
    }
    Ok(())
}

/// f64-buffer postprocessor application (`ApplyPostProcessor<double>`,
/// `predict.cc:307-323` with `InputT == double`).
///
/// Mirrors [`apply_postprocessor_f32`] arm-for-arm, but the non-identity bodies
/// run in f64 (the `*_f64` twins in [`postprocessor`]) so an f64-input model runs
/// its postprocessor at f64 precision (CR-01). `softmax` runs the
/// [`postprocessor::softmax_f64`] twin: upstream `softmax<double>` keeps the
/// `double* row` cells in f64 for the `row[i] - max_margin` subtraction,
/// `std::exp`, and the final `double /= float` divide — only `max_margin`/`t`/the
/// divisor are f32 (`postprocessor.cc:57-75`). The row is NOT narrowed to f32
/// first (that was CR-01 / WR-03). `sigmoid_alpha` / `ratio_c` stay f32 model
/// fields, cast into f64 at the operation site inside the twins.
fn apply_postprocessor_f64(
    model: &Model,
    shape: &OutputLayout<'_>,
    output: &mut [f64],
    num_row: usize,
) -> Result<(), GtilError> {
    match model.postprocessor.as_str() {
        "identity" => {
            // No-op (margin returned unchanged); keep f64 cells untouched.
        }
        "identity_multiclass" => {
            // No-op (upstream body is empty).
        }
        "sigmoid" => {
            for v in output.iter_mut() {
                *v = postprocessor::sigmoid_f64(model.sigmoid_alpha, *v);
            }
        }
        "signed_square" => {
            for v in output.iter_mut() {
                *v = postprocessor::signed_square_f64(*v);
            }
        }
        "hinge" => {
            // hinge is a step (1 iff > 0); the result is exactly 0.0/1.0 in any
            // width. Run it in f64 directly (no f32 intermediate).
            for v in output.iter_mut() {
                *v = if *v > 0.0 { 1.0 } else { 0.0 };
            }
        }
        "exponential" => {
            for v in output.iter_mut() {
                *v = postprocessor::exponential_f64(*v);
            }
        }
        "exponential_standard_ratio" => {
            for v in output.iter_mut() {
                *v = postprocessor::exponential_standard_ratio_f64(model.ratio_c, *v);
            }
        }
        "logarithm_one_plus_exp" => {
            for v in output.iter_mut() {
                *v = postprocessor::logarithm_one_plus_exp_f64(*v);
            }
        }
        "softmax" => {
            // softmax<double> (postprocessor.cc:57-75) keeps the `double* row`
            // cells in f64 for the `row[i] - max_margin` subtraction, `std::exp`,
            // and the final `double /= float` divide; only max_margin/t/divisor
            // are f32. Run the f64 twin in place — do NOT narrow the row to f32
            // first (that was CR-01 / WR-03).
            for r in 0..num_row {
                for t in 0..shape.num_target {
                    let n = shape.num_class_of(t);
                    if n <= 0 {
                        continue;
                    }
                    let start = shape.idx(r, t, 0);
                    let end = start + n as usize;
                    postprocessor::softmax_f64(&mut output[start..end]);
                }
            }
        }
        "multiclass_ova" => {
            for r in 0..num_row {
                for t in 0..shape.num_target {
                    let n = shape.num_class_of(t);
                    if n <= 0 {
                        continue;
                    }
                    let start = shape.idx(r, t, 0);
                    let end = start + n as usize;
                    postprocessor::multiclass_ova_f64(model.sigmoid_alpha, &mut output[start..end]);
                }
            }
        }
        other => return Err(GtilError::UnsupportedPostprocessor(other.to_string())),
    }
    Ok(())
}

#[cfg(test)]
mod categorical_guard {
    //! GREEN (Plan 05-03): the FULL categorical float-representability guard +
    //! child polarity (GTIL-06, RESEARCH Pitfall 3), plus the NaN→default-child
    //! routing confirmation (GTIL-05).
    //!
    //! 04-05 shipped a MINIMAL `next_node_categorical` guard
    //! (`fvalue < 0 || !finite || fvalue > u32::MAX`). Plan 05-03 replaces it
    //! with the FULL upstream guard (`predict.cc:135-143`), now generic over the
    //! input element `O` via [`PredictOut::category_match`]:
    //! `max_representable_int = min(u32::MAX, 2^digits)` with `digits = 24` for
    //! `f32` and `53` for `f64`, so f32's bound is `2^24` and f64's is `2^32-1`.
    //! A value whose f32 representation strictly EXCEEDS `2^24` (the f32 mantissa
    //! limit) is rejected by the full guard but is `u32`-fitting — so the minimal
    //! guard ACCEPTED it. The same source value, in f64, is well within the f64
    //! bound `2^32-1` and is decided purely by membership: this is the per-dtype
    //! `digits` distinction the edge-seeded categorical fixtures exercise.

    use super::{PredictOut, next_node_categorical};

    /// A u32-fitting categorical value whose f32 representation strictly exceeds
    /// the f32 representability limit `2^24` is REJECTED as a non-match by the
    /// FULL guard, even though it fits in `u32`. This is the f32 mantissa gap
    /// (RESEARCH Pitfall 3): the minimal Phase-4 guard accepted any `< u32::MAX`
    /// value, the full guard rejects past `2^24`. Was RED until Plan 05-03 ported
    /// `predict.cc:135-143`; now GREEN.
    #[test]
    fn categorical_full_guard_rejects_f32_value_past_mantissa_limit() {
        // 2^24 + 64 is exactly representable in f32 (a multiple of the f32 ULP at
        // this magnitude) and strictly greater than the f32 bound 2^24, so the
        // magnitude guard rejects it. It is well within u32, so the OLD minimal
        // guard would have accepted it (and matched the category) — the bug the
        // full guard fixes.
        let past_limit: f32 = (2.0_f32).powi(24) + 64.0; // == 16_777_280.0 > 2^24
        debug_assert!(past_limit > (2.0_f32).powi(24));
        let category_list: [u32; 1] = [16_777_280];
        let (left, right) = (10_i32, 20_i32);
        // category_list_right_child = false -> match routes LEFT, non-match RIGHT.
        let routed = next_node_categorical(past_limit, &category_list, false, left, right);
        // FULL GTIL-06 contract: value is REJECTED (fabs > 2^24 max_repr), i.e.
        // NON-match, so it must route RIGHT — even though 16_777_280 is in the
        // list and fits in u32.
        assert_eq!(
            routed, right,
            "FULL categorical guard must reject an f32 value past the 2^24 \
             mantissa limit as a non-match (route the not-matched direction)"
        );
    }

    /// `2.0**24` (== max_representable_int for f32) is NOT rejected by the
    /// magnitude guard — membership then decides. With the integer present in
    /// the list it matches (routes LEFT under `right_child = false`).
    #[test]
    fn categorical_f32_at_boundary_is_not_magnitude_rejected() {
        let at_max: f32 = (2.0_f32).powi(24); // exactly 16_777_216, representable
        let category_list: [u32; 1] = [16_777_216];
        let (left, right) = (10_i32, 20_i32);
        // Present in list -> MATCH -> LEFT (polarity false).
        assert_eq!(
            next_node_categorical(at_max, &category_list, false, left, right),
            left,
            "2^24 is exactly representable in f32; membership (present) must match"
        );
        // Boundary value NOT in the list -> non-match -> RIGHT (still not
        // magnitude-rejected, just absent).
        let empty: [u32; 0] = [];
        assert_eq!(
            next_node_categorical(at_max, &empty, false, left, right),
            right
        );
    }

    /// The per-dtype `digits` is exercised: the f64 boundary is `2^32 - 1`,
    /// distinct from the f32 boundary `2^24`. The SAME source value `2^24 + 64`
    /// — magnitude-rejected on the f32 path (it exceeds the f32 bound `2^24`) —
    /// is well within the f64 representable-integer range and is decided purely
    /// by membership on the f64 path; and a value just past `2^32 - 1` is
    /// magnitude-rejected for f64.
    #[test]
    fn categorical_f64_boundary_differs_from_f32() {
        let (left, right) = (10_i32, 20_i32);
        // 2^24 + 64 (the value the f32 test rejects) is exactly representable in
        // f64 and far below the f64 bound 2^32-1: the f64 path accepts it and
        // decides by membership (present -> MATCH -> LEFT).
        let v_f64: f64 = (2.0_f64).powi(24) + 64.0; // 16_777_280.0, exact in f64
        let list_present: [u32; 1] = [16_777_280];
        assert_eq!(
            next_node_categorical(v_f64, &list_present, false, left, right),
            left,
            "2^24+64 is exactly representable in f64 and below 2^32-1; \
             membership (present) must match — unlike the f32 path which rejects it"
        );
        // Direct cross-dtype contrast on the SAME numeric value: the f32 view
        // rejects (magnitude), the f64 view accepts (within bound).
        assert!(!((2.0_f32).powi(24) + 64.0).category_match(&[16_777_280]));
        assert!(((2.0_f64).powi(24) + 64.0).category_match(&[16_777_280]));

        // f64 boundary: 2^32 - 1 == 4_294_967_295 is the max; anything strictly
        // greater is magnitude-rejected (non-match -> RIGHT).
        let max_u32_f64: f64 = u32::MAX as f64; // 4_294_967_295.0
        let just_past: f64 = max_u32_f64 + 1.0; // 4_294_967_296.0 (exact in f64)
        let any_list: [u32; 1] = [0];
        assert_eq!(
            next_node_categorical(just_past, &any_list, false, left, right),
            right,
            "f64 value past 2^32-1 must be magnitude-rejected as a non-match"
        );
    }

    /// Negative and non-finite (±inf) fvalues are non-matches for BOTH dtypes
    /// (the guard's `fvalue < 0 || fabs > max` arm). (NaN never reaches this
    /// guard — `evaluate_tree` routes it to the default child first, GTIL-05;
    /// see [`negative_and_infinite_are_non_match`] and the NaN note below.)
    #[test]
    fn negative_and_infinite_are_non_match() {
        let list: [u32; 1] = [5];
        // f32
        assert!(!(-1.0_f32).category_match(&list));
        assert!(!f32::INFINITY.category_match(&list));
        assert!(!f32::NEG_INFINITY.category_match(&list));
        // f64
        assert!(!(-1.0_f64).category_match(&list));
        assert!(!f64::INFINITY.category_match(&list));
        assert!(!f64::NEG_INFINITY.category_match(&list));
    }

    /// Child polarity (`category_list_right_child`): with it TRUE, a MATCH routes
    /// RIGHT and a non-match LEFT; with it FALSE, the reverse. Verbatim from the
    /// upstream polarity block (`predict.cc:145-149`) — preserved unchanged.
    #[test]
    fn child_polarity_routes_correctly() {
        let list: [u32; 1] = [3];
        let (left, right) = (1_i32, 2_i32);
        // right_child = true: MATCH -> right, non-match -> left.
        assert_eq!(
            next_node_categorical(3.0_f32, &list, true, left, right),
            right
        );
        assert_eq!(
            next_node_categorical(9.0_f32, &list, true, left, right),
            left
        );
        // right_child = false: MATCH -> left, non-match -> right.
        assert_eq!(
            next_node_categorical(3.0_f32, &list, false, left, right),
            left
        );
        assert_eq!(
            next_node_categorical(9.0_f32, &list, false, left, right),
            right
        );
    }

    /// GTIL-05 confirmation: a NaN feature value is detected by
    /// [`PredictOut::is_nan_val`] (which `evaluate_tree` checks BEFORE the
    /// categorical-vs-numerical dispatch, routing it to `default_child`), so a
    /// missing value never enters the categorical guard. This unit-asserts the
    /// gate predicate the traversal relies on for both dtypes.
    #[test]
    fn nan_is_detected_before_categorical_guard() {
        assert!(f32::NAN.is_nan_val(), "f32 NaN must route to default_child");
        assert!(f64::NAN.is_nan_val(), "f64 NaN must route to default_child");
        // A normal category value is NOT flagged as NaN (it proceeds to the guard).
        assert!(!3.0_f32.is_nan_val());
        assert!(!3.0_f64.is_nan_val());
    }
}

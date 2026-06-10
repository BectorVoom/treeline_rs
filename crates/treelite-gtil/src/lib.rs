//! GTIL reference inference engine — scalar, single-threaded predict.
//!
//! Runs reference prediction over a [`treelite_core::Model`], porting the
//! upstream GTIL traversal and assembly order VERBATIM
//! (`treelite-mainline/src/gtil/predict.cc`). Per D-08, predict is a plain
//! function — there is NO `Predictor`/backend trait in Phase 1 (deferred to
//! Phase 6).

pub mod config;
pub mod error;
pub mod postprocessor;
pub mod shape;

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
    /// Narrow this element to `f32` for the postprocessor boundary. For `O =
    /// f32` this is the identity; for `O = f64` it narrows. (See
    /// [`apply_postprocessor`] — the f32-intermediate postprocessors are wired
    /// fully O-generic in Plan 05-03; here the f32 path is byte-identical.)
    fn out_to_f32(self) -> f32;
    /// Widen an `f32` postprocessor result back into this element. Identity for
    /// `O = f32`.
    fn out_from_f32(v: f32) -> Self;
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
    fn out_to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn out_from_f32(v: f32) -> Self {
        v
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
    fn out_to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn out_from_f32(v: f32) -> Self {
        v as f64
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
fn next_node(fvalue: f64, threshold: f64, op: Operator, left: i32, right: i32) -> i32 {
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

/// The `NextNodeCategorical` membership switch (`predict.cc:128-150`).
///
/// Tests whether `fvalue` (a non-NaN feature value) names a category present in
/// `category_list`, then routes per the `category_list_right_child` polarity: a
/// MATCH goes right when `category_list_right_child` is true, else left (and a
/// non-match goes the other way).
///
/// MINIMAL pull-forward (D-03): this ports the integer-category membership +
/// polarity that the captured `lightgbm_categorical` fixture exercises. The full
/// float-representability guard upstream (`predict.cc:132-139`: a category must
/// be exactly representable as the input type AND fit in u32) is GTIL-06,
/// deferred to Phase 5. Here we apply the load-bearing subset of that guard —
/// reject negative or non-finite values (which can never be a valid category) so
/// the `as u32` cast is well-defined — and leave the exhaustive
/// representability matrix to Phase 5.
#[inline]
fn next_node_categorical(
    fvalue: f32,
    category_list: &[u32],
    category_list_right_child: bool,
    left: i32,
    right: i32,
) -> i32 {
    // A valid category is a non-negative value that fits in u32 (the subset of
    // the upstream guard exercised by the fixture; full GTIL-06 is Phase 5).
    let category_matched = if fvalue < 0.0 || !fvalue.is_finite() || fvalue > u32::MAX as f32 {
        false
    } else {
        let category_value = fvalue as u32; // truncates toward zero (static_cast<u32>).
        category_list.contains(&category_value)
    };
    if category_list_right_child {
        if category_matched { right } else { left }
    } else if category_matched {
        left
    } else {
        right
    }
}

/// Bounds-safe category-list slice for `nid` (T-04-12).
///
/// The core `Tree::category_list(nid)` indexes the CSR `category_list_begin/end`
/// columns and slices the `category_list` value buffer directly. For builder- or
/// serializer-produced trees that is always in-bounds, but a hand-crafted /
/// malformed `Model` (short offset columns, or begin/end past the value buffer)
/// would panic. Read the offsets defensively and return an empty list on any
/// out-of-range / inverted slice rather than panicking (ERR-01) — an empty
/// category list simply makes every category a non-match.
fn category_list_safe<T: Copy>(tree: &Tree<T>, nid: usize) -> &[u32] {
    let begin = tree.category_list_begin.as_slice();
    let end = tree.category_list_end.as_slice();
    let values = tree.category_list.as_slice();
    match (begin.get(nid), end.get(nid)) {
        (Some(&b), Some(&e)) => {
            let b = b as usize;
            let e = e as usize;
            if b <= e && e <= values.len() {
                &values[b..e]
            } else {
                &[]
            }
        }
        _ => &[],
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
            // The input value is compared as an integer category; the minimal
            // float-representability guard subset lives in `next_node_categorical`
            // (full O-generic GTIL-06 guard is Plan 05-03). The category compare
            // is over the f32 view of the input value.
            next_node_categorical(
                fvalue.to_compare_f64() as f32,
                category_list_safe(tree, nid),
                tree.category_list_right_child(nid),
                tree.left_child(nid),
                tree.right_child(nid),
            )
        } else {
            // Compare in f64 (order-preserving across all (O, T) combinations).
            next_node(
                fvalue.to_compare_f64(),
                tree.threshold(nid).threshold_to_f64(),
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
struct OutputLayout<'m> {
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

/// Bounds-safe `HasLeafVector` check (`predict.cc:248`, `tree.h:233`).
///
/// Upstream sizes `leaf_vector_begin/end` to `num_nodes` for every tree, so the
/// raw `Tree::has_leaf_vector(nid)` indexes safely there. To stay panic-free on
/// a malformed model (or a hand-crafted scalar tree whose CSR offset columns are
/// empty), treat an out-of-range/absent offset as "no leaf vector" (ERR-01) and
/// fall through to the scalar `OutputLeafValue` path.
fn has_leaf_vector<T: Copy>(tree: &Tree<T>, leaf: usize) -> bool {
    let begin = tree.leaf_vector_begin.as_slice();
    let end = tree.leaf_vector_end.as_slice();
    match (begin.get(leaf), end.get(leaf)) {
        (Some(b), Some(e)) => b != e,
        _ => false,
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
    data: &[O],
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<O>, GtilError> {
    let cells_per_row = shape.cells_per_row();
    let mut output = vec![O::zero(); num_row * cells_per_row];
    let num_tree = trees.len();

    for r in 0..num_row {
        let row = &data[r * num_feature..(r + 1) * num_feature];
        // Serial tree accumulation in tree_id order — do NOT parallelize/reorder.
        for (tree_id, tree) in trees.iter().enumerate() {
            let leaf = evaluate_tree(tree, row)?;
            let target_id = shape.target_id.get(tree_id).copied().unwrap_or(-1);
            let class_id = shape.class_id.get(tree_id).copied().unwrap_or(-1);
            if has_leaf_vector(tree, leaf) {
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
    // Per-kind dispatch (PredictImpl, predict.cc:380-396). Default/Raw share the
    // sum-over-trees body and differ only in whether the postprocessor runs;
    // LeafId/ScorePerTree are wired in Plan 05-04 and return a typed error here.
    match config.kind {
        PredictKind::Default | PredictKind::Raw => {}
        PredictKind::LeafId => {
            return Err(GtilError::UnsupportedPredictKind { kind: "LeafId" });
        }
        PredictKind::ScorePerTree => {
            return Err(GtilError::UnsupportedPredictKind {
                kind: "ScorePerTree",
            });
        }
    }
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
            predict_preset(&preset.trees, &shape, data, num_row, num_feature)?
        }
        ModelVariant::F64(preset) => {
            predict_preset(&preset.trees, &shape, data, num_row, num_feature)?
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
    // The postprocessor functions run with the upstream-literal `f32`
    // intermediates (softmax `max_margin`/`t`, `sigmoid_alpha`/`ratio_c`) — they
    // must NOT be promoted to `O` (Pitfall 2). For `O = f32` this f32 view is the
    // identity (byte-identical Phase-1 path). For `O = f64` the cells are narrowed
    // to f32 here and widened back; the fully O-generic postprocessor element
    // arithmetic (e.g. f64 sigmoid) is wired in Plan 05-03 — the goldens this plan
    // validates use the precision-exact `identity` postprocessor.
    let mut buf: Vec<f32> = output.iter().map(|v| v.out_to_f32()).collect();
    apply_postprocessor_f32(model, shape, &mut buf, num_row)?;
    for (dst, src) in output.iter_mut().zip(buf) {
        *dst = O::out_from_f32(src);
    }
    Ok(())
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
        other => return Err(GtilError::UnsupportedPostprocessor(other.to_string())),
    }
    Ok(())
}

#[cfg(test)]
mod red_scaffolds {
    //! RED Wave-0 scaffold for the FULL categorical representability guard
    //! (GTIL-06, Pitfall 3).
    //!
    //! 04-05 shipped a MINIMAL `next_node_categorical` guard
    //! (`fvalue < 0 || !finite || fvalue > u32::MAX`). The FULL upstream guard
    //! (`predict.cc:127-150`) rejects `fvalue < 0 || fabs(fvalue) >
    //! max_representable_int` where, for f32 input,
    //! `max_representable_int = min(u32::MAX, 2^digits) = min(4294967295, 2^24)
    //! = 2^24`. The f32 representability-gap value `2.0**24 + 1.0`
    //! (`== 16_777_217.0`, which rounds to `16_777_216.0 = 2^24` in f32) sits in
    //! the `(2^24, u32::MAX]` gap: the minimal guard ACCEPTS it (and would route
    //! it as a category match), but the FULL guard must REJECT it (route the
    //! not-matched direction).
    //!
    //! This test asserts the FULL-guard outcome and is `#[ignore]`d with a
    //! "RED until Plan 03 full categorical guard" reason — the Wave-0 MISSING
    //! marker the Nyquist gate reads. It documents the exact edge value the
    //! frozen capture (`fixtures/gtil/`) seeds (`2**24 + 1`).

    use super::next_node_categorical;

    /// RED (Plan 03): an f32 input value of `2.0**24 + 1.0` in a categorical
    /// feature must be REJECTED as a non-match (routed the not-matched
    /// direction) by the FULL representability guard, even though the integer it
    /// rounds to (`2^24 = 16_777_216`) is present in the category list and fits
    /// in `u32`. The current minimal guard does NOT yet reject it, so this test
    /// is RED until Plan 03 ports the full `predict.cc:135-138` formula.
    #[test]
    #[ignore = "RED until Plan 03 full categorical guard (2^24+1 f32 representability gap)"]
    fn categorical_full_guard_red() {
        // The gap value: 2^24 + 1, which is NOT exactly representable in f32.
        let gap_value: f32 = (2.0_f32).powi(24) + 1.0; // rounds to 2^24 in f32
        // category_list contains the integer it rounds to (2^24 = 16_777_216),
        // so the *minimal* guard would treat it as a MATCH.
        let category_list: [u32; 1] = [16_777_216];
        let (left, right) = (10_i32, 20_i32);
        // category_list_right_child = false -> match routes LEFT, non-match RIGHT.
        let routed = next_node_categorical(gap_value, &category_list, false, left, right);
        // FULL GTIL-06 contract: gap value is REJECTED (fabs > 2^24 max_repr),
        // i.e. NON-match, so it must route RIGHT. The minimal guard currently
        // routes LEFT (it accepts the value), making this assertion RED.
        assert_eq!(
            routed, right,
            "FULL categorical guard must reject the 2^24+1 f32 gap value as a \
             non-match (route the not-matched direction)"
        );
    }
}

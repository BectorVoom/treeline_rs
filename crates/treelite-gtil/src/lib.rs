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
    let category_matched = if fvalue < 0.0
        || !fvalue.is_finite()
        || fvalue > u32::MAX as f32
    {
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
            // Missing value → default direction (predict.cc:158-159).
            tree.default_child(nid)
        } else if tree.node_type(nid) == TreeNodeType::kCategoricalTestNode {
            // Categorical split (predict.cc:161-165): membership in the node's
            // category list, routed by the category_list_right_child polarity.
            // `fvalue` is compared as an integer category (the float-
            // representability guard subset lives in `next_node_categorical`).
            next_node_categorical(
                fvalue,
                category_list_safe(tree, nid),
                tree.category_list_right_child(nid),
                tree.left_child(nid),
                tree.right_child(nid),
            )
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

/// Shape metadata extracted from the [`Model`] for output buffer indexing.
///
/// `num_target`, `num_class[target]`, `target_id[tree]`, and `class_id[tree]`
/// are loader-produced (untrusted). The output buffer is
/// `(num_row, num_target, max_num_class)` row-major; `cell(row, t, c)` lives at
/// `row * (num_target * max_num_class) + t * max_num_class + c`.
struct Shape<'m> {
    num_target: i32,
    max_num_class: i32,
    num_class: &'m [i32],
    target_id: &'m [i32],
    class_id: &'m [i32],
    average_tree_output: bool,
    base_scores: &'m [f64],
}

impl Shape<'_> {
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
fn predict_preset<T: PredictScalar + PartialOrd>(
    trees: &[Tree<T>],
    shape: &Shape<'_>,
    data: &[f32],
    num_row: usize,
    num_feature: usize,
) -> Result<Vec<f32>, GtilError> {
    let cells_per_row = shape.cells_per_row();
    let mut output = vec![0.0_f32; num_row * cells_per_row];
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
                        average_factor[target_id as usize * shape.max_num_class as usize
                            + c as usize] += 1;
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
                        output[shape.idx(r, t, c)] /= factor as f32;
                    }
                }
            }
        }
    }

    // Base scores: 2D f64 add per (target, class) cell, broadcast over rows
    // (predict.cc:294-304). base_scores is f64; mirror upstream
    // `float_view += double_view` (promote f32→f64, add, narrow back to f32).
    for r in 0..num_row {
        for t in 0..shape.num_target {
            for c in 0..shape.num_class_of(t) {
                let bi = t as usize * shape.max_num_class as usize + c as usize;
                if let Some(&base) = shape.base_scores.get(bi) {
                    let cell = shape.idx(r, t, c);
                    output[cell] = (output[cell] as f64 + base) as f32;
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
fn output_leaf_value<T: PredictScalar + PartialOrd>(
    output: &mut [f32],
    shape: &Shape<'_>,
    tree: &Tree<T>,
    leaf: usize,
    row_id: usize,
    target_id: i32,
    class_id: i32,
) -> Result<(), GtilError> {
    if target_id < 0 || class_id < 0 || target_id >= shape.num_target || class_id >= shape.max_num_class
    {
        return Err(GtilError::OutputRouteOutOfBounds {
            target_id,
            class_id,
            num_target: shape.num_target,
            max_num_class: shape.max_num_class,
        });
    }
    // static_cast<InputT>(tree.LeafValue(leaf)) then f32 += f32.
    output[shape.idx(row_id, target_id, class_id)] += tree.leaf_value(leaf).to_f32();
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
fn output_leaf_vector<T: PredictScalar + PartialOrd>(
    output: &mut [f32],
    shape: &Shape<'_>,
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
                let v = leaf_out.get(li).copied().ok_or(GtilError::LeafVectorTooShort {
                    needed: li + 1,
                    got: leaf_out.len(),
                })?;
                output[shape.idx(row_id, t, c)] += v.to_f32();
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
            output[shape.idx(row_id, t, class_id)] += v.to_f32();
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
            output[shape.idx(row_id, target_id, c)] += v.to_f32();
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
        let v = leaf_out.first().copied().ok_or(GtilError::LeafVectorTooShort {
            needed: 1,
            got: leaf_out.len(),
        })?;
        output[shape.idx(row_id, target_id, class_id)] += v.to_f32();
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

    // Output shape = (num_row, num_target, max_num_class). `max_num_class` is
    // max over num_class[target] (predict.cc:234-235); default to 1 for a
    // degenerate (no-target / empty) model so the binary scalar path stays
    // (num_row, 1, 1). num_target/num_class are loader-produced (untrusted);
    // clamp num_target to >= 0 and max_num_class to >= 1.
    let num_target = model.num_target.max(0);
    let max_num_class = model.num_class.iter().copied().max().unwrap_or(1).max(1);
    let shape = Shape {
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
    // predict.cc:307-323). Scalar postprocessors run per cell; `softmax` runs
    // per `(row, target)` over that target's `num_class` cells (predict.cc:318).
    apply_postprocessor(model, &shape, &mut output, num_row)?;

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
fn apply_postprocessor(
    model: &Model,
    shape: &Shape<'_>,
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

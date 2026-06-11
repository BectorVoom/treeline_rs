//! `sklearn` submodule loaders: thin `#[pyfunction]` wrappers over the Rust
//! `treelite-sklearn` array-signature loaders (PY-04).
//!
//! The heavy estimator→arrays extraction (the port of upstream `importer.py`)
//! stays Python-side in `treelite_rs/sklearn/__init__.py`; the estimator object
//! NEVER crosses the FFI boundary. Only numpy arrays do, and they cross
//! zero-copy: each per-tree column is borrowed as a `PyReadonlyArray1<'_, T>`,
//! `.as_slice()`d into a `&[T]`, and the per-tree `&[&[T]]` array-of-arrays is
//! assembled before dispatching to the matching `treelite_sklearn::load_*`
//! (the Phase-4 D-01 array signatures: `double const**`/`std::int64_t const**`
//! → `&[&[f64]]`/`&[&[i64]]`). A typed `PyReadonlyArray1<i64/f64>` rejects a
//! wrong-dtype array before the body runs (T-08-10); the loaders themselves
//! validate dimensions/topology → typed `SklError` → the single `TreeliteError`
//! (T-08-09, D-06). No tree internals are re-derived in Rust.

use numpy::{PyReadonlyArray1, PyUntypedArray};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

use crate::error::{PyResult2, TreelitePyErr};
use crate::model::Model;

/// Map a numpy non-contiguity failure to the single `TreeliteError`.
#[inline]
fn contiguity_err(field: &str) -> TreelitePyErr {
    use crate::error::TreeliteError;
    TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
        "sklearn array `{field}` must be C-contiguous; call np.ascontiguousarray(arr) first"
    )))
}

/// Build the per-element extraction error for an array-of-arrays field (WR-06).
/// Distinguish "the element is not a numpy array at all" (a string / `None` /
/// nested-list — report the concrete failing type) from "the element IS a numpy
/// array but of the wrong dtype" (report a dtype mismatch). The old code always
/// emitted "wrong dtype", which misled anyone debugging a malformed estimator
/// dump whose element was simply not an array.
#[inline]
fn array_element_err(item: &Bound<'_, PyAny>, field: &str) -> TreelitePyErr {
    use crate::error::TreeliteError;
    if item.cast::<PyUntypedArray>().is_ok() {
        TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "sklearn array `{field}` element has the wrong dtype; \
             no implicit cast is performed"
        )))
    } else {
        let ty = item
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "sklearn array `{field}` element must be a 1-D numpy array; got {ty}"
        )))
    }
}

/// A collection of per-tree `PyReadonlyArray1<T>` borrows. The guards keep the
/// numpy buffers alive; the `&[&[T]]` loader view is materialized on demand by
/// [`ArrayOfArrays::view`], whose slice lifetimes are compiler-tied to `&self`
/// (WR-01: no `transmute`-laundered lifetime — soundness is enforced by the
/// borrow checker, not a hand-maintained field-drop convention). Contiguity is
/// validated eagerly at extract time so a non-contiguous element fails fast with
/// the right `field` name before any loader dispatch.
struct ArrayOfArrays<'py, T: numpy::Element> {
    guards: Vec<PyReadonlyArray1<'py, T>>,
    field: &'static str,
}

impl<'py, T: numpy::Element> ArrayOfArrays<'py, T> {
    /// Extract a Python list of 1-D numpy arrays (dtype `T`) into a borrowed
    /// array-of-arrays. Each element is borrowed zero-copy; a wrong dtype raises
    /// before extraction completes (typed `PyReadonlyArray1<T>`), and a
    /// non-contiguous element is a typed `TreeliteError` (never a silent copy).
    fn extract(list: &Bound<'py, PyAny>, field: &'static str) -> PyResult2<Self> {
        use crate::error::TreeliteError;
        let seq = list.try_iter().map_err(|_| {
            TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
                "sklearn argument `{field}` must be a sequence of 1-D numpy arrays"
            )))
        })?;
        let mut guards: Vec<PyReadonlyArray1<'py, T>> = Vec::new();
        for item in seq {
            let item = item.map_err(TreelitePyErr::from_pyerr)?;
            let arr = item
                .extract::<PyReadonlyArray1<'py, T>>()
                .map_err(|_| array_element_err(&item, field))?;
            // Validate contiguity eagerly so a non-contiguous element is rejected
            // here (with `field`) rather than at `view()`, which is infallible.
            arr.as_slice().map_err(|_| contiguity_err(field))?;
            guards.push(arr);
        }
        Ok(ArrayOfArrays { guards, field })
    }

    /// Build the `&[&[T]]` loader view. Each slice borrows from a guard held by
    /// `self`, so the returned slices are compiler-tied to `&self` — no
    /// `transmute`. Contiguity was validated in `extract`, so `as_slice()` here
    /// cannot fail; the explicit `map_err` keeps the contract typed regardless.
    fn view(&self) -> PyResult2<Vec<&[T]>> {
        self.guards
            .iter()
            .map(|g| g.as_slice().map_err(|_| contiguity_err(self.field)))
            .collect()
    }
}

/// Borrow a flat 1-D numpy array (dtype `T`) zero-copy. Returns the guard; the
/// caller obtains the `&[T]` slice via `.as_slice()` at the use site so the slice
/// lifetime is compiler-tied to the guard (WR-01: no `transmute`). Contiguity is
/// validated eagerly so a non-contiguous array fails here with the right `field`
/// name; the subsequent `.as_slice()` at the use site is then infallible.
#[inline]
fn flat<'py, T: numpy::Element>(
    arr: &'py Bound<'py, PyAny>,
    field: &'static str,
) -> PyResult2<PyReadonlyArray1<'py, T>> {
    use crate::error::TreeliteError;
    let view = arr.extract::<PyReadonlyArray1<'py, T>>().map_err(|_| {
        TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
            "sklearn array `{field}` has the wrong dtype; no implicit cast is performed"
        )))
    })?;
    view.as_slice().map_err(|_| contiguity_err(field))?;
    Ok(view)
}

/// Borrow the `&[T]` from a `flat`-produced guard. Contiguity was already
/// validated in `flat`, so `as_slice` here is infallible; the slice lifetime is
/// compiler-tied to the borrowed guard (WR-01: no `transmute`).
#[inline]
fn nc_slice<'a, T: numpy::Element>(guard: &'a PyReadonlyArray1<'_, T>) -> &'a [T] {
    guard
        .as_slice()
        .expect("node_count contiguity validated in flat()")
}

// ---------------------------------------------------------------------------
// RandomForest / ExtraTrees
// ---------------------------------------------------------------------------

/// `RandomForestRegressor` / `ExtraTreesRegressor` share the same array dump and
/// the same loader; the Python shim routes both here.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_random_forest_regressor<'py>(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_random_forest_regressor(
        n_estimators,
        n_features,
        n_targets,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
    )?
    .into())
}

/// `RandomForestClassifier` / `ExtraTreesClassifier` (per-target `n_classes`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_random_forest_classifier<'py>(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    n_classes: Vec<i32>,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_random_forest_classifier(
        n_estimators,
        n_features,
        n_targets,
        &n_classes,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
    )?
    .into())
}

/// `ExtraTreesRegressor` — routes to the RF bulk path (sklearn does not
/// distinguish ExtraTrees from RandomForest in the loader).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_extra_trees_regressor<'py>(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_extra_trees_regressor(
        n_estimators,
        n_features,
        n_targets,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
    )?
    .into())
}

/// `ExtraTreesClassifier` — routes to the RF classifier bulk path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_extra_trees_classifier<'py>(
    n_estimators: i32,
    n_features: i32,
    n_targets: i32,
    n_classes: Vec<i32>,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_extra_trees_classifier(
        n_estimators,
        n_features,
        n_targets,
        &n_classes,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
    )?
    .into())
}

// ---------------------------------------------------------------------------
// GradientBoosting (leaf-shrink applied Python-side, not re-shrunk here)
// ---------------------------------------------------------------------------

/// `GradientBoostingRegressor` (MixIn path; `base_score` scalar header param).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_gradient_boosting_regressor<'py>(
    n_iter: i32,
    n_features: i32,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
    base_score: f64,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_gradient_boosting_regressor(
        n_iter,
        n_features,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
        base_score,
    )?
    .into())
}

/// `GradientBoostingClassifier` (`n_classes` + per-class `base_scores`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_gradient_boosting_classifier<'py>(
    n_iter: i32,
    n_features: i32,
    n_classes: i32,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
    base_scores: Vec<f64>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_gradient_boosting_classifier(
        n_iter,
        n_features,
        n_classes,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
        &base_scores,
    )?
    .into())
}

// ---------------------------------------------------------------------------
// IsolationForest (isolation depths in `value`, `ratio_c` scalar)
// ---------------------------------------------------------------------------

/// `IsolationForest` (SKL-03). `value` carries the precomputed isolation depths
/// (consumed AS-IS, no loader-side recomputation); `ratio_c` is the
/// `expected_depth(max_samples_)` scalar.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_isolation_forest<'py>(
    n_estimators: i32,
    n_features: i32,
    node_count: &Bound<'py, PyAny>,
    children_left: &Bound<'py, PyAny>,
    children_right: &Bound<'py, PyAny>,
    feature: &Bound<'py, PyAny>,
    threshold: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    n_node_samples: &Bound<'py, PyAny>,
    weighted_n_node_samples: &Bound<'py, PyAny>,
    impurity: &Bound<'py, PyAny>,
    ratio_c: f64,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let cl = ArrayOfArrays::<i64>::extract(children_left, "children_left")?;
    let cr = ArrayOfArrays::<i64>::extract(children_right, "children_right")?;
    let feat = ArrayOfArrays::<i64>::extract(feature, "feature")?;
    let thr = ArrayOfArrays::<f64>::extract(threshold, "threshold")?;
    let val = ArrayOfArrays::<f64>::extract(value, "value")?;
    let nns = ArrayOfArrays::<i64>::extract(n_node_samples, "n_node_samples")?;
    let wns = ArrayOfArrays::<f64>::extract(weighted_n_node_samples, "weighted_n_node_samples")?;
    let imp = ArrayOfArrays::<f64>::extract(impurity, "impurity")?;
    let (cl, cr, feat) = (cl.view()?, cr.view()?, feat.view()?);
    let (thr, val) = (thr.view()?, val.view()?);
    let (nns, wns, imp) = (nns.view()?, wns.view()?, imp.view()?);
    Ok(treelite_sklearn::load_isolation_forest(
        n_estimators,
        n_features,
        nc,
        &cl,
        &cr,
        &feat,
        &thr,
        &val,
        &nns,
        &wns,
        &imp,
        ratio_c,
    )?
    .into())
}

// ---------------------------------------------------------------------------
// HistGradientBoosting (raw packed node bytes per tree + features/categories map)
// ---------------------------------------------------------------------------

/// Owns the per-tree raw packed node byte buffers (one boxed `&[u8]` per tree)
/// plus the `&[&[u8]]` view over them. Unlike the other sklearn loaders HistGB
/// receives a RAW PACKED BYTE BUFFER per tree (the `HistGradientBoostingNode` C
/// struct), decoded field-by-field downstream at the 52/56-byte layout. The
/// `bytes` objects are copied out of Python into owned `Box<[u8]>` (an
/// acceptable one-time copy — these are small per-tree node tables, not the
/// zero-copy float matrices), and the slices borrow from those boxes.
struct NodeBuffers {
    boxes: Vec<Box<[u8]>>,
}

impl NodeBuffers {
    fn extract(list: &Bound<'_, PyAny>) -> PyResult2<Self> {
        use crate::error::TreeliteError;
        let seq = list.try_iter().map_err(|_| {
            TreelitePyErr::from_pyerr(TreeliteError::new_err(
                "sklearn argument `nodes` must be a sequence of bytes",
            ))
        })?;
        let mut boxes: Vec<Box<[u8]>> = Vec::new();
        for item in seq {
            let item = item.map_err(TreelitePyErr::from_pyerr)?;
            let bytes = item.extract::<Vec<u8>>().map_err(|_| {
                // WR-06: report the concrete failing type so a malformed dump is
                // actionable, not a flat "must be a bytes object".
                let ty = item
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "<unknown>".to_string());
                TreelitePyErr::from_pyerr(TreeliteError::new_err(format!(
                    "sklearn `nodes` element must be a bytes object; got {ty}"
                )))
            })?;
            boxes.push(bytes.into_boxed_slice());
        }
        Ok(NodeBuffers { boxes })
    }

    /// Build the `&[&[u8]]` loader view; each slice borrows from a box owned by
    /// `self`, so the slices are compiler-tied to `&self` (WR-01: no `transmute`).
    #[inline]
    fn view(&self) -> Vec<&[u8]> {
        self.boxes.iter().map(|b| &b[..]).collect()
    }
}

/// `HistGradientBoostingRegressor` (SKL-04).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_hist_gradient_boosting_regressor<'py>(
    n_iter: i32,
    n_features: i32,
    expected_sizeof_node_struct: usize,
    node_count: &Bound<'py, PyAny>,
    nodes: &Bound<'py, PyAny>,
    raw_left_cat_bitsets: &Bound<'py, PyAny>,
    features_map: Vec<i32>,
    categories_map: Option<Vec<Vec<i64>>>,
    baseline_prediction: f64,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let node_bufs = NodeBuffers::extract(nodes)?;
    let cat_bitsets = ArrayOfArrays::<u32>::extract(raw_left_cat_bitsets, "raw_left_cat_bitsets")?;
    let cat_map_ref: Option<&[Vec<i64>]> = categories_map.as_deref();
    let cat_bitsets_view = cat_bitsets.view()?;
    Ok(treelite_sklearn::load_hist_gradient_boosting_regressor(
        n_iter,
        n_features,
        expected_sizeof_node_struct,
        nc,
        &node_bufs.view(),
        &cat_bitsets_view,
        &features_map,
        cat_map_ref,
        baseline_prediction,
    )?
    .into())
}

/// `HistGradientBoostingClassifier` (SKL-04; `n_classes` + per-class baseline).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn load_hist_gradient_boosting_classifier<'py>(
    n_iter: i32,
    n_features: i32,
    n_classes: i32,
    expected_sizeof_node_struct: usize,
    node_count: &Bound<'py, PyAny>,
    nodes: &Bound<'py, PyAny>,
    raw_left_cat_bitsets: &Bound<'py, PyAny>,
    features_map: Vec<i32>,
    categories_map: Option<Vec<Vec<i64>>>,
    baseline_prediction: Vec<f64>,
) -> PyResult2<Model> {
    let nc_g = flat::<i64>(node_count, "node_count")?;
    let nc = nc_slice(&nc_g);
    let node_bufs = NodeBuffers::extract(nodes)?;
    let cat_bitsets = ArrayOfArrays::<u32>::extract(raw_left_cat_bitsets, "raw_left_cat_bitsets")?;
    let cat_map_ref: Option<&[Vec<i64>]> = categories_map.as_deref();
    let cat_bitsets_view = cat_bitsets.view()?;
    Ok(treelite_sklearn::load_hist_gradient_boosting_classifier(
        n_iter,
        n_features,
        n_classes,
        expected_sizeof_node_struct,
        nc,
        &node_bufs.view(),
        &cat_bitsets_view,
        &features_map,
        cat_map_ref,
        &baseline_prediction,
    )?
    .into())
}

/// Register all sklearn array-loader pyfunctions into the `sklearn` submodule.
pub fn register(sklearn: &Bound<'_, PyModule>) -> PyResult<()> {
    sklearn.add_function(wrap_pyfunction!(load_random_forest_regressor, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_random_forest_classifier, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_extra_trees_regressor, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_extra_trees_classifier, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_gradient_boosting_regressor, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_gradient_boosting_classifier, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(load_isolation_forest, sklearn)?)?;
    sklearn.add_function(wrap_pyfunction!(
        load_hist_gradient_boosting_regressor,
        sklearn
    )?)?;
    sklearn.add_function(wrap_pyfunction!(
        load_hist_gradient_boosting_classifier,
        sklearn
    )?)?;
    Ok(())
}

//! The `#[pyclass] Model` — a thin owning handle over `treelite_core::Model`.
//!
//! Wave 1 (Plan 08-02) exposes the read-only inspection getters the A/B suite
//! needs — `num_tree`, `num_feature`, `input_type`, `output_type` — mirroring the
//! upstream `treelite.Model` API shape (`treelite-mainline/python/treelite/model.py`).
//! Serialization / field accessors / `concatenate` land in later plans (08-03..05).
//!
//! Pitfall 2 (RESEARCH lines 397-401): `input_type`/`output_type` are derived
//! DIRECTLY from `ModelVariant` (the source of truth) — NOT from the staged
//! `threshold_type_`/`leaf_output_type_` `DType` fields, which read `kInvalid`
//! before serialization staging runs.

use pyo3::prelude::*;
use treelite_core::ModelVariant;

/// A loaded tree-ensemble model. Owns the underlying `treelite_core::Model`; the
/// loaders in `frontend`/`sklearn` and `gtil.predict_*` borrow `inner`.
///
/// Marked `unsendable`: `treelite_core::Model` contains a `TreeBuf::Borrowed`
/// raw-pointer variant (`crates/treelite-core/src/tree_buf.rs`), so the type is
/// `!Send + !Sync`. `unsendable` ties the pyclass to the thread that created it
/// (pyo3 panics on cross-thread access) — sound here because every method runs
/// under the GIL on a single thread.
#[pyclass(module = "treelite_rs._treelite_rs", name = "Model", unsendable)]
pub struct Model {
    /// The wrapped Rust core model (move-only by intent).
    pub inner: treelite_core::Model,
}

#[pymethods]
impl Model {
    /// Number of trees in the ensemble (`treelite.Model.num_tree`).
    ///
    /// Derived from the variant's preset (the source of truth), NOT
    /// `treelite_core::Model::num_tree()`, which returns the staged `num_tree_`
    /// bookkeeping field — `0` until serialization staging runs (same family as
    /// Pitfall 2 for `input_type`).
    #[getter]
    fn num_tree(&self) -> u64 {
        let n = match &self.inner.variant {
            ModelVariant::F32(p) => p.num_trees(),
            ModelVariant::F64(p) => p.num_trees(),
        };
        n as u64
    }

    /// Number of input features (`treelite.Model.num_feature`).
    #[getter]
    fn num_feature(&self) -> i32 {
        self.inner.num_feature
    }

    /// Threshold (input) numeric type as a dtype string. Derived from the model
    /// variant (Pitfall 2), NOT the staged `DType` which is `kInvalid` until
    /// serialization. `"float32"` for the `<f32,f32>` preset, `"float64"` for
    /// `<f64,f64>`.
    #[getter]
    fn input_type(&self) -> &'static str {
        match self.inner.variant {
            ModelVariant::F32(_) => "float32",
            ModelVariant::F64(_) => "float64",
        }
    }

    /// Leaf-output numeric type as a dtype string. The two concrete presets are
    /// `<float,float>` and `<double,double>`, so output type equals input type
    /// (variant-derived, Pitfall 2).
    #[getter]
    fn output_type(&self) -> &'static str {
        match self.inner.variant {
            ModelVariant::F32(_) => "float32",
            ModelVariant::F64(_) => "float64",
        }
    }
}

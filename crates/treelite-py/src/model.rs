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
//!
//! Wave 2 (Plan 08-03) widens the pyclass with the persistence + inspection
//! surface — `serialize_bytes`/`deserialize_bytes` (binary v5 round-trip),
//! `dump_as_json`, and `concatenate` — each a thin call into an already-1e-5-green
//! core seam (`treelite_core::serialize_to_buffer`/`deserialize`/`dump_as_json` and
//! `treelite_builder::concatenate`). Per RESEARCH Pattern 6 / A4, `serialize_bytes`
//! rides the BINARY serializer — the field-accessor / typed-layout surface is a
//! Phase-9 refinement and deliberately stays out of this Wave-2 vertical.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use treelite_core::ModelVariant;

use crate::error::{PyResult2, TreelitePyErr};

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

    /// Serialize the model to a fast binary byte sequence (the v5 wire format),
    /// recoverable via [`Model::deserialize_bytes`] (`treelite.Model.serialize_bytes`).
    ///
    /// Rides the BINARY serializer (`treelite_core::serialize_to_buffer`, RESEARCH
    /// Pattern 6 / A4 — not the typed field-accessor layout). Takes `&mut self` because the core
    /// serializer stages the v5 bookkeeping fields on the model in place. The
    /// `Vec<u8>` is copied out into a `PyBytes` (a small one-time copy at the
    /// boundary is acceptable per Pattern 6).
    fn serialize_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let buf = treelite_core::serialize_to_buffer(&mut self.inner);
        PyBytes::new(py, &buf)
    }

    /// Deserialize (recover) a model from a byte sequence produced by
    /// [`Model::serialize_bytes`] (`treelite.Model.deserialize_bytes`).
    ///
    /// `treelite_core::deserialize` is bounds-checked: adversarial / malformed
    /// bytes surface as a `SerializeError` → the single `TreeliteError` (T-08-07),
    /// never an out-of-bounds read or `transmute`.
    #[staticmethod]
    fn deserialize_bytes(buf: &[u8]) -> PyResult2<Model> {
        let inner = treelite_core::deserialize(buf).map_err(|e| {
            use crate::error::TreeliteError;
            TreelitePyErr::from_pyerr(TreeliteError::new_err(e.to_string()))
        })?;
        Ok(Model { inner })
    }

    /// Dump the model as a JSON string for inspection (`treelite.Model.dump_as_json`).
    ///
    /// `pretty_print` (default `True`, matching upstream) toggles indented vs
    /// compact output. The core `dump_as_json` returns a `serde_json::Value`
    /// (A3) which we render with `to_string_pretty` / `to_string`; equivalence to
    /// upstream is asserted at the PARSED-value level, never by byte-comparing the
    /// serialized text (Phase-2 D-04 discipline). Takes `&mut self` because the
    /// core dumper stages variant-derived type tags on the model in place.
    #[pyo3(signature = (*, pretty_print = true))]
    fn dump_as_json(&mut self, pretty_print: bool) -> PyResult2<String> {
        let value = treelite_core::dump_as_json(&mut self.inner);
        let s = if pretty_print {
            serde_json::to_string_pretty(&value)
        } else {
            serde_json::to_string(&value)
        }
        .map_err(|e| {
            use crate::error::TreeliteError;
            TreelitePyErr::from_pyerr(TreeliteError::new_err(e.to_string()))
        })?;
        Ok(s)
    }

    /// Concatenate multiple models into one by copying all member trees into a new
    /// destination model (`treelite.Model.concatenate`).
    ///
    /// Delegates to `treelite_builder::concatenate`. An empty input list yields
    /// `Ok(None)` from the builder, which is mapped to a typed `TreeliteError`
    /// rather than unwrapped (T-08-08 / RESEARCH Open Q1) — never a null deref.
    #[staticmethod]
    fn concatenate(models: Vec<PyRef<'_, Model>>) -> PyResult2<Model> {
        use crate::error::TreeliteError;
        let refs: Vec<&treelite_core::Model> = models.iter().map(|m| &m.inner).collect();
        match treelite_builder::concatenate(&refs)? {
            Some(inner) => Ok(Model { inner }),
            None => Err(TreelitePyErr::from_pyerr(TreeliteError::new_err(
                "concatenate requires at least one model",
            ))),
        }
    }
}

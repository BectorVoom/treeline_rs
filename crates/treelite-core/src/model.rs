//! `Model` — two-variant tree ensemble with header metadata (CORE-01, CORE-04).
//!
//! Ports `treelite-mainline/include/treelite/tree.h:437-573`. Upstream holds a
//! `std::variant<ModelPreset<float,float>, ModelPreset<double,double>>` and
//! dispatches via `std::visit`; here it is a two-variant enum dispatched via
//! `match`. The header metadata lives on `Model` itself, OUTSIDE the variant,
//! exactly as upstream.
//!
//! Critical deviation from the ROADMAP wording: `num_class`, `leaf_vector_shape`,
//! `target_id`, and `class_id` are ARRAYS (`Vec<i32>`), not scalars
//! (`tree.h:543-547`).
//!
//! XGBoost-JSON only ever produces the `F32` variant. Move-only by intent
//! (mirrors the upstream deleted copy ctor).

use crate::enums::{DType, TaskType};
use crate::tree::Tree;

/// A typed container for a vector of trees (`ModelPreset<T,T>` upstream).
pub struct ModelPreset<T: Copy> {
    /// The trees of this preset.
    pub trees: Vec<Tree<T>>,
}

impl<T: Copy> ModelPreset<T> {
    /// Construct from a vector of trees.
    pub fn new(trees: Vec<Tree<T>>) -> Self {
        ModelPreset { trees }
    }

    /// Number of trees.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
}

impl<T: Copy> Default for ModelPreset<T> {
    fn default() -> Self {
        ModelPreset { trees: Vec::new() }
    }
}

/// The two concrete preset variants (`ModelPresetVariant` upstream, tree.h:437).
pub enum ModelVariant {
    /// `<f32, f32>` preset (the variant XGBoost-JSON produces).
    F32(ModelPreset<f32>),
    /// `<f64, f64>` preset.
    F64(ModelPreset<f64>),
}

/// Central in-memory tree ensemble: a numeric-type variant plus header metadata.
///
/// Move-only by intent: header fields are array-typed exactly as upstream.
pub struct Model {
    /// The numeric-type-specialized preset.
    pub variant: ModelVariant,

    // --- header metadata (tree.h:535-553) ---
    /// Number of input features (tree.h:535).
    pub num_feature: i32,
    /// Prediction task kind (tree.h:537).
    pub task_type: TaskType,
    /// Whether tree outputs are averaged (tree.h:539; XGBoost hardcodes false).
    pub average_tree_output: bool,
    /// Number of targets (tree.h:542).
    pub num_target: i32,
    /// Per-target class counts (tree.h:543) — ARRAY, `[1]` for binary clf.
    pub num_class: Vec<i32>,
    /// Leaf-vector shape (tree.h:544) — ARRAY, `[1,1]` for binary clf.
    pub leaf_vector_shape: Vec<i32>,
    /// Per-tree target id (tree.h:546) — ARRAY, `[0]` for single-target.
    pub target_id: Vec<i32>,
    /// Per-tree class id (tree.h:547) — ARRAY, `[0]` for binary clf.
    pub class_id: Vec<i32>,
    /// Postprocessor name (tree.h:549) — e.g. `"sigmoid"`.
    pub postprocessor: String,
    /// Sigmoid scaling factor (tree.h:550); default `1.0`.
    pub sigmoid_alpha: f32,
    /// Tweedie/exponential ratio (tree.h:551); default `1.0`.
    pub ratio_c: f32,
    /// Margin-transformed base scores (tree.h:552) — f64.
    pub base_scores: Vec<f64>,
    /// Free-form attributes JSON blob (tree.h:553); may be empty.
    pub attributes: String,

    // --- private serialization bookkeeping (tree.h:556-567) ---
    // These are NOT loaded from any source; they are recomputed at serialize
    // time by `stage_serialization_fields` and then borrowed by the header
    // frame walk (Pattern 5: a recomputed scalar needs a `'a`-lived home so a
    // zero-copy borrowed frame can point at it). Module-private; the in-crate
    // serializer reads them via the `pub(crate)` accessors below.
    /// Number of trees in the variant (tree.h:558) — recomputed at serialize time.
    num_tree_: u64,
    /// Optional-field count in the model extension slot (tree.h:560) — always `0`.
    num_opt_field_per_model_: i32,
    /// Major version of the producing Treelite (tree.h:562) — staged to `4`.
    major_ver_: i32,
    /// Minor version of the producing Treelite (tree.h:563) — staged to `7`.
    minor_ver_: i32,
    /// Patch version of the producing Treelite (tree.h:564) — staged to `0`.
    patch_ver_: i32,
    /// Threshold numeric type tag (tree.h:566) — derived from the variant.
    threshold_type_: DType,
    /// Leaf-output numeric type tag (tree.h:567) — derived from the variant.
    leaf_output_type_: DType,
}

impl Model {
    /// Construct a `Model` wrapping `variant` with default header metadata
    /// (`sigmoid_alpha`/`ratio_c` default to `1.0`, all arrays empty).
    pub fn new(variant: ModelVariant) -> Self {
        Model {
            variant,
            num_feature: 0,
            task_type: TaskType::kRegressor,
            average_tree_output: false,
            num_target: 0,
            num_class: Vec::new(),
            leaf_vector_shape: Vec::new(),
            target_id: Vec::new(),
            class_id: Vec::new(),
            postprocessor: String::new(),
            sigmoid_alpha: 1.0,
            ratio_c: 1.0,
            base_scores: Vec::new(),
            attributes: String::new(),
            // Inert defaults; overwritten by `stage_serialization_fields` at
            // serialize time. Type tags start `kInvalid` exactly like upstream
            // (`tree.h:566-567`, `TypeInfo::kInvalid`).
            num_tree_: 0,
            num_opt_field_per_model_: 0,
            major_ver_: 0,
            minor_ver_: 0,
            patch_ver_: 0,
            threshold_type_: DType::kInvalid,
            leaf_output_type_: DType::kInvalid,
        }
    }

    /// Recompute and stage the private v5 header bookkeeping scalars, mirroring
    /// upstream `SerializeHeader` (`serializer.cc:93-106`; RESEARCH § "Header
    /// field walk"). Must be called at serialize time before the header frame
    /// walk borrows these fields.
    ///
    /// The version triple is the *producing Treelite version* `4.7.0` — NOT
    /// `5.x.x` (RESEARCH Pitfall 1 / Summary finding 1): "v5" names the wire
    /// generation, but the 4.7.0 wheel stamps `major_ver=4`. The type tags are
    /// derived from the active variant (`F32`→`kFloat32`, `F64`→`kFloat64`),
    /// `num_tree_` is the variant's tree count, and the model opt-field count is
    /// always `0` (the extension slot is never written).
    pub fn stage_serialization_fields(&mut self) {
        self.major_ver_ = 4;
        self.minor_ver_ = 7;
        self.patch_ver_ = 0;
        self.num_opt_field_per_model_ = 0;
        let (num_tree, dtype) = match &self.variant {
            ModelVariant::F32(p) => (p.num_trees(), DType::kFloat32),
            ModelVariant::F64(p) => (p.num_trees(), DType::kFloat64),
        };
        self.num_tree_ = num_tree as u64;
        self.threshold_type_ = dtype;
        self.leaf_output_type_ = dtype;
    }
}

// --- read accessors for the in-crate serializer (Pattern 5 borrow source) ---
// These mirror upstream's read-only privates (`major_ver`/`num_tree`/… reject
// `Set`): read-only by design, `pub(crate)` so only the serialize module sees
// them. Staged values are valid only after `stage_serialization_fields`.
//
// `allow(dead_code)`: their consumer is the in-crate serialize module added in a
// later Phase 2 plan (D-10); they are the `'a`-lived borrow source for the header
// frame walk and intentionally exist ahead of that consumer.
#[allow(dead_code)]
impl Model {
    /// Staged major version (tree.h:562).
    pub(crate) fn major_ver(&self) -> i32 {
        self.major_ver_
    }
    /// Staged minor version (tree.h:563).
    pub(crate) fn minor_ver(&self) -> i32 {
        self.minor_ver_
    }
    /// Staged patch version (tree.h:564).
    pub(crate) fn patch_ver(&self) -> i32 {
        self.patch_ver_
    }
    /// Staged tree count (tree.h:558).
    pub(crate) fn num_tree(&self) -> u64 {
        self.num_tree_
    }
    /// Staged model opt-field count (tree.h:560) — always `0`.
    pub(crate) fn num_opt_field_per_model(&self) -> i32 {
        self.num_opt_field_per_model_
    }
    /// Staged threshold type tag (tree.h:566).
    pub(crate) fn threshold_type(&self) -> DType {
        self.threshold_type_
    }
    /// Staged leaf-output type tag (tree.h:567).
    pub(crate) fn leaf_output_type(&self) -> DType {
        self.leaf_output_type_
    }

    // --- by-reference accessors for the zero-copy PyBuffer frame walk ---
    // The recomputed header scalars must live somewhere for `'a` so a borrowed
    // frame can point at them (RESEARCH Pattern 5). `stage_serialization_fields`
    // populates these private fields; the pybuffer backend borrows them here.

    /// Borrow the staged major version as a 4-byte LE-native `i32` (Pattern 5).
    pub(crate) fn major_ver_ref(&self) -> &i32 {
        &self.major_ver_
    }
    /// Borrow the staged minor version.
    pub(crate) fn minor_ver_ref(&self) -> &i32 {
        &self.minor_ver_
    }
    /// Borrow the staged patch version.
    pub(crate) fn patch_ver_ref(&self) -> &i32 {
        &self.patch_ver_
    }
    /// Borrow the staged tree count as a native `u64`.
    pub(crate) fn num_tree_ref(&self) -> &u64 {
        &self.num_tree_
    }
    /// Borrow the staged model opt-field count (always `0`).
    pub(crate) fn num_opt_field_per_model_ref(&self) -> &i32 {
        &self.num_opt_field_per_model_
    }
    /// Borrow the staged threshold type tag (1-byte repr) as a `u8` view source.
    pub(crate) fn threshold_type_ref(&self) -> &DType {
        &self.threshold_type_
    }
    /// Borrow the staged leaf-output type tag (1-byte repr).
    pub(crate) fn leaf_output_type_ref(&self) -> &DType {
        &self.leaf_output_type_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Tree;

    #[test]
    fn stage_serialization_fields_sets_version_triple_num_tree_and_type_tags() {
        // An F32 model with N empty trees.
        const N: usize = 3;
        let trees: Vec<Tree<f32>> = (0..N).map(|_| Tree::new()).collect();
        let mut model = Model::new(ModelVariant::F32(ModelPreset::new(trees)));

        // Before staging, the privates are inert (kInvalid type tags, 0 versions).
        assert_eq!(model.threshold_type(), DType::kInvalid);

        model.stage_serialization_fields();

        // Version triple is 4.7.0 (the producing Treelite version), NOT 5.x.x
        // (RESEARCH Pitfall 1).
        assert_eq!(
            (model.major_ver(), model.minor_ver(), model.patch_ver()),
            (4, 7, 0)
        );
        // num_tree_ reflects the variant's tree count.
        assert_eq!(model.num_tree(), N as u64);
        // F32 variant ⇒ both type tags are kFloat32.
        assert_eq!(model.threshold_type(), DType::kFloat32);
        assert_eq!(model.leaf_output_type(), DType::kFloat32);
        // The model extension slot is always 0.
        assert_eq!(model.num_opt_field_per_model(), 0);
    }

    #[test]
    fn stage_serialization_fields_f64_variant_uses_float64_tags() {
        let trees: Vec<Tree<f64>> = vec![Tree::new()];
        let mut model = Model::new(ModelVariant::F64(ModelPreset::new(trees)));
        model.stage_serialization_fields();
        assert_eq!(model.threshold_type(), DType::kFloat64);
        assert_eq!(model.leaf_output_type(), DType::kFloat64);
        assert_eq!(model.num_tree(), 1);
    }
}

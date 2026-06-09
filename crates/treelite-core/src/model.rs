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

use crate::enums::TaskType;
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
    // private serialization bookkeeping (tree.h:556-567) — deferred to Phase 2:
    // num_tree_, num_opt_field_per_model_, major/minor/patch_ver_,
    // threshold_type_, leaf_output_type_.
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
        }
    }
}

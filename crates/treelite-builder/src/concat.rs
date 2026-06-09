//! `ConcatenateModelObjects` (BLD-02).
//!
//! Ports `treelite-mainline/src/model_concat.cc:19-71` verbatim. Merges multiple
//! same-variant `Model` objects into one: copies the header from `objs[0]`,
//! requires every input to share the same `ModelVariant` discriminant and the
//! same `num_target` / `num_class` / `leaf_vector_shape`, deep-clones every tree,
//! and concatenates (`Extend`) `target_id` and `class_id`.
//!
//! Fidelity caveat (RESEARCH Pattern 4): upstream does NOT cross-check
//! `postprocessor` or `base_scores` equality across inputs — only the
//! copy-from-`objs[0]`. This port matches that exactly; no upstream-absent checks
//! are added.

use treelite_core::{Model, ModelPreset, ModelVariant, Tree};

use crate::error::BuilderError;

/// Deep-clone a single `Tree<T>` (column-by-column via `TreeBuf::deep_copy`),
/// mirroring upstream `Tree::Clone()` (`tree.h`).
fn clone_tree<T: Copy>(src: &Tree<T>) -> Tree<T> {
    let mut t = Tree::<T>::new();
    t.node_type = src.node_type.deep_copy();
    t.cleft = src.cleft.deep_copy();
    t.cright = src.cright.deep_copy();
    t.split_index = src.split_index.deep_copy();
    t.default_left = src.default_left.deep_copy();
    t.leaf_value = src.leaf_value.deep_copy();
    t.threshold = src.threshold.deep_copy();
    t.cmp = src.cmp.deep_copy();
    t.category_list_right_child = src.category_list_right_child.deep_copy();
    t.leaf_vector = src.leaf_vector.deep_copy();
    t.leaf_vector_begin = src.leaf_vector_begin.deep_copy();
    t.leaf_vector_end = src.leaf_vector_end.deep_copy();
    t.category_list = src.category_list.deep_copy();
    t.category_list_begin = src.category_list_begin.deep_copy();
    t.category_list_end = src.category_list_end.deep_copy();
    t.data_count = src.data_count.deep_copy();
    t.sum_hess = src.sum_hess.deep_copy();
    t.gain = src.gain.deep_copy();
    t.data_count_present = src.data_count_present.deep_copy();
    t.sum_hess_present = src.sum_hess_present.deep_copy();
    t.gain_present = src.gain_present.deep_copy();
    t.has_categorical_split = src.has_categorical_split;
    t.num_nodes = src.num_nodes;
    t.num_opt_field_per_tree = src.num_opt_field_per_tree;
    t.num_opt_field_per_node = src.num_opt_field_per_node;
    t
}

/// The `ModelVariant` discriminant, used for the same-variant check
/// (`model_concat.cc:44,47`).
fn variant_tag(m: &Model) -> u8 {
    match &m.variant {
        ModelVariant::F32(_) => 0,
        ModelVariant::F64(_) => 1,
    }
}

/// Concatenate multiple same-variant models into one (`model_concat.cc:19-71`).
///
/// Returns `Ok(None)` for an empty input (upstream returns a null `unique_ptr`).
/// Rejects a variant mismatch (`VariantMismatch`) or a `num_target` /
/// `num_class` / `leaf_vector_shape` mismatch (`HeaderMismatch`).
pub fn concatenate(objs: &[&Model]) -> Result<Option<Model>, BuilderError> {
    if objs.is_empty() {
        return Ok(None);
    }
    let first = objs[0];

    // Header copy from objs[0] (`model_concat.cc:26-39`).
    let mut out = Model::new(match &first.variant {
        ModelVariant::F32(_) => ModelVariant::F32(ModelPreset::new(Vec::new())),
        ModelVariant::F64(_) => ModelVariant::F64(ModelPreset::new(Vec::new())),
    });
    out.num_feature = first.num_feature;
    out.task_type = first.task_type;
    out.average_tree_output = first.average_tree_output;
    out.num_target = first.num_target;
    out.num_class = first.num_class.clone();
    out.leaf_vector_shape = first.leaf_vector_shape.clone();
    out.postprocessor = first.postprocessor.clone();
    out.sigmoid_alpha = first.sigmoid_alpha;
    out.ratio_c = first.ratio_c;
    out.base_scores = first.base_scores.clone();
    out.attributes = first.attributes.clone();

    let first_tag = variant_tag(first);

    // Per-input checks + deep-clone trees + Extend ids (`model_concat.cc:46-65`).
    let mut target_id: Vec<i32> = Vec::new();
    let mut class_id: Vec<i32> = Vec::new();

    // Collect cloned trees into a typed vec matching the variant.
    enum Acc {
        F32(Vec<Tree<f32>>),
        F64(Vec<Tree<f64>>),
    }
    let mut acc = match &first.variant {
        ModelVariant::F32(_) => Acc::F32(Vec::new()),
        ModelVariant::F64(_) => Acc::F64(Vec::new()),
    };

    for (i, m) in objs.iter().enumerate() {
        if variant_tag(m) != first_tag {
            return Err(BuilderError::VariantMismatch { index: i });
        }
        if m.num_target != out.num_target {
            return Err(BuilderError::HeaderMismatch {
                index: i,
                field: "num_target",
            });
        }
        if m.num_class != out.num_class {
            return Err(BuilderError::HeaderMismatch {
                index: i,
                field: "num_class",
            });
        }
        if m.leaf_vector_shape != out.leaf_vector_shape {
            return Err(BuilderError::HeaderMismatch {
                index: i,
                field: "leaf_vector_shape",
            });
        }

        match (&mut acc, &m.variant) {
            (Acc::F32(dst), ModelVariant::F32(p)) => {
                for t in &p.trees {
                    dst.push(clone_tree(t));
                }
            }
            (Acc::F64(dst), ModelVariant::F64(p)) => {
                for t in &p.trees {
                    dst.push(clone_tree(t));
                }
            }
            // Discriminant equality was already asserted above; unreachable.
            _ => return Err(BuilderError::VariantMismatch { index: i }),
        }

        target_id.extend_from_slice(&m.target_id);
        class_id.extend_from_slice(&m.class_id);
    }

    out.variant = match acc {
        Acc::F32(trees) => ModelVariant::F32(ModelPreset::new(trees)),
        Acc::F64(trees) => ModelVariant::F64(ModelPreset::new(trees)),
    };
    out.target_id = target_id;
    out.class_id = class_id;

    // Post-assert (`model_concat.cc:68-69`): id lengths == total tree count.
    let num_tree = match &out.variant {
        ModelVariant::F32(p) => p.num_trees(),
        ModelVariant::F64(p) => p.num_trees(),
    };
    debug_assert_eq!(out.target_id.len(), num_tree);
    debug_assert_eq!(out.class_id.len(), num_tree);

    Ok(Some(out))
}

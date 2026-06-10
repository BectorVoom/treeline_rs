//! HistGradientBoosting packed-node loader (SKL-04) — the Phase-4 tentpole.
//!
//! Ports `treelite-mainline/src/model_loader/sklearn.cc`
//! `LoadHistGradientBoosting{,Impl}` (`:260-369`) and the HistGB MixIns
//! (`:135-198`). Unlike the RF/GB sklearn paths (which receive parsed parallel
//! arrays), HistGB hands over a RAW PACKED BYTE BUFFER per tree — the
//! `HistGradientBoostingNode<FeatureIdT>` C struct dumped verbatim from
//! `sklearn._predictors[*].nodes`.
//!
//! ## Packed node decode (Phase-3 byte-cursor discipline, D-08)
//!
//! The struct is `#pragma pack(1)` (sklearn.cc:260) so every field sits at a
//! fixed byte offset with NO alignment padding. We decode each field
//! field-by-field via [`u32::from_le_bytes`] / [`f64::from_le_bytes`] at its
//! explicit offset — NEVER by reinterpreting the buffer as a `#[repr(C)]` struct
//! (`transmute`/`bytemuck`): that would be alignment/endianness UB and is the
//! Phase-3 D-08 ban (enforced by a grep gate in the plan's acceptance criteria).
//!
//! Two `itemsize`s exist depending on the `FeatureIdT` width:
//! - **52** — `feature_idx` is `i32` (32-bit feature index variant)
//! - **56** — `feature_idx` is `i64` (64-bit feature index variant)
//!
//! Field offsets (sklearn.cc:262-275; the 56-byte layout shifts every field
//! after `feature_idx` by +4 relative to the 52-byte layout):
//!
//! | field              | type | off (52) | off (56) |
//! |--------------------|------|----------|----------|
//! | `value`            | f64  | 0        | 0        |
//! | `count`            | u32  | 8        | 8        |
//! | `feature_idx`      | i32/i64 | 12    | 12       |
//! | `num_threshold`    | f64  | 16       | 20       |
//! | `missing_go_to_left`| u8  | 24       | 28       |
//! | `left`             | u32  | 25       | 29       |
//! | `right`            | u32  | 29       | 33       |
//! | `gain`             | f64  | 33       | 37       |
//! | `depth`            | u32  | 41       | 45       |
//! | `is_leaf`          | u8   | 45       | 49       |
//! | `bin_threshold`    | u8   | 46       | 50       |
//! | `is_categorical`   | u8   | 47       | 51       |
//! | `bitset_idx`       | u32  | 48       | 52       |
//!
//! ## features_map / categories_map (Pitfall 4)
//!
//! `split_index = features_map[node.feature_idx]` is ALWAYS applied
//! (sklearn.cc:325) — the packed node's `feature_idx` is in the model's internal
//! (categorical-first) ordering; `features_map` permutes it back to the input
//! feature ordering. `categories_map[fid][cat]` remaps categorical bit values
//! when a `categories_map` is present (sklearn.cc:300-305); identity otherwise.
//! Getting either wrong shifts the golden by a feature-permuted offset.
//!
//! ## Decode rules (sklearn.cc:314-346)
//! - leaf iff `left <= 0` (NOT `== -1` — HistGB uses `0` for the missing child)
//! - `default_left = (missing_go_to_left == 1)`
//! - numerical: `numerical_test(split_index, num_threshold, default_left, kLE,
//!   left, right)` — `num_threshold` read DIRECTLY (Pitfall 5: do NOT reconstruct
//!   from `_bin_mapper`; `known_cat_bitsets` is passed but UNUSED in v4.7.0, A3)
//! - categorical: see [`decode_categorical`] (Task 2)

use treelite_builder::{BuilderMetadata, ModelBuilder};
use treelite_core::{Model, Operator, TaskType};

use crate::error::SklError;

// ---------------------------------------------------------------------------
// Field offsets for the two packed-node layouts.
// ---------------------------------------------------------------------------

/// Offsets of every `HistGradientBoostingNode` field for a given `itemsize`.
///
/// `feature_idx` width is the only difference: 4 bytes (`i32`) for the 52-byte
/// variant, 8 bytes (`i64`) for the 56-byte variant. Every field AFTER
/// `feature_idx` shifts by `feat_width - 4`.
#[derive(Debug, Clone, Copy)]
struct NodeLayout {
    itemsize: usize,
    feat_is_i64: bool,
    off_value: usize,
    off_count: usize,
    off_feature_idx: usize,
    off_num_threshold: usize,
    off_missing: usize,
    off_left: usize,
    off_right: usize,
    off_gain: usize,
    off_is_categorical: usize,
    off_bitset_idx: usize,
    // `depth`, `is_leaf`, and `bin_threshold` are present in the packed struct
    // (offsets gain+25/+29/+30) but unused by this loader: leaf detection is
    // `left <= 0` (NOT the `is_leaf` byte, sklearn.cc:320), and depth/bin are
    // metadata Treelite does not consume. They are deliberately not decoded.
}

impl NodeLayout {
    /// Build the field-offset table for `itemsize`, rejecting any value not in
    /// {52, 56} with a typed [`SklError::HistGbDecode`] (T-04-18 — upstream
    /// `TREELITE_LOG(FATAL)`, sklearn.cc:366).
    fn for_itemsize(itemsize: usize) -> Result<Self, SklError> {
        let feat_is_i64 = match itemsize {
            52 => false,
            56 => true,
            _ => {
                return Err(SklError::HistGbDecode {
                    offset: 0,
                    detail: format!(
                        "unexpected sizeof node struct: {itemsize} (must be 52 or 56)"
                    ),
                });
            }
        };
        // value@0, count@8, feature_idx@12. The fields after feature_idx start
        // at 12 + feat_width and are laid out contiguously (pack(1), no padding).
        let feat_width = if feat_is_i64 { 8 } else { 4 };
        let after = 12 + feat_width; // num_threshold offset
        Ok(NodeLayout {
            itemsize,
            feat_is_i64,
            off_value: 0,
            off_count: 8,
            off_feature_idx: 12,
            off_num_threshold: after,        // f64 (8)
            off_missing: after + 8,          // u8  (1)
            off_left: after + 9,             // u32 (4)
            off_right: after + 13,           // u32 (4)
            off_gain: after + 17,            // f64 (8)
            // depth (u32) @ after+25, is_leaf (u8) @ after+29,
            // bin_threshold (u8) @ after+30 — not decoded (see struct comment).
            off_is_categorical: after + 31,  // u8  (1)
            off_bitset_idx: after + 32,      // u32 (4)
        })
    }
}

/// One decoded `HistGradientBoostingNode` (the fields this loader consumes).
#[derive(Debug, Clone, Copy)]
struct HistGbNode {
    value: f64,
    count: u32,
    feature_idx: i64,
    num_threshold: f64,
    missing_go_to_left: u8,
    left: u32,
    right: u32,
    gain: f64,
    is_categorical: u8,
    bitset_idx: u32,
}

// ---------------------------------------------------------------------------
// Bounds-checked little-endian field readers (D-08, never OOB).
// ---------------------------------------------------------------------------

/// Read a `u8` at `rec[off]`, mapping a short slice to a typed error.
fn read_u8(rec: &[u8], off: usize) -> Result<u8, SklError> {
    rec.get(off).copied().ok_or_else(|| SklError::HistGbDecode {
        offset: off,
        detail: "node buffer too short for u8 field".to_string(),
    })
}

/// Read a little-endian `u32` at `rec[off..off+4]`.
fn read_u32(rec: &[u8], off: usize) -> Result<u32, SklError> {
    let b = rec.get(off..off + 4).ok_or_else(|| SklError::HistGbDecode {
        offset: off,
        detail: "node buffer too short for u32 field".to_string(),
    })?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Read a little-endian `i32` at `rec[off..off+4]`.
fn read_i32(rec: &[u8], off: usize) -> Result<i32, SklError> {
    let b = rec.get(off..off + 4).ok_or_else(|| SklError::HistGbDecode {
        offset: off,
        detail: "node buffer too short for i32 field".to_string(),
    })?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Read a little-endian `i64` at `rec[off..off+8]`.
fn read_i64(rec: &[u8], off: usize) -> Result<i64, SklError> {
    let b = rec.get(off..off + 8).ok_or_else(|| SklError::HistGbDecode {
        offset: off,
        detail: "node buffer too short for i64 field".to_string(),
    })?;
    Ok(i64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// Read a little-endian `f64` at `rec[off..off+8]`.
fn read_f64(rec: &[u8], off: usize) -> Result<f64, SklError> {
    let b = rec.get(off..off + 8).ok_or_else(|| SklError::HistGbDecode {
        offset: off,
        detail: "node buffer too short for f64 field".to_string(),
    })?;
    Ok(f64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// Decode the single packed node at `rec` (exactly `layout.itemsize` bytes)
/// field-by-field via `from_le_bytes` (NEVER a transmute, D-08).
fn decode_node(rec: &[u8], layout: &NodeLayout) -> Result<HistGbNode, SklError> {
    let feature_idx = if layout.feat_is_i64 {
        read_i64(rec, layout.off_feature_idx)?
    } else {
        read_i32(rec, layout.off_feature_idx)? as i64
    };
    Ok(HistGbNode {
        value: read_f64(rec, layout.off_value)?,
        count: read_u32(rec, layout.off_count)?,
        feature_idx,
        num_threshold: read_f64(rec, layout.off_num_threshold)?,
        missing_go_to_left: read_u8(rec, layout.off_missing)?,
        left: read_u32(rec, layout.off_left)?,
        right: read_u32(rec, layout.off_right)?,
        gain: read_f64(rec, layout.off_gain)?,
        is_categorical: read_u8(rec, layout.off_is_categorical)?,
        bitset_idx: read_u32(rec, layout.off_bitset_idx)?,
    })
}

// ---------------------------------------------------------------------------
// Categorical bitset decode (Task 2).
// ---------------------------------------------------------------------------

/// `check(bitmap, val, row) = (bitmap[8*row + val/32] >> (val%32)) & 1`
/// (sklearn.cc:296-298). The `8*row` stride is load-bearing: each categorical
/// split occupies one 256-bit ROW = 8 consecutive `u32` words. This is NOT the
/// LightGBM `BitsetToList` layout — do NOT share code (RESEARCH No-Analog).
///
/// Returns `false` (bit unset) when the computed word index is out of range, so
/// a malformed bitmap can never OOB-panic (T-04-20). The 256-category caller
/// bounds-checks the full row up front before relying on this.
fn check_bit(bitmap: &[u32], val: u32, row: u32) -> bool {
    let word = 8 * (row as usize) + (val as usize) / 32;
    match bitmap.get(word) {
        Some(w) => ((w >> (val % 32)) & 1) == 1,
        None => false,
    }
}

/// Decode a categorical node into its list of LEFT category values
/// (sklearn.cc:328-336): walk `i in 0..256`, push `cat_transform(feature_idx, i)`
/// for every set bit in `bitmap` row `bitset_idx`. `cat_transform` is
/// `categories_map[fid][cat]` when `categories_map` is present, identity
/// otherwise.
///
/// `feature_idx` and `bitset_idx` are bounds-checked BEFORE the scan (T-04-19,
/// T-04-20): the 256-bit row spans words `8*bitset_idx ..= 8*bitset_idx + 7`, so
/// that whole range must be in `bitmap`.
fn decode_categorical(
    feature_idx: i64,
    bitset_idx: u32,
    bitmap: &[u32],
    categories_map: Option<&[Vec<i64>]>,
) -> Result<Vec<u32>, SklError> {
    // The 256-bit categorical row occupies 8 consecutive u32 words; verify the
    // whole row is present before any bit access (T-04-20).
    let row_start = 8usize
        .checked_mul(bitset_idx as usize)
        .ok_or_else(|| SklError::HistGbDecode {
            offset: 0,
            detail: format!("bitset_idx {bitset_idx} overflows row offset"),
        })?;
    let row_end = row_start + 8;
    if row_end > bitmap.len() {
        return Err(SklError::HistGbDecode {
            offset: row_start,
            detail: format!(
                "categorical bitset row [{row_start}, {row_end}) out of range (bitmap len {})",
                bitmap.len()
            ),
        });
    }

    // When a categories_map is present, resolve and bounds-check the per-feature
    // remap row once (categories_map[fid]).
    let remap_row: Option<&[i64]> = match categories_map {
        Some(cm) => {
            let fid = usize::try_from(feature_idx).map_err(|_| SklError::HistGbDecode {
                offset: 0,
                detail: format!("negative feature_idx {feature_idx} indexing categories_map"),
            })?;
            let row = cm.get(fid).ok_or_else(|| SklError::HistGbDecode {
                offset: 0,
                detail: format!(
                    "feature_idx {fid} out of range for categories_map (len {})",
                    cm.len()
                ),
            })?;
            Some(row.as_slice())
        }
        None => None,
    };

    let mut left_categories: Vec<u32> = Vec::new();
    for i in 0u32..256 {
        if check_bit(bitmap, i, bitset_idx) {
            let transformed: i64 = match remap_row {
                Some(row) => {
                    let cat = i as usize;
                    *row.get(cat).ok_or_else(|| SklError::HistGbDecode {
                        offset: 0,
                        detail: format!(
                            "category {cat} out of range for categories_map row (len {})",
                            row.len()
                        ),
                    })?
                }
                None => i as i64,
            };
            // Categories are non-negative small integers; the builder/CSR column
            // is u32. Reject a negative/oversized transformed value rather than
            // truncating it.
            let c = u32::try_from(transformed).map_err(|_| SklError::HistGbDecode {
                offset: 0,
                detail: format!("transformed category {transformed} out of u32 range"),
            })?;
            left_categories.push(c);
        }
    }
    Ok(left_categories)
}

// ---------------------------------------------------------------------------
// Tree decode loop (sklearn.cc:307-348).
// ---------------------------------------------------------------------------

/// Decode one tree's packed `nodes` buffer and emit every node through the f64
/// `ModelBuilder` (sklearn.cc:313-347).
#[allow(clippy::too_many_arguments)]
fn build_tree(
    builder: &mut ModelBuilder,
    tree: usize,
    node_count_i64: i64,
    nodes_bytes: &[u8],
    left_cat_bitmap: &[u32],
    layout: &NodeLayout,
    features_map: &[i32],
    categories_map: Option<&[Vec<i64>]>,
) -> Result<(), SklError> {
    // node_count <= INT_MAX overflow guard (sklearn.cc:308-310).
    if node_count_i64 < 0 {
        return Err(SklError::HistGbDecode {
            offset: 0,
            detail: format!("tree {tree}: negative node_count {node_count_i64}"),
        });
    }
    if node_count_i64 > i32::MAX as i64 {
        return Err(SklError::HistGbDecode {
            offset: 0,
            detail: format!("tree {tree}: node_count {node_count_i64} exceeds i32::MAX"),
        });
    }
    let n_nodes = node_count_i64 as usize;

    // Buffer-length guard BEFORE any field read (T-04-18): the packed buffer must
    // hold exactly `n_nodes * itemsize` bytes.
    let need = n_nodes
        .checked_mul(layout.itemsize)
        .ok_or_else(|| SklError::HistGbDecode {
            offset: 0,
            detail: format!("tree {tree}: node_count * itemsize overflows usize"),
        })?;
    if nodes_bytes.len() < need {
        return Err(SklError::HistGbDecode {
            offset: 0,
            detail: format!(
                "tree {tree}: nodes buffer {} bytes < node_count {n_nodes} x itemsize {} = {need}",
                nodes_bytes.len(),
                layout.itemsize
            ),
        });
    }

    builder.start_tree()?;
    for node_id in 0..n_nodes {
        let base = node_id * layout.itemsize;
        let rec = &nodes_bytes[base..base + layout.itemsize];
        let node = decode_node(rec, layout)?;

        builder.start_node(node_id as i32)?;
        // leaf iff left <= 0 (NOT == -1) — sklearn.cc:317,320. Upstream reads the
        // child id as `static_cast<int>(node.left)` (signed) before the `<= 0`
        // test, so a `node.left` in `[2^31, 2^32)` reinterprets as negative and
        // is treated as a LEAF. Replicate that signed-cast semantics here rather
        // than testing the raw `u32 == 0`.
        let left_child_id = node.left as i32;
        let right_child_id = node.right as i32;
        if left_child_id <= 0 {
            builder.leaf_scalar_f64(node.value)?;
        } else {
            // split_index = features_map[feature_idx] — ALWAYS remap (Pitfall 4).
            let fid = usize::try_from(node.feature_idx).map_err(|_| SklError::HistGbDecode {
                offset: 0,
                detail: format!(
                    "tree {tree} node {node_id}: negative feature_idx {}",
                    node.feature_idx
                ),
            })?;
            let split_index = *features_map.get(fid).ok_or_else(|| SklError::HistGbDecode {
                offset: 0,
                detail: format!(
                    "tree {tree} node {node_id}: feature_idx {fid} out of range for \
                     features_map (len {})",
                    features_map.len()
                ),
            })?;
            let default_left = node.missing_go_to_left == 1;
            let left_child = left_child_id;
            let right_child = right_child_id;

            if node.is_categorical == 1 {
                let left_categories = decode_categorical(
                    node.feature_idx,
                    node.bitset_idx,
                    left_cat_bitmap,
                    categories_map,
                )?;
                builder.categorical_test(
                    split_index,
                    default_left,
                    &left_categories,
                    false, // category_list_right_child (sklearn.cc:336)
                    left_child,
                    right_child,
                )?;
            } else {
                // num_threshold read DIRECTLY (Pitfall 5 — no _bin_mapper recon).
                builder.numerical_test_f64(
                    split_index,
                    node.num_threshold,
                    default_left,
                    Operator::kLE,
                    left_child,
                    right_child,
                )?;
            }
            builder.gain(node.gain)?;
        }
        builder.data_count(node.count as u64)?;
        builder.end_node()?;
    }
    builder.end_tree()?;
    Ok(())
}

/// Drive every tree through the builder and commit, given finalized metadata and
/// the per-tree packed buffers. Shared by the regressor and classifier entry
/// points (the only difference is the [`BuilderMetadata`]).
#[allow(clippy::too_many_arguments)]
fn build_model(
    n_trees: usize,
    metadata: BuilderMetadata,
    itemsize: usize,
    node_count: &[i64],
    nodes: &[&[u8]],
    raw_left_cat_bitsets: &[&[u32]],
    features_map: &[i32],
    categories_map: Option<&[Vec<i64>]>,
) -> Result<Model, SklError> {
    let layout = NodeLayout::for_itemsize(itemsize)?;

    if node_count.len() != n_trees {
        return Err(SklError::TreeCountMismatch {
            field: "node_count",
            expected: n_trees,
            got: node_count.len(),
        });
    }
    if nodes.len() != n_trees {
        return Err(SklError::TreeCountMismatch {
            field: "nodes",
            expected: n_trees,
            got: nodes.len(),
        });
    }
    if raw_left_cat_bitsets.len() != n_trees {
        return Err(SklError::TreeCountMismatch {
            field: "raw_left_cat_bitsets",
            expected: n_trees,
            got: raw_left_cat_bitsets.len(),
        });
    }

    let mut builder = ModelBuilder::new(metadata)?;
    for t in 0..n_trees {
        build_tree(
            &mut builder,
            t,
            node_count[t],
            nodes[t],
            raw_left_cat_bitsets[t],
            &layout,
            features_map,
            categories_map,
        )?;
    }
    Ok(builder.commit_model()?)
}

/// Validate a model scalar that must be at least 1.
fn require_positive(field: &'static str, value: i32) -> Result<i32, SklError> {
    if value < 1 {
        return Err(SklError::InvalidScalar {
            field,
            value: value as i64,
            reason: "must be at least 1",
        });
    }
    Ok(value)
}

/// Load a `HistGradientBoostingRegressor` (SKL-04).
///
/// Metadata per `HistGradientBoostingRegressorMixIn` (sklearn.cc:135-156):
/// `task=kRegressor`, `average_tree_output=false`, `num_target=1`,
/// `num_class={1}`, `leaf_vector_shape={1,1}`, `postprocessor="identity"`,
/// `base_scores={baseline_prediction}`.
///
/// `nodes` are the per-tree packed `HistGradientBoostingNode` byte buffers;
/// `expected_sizeof_node_struct` selects the 52/56-byte layout. `features_map` is
/// applied to EVERY split index; `categories_map` (when `Some`) remaps
/// categorical bit values.
#[allow(clippy::too_many_arguments)]
pub fn load_hist_gradient_boosting_regressor(
    n_iter: i32,
    n_features: i32,
    expected_sizeof_node_struct: usize,
    node_count: &[i64],
    nodes: &[&[u8]],
    raw_left_cat_bitsets: &[&[u32]],
    features_map: &[i32],
    categories_map: Option<&[Vec<i64>]>,
    baseline_prediction: f64,
) -> Result<Model, SklError> {
    let n_iter = require_positive("n_iter", n_iter)?;
    let n_features = require_positive("n_features", n_features)?;

    let metadata = BuilderMetadata {
        num_feature: n_features,
        task_type: TaskType::kRegressor,
        average_tree_output: false,
        num_target: 1,
        num_class: vec![1],
        leaf_vector_shape: vec![1, 1],
        target_id: vec![0; n_iter as usize],
        class_id: vec![0; n_iter as usize],
        postprocessor: "identity".to_string(),
        base_scores: vec![baseline_prediction],
        attributes: None,
    };
    build_model(
        n_iter as usize,
        metadata,
        expected_sizeof_node_struct,
        node_count,
        nodes,
        raw_left_cat_bitsets,
        features_map,
        categories_map,
    )
}

/// Load a `HistGradientBoostingClassifier` (SKL-04).
///
/// `n_classes >= 2` (sklearn.cc:386 analog). Binary
/// (`HistGradientBoostingBinaryClassifierMixIn`, sklearn.cc:158-174):
/// `task=kBinaryClf`, `postprocessor="sigmoid"`, `n_trees = n_iter`,
/// `class_id=vec![0; n_iter]`, `base_scores={baseline[0]}`. Multiclass
/// (`HistGradientBoostingMulticlassClassifierMixIn`, sklearn.cc:176-198):
/// `task=kMultiClf`, `postprocessor="softmax"`, `n_trees = n_iter * n_classes`,
/// `class_id[tree] = tree % n_classes` round-robin,
/// `base_scores = baseline[..n_classes]`.
#[allow(clippy::too_many_arguments)]
pub fn load_hist_gradient_boosting_classifier(
    n_iter: i32,
    n_features: i32,
    n_classes: i32,
    expected_sizeof_node_struct: usize,
    node_count: &[i64],
    nodes: &[&[u8]],
    raw_left_cat_bitsets: &[&[u32]],
    features_map: &[i32],
    categories_map: Option<&[Vec<i64>]>,
    baseline_prediction: &[f64],
) -> Result<Model, SklError> {
    let n_iter = require_positive("n_iter", n_iter)?;
    let n_features = require_positive("n_features", n_features)?;
    if n_classes < 2 {
        return Err(SklError::InvalidScalar {
            field: "n_classes",
            value: n_classes as i64,
            reason: "must be at least 2",
        });
    }

    if n_classes > 2 {
        // Multiclass — softmax, n_iter * n_classes trees, round-robin class_id.
        let n_trees = (n_iter as i64) * (n_classes as i64);
        let n_trees = usize::try_from(n_trees).map_err(|_| SklError::InvalidScalar {
            field: "n_iter*n_classes",
            value: n_trees,
            reason: "exceeds usize",
        })?;
        if baseline_prediction.len() < n_classes as usize {
            return Err(SklError::DimensionMismatch {
                tree: 0,
                field: "baseline_prediction",
                expected: n_classes as usize,
                got: baseline_prediction.len(),
            });
        }
        let class_id: Vec<i32> = (0..n_trees as i32).map(|t| t % n_classes).collect();
        let metadata = BuilderMetadata {
            num_feature: n_features,
            task_type: TaskType::kMultiClf,
            average_tree_output: false,
            num_target: 1,
            num_class: vec![n_classes],
            leaf_vector_shape: vec![1, 1],
            target_id: vec![0; n_trees],
            class_id,
            postprocessor: "softmax".to_string(),
            base_scores: baseline_prediction[..n_classes as usize].to_vec(),
            attributes: None,
        };
        build_model(
            n_trees,
            metadata,
            expected_sizeof_node_struct,
            node_count,
            nodes,
            raw_left_cat_bitsets,
            features_map,
            categories_map,
        )
    } else {
        // Binary — sigmoid, n_iter trees, class_id all 0.
        if baseline_prediction.is_empty() {
            return Err(SklError::DimensionMismatch {
                tree: 0,
                field: "baseline_prediction",
                expected: 1,
                got: 0,
            });
        }
        let metadata = BuilderMetadata {
            num_feature: n_features,
            task_type: TaskType::kBinaryClf,
            average_tree_output: false,
            num_target: 1,
            num_class: vec![1],
            leaf_vector_shape: vec![1, 1],
            target_id: vec![0; n_iter as usize],
            class_id: vec![0; n_iter as usize],
            postprocessor: "sigmoid".to_string(),
            base_scores: vec![baseline_prediction[0]],
            attributes: None,
        };
        build_model(
            n_iter as usize,
            metadata,
            expected_sizeof_node_struct,
            node_count,
            nodes,
            raw_left_cat_bitsets,
            features_map,
            categories_map,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 56-byte packed node from explicit field values (the i64
    /// feature_idx variant — the layout present in the frozen fixtures).
    #[allow(clippy::too_many_arguments)]
    fn pack_node_56(
        value: f64,
        count: u32,
        feature_idx: i64,
        num_threshold: f64,
        missing_go_to_left: u8,
        left: u32,
        right: u32,
        gain: f64,
        depth: u32,
        is_leaf: u8,
        bin_threshold: u8,
        is_categorical: u8,
        bitset_idx: u32,
    ) -> [u8; 56] {
        let mut b = [0u8; 56];
        b[0..8].copy_from_slice(&value.to_le_bytes());
        b[8..12].copy_from_slice(&count.to_le_bytes());
        b[12..20].copy_from_slice(&feature_idx.to_le_bytes());
        b[20..28].copy_from_slice(&num_threshold.to_le_bytes());
        b[28] = missing_go_to_left;
        b[29..33].copy_from_slice(&left.to_le_bytes());
        b[33..37].copy_from_slice(&right.to_le_bytes());
        b[37..45].copy_from_slice(&gain.to_le_bytes());
        b[45..49].copy_from_slice(&depth.to_le_bytes());
        b[49] = is_leaf;
        b[50] = bin_threshold;
        b[51] = is_categorical;
        b[52..56].copy_from_slice(&bitset_idx.to_le_bytes());
        b
    }

    /// Build a 52-byte packed node (the i32 feature_idx variant).
    #[allow(clippy::too_many_arguments)]
    fn pack_node_52(
        value: f64,
        count: u32,
        feature_idx: i32,
        num_threshold: f64,
        missing_go_to_left: u8,
        left: u32,
        right: u32,
        gain: f64,
        depth: u32,
        is_leaf: u8,
        bin_threshold: u8,
        is_categorical: u8,
        bitset_idx: u32,
    ) -> [u8; 52] {
        let mut b = [0u8; 52];
        b[0..8].copy_from_slice(&value.to_le_bytes());
        b[8..12].copy_from_slice(&count.to_le_bytes());
        b[12..16].copy_from_slice(&feature_idx.to_le_bytes());
        b[16..24].copy_from_slice(&num_threshold.to_le_bytes());
        b[24] = missing_go_to_left;
        b[25..29].copy_from_slice(&left.to_le_bytes());
        b[29..33].copy_from_slice(&right.to_le_bytes());
        b[33..41].copy_from_slice(&gain.to_le_bytes());
        b[41..45].copy_from_slice(&depth.to_le_bytes());
        b[45] = is_leaf;
        b[46] = bin_threshold;
        b[47] = is_categorical;
        b[48..52].copy_from_slice(&bitset_idx.to_le_bytes());
        b
    }

    #[test]
    fn histgb_decode_52byte_fields_at_documented_offsets() {
        let rec = pack_node_52(
            1.5, 42, 7, 0.625, 1, 3, 4, 2.25, 9, 0, 35, 1, 6,
        );
        let layout = NodeLayout::for_itemsize(52).unwrap();
        let n = decode_node(&rec, &layout).unwrap();
        assert_eq!(n.value, 1.5);
        assert_eq!(n.count, 42);
        assert_eq!(n.feature_idx, 7);
        assert_eq!(n.num_threshold, 0.625);
        assert_eq!(n.missing_go_to_left, 1);
        assert_eq!(n.left, 3);
        assert_eq!(n.right, 4);
        assert_eq!(n.gain, 2.25);
        assert_eq!(n.is_categorical, 1);
        assert_eq!(n.bitset_idx, 6);
    }

    #[test]
    fn histgb_decode_56byte_fields_at_shifted_offsets() {
        let rec = pack_node_56(
            -2.0, 100, 0x1_0000_0001, 0.5, 0, 11, 12, 0.0, 3, 0, 1, 0, 0,
        );
        let layout = NodeLayout::for_itemsize(56).unwrap();
        let n = decode_node(&rec, &layout).unwrap();
        assert_eq!(n.value, -2.0);
        assert_eq!(n.count, 100);
        // i64 feature_idx — a value that would NOT fit i32 proves 64-bit decode.
        assert_eq!(n.feature_idx, 0x1_0000_0001);
        assert_eq!(n.num_threshold, 0.5);
        assert_eq!(n.missing_go_to_left, 0);
        assert_eq!(n.left, 11);
        assert_eq!(n.right, 12);
        assert_eq!(n.is_categorical, 0);
    }

    #[test]
    fn histgb_decode_rejects_bad_itemsize() {
        let err = NodeLayout::for_itemsize(53).unwrap_err();
        assert!(matches!(err, SklError::HistGbDecode { .. }));
        let err = NodeLayout::for_itemsize(48).unwrap_err();
        assert!(matches!(err, SklError::HistGbDecode { .. }));
        // 52 and 56 are accepted.
        assert!(NodeLayout::for_itemsize(52).is_ok());
        assert!(NodeLayout::for_itemsize(56).is_ok());
    }

    #[test]
    fn histgb_decode_rejects_short_buffer_before_field_read() {
        // node_count says 2 nodes (2*56=112 bytes) but only 56 supplied.
        let one = pack_node_56(0.0, 1, 0, 0.0, 0, 0, 0, 0.0, 0, 1, 0, 0, 0);
        let metadata = BuilderMetadata {
            num_feature: 1,
            task_type: TaskType::kRegressor,
            average_tree_output: false,
            num_target: 1,
            num_class: vec![1],
            leaf_vector_shape: vec![1, 1],
            target_id: vec![0],
            class_id: vec![0],
            postprocessor: "identity".to_string(),
            base_scores: vec![0.0],
            attributes: None,
        };
        let res = build_model(
            1,
            metadata,
            56,
            &[2], // claims 2 nodes
            &[&one[..]],
            &[&[][..]],
            &[0],
            None,
        );
        assert!(matches!(res, Err(SklError::HistGbDecode { .. })));
    }

    #[test]
    fn histgb_leaf_detection_uses_le_zero_not_minus_one() {
        // A node with left==0 is a LEAF (HistGB rule), not an internal split.
        // Build a single-leaf tree (one node, left=0) and confirm it loads as a
        // leaf returning its value via prediction.
        let leaf = pack_node_56(3.5, 10, 0, 0.0, 0, 0, 0, 0.0, 0, 1, 0, 0, 0);
        let model = load_hist_gradient_boosting_regressor(
            1,
            1,
            56,
            &[1],
            &[&leaf[..]],
            &[&[][..]],
            &[0],
            None,
            0.0,
        )
        .expect("single-leaf histgb loads");
        let out = treelite_gtil::predict(&model, &[0.0_f32], 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 3.5_f32, epsilon = 1e-5);
    }

    #[test]
    fn histgb_features_map_remaps_split_index() {
        // Root splits on feature_idx=0 with num_threshold 0.5; features_map maps
        // internal index 0 -> input feature 1. So routing must read input col 1,
        // NOT col 0. Build: node0 internal (left=1,right=2), node1/node2 leaves.
        let root = pack_node_56(0.0, 10, 0, 0.5, 0, 1, 2, 1.0, 0, 0, 0, 0, 0);
        let left = pack_node_56(10.0, 6, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let right = pack_node_56(20.0, 4, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&root);
        buf.extend_from_slice(&left);
        buf.extend_from_slice(&right);
        // features_map: internal idx 0 -> input feature 1.
        let model = load_hist_gradient_boosting_regressor(
            1,
            2,
            56,
            &[3],
            &[&buf[..]],
            &[&[][..]],
            &[1, 0], // <- remap: feature_idx 0 reads input col 1
            None,
            0.0,
        )
        .expect("histgb loads");
        // Input col1 = 0.0 (<= 0.5) -> LEFT leaf (10.0); col0 is a decoy (99.0).
        let out = treelite_gtil::predict(&model, &[99.0_f32, 0.0], 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 10.0_f32, epsilon = 1e-5);
        // Input col1 = 1.0 (> 0.5) -> RIGHT leaf (20.0).
        let out = treelite_gtil::predict(&model, &[99.0_f32, 1.0], 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 20.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn histgb_check_bit_matches_8row_stride_reference() {
        // Two 256-bit rows. Row 0 has bits {2, 3} set (12 = 0b1100); row 1 has
        // bit {1} set (2 = 0b10). The `8*row` stride must select the correct row.
        let bitmap: [u32; 16] = [12, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
        // Row 0.
        assert!(!check_bit(&bitmap, 0, 0));
        assert!(!check_bit(&bitmap, 1, 0));
        assert!(check_bit(&bitmap, 2, 0));
        assert!(check_bit(&bitmap, 3, 0));
        assert!(!check_bit(&bitmap, 4, 0));
        // Row 1 — same words would be wrong without the 8*row stride.
        assert!(!check_bit(&bitmap, 0, 1));
        assert!(check_bit(&bitmap, 1, 1));
        assert!(!check_bit(&bitmap, 2, 1));
        // A high category (val 33) lands in word index 1 of its row.
        let mut bm2 = [0u32; 8];
        bm2[1] = 1 << (33 % 32); // bit 33 -> word 1, bit 1
        assert!(check_bit(&bm2, 33, 0));
        assert!(!check_bit(&bm2, 32, 0));
    }

    #[test]
    fn histgb_decode_categorical_identity_when_no_categories_map() {
        // Row 0 bits {2,3} set; no categories_map -> identity (cats 2 and 3).
        let bitmap: [u32; 8] = [12, 0, 0, 0, 0, 0, 0, 0];
        let cats = decode_categorical(0, 0, &bitmap, None).unwrap();
        assert_eq!(cats, vec![2, 3]);
    }

    #[test]
    fn histgb_decode_categorical_applies_categories_map_remap() {
        // Row 0 bits {0,1} set; categories_map[0] = [10, 20, 30, 40] remaps
        // cat 0 -> 10, cat 1 -> 20 (NOT identity 0,1 — Pitfall 4).
        let bitmap: [u32; 8] = [0b11, 0, 0, 0, 0, 0, 0, 0];
        let cm = vec![vec![10_i64, 20, 30, 40]];
        let cats = decode_categorical(0, 0, &bitmap, Some(&cm)).unwrap();
        assert_eq!(cats, vec![10, 20]);
    }

    #[test]
    fn histgb_decode_categorical_rejects_out_of_range_bitset_idx() {
        // bitmap has only one 256-bit row (8 words) but bitset_idx=1 needs row 1
        // (words 8..16) -> typed HistGbDecode, never OOB.
        let bitmap: [u32; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
        let res = decode_categorical(0, 1, &bitmap, None);
        assert!(matches!(res, Err(SklError::HistGbDecode { .. })));
    }

    #[test]
    fn histgb_categorical_node_routes_by_category_membership() {
        // Root is a categorical split on feature_idx 0 with left categories
        // {1, 2} (row 0 bits 1 and 2 = 0b110 = 6). A category MATCH routes LEFT.
        // node0 internal (left=1,right=2), node1/node2 leaves.
        let root = pack_node_56(0.0, 10, 0, 0.0, 0, 1, 2, 1.0, 0, 0, 0, 1, 0);
        let left = pack_node_56(10.0, 6, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let right = pack_node_56(20.0, 4, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&root);
        buf.extend_from_slice(&left);
        buf.extend_from_slice(&right);
        let bitmap: Vec<u32> = vec![0b110, 0, 0, 0, 0, 0, 0, 0]; // cats {1,2}
        let model = load_hist_gradient_boosting_regressor(
            1,
            1,
            56,
            &[3],
            &[&buf[..]],
            &[&bitmap[..]],
            &[0],
            None,
            0.0,
        )
        .expect("categorical histgb loads");
        // Feature 0 = 1.0 (category 1, in {1,2}) -> LEFT leaf (10.0).
        let out = treelite_gtil::predict(&model, &[1.0_f32], 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 10.0_f32, epsilon = 1e-5);
        // Feature 0 = 3.0 (category 3, NOT in {1,2}) -> RIGHT leaf (20.0).
        let out = treelite_gtil::predict(&model, &[3.0_f32], 1).expect("predict");
        approx::assert_abs_diff_eq!(out[0], 20.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn histgb_categorical_malformed_bitmap_is_typed_error_not_panic() {
        // A categorical node whose bitset_idx points past the supplied bitmap
        // must surface a typed error during load, never an OOB panic (T-04-20).
        let root = pack_node_56(0.0, 10, 0, 0.0, 0, 1, 2, 1.0, 0, 0, 0, 1, 5);
        let left = pack_node_56(1.0, 6, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let right = pack_node_56(2.0, 4, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&root);
        buf.extend_from_slice(&left);
        buf.extend_from_slice(&right);
        let bitmap: Vec<u32> = vec![0u32; 8]; // only row 0, but bitset_idx=5
        let res = load_hist_gradient_boosting_regressor(
            1, 1, 56, &[3], &[&buf[..]], &[&bitmap[..]], &[0], None, 0.0,
        );
        assert!(matches!(res, Err(SklError::HistGbDecode { .. })));
    }

    #[test]
    fn histgb_feature_idx_out_of_range_is_typed_error() {
        // feature_idx=5 but features_map has len 1 -> typed HistGbDecode.
        let root = pack_node_56(0.0, 10, 5, 0.5, 0, 1, 2, 1.0, 0, 0, 0, 0, 0);
        let left = pack_node_56(1.0, 6, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let right = pack_node_56(2.0, 4, 0, 0.0, 0, 0, 0, 0.0, 1, 1, 0, 0, 0);
        let mut buf = Vec::new();
        buf.extend_from_slice(&root);
        buf.extend_from_slice(&left);
        buf.extend_from_slice(&right);
        let res = load_hist_gradient_boosting_regressor(
            1, 1, 56, &[3], &[&buf[..]], &[&[][..]], &[0], None, 0.0,
        );
        assert!(matches!(res, Err(SklError::HistGbDecode { .. })));
    }
}

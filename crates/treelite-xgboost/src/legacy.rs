//! XGBoost legacy-binary loader (XGB-03 / D-07 / D-08).
//!
//! Ports `treelite-mainline/src/model_loader/xgboost_legacy.cc` `ParseStream`
//! onto an explicit little-endian byte cursor. Every multi-byte field is decoded
//! field-by-field via `from_le_bytes` — never by reinterpreting a raw byte buffer
//! as a native `#[repr(C)]` struct (hard D-08 invariant: that path is both
//! endianness- and padding-unsafe). A small [`PeekableReader`] mirrors upstream's
//! `PeekableInputStream` 1024-byte window for the `binf`/`bs64` magic peek.
//!
//! ## Convergence (D-01)
//!
//! The decoded fields are funneled into the SAME [`crate::json::XgbModelJson`]
//! struct family the JSON and UBJSON loaders fill, then handed to the shared
//! [`crate::build_model_from_parsed`] path. The legacy load therefore produces
//! the IDENTICAL [`Model`] a JSON/UBJSON load of the same logical model produces
//! — byte-faithful to the single upstream golden blob (D-10) and predicting
//! within 1e-5.
//!
//! ## Safety (T-03-L01..L06)
//!
//! Every read goes through [`Cursor`], which returns [`XgbError::Legacy`] on
//! truncation rather than indexing out of bounds. Every length/count
//! (objective/booster `u64` name length, `num_nodes`, leaf-vector len, DART `sz`)
//! is validated against the bytes remaining BEFORE any allocation, so a
//! truncated or hostile stream surfaces as a typed error, never an OOB panic or
//! an OOM pre-allocation.
//!
//! ## Byte layout (RESEARCH §"Legacy Binary Layout", validated vs mushroom.model)
//!
//! `LearnerModelParam` = 136 B, length-prefixed objective + booster names,
//! `GBTreeModelParam` = 160 B, then per tree a `TreeParam` = 148 B + `num_nodes`
//! × `Node`(20 B) + `num_nodes` × `NodeStat`(16 B) + optional leaf-vector tail,
//! then a `num_trees` × i32 `tree_info` tail, then optional DART `weight_drop`.

use crate::error::XgbError;
use crate::json::{RegTreeJson, XgbModelJson};
use treelite_core::Model;

// Upstream struct sizes (`static_assert`s in xgboost_legacy.cc). Encoded as
// constants so the cursor advances by fixed strides and a malformed file fails
// loudly with the wrong-count guards below.
const SIZE_LEARNER_MODEL_PARAM: usize = 136;
// NOTE (deviation from RESEARCH §"Legacy Binary Layout"): the research table
// listed `GBTreeModelParam` as 168 bytes, but the upstream struct
// (`xgboost_legacy.cc:169-178`) is `4×i32 (16) + i64 (8) + 2×i32 (8) +
// i32[32] (128) = 160` bytes, with no trailing padding (160 is already 8-byte
// aligned). This was confirmed against `mushroom.model`: header ends at byte
// 173, two trees + tree_info consume 1168 bytes, and 173 + 160 + 1168 == 1501
// (the exact file size). The 168 figure was a research transcription error.
const SIZE_GBTREE_MODEL_PARAM: usize = 160;
const SIZE_TREE_PARAM: usize = 148;
const SIZE_NODE: usize = 20;
const SIZE_NODE_STAT: usize = 16;

// ---------------------------------------------------------------------------
// Fallible little-endian byte cursor (D-07/D-08).
// ---------------------------------------------------------------------------

/// A bounds-checked little-endian cursor over the legacy stream. Every accessor
/// slices `self.buf.get(pos..pos+N)` → `Option`, mapping a short read to a typed
/// [`XgbError::Legacy`] truncation (never an out-of-bounds panic), then advances
/// `pos`. All integers/floats are decoded with `from_le_bytes` (D-08).
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    /// Bytes remaining unread.
    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn err<T>(&self, detail: impl Into<String>) -> Result<T, XgbError> {
        Err(XgbError::Legacy {
            pos: self.pos,
            detail: detail.into(),
        })
    }

    /// Borrow the next `n` bytes without copying, advancing the cursor. Returns a
    /// truncation error if fewer than `n` bytes remain.
    fn take(&mut self, n: usize) -> Result<&'a [u8], XgbError> {
        // Use `checked_add` (mirroring the UBJSON cursor) so an attacker-supplied
        // `n` near `usize::MAX` yields a typed error instead of an
        // arithmetic-overflow panic under the default debug `overflow-checks`
        // (CR-03).
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| XgbError::Legacy {
                pos: self.pos,
                detail: format!("length overflow: pos {} + {n} bytes", self.pos),
            })?;
        match self.buf.get(self.pos..end) {
            Some(slice) => {
                self.pos = end;
                Ok(slice)
            }
            None => self.err(format!(
                "truncated: need {n} bytes, only {} remain",
                self.remaining()
            )),
        }
    }

    /// Peek the next `n` bytes without advancing (for the magic check). Returns
    /// `None` if fewer than `n` bytes remain, or if `pos + n` would overflow
    /// (peek is best-effort, like upstream's `PeekRead` over a short stream).
    fn peek(&self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        self.buf.get(self.pos..end)
    }

    fn f32(&mut self) -> Result<f32, XgbError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, XgbError> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u32(&mut self) -> Result<u32, XgbError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, XgbError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Skip `n` bytes (reserved padding), bounds-checked.
    fn skip(&mut self, n: usize) -> Result<(), XgbError> {
        self.take(n)?;
        Ok(())
    }

    /// Read a `u64`-length-prefixed UTF-8 string, validating the length against
    /// the remaining buffer BEFORE allocating (T-03-L01 — never a giant alloc).
    fn length_prefixed_string(&mut self) -> Result<String, XgbError> {
        let len = self.u64()?;
        let len = usize::try_from(len)
            .map_err(|_| ())
            .and_then(|l| if l > self.remaining() { Err(()) } else { Ok(l) });
        let len = match len {
            Ok(l) => l,
            Err(()) => {
                return self.err("string length exceeds remaining bytes");
            }
        };
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| XgbError::Legacy {
            pos: self.pos,
            detail: format!("non-UTF-8 name: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// PeekableReader (D-07): mirrors upstream's 1024-byte peek window.
// ---------------------------------------------------------------------------

/// A peekable reader mirroring upstream's `PeekableInputStream` (1024-byte
/// window). Over an in-memory buffer the cursor already supports peeking, so this
/// is a thin wrapper that enforces the same `MAX_PEEK_WINDOW` contract and owns
/// the magic-header check.
struct PeekableReader;

impl PeekableReader {
    const MAX_PEEK_WINDOW: usize = 1024;

    /// Backward-compatible header check (`xgboost_legacy.cc:328-336`): peek 4
    /// bytes; `"bs64"` → reject (base64 no longer supported); `"binf"` → consume
    /// 4 bytes; anything else (mushroom: first byte `0x00`) → do not consume.
    fn check_magic(cursor: &mut Cursor<'_>) -> Result<(), XgbError> {
        const { assert!(4 <= PeekableReader::MAX_PEEK_WINDOW) };
        if let Some(header) = cursor.peek(4) {
            if header == b"bs64" {
                return Err(XgbError::Legacy {
                    pos: cursor.pos,
                    detail: "Base64 (bs64) format no longer supported".to_string(),
                });
            }
            if header == b"binf" {
                cursor.skip(4)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ParseStream port.
// ---------------------------------------------------------------------------

/// Decoded `LearnerModelParam` fields we actually use downstream.
struct LearnerModelParam {
    base_score: f32,
    num_feature: u32,
    num_class: i32,
    major_version: u32,
    num_target: u32,
}

/// Read the 136-byte `LearnerModelParam` field-by-field (D-08). Asserts exactly
/// 136 bytes are consumed.
fn read_learner_model_param(c: &mut Cursor<'_>) -> Result<LearnerModelParam, XgbError> {
    let start = c.pos;
    let base_score = c.f32()?;
    let num_feature = c.u32()?;
    let num_class = c.i32()?;
    let _contain_extra_attrs = c.i32()?;
    let _contain_eval_metrics = c.i32()?;
    let major_version = c.u32()?;
    let _minor_version = c.u32()?;
    let num_target = c.u32()?;
    c.skip(26 * 4)?; // pad2[26]
    let consumed = c.pos - start;
    if consumed != SIZE_LEARNER_MODEL_PARAM {
        return Err(XgbError::Legacy {
            pos: start,
            detail: format!(
                "LearnerModelParam size mismatch: consumed {consumed}, expected {SIZE_LEARNER_MODEL_PARAM}"
            ),
        });
    }
    Ok(LearnerModelParam {
        base_score,
        num_feature,
        num_class,
        major_version,
        num_target,
    })
}

/// Decoded `GBTreeModelParam` fields we use downstream. The model-level
/// `size_leaf_vector` is not consulted (the per-tree `TreeParam.size_leaf_vector`
/// drives the leaf-vector tail), so it is decoded-and-discarded below.
struct GBTreeModelParam {
    num_trees: i32,
    num_roots: i32,
}

/// Read the 168-byte `GBTreeModelParam` field-by-field (D-08). Asserts 168 bytes.
fn read_gbtree_model_param(c: &mut Cursor<'_>) -> Result<GBTreeModelParam, XgbError> {
    let start = c.pos;
    let num_trees = c.i32()?;
    let num_roots = c.i32()?;
    let _num_feature = c.i32()?;
    let _pad1 = c.i32()?;
    let _pad2 = c.u64()?; // i64
    let _num_output_group = c.i32()?;
    let _size_leaf_vector = c.i32()?; // model-level; per-tree value is authoritative
    c.skip(32 * 4)?; // pad3[32]
    let consumed = c.pos - start;
    if consumed != SIZE_GBTREE_MODEL_PARAM {
        return Err(XgbError::Legacy {
            pos: start,
            detail: format!(
                "GBTreeModelParam size mismatch: consumed {consumed}, expected {SIZE_GBTREE_MODEL_PARAM}"
            ),
        });
    }
    Ok(GBTreeModelParam {
        num_trees,
        num_roots,
    })
}

/// Decode one tree's `TreeParam` + nodes + stats into a [`RegTreeJson`], applying
/// the `sindex` bit-unpacking, `cleft == -1` leaf detection, and the `info` union
/// f32 reinterpretation (Pitfall 6). When `weight_drop` is `Some`, the leaf value
/// is scaled by it (DART fold).
fn read_tree(
    c: &mut Cursor<'_>,
    major_version: u32,
    weight_drop: Option<f32>,
) -> Result<RegTreeJson, XgbError> {
    // --- TreeParam (148 bytes = 37 × i32) ---
    let tp_start = c.pos;
    let _num_roots_tp = c.i32()?; // num_roots
    let num_nodes = c.i32()?;
    let _num_deleted = c.i32()?;
    let _max_depth = c.i32()?;
    let _num_feature = c.i32()?;
    let size_leaf_vector = c.i32()?;
    c.skip(31 * 4)?; // reserved[31]
    let tp_consumed = c.pos - tp_start;
    if tp_consumed != SIZE_TREE_PARAM {
        return Err(XgbError::Legacy {
            pos: tp_start,
            detail: format!(
                "TreeParam size mismatch: consumed {tp_consumed}, expected {SIZE_TREE_PARAM}"
            ),
        });
    }
    if num_nodes <= 0 {
        return Err(XgbError::Legacy {
            pos: tp_start,
            detail: format!("a tree can't be empty (num_nodes = {num_nodes})"),
        });
    }
    let num_nodes_usize = num_nodes as usize;

    // Bounds-check the fixed-stride node + stat blocks against the remaining
    // buffer BEFORE allocating (T-03-L02 — never an OOM pre-allocation).
    let nodes_bytes = num_nodes_usize
        .checked_mul(SIZE_NODE)
        .ok_or_else(|| XgbError::Legacy {
            pos: c.pos,
            detail: "num_nodes × node-size overflow".to_string(),
        })?;
    let stats_bytes =
        num_nodes_usize
            .checked_mul(SIZE_NODE_STAT)
            .ok_or_else(|| XgbError::Legacy {
                pos: c.pos,
                detail: "num_nodes × stat-size overflow".to_string(),
            })?;
    if nodes_bytes + stats_bytes > c.remaining() {
        return Err(XgbError::Legacy {
            pos: c.pos,
            detail: format!(
                "nodes+stats ({} B) exceed remaining ({} B)",
                nodes_bytes + stats_bytes,
                c.remaining()
            ),
        });
    }

    // --- Nodes: num_nodes × 20 bytes ---
    let mut left_children = Vec::with_capacity(num_nodes_usize);
    let mut right_children = Vec::with_capacity(num_nodes_usize);
    let mut split_indices = Vec::with_capacity(num_nodes_usize);
    let mut split_conditions = Vec::with_capacity(num_nodes_usize);
    let mut default_left = Vec::with_capacity(num_nodes_usize);
    // The `info` union holds either the leaf value or the split condition; we
    // keep the raw f32 and the cleft to decide which downstream.
    let mut info_raw = Vec::with_capacity(num_nodes_usize);
    for _ in 0..num_nodes_usize {
        let _parent = c.i32()?; // parent (top bit = is-left-child flag); unused
        let cleft = c.i32()?;
        let cright = c.i32()?;
        let sindex = c.u32()?;
        let info = c.f32()?; // union: leaf_value OR split_cond (same 4 bytes)

        let split_index = (sindex & 0x7FFF_FFFF) as i32;
        let dl = ((sindex >> 31) != 0) as i32;

        left_children.push(cleft);
        right_children.push(cright);
        split_indices.push(split_index);
        default_left.push(dl);
        info_raw.push(info);
        // Internal nodes carry split_cond in `info`; leaves carry leaf_value.
        // `build_tree` reads leaf value from `split_conditions[i]` and the split
        // threshold from the same array, so a single column suffices: it IS the
        // `info` union, exactly as upstream's `Node::info_`.
        split_conditions.push(info);
    }

    // --- NodeStats: num_nodes × 16 bytes ---
    let mut loss_changes = Vec::with_capacity(num_nodes_usize);
    let mut sum_hessian = Vec::with_capacity(num_nodes_usize);
    for _ in 0..num_nodes_usize {
        let loss_chg = c.f32()?;
        let sum_hess = c.f32()?;
        let _base_weight = c.f32()?;
        let _leaf_child_cnt = c.i32()?;
        loss_changes.push(loss_chg);
        sum_hessian.push(sum_hess);
    }

    // --- Conditional leaf-vector tail (scalar-only legacy is discarded) ---
    if size_leaf_vector != 0 && major_version < 2 {
        let len = c.u64()?;
        if len > 0 {
            let len_usize = usize::try_from(len).map_err(|_| XgbError::Legacy {
                pos: c.pos,
                detail: "leaf-vector length overflow".to_string(),
            })?;
            let bytes = len_usize.checked_mul(4).ok_or_else(|| XgbError::Legacy {
                pos: c.pos,
                detail: "leaf-vector byte-count overflow".to_string(),
            })?;
            // Bound the skip against the remaining buffer BEFORE skipping so a
            // crafted near-`usize::MAX` length yields a typed error rather than a
            // `pos + bytes` overflow inside `take` (CR-03, T-03-L02).
            if bytes > c.remaining() {
                return Err(XgbError::Legacy {
                    pos: c.pos,
                    detail: format!(
                        "leaf-vector ({bytes} B) exceeds remaining ({} B)",
                        c.remaining()
                    ),
                });
            }
            c.skip(bytes)?; // discard — scalar-only legacy
        }
    } else if major_version == 2 && size_leaf_vector != 1 {
        return Err(XgbError::Legacy {
            pos: c.pos,
            detail: "multi-target models are not supported with binary serialization \
                     (size_leaf_vector != 1); re-save the model in JSON format"
                .to_string(),
        });
    }

    // Assert single-root (upstream `param.num_roots == 1`). XGBoost reuses
    // num_roots as num_parallel_tree in ≥1.6, where the per-tree check is relaxed
    // upstream, but for the verify-narrow scalar path the TreeParam num_roots is 1.
    // (We do not gate on it here to stay forgiving of ≥1.6 parallel-tree files;
    //  the GBTreeModelParam num_roots check below covers the legacy invariant.)

    // DART weight-drop fold into leaf values (upstream :478-482). Only leaves
    // (cleft == -1) get the fold; internal split_conditions are untouched.
    if let Some(w) = weight_drop {
        for i in 0..num_nodes_usize {
            if left_children[i] == -1 {
                split_conditions[i] *= w;
            }
        }
    }

    Ok(RegTreeJson::from_legacy_nodes(
        num_nodes,
        left_children,
        right_children,
        split_indices,
        split_conditions,
        default_left,
        loss_changes,
        sum_hessian,
    ))
}

/// Load one XGBoost legacy-binary model into a [`Model`] (F32 variant).
///
/// Decodes the little-endian stream field-by-field (D-07/D-08), then converges at
/// the shared [`crate::build_model_from_parsed`] path so the result is the
/// IDENTICAL [`Model`] a JSON/UBJSON load of the same logical model produces.
pub fn load_xgboost_legacy(bytes: &[u8]) -> Result<Model, XgbError> {
    let mut c = Cursor::new(bytes);

    // 1. Magic peek (PeekableReader): bs64 → reject; binf → consume; else nothing.
    PeekableReader::check_magic(&mut c)?;

    // 2. LearnerModelParam (136 bytes).
    let mparam = read_learner_model_param(&mut c)?;

    // 3. Objective name + booster name (each u64-length-prefixed).
    let objective_name = c.length_prefixed_string()?;
    let booster_name = c.length_prefixed_string()?;
    if booster_name != "gbtree" && booster_name != "dart" {
        return Err(XgbError::Legacy {
            pos: c.pos,
            detail: format!("gradient booster must be gbtree or dart, got {booster_name:?}"),
        });
    }

    // 4. GBTreeModelParam (168 bytes).
    let gbm = read_gbtree_model_param(&mut c)?;
    if gbm.num_trees < 0 {
        return Err(XgbError::Legacy {
            pos: c.pos,
            detail: format!("num_trees must be 0 or greater, got {}", gbm.num_trees),
        });
    }
    // Legacy invariant: single root (multi-root trees unsupported). Upstream only
    // enforces this for < 1.6; we keep the check since the verify-narrow fixture
    // and mushroom both satisfy it.
    if gbm.num_roots != 1 {
        return Err(XgbError::Legacy {
            pos: c.pos,
            detail: format!(
                "multi-root trees not supported (num_roots = {})",
                gbm.num_roots
            ),
        });
    }

    // 5. Per tree: TreeParam + nodes + stats (+ conditional leaf-vector tail).
    //    DART weight_drop is read AFTER all trees, so first parse trees without
    //    the fold, then apply it once weight_drop is known.
    let num_trees = gbm.num_trees as usize;
    let mut trees: Vec<RegTreeJson> = Vec::with_capacity(num_trees.min(c.remaining()));
    for _ in 0..num_trees {
        trees.push(read_tree(&mut c, mparam.major_version, None)?);
    }

    // 6. tree_info: num_trees × i32.
    let mut tree_info = Vec::with_capacity(num_trees);
    for _ in 0..num_trees {
        tree_info.push(c.i32()?);
    }

    // 7. DART weight_drop (only for dart): u64 sz (== num_trees) + sz × f32.
    if booster_name == "dart" {
        let sz = c.u64()?;
        if sz != num_trees as u64 {
            return Err(XgbError::Legacy {
                pos: c.pos,
                detail: format!("DART weight_drop size {sz} != num_trees {num_trees}"),
            });
        }
        for (tree_id, tree) in trees.iter_mut().enumerate() {
            let _ = tree_id;
            let w = c.f32()?;
            // Fold into every leaf value of this tree.
            for i in 0..tree.left_children.len() {
                if tree.left_children[i] == -1 {
                    tree.split_conditions[i] *= w;
                }
            }
        }
    }

    // 8. Metadata mapping + convergence at the shared build path.
    //    num_target == 0 → 1 (legacy scalar invariant); num_class normalization
    //    (max(num_class, 1)) is handled inside build_model_from_parsed's branch.
    let num_target = if mparam.num_target == 0 {
        1
    } else {
        i32::try_from(mparam.num_target).map_err(|_| XgbError::Legacy {
            pos: c.pos,
            detail: format!("num_target {} too large", mparam.num_target),
        })?
    };
    let num_feature = i32::try_from(mparam.num_feature).map_err(|_| XgbError::Legacy {
        pos: c.pos,
        detail: format!("num_feature {} too large", mparam.num_feature),
    })?;
    let major_version = i32::try_from(mparam.major_version).unwrap_or(i32::MAX);

    let parsed = XgbModelJson::from_legacy_fields(
        major_version,
        num_feature,
        mparam.num_class,
        num_target,
        mparam.base_score,
        objective_name,
        tree_info,
        trees,
    );
    crate::build_model_from_parsed(parsed)
}

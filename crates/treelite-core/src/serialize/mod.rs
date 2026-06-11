//! The v5 binary (de)serializer (D-10, in-core).
//!
//! Ports `treelite-mainline/src/serializer.cc` (the field walk) and
//! `treelite-mainline/include/treelite/detail/serializer.h` (the framing
//! primitives). Upstream parameterizes `Serializer<MixIn>` over four mixins
//! (stream / buffer / size-calc / pybuffer); here a single
//! [`SerializerBackend`] trait carries the three framing primitives and the
//! header→per-tree field walk is written ONCE against the trait
//! (RESEARCH Pattern 2).
//!
//! The emission order is the load-bearing contract (D-01): it follows
//! `SerializeHeader` (`serializer.cc:91-126`) and `SerializeTree`
//! (`serializer.cc:140-175`) field-for-field — NOT struct declaration order.
//! Each emitted field is annotated with its `serializer.cc:NNN` source line for
//! auditability. The produced bytes are validated byte-for-byte against the
//! frozen upstream `fixtures/golden_v5.bin` (D-02).

pub mod binary;
pub mod error;
pub mod fields;
pub mod json;
pub mod pybuffer;

pub use binary::{BufferBackend, serialize_to_buffer};
pub use error::SerializeError;
pub use json::{dump_as_json, dump_as_json_string};
pub use pybuffer::{Frame, serialize_to_pybuffer};

use crate::enums::{DType, Operator, TaskType, TreeNodeType};
use crate::model::{Model, ModelPreset, ModelVariant};
use crate::serialize::binary::Reader;
use crate::tree::Tree;
use crate::tree_buf::TreeBuf;

/// The three framing primitives every serialize backend implements.
///
/// Mirrors upstream's `SerializeScalar` / `SerializeArray` / `SerializeString`
/// mixin methods (`serializer_mixins.h`). The byte framing is:
///
/// - **scalar**: exactly the raw little-endian bytes, no length prefix.
/// - **array**: a `u64` element count (8 LE bytes) then `count * sizeof(T)` raw
///   bytes; when the count is `0`, ONLY the 8-byte zero is written (no payload).
/// - **string**: a `u64` byte length (8 LE bytes) then the raw UTF-8 bytes (no
///   NUL terminator); when empty, ONLY the 8-byte zero is written.
///
/// (`serializer.h:163/181/201`; RESEARCH § "The v5 Wire Format".)
pub trait SerializerBackend {
    /// Emit a scalar as its raw little-endian bytes (no length prefix).
    fn scalar_le(&mut self, bytes: &[u8]);

    /// Emit an array: a `u64` element `count` then the already-little-endian
    /// `payload` (`count * sizeof(T)` bytes). When `count == 0`, only the
    /// 8-byte zero count is written (`payload` must be empty).
    fn array_u64_prefixed(&mut self, count: usize, payload: &[u8]);

    /// Emit a string: a `u64` byte length then the raw UTF-8 bytes. When empty,
    /// only the 8-byte zero length is written.
    fn string(&mut self, s: &str);
}

// --- field-order helpers shared by every backend (D-01) ---

/// Emit a single array column whose elements are already little-endian POD.
///
/// `count` is the element count; `payload` is the contiguous LE byte image of
/// the column (empty iff `count == 0`). Centralizes the empty-handling so the
/// `count == 0 ⇒ payload empty` invariant (Pitfall 3) holds for every column.
fn emit_array<B: SerializerBackend>(b: &mut B, count: usize, payload: &[u8]) {
    b.array_u64_prefixed(count, payload);
}

/// Reinterpret a `&[T]` POD column as its native-LE byte image (D-02: LE-host only).
///
/// Routes through the same validated [`bytemuck::cast_slice`] seam as
/// `tree_buf.rs::as_bytes` (D-01): no hand-rolled `from_raw_parts`/transmute.
/// `cast_slice` yields the platform's native-endian bytes, which on the
/// x86-64/ROCm manifest host (little-endian) are byte-identical to the old
/// `from_raw_parts` transmute and to upstream's `memcpy` image — gated by
/// `fixtures/golden_v5.bin` / `golden_v5_3format.bin` (D-03). There is NO
/// big-endian byte-swap path (D-02, out of scope); a big-endian host would emit
/// a different image and is unsupported for v1. The `T: bytemuck::Pod` bound
/// (narrowed from the old `T: Copy`) guarantees no padding/invalid bit patterns,
/// and every call site passes an `i32`/`f64`/`i64` Pod column, so the bound
/// change is mechanical. This recast is restricted to the safe `&[T] → &[u8]`
/// EMIT direction; the untrusted deserialize read path (`binary.rs Reader::array`)
/// keeps its element-wise bounds-checked decode and is NOT recast (V5 security).
fn le_bytes_of<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

/// Walk the 20 header fields in EXACT `SerializeHeader` order (D-01).
///
/// `stage_serialization_fields` MUST be called first so the recomputed version
/// triple, `num_tree`, and type tags are populated before they are borrowed
/// (RESEARCH Pattern 5). The caller (`serialize_to_buffer`) does this.
pub(crate) fn serialize_header<B: SerializerBackend>(m: &Model, b: &mut B) {
    // Header 1 — version triple + type tags + num_tree (serializer.cc:96-106).
    b.scalar_le(&m.major_ver().to_le_bytes()); // serializer.cc:96
    b.scalar_le(&m.minor_ver().to_le_bytes()); // serializer.cc:97
    b.scalar_le(&m.patch_ver().to_le_bytes()); // serializer.cc:98
    b.scalar_le(&[m.threshold_type() as u8]); // serializer.cc:101
    b.scalar_le(&[m.leaf_output_type() as u8]); // serializer.cc:102
    b.scalar_le(&m.num_tree().to_le_bytes()); // serializer.cc:106

    // Header 2 — model metadata (serializer.cc:109-121).
    b.scalar_le(&m.num_feature.to_le_bytes()); // serializer.cc:109
    b.scalar_le(&[m.task_type as u8]); // serializer.cc:110
    b.scalar_le(&[m.average_tree_output as u8]); // serializer.cc:111
    b.scalar_le(&m.num_target.to_le_bytes()); // serializer.cc:112
    emit_array(b, m.num_class.len(), le_bytes_of(&m.num_class)); // serializer.cc:113
    emit_array(
        b,
        m.leaf_vector_shape.len(),
        le_bytes_of(&m.leaf_vector_shape),
    ); // serializer.cc:114
    emit_array(b, m.target_id.len(), le_bytes_of(&m.target_id)); // serializer.cc:115
    emit_array(b, m.class_id.len(), le_bytes_of(&m.class_id)); // serializer.cc:116
    b.string(&m.postprocessor); // serializer.cc:117
    b.scalar_le(&m.sigmoid_alpha.to_le_bytes()); // serializer.cc:118
    b.scalar_le(&m.ratio_c.to_le_bytes()); // serializer.cc:119
    emit_array(b, m.base_scores.len(), le_bytes_of(&m.base_scores)); // serializer.cc:120
    b.string(&m.attributes); // serializer.cc:121

    // Extension Slot 1 — per-model optional fields, always 0 (serializer.cc:125).
    b.scalar_le(&m.num_opt_field_per_model().to_le_bytes()); // serializer.cc:125
}

/// Walk every tree in `num_tree` order (`SerializeTrees`, serializer.cc:128-138).
pub(crate) fn serialize_trees<B: SerializerBackend>(m: &Model, b: &mut B) {
    match &m.variant {
        ModelVariant::F32(p) => {
            for tree in &p.trees {
                serialize_tree(tree, b);
            }
        }
        ModelVariant::F64(p) => {
            for tree in &p.trees {
                serialize_tree(tree, b);
            }
        }
    }
}

/// Walk one tree's 25 fields in EXACT `SerializeTree` order (serializer.cc:140-175).
///
/// Note the divergence from struct declaration order: `leaf_value` is emitted
/// BEFORE `threshold`, and the node-statistics group (`data_count` …
/// `gain_present`) is emitted LAST, with each value column immediately followed
/// by its present-flag column (Pitfall 2).
pub(crate) fn serialize_tree<T: Copy + bytemuck::Pod, B: SerializerBackend>(t: &Tree<T>, b: &mut B) {
    b.scalar_le(&t.num_nodes.to_le_bytes()); // serializer.cc:142
    b.scalar_le(&[t.has_categorical_split as u8]); // serializer.cc:143

    let node_type_u8: Vec<u8> = t
        .node_type
        .as_slice()
        .iter()
        .map(|n| *n as i8 as u8)
        .collect();
    emit_array(b, node_type_u8.len(), &node_type_u8); // serializer.cc:144
    emit_array(b, t.cleft.len(), le_bytes_of(t.cleft.as_slice())); // serializer.cc:145
    emit_array(b, t.cright.len(), le_bytes_of(t.cright.as_slice())); // serializer.cc:146
    emit_array(
        b,
        t.split_index.len(),
        le_bytes_of(t.split_index.as_slice()),
    ); // serializer.cc:147
    emit_array(
        b,
        t.default_left.len(),
        bool_bytes(t.default_left.as_slice()),
    ); // serializer.cc:148
    emit_array(b, t.leaf_value.len(), le_bytes_of(t.leaf_value.as_slice())); // serializer.cc:149
    emit_array(b, t.threshold.len(), le_bytes_of(t.threshold.as_slice())); // serializer.cc:150

    let cmp_u8: Vec<u8> = t.cmp.as_slice().iter().map(|o| *o as i8 as u8).collect();
    emit_array(b, cmp_u8.len(), &cmp_u8); // serializer.cc:151
    emit_array(
        b,
        t.category_list_right_child.len(),
        bool_bytes(t.category_list_right_child.as_slice()),
    ); // serializer.cc:152

    emit_array(
        b,
        t.leaf_vector.len(),
        le_bytes_of(t.leaf_vector.as_slice()),
    ); // serializer.cc:153
    emit_array(
        b,
        t.leaf_vector_begin.len(),
        le_bytes_of(t.leaf_vector_begin.as_slice()),
    ); // serializer.cc:154
    emit_array(
        b,
        t.leaf_vector_end.len(),
        le_bytes_of(t.leaf_vector_end.as_slice()),
    ); // serializer.cc:155
    emit_array(
        b,
        t.category_list.len(),
        le_bytes_of(t.category_list.as_slice()),
    ); // serializer.cc:156
    emit_array(
        b,
        t.category_list_begin.len(),
        le_bytes_of(t.category_list_begin.as_slice()),
    ); // serializer.cc:157
    emit_array(
        b,
        t.category_list_end.len(),
        le_bytes_of(t.category_list_end.as_slice()),
    ); // serializer.cc:158

    // Node statistics — value column immediately followed by its present flag.
    emit_array(b, t.data_count.len(), le_bytes_of(t.data_count.as_slice())); // serializer.cc:161
    emit_array(
        b,
        t.data_count_present.len(),
        bool_bytes(t.data_count_present.as_slice()),
    ); // serializer.cc:162
    emit_array(b, t.sum_hess.len(), le_bytes_of(t.sum_hess.as_slice())); // serializer.cc:163
    emit_array(
        b,
        t.sum_hess_present.len(),
        bool_bytes(t.sum_hess_present.as_slice()),
    ); // serializer.cc:164
    emit_array(b, t.gain.len(), le_bytes_of(t.gain.as_slice())); // serializer.cc:165
    emit_array(
        b,
        t.gain_present.len(),
        bool_bytes(t.gain_present.as_slice()),
    ); // serializer.cc:166

    // Extension slots 2 & 3 — per-tree / per-node optional fields, always 0.
    b.scalar_le(&t.num_opt_field_per_tree.to_le_bytes()); // serializer.cc:170
    b.scalar_le(&t.num_opt_field_per_node.to_le_bytes()); // serializer.cc:174
}

/// View a `&[bool]` as its 1-byte-per-element byte image (NO bit-packing).
///
/// `ContiguousArray<bool>` upstream is a real 1-byte-per-element buffer, NOT a
/// bit-packed `std::vector<bool>` (`contiguous_array.h:59`; Pitfall 5). `bool`
/// in Rust is guaranteed 1 byte with `false == 0u8`, `true == 1u8`, so a direct
/// byte reinterpret reproduces upstream byte-for-byte.
fn bool_bytes(slice: &[bool]) -> &[u8] {
    // SAFETY: `bool` is 1 byte with valid bit patterns 0/1, matching `u8`.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) }
}

// --- deserialize (D-03 v5-gate, ASVS V5 panic-free) -----------------------

/// Deserialize a v5 byte stream back into a [`Model`] (the inverse of
/// [`serialize_to_buffer`]); round-trips bit-identically (SER-01).
///
/// Ports `DeserializeHeaderAndCreateModel` (`serializer.cc:186-245`) and
/// `DeserializeTree` (`serializer.cc:271-312`), reading every field in the SAME
/// order the serializer wrote it. Three gates make it panic-free on hostile
/// input (RESEARCH § Security):
///
/// - **D-03 version gate:** `major_ver != 4` → [`SerializeError::UnsupportedVersion`].
///   The legacy `major_ver == 3` V3 parse path is deliberately NOT ported — a
///   3.9 header is rejected, never mis-routed.
/// - **bounds:** every read is checked by [`Reader`]; a truncated blob →
///   [`SerializeError::TruncatedStream`].
/// - **allocation:** array/string counts are bound against the remaining buffer
///   before allocating → [`SerializeError::CountExceedsBuffer`].
///
/// A `major_ver == 4 && minor_ver > 7` blob is accepted (forward-compatible);
/// the `num_opt_field_*` extension slots are consumed via the bounded skip loop
/// so a forward-version blob never corrupts the cursor position.
pub fn deserialize(buf: &[u8]) -> Result<Model, SerializeError> {
    let mut r = Reader::new(buf);

    // Header 1 — version triple (serializer.cc:188-191).
    let major = r.i32()?;
    let minor = r.i32()?;
    let patch = r.i32()?;
    // D-03: accept only major_ver == 4; reject V3 and every other version.
    if major != 4 {
        return Err(SerializeError::UnsupportedVersion {
            major,
            minor,
            patch,
        });
    }
    // major == 4 && minor > 7 is forward-compatible: accept silently.

    // Type tags (serializer.cc:210-212). Only <f32,f32> / <f64,f64> presets
    // exist; the two tags must agree (tree.h:81-86).
    let thr = r.u8()?;
    let leaf = r.u8()?;
    let dtype = match (thr, leaf) {
        (2, 2) => DType::kFloat32,
        (3, 3) => DType::kFloat64,
        _ => {
            return Err(SerializeError::InvalidTypeTag {
                threshold: thr,
                leaf_output: leaf,
            });
        }
    };

    let num_tree = r.u64()?;

    // Header 2 (serializer.cc:223-235).
    let num_feature = r.i32()?;
    let task_type = decode_task_type(r.u8()?)?;
    let average_tree_output = decode_bool(r.u8()?);
    let num_target = r.i32()?;
    let num_class = r.array(4, decode_i32)?;
    let leaf_vector_shape = r.array(4, decode_i32)?;
    let target_id = r.array(4, decode_i32)?;
    let class_id = r.array(4, decode_i32)?;
    let postprocessor = r.string()?;
    let sigmoid_alpha = r.f32()?;
    let ratio_c = r.f32()?;
    let base_scores = r.array(8, decode_f64)?;
    let attributes = r.string()?;

    // Extension Slot 1 — per-model optional fields (serializer.cc:238-242).
    let nopt_model = r.i32()?;
    if nopt_model < 0 {
        return Err(SerializeError::NegativeOptFieldCount { count: nopt_model });
    }
    for _ in 0..nopt_model {
        r.skip_optional_field()?;
    }

    // Build the variant + trees in num_tree order.
    let variant = match dtype {
        DType::kFloat32 => {
            let mut trees = Vec::with_capacity(num_tree.min(MAX_TREES) as usize);
            for _ in 0..num_tree {
                trees.push(deserialize_tree::<f32>(&mut r, decode_f32)?);
            }
            ModelVariant::F32(ModelPreset::new(trees))
        }
        DType::kFloat64 => {
            let mut trees = Vec::with_capacity(num_tree.min(MAX_TREES) as usize);
            for _ in 0..num_tree {
                trees.push(deserialize_tree::<f64>(&mut r, decode_f64)?);
            }
            ModelVariant::F64(ModelPreset::new(trees))
        }
        _ => unreachable!("dtype already validated to float32/float64"),
    };

    // Reject trailing bytes (a well-formed v5 stream is fully consumed).
    if r.offset() != r.total() {
        return Err(SerializeError::TrailingBytes {
            offset: r.offset(),
            total: r.total(),
            trailing: r.total() - r.offset(),
        });
    }

    let mut model = Model::new(variant);
    model.num_feature = num_feature;
    model.task_type = task_type;
    model.average_tree_output = average_tree_output;
    model.num_target = num_target;
    // MEM-02: `r.array(..)`/`r.string()` still return `Vec`/`String`; the migrated
    // `Model` fields are `SmallVec`/`CompactString`. `.into()` converts via the
    // `SmallVec: From<Vec<T>>` / `CompactString: From<String>` impls (zero behavior
    // change — the bytes already decoded element-wise; this is the storage move).
    model.num_class = num_class.into();
    model.leaf_vector_shape = leaf_vector_shape.into();
    model.target_id = target_id.into();
    model.class_id = class_id.into();
    model.postprocessor = postprocessor.into();
    model.sigmoid_alpha = sigmoid_alpha;
    model.ratio_c = ratio_c;
    model.base_scores = base_scores.into();
    model.attributes = attributes.into();
    Ok(model)
}

/// A defensive pre-allocation cap on `num_tree` (the actual loop is still bound
/// by the per-read buffer checks; this only caps the speculative `with_capacity`
/// so a hostile `num_tree` cannot request a huge allocation up front).
const MAX_TREES: u64 = 1 << 20;

/// Read one tree's 25 fields in EXACT `DeserializeTree` order (serializer.cc:271-312).
fn deserialize_tree<T: Copy>(
    r: &mut Reader<'_>,
    decode_t: impl Fn(&[u8]) -> Result<T, SerializeError> + Copy,
) -> Result<Tree<T>, SerializeError> {
    let elem_t = std::mem::size_of::<T>();

    let num_nodes = r.i32()?;
    let has_categorical_split = decode_bool(r.u8()?);

    let node_type = r.array(1, decode_node_type)?;
    let cleft = r.array(4, decode_i32)?;
    let cright = r.array(4, decode_i32)?;
    let split_index = r.array(4, decode_i32)?;
    let default_left = r.array(1, |b| Ok(decode_bool(b[0])))?;
    let leaf_value = r.array(elem_t, decode_t)?;
    let threshold = r.array(elem_t, decode_t)?;
    let cmp = r.array(1, decode_operator)?;
    let category_list_right_child = r.array(1, |b| Ok(decode_bool(b[0])))?;
    let leaf_vector = r.array(elem_t, decode_t)?;
    let leaf_vector_begin = r.array(8, decode_u64)?;
    let leaf_vector_end = r.array(8, decode_u64)?;
    let category_list = r.array(4, decode_u32)?;
    let category_list_begin = r.array(8, decode_u64)?;
    let category_list_end = r.array(8, decode_u64)?;

    // Node statistics — value column immediately followed by its present flag.
    let data_count = r.array(8, decode_u64)?;
    let data_count_present = r.array(1, |b| Ok(decode_bool(b[0])))?;
    let sum_hess = r.array(8, decode_f64)?;
    let sum_hess_present = r.array(1, |b| Ok(decode_bool(b[0])))?;
    let gain = r.array(8, decode_f64)?;
    let gain_present = r.array(1, |b| Ok(decode_bool(b[0])))?;

    // Extension slot 2 — per-tree optional fields + bounded skip loop.
    let nopt_tree = r.i32()?;
    if nopt_tree < 0 {
        return Err(SerializeError::NegativeOptFieldCount { count: nopt_tree });
    }
    for _ in 0..nopt_tree {
        r.skip_optional_field()?;
    }
    // Extension slot 3 — per-node optional fields + bounded skip loop.
    let nopt_node = r.i32()?;
    if nopt_node < 0 {
        return Err(SerializeError::NegativeOptFieldCount { count: nopt_node });
    }
    for _ in 0..nopt_node {
        r.skip_optional_field()?;
    }

    let mut tree = Tree::<T>::new();
    tree.num_nodes = num_nodes;
    tree.has_categorical_split = has_categorical_split;
    tree.node_type = TreeBuf::from_owned(node_type);
    tree.cleft = TreeBuf::from_owned(cleft);
    tree.cright = TreeBuf::from_owned(cright);
    tree.split_index = TreeBuf::from_owned(split_index);
    tree.default_left = TreeBuf::from_owned(default_left);
    tree.leaf_value = TreeBuf::from_owned(leaf_value);
    tree.threshold = TreeBuf::from_owned(threshold);
    tree.cmp = TreeBuf::from_owned(cmp);
    tree.category_list_right_child = TreeBuf::from_owned(category_list_right_child);
    tree.leaf_vector = TreeBuf::from_owned(leaf_vector);
    tree.leaf_vector_begin = TreeBuf::from_owned(leaf_vector_begin);
    tree.leaf_vector_end = TreeBuf::from_owned(leaf_vector_end);
    tree.category_list = TreeBuf::from_owned(category_list);
    tree.category_list_begin = TreeBuf::from_owned(category_list_begin);
    tree.category_list_end = TreeBuf::from_owned(category_list_end);
    tree.data_count = TreeBuf::from_owned(data_count);
    tree.data_count_present = TreeBuf::from_owned(data_count_present);
    tree.sum_hess = TreeBuf::from_owned(sum_hess);
    tree.sum_hess_present = TreeBuf::from_owned(sum_hess_present);
    tree.gain = TreeBuf::from_owned(gain);
    tree.gain_present = TreeBuf::from_owned(gain_present);
    tree.num_opt_field_per_tree = nopt_tree;
    tree.num_opt_field_per_node = nopt_node;
    Ok(tree)
}

// --- small element decoders (each fed an exact-width LE chunk by `Reader`) ---

fn decode_i32(b: &[u8]) -> Result<i32, SerializeError> {
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn decode_u32(b: &[u8]) -> Result<u32, SerializeError> {
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn decode_u64(b: &[u8]) -> Result<u64, SerializeError> {
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}
fn decode_f32(b: &[u8]) -> Result<f32, SerializeError> {
    // Raw IEEE-754 bits; NaN/inf round-trip bit-exact (Pitfall 4).
    Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
fn decode_f64(b: &[u8]) -> Result<f64, SerializeError> {
    Ok(f64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}
fn decode_bool(b: u8) -> bool {
    b != 0
}
fn decode_task_type(b: u8) -> Result<TaskType, SerializeError> {
    match b {
        0 => Ok(TaskType::kBinaryClf),
        1 => Ok(TaskType::kRegressor),
        2 => Ok(TaskType::kMultiClf),
        3 => Ok(TaskType::kLearningToRank),
        4 => Ok(TaskType::kIsolationForest),
        other => Err(SerializeError::InvalidEnumTag {
            kind: "TaskType",
            value: other as i64,
        }),
    }
}
fn decode_node_type(b: &[u8]) -> Result<TreeNodeType, SerializeError> {
    match b[0] as i8 {
        0 => Ok(TreeNodeType::kLeafNode),
        1 => Ok(TreeNodeType::kNumericalTestNode),
        2 => Ok(TreeNodeType::kCategoricalTestNode),
        other => Err(SerializeError::InvalidEnumTag {
            kind: "TreeNodeType",
            value: other as i64,
        }),
    }
}
fn decode_operator(b: &[u8]) -> Result<Operator, SerializeError> {
    match b[0] as i8 {
        0 => Ok(Operator::kNone),
        1 => Ok(Operator::kEQ),
        2 => Ok(Operator::kLT),
        3 => Ok(Operator::kLE),
        4 => Ok(Operator::kGT),
        5 => Ok(Operator::kGE),
        other => Err(SerializeError::InvalidEnumTag {
            kind: "Operator",
            value: other as i64,
        }),
    }
}

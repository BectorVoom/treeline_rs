//! Zero-copy PyBuffer frame representation of the v5 serialization (SER-02, D-06).
//!
//! Ports `SerializeToPyBuffer` + `GetPyBufferFrom{Scalar,Array,String}`
//! (`serializer.cc:458-463`, `serializer.h:30-110`). Upstream emits a
//! `Vec<PyBufferFrame{buf, format, itemsize, nitem}>` aliasing live model memory;
//! D-06 represents this idiomatically as a Rust enum over borrowed slices, with
//! the lifetime tied to `&Model` so the model must outlive the frames (D-05).
//!
//! Each array frame borrows DIRECTLY into a [`crate::tree_buf::TreeBuf`] column
//! via `as_slice()` — no copy (the `.as_ptr()` equality proof is in
//! `tests/serialize_pybuffer.rs`). A scalar is a 1-element slice (upstream
//! `GetPyBufferFromScalar` uses `nitem == 1`); the recomputed header scalars
//! borrow the staged private `Model` fields (Pattern 5). The frame ORDER is the
//! binary field order (D-01) — header table then per-tree table.

use crate::model::{Model, ModelVariant};
use crate::tree::Tree;

/// One PyBuffer frame: a typed borrowed view into live model memory.
///
/// The variant encodes the element type (and thus upstream's format string and
/// itemsize, reproduced at the Phase 8 boundary). Enum/bool columns are carried
/// as their 1-byte underlying representation (`U8`/`I8`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Frame<'a> {
    /// `=B` — unsigned byte column (bool / `TaskType` / `TypeInfo`, 1 B/elem).
    U8(&'a [u8]),
    /// `=b` — signed byte column (`TreeNodeType` / `Operator`, 1 B/elem).
    I8(&'a [i8]),
    /// `=L` — `u32` column.
    U32(&'a [u32]),
    /// `=l` — `i32` column.
    I32(&'a [i32]),
    /// `=Q` — `u64` column.
    U64(&'a [u64]),
    /// `=q` — `i64` column.
    I64(&'a [i64]),
    /// `=f` — `f32` column.
    F32(&'a [f32]),
    /// `=d` — `f64` column.
    F64(&'a [f64]),
    /// `=c` — a string's raw bytes (itemsize 1).
    Str(&'a str),
}

/// View a `&[T]` of plain-old-data as a `&[u8]` (1-byte-per-source-byte image).
///
/// Used only for the enum/bool 1-byte columns so they surface as `U8`/`I8`
/// frames matching upstream's underlying-type reinterpret.
fn as_u8_slice<T: Copy>(slice: &[T]) -> &[u8] {
    // SAFETY: read-only byte view of POD; lifetime tied to `slice`.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

/// View a `&[bool]`/`&[Enum]` 1-byte column as a signed-byte `&[i8]`.
fn as_i8_slice<T: Copy>(slice: &[T]) -> &[i8] {
    // SAFETY: read-only byte view of a 1-byte-repr POD column.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i8, std::mem::size_of_val(slice)) }
}

/// Produce the ordered zero-copy frame list for `m` (SER-02, D-06).
///
/// Stages the recomputed header scalars FIRST (so the version triple, `num_tree`,
/// and type tags have an `'a`-lived home to borrow — Pattern 5), then walks the
/// header and every tree in EXACT binary field order (D-01). The returned
/// `Vec<Frame<'_>>` borrows `&m`, so `m` must outlive the frames (D-05).
// The frame walk is a long sequence of ordered, branch-interspersed `push`es
// (45+ frames, F32/F64 variant split); a single `vec![...]` literal would be
// far less readable than the field-by-field push list, so the
// `vec_init_then_push` lint is intentionally allowed here.
#[allow(clippy::vec_init_then_push)]
pub fn serialize_to_pybuffer(m: &mut Model) -> Vec<Frame<'_>> {
    m.stage_serialization_fields();
    let m: &Model = m; // re-borrow immutably for the borrowing frame walk.
    let mut f: Vec<Frame<'_>> = Vec::new();

    // Header 1 — version triple + type tags + num_tree (1-element scalar frames).
    f.push(Frame::I32(std::slice::from_ref(m.major_ver_ref())));
    f.push(Frame::I32(std::slice::from_ref(m.minor_ver_ref())));
    f.push(Frame::I32(std::slice::from_ref(m.patch_ver_ref())));
    f.push(Frame::U8(as_u8_slice(std::slice::from_ref(
        m.threshold_type_ref(),
    ))));
    f.push(Frame::U8(as_u8_slice(std::slice::from_ref(
        m.leaf_output_type_ref(),
    ))));
    f.push(Frame::U64(std::slice::from_ref(m.num_tree_ref())));

    // Header 2.
    f.push(Frame::I32(std::slice::from_ref(&m.num_feature)));
    f.push(Frame::U8(as_u8_slice(std::slice::from_ref(&m.task_type))));
    f.push(Frame::U8(as_u8_slice(std::slice::from_ref(
        &m.average_tree_output,
    ))));
    f.push(Frame::I32(std::slice::from_ref(&m.num_target)));
    f.push(Frame::I32(&m.num_class));
    f.push(Frame::I32(&m.leaf_vector_shape));
    f.push(Frame::I32(&m.target_id));
    f.push(Frame::I32(&m.class_id));
    f.push(Frame::Str(&m.postprocessor));
    f.push(Frame::F32(std::slice::from_ref(&m.sigmoid_alpha)));
    f.push(Frame::F32(std::slice::from_ref(&m.ratio_c)));
    f.push(Frame::F64(&m.base_scores));
    f.push(Frame::Str(&m.attributes));
    f.push(Frame::I32(std::slice::from_ref(
        m.num_opt_field_per_model_ref(),
    )));

    // Per-tree frames in num_tree order.
    match &m.variant {
        ModelVariant::F32(p) => {
            for tree in &p.trees {
                push_tree_frames(&mut f, tree, Frame::F32);
            }
        }
        ModelVariant::F64(p) => {
            for tree in &p.trees {
                push_tree_frames(&mut f, tree, Frame::F64);
            }
        }
    }
    f
}

/// Push one tree's 25 frames in EXACT binary field order.
///
/// `leaf_frame` wraps a `&[T]` (the threshold/leaf element type) into the right
/// float variant, so the single walk serves both the `<f32>` and `<f64>` trees.
fn push_tree_frames<'a, T: Copy>(
    f: &mut Vec<Frame<'a>>,
    t: &'a Tree<T>,
    leaf_frame: impl Fn(&'a [T]) -> Frame<'a>,
) {
    f.push(Frame::I32(std::slice::from_ref(&t.num_nodes)));
    f.push(Frame::U8(as_u8_slice(std::slice::from_ref(
        &t.has_categorical_split,
    ))));
    f.push(Frame::I8(as_i8_slice(t.node_type.as_slice())));
    f.push(Frame::I32(t.cleft.as_slice()));
    f.push(Frame::I32(t.cright.as_slice()));
    f.push(Frame::I32(t.split_index.as_slice()));
    f.push(Frame::U8(as_u8_slice(t.default_left.as_slice())));
    f.push(leaf_frame(t.leaf_value.as_slice()));
    f.push(leaf_frame(t.threshold.as_slice()));
    f.push(Frame::I8(as_i8_slice(t.cmp.as_slice())));
    f.push(Frame::U8(as_u8_slice(
        t.category_list_right_child.as_slice(),
    )));
    f.push(leaf_frame(t.leaf_vector.as_slice()));
    f.push(Frame::U64(t.leaf_vector_begin.as_slice()));
    f.push(Frame::U64(t.leaf_vector_end.as_slice()));
    f.push(Frame::U32(t.category_list.as_slice()));
    f.push(Frame::U64(t.category_list_begin.as_slice()));
    f.push(Frame::U64(t.category_list_end.as_slice()));
    f.push(Frame::U64(t.data_count.as_slice()));
    f.push(Frame::U8(as_u8_slice(t.data_count_present.as_slice())));
    f.push(Frame::F64(t.sum_hess.as_slice()));
    f.push(Frame::U8(as_u8_slice(t.sum_hess_present.as_slice())));
    f.push(Frame::F64(t.gain.as_slice()));
    f.push(Frame::U8(as_u8_slice(t.gain_present.as_slice())));
    f.push(Frame::I32(std::slice::from_ref(&t.num_opt_field_per_tree)));
    f.push(Frame::I32(std::slice::from_ref(&t.num_opt_field_per_node)));
}

//! Per-column ragged-SoA host→device upload (Wave 2 / plan 06-03).
//!
//! For each `Tree<T>` column (`tree.rs:18-52`) this concatenates the column
//! across EVERY tree in the forest into one host `Vec`, byte-casts it via
//! [`bytemuck::cast_slice`] (a checked size/alignment cast, never a hand-rolled
//! raw-pointer reinterpret, T-06-07), and uploads it as ONE device handle for the
//! whole forest via `client.create_from_slice` — exactly one handle for each
//! COLUMN. The SC3/GPU-05 anti-pattern this avoids is allocating a separate
//! handle for every tree (handle explosion). A
//! parallel `tree_node_offset` prefix sum over `num_nodes` lets a kernel address
//! tree `t`'s node `n` at `concat[tree_node_offset[t] + n]`; a
//! `tree_leafvec_offset` prefix sum does the same for the per-tree leaf-vector
//! value column (multiclass broadcast).
//!
//! Column type discipline (RESEARCH Pitfall 4):
//! - `cleft`/`cright`/`split_index` upload as `i32` (the native column type) via
//!   [`TreeBuf::as_bytes`] — zero-copy, the column is already the right numeric
//!   type.
//! - `threshold`/`leaf_value`/`leaf_vector` upload as the element width `F` via
//!   `as_bytes` — zero-copy.
//! - `leaf_vector_begin`/`leaf_vector_end` narrow from `u64` to `u32` at
//!   materialization (mirroring the kernel's u32-index discipline) — a small
//!   materialized `Vec<u32>` column.
//! - `default_left` (bool) materializes to a `Vec<u32>` of 0/1 — `bool` is not a
//!   cubecl `Array` element.
//! - `node_type` (enum) materializes to an `i32` discriminant column so a kernel
//!   can detect `kCategoricalTestNode` and route to fallback.
//!
//! Host-side validation precedes EVERY `client.create_from_slice`: a model whose
//! `split_index` exceeds `num_feature`, or whose `data` buffer is too small for
//! the declared `num_row × num_feature` shape, returns a typed [`CubeclError`]
//! BEFORE any device op — never an out-of-bounds device write (T-06-06,
//! mirroring `treelite_gtil::predict`'s up-front validation, lib.rs:902-926).
//
// cubecl 0.10.0 API: upload=ComputeClient::create_from_slice(&[u8]) -> Handle (zero-copy SoA, cubecl-runtime client.rs:265; create(Bytes) at :282 is the owned variant). The Wave-1 spike retired this against the read-back via client.read_one_unchecked + bytemuck::cast_slice.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use treelite_core::{ModelPreset, Tree};

use crate::error::CubeclError;

/// The per-column device handles + the two prefix-sum offset indices for one
/// uploaded forest.
///
/// Every field is ONE handle for the WHOLE forest (the concatenated ragged-SoA
/// column), never one allocated separately for each tree (SC3). The offset
/// vectors
/// are kept host-side (they are tiny and the kernel reads them as their own
/// uploaded `Array<u32>` — uploaded by the caller alongside the columns, or via
/// [`UploadedForest::node_off`] / [`UploadedForest::leafvec_off`]).
pub struct UploadedForest<R: Runtime> {
    /// `cleft` concatenated across the forest (one `i32` handle).
    pub cleft: cubecl::server::Handle,
    /// `cright` concatenated across the forest (one `i32` handle).
    pub cright: cubecl::server::Handle,
    /// `split_index` concatenated across the forest (one `i32` handle).
    pub split_index: cubecl::server::Handle,
    /// `threshold` concatenated across the forest (one `F`-width handle).
    pub threshold: cubecl::server::Handle,
    /// `leaf_value` concatenated across the forest (one `F`-width handle).
    pub leaf_value: cubecl::server::Handle,
    /// `leaf_vector` concatenated across the forest (one `F`-width handle).
    pub leaf_vector: cubecl::server::Handle,
    /// `leaf_vector_begin` narrowed to `u32`, concatenated (one handle).
    pub leaf_vector_begin: cubecl::server::Handle,
    /// `leaf_vector_end` narrowed to `u32`, concatenated (one handle).
    pub leaf_vector_end: cubecl::server::Handle,
    /// `default_left` materialized to `u32` 0/1, concatenated (one handle).
    pub default_left: cubecl::server::Handle,
    /// `node_type` materialized to its `i32` discriminant, concatenated (one
    /// handle).
    pub node_type: cubecl::server::Handle,

    /// Element count of each concatenated NODE column (`sum(num_nodes)`); the
    /// length to pass to `ArrayArg::from_raw_parts` for the node columns.
    pub num_nodes_total: usize,
    /// Element count of the concatenated leaf-vector value column
    /// (`sum(leaf_vector.len())`).
    pub num_leafvec_total: usize,

    /// Prefix sum over per-tree `num_nodes`: `[0, n0, n0+n1, …]` (length
    /// `num_tree + 1`). `tree_node_offset[t]` is the base of tree `t`'s nodes in
    /// every node column; `concat[tree_node_offset[t] + n]` addresses tree `t`'s
    /// node `n`.
    pub tree_node_offset: Vec<u32>,
    /// Prefix sum over per-tree leaf-vector lengths (length `num_tree + 1`).
    pub tree_leafvec_offset: Vec<u32>,

    _runtime: core::marker::PhantomData<R>,
}

impl<R: Runtime> UploadedForest<R> {
    /// Upload `tree_node_offset` as its own `u32` device handle (the kernel reads
    /// it as `Array<u32>`).
    pub fn node_off(&self, client: &ComputeClient<R>) -> cubecl::server::Handle {
        client.create_from_slice(bytemuck::cast_slice(&self.tree_node_offset))
    }

    /// Upload `tree_leafvec_offset` as its own `u32` device handle.
    pub fn leafvec_off(&self, client: &ComputeClient<R>) -> cubecl::server::Handle {
        client.create_from_slice(bytemuck::cast_slice(&self.tree_leafvec_offset))
    }
}

/// Build the host-side concatenated ragged-SoA columns + the two prefix-sum
/// offset indices for a forest, WITHOUT touching the device.
///
/// Returned in upload order so [`upload_forest`] (and tests) can round-trip each
/// column independently. `F` is the element width (`f32`/`f64`).
#[allow(clippy::type_complexity)]
pub fn concat_columns<F: Copy + bytemuck::Pod>(
    preset: &ModelPreset<F>,
) -> HostColumns<F> {
    let trees: &[Tree<F>] = &preset.trees;
    let num_tree = trees.len();

    let mut cleft: Vec<i32> = Vec::new();
    let mut cright: Vec<i32> = Vec::new();
    let mut split_index: Vec<i32> = Vec::new();
    let mut threshold: Vec<F> = Vec::new();
    let mut leaf_value: Vec<F> = Vec::new();
    let mut leaf_vector: Vec<F> = Vec::new();
    let mut leaf_vector_begin: Vec<u32> = Vec::new();
    let mut leaf_vector_end: Vec<u32> = Vec::new();
    let mut default_left: Vec<u32> = Vec::new();
    let mut node_type: Vec<i32> = Vec::new();

    // Prefix sums: length num_tree + 1, both starting at 0.
    let mut tree_node_offset: Vec<u32> = Vec::with_capacity(num_tree + 1);
    let mut tree_leafvec_offset: Vec<u32> = Vec::with_capacity(num_tree + 1);
    let mut node_acc: u32 = 0;
    let mut leafvec_acc: u32 = 0;

    for t in trees {
        tree_node_offset.push(node_acc);
        tree_leafvec_offset.push(leafvec_acc);

        cleft.extend_from_slice(t.cleft.as_slice());
        cright.extend_from_slice(t.cright.as_slice());
        split_index.extend_from_slice(t.split_index.as_slice());
        threshold.extend_from_slice(t.threshold.as_slice());
        leaf_value.extend_from_slice(t.leaf_value.as_slice());
        leaf_vector.extend_from_slice(t.leaf_vector.as_slice());
        // u64 CSR offsets narrow to u32 at materialization (kernel u32-index
        // discipline). leaf_vector_begin/end are per-node offsets INTO this
        // tree's leaf_vector; they ride as u32 columns of node length.
        leaf_vector_begin.extend(t.leaf_vector_begin.as_slice().iter().map(|&x| x as u32));
        leaf_vector_end.extend(t.leaf_vector_end.as_slice().iter().map(|&x| x as u32));
        // bool -> u32 0/1 (Pitfall 4): bool is not a cubecl Array element.
        default_left.extend(t.default_left.as_slice().iter().map(|&b| b as u32));
        // enum -> i32 discriminant (Pitfall 4).
        node_type.extend(t.node_type.as_slice().iter().map(|&n| n as i32));

        node_acc += t.cleft.as_slice().len() as u32;
        leafvec_acc += t.leaf_vector.as_slice().len() as u32;
    }
    tree_node_offset.push(node_acc);
    tree_leafvec_offset.push(leafvec_acc);

    HostColumns {
        cleft,
        cright,
        split_index,
        threshold,
        leaf_value,
        leaf_vector,
        leaf_vector_begin,
        leaf_vector_end,
        default_left,
        node_type,
        tree_node_offset,
        tree_leafvec_offset,
    }
}

/// The host-side concatenated ragged-SoA columns + offset indices (pre-upload).
///
/// Exposed so tests can assert the host concatenation before/after the device
/// round-trip, and so the host launcher can re-use a column without re-walking
/// the trees.
pub struct HostColumns<F: Copy> {
    /// Concatenated `cleft` (`i32`).
    pub cleft: Vec<i32>,
    /// Concatenated `cright` (`i32`).
    pub cright: Vec<i32>,
    /// Concatenated `split_index` (`i32`).
    pub split_index: Vec<i32>,
    /// Concatenated `threshold` (`F`).
    pub threshold: Vec<F>,
    /// Concatenated `leaf_value` (`F`).
    pub leaf_value: Vec<F>,
    /// Concatenated `leaf_vector` (`F`).
    pub leaf_vector: Vec<F>,
    /// Concatenated `leaf_vector_begin` narrowed to `u32`.
    pub leaf_vector_begin: Vec<u32>,
    /// Concatenated `leaf_vector_end` narrowed to `u32`.
    pub leaf_vector_end: Vec<u32>,
    /// Concatenated `default_left` materialized as `u32` 0/1.
    pub default_left: Vec<u32>,
    /// Concatenated `node_type` materialized as `i32` discriminant.
    pub node_type: Vec<i32>,
    /// Prefix sum over per-tree `num_nodes` (length `num_tree + 1`).
    pub tree_node_offset: Vec<u32>,
    /// Prefix sum over per-tree leaf-vector lengths (length `num_tree + 1`).
    pub tree_leafvec_offset: Vec<u32>,
}

/// Validate the model shape + input buffer up front, mirroring
/// [`treelite_gtil::predict`]'s checks (lib.rs:902-926), BEFORE any device op.
///
/// - A negative `num_feature` casts to a huge `usize`; treat it as a (0-sized)
///   impossible shape so the buffer-length check rejects it
///   ([`CubeclError::InvalidInputShape`]).
/// - The `data` buffer must hold at least `num_row × num_feature` elements
///   (saturating, so an overflow pins to `usize::MAX` and rejects).
/// - Every node's `split_index` (on a non-leaf node, `cleft != -1`) must be a
///   valid feature index `0 <= split_index < num_feature`
///   ([`CubeclError::FeatureIndexOutOfBounds`]).
///
/// This runs against the HOST columns — there is no device write until it
/// returns `Ok` (T-06-06: no OOB device write on a malformed model).
pub fn validate_shape<F: Copy>(
    num_feature: i32,
    num_row: usize,
    data_len: usize,
    cols: &HostColumns<F>,
) -> Result<(), CubeclError> {
    // num_feature < 0 -> impossible shape (mirror predict's WR-02 guard).
    if num_feature < 0 {
        return Err(CubeclError::InvalidInputShape {
            num_row,
            num_feature: 0,
            required: usize::MAX,
            got: data_len,
        });
    }
    let nf = num_feature as usize;
    let required = num_row.saturating_mul(nf);
    if data_len < required {
        return Err(CubeclError::InvalidInputShape {
            num_row,
            num_feature: nf,
            required,
            got: data_len,
        });
    }
    // Per-node split_index bounds on internal nodes (T-06-06 / T-03-01). A leaf
    // node has cleft == -1 and split_index == -1 (the sentinel); only an
    // internal node's split_index addresses a feature.
    for (node, (&cl, &fi)) in cols.cleft.iter().zip(cols.split_index.iter()).enumerate() {
        if cl != -1 && (fi < 0 || fi >= num_feature) {
            return Err(CubeclError::FeatureIndexOutOfBounds {
                node,
                feature: fi,
                num_feature,
            });
        }
    }
    Ok(())
}

/// Concatenate the forest's columns, VALIDATE the shape, then upload each column
/// as ONE device handle (no per-tree explosion). Validation precedes every
/// `client.create_from_slice` (T-06-06).
///
/// `num_row`/`data_len` describe the input matrix the kernel will read; they are
/// validated here so a malformed model never reaches a device launch.
pub fn upload_forest<R: Runtime, F: Copy + bytemuck::Pod>(
    client: &ComputeClient<R>,
    preset: &ModelPreset<F>,
    num_feature: i32,
    num_row: usize,
    data_len: usize,
) -> Result<UploadedForest<R>, CubeclError> {
    let cols = concat_columns(preset);
    // VALIDATE BEFORE ANY DEVICE OP (no OOB device write on a malformed model).
    validate_shape(num_feature, num_row, data_len, &cols)?;

    let num_nodes_total = cols.cleft.len();
    let num_leafvec_total = cols.leaf_vector.len();

    Ok(UploadedForest {
        cleft: client.create_from_slice(bytemuck::cast_slice(&cols.cleft)),
        cright: client.create_from_slice(bytemuck::cast_slice(&cols.cright)),
        split_index: client.create_from_slice(bytemuck::cast_slice(&cols.split_index)),
        threshold: client.create_from_slice(bytemuck::cast_slice(&cols.threshold)),
        leaf_value: client.create_from_slice(bytemuck::cast_slice(&cols.leaf_value)),
        leaf_vector: client.create_from_slice(bytemuck::cast_slice(&cols.leaf_vector)),
        leaf_vector_begin: client.create_from_slice(bytemuck::cast_slice(&cols.leaf_vector_begin)),
        leaf_vector_end: client.create_from_slice(bytemuck::cast_slice(&cols.leaf_vector_end)),
        default_left: client.create_from_slice(bytemuck::cast_slice(&cols.default_left)),
        node_type: client.create_from_slice(bytemuck::cast_slice(&cols.node_type)),
        num_nodes_total,
        num_leafvec_total,
        tree_node_offset: cols.tree_node_offset,
        tree_leafvec_offset: cols.tree_leafvec_offset,
        _runtime: core::marker::PhantomData,
    })
}

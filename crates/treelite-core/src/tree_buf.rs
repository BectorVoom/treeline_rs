//! `TreeBuf<T>` — the struct-of-arrays storage primitive (CORE-03).
//!
//! Ports `treelite-mainline/include/treelite/contiguous_array.h`. Upstream is
//! a manual `T* buffer_` + `bool owned_buffer_` with `UseForeignBuffer` for
//! zero-copy aliasing, a deleted copy ctor (move-only, explicit `Clone()`),
//! and `static_assert(std::is_pod<T>)`.
//!
//! Rust port: a two-mode enum. `Owned` holds a `Vec<T>`; `Borrowed` aliases
//! external memory via a raw pointer + length. The borrowed mode's real
//! consumer is the Phase 8 Python buffer protocol; Phase 1 only needs both
//! modes to exist and round-trip. `T: Copy` mirrors the upstream POD bound
//! (the `bytemuck::Pod` seam is deferred to Phase 9 — do not pull it in here).

use std::ops::Index;

/// Owned-or-borrowed flat column of `T`, indexed by node id.
///
/// Move-only by intent: no casual `#[derive(Clone)]` (mirrors the deleted
/// upstream copy ctor). Use [`TreeBuf::deep_copy`] for an explicit deep copy.
pub enum TreeBuf<T: Copy> {
    /// Owns its backing storage.
    Owned(Vec<T>),
    /// Aliases external memory (zero-copy). The pointer must remain valid for
    /// `len` elements for the lifetime of this `TreeBuf`.
    Borrowed {
        /// Pointer to the first element of the borrowed region.
        ptr: *const T,
        /// Number of elements in the borrowed region.
        len: usize,
    },
}

// SAFETY: `TreeBuf<T>` is `!Sync` only because the `Borrowed { ptr: *const T }`
// variant holds a raw const pointer (raw pointers are `!Sync` by default).
// Sharing `&TreeBuf<T>` across threads is sound on the predict path, the same
// argument that justifies `unsafe impl Sync for Model` (Phase-10 PAR-03,
// model.rs):
//   1. predict only ever takes `&Tree<T>` / `&TreeBuf<T>` (SHARED, never `&mut`)
//      and reads the buffer via `as_slice()` — no field is mutated;
//   2. the `Borrowed` pointer aliases external memory whose backing (by the
//      `from_borrowed` SAFETY contract) outlives the `TreeBuf` and is not
//      mutated while borrowed → concurrent READS are data-race-free;
//   3. `TreeBuf` exposes no interior mutability — there is no `&self` method
//      that writes through the pointer.
// This mirrors upstream Treelite sharing the forest `const&` across OpenMP
// threads (`predict.cc`). Only `Sync` is asserted — NOT `Send` (the model is
// shared by reference across rayon workers, never MOVED to another thread; A4).
//
// The bound is `T: Copy + Sync`, NOT merely `T: Copy` (WR-02). `T: Copy` alone
// does not forbid interior mutability: `Cell<f32>` is `Copy` but `!Sync`, and a
// `TreeBuf<Cell<f32>>` auto-claiming `Sync` would make the concurrent `Owned`
// reads a data race. Requiring `T: Sync` expresses the actual property — the
// element type must itself be shareable across threads. Every concrete `T` used
// in practice (`f32`, `f64`, `i32`, `u32`, `u64`, `bool`, `Operator`,
// `TreeNodeType`) satisfies `Sync`, so this is a zero-impact tightening.
unsafe impl<T: Copy + Sync> Sync for TreeBuf<T> {}

impl<T: Copy> TreeBuf<T> {
    /// Construct an owned buffer from a `Vec<T>`.
    pub fn from_owned(data: Vec<T>) -> Self {
        TreeBuf::Owned(data)
    }

    /// Construct an empty owned buffer.
    pub fn empty() -> Self {
        TreeBuf::Owned(Vec::new())
    }

    /// Construct a zero-copy borrowed buffer aliasing `slice`.
    ///
    /// # Safety
    /// The caller guarantees `slice`'s backing memory outlives this `TreeBuf`
    /// and is not mutated while it is borrowed. Mirrors upstream
    /// `UseForeignBuffer` (`contiguous_array.h:58-62`).
    pub unsafe fn from_borrowed(slice: &[T]) -> Self {
        TreeBuf::Borrowed {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    /// View the buffer as a slice.
    pub fn as_slice(&self) -> &[T] {
        match self {
            TreeBuf::Owned(v) => v.as_slice(),
            // SAFETY: `from_borrowed` requires the caller to guarantee the
            // pointer is valid for `len` elements for this buffer's lifetime.
            TreeBuf::Borrowed { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            TreeBuf::Owned(v) => v.len(),
            TreeBuf::Borrowed { len, .. } => *len,
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Explicit deep copy into an `Owned` buffer (mirrors upstream `Clone()`).
    pub fn deep_copy(&self) -> Self {
        TreeBuf::Owned(self.as_slice().to_vec())
    }
}

/// Zero-copy byte-view accessor, gated on the NARROWER `T: bytemuck::Pod`
/// bound (GPU-05 / SC3).
///
/// This is an ADDITIVE second `impl` block: it does NOT widen the enum's
/// primary `T: Copy` bound (the broad `bytemuck::Pod` seam on the whole API is
/// Phase 9 — module note at the top of this file). It exposes the bytes of a
/// numeric POD column so the Phase 6 cubecl SoA upload can feed
/// `bytemuck::cast_slice` → `cubecl::bytes::Bytes` → the device handle without
/// a copy. Only numeric POD columns qualify; bool/enum columns are excluded
/// (the host materializes them to numeric before upload, T-06-02).
impl<T: Copy + bytemuck::Pod> TreeBuf<T> {
    /// View the buffer's contents as raw bytes (zero-copy).
    ///
    /// Uses [`bytemuck::cast_slice`], which validates size/alignment and is
    /// never a hand-rolled `transmute` (T-06-02). The returned slice borrows
    /// `self` and has length `self.len() * size_of::<T>()`.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }
}

impl<T: Copy> Index<usize> for TreeBuf<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.as_slice()[idx]
    }
}

impl<T: Copy> Default for TreeBuf<T> {
    fn default() -> Self {
        TreeBuf::empty()
    }
}

#[cfg(test)]
mod tree_buf_as_bytes_tests {
    use super::TreeBuf;

    #[test]
    fn as_bytes_f32_roundtrip() {
        let buf = TreeBuf::<f32>::from_owned(vec![1.0_f32, 2.0]);
        // 2 elements × 4 bytes each.
        assert_eq!(buf.as_bytes().len(), 8);
        // Round-trip exact: bytes → f32 reproduces the source values.
        let back: &[f32] = bytemuck::cast_slice(buf.as_bytes());
        assert_eq!(back, &[1.0_f32, 2.0]);
    }

    #[test]
    fn as_bytes_i32_len() {
        let buf = TreeBuf::<i32>::from_owned(vec![5_i32]);
        assert_eq!(buf.as_bytes().len(), 4);
        let back: &[i32] = bytemuck::cast_slice(buf.as_bytes());
        assert_eq!(back, &[5_i32]);
    }

    #[test]
    fn as_bytes_empty() {
        let buf = TreeBuf::<f64>::empty();
        assert_eq!(buf.as_bytes().len(), 0);
    }
}

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

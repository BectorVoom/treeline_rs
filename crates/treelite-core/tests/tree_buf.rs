//! `TreeBuf<T>` owned + borrowed round-trip tests (CORE-03).

use treelite_core::TreeBuf;

#[test]
fn owned_len_and_index() {
    let buf = TreeBuf::from_owned(vec![1i32, 2, 3]);
    assert_eq!(buf.len(), 3);
    assert!(!buf.is_empty());
    assert_eq!(buf[1], 2);
    assert_eq!(buf.as_slice(), &[1, 2, 3]);
}

#[test]
fn empty_buffer() {
    let buf: TreeBuf<f32> = TreeBuf::empty();
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn borrowed_round_trip_zero_copy() {
    let backing = vec![10.0f64, 20.0, 30.0, 40.0];
    // SAFETY: `backing` outlives `buf` and is not mutated while borrowed.
    let buf = unsafe { TreeBuf::from_borrowed(&backing) };
    assert_eq!(buf.len(), backing.len());
    assert_eq!(buf.as_slice(), backing.as_slice());
    assert_eq!(buf[2], 30.0);
    // Zero-copy: the borrowed slice points at the same memory.
    assert_eq!(buf.as_slice().as_ptr(), backing.as_ptr());
}

#[test]
fn deep_copy_produces_owned() {
    let backing = vec![1u32, 2, 3];
    // SAFETY: `backing` outlives `borrowed`.
    let borrowed = unsafe { TreeBuf::from_borrowed(&backing) };
    let owned = borrowed.deep_copy();
    assert_eq!(owned.as_slice(), backing.as_slice());
    // The deep copy must NOT alias the original memory.
    assert_ne!(owned.as_slice().as_ptr(), backing.as_ptr());
}

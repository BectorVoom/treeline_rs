//! The buffer (byte) backend for the v5 serializer (SER-01).
//!
//! Ports `BufferSerializerMixIn` (`serializer_mixins.h`): every scalar is raw
//! little-endian bytes (`to_le_bytes`, no prefix), every array is a `u64` count
//! then the payload (count-only when empty — Pitfall 3), every string is a
//! `u64` length then the raw bytes. NaN/inf floats are copied as raw IEEE-754
//! bits and are NEVER normalized (Pitfall 4); bool columns are 1 byte/element
//! as one byte per element, never bit-compressed (Pitfall 5). On the
//! little-endian x86-64 manifest host
//! these bytes equal the upstream `memcpy` image (RESEARCH A2), validated
//! byte-for-byte against `fixtures/golden_v5.bin` (D-02).

use crate::model::Model;
use crate::serialize::error::SerializeError;
use crate::serialize::{SerializerBackend, serialize_header, serialize_trees};

/// An absolute sanity cap on any single array/string element count, applied
/// BEFORE the per-element bounds check as a cheap first gate.
///
/// No legitimate v5 column carries `2^32` elements; capping here means even a
/// `count * elem_size` product that would overflow `usize` on a 32-bit target
/// is rejected up front. The authoritative check remains
/// `count * size_of::<T>() <= remaining` (see [`Reader::check_count`]).
const MAX_ELEM_COUNT: u64 = u32::MAX as u64;

/// A [`SerializerBackend`] that appends framed bytes to an owned `Vec<u8>`.
///
/// Mirrors upstream `BufferSerializerMixIn`. Growing a `Vec` is sufficient for
/// Phase 2 (the upstream size-calculator pre-pass is an optimization, not a
/// correctness requirement — RESEARCH Pattern 2); the byte output is identical.
#[derive(Default)]
pub struct BufferBackend {
    /// The accumulated framed byte stream.
    pub buf: Vec<u8>,
}

impl BufferBackend {
    /// A fresh empty backend.
    pub fn new() -> Self {
        BufferBackend { buf: Vec::new() }
    }

    /// Consume the backend, yielding the accumulated bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

impl SerializerBackend for BufferBackend {
    fn scalar_le(&mut self, bytes: &[u8]) {
        // Raw LE bytes, no length prefix (serializer.h:163).
        self.buf.extend_from_slice(bytes);
    }

    fn array_u64_prefixed(&mut self, count: usize, payload: &[u8]) {
        // u64 element count, then payload; count-only when empty (serializer.h:181).
        self.buf.extend_from_slice(&(count as u64).to_le_bytes());
        if count == 0 {
            return; // empty array: only the 8-byte zero (Pitfall 3).
        }
        self.buf.extend_from_slice(payload);
    }

    fn string(&mut self, s: &str) {
        // u64 byte length, then raw UTF-8 bytes; length-only when empty
        // (serializer.h:201). No NUL terminator.
        let bytes = s.as_bytes();
        self.buf
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        if bytes.is_empty() {
            return; // empty string: only the 8-byte zero (Pitfall 3).
        }
        self.buf.extend_from_slice(bytes);
    }
}

/// Serialize `m` to a v5 byte stream (the `SerializeToBuffer` entry point).
///
/// Stages the recomputed header bookkeeping (version triple, `num_tree`, type
/// tags) via [`Model::stage_serialization_fields`] FIRST so the header walk can
/// borrow them (RESEARCH Pattern 5), then walks the 20 header fields and every
/// tree's 25 fields in EXACT `serializer.cc` order. The output is byte-for-byte
/// identical to the upstream `golden_v5.bin` (D-01/D-02).
pub fn serialize_to_buffer(m: &mut Model) -> Vec<u8> {
    m.stage_serialization_fields();
    let mut backend = BufferBackend::new();
    serialize_header(m, &mut backend);
    serialize_trees(m, &mut backend);
    backend.into_bytes()
}

/// A bounds-checked forward cursor over an untrusted v5 byte slice (ASVS V5).
///
/// Every read verifies `offset + n <= buf.len()` and returns a typed
/// [`SerializeError::TruncatedStream`] on a short read — safe Rust slicing would
/// panic, so we convert that into a `Result` (RESEARCH § Security T-02-S02).
/// Array/string counts are bound against the remaining buffer BEFORE any
/// allocation, so a hostile `u64` prefix can never drive a huge
/// `Vec::with_capacity` (T-02-S01). The cursor performs NO `unsafe` slicing.
pub struct Reader<'a> {
    buf: &'a [u8],
    off: usize,
}

impl<'a> Reader<'a> {
    /// Wrap `buf` with the cursor at offset 0.
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, off: 0 }
    }

    /// Current cursor offset.
    pub fn offset(&self) -> usize {
        self.off
    }

    /// Total buffer length.
    pub fn total(&self) -> usize {
        self.buf.len()
    }

    /// Bytes remaining from the cursor.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.off
    }

    /// Read exactly `n` raw bytes, advancing the cursor; short read → typed err.
    fn take(&mut self, n: usize) -> Result<&'a [u8], SerializeError> {
        let end = self
            .off
            .checked_add(n)
            .ok_or(SerializeError::TruncatedStream {
                offset: self.off,
                needed: n,
                available: self.remaining(),
            })?;
        if end > self.buf.len() {
            return Err(SerializeError::TruncatedStream {
                offset: self.off,
                needed: n,
                available: self.remaining(),
            });
        }
        let out = &self.buf[self.off..end];
        self.off = end;
        Ok(out)
    }

    /// Read a `u8` (1 byte).
    pub fn u8(&mut self) -> Result<u8, SerializeError> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `i32` (4 bytes).
    pub fn i32(&mut self) -> Result<i32, SerializeError> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a little-endian `f32` (4 bytes) — raw IEEE-754 bits (Pitfall 4).
    pub fn f32(&mut self) -> Result<f32, SerializeError> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a little-endian `u64` (8 bytes).
    pub fn u64(&mut self) -> Result<u64, SerializeError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Validate an array/string `count` against the remaining buffer BEFORE
    /// allocating: reject if `count > MAX_ELEM_COUNT` or
    /// `count * elem_size > remaining` (T-02-S01). Never trusts the prefix.
    fn check_count(&self, count: u64, elem_size: usize) -> Result<usize, SerializeError> {
        let needed = (count as u128) * (elem_size as u128);
        if count > MAX_ELEM_COUNT || needed > self.remaining() as u128 {
            return Err(SerializeError::CountExceedsBuffer {
                count,
                elem_size,
                needed,
                remaining: self.remaining(),
            });
        }
        Ok(count as usize)
    }

    /// Read a `u64`-prefixed array of fixed-size POD elements, decoding each via
    /// `decode` over its `elem_size`-byte little-endian image. The count is
    /// bound-checked before any allocation; an empty array consumes only the
    /// 8-byte zero prefix (Pitfall 3).
    pub fn array<T>(
        &mut self,
        elem_size: usize,
        mut decode: impl FnMut(&[u8]) -> Result<T, SerializeError>,
    ) -> Result<Vec<T>, SerializeError> {
        let count = self.u64()?;
        let n = self.check_count(count, elem_size)?;
        // `n` is now provably <= remaining/elem_size, so this capacity is safe.
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let chunk = self.take(elem_size)?;
            out.push(decode(chunk)?);
        }
        Ok(out)
    }

    /// Read a `u64`-length-prefixed UTF-8 string (no NUL terminator). The length
    /// is bound-checked before allocating; invalid UTF-8 is reported as a typed
    /// error rather than a panic.
    pub fn string(&mut self) -> Result<String, SerializeError> {
        let len = self.u64()?;
        let n = self.check_count(len, 1)?;
        let bytes = self.take(n)?;
        // Lossless decode; a hostile non-UTF-8 blob becomes a typed error.
        String::from_utf8(bytes.to_vec()).map_err(|_| SerializeError::TruncatedStream {
            offset: self.off - n,
            needed: n,
            available: n,
        })
    }

    /// Skip one forward-version optional field without corrupting position:
    /// consume a name-string then `{elem_size: u64, nelem: u64, payload}`
    /// (`SkipOptionalFieldInStream`, serializer.h:211). The payload size is
    /// bound-checked against the remaining buffer (T-02-S04).
    pub fn skip_optional_field(&mut self) -> Result<(), SerializeError> {
        let _name = self.string()?;
        let elem_size = self.u64()?;
        let nelem = self.u64()?;
        let nbytes = (elem_size as u128) * (nelem as u128);
        if nbytes > self.remaining() as u128 {
            return Err(SerializeError::CountExceedsBuffer {
                count: nelem,
                elem_size: elem_size as usize,
                needed: nbytes,
                remaining: self.remaining(),
            });
        }
        self.take(nbytes as usize)?;
        Ok(())
    }
}

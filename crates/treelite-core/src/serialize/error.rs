//! Typed errors for the v5 (de)serializer (SER-01, D-03, ASVS V5).
//!
//! Every upstream fatal path in `treelite-mainline/src/serializer.cc`
//! (`TREELITE_CHECK` / `TREELITE_LOG(FATAL)`) and every place where untrusted
//! deserialize input could panic, over-read, or over-allocate is converted to a
//! returned `Err` here. The deserializer is the phase's primary untrusted-input
//! surface (RESEARCH § Security Domain) and MUST be panic-free.

use thiserror::Error;

/// Errors raised by the `treelite-core` serialize module.
///
/// The write path is infallible for the in-memory `Vec<u8>` backend, so every
/// variant here describes a *deserialize* failure mode driven by hostile or
/// malformed input.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SerializeError {
    /// The stream ended before a read of `needed` bytes could complete at
    /// `offset` (the blob is truncated). Upstream this is an `istream` short
    /// read; here it is a typed error rather than a panic or over-read
    /// (`serializer.h:157-159`; RESEARCH § Security T-02-S02).
    #[error(
        "truncated stream: needed {needed} bytes at offset {offset}, only {available} available"
    )]
    TruncatedStream {
        /// Byte offset at which the read was attempted.
        offset: usize,
        /// Number of bytes the read required.
        needed: usize,
        /// Number of bytes actually remaining from `offset`.
        available: usize,
    },

    /// An array/string `u64` count prefix demanded more bytes than remain in
    /// the buffer. We reject *before* allocating, so an attacker-controlled
    /// huge count can never drive a `Vec::with_capacity` OOM
    /// (`serializer.h:170-177`; RESEARCH § Security T-02-S01).
    #[error(
        "count prefix {count} (× {elem_size} B = {needed} B) exceeds {remaining} remaining bytes"
    )]
    CountExceedsBuffer {
        /// The element count read from the stream.
        count: u64,
        /// `size_of::<T>()` for the element type.
        elem_size: usize,
        /// `count * elem_size`, the payload byte count demanded.
        needed: u128,
        /// Bytes remaining in the buffer after the prefix.
        remaining: usize,
    },

    /// The header's `major_ver` is not `4` (the only supported v5 generation).
    /// D-03: we reject `major_ver == 3` (the legacy V3 path) and every other
    /// version with a typed error instead of mis-parsing
    /// (`serializer.cc:192-200`).
    #[error(
        "unsupported Treelite model version {major}.{minor}.{patch}: only major_ver == 4 (v5 wire format) is supported"
    )]
    UnsupportedVersion {
        /// The major version read from the stream.
        major: i32,
        /// The minor version read from the stream.
        minor: i32,
        /// The patch version read from the stream.
        patch: i32,
    },

    /// A type-tag byte (`threshold_type` / `leaf_output_type`) did not map to a
    /// supported numeric type, or the threshold/leaf types disagree (only the
    /// `<f32,f32>` and `<f64,f64>` presets exist upstream — `tree.h:81-86`).
    #[error("invalid or unsupported type tags: threshold={threshold}, leaf_output={leaf_output}")]
    InvalidTypeTag {
        /// Raw `threshold_type` tag byte.
        threshold: u8,
        /// Raw `leaf_output_type` tag byte.
        leaf_output: u8,
    },

    /// An enum tag byte (`TaskType` / `TreeNodeType` / `Operator`) read from the
    /// stream did not correspond to a known enumerator.
    #[error("invalid {kind} tag byte: {value}")]
    InvalidEnumTag {
        /// The enum kind whose tag failed to map (e.g. `"TaskType"`).
        kind: &'static str,
        /// The offending raw byte value.
        value: i64,
    },

    /// A `num_opt_field_*` extension count was negative (we always write `0`;
    /// a negative count from a hostile blob is rejected so the skip loop can
    /// never be driven to amplify work — RESEARCH § Security T-02-S04).
    #[error("negative optional-field count {count} in extension slot")]
    NegativeOptFieldCount {
        /// The offending count.
        count: i32,
    },

    /// Trailing bytes remained after the full header + per-tree walk consumed
    /// the model — the blob is longer than a well-formed v5 stream.
    #[error("{trailing} trailing bytes after a complete model (offset {offset} of {total})")]
    TrailingBytes {
        /// Offset reached after parsing the model.
        offset: usize,
        /// Total buffer length.
        total: usize,
        /// `total - offset`, the unconsumed remainder.
        trailing: usize,
    },
}

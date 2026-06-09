//! Typed errors for the core crate (ERR-01).
//!
//! Every upstream fatal path (`TREELITE_LOG(FATAL)` / `TREELITE_CHECK`)
//! is converted to a returned `Err` here rather than a panic. See the
//! enum `FromString` paths in `treelite-mainline/src/enum/*.cc`.

use thiserror::Error;

/// Errors raised by the `treelite-core` crate.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// An unknown string was passed to an enum's `from_str`. Upstream this is
    /// a `TREELITE_LOG(FATAL)`; here it is a typed, recoverable error.
    #[error("unknown {kind} string: {value:?}")]
    UnknownEnumString {
        /// The enum kind that failed to parse (e.g. `"TaskType"`).
        kind: &'static str,
        /// The offending input string.
        value: String,
    },
}

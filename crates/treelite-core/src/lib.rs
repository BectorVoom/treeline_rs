//! `treelite-core` — the foundation layer of treelite-rs.
//!
//! Ports the in-memory tree-ensemble representation from upstream Treelite
//! v4.7.0 (`treelite-mainline/`): the four shared enums, the `TreeBuf<T>`
//! struct-of-arrays storage primitive, the `Tree<T>` node columns, and the
//! two-variant `Model` with full header metadata.

pub mod enums;
pub mod error;

pub use enums::{DType, Operator, TaskType, TreeNodeType};
pub use error::CoreError;

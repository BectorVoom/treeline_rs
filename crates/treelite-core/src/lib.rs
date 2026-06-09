//! `treelite-core` — the foundation layer of treelite-rs.
//!
//! Ports the in-memory tree-ensemble representation from upstream Treelite
//! v4.7.0 (`treelite-mainline/`): the four shared enums, the `TreeBuf<T>`
//! struct-of-arrays storage primitive, the `Tree<T>` node columns, and the
//! two-variant `Model` with full header metadata.

pub mod enums;
pub mod error;
pub mod model;
pub mod serialize;
pub mod tree;
pub mod tree_buf;

pub use enums::{DType, Operator, TaskType, TreeNodeType};
pub use error::CoreError;
pub use model::{Model, ModelPreset, ModelVariant};
pub use serialize::{
    Frame, SerializeError, deserialize, serialize_to_buffer, serialize_to_pybuffer,
};
pub use tree::Tree;
pub use tree_buf::TreeBuf;

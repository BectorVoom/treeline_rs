//! The four shared vocabulary enums, ported verbatim from upstream
//! Treelite v4.7.0 (`treelite-mainline/src/enum/*.cc`).
//!
//! The string forms are **non-uniform** across the four enums (ENUM-01):
//! `TaskType` uses `kXxx`-style strings; the others use lowercase/symbolic.
//! Integer reprs match the upstream `.h` files so they line up with the
//! wire format in later phases. `FromString` on an unknown value is a fatal
//! error upstream; here it returns a typed [`CoreError`] (ERR-01).
//!
//! Variant names deliberately mirror the upstream C++ `kXxx` enumerators
//! verbatim (porting fidelity), so the `non_camel_case_types` lint is
//! suppressed for this module.
#![allow(non_camel_case_types)]

use crate::error::CoreError;

/// Prediction task kind. Upstream: `TaskType` (`task_type.{h,cc}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TaskType {
    /// Binary classification.
    kBinaryClf = 0,
    /// Regression.
    kRegressor = 1,
    /// Multi-class classification.
    kMultiClf = 2,
    /// Learning-to-rank.
    kLearningToRank = 3,
    /// Isolation forest.
    kIsolationForest = 4,
}

impl TaskType {
    /// Exact upstream string form (`task_type.cc:15-47`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::kBinaryClf => "kBinaryClf",
            TaskType::kRegressor => "kRegressor",
            TaskType::kMultiClf => "kMultiClf",
            TaskType::kLearningToRank => "kLearningToRank",
            TaskType::kIsolationForest => "kIsolationForest",
        }
    }

    /// Parse from the exact upstream string. Unknown -> typed `Err`.
    pub fn from_str(s: &str) -> Result<Self, CoreError> {
        match s {
            "kBinaryClf" => Ok(TaskType::kBinaryClf),
            "kRegressor" => Ok(TaskType::kRegressor),
            "kMultiClf" => Ok(TaskType::kMultiClf),
            "kLearningToRank" => Ok(TaskType::kLearningToRank),
            "kIsolationForest" => Ok(TaskType::kIsolationForest),
            other => Err(CoreError::UnknownEnumString {
                kind: "TaskType",
                value: other.to_string(),
            }),
        }
    }
}

/// Node kind. Upstream: `TreeNodeType` (`tree_node_type.{h,cc}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum TreeNodeType {
    /// Leaf node.
    kLeafNode = 0,
    /// Numerical test node.
    kNumericalTestNode = 1,
    /// Categorical test node.
    kCategoricalTestNode = 2,
}

impl TreeNodeType {
    /// Exact upstream string form (`tree_node_type.cc:15-39`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TreeNodeType::kLeafNode => "leaf_node",
            TreeNodeType::kNumericalTestNode => "numerical_test_node",
            TreeNodeType::kCategoricalTestNode => "categorical_test_node",
        }
    }

    /// Parse from the exact upstream string. Unknown -> typed `Err`.
    pub fn from_str(s: &str) -> Result<Self, CoreError> {
        match s {
            "leaf_node" => Ok(TreeNodeType::kLeafNode),
            "numerical_test_node" => Ok(TreeNodeType::kNumericalTestNode),
            "categorical_test_node" => Ok(TreeNodeType::kCategoricalTestNode),
            other => Err(CoreError::UnknownEnumString {
                kind: "TreeNodeType",
                value: other.to_string(),
            }),
        }
    }
}

/// Split comparison operator. Upstream: `Operator` (`operator.{h,cc}`).
///
/// Note: `kNone` maps to the empty string `""` (`operator.cc:16-49`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Operator {
    /// No operator (string form is `""`).
    kNone = 0,
    /// Equality `==`.
    kEQ = 1,
    /// Less-than `<`.
    kLT = 2,
    /// Less-than-or-equal `<=`.
    kLE = 3,
    /// Greater-than `>`.
    kGT = 4,
    /// Greater-than-or-equal `>=`.
    kGE = 5,
}

impl Operator {
    /// Exact upstream string form (`operator.cc:16-49`); `kNone` -> `""`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Operator::kNone => "",
            Operator::kEQ => "==",
            Operator::kLT => "<",
            Operator::kLE => "<=",
            Operator::kGT => ">",
            Operator::kGE => ">=",
        }
    }

    /// Parse from the exact upstream string. Unknown -> typed `Err`.
    ///
    /// `""` round-trips back to `kNone`, mirroring the upstream default arm.
    pub fn from_str(s: &str) -> Result<Self, CoreError> {
        match s {
            "" => Ok(Operator::kNone),
            "==" => Ok(Operator::kEQ),
            "<" => Ok(Operator::kLT),
            "<=" => Ok(Operator::kLE),
            ">" => Ok(Operator::kGT),
            ">=" => Ok(Operator::kGE),
            other => Err(CoreError::UnknownEnumString {
                kind: "Operator",
                value: other.to_string(),
            }),
        }
    }
}

/// Numeric type tag. Upstream: `TypeInfo` (`typeinfo.{h,cc}`).
///
/// Note: `TypeInfoFromString` does NOT accept `"invalid"` as input — only
/// `uint32`/`float32`/`float64`. `DType::kInvalid.as_str()` still returns
/// `"invalid"`, but `from_str("invalid")` is an `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    /// Invalid / unset.
    kInvalid = 0,
    /// 32-bit unsigned integer.
    kUInt32 = 1,
    /// 32-bit float.
    kFloat32 = 2,
    /// 64-bit float.
    kFloat64 = 3,
}

impl DType {
    /// Exact upstream string form (`typeinfo.cc:15-42`).
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::kInvalid => "invalid",
            DType::kUInt32 => "uint32",
            DType::kFloat32 => "float32",
            DType::kFloat64 => "float64",
        }
    }

    /// Parse from the exact upstream string. Mirrors `TypeInfoFromString`,
    /// which rejects `"invalid"` as input (only uint32/float32/float64).
    /// Unknown (including `"invalid"`) -> typed `Err`.
    pub fn from_str(s: &str) -> Result<Self, CoreError> {
        match s {
            "uint32" => Ok(DType::kUInt32),
            "float32" => Ok(DType::kFloat32),
            "float64" => Ok(DType::kFloat64),
            other => Err(CoreError::UnknownEnumString {
                kind: "DType",
                value: other.to_string(),
            }),
        }
    }
}

//! Enum string round-trip tests (ENUM-01, ERR-01).
//!
//! Asserts each of the four enums round-trips through its EXACT upstream
//! string and that unknown strings return a typed `Err` rather than panic.

use treelite_core::{CoreError, DType, Operator, TaskType, TreeNodeType};

#[test]
fn task_type_exact_strings() {
    assert_eq!(TaskType::kBinaryClf.as_str(), "kBinaryClf");
    assert_eq!(TaskType::kRegressor.as_str(), "kRegressor");
    assert_eq!(TaskType::kMultiClf.as_str(), "kMultiClf");
    assert_eq!(TaskType::kLearningToRank.as_str(), "kLearningToRank");
    assert_eq!(TaskType::kIsolationForest.as_str(), "kIsolationForest");
}

#[test]
fn task_type_round_trip() {
    for v in [
        TaskType::kBinaryClf,
        TaskType::kRegressor,
        TaskType::kMultiClf,
        TaskType::kLearningToRank,
        TaskType::kIsolationForest,
    ] {
        assert_eq!(TaskType::from_str(v.as_str()), Ok(v));
    }
}

#[test]
fn tree_node_type_exact_strings_and_round_trip() {
    assert_eq!(TreeNodeType::kLeafNode.as_str(), "leaf_node");
    assert_eq!(
        TreeNodeType::kNumericalTestNode.as_str(),
        "numerical_test_node"
    );
    assert_eq!(
        TreeNodeType::kCategoricalTestNode.as_str(),
        "categorical_test_node"
    );
    for v in [
        TreeNodeType::kLeafNode,
        TreeNodeType::kNumericalTestNode,
        TreeNodeType::kCategoricalTestNode,
    ] {
        assert_eq!(TreeNodeType::from_str(v.as_str()), Ok(v));
    }
}

#[test]
fn operator_exact_strings_and_round_trip() {
    assert_eq!(Operator::kNone.as_str(), "");
    assert_eq!(Operator::kEQ.as_str(), "==");
    assert_eq!(Operator::kLT.as_str(), "<");
    assert_eq!(Operator::kLE.as_str(), "<=");
    assert_eq!(Operator::kGT.as_str(), ">");
    assert_eq!(Operator::kGE.as_str(), ">=");
    for v in [
        Operator::kNone,
        Operator::kEQ,
        Operator::kLT,
        Operator::kLE,
        Operator::kGT,
        Operator::kGE,
    ] {
        assert_eq!(Operator::from_str(v.as_str()), Ok(v));
    }
}

#[test]
fn dtype_exact_strings() {
    assert_eq!(DType::kInvalid.as_str(), "invalid");
    assert_eq!(DType::kUInt32.as_str(), "uint32");
    assert_eq!(DType::kFloat32.as_str(), "float32");
    assert_eq!(DType::kFloat64.as_str(), "float64");
}

#[test]
fn dtype_round_trip_excludes_invalid() {
    // The three accepted-as-input variants round-trip.
    for v in [DType::kUInt32, DType::kFloat32, DType::kFloat64] {
        assert_eq!(DType::from_str(v.as_str()), Ok(v));
    }
    // Upstream TypeInfoFromString rejects "invalid" as input.
    assert!(matches!(
        DType::from_str("invalid"),
        Err(CoreError::UnknownEnumString { kind: "DType", .. })
    ));
}

#[test]
fn unknown_strings_are_typed_errors_not_panics() {
    assert!(matches!(
        TaskType::from_str("not_a_task"),
        Err(CoreError::UnknownEnumString {
            kind: "TaskType",
            ..
        })
    ));
    assert!(matches!(
        TreeNodeType::from_str("nope"),
        Err(CoreError::UnknownEnumString {
            kind: "TreeNodeType",
            ..
        })
    ));
    assert!(matches!(
        Operator::from_str("=!="),
        Err(CoreError::UnknownEnumString {
            kind: "Operator",
            ..
        })
    ));
    assert!(matches!(
        DType::from_str("complex128"),
        Err(CoreError::UnknownEnumString { kind: "DType", .. })
    ));
}

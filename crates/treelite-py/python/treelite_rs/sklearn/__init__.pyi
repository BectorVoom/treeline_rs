"""Type stubs for treelite_rs.sklearn (PEP 561, D-10)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..model import Model

def import_model(sklearn_model: Any) -> Model: ...

# Raw compiled array loaders (Rust seam). The array-of-arrays parameters are
# sequences of 1-D numpy arrays (int64 / float64 per the dtype contract);
# `node_count` is an int64 numpy array; HistGB `nodes` is a sequence of bytes.
def load_random_forest_regressor(
    n_estimators: int,
    n_features: int,
    n_targets: int,
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
) -> Model: ...
def load_random_forest_classifier(
    n_estimators: int,
    n_features: int,
    n_targets: int,
    n_classes: List[int],
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
) -> Model: ...
def load_extra_trees_regressor(
    n_estimators: int,
    n_features: int,
    n_targets: int,
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
) -> Model: ...
def load_extra_trees_classifier(
    n_estimators: int,
    n_features: int,
    n_targets: int,
    n_classes: List[int],
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
) -> Model: ...
def load_gradient_boosting_regressor(
    n_iter: int,
    n_features: int,
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
    base_score: float,
) -> Model: ...
def load_gradient_boosting_classifier(
    n_iter: int,
    n_features: int,
    n_classes: int,
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
    base_scores: List[float],
) -> Model: ...
def load_isolation_forest(
    n_estimators: int,
    n_features: int,
    node_count: Any,
    children_left: Sequence[Any],
    children_right: Sequence[Any],
    feature: Sequence[Any],
    threshold: Sequence[Any],
    value: Sequence[Any],
    n_node_samples: Sequence[Any],
    weighted_n_node_samples: Sequence[Any],
    impurity: Sequence[Any],
    ratio_c: float,
) -> Model: ...
def load_hist_gradient_boosting_regressor(
    n_iter: int,
    n_features: int,
    expected_sizeof_node_struct: int,
    node_count: Any,
    nodes: Sequence[bytes],
    raw_left_cat_bitsets: Sequence[Any],
    features_map: List[int],
    categories_map: Optional[List[List[int]]],
    baseline_prediction: float,
) -> Model: ...
def load_hist_gradient_boosting_classifier(
    n_iter: int,
    n_features: int,
    n_classes: int,
    expected_sizeof_node_struct: int,
    node_count: Any,
    nodes: Sequence[bytes],
    raw_left_cat_bitsets: Sequence[Any],
    features_map: List[int],
    categories_map: Optional[List[List[int]]],
    baseline_prediction: List[float],
) -> Model: ...

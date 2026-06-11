"""scikit-learn estimator import shim (PY-04) — port of upstream ``importer.py``.

``import_model(fitted_estimator)`` extracts a fitted scikit-learn estimator's
per-tree arrays Python-side (branch-for-branch with
``treelite/sklearn/importer.py``) and hands them to the compiled
``_treelite_rs.sklearn.load_*`` array-signature loaders. The estimator object
NEVER crosses the FFI boundary — only numpy arrays do (D-01 discretion). The
returned :class:`~treelite_rs.Model` predicts within 1e-5 of upstream
``treelite.sklearn.import_model`` across RF / ET / GB / IsolationForest / HistGB.

The dtype contract mirrors ``importer.py``'s ``ArrayOfArrays`` choices and the
Rust loader signatures (``&[&[i64]]`` / ``&[&[f64]]``): ``children_left`` /
``children_right`` / ``feature`` / ``n_node_samples`` are ``int64``; ``threshold``
/ ``value`` / ``weighted_n_node_samples`` / ``impurity`` are ``float64``;
``node_count`` is an ``int64`` array. HistGB hands the raw packed node ``bytes``
per tree.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .. import _treelite_rs

# The compiled sklearn array loaders (Rust seam). Names mirror src/sklearn.rs.
_S = _treelite_rs.sklearn

# Re-export the raw typed entry points so the A/B suite can hit a specific
# loader directly (parity with the gtil shim).
load_random_forest_regressor = _S.load_random_forest_regressor
load_random_forest_classifier = _S.load_random_forest_classifier
load_extra_trees_regressor = _S.load_extra_trees_regressor
load_extra_trees_classifier = _S.load_extra_trees_classifier
load_gradient_boosting_regressor = _S.load_gradient_boosting_regressor
load_gradient_boosting_classifier = _S.load_gradient_boosting_classifier
load_isolation_forest = _S.load_isolation_forest
load_hist_gradient_boosting_regressor = _S.load_hist_gradient_boosting_regressor
load_hist_gradient_boosting_classifier = _S.load_hist_gradient_boosting_classifier

__all__ = [
    "import_model",
    "load_random_forest_regressor",
    "load_random_forest_classifier",
    "load_extra_trees_regressor",
    "load_extra_trees_classifier",
    "load_gradient_boosting_regressor",
    "load_gradient_boosting_classifier",
    "load_isolation_forest",
    "load_hist_gradient_boosting_regressor",
    "load_hist_gradient_boosting_classifier",
]


def _harmonic(number: float) -> float:
    """The n-th harmonic number (isolation_forest.py port)."""
    return float(np.log(number) + np.euler_gamma)


def _expected_depth(n_remainder: int) -> float:
    """Expected isolation depth for a remainder of uniform points (port)."""
    if n_remainder <= 1:
        return 0.0
    if n_remainder == 2:
        return 1.0
    return float(2 * _harmonic(n_remainder - 1) - 2 * (n_remainder - 1) / n_remainder)


def _calculate_depths(isolation_depths, tree, curr_node: int, curr_depth: float) -> None:
    """Fill an array of isolation depths for an isolation-forest tree (port)."""
    if tree.children_left[curr_node] == -1:
        isolation_depths[curr_node] = curr_depth + _expected_depth(
            tree.n_node_samples[curr_node]
        )
    else:
        _calculate_depths(
            isolation_depths, tree, tree.children_left[curr_node], curr_depth + 1
        )
        _calculate_depths(
            isolation_depths, tree, tree.children_right[curr_node], curr_depth + 1
        )


class _ArrayOfArrays:
    """Collect per-tree 1-D numpy columns of a fixed dtype (importer.py port).

    Each ``add``-ed array is cast to ``dtype`` and made C-contiguous so the
    compiled loader can borrow it zero-copy as the matching ``&[&[T]]`` element.
    """

    def __init__(self, *, dtype):
        self.dtype = dtype
        self.collection: List[np.ndarray] = []

    def add(self, array) -> None:
        self.collection.append(np.ascontiguousarray(np.asarray(array, dtype=self.dtype)))

    def as_list(self) -> List[np.ndarray]:
        return self.collection


def _extract_forest(sklearn_model, *, is_gb: bool, is_iforest: bool):
    """Extract the per-tree node arrays (the importer.py loop, branch-for-branch).

    Returns ``(node_count, children_left, children_right, feature, threshold,
    value, n_node_samples, weighted_n_node_samples, impurity)`` where the
    ``children_*`` / ``feature`` / ``n_node_samples`` lists are int64 and the
    ``threshold`` / ``value`` / ``weighted_*`` / ``impurity`` lists are float64.
    """
    node_count: List[int] = []
    children_left = _ArrayOfArrays(dtype=np.int64)
    children_right = _ArrayOfArrays(dtype=np.int64)
    feature = _ArrayOfArrays(dtype=np.int64)
    threshold = _ArrayOfArrays(dtype=np.float64)
    value = _ArrayOfArrays(dtype=np.float64)
    n_node_samples = _ArrayOfArrays(dtype=np.int64)
    weighted_n_node_samples = _ArrayOfArrays(dtype=np.float64)
    impurity = _ArrayOfArrays(dtype=np.float64)

    estimators_features = getattr(sklearn_model, "estimators_features_", None)
    for tree_idx, estimator in enumerate(sklearn_model.estimators_):
        if is_gb:
            estimator_range = list(estimator)  # one sub-estimator per class
            learning_rate = sklearn_model.learning_rate
        else:
            estimator_range = [estimator]
            learning_rate = 1.0
        if is_iforest:
            isolation_depths = np.zeros(
                estimator.tree_.n_node_samples.shape[0], dtype="float64"
            )
            _calculate_depths(isolation_depths, estimator.tree_, 0, 0.0)
        for sub_estimator in estimator_range:
            tree = sub_estimator.tree_
            node_count.append(int(tree.node_count))
            children_left.add(tree.children_left)
            children_right.add(tree.children_right)
            threshold.add(tree.threshold)
            if is_iforest:
                value.add(isolation_depths.reshape(-1))
                # importer.py:208-216 — subsample feature-index remap.
                feature_subsample = np.full(tree.feature.shape, -2, dtype=np.int64)
                mask = tree.feature != -2
                feature_subsample[mask] = np.asarray(
                    estimators_features[tree_idx]
                )[tree.feature[mask]]
                feature.add(feature_subsample)
            else:
                # GB: shrink each leaf output by the learning rate (importer.py
                # :218-223). For RF/ET learning_rate == 1.0 (no-op).
                value.add((tree.value * learning_rate).reshape(-1))
                feature.add(tree.feature)
            n_node_samples.add(tree.n_node_samples)
            weighted_n_node_samples.add(tree.weighted_n_node_samples)
            impurity.add(tree.impurity)

    return (
        np.ascontiguousarray(np.asarray(node_count, dtype=np.int64)),
        children_left.as_list(),
        children_right.as_list(),
        feature.as_list(),
        threshold.as_list(),
        value.as_list(),
        n_node_samples.as_list(),
        weighted_n_node_samples.as_list(),
        impurity.as_list(),
    )


# Minimum sklearn version whose HistGradientBoosting private layout
# (``_predictors`` → per-``TreePredictor`` ``.nodes`` / ``.raw_left_cat_bitsets``,
# and ``_baseline_prediction``) matches what this importer reads. The extraction
# loop below reaches into these private attributes; on an older / unexpected
# sklearn the layout may differ, which would otherwise raise a bare
# ``AttributeError`` instead of the single ``TreeliteError`` (D-06, WR-05).
_HISTGB_MIN_SKLEARN = "1.0.0"


def _import_hist_gradient_boosting(sklearn_model):
    """Load HistGradientBoosting{Regressor,Classifier} (importer.py:355 port)."""
    from packaging.version import parse as parse_version

    import sklearn as _sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier as _HistC
    from sklearn.ensemble import HistGradientBoostingRegressor as _HistR

    # WR-05: guard the private-attribute access against an unsupported sklearn
    # version / private layout BEFORE reaching into ``_predictors`` etc., so any
    # incompatibility surfaces as one actionable ``TreeliteError`` (D-06) rather
    # than a bare ``AttributeError`` silently dependent on the installed ABI.
    if parse_version(_sklearn.__version__) < parse_version(_HISTGB_MIN_SKLEARN):
        raise _treelite_rs.TreeliteError(
            f"HistGradientBoosting import requires scikit-learn >= "
            f"{_HISTGB_MIN_SKLEARN}; found {_sklearn.__version__}. The importer "
            f"reads private estimator internals (_predictors / nodes / "
            f"_baseline_prediction) whose layout is not supported on this version."
        )
    for _attr in ("_predictors", "_baseline_prediction"):
        if not hasattr(sklearn_model, _attr):
            raise _treelite_rs.TreeliteError(
                f"HistGradientBoosting import: fitted estimator is missing the "
                f"private attribute '{_attr}' expected by this importer "
                f"(scikit-learn {_sklearn.__version__}); the private layout is "
                f"incompatible. Re-fit with a supported scikit-learn version."
            )

    # features_map (feat_remapper): identity arange when no categorical
    # preprocessor, else the embedded-OrdinalEncoder remap (importer.py:371-410).
    categories_map: List[List[int]] = []
    preprocessor = getattr(sklearn_model, "_preprocessor", None)
    if (
        parse_version(_sklearn.__version__) >= parse_version("1.4.0")
        and preprocessor is not None
    ):
        for cats in preprocessor.transformers_[0][1].categories_:
            if cats.dtype.type is np.str_:
                raise _treelite_rs.TreeliteError("String categories are not supported")
            categories_map.append(cats.astype(np.int64).tolist())
        feat_remapper = np.zeros((sklearn_model.n_features_in_,), dtype=np.int32)
        n_categorical = int(sklearn_model.is_categorical_.sum())
        cat_idx, num_idx = 0, 0
        for i, e in enumerate(sklearn_model.is_categorical_):
            if e:
                feat_remapper[cat_idx] = i
                cat_idx += 1
            else:
                feat_remapper[n_categorical + num_idx] = i
                num_idx += 1
    else:
        feat_remapper = np.arange(
            start=0, stop=sklearn_model.n_features_in_, dtype=np.int32
        )

    nodes: List[bytes] = []
    raw_left_cat_bitsets: List[np.ndarray] = []
    node_count: List[int] = []
    itemsize = None
    for estimator in sklearn_model._predictors:
        for sub_estimator in estimator:
            nodes.append(bytes(sub_estimator.nodes.tobytes()))
            raw_left_cat_bitsets.append(
                np.ascontiguousarray(
                    np.asarray(sub_estimator.raw_left_cat_bitsets, dtype=np.uint32).reshape(-1)
                )
            )
            node_count.append(int(len(sub_estimator.nodes)))
            if itemsize is None:
                itemsize = int(sub_estimator.nodes.itemsize)
            elif itemsize != int(sub_estimator.nodes.itemsize):
                raise _treelite_rs.TreeliteError("HistGB node itemsize mismatch")
    assert itemsize is not None

    node_count_arr = np.ascontiguousarray(np.asarray(node_count, dtype=np.int64))
    features_map = feat_remapper.astype(np.int32).reshape(-1).tolist()
    cat_map = categories_map if categories_map else None

    if isinstance(sklearn_model, _HistR):
        baseline = np.asarray(
            sklearn_model._baseline_prediction, dtype=np.float64
        ).reshape(-1)
        # WR-05: the Rust regressor loader signature takes a SINGLE
        # ``baseline_prediction: f64``, so we pass ``baseline[0]`` below. A
        # multi-target ``HistGradientBoostingRegressor`` (sklearn >= 1.4) has one
        # baseline per target; silently keeping only ``[0]`` would drop the rest
        # and emit numerically wrong predictions (a 1e-5 violation). Reject it
        # explicitly until the multi-target baseline path is supported.
        if baseline.size != 1:
            raise _treelite_rs.TreeliteError(
                "multi-target HistGradientBoostingRegressor is not yet supported "
                f"(got {baseline.size} baseline predictions; expected 1)"
            )
        return load_hist_gradient_boosting_regressor(
            int(sklearn_model.n_iter_),
            int(sklearn_model.n_features_in_),
            int(itemsize),
            node_count_arr,
            nodes,
            raw_left_cat_bitsets,
            features_map,
            cat_map,
            float(baseline[0]),
        )
    if isinstance(sklearn_model, _HistC):
        baseline = np.asarray(
            sklearn_model._baseline_prediction, dtype=np.float64
        ).reshape(-1)
        return load_hist_gradient_boosting_classifier(
            int(sklearn_model.n_iter_),
            int(sklearn_model.n_features_in_),
            int(len(sklearn_model.classes_)),
            int(itemsize),
            node_count_arr,
            nodes,
            raw_left_cat_bitsets,
            features_map,
            cat_map,
            baseline.tolist(),
        )
    raise _treelite_rs.TreeliteError(
        f"Unsupported model type {sklearn_model.__class__.__name__}"
    )


def import_model(sklearn_model):
    """Load a tree-ensemble model from a fitted scikit-learn estimator (PY-04).

    Supports RandomForest / ExtraTrees (Regressor + Classifier), GradientBoosting
    (Regressor + Classifier), IsolationForest, and HistGradientBoosting
    (Regressor + Classifier). The returned :class:`~treelite_rs.Model` predicts
    within 1e-5 of upstream ``treelite.sklearn.import_model``.

    Note
    ----
    For :py:class:`~sklearn.ensemble.IsolationForest`, the loaded model produces
    the standardized-ratio outlier score: ``predict(model, X) ==
    -clf.score_samples(X)`` (D-07), which differs from the framework's
    ``decision_function`` / ``predict``.
    """
    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesC
        from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesR
        from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingC
        from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingR
        from sklearn.ensemble import (
            HistGradientBoostingClassifier as HistGradientBoostingC,
        )
        from sklearn.ensemble import (
            HistGradientBoostingRegressor as HistGradientBoostingR,
        )
        from sklearn.ensemble import IsolationForest
        from sklearn.ensemble import RandomForestClassifier as RandomForestC
        from sklearn.ensemble import RandomForestRegressor as RandomForestR
    except ImportError as e:  # pragma: no cover
        raise _treelite_rs.TreeliteError(
            "This function requires the scikit-learn package"
        ) from e

    if isinstance(sklearn_model, (HistGradientBoostingR, HistGradientBoostingC)):
        return _import_hist_gradient_boosting(sklearn_model)

    if isinstance(sklearn_model, (RandomForestR, ExtraTreesR)):
        (
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        ) = _extract_forest(sklearn_model, is_gb=False, is_iforest=False)
        loader = (
            load_extra_trees_regressor
            if isinstance(sklearn_model, ExtraTreesR)
            else load_random_forest_regressor
        )
        return loader(
            int(sklearn_model.n_estimators),
            int(sklearn_model.n_features_in_),
            int(sklearn_model.n_outputs_),
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        )

    if isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        (
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        ) = _extract_forest(sklearn_model, is_gb=False, is_iforest=False)
        n_classes = np.atleast_1d(
            np.asarray(sklearn_model.n_classes_, dtype=np.int32)
        ).tolist()
        loader = (
            load_extra_trees_classifier
            if isinstance(sklearn_model, ExtraTreesC)
            else load_random_forest_classifier
        )
        return loader(
            int(sklearn_model.n_estimators),
            int(sklearn_model.n_features_in_),
            int(sklearn_model.n_outputs_),
            n_classes,
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        )

    if isinstance(sklearn_model, IsolationForest):
        (
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        ) = _extract_forest(sklearn_model, is_gb=False, is_iforest=True)
        ratio_c = _expected_depth(sklearn_model.max_samples_)
        return load_isolation_forest(
            int(sklearn_model.n_estimators),
            int(sklearn_model.n_features_in_),
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
            float(ratio_c),
        )

    if isinstance(sklearn_model, GradientBoostingR):
        (
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        ) = _extract_forest(sklearn_model, is_gb=True, is_iforest=False)
        if sklearn_model.init_ == "zero":
            base_score = 0.0
        elif isinstance(sklearn_model.init_, DummyRegressor):
            base_scores = np.asarray(sklearn_model.init_.constant_, dtype=np.float64).reshape(-1)
            assert base_scores.size == 1
            base_score = float(base_scores[0])
        else:
            raise _treelite_rs.TreeliteError("Custom init estimator not supported")
        return load_gradient_boosting_regressor(
            int(sklearn_model.n_estimators),
            int(sklearn_model.n_features_in_),
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
            base_score,
        )

    if isinstance(sklearn_model, GradientBoostingC):
        (
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
        ) = _extract_forest(sklearn_model, is_gb=True, is_iforest=False)
        if sklearn_model.init_ == "zero":
            base_scores = np.zeros(shape=(sklearn_model.n_classes_,), dtype=np.float64)
        elif isinstance(sklearn_model.init_, DummyClassifier):
            if sklearn_model.init_.strategy != "prior":
                raise _treelite_rs.TreeliteError("Custom init estimator not supported")
            base_scores = np.asarray(
                sklearn_model._raw_predict_init(
                    np.zeros((1, sklearn_model.n_features_in_))
                ),
                dtype=np.float64,
            ).reshape(-1)
        else:
            raise _treelite_rs.TreeliteError("Custom init estimator not supported")
        return load_gradient_boosting_classifier(
            int(sklearn_model.n_estimators),
            int(sklearn_model.n_features_in_),
            int(sklearn_model.n_classes_),
            node_count,
            children_left,
            children_right,
            feature,
            threshold,
            value,
            n_node_samples,
            weighted_n_node_samples,
            impurity,
            base_scores.tolist(),
        )

    raise _treelite_rs.TreeliteError(
        f"Not supported model type: {sklearn_model.__class__.__name__}"
    )

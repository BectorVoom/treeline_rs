"""PY-04: sklearn estimator import -> A/B predict within 1e-5 of upstream.

Fits a small scikit-learn estimator, imports it through both
``treelite_rs.sklearn.import_model`` and upstream
``treelite.sklearn.import_model``, predicts the same X through each, and asserts
the outputs agree within 1e-5 (D-11 live A/B). The estimator object never
crosses the FFI boundary — only numpy arrays do.

Covers the full estimator set: RandomForest / ExtraTrees (Regressor +
Classifier), GradientBoosting (Regressor + Classifier), IsolationForest, and
HistGradientBoosting (Regressor + Classifier, numerical + categorical).
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 1234


def _ab_predict(treelite_upstream, treelite_rs, clf, X):
    """Import ``clf`` through both stacks, predict ``X``, return (rs_out, up_out)."""
    data = np.ascontiguousarray(X.astype(np.float64))
    rs_model = treelite_rs.sklearn.import_model(clf)
    up_model = treelite_upstream.sklearn.import_model(clf)
    rs_out = np.asarray(treelite_rs.gtil.predict(rs_model, data))
    up_out = np.asarray(treelite_upstream.gtil.predict(up_model, data))
    return rs_out, up_out


def test_random_forest_regressor_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = rs.rand(40).astype(np.float64)
    clf = sklearn_ensemble.RandomForestRegressor(
        n_estimators=5, max_depth=4, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_extra_trees_regressor_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = rs.rand(40).astype(np.float64)
    clf = sklearn_ensemble.ExtraTreesRegressor(
        n_estimators=5, max_depth=4, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_random_forest_classifier_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    clf = sklearn_ensemble.RandomForestClassifier(
        n_estimators=5, max_depth=4, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_extra_trees_classifier_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    clf = sklearn_ensemble.ExtraTreesClassifier(
        n_estimators=5, max_depth=4, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_gradient_boosting_regressor_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = rs.rand(40).astype(np.float64)
    clf = sklearn_ensemble.GradientBoostingRegressor(
        n_estimators=5, max_depth=3, learning_rate=0.1, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_gradient_boosting_classifier_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(40, 4).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    clf = sklearn_ensemble.GradientBoostingClassifier(
        n_estimators=5, max_depth=3, learning_rate=0.1, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_isolation_forest_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(50, 4).astype(np.float64)
    clf = sklearn_ensemble.IsolationForest(
        n_estimators=5, max_samples=32, random_state=SEED
    ).fit(X)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_hist_gradient_boosting_regressor_numerical_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(60, 4).astype(np.float64)
    y = rs.rand(60).astype(np.float64)
    clf = sklearn_ensemble.HistGradientBoostingRegressor(
        max_iter=5, max_depth=3, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_hist_gradient_boosting_regressor_categorical_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    n = 80
    num_feat = rs.rand(n, 2).astype(np.float64)
    cat_feat = rs.randint(0, 4, size=(n, 1)).astype(np.float64)
    X = np.hstack([num_feat, cat_feat]).astype(np.float64)
    y = (cat_feat.reshape(-1) + num_feat[:, 0]).astype(np.float64)
    clf = sklearn_ensemble.HistGradientBoostingRegressor(
        max_iter=5,
        max_depth=3,
        categorical_features=[False, False, True],
        random_state=SEED,
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_hist_gradient_boosting_classifier_ab_1e5(treelite_upstream, treelite_rs):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
    rs = np.random.RandomState(SEED)
    X = rs.rand(60, 4).astype(np.float64)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    clf = sklearn_ensemble.HistGradientBoostingClassifier(
        max_iter=5, max_depth=3, random_state=SEED
    ).fit(X, y)

    rs_out, up_out = _ab_predict(treelite_upstream, treelite_rs, clf, X)
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)

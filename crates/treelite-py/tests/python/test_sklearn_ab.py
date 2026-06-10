"""PY-04: sklearn estimator import -> A/B predict within 1e-5 of upstream.

Fits a small scikit-learn estimator, imports it through both treelite_rs.sklearn
and upstream treelite.sklearn, and asserts predictions agree within 1e-5. RED
scaffold (flips GREEN in 08-04 once the sklearn loaders land).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.skip(reason="MISSING — sklearn.import_model A/B implemented in 08-04")
def test_random_forest_regressor_ab_1e5(treelite_upstream, treelite_rs, rng):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")

    x = rng.standard_normal((64, 5))
    y = rng.standard_normal(64)
    clf = sklearn_ensemble.RandomForestRegressor(n_estimators=4, random_state=0)
    clf.fit(x, y)

    rs_model = treelite_rs.sklearn.load_random_forest_regressor(clf)
    up_model = treelite_upstream.sklearn.import_model(clf)

    data = rng.standard_normal((16, 5)).astype(np.float64)
    rs_out = treelite_rs.gtil.predict_f64(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


@pytest.mark.skip(reason="MISSING — sklearn.import_model classifier A/B implemented in 08-04")
def test_gradient_boosting_classifier_ab_1e5(treelite_upstream, treelite_rs, rng):
    sklearn_ensemble = pytest.importorskip("sklearn.ensemble")

    x = rng.standard_normal((64, 5))
    y = (rng.standard_normal(64) > 0).astype(int)
    clf = sklearn_ensemble.GradientBoostingClassifier(n_estimators=4, random_state=0)
    clf.fit(x, y)

    rs_model = treelite_rs.sklearn.load_gradient_boosting_classifier(clf)
    up_model = treelite_upstream.sklearn.import_model(clf)

    data = rng.standard_normal((16, 5)).astype(np.float64)
    rs_out = treelite_rs.gtil.predict_f64(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)

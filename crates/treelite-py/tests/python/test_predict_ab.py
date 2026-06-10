"""PY-02 (D-11): the A/B headline test — treelite_rs.gtil.predict matches upstream
treelite.gtil.predict within 1e-5 across model classes, presets, and predict kinds.

This is the phase's core-value instrument: predictions must match upstream within
1e-5. RED scaffold (flips GREEN in 08-03 once gtil.predict* lands). The upstream
import is skip-guarded in conftest (importorskip) so this never errors where the
C++ reference is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURES

# (model fixture, loader-attr) pairs spanning the A/B model classes.
MODEL_CASES = [
    ("xgb_3format.json", "load_xgboost_json_str", "text"),
    ("lightgbm_categorical.txt", "load_lightgbm_str", "text"),
]


@pytest.mark.skip(reason="MISSING — gtil.predict A/B parity implemented in 08-03")
@pytest.mark.parametrize("fixture,loader,mode", MODEL_CASES)
def test_predict_matches_upstream_1e5(treelite_upstream, treelite_rs, rng, fixture, loader, mode):
    path = FIXTURES / fixture
    payload = path.read_text() if mode == "text" else path.read_bytes()

    rs_model = getattr(treelite_rs.frontend, loader)(payload)
    up_model = treelite_upstream.frontend.load_xgboost_model(str(path))

    data = rng.standard_normal((16, rs_model.num_feature)).astype(np.float32)

    rs_out = treelite_rs.gtil.predict_f32(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


@pytest.mark.skip(reason="MISSING — gtil.predict f64 preset parity implemented in 08-03")
def test_predict_f64_preset_matches_upstream_1e5(treelite_upstream, treelite_rs, rng):
    path = FIXTURES / "xgb_3format.json"
    rs_model = treelite_rs.frontend.load_xgboost_json_str(path.read_text())
    up_model = treelite_upstream.frontend.load_xgboost_model(str(path))

    data = rng.standard_normal((16, rs_model.num_feature)).astype(np.float64)

    rs_out = treelite_rs.gtil.predict_f64(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)

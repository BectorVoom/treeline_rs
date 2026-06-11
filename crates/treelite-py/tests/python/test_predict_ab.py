"""PY-02 (D-11): the A/B headline test — treelite_rs.gtil.predict matches upstream
treelite.gtil.predict within 1e-5 across model classes, presets, and predict kinds.

This is the phase's core-value instrument: predictions must match upstream within
1e-5. The upstream import is skip-guarded in conftest (importorskip) so this never
errors where the C++ reference is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURES

# (model fixture, rs-loader-attr, upstream-loader-attr) triples spanning the A/B
# model classes. The upstream loader differs per format (XGBoost vs LightGBM).
MODEL_CASES = [
    ("xgb_3format.json", "load_xgboost_json_str", "load_xgboost_model"),
    ("lightgbm_categorical.txt", "load_lightgbm_str", "load_lightgbm_model"),
]


@pytest.mark.parametrize("fixture,rs_loader,up_loader", MODEL_CASES)
def test_predict_matches_upstream_1e5(
    treelite_upstream, treelite_rs, rng, fixture, rs_loader, up_loader
):
    path = FIXTURES / fixture

    rs_model = getattr(treelite_rs.frontend, rs_loader)(path.read_text())
    up_model = getattr(treelite_upstream.frontend, up_loader)(str(path))

    data = np.ascontiguousarray(
        rng.standard_normal((16, rs_model.num_feature)), dtype=np.float32
    )

    # The headline shim: dtype-dispatch + flat→N-D reshape (D-11).
    rs_out = treelite_rs.gtil.predict(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    # Shape parity (num_row, num_target, max_num_class) AND value parity within 1e-5.
    assert rs_out.shape == up_out.shape
    np.testing.assert_allclose(rs_out, up_out, atol=1e-5, rtol=0)


def test_predict_f64_input_matches_upstream_1e5(treelite_upstream, treelite_rs, rng):
    """f64-INPUT path: predict_f64 over an f64 matrix (the f32 XGBoost model's leaf
    values accumulate into an f64 buffer — the InputT-as-accumulator contract the
    monomorphized entry points preserve, no f32→f64 pre-cast)."""
    path = FIXTURES / "xgb_3format.json"
    rs_model = treelite_rs.frontend.load_xgboost_json_str(path.read_text())
    up_model = treelite_upstream.frontend.load_xgboost_model(str(path))

    data = np.ascontiguousarray(
        rng.standard_normal((16, rs_model.num_feature)), dtype=np.float64
    )

    rs_flat = treelite_rs.gtil.predict_f64(rs_model, data)
    up_out = treelite_upstream.gtil.predict(up_model, data)

    # The raw entry point returns a flat 1-D buffer; reshape to compare element-wise.
    np.testing.assert_allclose(
        rs_flat.reshape(up_out.shape), up_out, atol=1e-5, rtol=0
    )

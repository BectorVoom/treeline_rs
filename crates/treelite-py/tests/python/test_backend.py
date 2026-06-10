"""D-05 / D-08: the gtil.predict* ``backend=`` kwarg.

Asserts (a) ``backend='cpu'`` works (always-available default backend), (b)
requesting a backend the wheel was NOT built with raises TreeliteError, and (c)
the ROCm cell is hardware-gated (skip-not-fail where no AMD/ROCm device exists).
RED scaffold (flips GREEN in 08-05 once the backend kwarg lands).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import FIXTURES


@pytest.mark.skip(reason="MISSING — gtil.predict backend= kwarg implemented in 08-05")
def test_predict_backend_cpu_works(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    out = treelite_rs.gtil.predict_f32(model, data, backend="cpu")
    assert out is not None


@pytest.mark.skip(reason="MISSING — un-built backend rejection implemented in 08-05")
def test_predict_unbuilt_backend_raises(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, data, backend="cuda")


@pytest.mark.skipif(
    os.environ.get("TREELITE_RS_ROCM") != "1",
    reason="ROCm backend cell is hardware-gated (set TREELITE_RS_ROCM=1 on an AMD/ROCm box)",
)
@pytest.mark.skip(reason="MISSING — rocm backend predict implemented in 08-05")
def test_predict_backend_rocm_matches_cpu(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    cpu_out = treelite_rs.gtil.predict_f32(model, data, backend="cpu")
    rocm_out = treelite_rs.gtil.predict_f32(model, data, backend="rocm")
    np.testing.assert_allclose(rocm_out, cpu_out, atol=1e-5, rtol=0)

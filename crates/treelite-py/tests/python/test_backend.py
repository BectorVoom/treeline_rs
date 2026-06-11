"""D-05 / D-08: the gtil.predict* ``backend=`` kwarg.

Asserts (a) ``backend='cpu'`` works and matches the no-kwarg path within 1e-5
(always-available default backend), (b) requesting a backend the wheel was NOT
built with raises TreeliteError (never a silent CPU fallback, D-08), and (c) the
ROCm cell is hardware-gated (skip-not-fail where no AMD/ROCm device exists —
validated on-device in Task 3 with TREELITE_RS_ROCM=1).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import FIXTURES


def _load_model(treelite_rs):
    return treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )


def test_predict_backend_cpu_matches_no_kwarg(treelite_rs, rng):
    # backend='cpu' must produce the SAME output as omitting the kwarg (D-05:
    # the no-kwarg call stays upstream-identical; cpu is the default).
    model = _load_model(treelite_rs)
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    default_out = treelite_rs.gtil.predict_f32(model, data)
    cpu_out = treelite_rs.gtil.predict_f32(model, data, backend="cpu")
    np.testing.assert_allclose(cpu_out, default_out, atol=1e-5, rtol=0)


def test_predict_unbuilt_backend_raises(treelite_rs, rng):
    # A backend name not compiled into THIS (default cpu-only) wheel must raise
    # TreeliteError, never silently fall back to CPU (D-08 / T-08-13).
    model = _load_model(treelite_rs)
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, data, backend="cuda")


def test_predict_nonexistent_backend_raises(treelite_rs, rng):
    # An entirely unknown backend name must also raise TreeliteError (T-08-13).
    model = _load_model(treelite_rs)
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, data, backend="nonexistent")


@pytest.mark.skipif(
    os.environ.get("TREELITE_RS_ROCM") != "1",
    reason="ROCm backend cell is hardware-gated (set TREELITE_RS_ROCM=1 on an AMD/ROCm box)",
)
def test_predict_backend_rocm_matches_cpu(treelite_rs, rng):
    model = _load_model(treelite_rs)
    data = rng.standard_normal((16, model.num_feature)).astype(np.float32)
    cpu_out = treelite_rs.gtil.predict_f32(model, data, backend="cpu")
    rocm_out = treelite_rs.gtil.predict_f32(model, data, backend="rocm")
    np.testing.assert_allclose(rocm_out, cpu_out, atol=1e-5, rtol=0)

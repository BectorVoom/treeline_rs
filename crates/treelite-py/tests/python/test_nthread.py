"""PAR-04 (Phase 10): the Python ``nthread=`` kwarg drives the scalar predict path
end-to-end. ``gtil.predict(model, data, nthread=2)`` must equal
``gtil.predict(model, data, nthread=1)`` within 1e-5 over a scalar-fallback model.

The LightGBM categorical fixture routes WHOLE-model to the scalar
``treelite_gtil::predict`` fallback (categorical splits are not on the cubecl
kernel path), so this exercises exactly the row-parallelized scalar engine that
Phase 10 wires ``Config.nthread`` into. ``nthread`` bounds the rayon pool but must
not change the result (the per-row split is independent; GTIL-08 keeps the
per-row tree sum serial).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURES


def test_nthread_two_equals_one_scalar_fallback(treelite_rs, rng):
    """nthread=2 == nthread=1 within 1e-5 over the LightGBM categorical
    scalar-fallback fixture (the Python kwarg now drives the scalar rayon pool)."""
    path = FIXTURES / "lightgbm_categorical.txt"
    rs_model = treelite_rs.frontend.load_lightgbm_str(path.read_text())

    # Enough rows that the rayon par_chunks_mut split is real, not a 1-row no-op.
    data = np.ascontiguousarray(
        rng.standard_normal((256, rs_model.num_feature)), dtype=np.float32
    )

    out_n1 = treelite_rs.gtil.predict(rs_model, data, nthread=1)
    out_n2 = treelite_rs.gtil.predict(rs_model, data, nthread=2)

    assert out_n1.shape == out_n2.shape
    np.testing.assert_allclose(out_n2, out_n1, atol=1e-5, rtol=0)


def test_nthread_all_cores_equals_one_scalar_fallback(treelite_rs, rng):
    """nthread<=0 (all cores, global pool) == nthread=1 within 1e-5 — the pool
    size must not change the prediction (PAR-04)."""
    path = FIXTURES / "lightgbm_categorical.txt"
    rs_model = treelite_rs.frontend.load_lightgbm_str(path.read_text())

    data = np.ascontiguousarray(
        rng.standard_normal((256, rs_model.num_feature)), dtype=np.float32
    )

    out_n1 = treelite_rs.gtil.predict(rs_model, data, nthread=1)
    out_all = treelite_rs.gtil.predict(rs_model, data, nthread=-1)

    assert out_n1.shape == out_all.shape
    np.testing.assert_allclose(out_all, out_n1, atol=1e-5, rtol=0)

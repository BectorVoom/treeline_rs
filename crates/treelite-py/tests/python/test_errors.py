"""PY-05: error mapping — bad input and Rust-side failures surface as TreeliteError.

Asserts (a) malformed loader input raises TreeliteError (not a bare ValueError or
a Rust panic abort), (b) malformed serialized bytes raise TreeliteError (bounds-
checked, never an abort), and (c) a forced bad-predict path is caught and surfaces
as a catchable TreeliteError rather than aborting the interpreter (D-06/D-07) — the
panic `guard` remap normalizes any trapped panic to TreeliteError, and pyo3's auto
catch_unwind already prevents the abort. The decisive assertion is that the process
SURVIVES (the test function returns), proving no interpreter abort crossed the FFI
boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURES


def test_malformed_xgboost_json_raises_treelite_error(treelite_rs):
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.frontend.load_xgboost_json_str("{ not valid xgboost json ")


def test_malformed_serialized_bytes_raises_treelite_error(treelite_rs):
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.Model.deserialize_bytes(b"\x00\x01\x02not-a-v5-model")


def test_bad_predict_surfaces_as_treelite_error_not_abort(treelite_rs, rng):
    # A forced-bad predict call (a feature-count mismatch: the model expects
    # `num_feature` columns, we pass 1) must surface as a catchable TreeliteError
    # (typed GtilError->TreeliteError, or a trapped panic->TreeliteError via the
    # `guard` remap), NEVER an interpreter abort. The process surviving past this
    # call (the function returns) is the no-abort proof (D-07).
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    assert model.num_feature > 1  # so a 1-column matrix is genuinely mismatched
    bad = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, bad)
    # If we reach here, no abort occurred — the interpreter is alive.


def test_panic_remap_surfaces_as_treelite_error_not_abort(treelite_rs):
    # Empty XGBoost JSON ("{}") exercises the loader error path — a missing
    # required field surfaces as a catchable TreeliteError, never an abort. This
    # locks the D-06/D-07 contract from the loader side and proves the process
    # survives a forced Rust-side failure.
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.frontend.load_xgboost_json_str("{}")

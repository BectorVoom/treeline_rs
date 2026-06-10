"""PY-05: error mapping — bad input and Rust-side failures surface as TreeliteError.

Asserts (a) malformed loader input raises TreeliteError (not a bare ValueError or
a Rust panic abort), and (b) a forced Rust panic is caught and remapped to
TreeliteError rather than aborting the interpreter (D-07). RED scaffold (flips
GREEN in 08-02/08-03 as error mapping + the panic remap land).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="MISSING — TreeliteError mapping implemented in 08-02")
def test_malformed_xgboost_json_raises_treelite_error(treelite_rs):
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.frontend.load_xgboost_json_str("{ not valid xgboost json ")


@pytest.mark.skip(reason="MISSING — TreeliteError mapping implemented in 08-02")
def test_malformed_serialized_bytes_raises_treelite_error(treelite_rs):
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.Model.deserialize_bytes(b"\x00\x01\x02not-a-v5-model")


@pytest.mark.skip(reason="MISSING — panic->TreeliteError remap implemented in 08-03")
def test_panic_surfaces_as_treelite_error_not_abort(treelite_rs):
    # A forced-bad predict call (e.g. mismatched feature count) must surface as a
    # catchable TreeliteError, never an interpreter abort.
    import numpy as np

    bad = np.zeros((1, 1), dtype=np.float32)
    model = treelite_rs.frontend.load_xgboost_json_str("{}")  # also exercises the loader error
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, bad)

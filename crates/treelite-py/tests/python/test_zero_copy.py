"""MEM-04 / D-03: zero-copy numpy input borrow + strict dtype/contiguity gate.

Proves (a) the input array's data pointer is unchanged across the predict call
(no copy of the borrowed buffer), mirroring the Rust zero-copy PyBuffer proof at
crates/treelite-core/tests/serialize_pybuffer.rs:80-90, and (b) a non-contiguous
or wrong-dtype array raises TreeliteError rather than silently copying.

RED scaffold (flips GREEN in 08-03).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURES


def test_predict_input_is_zero_copy(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    data = np.ascontiguousarray(
        rng.standard_normal((16, model.num_feature)), dtype=np.float32
    )
    ptr_before = data.__array_interface__["data"][0]
    treelite_rs.gtil.predict_f32(model, data)
    ptr_after = data.__array_interface__["data"][0]
    assert ptr_before == ptr_after


def test_predict_noncontiguous_raises(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    base = np.ascontiguousarray(
        rng.standard_normal((16, model.num_feature * 2)), dtype=np.float32
    )
    noncontig = base[:, ::2]  # strided view -> not C-contiguous
    assert not noncontig.flags["C_CONTIGUOUS"]
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, noncontig)


def test_predict_wrong_dtype_raises(treelite_rs, rng):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    data = np.ascontiguousarray(
        rng.standard_normal((16, model.num_feature)), dtype=np.float64
    )
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.gtil.predict_f32(model, data)  # f64 array into f32 entry point

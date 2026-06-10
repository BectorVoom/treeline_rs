"""PY-01: frontend loaders import external model files into a treelite_rs Model.

RED scaffold (flips GREEN in 08-02/08-04 as the loaders land). Each case loads a
frozen fixture and asserts the resulting Model's ``num_tree`` / ``num_feature``.
"""

from __future__ import annotations

import pytest

from conftest import FIXTURES


@pytest.mark.skip(reason="MISSING — frontend.load_xgboost_json_str implemented in 08-02")
def test_load_xgboost_json(treelite_rs):
    text = (FIXTURES / "xgb_3format.json").read_text()
    model = treelite_rs.frontend.load_xgboost_json_str(text)
    assert model.num_feature == 4
    assert model.num_tree >= 1


@pytest.mark.skip(reason="MISSING — frontend.load_xgboost_ubjson_bytes implemented in 08-02")
def test_load_xgboost_ubjson(treelite_rs):
    data = (FIXTURES / "xgb_3format.ubj").read_bytes()
    model = treelite_rs.frontend.load_xgboost_ubjson_bytes(data)
    assert model.num_feature == 4
    assert model.num_tree >= 1


@pytest.mark.skip(reason="MISSING — frontend.load_xgboost_legacy_bytes implemented in 08-02")
def test_load_xgboost_legacy(treelite_rs):
    data = (FIXTURES / "xgb_3format.model").read_bytes()
    model = treelite_rs.frontend.load_xgboost_legacy_bytes(data)
    assert model.num_feature == 4
    assert model.num_tree >= 1


@pytest.mark.skip(reason="MISSING — frontend.load_lightgbm_str implemented in 08-02")
def test_load_lightgbm_numerical(treelite_rs):
    text = (FIXTURES / "lightgbm_categorical.txt").read_text()
    model = treelite_rs.frontend.load_lightgbm_str(text)
    assert model.num_feature >= 1
    assert model.num_tree >= 1


@pytest.mark.skip(reason="MISSING — frontend.detect_xgboost_format_bytes implemented in 08-02")
def test_detect_xgboost_format(treelite_rs):
    json_bytes = (FIXTURES / "xgb_3format.json").read_bytes()
    ubj_bytes = (FIXTURES / "xgb_3format.ubj").read_bytes()
    assert treelite_rs.frontend.detect_xgboost_format_bytes(json_bytes) == "json"
    assert treelite_rs.frontend.detect_xgboost_format_bytes(ubj_bytes) == "ubjson"

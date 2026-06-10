"""PY-03: serialize/deserialize round-trip + dump_as_json.

Asserts serialize_bytes -> deserialize_bytes preserves the model (same num_tree /
num_feature, same re-serialized bytes) and that dump_as_json produces parseable
JSON. RED scaffold (flips GREEN in 08-02/08-03).
"""

from __future__ import annotations

import json

import pytest

from conftest import FIXTURES


@pytest.mark.skip(reason="MISSING — Model.serialize_bytes/deserialize_bytes implemented in 08-02")
def test_serialize_roundtrip_bytes(treelite_rs):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    blob = model.serialize_bytes()
    assert isinstance(blob, (bytes, bytearray))
    restored = treelite_rs.Model.deserialize_bytes(blob)
    assert restored.num_tree == model.num_tree
    assert restored.num_feature == model.num_feature
    # Re-serialization must be byte-stable across the round-trip.
    assert restored.serialize_bytes() == blob


@pytest.mark.skip(reason="MISSING — Model.dump_as_json implemented in 08-02")
def test_dump_as_json_parses(treelite_rs):
    model = treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )
    parsed = json.loads(model.dump_as_json())
    assert isinstance(parsed, dict)

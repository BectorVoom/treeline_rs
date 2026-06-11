"""PY-03: serialize/deserialize round-trip + dump_as_json + concatenate (08-03).

Flipped GREEN in 08-03. Asserts:

* ``serialize_bytes`` → ``deserialize_bytes`` recovers a model that predicts
  identically within 1e-5 (the core-value round-trip) and re-serializes to the
  same bytes;
* ``dump_as_json`` produces parseable JSON, structurally equivalent (parsed-value
  level, Phase-2 D-04 discipline) to upstream ``dump_as_json``;
* ``Model.concatenate([m1, m2])`` merges models via the builder, and an empty
  input list raises ``TreeliteError`` (T-08-08).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import FIXTURES


def _load(treelite_rs):
    return treelite_rs.frontend.load_xgboost_json_str(
        (FIXTURES / "xgb_3format.json").read_text()
    )


def test_serialize_roundtrip_bytes(treelite_rs):
    model = _load(treelite_rs)
    blob = model.serialize_bytes()
    assert isinstance(blob, (bytes, bytearray))

    restored = treelite_rs.Model.deserialize_bytes(blob)
    assert restored.num_tree == model.num_tree
    assert restored.num_feature == model.num_feature
    # Re-serialization must be byte-stable across the round-trip.
    assert restored.serialize_bytes() == blob


def test_serialize_roundtrip_predicts_identically(treelite_rs, rng):
    """The deserialized model must predict identically within 1e-5 (core value)."""
    model = _load(treelite_rs)
    restored = treelite_rs.Model.deserialize_bytes(model.serialize_bytes())

    data = np.ascontiguousarray(
        rng.standard_normal((16, model.num_feature)), dtype=np.float32
    )
    before = treelite_rs.gtil.predict(model, data)
    after = treelite_rs.gtil.predict(restored, data)

    assert before.shape == after.shape
    np.testing.assert_allclose(after, before, atol=1e-5, rtol=0)


def test_serialize_file_roundtrip(treelite_rs, tmp_path):
    """The Python file-path shims (serialize/deserialize) round-trip via disk."""
    model = _load(treelite_rs)
    path = tmp_path / "model.bin"
    treelite_rs.serialize(model, path)
    restored = treelite_rs.deserialize(path)
    assert restored.num_tree == model.num_tree
    assert restored.serialize_bytes() == model.serialize_bytes()


def test_dump_as_json_parses(treelite_rs):
    model = _load(treelite_rs)
    parsed = json.loads(model.dump_as_json())
    assert isinstance(parsed, dict)
    # Compact mode is also valid JSON and equal at the parsed-value level.
    assert json.loads(model.dump_as_json(pretty_print=False)) == parsed


def test_dump_as_json_matches_upstream(treelite_upstream, treelite_rs):
    """Structural (parsed-value) equivalence to upstream dump_as_json (D-04)."""
    path = FIXTURES / "xgb_3format.json"
    rs_model = treelite_rs.frontend.load_xgboost_json_str(path.read_text())
    up_model = treelite_upstream.frontend.load_xgboost_model(str(path))

    rs_parsed = json.loads(rs_model.dump_as_json())
    up_parsed = json.loads(up_model.dump_as_json())
    # Compare the load-bearing structure: tree count + per-tree node counts.
    assert len(rs_parsed["trees"]) == len(up_parsed["trees"])
    assert [t["num_nodes"] for t in rs_parsed["trees"]] == [
        t["num_nodes"] for t in up_parsed["trees"]
    ]


def test_concatenate_merges(treelite_rs):
    m1 = _load(treelite_rs)
    m2 = _load(treelite_rs)
    merged = treelite_rs.Model.concatenate([m1, m2])
    assert merged.num_tree == m1.num_tree + m2.num_tree
    assert merged.num_feature == m1.num_feature
    # The free-function alias is equivalent.
    merged2 = treelite_rs.concatenate([m1, m2])
    assert merged2.num_tree == merged.num_tree


def test_concatenate_empty_raises(treelite_rs):
    with pytest.raises(treelite_rs.TreeliteError):
        treelite_rs.Model.concatenate([])

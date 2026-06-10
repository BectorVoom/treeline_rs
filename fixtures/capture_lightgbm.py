#!/usr/bin/env python3
"""One-time FROZEN LightGBM golden capture (Phase 4, D-06/D-07).

Run ONCE on the main worktree, then commit the resulting
``fixtures/lightgbm_*.golden.json`` + ``fixtures/lightgbm_categorical.txt``
read-only. CI NEVER regenerates these (same "run once, commit, never
regenerate" discipline as ``capture_golden_v5.py``).

Provenance / reproduction::

    # treelite==4.7.0 + numpy already in the repo venv; lightgbm is capture-only
    # and NEVER enters the Rust build graph / CI runtime (D-06):
    uv pip install scikit-learn lightgbm
    uv run python fixtures/capture_lightgbm.py

What it freezes:

  * NUMERICAL (LGB-01): the VENDORED ``treelite-mainline/tests/examples/
    deep_lightgbm/model.txt`` (32 leaves, single feature, no categorical -- the
    RESEARCH A2 numerical smoke source) is loaded via
    ``treelite.frontend.load_lightgbm_model``. The golden references that path
    (does NOT duplicate the model bytes) + freezes a seeded input matrix X and
    ``treelite.gtil.predict(model, X)``.
  * CATEGORICAL (LGB-02 bitset coverage): a small LightGBM model is fit WITH a
    categorical feature, its text form dumped to
    ``fixtures/lightgbm_categorical.txt``, then ``treelite.gtil.predict`` is
    frozen on a seeded X. The model is asserted at capture time to contain at
    least one categorical split (``num_cat > 0`` / a ``cat_threshold`` line) so
    the LGB-02 bitset decode has a real target.

For BOTH cases the committed golden prediction vector is
``treelite.gtil.predict(model, X)`` (D-07), NOT the LightGBM framework
``predict()`` (recorded only as a secondary ``framework_predict_sanity`` field).
Each manifest pins lightgbm / treelite / numpy versions + seed (D-06).
"""

import hashlib
import json
import os
import platform

import numpy as np

import treelite
import lightgbm

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1234

# Vendored numerical fixture (RESEARCH A2). Referenced, never duplicated.
VENDORED_NUMERICAL_REL = "treelite-mainline/tests/examples/deep_lightgbm/model.txt"
REPO_ROOT = os.path.dirname(HERE)
VENDORED_NUMERICAL_ABS = os.path.join(REPO_ROOT, VENDORED_NUMERICAL_REL)


def _manifest(extra=None):
    m = {
        "treelite": treelite.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "seed": SEED,
    }
    if extra:
        m.update(extra)
    return m


def _payload_sha256(input_list, output_list):
    blob = json.dumps(
        {"input": input_list, "output": output_list},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _write_golden(name, payload):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def _max_feature_idx(model_text):
    for line in model_text.splitlines():
        if line.startswith("max_feature_idx="):
            return int(line.split("=", 1)[1])
    raise RuntimeError("max_feature_idx not found in LightGBM text model")


def capture_numerical():
    """LGB-01: vendored deep_lightgbm/model.txt -> treelite-GTIL golden."""
    assert os.path.isfile(VENDORED_NUMERICAL_ABS), (
        f"vendored numerical fixture missing: {VENDORED_NUMERICAL_ABS}"
    )
    with open(VENDORED_NUMERICAL_ABS, "r", encoding="utf-8") as f:
        model_text = f.read()
    n_features = _max_feature_idx(model_text) + 1  # max_feature_idx=0 -> 1 feature

    model = treelite.frontend.load_lightgbm_model(VENDORED_NUMERICAL_ABS)

    rng = np.random.RandomState(SEED)
    # feature_infos=[0:100] in the vendored model -> sample inside that range.
    X = rng.uniform(0.0, 100.0, size=(20, n_features)).astype(np.float64)
    gtil_out = np.asarray(treelite.gtil.predict(model, X), dtype=np.float64)

    payload = {
        # Reference the vendored model by path -- do NOT copy its bytes here.
        "model_path": VENDORED_NUMERICAL_REL,
        "n_features": int(n_features),
        "input": X.tolist(),
        "output": gtil_out.reshape(-1).tolist(),
        "output_shape": list(gtil_out.shape),
        "manifest": _manifest({"variant": "numerical", "source": "vendored"}),
        "sha256": _payload_sha256(X.tolist(), gtil_out.reshape(-1).tolist()),
    }
    _write_golden("lightgbm_numerical.golden.json", payload)


def capture_categorical():
    """LGB-02: fresh categorical LightGBM model exercising a bitset split."""
    rng = np.random.RandomState(SEED)
    n = 200
    num0 = rng.rand(n).astype(np.float64)
    num1 = rng.rand(n).astype(np.float64)
    cat = rng.randint(0, 6, size=n).astype(np.float64)  # 6-level categorical
    X = np.column_stack([num0, num1, cat]).astype(np.float64)
    # Make the target strongly depend on the categorical level so LightGBM
    # actually builds categorical splits.
    y = (cat * 2.0 + num0 - num1).astype(np.float64)

    train = lightgbm.Dataset(
        X,
        label=y,
        categorical_feature=[2],
        free_raw_data=False,
    )
    params = {
        "objective": "regression",
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "min_data_per_group": 5,
        "cat_smooth": 1.0,
        "max_cat_to_onehot": 1,  # force bitset (non-onehot) categorical splits
        "seed": SEED,
        "deterministic": True,
        "verbose": -1,
    }
    booster = lightgbm.train(params, train, num_boost_round=10)

    txt_path = os.path.join(HERE, "lightgbm_categorical.txt")
    booster.save_model(txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        model_text = f.read()

    # Assert a real categorical split exists (LGB-02 bitset target).
    has_cat = ("\nnum_cat=" in model_text and not all(
        line.endswith("=0")
        for line in model_text.splitlines()
        if line.startswith("num_cat=")
    )) or "cat_threshold=" in model_text
    assert has_cat, (
        "fresh LightGBM model has no categorical split -- LGB-02 bitset decode "
        "has no target. Adjust params/data."
    )
    print("LightGBM categorical split present (LGB-02 bitset target OK)")

    model = treelite.frontend.load_lightgbm_model(txt_path)
    gtil_out = np.asarray(treelite.gtil.predict(model, X), dtype=np.float64)
    framework = np.asarray(booster.predict(X), dtype=np.float64)

    payload = {
        "model_path": "fixtures/lightgbm_categorical.txt",
        "n_features": 3,
        "categorical_features": [2],
        "input": X.tolist(),
        "output": gtil_out.reshape(-1).tolist(),
        "output_shape": list(gtil_out.shape),
        "framework_predict_sanity": framework.reshape(-1).tolist(),
        "manifest": _manifest({"variant": "categorical", "source": "fresh"}),
        "sha256": _payload_sha256(X.tolist(), gtil_out.reshape(-1).tolist()),
    }
    _write_golden("lightgbm_categorical.golden.json", payload)


def main():
    print("=== treelite.gtil.predict signature ===")
    import inspect

    sig = inspect.signature(treelite.gtil.predict)
    assert sig.parameters["pred_margin"].default is False, (
        "default pred_margin changed -- goldens assume post-processed output"
    )
    print(sig)

    capture_numerical()
    capture_categorical()
    print("All LightGBM goldens captured (treelite-GTIL, D-07).")


if __name__ == "__main__":
    main()

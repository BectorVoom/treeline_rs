#!/usr/bin/env python3
"""One-time FROZEN capture of the two GTIL-matrix MODEL artifacts (Phase 5).

`capture_gtil_matrix.py` (Plan 05-01) authored two representative models
in-script (a binary:logistic XGBoost model and a 4-class multi:softprob
leaf-vector-broadcast model), captured their upstream `treelite.gtil.*`
predictions into the frozen `fixtures/gtil/*.golden.json` matrix, and then
*discarded the models* — its `model_path` recorded only the sentinel
`fixtures/gtil/<model>.captured-in-script` (no model file on disk).

The Plan 05-05 Rust equivalence runner (`tests/gtil_matrix.rs`) must load an
actual `treelite_core::Model` to predict against those goldens. This script
re-authors the SAME two seeded models (identical seed/hyperparameters to
`capture_gtil_matrix.py:build_binary_model`/`build_leaf_vector_model`) and
freezes each as the exact treelite v5 byte stream
(`treelite.Model.serialize_bytes()`) at:

    fixtures/gtil/binary.model.bin
    fixtures/gtil/leaf_vec_mc.model.bin

These bytes are the SAME model object that produced the frozen golden vectors
(verified at capture time: re-serialized model reproduces every golden cell to
max |delta| == 0.0 in Python). Loading them via `treelite_core::deserialize`
gives the Rust runner the precise model the goldens were captured from — with
NO xgboost-frontend re-derivation drift.

Run ONCE on the main worktree (the `uv` venv lives only here, per MEMORY.md):

    uv run python fixtures/capture_gtil_models.py

The committed `*.model.bin` bytes are the frozen contract; CI never
regenerates them. The frozen `*.golden.json` matrices (D-08) are NOT touched by
this script.
"""

import os

import numpy as np

import treelite
import xgboost  # capture-only pin; never in the Rust build graph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "gtil")
# The seed `capture_gtil_matrix.py:main` uses to AUTHOR each model (SEEDS[0]).
MODEL_SEED = 1234


def build_binary_model(seed):
    """XGBoost binary:logistic — identical to capture_gtil_matrix.py."""
    rng = np.random.RandomState(seed)
    n, n_feat = 160, 4
    X = rng.uniform(-3.0, 3.0, size=(n, n_feat)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0.0).astype(np.float32)
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 4, "seed": seed, "eta": 0.3},
        xgboost.DMatrix(X, label=y),
        num_boost_round=6,
    )
    return treelite.frontend.from_xgboost(booster)


def build_leaf_vector_model(seed):
    """4-class multi:softprob leaf-vector model — identical to capture_gtil_matrix.py."""
    rng = np.random.RandomState(seed)
    n, n_feat, n_class = 180, 5, 4
    X = rng.uniform(-4.0, 4.0, size=(n, n_feat)).astype(np.float32)
    score = X[:, 0] * 1.5 - X[:, 1] + X[:, 3] * 0.7
    y = np.clip(
        ((score - score.min()) / (np.ptp(score) + 1e-9) * n_class), 0, n_class - 1
    ).astype(int)
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": n_class,
            "max_depth": 4,
            "seed": seed,
            "eta": 0.3,
        },
        xgboost.DMatrix(X, label=y),
        num_boost_round=6,
    )
    return treelite.frontend.from_xgboost(booster)


def _freeze(name, model):
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = model.serialize_bytes()
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"Wrote {path} ({len(raw)} bytes)")


def main():
    _freeze("binary.model.bin", build_binary_model(MODEL_SEED))
    _freeze("leaf_vec_mc.model.bin", build_leaf_vector_model(MODEL_SEED))
    print("GTIL matrix model artifacts frozen (treelite v5 serialize_bytes).")


if __name__ == "__main__":
    main()

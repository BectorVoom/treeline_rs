#!/usr/bin/env python3
"""One-time FROZEN scikit-learn golden capture (Phase 4, D-06/D-07).

Run ONCE on the main worktree (the venv/pyproject live only here), then commit
the resulting ``fixtures/sklearn_*.golden.json`` files read-only. CI NEVER
regenerates these (mirrors ``capture_golden_v5.py`` "run once, commit, never
regenerate" discipline).

Provenance / reproduction::

    # treelite==4.7.0 + numpy already in the repo venv; sklearn + lightgbm are
    # capture-only and NEVER enter the Rust build graph / CI runtime (D-06):
    uv pip install scikit-learn lightgbm
    uv run python fixtures/capture_sklearn.py

What it freezes, per estimator family, with a fixed seed:

  * RandomForest + ExtraTrees (classifier + regressor) -> node arrays in the
    ``importer.py`` dtype contract (children_left/children_right/feature i64;
    threshold/value/weighted_n_node_samples/impurity f64; n_node_samples i64).
  * GradientBoosting (classifier + regressor) -> arrays with the leaf-shrink
    ALREADY applied capture-side (``value * learning_rate``, importer.py:220-223).
    The Rust loader MUST NOT re-shrink (it consumes these pre-shrunk values).
  * IsolationForest -> node arrays + isolation depths in ``value`` +
    ``ratio_c = expected_depth(max_samples_)`` + subsampled feature indices.
    The committed golden is cross-checked at capture time against
    ``-clf.score_samples(X)`` (D-07 IsolationForest == "Treelite != framework").
  * HistGradientBoosting numerical-only (features_map == identity arange,
    Pitfall 4) AND a separate categorical HistGB (exercising categories_map /
    features_map) -> the packed ``nodes`` byte buffer (base64) +
    ``expected_sizeof_node_struct`` (nodes.itemsize in {52, 56}) + features_map
    + categories_map + known_cat_bitsets + raw_left_cat_bitsets.

For EVERY family the committed golden prediction vector is
``treelite.gtil.predict(model, X)`` on the captured input matrix X (D-07) -- NOT
the framework's ``predict()``. The framework predict is recorded only as a
secondary ``framework_predict_sanity`` field, never the asserted target.

Each golden carries a manifest block pinning sklearn / lightgbm / treelite /
numpy versions + the seed (D-06), plus a sha256 of the (input, output) payload.
"""

import base64
import hashlib
import json
import os
import platform
import struct

import numpy as np

import treelite
import treelite.sklearn
from treelite.sklearn.isolation_forest import calculate_depths, expected_depth

import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)

import lightgbm  # capture-env pin only (recorded in manifest)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1234


def _manifest(extra=None):
    m = {
        "treelite": treelite.__version__,
        "sklearn": sklearn.__version__,
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


def _dump_node_arrays(sklearn_model, *, is_gb=False, is_iforest=False):
    """Replicate importer.py's per-tree ArrayOfArrays extraction (dtype contract).

    Returns a JSON-safe dict of per-tree column lists. The Rust sklearn loader
    consumes exactly these columns; the GB leaf-shrink is applied here
    (capture-side) so the loader must NOT re-shrink (importer.py:218-223).
    """
    trees = []
    estimators_features = getattr(sklearn_model, "estimators_features_", None)
    for tree_idx, estimator in enumerate(sklearn_model.estimators_):
        if is_gb:
            estimator_range = list(estimator)  # GB: one sub-estimator per class
            learning_rate = sklearn_model.learning_rate
        else:
            estimator_range = [estimator]
            learning_rate = 1.0
        if is_iforest:
            isolation_depths = np.zeros(
                estimator.tree_.n_node_samples.shape[0], dtype="float64"
            )
            calculate_depths(isolation_depths, estimator.tree_, 0, 0.0)
        for sub_estimator in estimator_range:
            tree = sub_estimator.tree_
            children_left = tree.children_left.astype(np.int64)
            children_right = tree.children_right.astype(np.int64)
            threshold = tree.threshold.astype(np.float64)
            n_node_samples = tree.n_node_samples.astype(np.int64)
            weighted = tree.weighted_n_node_samples.astype(np.float64)
            impurity = tree.impurity.astype(np.float64)
            if is_iforest:
                value = isolation_depths.reshape((-1, 1, 1)).astype(np.float64)
                # importer.py:208-212 -- subsample feature index remap
                feature = np.full(tree.feature.shape, -2, dtype=np.int64)
                mask = tree.feature != -2
                feature[mask] = np.array(estimators_features[tree_idx])[
                    tree.feature[mask]
                ]
            else:
                # GB: shrink leaf output by learning_rate CAPTURE-SIDE.
                value = (tree.value * learning_rate).astype(np.float64)
                feature = tree.feature.astype(np.int64)
            trees.append(
                {
                    "node_count": int(tree.node_count),
                    "children_left": children_left.tolist(),
                    "children_right": children_right.tolist(),
                    "feature": feature.tolist(),
                    "threshold": threshold.tolist(),
                    "value": value.reshape(-1).tolist(),
                    "value_shape": list(value.shape),
                    "n_node_samples": n_node_samples.tolist(),
                    "weighted_n_node_samples": weighted.tolist(),
                    "impurity": impurity.tolist(),
                }
            )
    return trees


def _write_golden(name, payload):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def capture_rf_et():
    """RandomForest + ExtraTrees, classifier + regressor (SKL-01)."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(40, 4).astype(np.float64)
    y_reg = rng.rand(40).astype(np.float64)
    y_clf = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    estimators = {
        "rf_regressor": RandomForestRegressor(
            n_estimators=5, max_depth=4, random_state=SEED
        ).fit(X, y_reg),
        "et_regressor": ExtraTreesRegressor(
            n_estimators=5, max_depth=4, random_state=SEED
        ).fit(X, y_reg),
        "rf_classifier": RandomForestClassifier(
            n_estimators=5, max_depth=4, random_state=SEED
        ).fit(X, y_clf),
        "et_classifier": ExtraTreesClassifier(
            n_estimators=5, max_depth=4, random_state=SEED
        ).fit(X, y_clf),
    }

    families = {}
    for key, clf in estimators.items():
        tl_model = treelite.sklearn.import_model(clf)
        gtil_out = np.asarray(treelite.gtil.predict(tl_model, X), dtype=np.float64)
        families[key] = {
            "n_estimators": int(clf.n_estimators),
            "n_features_in": int(clf.n_features_in_),
            "n_outputs": int(clf.n_outputs_),
            "n_classes": (
                np.asarray(clf.n_classes_).tolist()
                if hasattr(clf, "n_classes_")
                else None
            ),
            "trees": _dump_node_arrays(clf),
            "output": gtil_out.reshape(-1).tolist(),
            "output_shape": list(gtil_out.shape),
            "framework_predict_sanity": np.asarray(
                clf.predict(X), dtype=np.float64
            ).reshape(-1).tolist(),
        }

    flat_out = families["rf_regressor"]["output"]
    payload = {
        "input": X.tolist(),
        "families": families,
        "manifest": _manifest(),
        "sha256": _payload_sha256(X.tolist(), flat_out),
    }
    _write_golden("sklearn_rf.golden.json", payload)


def capture_gb():
    """GradientBoosting classifier + regressor (SKL-02), leaf-shrink capture-side."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(40, 4).astype(np.float64)
    y_reg = rng.rand(40).astype(np.float64)
    y_clf = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    gb_reg = GradientBoostingRegressor(
        n_estimators=5, max_depth=3, learning_rate=0.1, random_state=SEED
    ).fit(X, y_reg)
    gb_clf = GradientBoostingClassifier(
        n_estimators=5, max_depth=3, learning_rate=0.1, random_state=SEED
    ).fit(X, y_clf)

    families = {}
    for key, clf in {"gb_regressor": gb_reg, "gb_classifier": gb_clf}.items():
        tl_model = treelite.sklearn.import_model(clf)
        gtil_out = np.asarray(treelite.gtil.predict(tl_model, X), dtype=np.float64)
        families[key] = {
            "n_estimators": int(clf.n_estimators),
            "n_features_in": int(clf.n_features_in_),
            "learning_rate": float(clf.learning_rate),
            # NOTE: trees[].value already has *learning_rate applied capture-side
            # (importer.py:220-223). The Rust loader must NOT re-shrink.
            "leaf_shrink_applied_capture_side": True,
            "trees": _dump_node_arrays(clf, is_gb=True),
            "output": gtil_out.reshape(-1).tolist(),
            "output_shape": list(gtil_out.shape),
            "framework_predict_sanity": np.asarray(
                clf.predict(X), dtype=np.float64
            ).reshape(-1).tolist(),
        }

    flat_out = families["gb_regressor"]["output"]
    payload = {
        "input": X.tolist(),
        "families": families,
        "manifest": _manifest(),
        "sha256": _payload_sha256(X.tolist(), flat_out),
    }
    _write_golden("sklearn_gb.golden.json", payload)


def capture_iforest():
    """IsolationForest (SKL-03). Golden == -clf.score_samples(X) (D-07 cross-check)."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(50, 4).astype(np.float64)

    clf = IsolationForest(
        n_estimators=5, max_samples=32, random_state=SEED
    ).fit(X)
    ratio_c = expected_depth(clf.max_samples_)

    tl_model = treelite.sklearn.import_model(clf)
    gtil_out = np.asarray(treelite.gtil.predict(tl_model, X), dtype=np.float64)
    neg_score = (-clf.score_samples(X)).astype(np.float64)

    # D-07 capture-time cross-check: the Treelite GTIL golden MUST equal
    # -score_samples within 1e-5 (the canonical "Treelite != framework" case;
    # framework decision_function/predict are something else entirely).
    max_delta = float(np.max(np.abs(gtil_out.reshape(-1) - neg_score.reshape(-1))))
    assert max_delta < 1e-5, (
        f"IsolationForest golden != -score_samples (max |delta|={max_delta}); "
        "D-07 cross-check failed -- do NOT commit."
    )
    print(f"IsolationForest D-07 cross-check OK (max |delta| = {max_delta:g})")

    payload = {
        "input": X.tolist(),
        "n_estimators": int(clf.n_estimators),
        "n_features_in": int(clf.n_features_in_),
        "max_samples": int(clf.max_samples_),
        "ratio_c": float(ratio_c),
        "trees": _dump_node_arrays(clf, is_iforest=True),
        "output": gtil_out.reshape(-1).tolist(),
        "output_shape": list(gtil_out.shape),
        "neg_score_samples_cross_check": neg_score.reshape(-1).tolist(),
        "cross_check_max_delta": max_delta,
        "manifest": _manifest(),
        "sha256": _payload_sha256(X.tolist(), gtil_out.reshape(-1).tolist()),
    }
    _write_golden("sklearn_iforest.golden.json", payload)


def _dump_histgb(sklearn_model):
    """Replicate importer.py:_import_hist_gradient_boosting capture-side dumps."""
    known_cat_bitsets, f_idx_map = (
        sklearn_model._bin_mapper.make_known_categories_bitsets()
    )

    # features_map (feat_remapper) -- identity arange when no categorical
    # preprocessor, else the embedded-OrdinalEncoder remap (importer.py:371-410).
    cat_remapper = []
    preprocessor = getattr(sklearn_model, "_preprocessor", None)
    from packaging.version import parse as parse_version

    if (
        parse_version(sklearn.__version__) >= parse_version("1.4.0")
        and preprocessor is not None
    ):
        for cats in preprocessor.transformers_[0][1].categories_:
            if cats.dtype.type is np.str_:
                raise NotImplementedError("String categories not supported")
            cat_remapper.append(cats.astype(np.int64).tolist())
        feat_remapper = np.zeros((sklearn_model.n_features_in_,), dtype=np.int32)
        n_categorical = int(sklearn_model.is_categorical_.sum())
        cat_idx, num_idx = 0, 0
        for i, e in enumerate(sklearn_model.is_categorical_):
            if e:
                feat_remapper[cat_idx] = i
                cat_idx += 1
            else:
                feat_remapper[n_categorical + num_idx] = i
                num_idx += 1
    else:
        feat_remapper = np.arange(
            start=0, stop=sklearn_model.n_features_in_, dtype=np.int32
        )

    nodes_b64 = []
    raw_left_cat_bitsets = []
    node_count = []
    itemsize = None
    for estimator in sklearn_model._predictors:
        for sub_estimator in estimator:
            buf = bytes(sub_estimator.nodes.tobytes())
            nodes_b64.append(base64.b64encode(buf).decode("ascii"))
            raw_left_cat_bitsets.append(
                np.asarray(sub_estimator.raw_left_cat_bitsets, dtype=np.uint32)
                .reshape(-1)
                .tolist()
            )
            node_count.append(int(len(sub_estimator.nodes)))
            if itemsize is None:
                itemsize = int(sub_estimator.nodes.itemsize)
            elif itemsize != int(sub_estimator.nodes.itemsize):
                raise RuntimeError("itemsize mismatch")
    assert itemsize in (52, 56), f"unexpected HistGB node itemsize {itemsize}"

    return {
        "n_iter": int(sklearn_model.n_iter_),
        "n_features_in": int(sklearn_model.n_features_in_),
        "expected_sizeof_node_struct": itemsize,
        "node_count": node_count,
        "nodes_b64": nodes_b64,
        "n_categorical_splits": int(known_cat_bitsets.shape[0]),
        "raw_left_cat_bitsets": raw_left_cat_bitsets,
        "known_cat_bitsets": np.asarray(known_cat_bitsets, dtype=np.uint32)
        .reshape(-1)
        .tolist(),
        "f_idx_map": np.asarray(f_idx_map, dtype=np.uint32).reshape(-1).tolist(),
        "features_map": feat_remapper.astype(np.int32).reshape(-1).tolist(),
        "categories_map": cat_remapper,
        "baseline_prediction": np.asarray(
            sklearn_model._baseline_prediction, dtype=np.float64
        )
        .reshape(-1)
        .tolist(),
    }


def capture_histgb_numerical():
    """HistGB numerical-only (SKL-04): features_map == identity arange (Pitfall 4)."""
    rng = np.random.RandomState(SEED)
    X = rng.rand(60, 4).astype(np.float64)
    y = rng.rand(60).astype(np.float64)

    clf = HistGradientBoostingRegressor(
        max_iter=5, max_depth=3, random_state=SEED
    ).fit(X, y)

    histgb = _dump_histgb(clf)
    # Numerical-only: feature remap must be the identity arange (Pitfall 4).
    assert histgb["features_map"] == list(range(clf.n_features_in_)), (
        "numerical HistGB features_map is not identity -- not a clean "
        "non-categorical fixture (Pitfall 4)."
    )
    assert histgb["categories_map"] == [], "numerical HistGB has categories_map"

    tl_model = treelite.sklearn.import_model(clf)
    gtil_out = np.asarray(treelite.gtil.predict(tl_model, X), dtype=np.float64)

    payload = {
        "input": X.tolist(),
        "histgb": histgb,
        "output": gtil_out.reshape(-1).tolist(),
        "output_shape": list(gtil_out.shape),
        "framework_predict_sanity": np.asarray(
            clf.predict(X), dtype=np.float64
        ).reshape(-1).tolist(),
        "manifest": _manifest({"variant": "numerical", "features_map": "identity"}),
        "sha256": _payload_sha256(X.tolist(), gtil_out.reshape(-1).tolist()),
    }
    _write_golden("sklearn_histgb_numerical.golden.json", payload)


def capture_histgb_categorical():
    """HistGB categorical (SKL-04): exercises categories_map / features_map (Pitfall 4)."""
    rng = np.random.RandomState(SEED)
    n = 80
    num_feat = rng.rand(n, 2).astype(np.float64)
    cat_feat = rng.randint(0, 4, size=(n, 1)).astype(np.float64)
    X = np.hstack([num_feat, cat_feat]).astype(np.float64)
    y = (cat_feat.reshape(-1) + num_feat[:, 0]).astype(np.float64)

    is_categorical = [False, False, True]
    clf = HistGradientBoostingRegressor(
        max_iter=5,
        max_depth=3,
        categorical_features=is_categorical,
        random_state=SEED,
    ).fit(X, y)

    histgb = _dump_histgb(clf)
    assert histgb["n_categorical_splits"] >= 0
    # Categorical fixture MUST carry a categories_map (the embedded OrdinalEncoder
    # remap path, importer.py:391-410). This is the Pitfall-4 categorical half.
    assert histgb["categories_map"], (
        "categorical HistGB carries no categories_map -- categorical preprocessor "
        "path not exercised (Pitfall 4)."
    )

    tl_model = treelite.sklearn.import_model(clf)
    gtil_out = np.asarray(treelite.gtil.predict(tl_model, X), dtype=np.float64)

    payload = {
        "input": X.tolist(),
        "is_categorical": is_categorical,
        "histgb": histgb,
        "output": gtil_out.reshape(-1).tolist(),
        "output_shape": list(gtil_out.shape),
        "framework_predict_sanity": np.asarray(
            clf.predict(X), dtype=np.float64
        ).reshape(-1).tolist(),
        "manifest": _manifest({"variant": "categorical"}),
        "sha256": _payload_sha256(X.tolist(), gtil_out.reshape(-1).tolist()),
    }
    _write_golden("sklearn_histgb_categorical.golden.json", payload)


def main():
    # Confirm the GTIL default predict kind once (as Phase 1 did) and assert the
    # golden is the DEFAULT (post-processed, not pred_margin) output.
    print("=== treelite.gtil.predict signature ===")
    import inspect

    sig = inspect.signature(treelite.gtil.predict)
    assert "pred_margin" in sig.parameters, "gtil.predict lost pred_margin kw"
    assert sig.parameters["pred_margin"].default is False, (
        "default pred_margin changed -- goldens assume post-processed output"
    )
    print(sig)

    capture_rf_et()
    capture_gb()
    capture_iforest()
    capture_histgb_numerical()
    capture_histgb_categorical()
    print("All sklearn goldens captured (treelite-GTIL, D-07).")


if __name__ == "__main__":
    main()

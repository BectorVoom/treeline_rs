#!/usr/bin/env python3
"""One-time FROZEN exhaustive GTIL equivalence-matrix capture (Phase 5).

Run ONCE on the main worktree (the ``uv`` venv/pyproject live ONLY here and are
absent from worktrees, per MEMORY.md), then commit the resulting
``fixtures/gtil/*.golden.json`` files read-only. CI NEVER regenerates these and
NEVER re-draws from a seed: the committed input matrices + golden output vectors
are the frozen contract (D-08). The seed + this script are committed only as a
regeneration aid.

Provenance / reproduction::

    # treelite==4.7.0 + numpy + scipy already in the repo venv; xgboost / sklearn
    # are capture-only and NEVER enter the Rust build graph / CI runtime (D-06):
    uv run python fixtures/capture_gtil_matrix.py

What it freezes (the exhaustive cross-product, D-01/D-03 — "we tested
everything", no invariance pruning):

  model (one representative per capability axis, D-02 / RESEARCH Open Q2)
    x preset (the model's own <f32,f32> or <f64,f64> — fixed per model)
    x input dtype  (f32 and f64, D-05 — orthogonal to preset)
    x predict kind (default / raw / leaf_id / score_per_tree, GTIL-03)
    x {dense, sparse CSR} (D-04 parity)
    x seed  ({1234, 5678}, D-02 few seeds, wide edge-seeded matrices)

Each cell is one committed ``fixtures/gtil/<model>.<preset>.<indtype>.<kind>.
<dense|sparse>.s<seed>.golden.json`` with the payload shape
``{model_path, n_features, input, output, output_shape, manifest, sha256}``
mirroring ``fixtures/lightgbm_categorical.golden.json``.

The asserted golden is ALWAYS ``treelite.gtil.*`` (upstream Treelite GTIL, the
C++ truth), NOT the source framework's ``predict()`` (D-07).

### Edge-seeded wide matrices (RESEARCH Code Examples / Pitfall 3)

Each input matrix is wide (~120-200 rows) and deliberately injected with edge
values: ``np.nan`` (missing -> default-direction routing, GTIL-05),
``+/-np.inf``, boundary thresholds, and ``2**24 + 1`` placed in a categorical
feature column — the f32 representability-gap value that exercises the FULL
categorical guard (GTIL-06, Pitfall 3): ``2**24 + 1`` is NOT exactly
representable in f32 and sits in the gap between the f32 ``max_repr = 2**24`` and
``u32::MAX``, so it must route as a non-match.

### Sparse-with-NaN construction (RESEARCH Open Q1 / D-04)

absent CSR entries materialize as NaN (NOT 0) inside treelite's
``SparseMatrixAccessor`` (``predict.cc:81``). To make dense and sparse predict
identically on identical logical data we:

  1. build a presence mask,
  2. build a DENSE matrix with ``np.nan`` in absent positions,
  3. build a ``scipy.sparse.csr_matrix`` from ONLY the present positions,
  4. ASSERT at capture time ``treelite.gtil.predict(dense_with_nan) ==
     treelite.gtil.predict(csr)`` with ``equal_nan=True`` BEFORE freezing both
     goldens (the canonical D-04 construction).

The unconditional multiclass leaf-vector-broadcast model (GTIL-07, D-03) is
authored fresh in-script on EVERY run (see ``build_leaf_vector_model``) so the
leaf-vector axis always has at least one committed golden — never gated behind a
"if no vendored example" branch.
"""

import hashlib
import json
import os
import platform

import numpy as np
import scipy.sparse

import treelite
import xgboost  # capture-only pin (recorded in manifest); never in the Rust build graph
import lightgbm  # capture-only pin (kLE numerical model class, CR-01); never in the Rust build graph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "gtil")
SEEDS = [1234, 5678]

# Predict kinds (GTIL-03). The value is the fixture-name token; the captured
# vector comes from the matching treelite.gtil.* entry point.
KINDS = ["default", "raw", "leaf_id", "score_per_tree"]


def _manifest(model_name, preset, indtype, kind, layout, seed, extra=None):
    """Full-provenance manifest with the D-09 ``backend`` field.

    Records OS/arch, every capture framework version, the seed, rustc, and a
    ``cubecl`` placeholder ("n/a" this phase, D-09). ``backend`` is ALWAYS
    ``scalar-cpu`` for Phase 5 — the plain-Rust reference every later backend is
    measured against.
    """
    m = {
        "backend": "scalar-cpu",
        "treelite": treelite.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "rustc": _rustc_version(),
        "cubecl": "n/a",  # forward-only placeholder (D-09); no cubecl this phase
        "model": model_name,
        "preset": preset,
        "input_dtype": indtype,
        "kind": kind,
        "layout": layout,
        "seed": seed,
    }
    if extra:
        m.update(extra)
    return m


def _rustc_version():
    """Best-effort rustc version string for the manifest (provenance only)."""
    import subprocess

    try:
        out = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _payload_sha256(input_list, output_list):
    blob = json.dumps(
        {"input": input_list, "output": output_list},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _write_golden(name, payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def _json_safe(arr):
    """Convert a numpy array to a JSON-safe nested list.

    Non-finite values (NaN, +/-inf) are written as the JSON tokens the harness
    normalizes on read (the existing ``golden.json`` discipline: bare ``NaN`` ->
    ``null`` mapping). We emit ``None`` for NaN and the strings ``"inf"`` /
    ``"-inf"`` for infinities so the committed JSON is valid and round-trippable.
    """
    out = []
    flat = arr.reshape(-1)
    for v in flat:
        fv = float(v)
        if np.isnan(fv):
            out.append(None)
        elif np.isposinf(fv):
            out.append("inf")
        elif np.isneginf(fv):
            out.append("-inf")
        else:
            out.append(fv)
    return np.asarray(out, dtype=object).reshape(arr.shape).tolist()


def _gtil_capture(model, kind, X):
    """Run the upstream treelite.gtil.* entry point for ``kind`` (GTIL-03)."""
    if kind == "default":
        out = treelite.gtil.predict(model, X)
    elif kind == "raw":
        out = treelite.gtil.predict(model, X, pred_margin=True)
    elif kind == "leaf_id":
        out = treelite.gtil.predict_leaf(model, X)
    elif kind == "score_per_tree":
        out = treelite.gtil.predict_per_tree(model, X)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return np.asarray(out, dtype=np.float64)


def _build_edge_matrix(seed, n_rows, n_feat, cat_col):
    """A wide RandomState matrix injected with edge values (Pitfall 3).

    Returns the dense matrix in f64 (callers cast to the capture dtype). Edge
    values: NaN (missing), +/-inf, boundary thresholds, and ``2**24 + 1`` in the
    categorical column (the f32 representability-gap value).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-5.0, 5.0, size=(n_rows, n_feat)).astype(np.float64)
    # Edge injections (spread across distinct rows so each path is exercised).
    X[0, 0] = np.nan  # missing -> default direction (GTIL-05)
    X[1, min(1, n_feat - 1)] = np.inf
    X[2, min(2, n_feat - 1)] = -np.inf
    X[3, 0] = 0.0  # boundary threshold
    # The f32 categorical-gap value (Pitfall 3): 2**24 + 1 is NOT representable
    # in f32 and sits in the [2**24, u32::MAX] gap of the f32 categorical guard.
    X[4, cat_col] = float(2**24 + 1)
    if n_rows > 5:
        X[5, cat_col] = np.nan
    return X


def _present_mask(seed, n_rows, n_feat):
    """A deterministic ~60%-dense presence mask for the sparse construction."""
    rng = np.random.RandomState(seed + 9999)
    return rng.rand(n_rows, n_feat) < 0.6


def _dense_and_csr(X_dense_full, dtype, seed):
    """Build the D-04 dense-with-NaN matrix + matching CSR from a presence mask.

    The dense matrix carries ``NaN`` in absent positions; the CSR is built from
    ONLY the present positions, so treelite materializes ``NaN`` for exactly the
    absent columns (predict.cc:81). Returns (dense_nan, csr, csr_triple) in
    ``dtype``, where ``csr_triple`` is the REAL frozen CSR
    ``(data, indices, indptr)`` (WR-01) — the exact arrays the Rust sparse path
    must consume verbatim instead of re-deriving a CSR from NaN-presence.
    """
    n_rows, n_feat = X_dense_full.shape
    mask = _present_mask(seed, n_rows, n_feat)
    dense_nan = np.where(mask, X_dense_full, np.nan).astype(dtype)
    # CSR from present positions only. scipy stores the present VALUES; absent
    # positions are implicit and become NaN inside treelite's accessor.
    present_vals = np.where(mask, X_dense_full, 0.0).astype(dtype)
    csr = scipy.sparse.csr_matrix(present_vals)
    # Drop any present-but-zero explicit entries so the present set == mask.
    csr.data = present_vals[mask].astype(dtype)
    # Rebuild indices/indptr from the mask to guarantee present == mask exactly.
    indices = []
    indptr = [0]
    data = []
    for r in range(n_rows):
        cols = np.nonzero(mask[r])[0]
        indices.extend(cols.tolist())
        data.extend(X_dense_full[r, cols].astype(dtype).tolist())
        indptr.append(len(indices))
    data_arr = np.asarray(data, dtype=dtype)
    indices_arr = np.asarray(indices, dtype=np.int64)
    indptr_arr = np.asarray(indptr, dtype=np.int64)
    csr = scipy.sparse.csr_matrix(
        (data_arr, indices_arr, indptr_arr),
        shape=(n_rows, n_feat),
    )
    # WR-01: the REAL captured CSR triple (data/indices/indptr) — frozen into the
    # sparse golden so the Rust runner loads it verbatim, never reconstructing a
    # CSR from NaN-presence (which can mistake a present NaN for absent).
    csr_triple = (data_arr, indices_arr, indptr_arr)
    return dense_nan, csr, csr_triple


# --------------------------------------------------------------------------- #
# Model authoring (one representative per capability axis, D-02 / Open Q2)
# --------------------------------------------------------------------------- #

def build_binary_model(seed):
    """XGBoost binary:logistic — the scalar binary axis (reuses the corpus shape)."""
    rng = np.random.RandomState(seed)
    n, n_feat = 160, 4
    X = rng.uniform(-3.0, 3.0, size=(n, n_feat)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0.0).astype(np.float32)
    dtrain = xgboost.DMatrix(X, label=y)
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 4, "seed": seed, "eta": 0.3},
        dtrain,
        num_boost_round=6,
    )
    model = treelite.frontend.from_xgboost(booster)
    return model, n_feat, 2  # cat_col index for edge injection


def build_leaf_vector_model(seed):
    """UNCONDITIONAL multiclass leaf-vector-broadcast model (GTIL-07, D-03).

    Authored fresh on EVERY run — NOT gated behind any "if no vendored example"
    branch — so the leaf-vector-broadcast axis ALWAYS has >= 1 committed golden
    and the 05-05 runner can never silently skip it. A 4-class
    ``multi:softprob`` XGBoost model emits a per-class leaf vector that is
    broadcast across the output targets (the OutputLeafVector path at
    predict.cc:174-216).
    """
    rng = np.random.RandomState(seed)
    n, n_feat, n_class = 180, 5, 4
    X = rng.uniform(-4.0, 4.0, size=(n, n_feat)).astype(np.float32)
    # Class label depends on several features so trees actually split.
    score = X[:, 0] * 1.5 - X[:, 1] + X[:, 3] * 0.7
    y = np.clip(
        ((score - score.min()) / (np.ptp(score) + 1e-9) * n_class), 0, n_class - 1
    )
    y = y.astype(int)
    dtrain = xgboost.DMatrix(X, label=y)
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": n_class,
            "max_depth": 4,
            "seed": seed,
            "eta": 0.3,
        },
        dtrain,
        num_boost_round=6,
    )
    model = treelite.frontend.from_xgboost(booster)
    return model, n_feat, 2  # cat_col index for edge injection


def build_large_margin_model(seed):
    """LARGE-MARGIN XGBoost binary:logistic — the CR-01 ``sigmoid`` stressor.

    Authored to drive prediction margins WELL past ±10 (deep trees + ``eta=1.0``
    + many rounds over a cleanly-separable target), the regime where the f64 and
    f32 ``sigmoid`` (``1/(1+exp(-x))``) differ by ~1e-7 (empirically ~6e-8 on
    ±20 margins). That divergence sits INSIDE the 1e-5 gate — exactly the band
    that masked CR-01 (the pre-05-06 engine narrowed the f64 buffer to f32 before
    the postprocessor). The committed ``f64`` cells of this model are captured
    from upstream ``treelite.gtil.predict`` which runs ``ApplyPostProcessor<double>``
    (f64 sigmoid); the Rust engine must reproduce them via its ``sigmoid_f64``
    twin (05-06). A silent collapse to the f32 path would deviate from THIS
    golden by ~6e-8 — measured, not absorbed (GTIL-04, EQV-03/04).
    """
    rng = np.random.RandomState(seed)
    n, n_feat = 200, 4
    X = rng.uniform(-3.0, 3.0, size=(n, n_feat)).astype(np.float32)
    # Cleanly separable target so the booster can drive scores to large margins.
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0.0).astype(np.float32)
    dtrain = xgboost.DMatrix(X, label=y)
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "max_depth": 6,
            "seed": seed,
            "eta": 1.0,
            "lambda": 0.0,
            "min_child_weight": 0.0,
        },
        dtrain,
        num_boost_round=20,
    )
    model = treelite.frontend.from_xgboost(booster)
    return model, n_feat, 2  # cat_col index for edge injection


def build_lgbm_numerical_model(seed):
    """NUMERICAL-only LightGBM regressor -> the CR-01 kLE model class.

    LightGBM emits ``Operator::kLE`` on EVERY numerical split (lightgbm.cc:585,
    mirrored in crates/treelite-lightgbm/src/lib.rs:273), and the
    ``treelite.frontend.load_lightgbm_model`` frontend yields the <f64,f64>
    preset. This is precisely the model class CR-01's operator-coverage fallback
    gate routes WHOLE to the scalar reference (the cubecl kernel implements only
    kLT). Before 06-06 such a model silently reached the kLT-hardcoded kernel and
    mis-routed every ``fv == threshold`` tie; this fixture proves the fallback
    gate now produces upstream-correct predictions on the kLE class.

    A handful of leaves/rounds is enough to exercise multiple kLE splits. NO
    categorical_feature is set, so every split is a numerical kLE test (never a
    bitset) — the matrix cell routes through the CR-01 fallback gate, not the
    categorical gate. Returns (model, n_feat, cat_col) like the other authors.
    """
    import lightgbm
    import tempfile

    rng = np.random.RandomState(seed)
    n, n_feat = 200, 4
    X = rng.uniform(0.0, 5.0, size=(n, n_feat)).astype(np.float64)
    # Target depends on several features so the booster builds real kLE splits.
    y = (X[:, 0] * 1.5 - X[:, 1] + X[:, 2] * 0.7 - X[:, 3]).astype(np.float64)
    train = lightgbm.Dataset(X, label=y, free_raw_data=False)
    params = {
        "objective": "regression",
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "seed": seed,
        "deterministic": True,
        "verbose": -1,
    }
    booster = lightgbm.train(params, train, num_boost_round=8)
    txt_path = os.path.join(tempfile.gettempdir(), f"lgbm_numerical_{seed}.txt")
    booster.save_model(txt_path)
    model = treelite.frontend.load_lightgbm_model(txt_path)
    return model, n_feat, 2  # cat_col index for edge injection (purely numerical)


# The f32-unrepresentable split threshold for the mixed-width stressor. 0.1 has
# no exact binary fraction; rounded to f32 it becomes 0.10000000149011612 — a
# DIFFERENT f64 value than the literal 0.1 the model stores. A kernel that
# narrows the f64 threshold to f32 before comparing would compare against
# 0.10000000149.., routing an input of float32(0.1) (== 0.10000000149.. in f64,
# i.e. STRICTLY > 0.1) to the WRONG child. The CR-02 f64-promoted comparison
# keeps the threshold at 0.1, so float32(0.1) correctly tests kLE-false (> 0.1)
# and routes RIGHT, matching upstream Treelite (which promotes the f32 input to
# f64 before the f64-threshold compare).
_MIXEDWIDTH_THRESHOLD = 0.1


def build_mixedwidth_model(seed):
    """<f64,f64> single-split model whose threshold is NOT f32-representable.

    Why 0.1: ``0.1`` is not exactly representable in binary floating point;
    ``float32(0.1) == 0.10000000149011612`` as an f64 — a strictly LARGER value
    than the literal ``0.1`` the model stores as an f64 threshold. The split uses
    ``opname='<='`` (kLE), so an input equal to ``float32(0.1)`` tests
    ``0.10000000149.. <= 0.1`` -> FALSE -> RIGHT child. A buggy kernel that
    narrowed the f64 threshold to f32 (0.10000000149..) would instead test
    ``0.10000000149.. <= 0.10000000149..`` -> TRUE -> LEFT child, diverging from
    upstream. This is the exact CR-02 stressor: the f32-input cells of this
    fixture only match the upstream golden if the kernel promotes BOTH operands
    to f64 (the 06-06 fix). Authored via the treelite ModelBuilder with
    ``threshold_type='float64'`` so the stored threshold is EXACTLY the f64
    ``0.1`` (no frontend re-drift). Returns (model, n_feat, cat_col).
    """
    from treelite.model_builder import (
        Metadata,
        ModelBuilder,
        PostProcessorFunc,
        TreeAnnotation,
    )

    n_feat = 2
    b = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=n_feat,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[0.0],
    )
    b.start_tree()
    b.start_node(0)
    b.numerical_test(
        0,
        _MIXEDWIDTH_THRESHOLD,
        default_left=True,
        opname="<=",  # kLE — also rides the CR-01 fallback gate (non-kLT)
        left_child_key=1,
        right_child_key=2,
    )
    b.end_node()
    b.start_node(1)
    b.leaf(-1.0)
    b.end_node()
    b.start_node(2)
    b.leaf(1.0)
    b.end_node()
    b.end_tree()
    model = b.commit()
    # Sanity: the literal 0.1 really is f32-unrepresentable (guards against a
    # future numpy/platform surprise that would defang the CR-02 stressor).
    assert float(np.float32(_MIXEDWIDTH_THRESHOLD)) != _MIXEDWIDTH_THRESHOLD, (
        "0.1 unexpectedly f32-representable on this platform — pick another "
        "f32-unrepresentable threshold for the mixed-width CR-02 stressor"
    )
    return model, n_feat, 0  # cat_col unused (custom matrix_builder supplies rows)


def _build_mixedwidth_matrix(seed, n_rows, n_feat):
    """Input rows that STRADDLE the f32-unrepresentable 0.1 threshold (CR-02).

    Beyond a wide RandomState fill (so the matrix has the usual ~120-row body),
    the leading rows are deliberately pinned to values around the f32/f64
    boundary of 0.1 so a wrong f32 narrowing would route them to the wrong child:

      row 0: float32(0.1) (== 0.10000000149.. in f64) — the canonical stressor:
              f64-compare -> RIGHT, f32-narrowed-compare -> LEFT.
      row 1: f32 value just below float32(0.1)        — both routes LEFT.
      row 2: literal 0.1 (f64)                        — kLE-true -> LEFT.
      row 3: 0.2                                       — clearly RIGHT.
      row 4: NaN -> default-direction routing (GTIL-05).

    The matrix is built in f64; ``capture_model`` casts it to the per-cell input
    dtype, so the f32 cells materialize exactly the f32 bit patterns above.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_rows, n_feat)).astype(np.float64)
    f32_thr = float(np.float32(_MIXEDWIDTH_THRESHOLD))  # 0.10000000149011612
    just_below = float(np.nextafter(np.float32(_MIXEDWIDTH_THRESHOLD), np.float32(-1)))
    X[0, 0] = f32_thr  # the CR-02 stressor row
    X[1, 0] = just_below
    X[2, 0] = _MIXEDWIDTH_THRESHOLD
    X[3, 0] = 0.2
    X[4, 0] = np.nan  # missing -> default direction (GTIL-05)
    return X


def _preset_of(model, override=None):
    """Return the model's preset token (``f32`` / ``f64``) for the fixture name.

    XGBoost-frontend models are the <f32,f32> preset; LightGBM-frontend and the
    explicitly-f64 ModelBuilder models are the <f64,f64> preset. This is recorded
    for provenance and the fixture name (the input-dtype axis is separate, D-05).
    Callers that know the preset (LightGBM/ModelBuilder always f64) pass it via
    ``override`` so the fixture name and manifest stay honest.
    """
    if override is not None:
        return override
    # treelite exposes leaf/threshold types via the model; XGBoost is f32/f32.
    return "f32"


def capture_model(model_name, model, n_feat, cat_col, preset_override=None,
                  matrix_builder=None):
    """Freeze the full (dtype x kind x {dense,sparse} x seed) cross-product.

    ``preset_override`` lets a caller that knows the model's preset (LightGBM and
    the explicit-f64 ModelBuilder models are <f64,f64>) tag the fixture name +
    manifest honestly. ``matrix_builder(seed, n_rows, n_feat) -> ndarray`` lets a
    model supply its own input rows (e.g. the mixed-width straddle rows that
    stress the f32/f64 threshold boundary); it defaults to ``_build_edge_matrix``.
    """
    preset = _preset_of(model, override=preset_override)
    n_rows = 140
    for seed in SEEDS:
        if matrix_builder is not None:
            X_full = matrix_builder(seed, n_rows, n_feat)
        else:
            X_full = _build_edge_matrix(seed, n_rows, n_feat, cat_col)
        for indtype, np_dtype in (("f32", np.float32), ("f64", np.float64)):
            Xc = X_full.astype(np_dtype)
            dense_nan, csr, csr_triple = _dense_and_csr(X_full, np_dtype, seed)

            for kind in KINDS:
                # ---- DENSE cell --------------------------------------------
                dense_out = _gtil_capture(model, kind, Xc)
                _freeze_cell(
                    model_name, preset, indtype, kind, "dense", seed,
                    n_feat, Xc, dense_out,
                )

                # ---- SPARSE cell (D-04 parity asserted at capture time) -----
                sparse_out = _gtil_capture(model, kind, csr)
                dense_nan_out = _gtil_capture(model, kind, dense_nan)
                # D-04 / Open Q1: dense-with-NaN == CSR on identical logical data
                # (absent == NaN). Assert BEFORE freezing. equal_nan tolerates the
                # NaN-routed cells; leaf_id is integer-exact.
                assert np.allclose(
                    np.nan_to_num(dense_nan_out, nan=0.0),
                    np.nan_to_num(sparse_out, nan=0.0),
                    rtol=0.0,
                    atol=1e-5,
                    equal_nan=True,
                ), (
                    f"D-04 dense-with-NaN != CSR parity failed for "
                    f"{model_name}.{indtype}.{kind}.s{seed} "
                    f"(max |delta|="
                    f"{float(np.nanmax(np.abs(dense_nan_out - sparse_out))):g})"
                )
                _freeze_cell(
                    model_name, preset, indtype, kind, "sparse", seed,
                    n_feat, dense_nan, sparse_out, csr_triple=csr_triple,
                )


def _freeze_cell(
    model_name, preset, indtype, kind, layout, seed, n_feat, X, out,
    csr_triple=None,
):
    name = f"{model_name}.{preset}.{indtype}.{kind}.{layout}.s{seed}.golden.json"
    payload = {
        "model_path": f"fixtures/gtil/{model_name}.captured-in-script",
        "n_features": int(n_feat),
        "input": _json_safe(np.asarray(X)),
        "output": _json_safe(out),
        "output_shape": list(out.shape),
        "manifest": _manifest(model_name, preset, indtype, kind, layout, seed),
        "sha256": _payload_sha256(
            _json_safe(np.asarray(X)), _json_safe(out)
        ),
    }
    # WR-01: sparse cells carry the REAL frozen CSR triple (data/indices/indptr)
    # so the Rust runner loads it verbatim instead of re-deriving a CSR from
    # NaN-presence. ``data`` runs through ``_json_safe`` for non-finite safety
    # (an edge cell may legitimately be +/-inf); indices/indptr are exact ints.
    if layout == "sparse" and csr_triple is not None:
        data_arr, indices_arr, indptr_arr = csr_triple
        payload["csr"] = {
            "data": _json_safe(np.asarray(data_arr)),
            "indices": [int(v) for v in np.asarray(indices_arr).tolist()],
            "indptr": [int(v) for v in np.asarray(indptr_arr).tolist()],
        }
    _write_golden(name, payload)


def main():
    # Capture-time API assert: treelite.gtil.predict must keep the expected
    # signature so an upstream API change is caught (mirror
    # capture_lightgbm.py:191-198).
    print("=== treelite.gtil.predict signature ===")
    import inspect

    sig = inspect.signature(treelite.gtil.predict)
    assert "pred_margin" in sig.parameters, "gtil.predict lost the pred_margin kw"
    assert sig.parameters["pred_margin"].default is False, (
        "default pred_margin changed -- goldens assume post-processed output"
    )
    assert hasattr(treelite.gtil, "predict_leaf"), "gtil.predict_leaf missing"
    assert hasattr(treelite.gtil, "predict_per_tree"), "gtil.predict_per_tree missing"
    print(sig)

    # Use the first seed to author each representative model; the per-seed input
    # matrices are drawn separately inside capture_model.
    bin_model, bin_feat, bin_cat = build_binary_model(SEEDS[0])
    capture_model("binary", bin_model, bin_feat, bin_cat)

    # UNCONDITIONAL leaf-vector-broadcast axis (GTIL-07, D-03).
    lv_model, lv_feat, lv_cat = build_leaf_vector_model(SEEDS[0])
    capture_model("leaf_vec_mc", lv_model, lv_feat, lv_cat)

    # CR-01 large-margin sigmoid axis (the f64 cells exercise sigmoid_f64). Its
    # model.bin is frozen from the SAME object that produced its goldens, so the
    # Rust runner deserializes the exact model — no xgboost-frontend re-drift.
    lm_model, lm_feat, lm_cat = build_large_margin_model(SEEDS[0])
    capture_model("large_margin", lm_model, lm_feat, lm_cat)
    _freeze_model_bin("large_margin.model.bin", lm_model)

    # CR-01 kLE coverage (06-07): a NUMERICAL-only LightGBM model — every split is
    # kLE, so predict_cpu's operator-coverage gate routes the WHOLE model to the
    # scalar reference (D-02). The <f64,f64> preset is recorded honestly. Its
    # model.bin is frozen from the SAME object that produced its goldens.
    ln_model, ln_feat, ln_cat = build_lgbm_numerical_model(SEEDS[0])
    capture_model("lgbm_numerical", ln_model, ln_feat, ln_cat, preset_override="f64")
    _freeze_model_bin("lgbm_numerical.model.bin", ln_model)

    # CR-02 mixed-width coverage (06-07): a <f64,f64> single-split model whose
    # threshold (0.1) is f32-unrepresentable. The f32-input cells only match the
    # upstream golden if the kernel promotes both operands to f64 before the kLE
    # compare (the 06-06 CR-02 fix). Custom straddle rows guarantee the boundary
    # is exercised. Its model.bin is frozen from the exact built object.
    mw_model, mw_feat, mw_cat = build_mixedwidth_model(SEEDS[0])
    capture_model(
        "mixedwidth", mw_model, mw_feat, mw_cat,
        preset_override="f64", matrix_builder=_build_mixedwidth_matrix,
    )
    _freeze_model_bin("mixedwidth.model.bin", mw_model)

    print("All GTIL matrix goldens captured (treelite-GTIL, scalar-cpu, D-07/D-09).")


def _freeze_model_bin(name, model):
    """Freeze a model's treelite v5 byte stream beside its goldens.

    The Rust runner (``tests/gtil_matrix.rs``) loads ``<model>.model.bin`` via
    ``treelite_core::deserialize``. The large-margin model is authored fresh
    here, so its ``model.bin`` MUST be frozen from this exact object (the same
    one that produced its golden cells) — mirroring ``capture_gtil_models.py``.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = model.serialize_bytes()
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"Wrote {path} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()

"""GTIL prediction shim — dtype dispatch + flat→N-D reshape over the compiled
zero-copy ``predict_f32`` / ``predict_f64`` entry points.

Ports the upstream ``treelite.gtil`` API shape (D-01): ``predict`` /
``predict_leaf`` / ``predict_per_tree``. The compiled Rust side exposes two
MONOMORPHIZED dense entry points (no f32↔f64 pre-cast) returning a FLAT 1-D
buffer; this shim:

* dispatches on ``model.input_type`` (variant-derived — Pitfall 2) so an f32
  model goes through ``predict_f32`` and an f64 model through ``predict_f64``;
* reshapes the flat output to the upstream N-D shape
  ``(num_row, num_target_or_1, max_num_class)`` via ``predict_output_shape``
  (a view, no copy — Pitfall 3).

Dtype / contiguity rejection (D-03) is enforced Rust-side: a wrong-dtype array is
rejected by the typed ``PyReadonlyArray2`` param, a non-contiguous array by the
``as_slice`` contiguity check — both raise ``TreeliteError`` with no silent cast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import _treelite_rs

if TYPE_CHECKING:
    import numpy as np

# The compiled gtil entry points (Rust seam). Names mirror src/gtil.rs.
_G = _treelite_rs.gtil

# Re-export the raw typed entry points so the A/B suite (and advanced callers) can
# hit a specific monomorphized path directly.
predict_f32 = _G.predict_f32
predict_f64 = _G.predict_f64
predict_output_shape = _G.predict_output_shape

__all__ = [
    "predict",
    "predict_leaf",
    "predict_per_tree",
    "predict_f32",
    "predict_f64",
    "predict_output_shape",
]


def _dense_predict(model, data, *, nthread: int, pred_margin: bool):
    """Dispatch to predict_f32/_f64 by the DATA dtype, then reshape flat→N-D.

    The monomorphized entry point is selected by the INPUT array dtype, not the
    model's preset: GTIL's input element type is an axis independent of the model
    variant (an f32 model accepts an f64 input matrix and vice versa, accumulating
    leaf values into the input-typed buffer — the harness InputT-as-accumulator
    contract). Routing on the model variant instead would feed an f32 array into
    the f64-typed entry point and trip the strict dtype gate (D-03).
    """
    dtype = data.dtype
    if dtype == "float32":
        flat = predict_f32(model, data, nthread=nthread, pred_margin=pred_margin)
    elif dtype == "float64":
        flat = predict_f64(model, data, nthread=nthread, pred_margin=pred_margin)
    else:
        raise _treelite_rs.TreeliteError(
            f"unsupported input dtype {dtype!r}; expected float32 or float64"
        )

    num_row = data.shape[0]
    shape = tuple(predict_output_shape(model, num_row, pred_margin=pred_margin))
    # Flat→N-D is a view (no copy, Pitfall 3).
    return flat.reshape(shape)


def predict(
    model,
    data: "np.ndarray",
    *,
    nthread: int = -1,
    pred_margin: bool = False,
) -> "np.ndarray":
    """Predict with a Treelite model over a dense numpy matrix (GTIL default kind).

    Parameters
    ----------
    model : Model
        The loaded model.
    data : numpy.ndarray
        A C-contiguous 2-D matrix whose dtype matches ``model.input_type``
        (``float32`` or ``float64``). A wrong-dtype or non-contiguous array raises
        ``TreeliteError`` (strict, no silent cast — D-03).
    nthread : int
        Requested CPU core count (recorded; the scalar reference is single-threaded).
    pred_margin : bool
        If ``True``, produce raw margin scores (skip post-processing).

    Returns
    -------
    numpy.ndarray
        Prediction output, shaped ``(num_row, num_target_or_1, max_num_class)``.
    """
    return _dense_predict(model, data, nthread=nthread, pred_margin=pred_margin)


def predict_leaf(model, data: "np.ndarray", *, nthread: int = -1) -> "np.ndarray":
    """Per-row leaf-node ID prediction (upstream ``predict_leaf``).

    The ``LeafId`` kind is not yet wired through the binding (it surfaces a typed
    ``TreeliteError`` from the engine); the signature is provided for 1:1 upstream
    parity (D-01) and lands fully in a later slice.
    """
    raise _treelite_rs.TreeliteError(
        "predict_leaf (LeafId kind) is not yet wired in the binding"
    )


def predict_per_tree(model, data: "np.ndarray", *, nthread: int = -1) -> "np.ndarray":
    """Per-tree margin-score prediction (upstream ``predict_per_tree``).

    The ``ScorePerTree`` kind is not yet wired through the binding; the signature
    is provided for 1:1 upstream parity (D-01) and lands fully in a later slice.
    """
    raise _treelite_rs.TreeliteError(
        "predict_per_tree (ScorePerTree kind) is not yet wired in the binding"
    )

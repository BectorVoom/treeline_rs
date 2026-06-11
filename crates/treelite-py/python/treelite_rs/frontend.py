"""Model loaders — the path→bytes/text shim over the compiled Rust loaders.

Ports the upstream ``treelite.frontend`` API shape (D-01): the same public
signatures (``load_xgboost_model`` / ``load_xgboost_model_legacy_binary`` /
``load_lightgbm_model``), with file I/O staying Python-side. The Rust loaders in
``_treelite_rs.frontend`` take already-read ``str`` / ``bytes`` and return a
``Model``; format detection ports upstream ``_detect_xgboost_format``.
"""

from __future__ import annotations

import pathlib
from typing import Union

from . import _treelite_rs

# The compiled loaders (Rust seam). Names mirror src/frontend.rs.
_F = _treelite_rs.frontend
Model = _treelite_rs.Model

__all__ = [
    "load_xgboost_model",
    "load_xgboost_model_legacy_binary",
    "load_lightgbm_model",
    "load_xgboost_json_str",
    "load_xgboost_ubjson_bytes",
    "load_xgboost_legacy_bytes",
    "load_lightgbm_str",
    "detect_xgboost_format_bytes",
]

# Re-export the raw compiled entry points so tests (and advanced callers) can hit
# the str/bytes seam directly without the path shim.
load_xgboost_json_str = _F.load_xgboost_json_str
load_xgboost_ubjson_bytes = _F.load_xgboost_ubjson_bytes
load_xgboost_legacy_bytes = _F.load_xgboost_legacy_bytes
load_lightgbm_str = _F.load_lightgbm_str
detect_xgboost_format_bytes = _F.detect_xgboost_format_bytes


def _normalize_path(filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """Fully expand a path and convert it to an absolute path (upstream parity)."""
    return pathlib.Path(filename).expanduser().resolve()


def load_xgboost_model(
    filename: Union[str, pathlib.Path],
    *,
    format_choice: str = "use_suffix",
) -> Model:
    """Load an XGBoost model stored in JSON or UBJSON format.

    Parameters
    ----------
    filename :
        Path to the model file.
    format_choice :
        How to select the wire format:

        * ``use_suffix`` (default): files ending in ``.json`` are parsed as JSON,
          all others as UBJSON.
        * ``inspect``: sniff the leading bytes to disambiguate JSON vs UBJSON.
        * ``json`` / ``ubjson``: force the respective parser.

    Returns
    -------
    model : Model
        The loaded model.
    """
    path = _normalize_path(filename)

    if format_choice == "use_suffix":
        if path.name.endswith(".json"):
            return load_xgboost_json_str(path.read_text())
        return load_xgboost_ubjson_bytes(path.read_bytes())

    if format_choice == "inspect":
        data = path.read_bytes()
        detected = detect_xgboost_format_bytes(data)
        if detected == "json":
            # A detected-JSON file with invalid UTF-8 bytes must surface as the
            # single ``TreeliteError`` (D-06), not a bare ``UnicodeDecodeError``
            # (WR-04).
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _treelite_rs.TreeliteError(
                    f"XGBoost JSON model is not valid UTF-8: {exc}"
                ) from exc
            return load_xgboost_json_str(text)
        if detected == "ubjson":
            return load_xgboost_ubjson_bytes(data)
        # Undetected format: surface as ``TreeliteError`` so callers branch on
        # the single binding exception (D-06, WR-04) rather than ``ValueError``.
        raise _treelite_rs.TreeliteError(
            "Could not detect whether the given XGBoost model is JSON or UBJSON. "
            "Please explicitly set format_choice='json' or format_choice='ubjson'"
        )

    if format_choice == "ubjson":
        return load_xgboost_ubjson_bytes(path.read_bytes())

    if format_choice == "json":
        return load_xgboost_json_str(path.read_text())

    raise ValueError(f"Unknown format_choice argument: {format_choice}")


def load_xgboost_model_legacy_binary(filename: Union[str, pathlib.Path]) -> Model:
    """Load an XGBoost model stored in the legacy binary format (``*.model``)."""
    path = _normalize_path(filename)
    return load_xgboost_legacy_bytes(path.read_bytes())


def load_lightgbm_model(filename: Union[str, pathlib.Path]) -> Model:
    """Load a LightGBM model from its text dump (``model.txt``)."""
    path = _normalize_path(filename)
    return load_lightgbm_str(path.read_text())

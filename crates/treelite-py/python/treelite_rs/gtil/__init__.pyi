"""Type stubs for treelite_rs.gtil (PEP 561, D-10)."""

from __future__ import annotations

from typing import List

import numpy as np

from ..model import Model

def predict(
    model: Model,
    data: np.ndarray,
    *,
    nthread: int = ...,
    pred_margin: bool = ...,
    backend: str = ...,
) -> np.ndarray: ...
def predict_leaf(
    model: Model, data: np.ndarray, *, nthread: int = ..., backend: str = ...
) -> np.ndarray: ...
def predict_per_tree(
    model: Model, data: np.ndarray, *, nthread: int = ..., backend: str = ...
) -> np.ndarray: ...

# Compiled monomorphized entry points (Rust seam) — return a FLAT 1-D array.
def predict_f32(
    model: Model,
    data: np.ndarray,
    *,
    nthread: int = ...,
    pred_margin: bool = ...,
    backend: str = ...,
) -> np.ndarray: ...
def predict_f64(
    model: Model,
    data: np.ndarray,
    *,
    nthread: int = ...,
    pred_margin: bool = ...,
    backend: str = ...,
) -> np.ndarray: ...
def predict_output_shape(
    model: Model, num_row: int, *, pred_margin: bool = ...
) -> List[int]: ...

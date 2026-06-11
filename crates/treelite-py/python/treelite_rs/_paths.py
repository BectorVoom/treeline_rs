"""Shared path-normalization helper (IN-01).

``_normalize_path`` was previously defined verbatim in both ``__init__.py`` and
``frontend.py``; the duplication risked the two copies drifting apart if the
upstream path-handling parity ever changed. Define it ONCE here and import it in
both modules.
"""

from __future__ import annotations

import pathlib
from typing import Union


def _normalize_path(filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """Fully expand a path and convert it to an absolute path (upstream parity)."""
    return pathlib.Path(filename).expanduser().resolve()

"""Shared pytest fixtures for the treelite_rs binding A/B suite.

Wave 0 (Plan 08-01) lands the test INSTRUMENT for the whole phase: this conftest
plus the seven RED/skip test files collect cleanly and run all-skipped. Each
downstream slice (08-02 .. 08-05) flips its file(s) to GREEN as the capability
lands. The Nyquist contract is that the harness exists and collects BEFORE any
slice depends on it.

Key gates:
- ``treelite`` (upstream) is imported with ``pytest.importorskip`` so the A/B
  suite skips cleanly on a machine without the C++ reference (D-11 witness).
- ``scipy`` cells gate on ``pytest.importorskip("scipy")`` (sparse CSR input).
- ``FIXTURES`` points at the repo ``fixtures/`` dir; tests reference goldens and
  model files through it, never via a hardcoded absolute path.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# Repo root = four levels up from this file:
#   conftest.py(0)/python(1)/tests(2)/treelite-py(3)/crates(4=repo root).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

#: Frozen test corpus shared with the Rust suite (models + upstream-GTIL goldens).
FIXTURES = REPO_ROOT / "fixtures"

#: Per-kind GTIL goldens captured from upstream treelite.gtil.predict.
GTIL_FIXTURES = FIXTURES / "gtil"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG so synthetic input matrices are reproducible across runs."""
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def treelite_upstream():
    """The upstream C++ Treelite Python package (skip the A/B cell if absent)."""
    return pytest.importorskip("treelite")


@pytest.fixture(scope="session")
def treelite_rs():
    """The Rust binding under test."""
    return pytest.importorskip("treelite_rs")


@pytest.fixture(scope="session")
def scipy_sparse():
    """scipy.sparse (gate sparse-CSR predict cells on its presence)."""
    return pytest.importorskip("scipy.sparse")


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    """Path to the frozen fixtures directory."""
    return FIXTURES

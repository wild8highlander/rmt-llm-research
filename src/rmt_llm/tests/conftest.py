"""Test infrastructure for rmt_llm verification package."""

import sys
from pathlib import Path

import numpy as np
import pytest


# Flat module imports (`import caputo_fractional`, ...) need this directory
# (src/rmt_llm) itself on sys.path when the package is not pip-installed.
_PKG_DIR = str(Path(__file__).resolve().parents[1])
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def default_q():
    """Default aspect ratio for Marchenko-Pastur tests."""
    return 0.5


@pytest.fixture
def default_sigma2():
    """Default population variance."""
    return 1.0

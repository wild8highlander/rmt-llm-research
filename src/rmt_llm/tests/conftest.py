"""Test infrastructure for rmt_llm verification package."""

import numpy as np
import pytest


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

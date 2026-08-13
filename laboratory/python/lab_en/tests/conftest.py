"""
Shared pytest fixtures for the lab_en test suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the lab_en directory is importable
_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Session-shared RNG for reproducible stochastic tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def small_config():
    """Tiny config that runs fast — used by most tests."""
    from tiny_gpt import TinyGPTConfig

    return TinyGPTConfig(
        vocab_size=64,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        max_seq_len=64,
        mlp_ratio=4,
        use_layernorm=True,
        use_mlp=True,
        activation="gelu",
        seed=42,
    )


@pytest.fixture
def small_model(small_config):
    """A small untrained TinyGPT model — fast to construct & forward."""
    from tiny_gpt import TinyGPT

    return TinyGPT(small_config)


@pytest.fixture
def small_corpus() -> bytes:
    """A small text corpus suitable for BPE training (≥ 1 KB)."""
    return (
        b"The Quick Brown Fox Jumps Over The Lazy Dog.\n"
        b"Random Matrix Theory (RMT) studies the eigenvalue distribution of large random matrices.\n"
        b"Marchenko-Pastur law describes the asymptotic spectral density of sample covariance matrices.\n"
        b"The BBP phase transition occurs when a spike eigenvalue separates from the bulk.\n"
        b"Tracy-Widom fluctuations govern the largest eigenvalue of Wigner matrices.\n"
        b"In large language models, activation covariance matrices exhibit RMT signatures.\n"
        b"Hallucinations are inevitable past the critical token count N_crit.\n"
        b"The Caputo fractional derivative introduces long-range memory in time.\n"
        b"The Non-Hermitian Skin Effect causes eigenvalue collapse to the boundary.\n"
        b"Keating-Snaith corrections refine the gamma function for finite matrix sizes.\n"
    ) * 5  # ~3 KB


@pytest.fixture
def tiny_corpus() -> bytes:
    """Even smaller corpus (~200 bytes) for ultra-fast smoke tests."""
    return (
        b"TinyGPT is a small transformer model.\n"
        b"It uses byte-level or BPE tokenization.\n"
        b"Training uses Adam with cosine learning rate.\n"
        b"Reverse-mode autodiff is implemented in pure NumPy.\n"
    ) * 3


@pytest.fixture
def tmp_weights_path(tmp_path):
    """Path to a temporary .npz weights file (cleaned up automatically)."""
    return str(tmp_path / "tiny_gpt_test_weights.npz")


@pytest.fixture
def tmp_bpe_path(tmp_path):
    """Path to a temporary BPE JSON file (cleaned up automatically)."""
    return str(tmp_path / "tiny_gpt_test_bpe.json")


@pytest.fixture(autouse=True)
def _set_seed():
    """Reset NumPy RNG before every test — keep tests deterministic."""
    np.random.seed(42)
    yield
    # Cleanup not needed — pytest manages tmp_path

"""
Performance benchmarks for TinyGPT + BPE + Adam using pytest-benchmark.

These benchmarks measure hot-path performance so we can detect regressions
in the pure-NumPy forward/backward pass. They run as part of `make bench`
and in CI under the `benchmark.yml` workflow.

Run locally:
    pytest laboratory/python/lab_en/tests/test_benchmark.py --benchmark-only
    pytest laboratory/python/lab_en/tests/test_benchmark.py --benchmark-only --benchmark-compare
    pytest laboratory/python/lab_en/tests/test_benchmark.py --benchmark-only --benchmark-save=baseline

The benchmarks are intentionally small enough to complete in < 1 second each
so CI stays fast, but large enough to surface O(n²) vs O(n³) regressions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# Make the lab package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    pytest.importorskip("pytest_benchmark")
except ImportError:  # pragma: no cover
    pytest.skip("pytest-benchmark not installed", allow_module_level=True)

from tiny_gpt import TinyGPT, TinyGPTConfig
from tiny_gpt_trainer import (
    AdamState,
    BPETokenizer,
    backward,
    cross_entropy_loss,
    forward_with_cache,
    softmax,
)


_LAYER_KEYS = (
    "W_q",
    "W_k",
    "W_v",
    "W_o",
    "b_q",
    "b_k",
    "b_v",
    "b_o",
    "ln1_gamma",
    "ln1_beta",
    "ln2_gamma",
    "ln2_beta",
    "W_fc1",
    "W_fc2",
    "b_fc1",
    "b_fc2",
)


def _zero_grads(model: TinyGPT) -> dict:
    """Zero gradients matching the structure produced by ``backward()``."""
    layers = []
    for layer in model.layers:
        lg = {}
        for name in _LAYER_KEYS:
            arr = getattr(layer, name)
            lg[name] = np.zeros_like(arr) if arr is not None else None
        layers.append(lg)
    return {
        "token_emb": np.zeros_like(model.token_emb),
        "pos_emb": np.zeros_like(model.pos_emb),
        "lm_head": np.zeros_like(model.lm_head),
        "ln_f_gamma": np.zeros_like(model.ln_f_gamma),
        "ln_f_beta": np.zeros_like(model.ln_f_beta),
        "layers": layers,
    }


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def small_model():
    """1-layer TinyGPT with hidden=32 — fast enough for CI."""
    cfg = TinyGPTConfig(
        vocab_size=128,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=16,
        use_mlp=True,
        use_layernorm=True,
        seed=0,
    )
    return TinyGPT(cfg)


@pytest.fixture
def medium_model():
    """4-layer TinyGPT with hidden=64 — closer to training config."""
    cfg = TinyGPTConfig(
        vocab_size=256,
        hidden_dim=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        use_mlp=True,
        use_layernorm=True,
        seed=1,
    )
    return TinyGPT(cfg)


@pytest.fixture
def trained_bpe():
    """A BPE tokenizer trained on a small corpus."""
    corpus = ("def hello_world():\n    print('Hello, world!')\n    return 42\n") * 20
    tok = BPETokenizer(vocab_size=512)
    tok.train(corpus.encode("utf-8"), target_merges=64)
    return tok, corpus


# ─── Forward pass benchmarks ─────────────────────────────────────────────────


class TestForwardBenchmarks:
    """Measure forward pass throughput."""

    def test_bench_forward_small(self, benchmark, small_model):
        """Benchmark: 1-layer forward pass."""
        x = np.random.randint(0, 128, size=16)

        def run():
            logits, _ = small_model.forward(x)
            return logits

        result = benchmark(run)
        assert result.shape == (16, 128)

    def test_bench_forward_medium(self, benchmark, medium_model):
        """Benchmark: 4-layer forward pass."""
        x = np.random.randint(0, 256, size=32)

        def run():
            logits, _ = medium_model.forward(x)
            return logits

        result = benchmark(run)
        assert result.shape == (32, 256)

    def test_bench_forward_batched(self, benchmark, small_model):
        """Benchmark: 8-sequence batched forward (manual loop)."""
        batch = [np.random.randint(0, 128, size=16) for _ in range(8)]

        def run():
            return [small_model.forward(x)[0] for x in batch]

        result = benchmark(run)
        assert len(result) == 8


# ─── Softmax / cross-entropy benchmarks ──────────────────────────────────────


class TestNumericsBenchmarks:
    """Measure softmax / CE throughput — these are called per sample."""

    def test_bench_softmax(self, benchmark):
        """Benchmark: softmax over 512 logits (vocab size)."""
        x = np.random.randn(512)

        def run():
            return softmax(x)

        benchmark(run)

    def test_bench_cross_entropy(self, benchmark):
        """Benchmark: single cross-entropy call."""
        logits = np.random.randn(512)
        target = 42

        def run():
            return cross_entropy_loss(logits, target)

        benchmark(run)

    def test_bench_softmax_batch(self, benchmark):
        """Benchmark: softmax over batch of 32 × 512."""
        x = np.random.randn(32, 512)

        def run():
            return softmax(x, axis=-1)

        benchmark(run)


# ─── BPE tokenizer benchmarks ────────────────────────────────────────────────


class TestBPEBenchmarks:
    """Measure BPE train + encode throughput."""

    def test_bench_bpe_train(self, benchmark, trained_bpe):
        """Benchmark: train BPE with 64 merges on a small corpus."""
        _, corpus = trained_bpe
        corpus_bytes = corpus.encode("utf-8")

        def run():
            tok = BPETokenizer(vocab_size=512)
            tok.train(corpus_bytes, target_merges=64)
            return tok.n_merges

        result = benchmark(run)
        assert 0 < result <= 64

    def test_bench_bpe_encode(self, benchmark, trained_bpe):
        """Benchmark: encode a string with trained BPE."""
        tok, corpus = trained_bpe
        sample = corpus[:200]

        def run():
            return tok.encode(sample)

        result = benchmark(run)
        assert len(result) > 0

    def test_bench_bpe_decode(self, benchmark, trained_bpe):
        """Benchmark: decode a token sequence back to string."""
        tok, corpus = trained_bpe
        ids = tok.encode(corpus[:200])

        def run():
            return tok.decode(ids)

        result = benchmark(run)
        assert len(result) > 0


# ─── Adam optimizer benchmarks ───────────────────────────────────────────────


class TestAdamBenchmarks:
    """Measure Adam.step() throughput — this is called per parameter update."""

    def test_bench_adam_step(self, benchmark, small_model):
        """Benchmark: single Adam.step() on the small model."""
        optimizer = AdamState(small_model, lr=5e-4, warmup_epochs=0)
        grads = _zero_grads(small_model)

        def run():
            optimizer.step(small_model, grads)
            return small_model.token_emb

        benchmark(run)

    def test_bench_adam_step_medium(self, benchmark, medium_model):
        """Benchmark: Adam.step() on the 4-layer model (≈ full TinyGPT size)."""
        optimizer = AdamState(medium_model, lr=5e-4, warmup_epochs=0)
        grads = _zero_grads(medium_model)

        def run():
            optimizer.step(medium_model, grads)
            return medium_model.token_emb

        benchmark(run)


# ─── End-to-end training step benchmark ──────────────────────────────────────


class TestEndToEndBenchmarks:
    """One full training step: forward → loss → backward → Adam update."""

    def test_bench_single_train_step(self, benchmark, small_model):
        """Benchmark: one full training step (forward + backward + Adam).

        This is THE critical perf number — every 1% improvement here
        translates directly to wall-clock training time.
        """
        cfg = small_model.config
        x = np.random.randint(0, cfg.vocab_size, size=cfg.max_seq_len)
        y = np.random.randint(0, cfg.vocab_size, size=cfg.max_seq_len)
        optimizer = AdamState(small_model, lr=5e-4, warmup_epochs=0)

        def run():
            # Forward with cache
            _, cache = forward_with_cache(small_model, x)
            loss = cross_entropy_loss(cache.logits[-1], int(y[-1]))
            # Backward — reverse-mode autodiff through the whole model
            grads = backward(small_model, cache, y)
            # Adam update
            optimizer.step(small_model, grads)
            return loss

        result = benchmark(run)
        assert np.isfinite(result)


# ─── Memory-pressure benchmarks (informational) ──────────────────────────────


class TestMemoryBenchmarks:
    """These benchmarks surface memory regressions. Run with --benchmark-only.

    Note: pytest-benchmark does not measure memory directly, but the wall time
    is a good proxy for cache pressure. If `test_bench_forward_long_seq` slows
    down by >2x, suspect O(n²) regression in attention.
    """

    def test_bench_forward_long_seq(self, benchmark):
        """Benchmark: forward on max_seq_len=128 (cache pressure)."""
        cfg = TinyGPTConfig(
            vocab_size=128,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
            use_mlp=True,
            use_layernorm=True,
            seed=42,
        )
        model = TinyGPT(cfg)
        x = np.random.randint(0, 128, size=128)

        def run():
            logits, _ = model.forward(x)
            return logits

        result = benchmark(run)
        assert result.shape == (128, 128)

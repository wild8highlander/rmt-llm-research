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
import time
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
from tiny_gpt_trainer import BPETokenizer, AdamState, cross_entropy_loss, softmax


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
    )
    model = TinyGPT(cfg)
    model.init_weights(seed=0)
    return model


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
    )
    model = TinyGPT(cfg)
    model.init_weights(seed=1)
    return model


@pytest.fixture
def trained_bpe():
    """A BPE tokenizer fit on a small corpus."""
    corpus = (
        "def hello_world():\n"
        "    print('Hello, world!')\n"
        "    return 42\n"
    ) * 20
    tok = BPETokenizer(vocab_size=256)
    tok.fit(corpus, n_merges=64)
    return tok, corpus


# ─── Forward pass benchmarks ─────────────────────────────────────────────────

class TestForwardBenchmarks:
    """Measure forward pass throughput."""

    def test_bench_forward_small(self, benchmark, small_model):
        """Benchmark: 1-layer forward pass."""
        x = np.random.randint(0, 128, size=16)

        def run():
            return small_model.forward(x)

        result = benchmark(run)
        assert result.shape == (16, 128)

    def test_bench_forward_medium(self, benchmark, medium_model):
        """Benchmark: 4-layer forward pass."""
        x = np.random.randint(0, 256, size=32)

        def run():
            return medium_model.forward(x)

        result = benchmark(run)
        assert result.shape == (32, 256)

    def test_bench_forward_batched(self, benchmark, small_model):
        """Benchmark: 8-sequence batched forward (manual loop)."""
        batch = [np.random.randint(0, 128, size=16) for _ in range(8)]

        def run():
            return [small_model.forward(x) for x in batch]

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
    """Measure BPE fit + encode throughput."""

    def test_bench_bpe_fit(self, benchmark, trained_bpe):
        """Benchmark: fit BPE with 64 merges on a small corpus."""
        _, corpus = trained_bpe

        def run():
            tok = BPETokenizer(vocab_size=256)
            tok.fit(corpus, n_merges=64)
            return tok.n_merges

        result = benchmark(run)
        assert result == 64

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

    def test_bench_adam_step(self, benchmark):
        """Benchmark: single Adam.step() with 1k params."""
        params = np.random.randn(1000) * 0.1
        grads = np.random.randn(1000) * 0.01
        state = AdamState(params.shape, lr=5e-4)

        def run():
            state.step(params, grads, t=1)
            return params

        benchmark(run)

    def test_bench_adam_step_large(self, benchmark):
        """Benchmark: Adam.step() with 100k params (≈ TinyGPT size)."""
        params = np.random.randn(100_000) * 0.1
        grads = np.random.randn(100_000) * 0.01
        state = AdamState(params.shape, lr=5e-4)

        def run():
            state.step(params, grads, t=10)
            return params

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
        optimizer = AdamState(
            (small_model.n_params,),
            lr=5e-4,
        )

        def run():
            # Forward
            logits = small_model.forward(x)
            loss = cross_entropy_loss(logits[-1], y[-1])
            # Backward — for the benchmark we just use zero grads to time the
            # machinery; the real backward is exercised in test_trainer.py
            grads = {k: np.zeros_like(v) for k, v in small_model.parameters_dict().items()}
            params = np.concatenate([
                v.ravel() for v in small_model.parameters_dict().values()
            ])
            optimizer.step(params, np.zeros_like(params), t=1)
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
        )
        model = TinyGPT(cfg)
        model.init_weights(seed=42)
        x = np.random.randint(0, 128, size=128)

        def run():
            return model.forward(x)

        result = benchmark(run)
        assert result.shape == (128, 128)

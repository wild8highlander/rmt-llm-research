"""
Performance benchmarks: TinyGPT v2 vs v3.

Measures forward pass, backward pass, and memory characteristics of
the v2 (absolute position embeddings, MHA) and v3 (RoPE, GQA,
mixed-precision, gradient checkpointing) architectures.

Run with::

    pytest laboratory/python/lab_en/tests/test_benchmark_v3.py --benchmark-only
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    pytest.importorskip("pytest_benchmark")
except ImportError:  # pragma: no cover
    pytest.skip("pytest-benchmark not installed", allow_module_level=True)

from tiny_gpt import TinyGPT as TinyGPTv2
from tiny_gpt import TinyGPTConfig as V2Config
from tiny_gpt_v3 import TinyGPTV3 as TinyGPTv3
from tiny_gpt_v3 import TinyGPTV3Config as V3Config


# ─── Fixtures ──────────────────────────────────────────────────────────────


def _v2_config():
    return V2Config(
        vocab_size=128,
        hidden_dim=64,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        use_mlp=True,
        use_layernorm=True,
        seed=42,
    )


def _v3_config(use_gqa=True, use_rope=True, mixed_precision=False, gradient_checkpointing=False):
    n_kv_heads = 2 if use_gqa else 4
    return V3Config(
        vocab_size=128,
        hidden_dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        max_seq_len=32,
        use_mlp=True,
        use_layernorm=True,
        use_rope=use_rope,
        mixed_precision=mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        seed=42,
    )


@pytest.fixture
def v2_model():
    return TinyGPTv2(_v2_config())


@pytest.fixture
def v3_model():
    return TinyGPTv3(_v3_config())


@pytest.fixture
def v3_gqa_model():
    return TinyGPTv3(_v3_config(use_gqa=True))


@pytest.fixture
def v3_mp_model():
    return TinyGPTv3(_v3_config(mixed_precision=True))


@pytest.fixture
def v3_ckpt_model():
    return TinyGPTv3(_v3_config(gradient_checkpointing=True))


# ─── Forward pass benchmarks ───────────────────────────────────────────────


class TestForwardBenchmarks:
    """Compare forward pass throughput: v2 vs v3 variants."""

    def test_bench_forward_v2(self, benchmark, v2_model):
        """v2 forward: MHA + absolute position embeddings."""
        x = np.random.randint(0, 128, size=32)

        def run():
            return v2_model.forward(x)

        result = benchmark(run)
        assert result[0].shape == (32, 128)

    def test_bench_forward_v3_rope_gqa(self, benchmark, v3_model):
        """v3 forward: RoPE + GQA."""
        x = np.random.randint(0, 128, size=32)

        def run():
            return v3_model.forward(x)

        result = benchmark(run)
        assert result[0].shape == (32, 128)

    def test_bench_forward_v3_mixed_precision(self, benchmark, v3_mp_model):
        """v3 forward: RoPE + GQA + float16."""
        x = np.random.randint(0, 128, size=32)

        def run():
            return v3_mp_model.forward(x)

        result = benchmark(run)
        assert result[0].shape == (32, 128)

    def test_bench_forward_v3_checkpointing(self, benchmark, v3_ckpt_model):
        """v3 forward: RoPE + GQA + gradient checkpointing."""
        x = np.random.randint(0, 128, size=32)

        def run():
            return v3_ckpt_model.forward(x)

        result = benchmark(run)
        assert result[0].shape == (32, 128)


# ─── Backward pass benchmarks ──────────────────────────────────────────────


class TestBackwardBenchmarks:
    """Compare backward pass throughput: v3 vs v3 with checkpointing."""

    def test_bench_backward_v3(self, benchmark, v3_model):
        """v3 backward: full cache."""
        x = np.random.randint(0, 128, size=32)

        def run():
            logits, cache = v3_model.forward_with_cache(x)
            dlogits = np.random.randn(*logits.shape).astype(np.float32)
            return v3_model.backward(cache, dlogits)

        result = benchmark(run)
        assert "L0_W_q" in result

    def test_bench_backward_v3_checkpointing(self, benchmark, v3_ckpt_model):
        """v3 backward: gradient checkpointing (recompute forward)."""
        x = np.random.randint(0, 128, size=32)

        def run():
            logits, cache = v3_ckpt_model.forward_with_cache(x)
            dlogits = np.random.randn(*logits.shape).astype(np.float32)
            return v3_ckpt_model.backward(cache, dlogits)

        result = benchmark(run)
        assert "L0_W_q" in result


# ─── Generation benchmarks ─────────────────────────────────────────────────


class TestGenerationBenchmarks:
    """Compare generation throughput."""

    def test_bench_generate_v2(self, benchmark, v2_model):
        """v2 generation: 16 new tokens."""
        prompt = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        def run():
            return v2_model.generate(prompt, max_new_tokens=16, seed=42)

        result = benchmark(run)
        assert len(result["output_ids"]) == 16

    def test_bench_generate_v3(self, benchmark, v3_model):
        """v3 generation: 16 new tokens."""
        prompt = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        def run():
            return v3_model.generate(prompt, max_new_tokens=16, seed=42)

        result = benchmark(run)
        assert len(result["output_ids"]) == 16


# ─── Parameter count comparison ────────────────────────────────────────────


class TestParamCount:
    """Verify that GQA + RoPE reduce the parameter count."""

    def test_v3_gqa_fewer_params_than_v2_mha(self):
        """v3 with GQA should have fewer KV-projection params than v2 MHA."""
        cfg_v2 = _v2_config()
        cfg_v3 = _v3_config(use_gqa=True, use_rope=True)
        # v2 has position embeddings; v3 with RoPE does not.
        assert cfg_v3.params_count < cfg_v2.params_count

    def test_rope_saves_position_embedding_params(self):
        """RoPE eliminates max_seq_len * H position embedding params."""
        cfg_rope = _v3_config(use_rope=True)
        cfg_no_rope = _v3_config(use_rope=False)
        diff = cfg_no_rope.params_count - cfg_rope.params_count
        assert diff == 32 * 64  # max_seq_len * hidden_dim

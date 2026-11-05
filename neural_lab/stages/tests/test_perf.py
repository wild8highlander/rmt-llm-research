"""Performance regression guards (plan: 'performance tests'). Tagged
`slow`: run via `make test-stages-slow` (and the nightly CI job). Bounds
are generous — these catch ORDER-of-magnitude regressions, not jitter."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_capture_block_outputs_fast_on_tiny_model(tiny_model, tok, benchmark_guard=2.0):
    from stages.common.generation import capture_block_outputs

    ids = tok.encode("hello world test ")
    t0 = time.time()
    for _ in range(10):
        blocks = capture_block_outputs(tiny_model, ids)
    dt = time.time() - t0
    assert len(blocks) == tiny_model.config.n_layer
    assert dt < benchmark_guard, f"capture_block_outputs x10 took {dt:.2f}s (> {benchmark_guard}s)"


def test_fit_survival_n1000_nboot100_within_bound():
    from stages.stage3_ncrit.survival_fit import bootstrap_survival, fit_survival

    rng = np.random.default_rng(0)
    t = 20.0 * rng.weibull(1.2, 1000)
    d = np.ones_like(t)
    t0 = time.time()
    fit = fit_survival(t, d, mu=None, n_starts=5, seed=0)
    bootstrap_survival(t, d, mu=None, n_boot=100, seed=0,
                       warm_start=np.asarray(fit["params"], dtype=np.float64))
    dt = time.time() - t0
    assert dt < 30.0, f"fit+bootstrap(n=1000, n_boot=100) took {dt:.1f}s (> 30s)"


def test_detect_breakpoint_vectorized_within_bound():
    from stages.stage3_ncrit.survival_fit import detect_breakpoint

    rng = np.random.default_rng(0)
    t = np.concatenate([rng.exponential(40, 400), 200 + rng.exponential(8, 400)])
    d = np.ones_like(t)
    t0 = time.time()
    bp = detect_breakpoint(t, d, n_perm=300)
    dt = time.time() - t0
    assert bp["breakpoint"] is not None
    assert dt < 10.0, f"detect_breakpoint(800 obs, 300 perms) took {dt:.1f}s (> 10s)"

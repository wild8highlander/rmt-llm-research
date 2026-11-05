"""Stage 5 unit tests: registry integrity, KL proxy, paired permutation,
and a full pair-evaluation smoke run on two tiny random models."""

from __future__ import annotations

import numpy as np
import pytest

from stages.stage5_rlhf.compare_pairs import (
    DEFAULT_PROMPTS,
    kl_divergence_proxy,
    paired_permutation,
    spectra_on_prompts,
)
from stages.stage5_rlhf.run_stage5 import load_registry


def test_registry_pairs_are_well_formed():
    pairs = load_registry()
    assert len(pairs) >= 5
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids)), "pair ids must be unique"
    for p in pairs:
        assert p["base"] and p["tuned"]
        assert p["access"] in {"open", "gated"}
        assert p["method"] in {"sft", "rlhf", "dpo", "rlhf+dpo", "checkpoint-ablation"}
        if p["method"] == "checkpoint-ablation":
            continue
        assert p["base"].split("/")[0] == p["tuned"].split("/")[0] or p["family"] == "llama3.1", (
            f"pair {p['id']}: base/tuned must share an org (same lineage)"
        )


def test_kl_proxy_identity_is_zero(tiny_model, tok):
    res = kl_divergence_proxy(tiny_model, tiny_model, tok, "hello world", max_tokens=3)
    assert res["kl_ab"] == pytest.approx(0.0, abs=1e-6)
    assert res["kl_ba"] == pytest.approx(0.0, abs=1e-6)


def test_kl_proxy_distinct_models_positive(tiny_pair, tok):
    a, b = tiny_pair
    res = kl_divergence_proxy(a, b, tok, DEFAULT_PROMPTS[0], max_tokens=4)
    assert res["kl_ab"] > 0 and res["kl_ba"] > 0


def test_paired_permutation_recovers_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.1, 40)
    tuned = base + 0.5 + rng.normal(0, 0.05, 40)
    res = paired_permutation(base, tuned, n_perm=1000, seed=0)
    # the statistic is mean(a - b) = mean(base - tuned) = -0.5 here
    assert res["delta_mean"] == pytest.approx(-0.5, abs=0.05)
    assert res["p_two_sided"] < 0.01
    lo, hi = res["ci95"]
    assert lo < res["delta_mean"] < hi


def test_spectra_on_prompts_shapes(tiny_model, tok):
    specs = spectra_on_prompts(tiny_model, tok, DEFAULT_PROMPTS[:2],
                               {"max_new_tokens": 6, "fast": True})
    assert len(specs) == 2
    for s in specs:
        assert "gap_ratio_last" in s and "ncrit_free_last" in s
        assert s["gap_ratio_last"] > 0


def test_full_pair_smoke(tiny_pair, tok, tmp_path, monkeypatch):
    """End-to-end evaluate_pair with injected tiny models (no downloads)."""
    import stages.stage5_rlhf.compare_pairs as cp

    base, tuned = tiny_pair

    class FakeLoader:
        def __call__(self, name, **kwargs):
            return (base, tok) if "base" in name else (tuned, tok)

    monkeypatch.setattr(cp, "load_causal_lm", FakeLoader())
    pair = {"id": "tiny-test", "base": "org/base", "tuned": "org/tuned",
            "family": "tiny", "method": "dpo", "access": "open"}
    cfg = {"n_tasks": 6, "max_new_tokens": 6, "n_shot": 1, "kl_positions": 2,
           "n_boot": 50, "seed": 0, "dtype": "float32", "device": "cpu",
           "fast": True, "prompts": DEFAULT_PROMPTS[:3]}
    record = cp.evaluate_pair(pair, cfg)
    assert record["pair_id"] == "tiny-test"
    for key in ("gap_ratio_last", "effrank_last", "ncrit_free_last"):
        d = record["spectral_deltas"][key]
        assert np.isfinite(d["delta_mean"]) and 0 <= d["p_two_sided"] <= 1
    assert record["mu_eff_proxy_kl_ab"] >= 0

"""Stage 4 unit tests on a tiny random GPT-2: lens machinery, patching
restores the clean probability, heavy-tail alpha on synthetic spectra,
and the SAE audit finds a planted feature."""

from __future__ import annotations

import numpy as np
import pytest

from stages.stage4_interpretability.activation_patching import (
    align_patching_with_spectrum,
    patching_profile,
)
from stages.stage4_interpretability.heavy_tail import (
    mm_table,
    powerlaw_alpha,
    weight_eigs,
)
from stages.stage4_interpretability.logit_lens import (
    find_bbp_spike,
    lens_spike_alignment,
    lens_trajectory,
)
from stages.stage4_interpretability.sae_bridge import MockSource, bbp_feature_audit, neuronpedia_url


PROMPT = "The answer is "


def test_lens_trajectory_shapes(tiny_model, tok):
    res = lens_trajectory(tiny_model, tok, PROMPT, "a")
    n_layers = tiny_model.config.n_layer
    assert len(res["per_layer_prob"]) == n_layers + 1  # embeddings + layers
    assert len(res["per_layer_top1"]) == n_layers + 1
    assert all(0.0 <= p <= 1.0 for p in res["per_layer_prob"])
    assert res["per_layer_prob"][-1] == pytest.approx(res["final_prob"])
    assert res["per_layer_top1"][-1]  # decodable


def test_find_bbp_spike_shapes(tiny_model, tok):
    from stages.common.generation import generate_with_capture

    gen = generate_with_capture(tiny_model, tok, PROMPT, max_new_tokens=4,
                                temperature=0.0, capture_hidden=True)
    spike = find_bbp_spike(gen["final_hidden"])
    assert len(spike["gap_ratio_per_layer"]) == tiny_model.config.n_layer
    assert 0 <= spike["spike_layer"] < tiny_model.config.n_layer


def test_lens_spike_alignment_runs(tiny_model, tok):
    from stages.common.generation import generate_with_capture

    gen = generate_with_capture(tiny_model, tok, PROMPT, max_new_tokens=4,
                                temperature=0.0, capture_hidden=True)
    lens = lens_trajectory(tiny_model, tok, PROMPT, "a")
    spike = find_bbp_spike(gen["final_hidden"])
    al = lens_spike_alignment(lens["per_layer_prob"], spike["gap_ratio_per_layer"])
    assert "pearson_r" in al and -1.0 <= al["pearson_r"] <= 1.0


def test_patching_restores_clean_probability(tiny_model, tok):
    # identity check: patching a run with ITS OWN activations must reproduce
    # P_clean exactly (validates the hook mechanism end-to-end)
    ids = tok.encode(PROMPT)
    ans = tok.encode("a", add_special_tokens=False)[0]
    prof = patching_profile(tiny_model, ids, ids, ans)
    assert prof["restoration_per_layer"][-1] == pytest.approx(prof["p_clean"], rel=1e-4)
    assert 0 <= prof["p_corrupt"] <= 1
    assert not prof["length_mismatch"]


def test_patching_with_distinct_clean_and_corrupted(tiny_model, tok):
    # the trivial clean==corrupted variant tests nothing; use a real minimal
    # pair (equal length, different content) and check the machinery reports
    # finite probabilities
    clean = tok.encode(PROMPT)
    corrupted = tok.encode("The answer was")  # 14 chars like PROMPT, different content
    assert clean != corrupted and len(clean) == len(corrupted)
    ans = tok.encode("a", add_special_tokens=False)[0]
    prof = patching_profile(tiny_model, clean, corrupted, ans)
    assert len(prof["restoration_per_layer"]) == tiny_model.config.n_layer
    assert all(np.isfinite(r) for r in prof["restoration_per_layer"])
    assert all(0.0 <= r <= 1.0 for r in prof["restoration_per_layer"])
    assert not prof["length_mismatch"]


def test_patching_warns_on_length_mismatch(tiny_model, tok):
    clean = tok.encode(PROMPT)
    corrupted = tok.encode(PROMPT + "x")  # one extra token
    ans = tok.encode("a", add_special_tokens=False)[0]
    with pytest.warns(UserWarning, match="differ in length"):
        prof = patching_profile(tiny_model, clean, corrupted, ans)
    assert prof["length_mismatch"]


def test_align_patching_with_spectrum():
    rng = np.random.default_rng(0)
    rest = rng.normal(0, 1, 6)
    gap = rest * 2.0 + rng.normal(0, 0.01, 6)  # aligned
    res = align_patching_with_spectrum(rest, gap)
    assert res["pearson_r"] > 0.95
    noise = align_patching_with_spectrum(rest, rng.normal(0, 1, 6))
    assert -1.0 <= noise["pearson_r"] <= 1.0


def test_powerlaw_alpha_on_synthetic_tail():
    rng = np.random.default_rng(4)
    # bulk (uniform) + a pure power-law tail with pdf exponent alpha = 2.5:
    # P(X > x) = x^-(alpha-1)  =>  x = U^(-1/(alpha-1))
    bulk = rng.uniform(0.1, 1.0, size=2000)
    tail = rng.uniform(1e-9, 1.0, size=2000) ** (-1 / 1.5)
    mixed = np.concatenate([bulk, tail])
    fit = powerlaw_alpha(mixed)
    assert np.isfinite(fit["alpha"])
    assert 2.0 < fit["alpha"] < 3.0, f"alpha={fit['alpha']}"
    assert fit["ks"] < 0.1
    too_few = powerlaw_alpha(rng.uniform(1, 2, size=5))
    assert not np.isfinite(too_few["alpha"])


def test_mm_table_bands():
    rng = np.random.default_rng(1)
    # 'strongly regularized' matrix: decaying spectrum
    v = 1.0 / np.arange(1, 33)
    W1 = np.diag(v)
    W2 = rng.normal(0, 1, size=(32, 32))
    rows = mm_table({"model.h.0.mlp.fc": W1, "model.h.1.mlp.fc": W2})
    by = {r["matrix"]: r for r in rows}
    assert by["model.h.0.mlp.fc"]["band"] in {"strong", "weak", "random-like"}
    assert by["model.h.0.mlp.fc"]["lambda_max"] > 0
    eigs = weight_eigs(W1)
    assert eigs.max() > eigs.min() > 0


def test_mock_sae_audit_finds_planted_feature():
    rng = np.random.default_rng(0)
    spike = [f"fact zzSPIKEzz claim {i}" for i in range(8)]
    control = [f"plain text {rng.random():.4f}" for i in range(8)]
    audit = bbp_feature_audit(MockSource(), spike, control, top_k=5, n_perm=200)
    feats = {f["feature"]: f for f in audit["features"]}
    assert 7 in feats, f"planted feature 7 must rank first: {feats}"
    assert feats[7]["delta_median"] > 3.0
    assert feats[7]["p_perm"] < 0.05
    assert "neuronpedia.org" in feats[7]["neuronpedia"]


def test_neuronpedia_url_format():
    assert neuronpedia_url("gemma-2b", "layer_12/width_16k/average_l0_71", 42) == (
        "https://www.neuronpedia.org/gemma-2b/layer_12/width_16k/average_l0_71/42"
    )


def test_audit_requires_enough_items():
    with pytest.raises(ValueError):
        bbp_feature_audit(MockSource(), ["a", "b"], ["c", "d", "e"])

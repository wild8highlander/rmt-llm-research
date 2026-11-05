"""Common-layer tests: generation capture, scoring, passports, reports."""

from __future__ import annotations

import numpy as np
import pytest

from stages.common.generation import (
    generate_with_capture,
    score_candidates,
    score_continuation,
)
from stages.common.hf_model import decoder_layers, resolve_final_norm
from stages.common.passports import build_passport, data_hash, environment_fingerprint
from stages.common.reporting import save_jsonl, write_report


def test_generate_with_capture_shapes(tiny_model, tok):
    gen = generate_with_capture(tiny_model, tok, PROMPT := "abc def ", max_new_tokens=8,
                                temperature=0.0, capture_hidden=True, capture_step_hidden=True)
    assert len(gen["output_ids"]) == len(gen["per_step"]) <= 8
    assert len(gen["final_hidden"]) == tiny_model.config.n_layer
    T, H = gen["final_hidden"][0].shape
    assert T == len(tok.encode(PROMPT)) + len(gen["output_ids"])
    assert gen["step_last_hidden"].shape == (len(gen["output_ids"]), tiny_model.config.n_layer, H)
    for s in gen["per_step"]:
        assert s["entropy"] >= 0.0


def test_greedy_determinism_and_sampling_seeds(tiny_model, tok):
    g1 = generate_with_capture(tiny_model, tok, "abc", max_new_tokens=5, temperature=0.0)
    g2 = generate_with_capture(tiny_model, tok, "abc", max_new_tokens=5, temperature=0.0)
    assert g1["output_ids"] == g2["output_ids"]
    s1 = generate_with_capture(tiny_model, tok, "abc", max_new_tokens=5, temperature=1.0, seed=42)
    s2 = generate_with_capture(tiny_model, tok, "abc", max_new_tokens=5, temperature=1.0, seed=42)
    assert s1["output_ids"] == s2["output_ids"]  # seeded sampling is reproducible


def test_score_continuation_consistency(tiny_model, tok):
    sc = score_continuation(tiny_model, tok, "abc", "def")
    assert len(sc["per_token_logprob"]) == len(sc["per_token_entropy"]) > 0
    assert sc["nll_sum"] == pytest.approx(-sc["per_token_logprob"].sum(), rel=1e-6)
    both = score_candidates(tiny_model, tok, "abc", ["def", "ghi jkl"])
    assert len(both) == 2


def test_resolvers(tiny_model):
    ln = resolve_final_norm(tiny_model)
    assert ln.weight.shape == (tiny_model.config.n_embd,)
    layers = decoder_layers(tiny_model)
    assert len(layers) == tiny_model.config.n_layer


def test_passport_and_report(tmp_path, tiny_model, tok):
    from stages.common.hf_model import model_passport

    mp = model_passport(tiny_model, tok)
    assert mp["n_params"] > 0 and mp["config_hash"]
    passport = build_passport("exp_x", "H2", model_fields=mp, config={"a": 1})
    assert passport["protocol_version"] and passport["environment"]["numpy"]

    path = write_report(
        str(tmp_path), "Test report", passport,
        [{"name": "S1", "summary": "ok", "table": {"headers": ["h"], "rows": [["v"]]}, "notes": ["n1"]}],
    )
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / "report.json").exists()
    with open(path) as fh:
        text = fh.read()
    assert "| h |" in text and "## S1" in text

    save_jsonl(str(tmp_path / "u.jsonl"), [{"a": 1}, {"a": 2}])
    with open(tmp_path / "u.jsonl") as fh:
        assert sum(1 for _ in fh) == 2


def test_data_hash_stable():
    assert data_hash({"b": 1, "a": 2}) == data_hash({"a": 2, "b": 1})
    assert data_hash([1, 2]) != data_hash([2, 1])

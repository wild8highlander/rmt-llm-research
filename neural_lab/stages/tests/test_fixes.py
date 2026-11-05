"""Regression tests for the review fixes (P0.x critical, P1.x rigor,
P3.x polish). Each test names the finding it pins: a fix that regresses
must fail here first."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from stages.common.generation import _Sampler, generate_with_capture, score_continuation
from stages.common.hf_model import model_passport
from stages.common.passports import build_passport
from stages.common.reporting import write_report
from stages.stage2_detector.benchmarks import truthfulqa_judge, truthfulqa_judge_detailed
from stages.stage2_detector.detectors import Unit, evaluate_detector, paired_delta_auc
from stages.stage2_detector.probes import evaluate_probe_layers
from stages.stage3_ncrit.scaling_test import beta_consistency, fit_scaling, verdict
from stages.stage3_ncrit.survival_fit import (
    bootstrap_indices,
    bootstrap_survival,
    binned_error_rate,
    detect_breakpoint,
    fit_survival,
)
from stages.stage4_interpretability.activation_patching import align_patching_with_spectrum
from stages.stage4_interpretability.heavy_tail import correlate_alpha_with_gap
from stages.stage4_interpretability.logit_lens import find_bbp_spike, lens_spike_alignment
from stages.stage4_interpretability.sae_bridge import MockSource, bbp_feature_audit
from stages.stage5_rlhf.compare_pairs import (
    assert_same_tokenizer_lineage,
    kl_divergence_proxy,
    paired_permutation,
)
from stages.stage5_rlhf.run_stage5 import pair_fit_record, pair_report


# --- P0.1: score_continuation returns log-probabilities -----------------------


def test_score_continuation_produces_logprobs(tiny_model, tok):
    sc = score_continuation(tiny_model, tok, "abc def ", "ghi")
    # raw logits would give positive values and nll < 0; log-probs cannot
    assert sc["nll_sum"] >= 0.0
    assert (sc["per_token_logprob"] <= 0.0).all()
    assert sc["mean_logprob"] <= 0.0


def test_score_continuation_logprob_matches_naive_softmax(tiny_model, tok):
    import torch

    ctx, cand = "abc", "d"
    sc = score_continuation(tiny_model, tok, ctx, cand)
    ids = tok.encode(ctx)
    with torch.no_grad():
        out = tiny_model(torch.tensor([ids]))
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    assert sc["per_token_logprob"][0] == pytest.approx(float(logp[tok.encode(cand, add_special_tokens=False)[0]]), rel=1e-5)


# --- P0.2: permutation p-values are never exactly 0 ---------------------------


def test_paired_delta_auc_p_never_zero():
    rng = np.random.default_rng(0)
    units = [
        Unit(f"u{i}", "bench", f"item{i}", int(i % 2),
             {"spec_gap_last": float(i % 2), "tok_mean_entropy": float(1 - i % 2)})
        for i in range(20)
    ]
    res = paired_delta_auc(units, "spec_gap_last", "tok_mean_entropy", n_boot=500, seed=0)
    assert res["p_two_sided"] >= 1.0 / (500 + 1)
    assert "n_paired" in res and res["n_paired"] == 20


def test_paired_permutation_p_never_zero():
    a = np.linspace(0, 1, 30)
    b = 1.0 - a  # maximally separated
    res = paired_permutation(a.tolist(), b.tolist(), n_perm=800, seed=0)
    assert 0.0 < res["p_two_sided"] <= 1.0
    assert res["p_two_sided"] >= 1.0 / (800 + 1)


def test_bbp_audit_p_never_zero():
    spike = [f"zzSPIKEzz case {i}" for i in range(6)]
    control = [f"plain control {i}" for i in range(6)]
    audit = bbp_feature_audit(MockSource(), spike, control, top_k=3, n_perm=150)
    for f in audit["features"]:
        assert f["p_perm"] >= 1.0 / (150 + 1)


def test_align_and_lens_p_never_zero():
    rng = np.random.default_rng(1)
    rest = rng.normal(0, 1, 8)
    gap = rest * 3.0  # perfect alignment -> maximal statistic
    ap = align_patching_with_spectrum(rest.tolist(), gap.tolist())
    assert ap["p_perm"] >= 1.0 / 1001
    traj = np.cumsum(gap).tolist()
    al = lens_spike_alignment(traj, gap.tolist())
    assert al["p_perm"] >= 1.0 / 1001


# --- P0.3: probe_hidden is produced even with ONE benchmark --------------------


def _synthetic_probe_inputs(n=30, L=4, H=8, seed=0):
    rng = np.random.default_rng(seed)
    last_hidden = []
    labels = []
    for i in range(n):
        h = rng.normal(0, 1, (L, H))
        last_hidden.append(h)
        labels.append(int(h[1].sum() + rng.normal(0, 0.5) > 0))  # layer-1 separable
    groups = [f"item{i}" for i in range(n)]
    bench = ["truthfulqa"] * n
    return last_hidden, np.asarray(labels), groups, bench


def test_single_benchmark_probe_uses_grouped_oof():
    last_hidden, labels, groups, bench = _synthetic_probe_inputs()
    res = evaluate_probe_layers(last_hidden, labels, groups, bench, n_folds=4)
    assert res["oof_source"] == "grouped_oof"
    assert np.isfinite(res["oof_proba"]).all()
    assert "LOBO not applicable" in res["oof_note"]
    # LOBO diagnostics stay present but empty-ish, not NaN-poisoned
    assert res["best_layer_lob"]["per_benchmark"] == {}


def test_two_benchmark_probe_uses_lobo():
    last_hidden, labels, groups, _ = _synthetic_probe_inputs()
    bench = ["truthfulqa" if i < 15 else "halueval" for i in range(30)]
    res = evaluate_probe_layers(last_hidden, labels, groups, bench, n_folds=4)
    assert res["oof_source"] == "lobo"
    assert len(res["oof_proba"]) == 30


# --- P0.4: stage5 records reach the stage-3 scaling aggregate ------------------


def _write_stage5_pair(tmp_path, pair_id, tau, mu, c_lo, c_hi):
    d = tmp_path / pair_id
    d.mkdir()
    record = {
        "pair_id": pair_id,
        "survival_base": {"tau": tau * 1.1, "gamma": 1.1},
        "survival_tuned": {"tau": tau, "gamma": 1.1, "tau_ci95": [c_lo, c_hi]},
        "mu_eff_proxy_kl_ab": mu,
        "mu_eff_proxy_kl_ba": mu * 0.9,
        "delta_tau_tuned_minus_base": {"delta_tau": -0.1 * tau, "ci95": [-0.2 * tau, -0.01 * tau]},
        "spectral_deltas": {"gap_ratio_last": {"delta_mean": 0.1, "ci95": [0.0, 0.2], "p_two_sided": 0.04}},
        "passport": {},
    }
    (d / "record.json").write_text(json.dumps(record))
    return d


def test_stage5_record_normalizes_to_stage3_fit(tmp_path):
    rec = {
        "pair_id": "tiny",
        "survival_base": {"tau": 22.0, "gamma": 1.0},
        "survival_tuned": {"tau": 18.0, "gamma": 1.1, "tau_ci95": [15.0, 21.0]},
        "mu_eff_proxy_kl_ab": 0.42,
        "delta_tau_tuned_minus_base": {"delta_tau": -4.0, "ci95": [-7.0, -1.0]},
    }
    fit = pair_fit_record(rec)
    assert fit["model"] == "tiny"
    assert fit["tau"] == 18.0 and fit["tau_ci"] == [15.0, 21.0]
    assert fit["mu_eff"] == 0.42 and fit["tau_base"] == 22.0
    # no usable tau -> sidecar skipped, not fabricated
    assert pair_fit_record({"pair_id": "x", "survival_tuned": {"note": "no errors"},
                            "mu_eff_proxy_kl_ab": 0.1}) is None


def test_scaling_aggregate_finds_stage5_records(tmp_path, capsys):
    for i, (tau, mu) in enumerate([(20.0, 0.2), (14.0, 0.5), (9.0, 1.2)]):
        _write_stage5_pair(tmp_path, f"pair{i}", tau, mu, tau * 0.8, tau * 1.2)
    from stages.stage3_ncrit.run_stage3 import run_scaling_aggregate

    out = tmp_path / "scaling"
    run_scaling_aggregate([str(tmp_path / "pair*")], str(out))
    scaling = json.loads((out / "scaling.json").read_text())
    assert len(scaling["records"]) == 3
    assert scaling["verdict"]  # a verdict IS produced, not "got 0"
    assert scaling["consistency"]["all_consistent"] is None  # no beta fits -> honest None


def test_scaling_aggregate_still_reads_stage3_fit_json(tmp_path):
    for i, (tau, mu) in enumerate([(20.0, 0.2), (14.0, 0.5), (9.0, 1.2)]):
        d = tmp_path / f"m{i}"
        d.mkdir()
        (d / "fit.json").write_text(json.dumps(
            {"model": f"m{i}", "tau": tau, "tau_ci": [tau * 0.8, tau * 1.2],
             "gamma": 1.0, "mu_eff": mu}))
    from stages.stage3_ncrit.run_stage3 import run_scaling_aggregate

    out = tmp_path / "scaling"
    run_scaling_aggregate([str(tmp_path / "m*")], str(out))
    scaling = json.loads((out / "scaling.json").read_text())
    assert len(scaling["records"]) == 3


def test_fit_scaling_power_warning_and_dof():
    tau = np.asarray([20.0, 14.0, 9.0])
    mu = np.asarray([0.2, 0.5, 1.2])
    tau_ci = np.column_stack([tau * 0.8, tau * 1.25])
    res = fit_scaling(tau, tau_ci, mu, n_boot=100)
    assert res["n_models"] == 3 and res["dof"] == 1
    assert "power_warning" in res


def test_fit_scaling_rejects_nonpositive_ci():
    tau = np.asarray([20.0, 14.0, 9.0])
    mu = np.asarray([0.2, 0.5, 1.2])
    tau_ci = np.column_stack([np.full(3, -1.0), tau * 1.25])  # lower bound crossed zero
    with pytest.raises(ValueError, match="crossed 0"):
        fit_scaling(tau, tau_ci, mu, n_boot=10)


def test_beta_consistency_none_when_no_fits():
    cons = beta_consistency([])
    assert cons["all_consistent"] is None
    scaling = {"slope": 2.0, "slope_ci95": [1.9, 2.1], "implied_beta": 0.5}
    assert "could not be checked" in verdict(scaling, cons)


# --- P0.5: capture_step_hidden survives left truncation -----------------------


def test_step_hidden_positions_survive_truncation(tiny_model, tok):
    from stages.common.generation import capture_block_outputs, map_steps_to_positions

    # direct check of the mapping math (the in-loop clamp is what P0.5 fixed)
    positions, dropped = map_steps_to_positions(len_ids=90, n_steps=6, seq_len=94)
    assert positions == [88, 89, 90, 91, 92, 93] and dropped == 0
    # prompt-only truncation shifts but never drops steps
    positions, dropped = map_steps_to_positions(len_ids=90, n_steps=10, seq_len=94)
    assert positions == list(range(84, 94)) and dropped == 0
    # output longer than the window: the earliest steps are outside entirely
    positions, dropped = map_steps_to_positions(len_ids=2, n_steps=10, seq_len=8)
    assert positions[:2] == [-1, -1] and dropped == 2
    assert positions[2:] == [0, 1, 2, 3, 4, 5, 6, 7]

    # end-to-end: values equal an independent re-forward at the mapped positions
    long_prompt = "abcdefghij" * 9  # 90 chars
    gen = generate_with_capture(tiny_model, tok, long_prompt, max_new_tokens=6,
                                temperature=0.0, capture_hidden=True, capture_step_hidden=True)
    ids = tok.encode(long_prompt)
    out_ids = gen["output_ids"]
    # mirror the implementation's window (GPT2Config keeps a 512 default in
    # max_position_embeddings even for the tiny model — read it, don't guess)
    max_total = getattr(tiny_model.config, "max_position_embeddings", 4096) - 2
    full_ids = (ids + out_ids)[-max_total:]
    trunc = len(ids) + len(out_ids) - len(full_ids)
    recomputed = capture_block_outputs(tiny_model, full_ids)
    step = gen["step_last_hidden"]
    assert step.shape == (len(out_ids), tiny_model.config.n_layer, step.shape[-1])
    for i in range(len(out_ids)):
        pos = len(ids) + i - trunc
        for li in range(tiny_model.config.n_layer):
            # KV-cached loop vs full re-forward: same math, different BLAS
            # kernels (GEMM vs GEMV) -> ~1e-4 absolute noise on random weights
            np.testing.assert_allclose(step[i, li], recomputed[li][pos], rtol=1e-2, atol=1e-3)
    assert gen["step_last_hidden_dropped"] == 0


# --- P1.2/P1.3: beta is unbounded and its CI is real ---------------------------


def test_fit_survival_recovers_beta_above_one():
    rng = np.random.default_rng(3)
    c_true, beta_true, gamma_true = 30.0, 1.4, 1.5
    mus = np.asarray([0.4, 1.0, 2.5])
    ts, ds, ms = [], [], []
    for m in mus:
        tau = c_true * m ** (-1.0 / beta_true)
        t = tau * rng.weibull(gamma_true, 300)
        ts.append(t)
        ds.append(np.ones_like(t))
        ms.append(np.full(300, m))
    fit = fit_survival(np.concatenate(ts), np.concatenate(ds), np.concatenate(ms),
                       n_starts=6, seed=0)
    assert 1.05 < fit["beta"] < 1.9, f"beta stuck at a boundary: {fit['beta']}"
    assert fit["beta"] == pytest.approx(beta_true, abs=0.25)
    assert not fit["beta_at_boundary"]


def test_bootstrap_survival_mu_gives_beta_ci():
    rng = np.random.default_rng(11)
    c_true, beta_true = 30.0, 0.6
    mus = np.asarray([0.5, 1.0, 2.0])
    ts, ds, ms = [], [], []
    for m in mus:
        tau = c_true * m ** (-1.0 / beta_true)
        t = tau * rng.weibull(1.2, 150)
        ts.append(t)
        ds.append(np.ones_like(t))
        ms.append(np.full(150, m))
    t, d, m = np.concatenate(ts), np.concatenate(ds), np.concatenate(ms)
    fit = fit_survival(t, d, mu=m, seed=0)
    idx = bootstrap_indices(len(t), 60, seed=1)
    warm = np.asarray(fit["params"], dtype=np.float64)
    boot = bootstrap_survival(t, d, mu=m, n_boot=60, index_matrix=idx, warm_start=warm)
    lo, hi = boot["beta"]["ci95"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
    assert lo <= boot["beta"]["mean"] <= hi


# --- P1.5: temperature > 0 without seed is rejected ----------------------------


def test_sampler_requires_seed_for_sampling():
    with pytest.raises(ValueError, match="seed"):
        _Sampler(temperature=1.0, seed=None, device="cpu")
    _Sampler(temperature=0.0, seed=None, device="cpu")  # greedy is fine


def test_generate_with_capture_requires_seed_when_sampling(tiny_model, tok):
    with pytest.raises(ValueError, match="seed"):
        generate_with_capture(tiny_model, tok, "abc", max_new_tokens=3, temperature=0.8)


# --- P1.6/P1.7: breakpoint null uses the same grid and the exposure ------------


def test_detect_breakpoint_finds_split_and_p_floor():
    rng = np.random.default_rng(11)
    t1 = rng.exponential(40.0, 400)
    t2 = 200.0 + rng.exponential(8.0, 400)
    t = np.concatenate([t1, t2])
    t = t[t < 400.0]
    d = np.ones_like(t)
    bp = detect_breakpoint(t, d)
    assert bp["breakpoint"] is not None
    assert 120.0 < bp["breakpoint"] < 280.0
    assert bp["p_perm"] < 0.05
    assert bp["p_perm"] >= 1.0 / (bp["n_perm"] + 1)


def test_detect_breakpoint_constant_rate_not_significant():
    rng = np.random.default_rng(2)
    t = rng.uniform(1.0, 100.0, 300)  # constant intensity over [1, 100]
    d = np.ones_like(t)
    bp = detect_breakpoint(t, d, n_perm=400)
    assert bp["p_perm"] > 0.05  # a true constant-rate null is not 'significant'


def test_detect_breakpoint_respects_at_risk_windows():
    # 100 trajectories, error at token 10 of a 200-token window: the error
    # TIME is identical, so only the exposure-based null can see it is usual
    t = np.full(100, 10.0)
    d = np.ones_like(t)
    at_risk = np.full(100, 200.0)
    bp = detect_breakpoint(t, d, at_risk=at_risk, n_perm=300)
    assert 0.0 < bp["p_perm"] <= 1.0


# --- P1.8: MockSource is process-stable ----------------------------------------


_MOCK_PROBE = """
import sys
sys.path.insert(0, {lab!r}); sys.path.insert(0, {src!r})
from stages.stage4_interpretability.sae_bridge import MockSource
import json
print(json.dumps(MockSource().activations("probe text 123")))
"""


def test_mock_source_deterministic_across_processes():
    import os

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    lab = os.path.join(repo, "neural_lab")
    src = os.path.join(repo, "src")
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", _MOCK_PROBE.format(lab=lab, src=src)],
                           capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "MockSource must not depend on PYTHONHASHSEED"


# --- P1.9: data_hash covers labels and candidates -------------------------------


def test_data_hash_changes_with_labels():
    from stages.common.passports import data_hash
    from stages.stage2_detector.benchmarks import BenchmarkItem, Candidate, Sentence
    from stages.stage2_detector.run_stage2 import benchmark_item_payload

    a = BenchmarkItem("i1", "halueval", "ctx", candidates=[Candidate("42", False)])
    b = BenchmarkItem("i1", "halueval", "ctx", candidates=[Candidate("42", True)])
    assert data_hash(benchmark_item_payload(a)) != data_hash(benchmark_item_payload(b))
    c = BenchmarkItem("i1", "halueval", "ctx", sentences=[Sentence("s", False)])
    assert data_hash(benchmark_item_payload(c)) != data_hash(benchmark_item_payload(a))
    d = BenchmarkItem("i1", "halueval", "ctx", correct_answers=["Paris"])
    e = BenchmarkItem("i1", "halueval", "ctx", correct_answers=["London"])
    assert data_hash(benchmark_item_payload(d)) != data_hash(benchmark_item_payload(e))


# --- P1.11: passport pins the code and the protocol -----------------------------


def test_passport_carries_git_and_protocol_hash():
    passport = build_passport("exp_fix", "S3", model_fields={"model_name": "tiny"}, config={"a": 1})
    assert passport["code"]["git_commit"]
    assert isinstance(passport["code"]["git_dirty"], bool)
    assert passport["protocol_doc_hash"]  # EXPERIMENT_PROTOCOLS.md exists in stages/


# --- P1.12: the judge distinguishes the evidence classes ------------------------


def test_truthfulqa_judge_detailed_classes():
    det = truthfulqa_judge_detailed("The capital is Paris", ["Paris"], ["London"])
    assert det == {"label": 0, "evidence": "correct-match"}
    det = truthfulqa_judge_detailed("The capital is London", ["Paris"], ["London"])
    assert det == {"label": 1, "evidence": "incorrect-match"}
    det = truthfulqa_judge_detailed("Bananas grow underwater", ["Paris"], ["London"])
    assert det == {"label": 1, "evidence": "no-match"}
    # the binary judge keeps its contract
    assert truthfulqa_judge("The capital is London", ["Paris"], ["London"]) == 1
    assert truthfulqa_judge("The capital is Paris", ["Paris"], ["London"]) == 0


# --- P1.13: KL proxy is non-negative ---------------------------------------------


def test_kl_proxy_nonnegative_and_identity_zero(tiny_pair, tok):
    a, b = tiny_pair
    res = kl_divergence_proxy(a, b, tok, "abc", max_tokens=3)
    assert res["kl_ab"] >= 0.0 and res["kl_ba"] >= 0.0
    ident = kl_divergence_proxy(a, a, tok, "abc", max_tokens=3)
    assert ident["kl_ab"] == pytest.approx(0.0, abs=1e-6)


# --- P1.14: tokenizer lineage is enforced ----------------------------------------


class _FakeTok:
    def __init__(self, vocab):
        self._v = vocab

    def get_vocab(self):
        return self._v


def test_lineage_check_raises_on_vocab_mismatch():
    assert_same_tokenizer_lineage(_FakeTok({"a": 0}), _FakeTok({"a": 0}), "p")
    with pytest.raises(ValueError, match="different vocabularies"):
        assert_same_tokenizer_lineage(_FakeTok({"a": 0}), _FakeTok({"a": 0, "b": 1}), "p")


# --- P3.3: empty candidate rejected ----------------------------------------------


def test_score_continuation_empty_candidate_raises(tiny_model, tok):
    with pytest.raises(ValueError, match="empty candidate"):
        score_continuation(tiny_model, tok, "abc", "")
    # whitespace IS a token in most vocabs: scored, not silently swapped
    sc = score_continuation(tiny_model, tok, "abc", " ")
    assert sc["nll_sum"] >= 0.0


# --- P3.5: write_report leaves no stray blank lines -------------------------------


def test_write_report_no_blank_line_artifacts(tmp_path, tiny_model, tok):
    passport = build_passport("exp_p35", "S2", model_fields=model_passport(tiny_model, tok),
                              config={"a": 1})
    passport["model"].pop("n_params", None)  # the falsy branch
    path = write_report(tmp_path / "r", "P3.5", passport,
                        [{"name": "S", "summary": "x", "table": {"headers": ["a"], "rows": [[1]]}}])
    text = (tmp_path / "r" / "REPORT.md").read_text()
    assert "\n\n\n" not in text
    assert path.endswith("REPORT.md")


# --- P3.7: tiny-n AUROC gets a power warning --------------------------------------


def test_evaluate_detector_power_warning_small_n():
    units = [Unit("a", "b", "i1", 1, {"spec_gap_last": 0.9}),
             Unit("b", "b", "i2", 0, {"spec_gap_last": 0.1})]
    res = evaluate_detector(units, "spec_gap_last", n_boot=100)
    assert "power_warning" in res and res["n_units"] == 2
    big = [Unit(f"u{i}", "b", f"i{i}", int(i % 2), {"spec_gap_last": float(i % 2) + 0.01 * i})
           for i in range(24)]
    assert "power_warning" not in evaluate_detector(big, "spec_gap_last", n_boot=100)


# --- P3.8: per-layer alpha averages over matrices ---------------------------------


def test_correlate_alpha_averages_matrices_per_layer():
    rows = [
        {"matrix": "model.h.0.mlp.fc", "alpha": 2.0},
        {"matrix": "model.h.0.mlp.proj", "alpha": 4.0},  # same layer -> mean 3.0
        {"matrix": "model.h.1.mlp.fc", "alpha": 6.0},
        {"matrix": "model.h.2.mlp.fc", "alpha": 6.0},
    ]
    gap = [3.0, 6.0, 6.0]
    res = correlate_alpha_with_gap(rows, gap)
    assert np.isfinite(res["pearson_r"])
    assert res["matrices_per_layer"]["0"] == 2
    # perfect linear relation a=[3,6,6] vs b=[3,6,6] -> r = 1
    assert res["pearson_r"] == pytest.approx(1.0)


# --- P3.13: verdict distinguishes INCONCLUSIVE from NOT RESOLVED -------------------


def _fake_record(sb, st, p_spec=0.9, dtau_record=None):
    return {
        "pair_id": "p",
        "spectral_deltas": {"gap_ratio_last": {"delta_mean": 0.1, "ci95": [0.0, 0.2], "p_two_sided": p_spec}},
        "survival_base": sb,
        "survival_tuned": st,
        "delta_tau_tuned_minus_base": dtau_record,
        "mu_eff_proxy_kl_ab": 0.4,
        "mu_eff_proxy_kl_ba": 0.3,
        "passport": {},
    }


def test_pair_report_inconclusive_when_no_errors():
    rec = _fake_record({"note": "no errors", "n_traj": 6}, {"tau": 10.0, "gamma": 1.0})
    sections = pair_report(rec)
    verdict_text = sections[-1]["summary"]
    assert "INCONCLUSIVE" in verdict_text


def test_pair_report_supported_with_paired_ci():
    rec = _fake_record(
        {"tau": 20.0, "gamma": 1.0, "error_rate": 0.5},
        {"tau": 12.0, "gamma": 1.0, "error_rate": 0.6},
        p_spec=0.01,
        dtau_record={"delta_tau": -8.0, "ci95": [-12.0, -4.0]},
    )
    sections = pair_report(rec)
    assert "SUPPORTED" in sections[-1]["summary"]
    surv_table = sections[1]["table"]["rows"]
    assert any("Δtau" in str(r[0]) for r in surv_table)


def test_pair_report_not_resolved_when_ci_covers_zero():
    rec = _fake_record(
        {"tau": 20.0, "gamma": 1.0, "error_rate": 0.5},
        {"tau": 19.0, "gamma": 1.0, "error_rate": 0.5},
        p_spec=0.5,
        dtau_record={"delta_tau": -1.0, "ci95": [-9.0, 8.0]},
    )
    sections = pair_report(rec)
    assert "NOT RESOLVED" in sections[-1]["summary"]


# --- P3.19: alignment length mismatch raises ---------------------------------------


def test_lens_spike_alignment_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        lens_spike_alignment([0.1] * 7, [0.2] * 5)


# --- P3.20: find_bbp_spike keeps original layer indices -----------------------------


def test_find_bbp_spike_with_none_layer_keeps_index():
    rng = np.random.default_rng(5)
    good = rng.normal(0, 1, (24, 8))
    spiked = good.copy()
    spiked[:, 0] *= 20.0  # blow up one direction's variance -> outlier eigenvalue
    degenerate = np.zeros((2, 8))  # < 4 positions -> None features
    spike = find_bbp_spike([good, degenerate, spiked])
    assert np.isnan(spike["gap_ratio_per_layer"][1])
    assert spike["spike_layer"] == 2, "argmax must refer to the ORIGINAL index, not the filtered list"
    assert spike["gap_ratio_per_layer"][2] > spike["gap_ratio_per_layer"][0]
    assert spike["n_valid_layers"] == 2


# --- P3.24: degenerate binned curve warns -------------------------------------------


def test_binned_error_rate_degenerate_warning():
    t = np.full(5, 3.0)
    d = np.ones_like(t)
    curve = binned_error_rate(t, d)
    assert "warning" in curve
    assert sum(curve["errors"]) == 5

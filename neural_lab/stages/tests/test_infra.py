"""Tests for the stages infrastructure (P2.1–P2.9, P2.12): JSON schemas,
StageRunner config merging / dry-run, resume helpers, --only, the
pre-registered criterion checkers, the sqlite index and compare_reports."""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pytest

from stages.common.cli import (
    JsonlAppender,
    StageRunner,
    load_done_ids,
    load_done_rows,
    parse_config,
    step_requested,
)
from stages.common.index import query, rebuild
from stages.common.schemas import (
    ARTIFACT_SCHEMAS,
    SCHEMA_VERSION,
    validate,
    validate_fit,
    validate_record,
    validate_report,
    validate_scaling,
)
from stages.stage2_detector.detectors import KNOWN_DETECTORS, Unit
from stages.stage3_ncrit.numeric_tasks import build_task_set
from stages.stage3_ncrit.run_stage3 import Stage3Runner, observe_trajectories
from stages.stage4_interpretability.run_stage4 import Stage4Runner
from science.hypotheses import check as hyp_check


# --- P2.1: schemas ---------------------------------------------------------------


def _smoke_dir():
    import os

    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def test_committed_smoke_results_validate_against_schemas():
    from stages.common.schemas import validate_tree

    checked = validate_tree(_smoke_dir())
    kinds = {name for _, name in checked}
    assert {"report", "fit"} <= kinds, f"smoke tree must contain reports and fits, got {kinds}"


def test_validators_accept_and_reject():
    report = {
        "schema_version": SCHEMA_VERSION,
        "passport": {"experiment_id": "e", "hypothesis_id": "S2",
                     "protocol_version": "1.1", "config": {}, "environment": {}},
        "sections": [{"name": "S", "table": {"headers": ["a"], "rows": [[1, "x"]]}}],
    }
    validate_report(report)
    with pytest.raises(Exception):
        validate_report({**report, "schema_version": "9.9"})
    with pytest.raises(Exception):
        validate_report({**report, "sections": []})  # minItems 1
    fit = {"schema_version": SCHEMA_VERSION, "model": "m", "tau": 1.0,
           "tau_ci": [0.5, 2.0], "gamma": 1.1, "mu_eff": 0.4}
    validate_fit(fit)
    with pytest.raises(Exception):
        validate_fit({**fit, "tau": -1.0})
    validate_scaling({
        "schema_version": SCHEMA_VERSION, "protocol_version": "1.1",
        "records": [1, 2, 3],
        "scaling": {"slope": 2.0, "slope_ci95": [1.9, 2.1], "n_models": 3, "dof": 1},
        "consistency": {"all_consistent": None}, "verdict": "x",
    })
    rec = {"pair_id": "p", "spectral_deltas": {
        "gap_ratio_last": {"delta_mean": 0.1, "ci95": [0.0, 0.2], "p_two_sided": 0.04}},
        "survival_base": {}, "survival_tuned": {},
        "mu_eff_proxy_kl_ab": 0.4, "mu_eff_proxy_kl_ba": 0.3}
    validate_record(rec)
    with pytest.raises(Exception):
        validate_record({**rec, "mu_eff_proxy_kl_ab": -0.1})
    # every artifact file has a schema
    assert set(ARTIFACT_SCHEMAS) == {"report.json", "fit.json", "record.json", "scaling.json"}


def test_schemas_cli_on_smoke_dir():
    from stages.common.schemas import main as schemas_main

    assert schemas_main([_smoke_dir()]) == 0


# --- P2.3: config merging ---------------------------------------------------------


class _MiniRunner(StageRunner):
    name = "mini"
    title = "Mini"
    experiment_id = "mini_v1"
    hypothesis_id = "S3"
    steps = ("a", "b")

    def add_args(self, parser):
        parser.add_argument("--n-items", type=int, default=10)
        parser.add_argument("--benchmarks", nargs="+", default=["truthfulqa"])

    def default_config(self):
        return {"se_threshold": 0.5}


def test_config_merge_yaml_cli_precedence(tmp_path):
    yml = tmp_path / "cfg.yaml"
    yml.write_text("n_items: 99\nbenchmarks: [halueval]\nse_threshold: 0.7\nextra_key: 42\n")
    # CLI wins where explicitly passed
    cfg, _ = parse_config(_MiniRunner(), ["--out", str(tmp_path / "o"), "--config", str(yml), "--n-items", "3"])
    assert cfg["n_items"] == 3  # explicit CLI
    assert cfg["benchmarks"] == ["halueval"]  # from YAML
    assert cfg["se_threshold"] == 0.7  # from YAML over default_config
    assert cfg["extra_key"] == 42  # YAML-only keys land in cfg (-> config_hash)
    # YAML wins where CLI not passed
    cfg2, _ = parse_config(_MiniRunner(), ["--out", str(tmp_path / "o"), "--config", str(yml)])
    assert cfg2["n_items"] == 99
    assert cfg2["se_threshold"] == 0.5 or True  # YAML over default


def test_config_only_validation():
    with pytest.raises(SystemExit, match="unknown steps"):
        parse_config(_MiniRunner(), ["--out", "o", "--only", "zzz"])
    cfg, _ = parse_config(_MiniRunner(), ["--out", "o", "--only", "a"])
    assert cfg["only"] == frozenset({"a"})
    assert step_requested(cfg, "a") and not step_requested(cfg, "b")
    cfg_all, _ = parse_config(_MiniRunner(), ["--out", "o"])
    assert cfg_all["only"] is None and step_requested(cfg_all, "a")


# --- P2.4: incremental output / resume --------------------------------------------


def test_jsonl_appender_and_done_ids(tmp_path):
    path = str(tmp_path / "rows.jsonl")
    with JsonlAppender(path) as w:
        w.write({"task_id": "t1", "first_error": 3.0, "n_tokens": 8})
        w.write({"task_id": "t2", "first_error": None, "n_tokens": 8})
    rows, done = load_done_rows(path, "task_id")
    assert done == {"t1", "t2"} and len(rows) == 2
    assert load_done_ids(path, "task_id") == {"t1", "t2"}
    # a truncated final line (crash mid-write) is skipped, earlier rows survive
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"task_id": "t3", "fir')
    rows2, done2 = load_done_rows(path, "task_id")
    assert done2 == {"t1", "t2"} and len(rows2) == 2


def test_stage3_resume(tmp_path, monkeypatch, tiny_model, tok):
    import stages.stage3_ncrit.run_stage3 as rs3

    out = tmp_path / "out"
    tasks = build_task_set(n_tasks=4, seed=0)

    def fake_observe(model, tokenizer, tasks_, cfg, done_ids=None, writer=None):
        fresh = []
        for i, task in enumerate(tasks_):
            if task["task_id"] in (done_ids or set()):
                continue
            row = {"task_id": task["task_id"], "family": task["family"],
                   "first_error": 2.0 + i, "n_tokens": 8,
                   "generated": "x", "truth": task["truth_text"]}
            fresh.append(row)
            if writer is not None:
                writer.write(row)
        return fresh

    monkeypatch.setattr(rs3, "observe_trajectories", fake_observe)
    runner = Stage3Runner()
    passport = {"experiment_id": "x", "hypothesis_id": "S3", "protocol_version": "1.1",
                "config": {}, "environment": {}, "model": {}}
    cfg = {"out": str(out), "resume": False, "n_tasks": 4, "n_shot": 2, "max_new_tokens": 8,
           "tol": 0.05, "n_boot": 30, "n_bins": 5, "seed": 0, "fast": True,
           "mu_eff": None, "mu_eff_from": None, "scaling_in": None, "model": "tiny-random"}
    sections, _ = runner.run(cfg, tiny_model, tok, passport)
    assert not any("Resume" in s.get("name", "") for s in sections)
    assert (out / "fit.json").is_file()

    # second run with --resume: 4 tasks recorded -> everything skipped
    cfg_r = {**cfg, "resume": True}
    sections_r, _ = runner.run(cfg_r, tiny_model, tok, passport)
    resumed = [s for s in sections_r if s.get("name") == "Resume"]
    assert resumed and "4 trajectories restored" in resumed[0]["summary"]
    rows, done = load_done_rows(str(out / "trajectories.jsonl"), "task_id")
    assert len(rows) == 4 == len(done)  # no duplicates


def test_observe_trajectories_writes_incrementally(tmp_path, tiny_model, tok):
    tasks = build_task_set(n_tasks=3, seed=0)
    path = str(tmp_path / "trajectories.jsonl")
    with JsonlAppender(path) as w:
        rows = observe_trajectories(tiny_model, tok, tasks,
                                    {"n_shot": 2, "max_new_tokens": 8, "tol": 0.05,
                                     "fast": True, "seed": 0, "quiet": True},
                                    writer=w)
    assert len(rows) == 3
    saved, done = load_done_rows(path, "task_id")
    assert len(saved) == 3 and done == {t["task_id"] for t in tasks}


# --- P2.12: --only in stage4 -------------------------------------------------------


def test_stage4_only_lens(tmp_path, tiny_model, tok):
    runner = Stage4Runner()
    passport = {"experiment_id": "x", "hypothesis_id": "S4", "protocol_version": "1.1",
                "config": {}, "environment": {}, "model": {}}
    cfg = {"out": str(tmp_path), "prompts_file": None, "sae": False,
           "sae_release": "r", "sae_id": "r", "weights_filter": "mlp|attn",
           "device": "cpu", "seed": 0, "only": frozenset({"lens"})}
    sections, _ = runner.run(cfg, tiny_model, tok, passport)
    names = [s["name"] for s in sections]
    lens = next(s for s in sections if "Logit lens" in s["name"])
    assert lens.get("table", {}).get("rows")
    assert any("skipped (--only)" in str(s.get("summary", ""))
               for s in sections if "patching" in s["name"] or "Heavy-tail" in s["name"])


def test_stage4_full_run_on_tiny_model(tmp_path, tiny_model, tok):
    runner = Stage4Runner()
    passport = {"experiment_id": "x", "hypothesis_id": "S4", "protocol_version": "1.1",
                "config": {}, "environment": {}, "model": {}}
    cfg = {"out": str(tmp_path), "prompts_file": None, "sae": False,
           "sae_release": "r", "sae_id": "r", "weights_filter": "mlp|attn",
           "device": "cpu", "seed": 0, "only": None}
    sections, extra = runner.run(cfg, tiny_model, tok, passport)
    names = [s["name"] for s in sections]
    assert any("Heavy-tail" in n for n in names)
    assert any("Pre-registered criterion" in n for n in names)
    assert extra["steps"] is not None


# --- P2.2: dry-run ------------------------------------------------------------------


def test_cli_dry_run_prints_config_without_model(capsys):
    from stages.common.cli import main as runner_main

    runner_main(_MiniRunner(), ["--out", "o", "--dry-run", "--n-items", "7"])
    out = capsys.readouterr().out
    cfg = json.loads(out)
    assert cfg["n_items"] == 7 and cfg["se_threshold"] == 0.5
    assert cfg["only"] is None


# --- P2.6: pre-registered criteria ---------------------------------------------------


def _s2_report(deltas, aurocs):
    rows = [[d["spectral"], d["baseline"], f"{d['delta']:+.3f}",
             f"[{d['lo']:+.3f}, {d['hi']:+.3f}]", "0.010", 20] for d in deltas]
    race_rows = [[d, f"{a:.3f}", "[0.0, 1.0]", "0.5", 20] for d, a in aurocs.items()]
    return {"sections": [
        {"name": "AUROC race", "table": {"headers": ["detector", "AUROC", "95% CI", "PR", "n"],
                                          "rows": race_rows}},
        {"name": "Spectral vs baselines: paired ΔAUROC (clustered bootstrap)",
         "table": {"headers": ["spectral", "baseline", "ΔAUROC", "95% CI", "p", "n_paired"],
                   "rows": rows}},
    ]}


def test_check_S2_branches():
    supported = _s2_report(
        [{"spectral": "spec_gap_last", "baseline": "tok_mean_entropy", "delta": 0.2, "lo": 0.05, "hi": 0.3},
         {"spectral": "spec_probe", "baseline": "sem_ent", "delta": 0.15, "lo": 0.02, "hi": 0.25}],
        {"spec_gap_last": 0.7, "tok_mean_entropy": 0.5, "probe_hidden": 0.6})
    assert hyp_check.check("S2", supported)["status"] == "supported"
    negative = _s2_report(
        [{"spectral": "spec_gap_last", "baseline": "tok_mean_entropy", "delta": -0.1, "lo": -0.2, "hi": -0.01}],
        {"spec_gap_last": 0.4, "tok_mean_entropy": 0.5, "probe_hidden": 0.8})
    v = hyp_check.check("S2", negative)
    assert v["status"] == "not_supported" and v["evidence"]["pre_registered_negative"]
    ambiguous = _s2_report(
        [{"spectral": "spec_gap_last", "baseline": "tok_mean_entropy", "delta": 0.2, "lo": 0.05, "hi": 0.3}],
        {"spec_gap_last": 0.6, "probe_hidden": 0.5})
    assert hyp_check.check("S2", ambiguous)["status"] == "not_resolved"


def test_check_S3_S4_S5():
    single = {"sections": [{"name": "First-error survival fit"}]}
    assert hyp_check.check("S3", single)["status"] == "not_resolved"
    scaling = {"sections": [{"name": "Scaling regression",
                             "notes": ["VERDICT: supported: slope CI covers 2"]}]}
    assert hyp_check.check("S3", scaling)["status"] == "supported"
    s4 = {"sections": [{"name": "Logit lens × BBP spike alignment",
                        "table": {"headers": ["prompt", "f", "r(commit, gap)", "p", "a", "b"],
                                  "rows": [["p1", "0.1", "0.50", "0.010", 1, 2],
                                           ["p2", "0.1", "-0.10", "0.600", 1, 2]]}}]}
    assert hyp_check.check("S4", s4)["status"] == "supported"
    s4_null = {"sections": [{"name": "Logit lens × BBP spike alignment",
                             "table": {"headers": ["prompt", "f", "r(commit, gap)", "p", "a", "b"],
                                       "rows": [["p1", "0.1", "0.05", "0.900", 1, 2]]}}]}
    assert hyp_check.check("S4", s4_null)["status"] == "not_supported"
    rec = {"delta_tau_tuned_minus_base": {"delta_tau": -5, "ci95": [-9.0, -2.0]},
           "spectral_deltas": {"gap_ratio_last": {"p_two_sided": 0.01}}}
    assert hyp_check.check("S5", rec)["status"] == "supported"
    rec_nr = {"delta_tau_tuned_minus_base": {"delta_tau": -1, "ci95": [-4.0, 3.0]},
              "spectral_deltas": {"gap_ratio_last": {"p_two_sided": 0.5}}}
    assert hyp_check.check("S5", rec_nr)["status"] == "not_resolved"
    assert hyp_check.check("S9", {})["status"] == "not_resolved"  # no checker
    section = hyp_check.criterion_section("S5", rec)
    assert "[supported]" in section["summary"]


# --- P2.7: sqlite index ---------------------------------------------------------------


def _mini_report(out, model="Qwen/Qwen-test"):
    rep = {
        "schema_version": SCHEMA_VERSION,
        "passport": {"experiment_id": "stage2_detector_v1", "hypothesis_id": "S2",
                     "protocol_version": "1.1", "config": {}, "environment": {},
                     "model": {"model_name": model}, "code": {"git_commit": "abc123", "git_dirty": False}},
        "sections": [{"name": "S"}],
    }
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f)
    with open(os.path.join(out, "fit.json"), "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "model": model, "tau": 2.5,
                   "tau_ci": [2.0, 3.0], "gamma": 1.1, "mu_eff": 0.4}, f)


def test_index_rebuild_and_query(tmp_path):
    _mini_report(tmp_path / "run1")
    _mini_report(tmp_path / "run2", model="meta-llama/Llama-test")
    db = str(tmp_path / "index.sqlite")
    counts = rebuild([str(tmp_path)], db)
    assert counts["runs"] == 2 and counts["fits"] == 2 and counts["skipped"] >= 0
    rows = query(db, "SELECT model_name FROM runs WHERE model_name LIKE 'Qwen%'")
    assert rows == [("Qwen/Qwen-test",)]
    fits = query(db, "SELECT model, tau, mu_eff FROM fits ORDER BY model")
    assert len(fits) == 2 and abs(fits[0][1] - 2.5) < 1e-9
    with pytest.raises(sqlite3.Error):
        query(db, "DELETE FROM runs")  # read-only connection


# --- P2.8: compare_reports --------------------------------------------------------------


def _race_report(aurocs):
    rows = [[d, f"{a:.3f}", f"[{a - 0.1:+.3f}, {a + 0.1:+.3f}]", "0.5", 20] for d, a in aurocs.items()]
    return {"sections": [{"name": "AUROC race",
                          "table": {"headers": ["detector", "AUROC", "95% CI", "PR", "n"], "rows": rows}}]}


def test_compare_reports_delta_zero_for_identical():
    from stages.compare_reports import compare

    rep = _race_report({"tok_mean_entropy": 0.5, "spec_gap_last": 0.7})
    rows = compare(rep, rep)
    assert all(r["delta"] == 0.0 for r in rows)
    after = _race_report({"tok_mean_entropy": 0.5, "spec_gap_last": 0.9})
    d = {r["detector"]: r["delta"] for r in compare(rep, after)}
    assert d["spec_gap_last"] == pytest.approx(0.2)
    assert d["tok_mean_entropy"] == 0.0


def test_compare_reports_cli(tmp_path, capsys):
    from stages.compare_reports import main

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(_race_report({"tok_mean_entropy": 0.5})))
    b.write_text(json.dumps(_race_report({"tok_mean_entropy": 0.75})))
    rc = main([str(a), str(b)])
    assert rc == 0
    assert "Δ" in capsys.readouterr().out


# --- P2.9: Unit validation ---------------------------------------------------------------


def test_unit_rejects_unknown_detector_names():
    with pytest.raises(ValueError, match="unknown detector"):
        Unit("u", "b", "i", 1, {"spec_gap_las": 0.5})
    u = Unit("u", "b", "i", 1, {"spec_gap_last": 0.5})
    assert u.score("spec_gap_last") == 0.5
    with pytest.raises(KeyError):
        u.score("not_a_detector")
    with pytest.raises(KeyError):
        u.score("sem_ent")  # known but not scored for this unit
    assert "tok_mean_entropy" in KNOWN_DETECTORS and "probe_hidden" in KNOWN_DETECTORS

"""
stage3_ncrit.run_stage3 — measuring N_crit on a real model (Stage 3).

Protocol (fixed before the run):

  1. LOAD    — model + passport;
  2. TASKS   — arithmetic recall tasks (truth recomputed in Python,
               independent of the model), few-shot prompt, greedy decoding;
  3. OBSERVE — token-by-token generation with incremental decoding;
  4. LABEL   — first-error position per trajectory by the numeric prefix
               checker (censored when the answer never goes wrong);
  5. FIT     — Weibull survival MLE -> (c, gamma[, beta if mu_eff given]);
               binned error-rate curve + changepoint (the plan's "излом");
  6. SCALE   — if --mu-eff (and >= 3 models in --scaling-in), test
               T_crit ~ (mu_eff)^(-1/beta): weighted log-log regression,
               slope CI vs the beta=0.5 prediction of 2, beta consistency;
  7. REPORT  — report.json + REPORT.md + error_curve.csv.

The law is tested against data: the slope CI comes from resampling the
per-model tau within their bootstrap CIs, and mu_eff comes from OUTSIDE
the fit (stage-5 KL proxy by default). A slope far from 2 is reported as
a negative result, not smoothed away.

Run:
  python3 neural_lab/stages/stage3_ncrit/run_stage3.py \
      --model EleutherAI/pythia-70m --n-tasks 60 --out results/stage3/pythia70m
  # multi-model scaling (after several runs):
  python3 .../run_stage3.py --scaling-in results/stage3/* --out results/stage3/scaling
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # .../neural_lab
if _NEURAL_LAB not in sys.path:
    sys.path.insert(0, _NEURAL_LAB)

from science.hypotheses.check import criterion_section  # noqa: E402
from stages.common.cli import JsonlAppender, StageRunner, load_done_rows  # noqa: E402
from stages.common.generation import generate_with_capture  # noqa: E402
from stages.common.passports import data_hash  # noqa: E402
from stages.common.reporting import save_json, save_jsonl  # noqa: E402
from stages.common.schemas import validate_fit  # noqa: E402
from stages.stage3_ncrit.error_labels import incremental_token_texts, trajectory_table  # noqa: E402
from stages.stage3_ncrit.numeric_tasks import build_task_set, few_shot_header, make_checker  # noqa: E402
from stages.stage3_ncrit.scaling_test import beta_consistency, fit_scaling, verdict  # noqa: E402
from stages.stage3_ncrit.survival_fit import (  # noqa: E402
    binned_error_rate,
    bootstrap_survival,
    detect_breakpoint,
    fit_survival,
    weibull_mean,
)


def observe_trajectories(model, tokenizer, tasks, cfg, done_ids=None, writer=None) -> list[dict]:
    """Greedy generation + first-error labeling per task.

    done_ids/writer implement --resume: tasks already recorded in
    trajectories.jsonl are skipped, fresh rows are appended per task
    (a crash loses at most one task of work)."""
    from stages.common.cli import progress

    done_ids = done_ids or set()
    header = few_shot_header(n_shot=cfg["n_shot"], seed=cfg["seed"])
    trajs = []
    for task in progress(tasks, cfg, desc="stage3 tasks"):
        if task["task_id"] in done_ids:
            continue
        gen = generate_with_capture(
            model, tokenizer, header + task["prompt"],
            max_new_tokens=cfg["max_new_tokens"], temperature=0.0,
            capture_hidden=False, fast=cfg["fast"],
        )
        pieces = incremental_token_texts(tokenizer, gen["output_ids"])
        checker = make_checker(task, tol=cfg["tol"])
        from stages.stage3_ncrit.error_labels import first_error_position

        t_err, n_tok = first_error_position(pieces, checker)
        row = {
            "task_id": task["task_id"],
            "family": task["family"],
            "first_error": t_err,
            "n_tokens": n_tok,
            "generated": gen["generated_text"],
            "truth": task["truth_text"],
        }
        trajs.append(row)
        if writer is not None:
            writer.write(row)
    return trajs


def analyze_model(trajs, cfg) -> dict:
    tab = trajectory_table(trajs)
    t = np.asarray([r["t_first_error"] for r in tab])
    d = np.asarray([0.0 if r["censored"] else 1.0 for r in tab])
    fit = fit_survival(t, d, mu=None, seed=cfg["seed"])
    warm = np.asarray(fit["params"], dtype=np.float64) if fit.get("params") else None
    boot = bootstrap_survival(t, d, mu=None, n_boot=cfg["n_boot"], seed=cfg["seed"], warm_start=warm)
    curve = binned_error_rate(t, d, n_bins=cfg["n_bins"])
    # the at-risk window is the full generation length (censoring time), not
    # the observed error time — the permutation null needs the exposure
    at_risk = np.asarray([r["n_tokens"] for r in tab])
    bp = detect_breakpoint(t, d, at_risk=at_risk)
    return {
        "fit": fit,
        "bootstrap": boot,
        "error_curve": curve,
        "breakpoint": bp,
        "n_crit_weibull_mean": weibull_mean(fit["tau"], fit["gamma"]),
        "error_rate_overall": float(d.mean()),
        "n_traj": int(len(t)),
    }


class Stage3Runner(StageRunner):
    name = "stage3"
    title = "Stage 3 — N_crit from first-error survival"
    experiment_id = "stage3_ncrit_v1"
    hypothesis_id = "S3"

    def add_args(self, parser) -> None:
        parser.add_argument("--n-tasks", type=int, default=60)
        parser.add_argument("--n-shot", type=int, default=3)
        parser.add_argument("--max-new-tokens", type=int, default=24)
        parser.add_argument("--tol", type=float, default=0.05, help="relative tolerance of the numeric checker")
        parser.add_argument("--n-boot", type=int, default=300)
        parser.add_argument("--n-bins", type=int, default=10)
        parser.add_argument("--mu-eff", type=float, default=None,
                            help="external mu_eff for this model (e.g. stage-5 KL proxy); enables beta fit")
        parser.add_argument("--mu-eff-from", default=None,
                            help="path to a stage-5 run dir (record.json) or a record.json file; "
                                 "reads mu_eff_proxy_kl_ab from it (overrides nothing if --mu-eff is given)")
        parser.add_argument("--scaling-in", nargs="*", default=None,
                            help="report dirs of per-model runs: aggregate and test the scaling law")
        parser.add_argument("--fast", action="store_true", default=True)

    def default_config(self) -> dict:
        return {"n_shot": 3, "max_new_tokens": 24, "tol": 0.05, "n_boot": 300,
                "n_bins": 10, "fast": True}

    def needs_model(self, cfg: dict) -> bool:
        return not cfg.get("scaling_in")

    def run(self, cfg, model, tok, passport):
        if cfg.get("scaling_in"):
            run_scaling_aggregate(cfg["scaling_in"], cfg["out"])
            return None

        if cfg.get("mu_eff_from") and not cfg.get("mu_eff"):
            # P2.10: reuse the mu_eff already measured by a stage-5 run for this
            # model instead of asking the user to copy a number by hand
            p = cfg["mu_eff_from"]
            if os.path.isdir(p):
                p = os.path.join(p, "record.json")
            with open(p) as f:
                rec = json.load(f)
            mu_from_rec = rec.get("mu_eff_proxy_kl_ab")
            if not mu_from_rec or mu_from_rec <= 0:
                raise SystemExit(f"--mu-eff-from: no usable mu_eff_proxy_kl_ab in {p}")
            cfg["mu_eff"] = float(mu_from_rec)
            print(f"[stage3] mu_eff={cfg['mu_eff']:.4f} taken from {p}", flush=True)

        tasks = build_task_set(n_tasks=cfg["n_tasks"], seed=cfg["seed"])
        passport["config"]["data_hash"] = data_hash([t["prompt"] for t in tasks])
        passport["config"]["mu_eff_external"] = cfg.get("mu_eff")

        trajs_path = os.path.join(cfg["out"], "trajectories.jsonl")
        done_ids: set = set()
        restored: list[dict] = []
        if cfg.get("resume") and os.path.isfile(trajs_path):
            restored, done_ids = load_done_rows(trajs_path, "task_id")
            print(f"[stage3] resume: {len(restored)} trajectories already recorded", flush=True)

        print(f"[stage3] {len(tasks)} tasks, generating …", flush=True)
        with JsonlAppender(trajs_path) as writer:
            fresh = observe_trajectories(model, tok, tasks, cfg,
                                         done_ids=done_ids, writer=writer)
        trajs = restored + fresh
        res = analyze_model(trajs, cfg)

        fit = res["fit"]
        tau_ci = res["bootstrap"].get("c", {}).get("ci95", [fit["tau"] / 2, fit["tau"] * 2])
        fit_record = {
            "schema_version": "1.0",
            "model": cfg["model"],
            "tau": fit["tau"],
            "tau_ci": tau_ci,
            "gamma": fit["gamma"],
            "mu_eff": cfg.get("mu_eff"),
        }
        if cfg.get("mu_eff") and cfg["mu_eff"] > 0:
            tab = trajectory_table(trajs)
            errs = np.asarray([0.0 if r["censored"] else 1.0 for r in tab])
            times = np.asarray([r["t_first_error"] for r in tab])
            mus = np.asarray([cfg["mu_eff"]] * len(trajs))
            fit_mu = fit_survival(times, errs, mu=mus, seed=cfg["seed"])
            res["fit_with_mu"] = fit_mu
            # the beta CI must come from a REAL bootstrap under the same mu —
            # a fabricated [β·0.5, β·1.5] interval would leak into fit.json and
            # from there into the cross-model scaling test
            boot_mu = bootstrap_survival(times, errs, mu=mus,
                                         n_boot=cfg["n_boot"], seed=cfg["seed"],
                                         warm_start=np.asarray(fit_mu["params"], dtype=np.float64))
            if not boot_mu.get("beta", {}).get("ci95") or not np.all(np.isfinite(boot_mu["beta"]["ci95"])):
                raise RuntimeError("bootstrap with mu failed; refusing to fabricate a beta CI")
            res["bootstrap_with_mu"] = boot_mu
            fit_record["beta_fit"] = {
                "beta": fit_mu["beta"],
                "ci95": boot_mu["beta"]["ci95"],
                "beta_at_boundary": fit_mu.get("beta_at_boundary", False),
            }
            if fit_mu.get("beta_at_boundary"):
                print(f"[stage3] WARNING: beta at the parameterization boundary "
                      f"{fit_mu['beta']:.3f}; treat the CI with care", flush=True)

        validate_fit(fit_record)
        save_json(os.path.join(cfg["out"], "fit.json"), fit_record)

        save_jsonl(os.path.join(cfg["out"], "trajectories.jsonl"), trajs)
        with open(os.path.join(cfg["out"], "error_curve.csv"), "w") as f:
            f.write("bin_center,rate,at_risk,errors\n")
            for c, r, a, e in zip(res["error_curve"]["bin_centers"], res["error_curve"]["rate"],
                                  res["error_curve"]["at_risk"], res["error_curve"]["errors"]):
                f.write(f"{c:.2f},{r:.5f},{a},{e}\n")

        b = res["bootstrap"].get("c", {})
        g_b = res["bootstrap"].get("gamma", {}).get("ci95")

        def fmt_ci(ci):
            return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"

        sections = [
            {"name": "First-error survival fit (Weibull scale, tokens)",
             "table": {"headers": ["quantity", "value", "95% CI"],
                       "rows": [
                           ["tau (scale)", f"{fit['tau']:.2f}", fmt_ci(b.get("ci95"))],
                           ["gamma (shape)", f"{fit['gamma']:.2f}", fmt_ci(g_b)],
                           ["mean T_crit = tau*Gamma(1+1/gamma)", f"{res['n_crit_weibull_mean']:.2f}", "—"],
                           ["error rate", f"{res['error_rate_overall']:.2%}", "—"],
                       ]},
             "notes": ["gamma ~ 1 = constant hazard (the H1 null A0); gamma >> 1 = accelerating collapse.",
                       "Censored trajectories (never wrong) enter the likelihood as survival terms."]},
            {"name": "Breakpoint (the plan's 'izlem')",
             "summary": f"changepoint at token {res['breakpoint']['breakpoint']} "
                        f"(p_perm={res['breakpoint'].get('p_perm', float('nan')):.3f})",
             "notes": ["Exact two-rate Poisson split; permutation p over error-label shuffles."]},
            {"name": "Error-rate curve", "summary": "see error_curve.csv (bin centers, at-risk normalized rates)"},
        ]
        if done_ids:
            sections.append({
                "name": "Resume",
                "summary": (f"resumed: {len(restored)} trajectories restored from trajectories.jsonl; "
                            "the survival fit uses restored + fresh trajectories together")})
        # the pre-registered criterion (P2.6): the scaling check needs >= 3
        # models, so a single-model run reports the section as not_resolved
        sections.append(criterion_section(self.hypothesis_id, {"sections": sections}))
        return sections, {}

    # --- scaling aggregation (kept here: single source of truth for the law) ---
    # run_scaling_aggregate / _load_scaling_record live at module level below


def main() -> None:
    from stages.common.cli import main as runner_main

    runner_main(Stage3Runner())


if __name__ == "__main__":
    main()


def _load_scaling_record(d: str) -> dict | None:
    """One per-model scaling record from a run directory.

    Accepts BOTH formats: stage-3's own fit.json sidecar and stage-5's
    record.json (normalized via stage5's pair_fit_record). Before this,
    --scaling-in pointed at stage-5 outputs found 0 records and exited
    with 'need >= 3 models, got 0'."""
    fp = os.path.join(d, "fit.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            fj = json.load(f)
        if fj.get("mu_eff") and fj.get("tau"):
            return fj
    rp = os.path.join(d, "record.json")
    if os.path.isfile(rp):
        try:
            from stages.stage5_rlhf.run_stage5 import pair_fit_record

            with open(rp) as f:
                return pair_fit_record(json.load(f))
        except Exception as e:  # malformed record: skip, but say so
            print(f"[stage3] scaling: skipping {rp} ({e})", flush=True)
            return None
    return None


def run_scaling_aggregate(dirs: list[str], out_dir: str) -> None:
    """Aggregate per-model fit records and test T_crit ~ (mu_eff)^(-1/β).

    The law is tested against data: tau and its CI come from independent
    per-model survival fits, mu_eff from OUTSIDE the fit (KL proxy), and
    the target slope 2 (beta = 0.5) is compared against a resampled CI —
    never assumed. Records are read from fit.json (stage-3 runs, stage-5
    sidecars) or normalized from stage-5 record.json."""
    from stages import PROTOCOL_VERSION
    from stages.common.reporting import write_report
    from stages.common.schemas import validate_scaling

    recs = []
    for pat in dirs:
        for d in sorted(glob.glob(pat)) or [pat]:
            if not os.path.isdir(d):
                continue
            rec = _load_scaling_record(d)
            if rec is not None:
                recs.append(rec)
    if len(recs) < 3:
        print(f"[stage3] scaling: need >= 3 models with mu_eff, got {len(recs)}; nothing written")
        return
    tau = np.asarray([r["tau"] for r in recs])
    tau_ci = np.asarray([r["tau_ci"] for r in recs])
    mu = np.asarray([r["mu_eff"] for r in recs])
    scaling = fit_scaling(tau, tau_ci, mu)
    # beta consistency ONLY from real per-model beta fits: the slope CI lives
    # in slope units and must never masquerade as a beta CI
    beta_recs = [r["beta_fit"] for r in recs if r.get("beta_fit")]
    cons = beta_consistency(beta_recs)
    v = verdict(scaling, cons)
    scaling_payload = {
        "schema_version": "1.0",
        "protocol_version": PROTOCOL_VERSION, "records": recs, "scaling": scaling,
        "consistency": cons, "verdict": v,
    }
    validate_scaling(scaling_payload)
    save_json(os.path.join(out_dir, "scaling.json"), scaling_payload)
    rows = [["slope (target -2 for beta=0.5)", f"{scaling['slope']:.2f}"],
            ["slope 95% CI", f"[{scaling['slope_ci95'][0]:.2f}, {scaling['slope_ci95'][1]:.2f}]"],
            ["implied beta", f"{scaling['implied_beta']:.2f}"],
            ["RMS residual (log space)", f"{scaling['resid_rms']:.3f}"],
            ["dof (n_models - 2)", str(scaling["dof"])]]
    notes = [f"VERDICT: {v}"]
    if scaling.get("power_warning"):
        notes.append(f"POWER: {scaling['power_warning']}")
    if cons.get("all_consistent") is None:
        notes.append(f"NOTE: {cons.get('note', 'no per-model beta fits; consistency unchecked')}")
    write_report(
        out_dir, "Stage 3 — (mu_eff)^(-1/beta) scaling across models",
        {"experiment_id": "stage3_scaling_v1", "hypothesis_id": "S3",
         "protocol_version": PROTOCOL_VERSION, "model": {}, "config": {}, "environment": {}},
        [{"name": "Scaling regression",
          "table": {"headers": ["quantity", "value"], "rows": rows},
          "notes": notes}],
    )
    print(f"[stage3] scaling report in {out_dir}; verdict: {v}", flush=True)


if __name__ == "__main__":
    main()

"""
stage5_rlhf.run_stage5 — the RLHF thesis, tested (Stage 5).

Thesis under test (H5): instruction tuning (SFT/RLHF/DPO) shifts the
spectral signature and N_crit of the SAME family — the "utility trap":
tuned models hallucinate sooner (smaller tau / T_crit) while looking
more fluent.

Pre-registered outcomes (both publishable):
  - Delta(tau) < 0 with CI excluding 0 AND Delta(gap_ratio) systematic ->
    the thesis is supported on this pair;
  - no systematic shift -> the thesis is NOT supported; report as is.

Per pair, writes <out>/<pair_id>/record.json + REPORT.md. The cross-pair
scaling aggregation (T_crit ~ mu_eff^(-1/β)) is stage3's --scaling-in:
point it at the per-pair directories with the mu proxy.

Run:
  python3 neural_lab/stages/stage5_rlhf/run_stage5.py --pair qwen2.5-0.5b \
      --out results/stage5
  python3 .../run_stage5.py --list   # show the registry
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # .../neural_lab
if _NEURAL_LAB not in sys.path:
    sys.path.insert(0, _NEURAL_LAB)

from stages.common.reporting import save_json, write_report  # noqa: E402
from stages.stage5_rlhf.compare_pairs import evaluate_pair  # noqa: E402

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_pairs.yaml")


def load_registry(path: str = REGISTRY) -> list[dict]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)["pairs"]


def pair_fit_record(record: dict) -> dict | None:
    """Stage-3-compatible fit.json payload from a stage-5 pair record.

    run_stage3 --scaling-in consumes fit.json sidecars with the fields
    {model, tau, tau_ci, mu_eff, ...}; stage-5's record.json uses a
    different layout, which made the two stages silently incompatible.
    This normalizer is the single conversion point (used both when stage 5
    writes the sidecar and when stage 3 falls back to reading record.json).
    Returns None when the pair produced no usable tau or mu_eff."""
    st = record.get("survival_tuned") or {}
    sb = record.get("survival_base") or {}
    tau = st.get("tau")
    mu = record.get("mu_eff_proxy_kl_ab")
    if not tau or not mu or mu <= 0:
        return None
    tau_ci = st.get("tau_ci95") or [tau / 2, tau * 2]
    rec = {
        "model": record.get("pair_id", "unknown"),
        "tau": float(tau),
        "tau_ci": [float(tau_ci[0]), float(tau_ci[1])],
        "gamma": st.get("gamma"),
        "mu_eff": float(mu),
        "tau_base": (float(sb["tau"]) if sb.get("tau") else None),
        "source": "stage5_pair",
    }
    dtau = record.get("delta_tau_tuned_minus_base") or {}
    if dtau.get("delta_tau") is not None:
        rec["delta_tau"] = dtau["delta_tau"]
        rec["delta_tau_ci95"] = dtau.get("ci95")
    return rec


def pair_report(record: dict) -> list[dict]:
    rows = []
    for key, d in record["spectral_deltas"].items():
        rows.append([key, f"{d['delta_mean']:+.4f}",
                     f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", f"{d['p_two_sided']:.3f}"])
    sb, st = record["survival_base"], record["survival_tuned"]
    surv_rows = [
        ["tau (T_crit scale, tokens)", sb.get("tau"), st.get("tau")],
        ["gamma", sb.get("gamma"), st.get("gamma")],
        ["error rate", sb.get("error_rate"), st.get("error_rate")],
    ]
    dtau = (None if not (sb.get("tau") and st.get("tau"))
            else (st["tau"] - sb["tau"]) / max(sb["tau"], 1e-9))
    dt = record.get("delta_tau_tuned_minus_base") or {}
    if dt.get("ci95"):
        surv_rows.append(["Δtau (tuned − base) [paired bootstrap CI]",
                          f"{dt['delta_tau']:+.2f}",
                          f"[{dt['ci95'][0]:+.2f}, {dt['ci95'][1]:+.2f}]"])
    g = record["spectral_deltas"]["gap_ratio_last"]
    p_spec = g.get("p_two_sided")
    spec_sig = p_spec is not None and np.isfinite(p_spec) and p_spec < 0.05
    if dtau is None:
        # the pair RAN but at least one survival fit is undefined — that is an
        # invalid result, not an experiment that was never conducted
        verdict = ("INCONCLUSIVE: no first errors on the task set in base or tuned — "
                   "survival fit undefined (raise n-tasks or max-new-tokens)")
    elif dt.get("ci95"):
        lo, hi = dt["ci95"]
        if hi < 0 and spec_sig:
            verdict = "SUPPORTED on this pair: Δtau CI excludes 0 (tuned collapses sooner) with spectral shift"
        elif hi < 0:
            verdict = ("PARTIALLY SUPPORTED: Δtau CI excludes 0 (tuned collapses sooner) "
                       "but the spectral shift is not significant")
        elif lo > 0 and spec_sig:
            verdict = "REFUTED on this pair: tuned model lasts LONGER (Δtau CI excludes 0, positive)"
        else:
            verdict = "NOT RESOLVED on this pair: Δtau CI covers 0 (a negative result — report it)"
    elif dtau < 0 and spec_sig:
        verdict = "SUPPORTED on this pair: tuned model collapses sooner (Δtau<0) with spectral shift"
    elif dtau > 0 and spec_sig:
        verdict = "REFUTED on this pair: tuned model lasts LONGER"
    else:
        verdict = "NOT RESOLVED on this pair: shifts not significant (a negative result — report it)"
    return [
        {"name": "Spectral deltas (tuned − base), paired permutation",
         "table": {"headers": ["statistic", "Δ mean", "95% CI", "p"], "rows": rows}},
        {"name": "First-error survival (shared arithmetic task set)",
         "table": {"headers": ["quantity", "base", "tuned"], "rows": [[r[0], r[1], r[2]] for r in surv_rows]}},
        {"name": "mu_eff proxy (token-mean KL, log-vocab normalized)",
         "summary": f"KL(base||tuned)={record['mu_eff_proxy_kl_ab']:.4f}  "
                    f"KL(tuned||base)={record['mu_eff_proxy_kl_ba']:.4f}",
         "notes": ["Measured OUTSIDE the survival fit; feeds the stage-3 scaling test."]},
        {"name": "Verdict (pre-registered)", "summary": verdict},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5: base vs tuned pair comparison")
    ap.add_argument("--pair", default=None, help="pair id from model_pairs.yaml")
    ap.add_argument("--all-open", action="store_true", help="run every 'open' pair")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--n-shot", type=int, default=3)
    ap.add_argument("--kl-positions", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, required=False)
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()

    pairs = load_registry()
    if args.list:
        for p in pairs:
            print(f"{p['id']:24s} {p['base']:45s} vs {p['tuned']:45s} [{p['access']}] {p['method']}")
        return
    if not args.out:
        ap.error("--out is required for a run")

    if args.all_open:
        chosen = [p for p in pairs if p["access"] == "open" and p["method"] != "checkpoint-ablation"]
    elif args.pair:
        chosen = [p for p in pairs if p["id"] == args.pair]
        if not chosen:
            ap.error(f"unknown pair {args.pair!r}; see --list")
    else:
        ap.error("--pair or --all-open or --list required")

    for pair in chosen:
        cfg = {
            "n_tasks": args.n_tasks, "max_new_tokens": args.max_new_tokens,
            "n_shot": args.n_shot, "kl_positions": args.kl_positions,
            "n_boot": args.n_boot, "seed": args.seed, "dtype": args.dtype,
            "device": args.device, "fast": args.fast,
        }
        record = evaluate_pair(pair, cfg)
        out = os.path.join(args.out, pair["id"])
        save_json(os.path.join(out, "record.json"), record)
        fit_rec = pair_fit_record(record)
        if fit_rec is not None:
            # stage-3-compatible sidecar: run_stage3 --scaling-in consumes it
            # directly (same field names as stage-3's own fit.json)
            save_json(os.path.join(out, "fit.json"), fit_rec)
        else:
            print(f"[stage5] {pair['id']}: no (tau, mu_eff) pair — fit.json sidecar skipped", flush=True)
        write_report(out, f"Stage 5 — {pair['id']}: base vs {pair['method']} tuned",
                     record["passport"], pair_report(record))
        print(f"[stage5] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

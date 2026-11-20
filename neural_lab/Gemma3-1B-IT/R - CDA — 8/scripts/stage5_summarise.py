#!/usr/bin/env python3
"""stage5_summarise.py — the cross-arm summary of the RCDA-9 formula round.

Reads part9_eval_{base,formulas,pen}.json + the pen training log and writes
results/summary_round9.json: the one table the README and the report read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda9_lab as L                                          # noqa: E402


def main() -> None:
    res = L.RESULTS_DIR
    tags = ("base", "formulas", "pen")
    ev = {}
    for t in tags:
        p = res / f"part9_eval_{t}.json"
        if p.exists():
            ev[t] = json.loads(p.read_text(encoding="utf-8"))
    if len(ev) < 2:
        print("not enough arm evals yet — nothing to summarise", flush=True)
        return

    tl_path = res / "pen_arm" / "training_log.json"
    tl = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.exists() else None
    pen_descent = None
    if tl:
        bad = [s for s in tl["steps"] if s.get("kind") == "bad" and "pen_norm" in s]
        if len(bad) >= 20:
            first10 = float(np.mean([s["pen_norm"] for s in bad[:10]]))
            last10 = float(np.mean([s["pen_norm"] for s in bad[-10:]]))
            pen_descent = round(100 * (last10 / first10 - 1), 1)

    summary = {
        "round": "RCDA-9 formula round",
        "model": L.MODEL_SNAPSHOT,
        "arms": {},
        "pen_descent_pct": pen_descent,
    }
    for t, e in ev.items():
        fk = e.get("formula_knowledge", {})
        summary["arms"][t] = {
            "rho_bad": e["rho_bad"],
            "rho_good": e["rho_good"],
            "loop_share_mean_bad": e["loop_share_mean_bad"],
            "loop_waste_mean_bad": e["loop_waste_mean_bad"],
            "hard12_loop_share": (round(float(np.mean(e["hard12_loop_share"])), 3)
                                  if e.get("hard12_loop_share") else None),
            "mild8_loop_share": (round(float(np.mean(e["mild8_loop_share"])), 3)
                                 if e.get("mild8_loop_share") else None),
            "prompts_fully_repaired": e.get("prompts_fully_repaired"),
            "kl_ctrl_mean": e["kl_ctrl_mean"],
            "ppl_control_mean": e["ppl_control_mean"],
            "corpus_val_ppl": fk.get("corpus_val_ppl"),
            "qa_exact": fk.get("qa_exact"),
            "qa_pairs": fk.get("qa_pairs"),
            "qa_exact_rate": fk.get("qa_exact_rate"),
            "qa_partial_rate": fk.get("qa_partial_rate"),
            "qa_mean_rel_err": fk.get("qa_mean_rel_err"),
            "forensics_tensors": len(e.get("lora_forensics", {})),
        }

    # the honest cross-reads the report needs
    notes = []
    if "formulas" in ev and "base" in ev:
        d = ev["formulas"]["ppl_control_mean"] - ev["base"]["ppl_control_mean"]
        notes.append(f"formulas arm control-PPL drift: {d:+.1f}")
        if ev["formulas"].get("formula_knowledge", {}).get("qa_exact") is not None:
            notes.append(f"formulas arm QA exact: "
                         f"{ev['formulas']['formula_knowledge']['qa_exact']}/"
                         f"{ev['formulas']['formula_knowledge']['qa_pairs']} vs base "
                         f"{ev['base']['formula_knowledge']['qa_exact']}/"
                         f"{ev['base']['formula_knowledge']['qa_pairs']}")
    if "pen" in ev and "base" in ev:
        m = ev["pen"].get("mild8_loop_share")
        if m:
            notes.append(f"pen arm mild-8 migration check: base "
                         f"{round(float(np.mean(ev['base']['per_traj_loop_share_bad'][12:])), 3)} "
                         f"-> pen {round(float(np.mean(m)), 3)}")
    summary["notes"] = notes

    L.dump_json(summary, res / "summary_round9.json")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

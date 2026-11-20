#!/usr/bin/env python3
"""pipeline.py — the RCDA-9 formula round, one chunk at a time.

The validation box kills any process between tool calls, so every stage is
resumable and this orchestrator simply picks the next unfinished unit of
work and runs it:

  1. probe          stage1_probe.py        (frozen generations, directions)
  2. base eval      stage4_eval.py base    (the baseline arm)
  3. formulas arm   llamafactory-cli train (formula corpus SFT, resume)
  4. pen arm        stage3_pen_train.py    (the formula-as-loss arm, resume)
  5. eval arms      stage4_eval.py formulas / pen
  6. summary        stage5_summarise.py

Usage: python3 scripts/pipeline.py [--stage NAME]
Prints PIPELINE_DONE when everything is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
RES = WORK / "results"
PY = sys.executable


def sh(cmd, timeout=None):
    print(f"\n=== RUN: {' '.join(map(str, cmd))}", flush=True)
    sys.stdout.flush()
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/home/z/my-project/.cache/hf")
    env.setdefault("HF_HUB_OFFLINE", "1")
    r = subprocess.run(cmd, cwd=WORK, env=env)
    return r.returncode


def latest_ckpt(out_dir: Path) -> str | None:
    import glob
    cks = sorted(glob.glob(str(out_dir / "checkpoint-*")),
                 key=lambda p: int(p.rsplit("-", 1)[-1]))
    return cks[-1] if cks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    args = ap.parse_args()

    def stage(name):
        return args.stage in (None, name)

    # 1. probe --------------------------------------------------------------
    if stage("probe") and not (RES / "probe" / "probe_phase_report.json").exists():
        sh([PY, "scripts/stage1_probe.py"])

    # 2. base arm eval ------------------------------------------------------
    if stage("base_eval") and not (RES / "part9_eval_base.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "base"])

    # 3. formulas arm (llamafactory-cli, auto-resume) ------------------------
    fml_out = RES / "formulas_arm" / "gemma3_formula_lora"
    fml_done = (fml_out / "adapter_model.safetensors").exists()
    if stage("formulas_train") and not fml_done:
        cmd = ["llamafactory-cli", "train", "configs/sft_formula_cpu.yaml"]
        ck = latest_ckpt(fml_out)
        if ck:
            cmd += [f"resume_from_checkpoint={ck}"]
        sh(cmd)
        if (fml_out / "adapter_model.safetensors").exists():
            print("formulas arm complete", flush=True)

    # 4. pen arm (the formula-as-loss arm) -----------------------------------
    pen_dir = RES / "pen_arm"
    pen_done = (pen_dir / "adapter" / "adapter_model.safetensors").exists()
    if stage("pen_train") and not pen_done:
        cmd = [PY, "scripts/stage3_pen_train.py", "--resume"]
        sh(cmd)

    # 5. arm evals ----------------------------------------------------------
    if stage("eval_formulas") and fml_done and \
            not (RES / "part9_eval_formulas.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "formulas",
            "--lora", str(fml_out)])
    if stage("eval_pen") and pen_done and \
            not (RES / "part9_eval_pen.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "pen",
            "--lora", str(pen_dir / "adapter")])

    # 6. summary ------------------------------------------------------------
    if stage("summary"):
        ok = all((RES / f"part9_eval_{t}.json").exists()
                 for t in ("base", "formulas", "pen"))
        if ok and not (RES / "summary_round9.json").exists():
            sh([PY, "scripts/stage5_summarise.py"])

    # final report ----------------------------------------------------------
    state = {
        "probe": (RES / "probe" / "probe_phase_report.json").exists(),
        "eval_base": (RES / "part9_eval_base.json").exists(),
        "formulas_adapter": (fml_out / "adapter_model.safetensors").exists()
        if fml_out.exists() else False,
        "pen_adapter": pen_done,
        "eval_formulas": (RES / "part9_eval_formulas.json").exists(),
        "eval_pen": (RES / "part9_eval_pen.json").exists(),
        "summary": (RES / "summary_round9.json").exists(),
    }
    print("\nSTATE: " + json.dumps(state), flush=True)
    if all(state.values()):
        print("PIPELINE_DONE", flush=True)
    else:
        print("PIPELINE_INCOMPLETE — re-run for the next chunk", flush=True)


if __name__ == "__main__":
    main()

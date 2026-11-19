#!/usr/bin/env python3
"""pipeline.py — the RCDA-8 LlamaFactory round, one chunk at a time.

The validation box kills any process between tool calls, so every stage is
resumable and this orchestrator simply picks the next unfinished unit of
work and runs it (until it finishes or the chunk budget expires):

  1. probe        stage1_probe.py       (frozen generations, directions)
  2. base eval    stage4_eval.py base   (the baseline arm)
  3. sft data     stage2_build_sft_data.py
  4. sft train    llamafactory-cli train (anti-loop SFT arm, resume)
  5. rcda8 train  stage3_rcda8_train.py (the formula arm, resume)
  6. eval arms    stage4_eval.py rcda8 / sft
  7. summary      stage5_summarise.py

Usage: python3 scripts/pipeline.py [--next] [--stage NAME]
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
    if stage("base_eval") and not (RES / "part8_eval_base.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "base"])

    # 3. sft data -----------------------------------------------------------
    if stage("sft_data") and not (WORK / "data" / "sft_antiloop.jsonl").exists():
        sh([PY, "scripts/stage2_build_sft_data.py"])

    # 4. sft train (llamafactory-cli, auto-resume) --------------------------
    sft_out = RES / "sft_arm" / "gemma3_sft_lora"
    sft_done = (sft_out / "adapter_model.safetensors").exists()
    if stage("sft_train") and not sft_done:
        cmd = ["llamafactory-cli", "train", "configs/sft_antiloop_cpu.yaml"]
        ck = latest_ckpt(sft_out)
        if ck:
            cmd += [f"resume_from_checkpoint={ck}"]
        sh(cmd)
        # final export of the adapter if training finished
        if (sft_out / "adapter_model.safetensors").exists():
            print("SFT arm complete", flush=True)

    # 5. rcda8 train (the formula arm) --------------------------------------
    rcda_dir = RES / "rcda8_arm"
    rcda_done = (rcda_dir / "adapter" / "adapter_model.safetensors").exists()
    if stage("rcda8_train") and not rcda_done:
        cmd = [PY, "scripts/stage3_rcda8_train.py", "--resume"]
        sh(cmd)

    # 6. arm evals ----------------------------------------------------------
    if stage("eval_rcda8") and rcda_done and \
            not (RES / "part8_eval_rcda8.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "rcda8",
            "--lora", str(rcda_dir / "adapter")])
    if stage("eval_sft") and (RES / "sft_arm" / "gemma3_sft_lora" /
                              "adapter_model.safetensors").exists() and \
            not (RES / "part8_eval_sft.json").exists():
        sh([PY, "scripts/stage4_eval.py", "--tag", "sft",
            "--lora", str(RES / "sft_arm" / "gemma3_sft_lora")])

    # 7. summary ------------------------------------------------------------
    if stage("summary"):
        ok = all((RES / f"part8_eval_{t}.json").exists()
                 for t in ("base", "rcda8", "sft"))
        if ok and not (RES / "summary_round8.json").exists():
            sh([PY, "scripts/stage5_summarise.py"])

    # final report ----------------------------------------------------------
    state = {
        "probe": (RES / "probe" / "probe_phase_report.json").exists(),
        "eval_base": (RES / "part8_eval_base.json").exists(),
        "sft_data": (WORK / "data" / "sft_antiloop.jsonl").exists(),
        "sft_adapter": (sft_out / "adapter_model.safetensors").exists()
        if sft_out.exists() else False,
        "rcda8_adapter": rcda_done,
        "eval_rcda8": (RES / "part8_eval_rcda8.json").exists(),
        "eval_sft": (RES / "part8_eval_sft.json").exists(),
        "summary": (RES / "summary_round8.json").exists(),
    }
    print("\nSTATE: " + json.dumps(state), flush=True)
    if all(state.values()):
        print("PIPELINE_DONE", flush=True)
    else:
        print("PIPELINE_INCOMPLETE — re-run for the next chunk", flush=True)


if __name__ == "__main__":
    main()

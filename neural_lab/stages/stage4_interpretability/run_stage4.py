"""
stage4_interpretability.run_stage4 — the interpretability bridge (Stage 4).

Per model and prompt set:
  A. logit lens trajectory of the target answer token vs layer;
  B. BBP spike layer (gap_ratio argmax over layers) + alignment statistic
     (lens commitment ~ spectral gap; restoration ~ gap);
  C. activation patching on minimal pairs (clean vs corrupted prompt);
  D. heavy-tail alpha of every weight matrix (MM bands per layer) and its
     correlation with the activation gap profile;
  E. (optional, --sae) SAE audit of the spike window (Gemma Scope).

The mechanism claim is accepted only if the alignments hold with their
permutation p-values on the REAL model; a null alignment is reported as
such. Run:

  python3 neural_lab/stages/stage4_interpretability/run_stage4.py \
      --model EleutherAI/pythia-70m --prompts-file prompts.txt --out results/stage4/pythia70m
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # .../neural_lab
if _NEURAL_LAB not in sys.path:
    sys.path.insert(0, _NEURAL_LAB)

from science.hypotheses.check import criterion_section  # noqa: E402
from stages.common.cli import StageRunner  # noqa: E402
from stages.common.generation import generate_with_capture  # noqa: E402
from stages.stage2_detector.spectral_features import covariance_spectral_features  # noqa: E402
from stages.stage4_interpretability.activation_patching import (  # noqa: E402
    align_patching_with_spectrum,
    patching_profile,
)
from stages.stage4_interpretability.heavy_tail import correlate_alpha_with_gap, mm_table  # noqa: E402
from stages.stage4_interpretability.logit_lens import (  # noqa: E402
    find_bbp_spike,
    lens_spike_alignment,
    lens_trajectory,
)


def default_prompts() -> list[dict]:
    """Minimal pairs bundled with the repo (small factual set; extend via
    --prompts-file). Each pair: clean/corrupted prompt + expected answer."""
    return [
        {"clean": "The capital of France is", "corrupted": "The capital of Germany is", "answer": " Paris"},
        {"clean": "The capital of Japan is", "corrupted": "The capital of Brazil is", "answer": " Tokyo"},
        {"clean": "Two plus two equals", "corrupted": "Two plus five equals", "answer": " four"},
        {"clean": "The author of Romeo and Juliet is", "corrupted": "The author of War and Peace is", "answer": " Shakespeare"},
        {"clean": "Water boils at a temperature of 100 degrees", "corrupted": "Gold melts at a temperature of 100 degrees", "answer": " Celsius"},
        {"clean": "The largest planet in the solar system is", "corrupted": "The smallest planet in the solar system is", "answer": " Jupiter"},
    ]


def gap_profile_of(model, tokenizer, text: str) -> list[float]:
    gen = generate_with_capture(model, tokenizer, text, max_new_tokens=1,
                                temperature=0.0, capture_hidden=True, fast=True)
    feats = [covariance_spectral_features(np.asarray(h, dtype=np.float64), li)
             for li, h in enumerate(gen["final_hidden"])]
    return [f["gap_ratio"] for f in feats if f is not None]


def mean_gap_profile(gap_profiles: list[list[float]]) -> list[float]:
    """Element-wise mean over prompts, truncated to the SHORTEST profile.

    Padding by repeating the last value would fabricate constant tails and
    distort the correlation with alpha; different lengths (e.g. from
    truncated prompts) are truncated instead."""
    if not gap_profiles:
        return []
    min_len = min(map(len, gap_profiles))
    return np.mean([g[:min_len] for g in gap_profiles], axis=0).tolist()


class Stage4Runner(StageRunner):
    name = "stage4"
    title = "Stage 4 — Interpretability bridge"
    experiment_id = "stage4_interpretability_v1"
    hypothesis_id = "S4"
    steps = ("lens", "patching", "heavy-tail", "sae")

    def add_args(self, parser) -> None:
        parser.add_argument("--prompts-file", default=None,
                            help='JSONL: {"clean":..., "corrupted":..., "answer":...}')
        parser.add_argument("--sae", action="store_true", help="run the SAE audit (needs sae_lens + Gemma)")
        parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
        parser.add_argument("--sae-id", default="layer_12/width_16k/average_l0_71")
        parser.add_argument("--weights-filter", default="mlp|attn|c_attn|c_proj|c_fc",
                            help="regex over parameter names for the heavy-tail step (P2.11)")

    def default_config(self) -> dict:
        return {}

    def run(self, cfg, model, tok, passport) -> tuple[list[dict], dict]:
        import time as _time

        from stages.common.cli import step_requested

        if cfg.get("prompts_file"):
            with open(cfg["prompts_file"]) as f:
                pairs = [json.loads(l) for l in f if l.strip()]
        else:
            pairs = default_prompts()

        # the gap profiles feed every mechanism step; compute them when any
        # consumer is requested (cheapest shared observation)
        need_gap = any(step_requested(cfg, s) for s in ("lens", "patching", "heavy-tail", "sae"))
        lens_rows, patch_rows, gap_profiles = [], [], []
        step_notes: list[list] = []
        if need_gap:
            t0 = _time.time()
            for pr in pairs:
                ids = tok.encode(pr["clean"])
                ans_ids = tok.encode(pr["answer"], add_special_tokens=False)
                if not ans_ids:
                    continue
                want_lens = step_requested(cfg, "lens")
                want_patch = step_requested(cfg, "patching")
                if not (want_lens or want_patch):
                    break
                gp = gap_profile_of(model, tok, pr["clean"])
                if want_patch:
                    gap_profiles.append(gp)
                if want_lens:
                    lens = lens_trajectory(model, tok, pr["clean"], pr["answer"])
                    align = lens_spike_alignment(lens["per_layer_prob"], gp)
                    lens_rows.append({"prompt": pr["clean"], "final_prob": lens["final_prob"], "alignment": align})
                if want_patch:
                    try:
                        patch = patching_profile(model, ids, tok.encode(pr["corrupted"]), int(ans_ids[0]))
                        pa = align_patching_with_spectrum(patch["normalized"], gp)
                        patch_rows.append({"prompt": pr["clean"], "p_clean": patch["p_clean"],
                                           "p_corrupt": patch["p_corrupt"], "alignment": pa})
                    except Exception as e:  # architecture-specific hook failures are recorded, not fatal
                        patch_rows.append({"prompt": pr["clean"], "error": str(e)})
            step_notes.append(["observation (lens/patching)", "done", f"{_time.time() - t0:.1f}s"])

        sections: list[dict] = []
        if step_requested(cfg, "lens"):
            sections.append(
                {"name": "Logit lens × BBP spike alignment",
                 "table": {"headers": ["prompt", "final P(answer)", "r(commit, gap)", "p", "spike L", "commit L"],
                           "rows": [[r["prompt"][:38], f"{r['final_prob']:.3f}",
                                     f"{r['alignment'].get('pearson_r', float('nan')):.2f}",
                                     f"{r['alignment'].get('p_perm', float('nan')):.3f}",
                                     r["alignment"].get("argmax_gap_layer"), r["alignment"].get("argmax_commit_layer")]
                                    for r in lens_rows]},
                 "notes": ["Positive r = the layer that commits the answer is the layer with the spectral spike."]})
        else:
            sections.append({"name": "Logit lens × BBP spike alignment", "summary": "skipped (--only)"})

        if step_requested(cfg, "patching"):
            sections.append(
                {"name": "Activation patching × BBP spike alignment",
                 "table": {"headers": ["prompt", "P_clean", "P_corrupt", "r(restore, gap)", "p"],
                           "rows": [[r["prompt"][:38], f"{r.get('p_clean', float('nan')):.3f}",
                                     f"{r.get('p_corrupt', float('nan')):.3f}",
                                     f"{r.get('alignment', {}).get('pearson_r', float('nan')):.2f}",
                                     f"{r.get('alignment', {}).get('p_perm', float('nan')):.3f}"]
                                    for r in patch_rows]}})
        else:
            sections.append({"name": "Activation patching × BBP spike alignment", "summary": "skipped (--only)"})

        alpha_gap: dict = {}
        if step_requested(cfg, "heavy-tail"):
            t0 = _time.time()
            # one matrix at a time: materializing ALL 2D weights as numpy copies at
            # once needs ~2x model size in RAM (OOM for 7B on CPU); iterating keeps
            # peak usage at one matrix (P2.11)
            import re as _re

            wpat = _re.compile(cfg["weights_filter"])
            mm: list[dict] = []
            for n, p in model.named_parameters():
                if p.ndim == 2 and wpat.search(n):
                    mm.extend(mm_table({n: p.detach().float().cpu().numpy()}))
            mean_gap = mean_gap_profile(gap_profiles)
            alpha_gap = correlate_alpha_with_gap(mm, mean_gap)
            sections.append(
                {"name": "Heavy-tail alpha (Martin & Mahoney bands)",
                 "table": {"headers": ["matrix", "alpha", "xmin", "KS", "lambda_max", "band"],
                           "rows": [[r["matrix"], f"{r['alpha']:.2f}", f"{r['xmin']:.2e}", f"{r['ks']:.3f}",
                                     f"{r['lambda_max']:.2e}", r["band"]] for r in mm[:40]]},
                 "notes": [f"alpha(x gap_ratio): {alpha_gap}",
                           step_duration_note("heavy-tail", t0, _time)]})
        else:
            sections.append({"name": "Heavy-tail alpha (Martin & Mahoney bands)", "summary": "skipped (--only)"})

        if step_requested(cfg, "sae"):
            if not cfg["sae"]:
                sections.append({"name": "SAE audit of the spike window",
                                 "summary": "skipped (enable with --sae; needs sae_lens + Gemma access)"})
            else:
                from stages.stage4_interpretability.sae_bridge import GemmaScopeSource, bbp_feature_audit

                gen_gap = [g[-1] for g in gap_profiles]
                thr = float(np.quantile(gen_gap, 0.7))
                spike_texts = [pr["clean"] for pr, g in zip(pairs, gen_gap) if g >= thr]
                control_texts = [pr["clean"] for pr, g in zip(pairs, gen_gap) if g < thr]
                if len(spike_texts) < 3 or len(control_texts) < 3:
                    # the default 6-prompt set cannot fill both groups (30/70 split
                    # -> ~2 per side): report the skip instead of crashing the run
                    sections.append({
                        "name": "SAE audit of the spike window",
                        "summary": (f"skipped: need >= 3 prompts per group, have "
                                    f"{len(spike_texts)} spike / {len(control_texts)} control "
                                    "(add prompts via --prompts-file)"),
                    })
                else:
                    source = GemmaScopeSource(cfg["sae_release"], cfg["sae_id"], device=cfg["device"],
                                              model=model, tokenizer=tok)
                    audit = bbp_feature_audit(source, spike_texts, control_texts)
                    sections.append({"name": "SAE audit of the spike window",
                                     "table": {"headers": ["feature", "Δ median", "p", "neuronpedia"],
                                               "rows": [[f["feature"], f"{f['delta_median']:.3f}", f"{f['p_perm']:.3f}",
                                                         f["neuronpedia"]] for f in audit["features"]]}})
        else:
            sections.append({"name": "SAE audit of the spike window", "summary": "skipped (--only)"})

        sections.append(criterion_section(self.hypothesis_id, {"sections": sections}))
        return sections, {"steps": step_notes or None}


def step_duration_note(name: str, t0: float, clock) -> str:
    return f"step '{name}' took {clock.time() - t0:.1f}s"


def main() -> None:
    from stages.common.cli import main as runner_main

    runner_main(Stage4Runner())


if __name__ == "__main__":
    main()

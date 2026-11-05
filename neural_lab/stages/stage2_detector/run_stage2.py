"""
stage2_detector.run_stage2 — hallucination detector vs baselines (Stage 2).

Race protocol (fixed before any run):

  1. LOAD   — model + tokenizer with pinned revision, run passport;
  2. DATA   — TruthfulQA (generation), HaluEval QA (candidate scoring),
              SelfCheckGPT wiki_bio (sentence scoring); labels are
              established by the benchmark annotation, NOT by the scores;
  3. OBSERVE — per unit: greedy answer (TruthfulQA) / teacher-forced
              scoring (HaluEval, SelfCheck) + K sampled answers
              (semantic entropy) + last-step hidden states (spectral
              features, probes);
  4. DETECT — tok_* (token entropy family), sem_ent (semantic entropy),
              spec_* (RMT spectral detector), probe_hidden (Azaria &
              Mitchell linear probes per layer);
  5. EVAL   — AUROC + PR-AUC with 95% item-clustered bootstrap CIs;
              paired delta-AUC of every spectral detector vs every
              baseline (clustered permutation p);
  6. REPORT — report.json + REPORT.md + auroc_table.csv.

INTERPRETATION (pre-registered, mirrors the plan):
  - spectral features do NOT beat the probes -> honest negative result,
    publishable as such (the plan explicitly asks for this outcome);
  - spectral features add gain ON TOP of probe + entropy (D vs B style
    delta) -> strong result.

Run:
  python3 neural_lab/stages/stage2_detector/run_stage2.py \
      --model EleutherAI/pythia-70m --benchmarks truthfulqa halueval selfcheckgpt \
      --n-items 25 --out results/stage2/pythia70m
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # .../neural_lab
for p in (_NEURAL_LAB,):
    if p not in sys.path:
        sys.path.insert(0, p)

from stages.common.generation import generate_with_capture, score_candidates  # noqa: E402
from stages.common.hf_model import load_causal_lm, model_passport  # noqa: E402
from stages.common.passports import build_passport, data_hash  # noqa: E402
from stages.common.reporting import save_jsonl, write_report  # noqa: E402
from stages.stage2_detector.benchmarks import (  # noqa: E402
    load_benchmarks,
    make_truthfulqa_judge,
)
from stages.stage2_detector.detectors import Unit, evaluate_detector, paired_delta_auc  # noqa: E402
from stages.stage2_detector.probes import evaluate_probe_layers  # noqa: E402
from stages.stage2_detector.semantic_entropy import JaccardBackend, semantic_entropy_score  # noqa: E402
from stages.stage2_detector.spectral_features import (  # noqa: E402
    last_token_hidden,
    spectral_feature_matrix,
    spectral_summary,
)
from stages.stage2_detector.token_entropy import (  # noqa: E402
    detector_scores_from_features,
    token_entropy_features,
)


def benchmark_item_payload(i) -> dict:
    """Full hashable representation of a benchmark item: every field that
    influences the scored units (labels, candidates, sentences, GPT-3
    samples) — so a dataset update can never keep the old data_hash."""
    return {
        "item_id": i.item_id,
        "benchmark": i.benchmark,
        "context": i.context,
        "question": i.question,
        "correct_answers": list(i.correct_answers),
        "incorrect_answers": list(i.incorrect_answers),
        "candidates": [{"text": c.text, "is_hallucinated": c.is_hallucinated} for c in i.candidates],
        "sentences": [{"text": s.text, "is_nonfactual": s.is_nonfactual} for s in i.sentences],
        "gpt3_samples": list(i.gpt3_samples),
    }


def maybe_chat(tokenizer, text: str, use_chat: bool) -> str:
    """Apply the chat template for instruct models (flag-controlled)."""
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return text
    return text


def observe_units(items, model, tokenizer, cfg) -> list[Unit]:
    """Generate/score everything and produce labeled detector units."""
    judge = make_truthfulqa_judge()
    se_backend = JaccardBackend(threshold=cfg["se_threshold"])
    units: list[Unit] = []
    spectral_summaries: list[dict] = []
    last_hidden: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    bench: list[str] = []

    t0 = time.time()
    for it in items:
        ctx = maybe_chat(tokenizer, it.context, cfg["use_chat"])
        if it.candidates:  # --- HaluEval: candidate scoring ----------------
            scored = score_candidates(model, tokenizer, ctx, [c.text for c in it.candidates])
            for cand, sc in zip(it.candidates, scored):
                feats = token_entropy_features(
                    [{"entropy": e, "logprob": lp} for lp, e in zip(sc["per_token_logprob"], sc["per_token_entropy"])]
                )
                scores = detector_scores_from_features(feats)
                units.append(Unit(cand.text[:40], it.benchmark, it.item_id,
                                  int(cand.is_hallucinated), scores))
        elif it.sentences:  # --- SelfCheckGPT: sentence scoring -------------
            for si, s in enumerate(it.sentences):
                sc = score_candidates(model, tokenizer, it.context, [s.text])[0]
                feats = token_entropy_features(
                    [{"entropy": e, "logprob": lp} for lp, e in zip(sc["per_token_logprob"], sc["per_token_entropy"])]
                )
                scores = detector_scores_from_features(feats)
                if cfg["n_samples"] >= 2:
                    samples = []
                    for k in range(cfg["n_samples"]):
                        gen = generate_with_capture(
                            model, tokenizer, it.context, max_new_tokens=cfg["max_new_tokens"],
                            temperature=cfg["temperature"], seed=cfg["seed"] + 1000 * k + si,
                            capture_hidden=False, fast=cfg["fast"],
                        )
                        samples.append(gen["generated_text"])
                    scores["sem_ent"] = semantic_entropy_score(samples, se_backend)
                units.append(Unit(f"selfcheck-{it.item_id}-s{si}", it.benchmark, it.item_id,
                                  int(s.is_nonfactual), scores))
        else:  # --- TruthfulQA: generation + judge -------------------------
            gen = generate_with_capture(
                model, tokenizer, ctx, max_new_tokens=cfg["max_new_tokens"],
                temperature=0.0, capture_hidden=True, fast=cfg["fast"],
            )
            answer = gen["generated_text"].strip()
            label = judge(answer, it.correct_answers, it.incorrect_answers)
            scores = detector_scores_from_features(token_entropy_features(gen["per_step"]))
            if cfg["n_samples"] >= 2:
                samples = []
                for k in range(cfg["n_samples"]):
                    g = generate_with_capture(
                        model, tokenizer, ctx, max_new_tokens=cfg["max_new_tokens"],
                        temperature=cfg["temperature"], seed=cfg["seed"] + 1000 * k,
                        capture_hidden=False, fast=cfg["fast"],
                    )
                    samples.append(g["generated_text"])
                scores["sem_ent"] = semantic_entropy_score(samples, se_backend)
            summ = spectral_summary(gen["final_hidden"])
            scores["spec_gap_last"] = summ["gap_ratio_last"]
            spectral_summaries.append(summ)
            last_hidden.append(last_token_hidden(gen["final_hidden"], gen["prompt_len"]))
            labels.append(label)
            groups.append(it.item_id)
            bench.append(it.benchmark)
            units.append(Unit(it.item_id, it.benchmark, it.item_id, label, scores))
            it.label = label
        print(f"  [{it.item_id}] done ({len(units)} units, {time.time() - t0:.0f}s)", flush=True)
    return units, spectral_summaries, last_hidden, np.asarray(labels), groups, bench


def add_probe_units(units, last_hidden, labels, groups, bench, cfg) -> tuple[list[Unit], dict]:
    """Fit per-layer probes and add `probe_hidden` scores to the
    generation-scored units (TruthfulQA).

    With a single benchmark (TruthfulQA alone) the leave-one-benchmark-out
    protocol is undefined, so the within-benchmark grouped OOF is used and
    the limitation is reported instead of silently producing NaN scores."""
    if len(last_hidden) < 20 or len(np.unique(labels)) < 2:
        return units, {"skipped": "insufficient labeled units for probe fitting"}
    res = evaluate_probe_layers(last_hidden, labels, groups, bench, n_folds=cfg["n_folds"], seed=cfg["seed"])
    oof = np.asarray(res["oof_proba"], dtype=np.float64)
    tqa_units = [u for u in units if u.benchmark == "truthfulqa"]
    oof_map = dict(zip([u.item_id for u in tqa_units], oof))
    n_assigned = 0
    for u in units:
        if u.benchmark == "truthfulqa" and u.item_id in oof_map and np.isfinite(oof_map[u.item_id]):
            u.scores["probe_hidden"] = float(oof_map[u.item_id])
            n_assigned += 1
    info = {
        "per_layer_auc": res["per_layer_auc"],
        "best_layer": res["best_layer"],
        "oof_source": res.get("oof_source"),
        "oof_note": res.get("oof_note"),
        "n_units_scored": n_assigned,
    }
    if len(set(bench)) >= 2:
        info["best_layer_leave_one_benchmark_out"] = res["best_layer_lob"]["per_benchmark"]
    return units, info


def add_spectral_units(units, spectral_summaries, labels, groups, cfg) -> list[Unit]:
    """Single-feature spectral scores + honest OOF spectral probe.

    spec_gap_last is already written by observe_units (from the fresh
    per-item summary) — this adds only the features that exist solely
    here, so each detector keeps ONE source."""
    n = len(spectral_summaries)
    if n == 0:
        return units
    X, keys = spectral_feature_matrix(spectral_summaries)
    tqa = [u for u in units if u.benchmark == "truthfulqa"]
    for row_i, u in enumerate(tqa):
        u.scores["spec_ncrit_free"] = float(X[row_i, keys.index("ncrit_free_last")])
    if n >= 20 and len(np.unique(labels)) >= 2:
        from stages.stage2_detector.probes import probe_layer_auroc, grouped_folds
        from science.predictors import LogisticRegression

        folds = grouped_folds(groups, n_folds=cfg["n_folds"], seed=cfg["seed"])
        oof = np.full(n, np.nan)
        for tr, te in folds:
            if len(np.unique(labels[tr])) < 2:
                continue
            clf = LogisticRegression(l2=1.0)
            clf.fit(X[tr], labels[tr])
            oof[te] = clf.predict_proba(X[te])
        for u, p in zip(tqa, oof):
            if np.isfinite(p):
                u.scores["spec_probe"] = float(p)
    return units


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: detector vs baselines (AUROC)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["truthfulqa", "halueval", "selfcheckgpt"])
    ap.add_argument("--n-items", type=int, default=25)
    ap.add_argument("--n-samples", type=int, default=5, help="K for semantic entropy")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-chat", action="store_true", help="apply chat template (instruct models)")
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()

    cfg = {
        "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "seed": args.seed, "n_folds": args.n_folds,
        "use_chat": args.use_chat, "se_threshold": 0.5, "fast": args.fast,
    }
    print(f"[stage2] loading {args.model} …", flush=True)
    model, tokenizer = load_causal_lm(args.model, revision=args.revision, dtype=args.dtype, device=args.device)
    passport = build_passport(
        experiment_id="stage2_detector_v1",
        hypothesis_id="S2",
        model_fields=model_passport(model, tokenizer),
        config={**cfg, "model": args.model, "benchmarks": args.benchmarks, "n_items": args.n_items},
    )
    items = load_benchmarks(args.benchmarks, n_items=args.n_items, seed=args.seed)
    passport["config"]["data_hash"] = data_hash([benchmark_item_payload(i) for i in items])
    print(f"[stage2] {len(items)} benchmark items", flush=True)

    units, spec_summ, last_h, labels, groups, bench = observe_units(items, model, tokenizer, cfg)
    units, probe_info = add_probe_units(units, last_h, labels, groups, bench, cfg)
    units = add_spectral_units(units, spec_summ, labels, groups, cfg)

    det_names = sorted({d for u in units for d in u.scores})
    table, results = [], {}
    for d in det_names:
        scored = [u for u in units if d in u.scores]
        try:
            r = evaluate_detector(scored, d, n_boot=args.n_boot, seed=args.seed)
        except ValueError as e:
            r = {"detector": d, "error": str(e)}
        results[d] = r
        if "auroc" in r:
            table.append([d, f"{r['auroc']:.3f}",
                          f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]", f"{r['pr_auc']:.3f}", r["n_units"]])

    deltas = []
    spec_dets = [d for d in det_names if d.startswith(("spec_", "probe_hidden"))]
    base_dets = [d for d in det_names if d.startswith(("tok_", "sem_ent"))]
    for a in spec_dets:
        for b in base_dets:
            if a in results and b in results and "auroc" in results.get(a, {}) and "auroc" in results.get(b, {}):
                dd = paired_delta_auc(units, a, b, n_boot=args.n_boot, seed=args.seed)
                deltas.append({"spectral": a, "baseline": b, **dd})

    save_jsonl(os.path.join(args.out, "units.jsonl"), [u.__dict__ for u in units])
    sections = [
        {"name": "AUROC race (higher = better hallucination detection)",
         "table": {"headers": ["detector", "AUROC", "95% CI", "PR-AUC", "n"], "rows": table},
         "notes": [
             "tok_* = token-entropy baselines; sem_ent = semantic entropy (Farquhar et al. 2024);",
             "probe_hidden = linear probe on last-token hidden states (Azaria & Mitchell protocol, OOF);",
             "spec_* = RMT spectral detector (gap_ratio / ncrit_free / spectral probe).",
         ]},
        {"name": "Spectral vs baselines: paired ΔAUROC (clustered bootstrap)",
         "table": {"headers": ["spectral", "baseline", "ΔAUROC", "95% CI", "p", "n_paired"],
                   "rows": [[d["spectral"], d["baseline"], f"{d['delta_auc']:+.3f}",
                             f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", f"{d['p_two_sided']:.3f}",
                             d.get("n_paired")]
                            for d in deltas]},
         "notes": ["CI excluding 0 = the difference survives item-clustered resampling.",
                   "n_paired: the Δ was computed only on units carrying BOTH detectors."]},
        {"name": "Probe diagnostics", "summary": str(probe_info)},
    ]
    path = write_report(args.out, "Stage 2 — Hallucination detector vs baselines", passport, sections)
    print(f"[stage2] report: {path}", flush=True)


if __name__ == "__main__":
    main()

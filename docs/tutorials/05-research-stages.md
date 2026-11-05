# 5. Research Stages 2–5 on Real Models

This tutorial walks through the empirical program in
[`neural_lab/stages/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/neural_lab/stages):
four pre-registered experiments on real HuggingFace checkpoints, sharing
the `science/` core's ground rules (passports, frozen protocols, honest
out-of-fold evaluation, negative results reported as is).

## Install

```bash
pip install torch transformers datasets pyyaml scipy
```

CPU is enough for models up to ~1B parameters.

## Stage 2 — hallucination detector vs baselines

The race: spectral features vs token entropy, semantic entropy
(Farquhar et al. 2024) and linear probes (Azaria & Mitchell), all
evaluated by AUROC with item-clustered bootstrap CIs on TruthfulQA,
HaluEval and SelfCheckGPT.

```bash
python3 neural_lab/stages/stage2_detector/run_stage2.py \
    --model EleutherAI/pythia-70m --n-items 25 \
    --out results/stage2/pythia70m
```

The report writes the AUROC table and the pre-registered verdict:
spectral features losing to the probes is a publishable negative result;
a ΔAUROC with a CI excluding 0 on top of the baselines is the strong
outcome. For instruct models add `--use-chat`.

> **Warning.** `--use-chat` applies the tokenizer's chat template; on a
> BASE model without one it may silently produce a wrong prompt format.
> Only use it for models whose card documents a chat template, and check
> the rendered prompts in the passport. Note also that stage 2 sampling
> (`--temperature > 0`) requires an explicit `--seed`: the run refuses to
> start otherwise, since unseeded sampling is not reproducible.

## Stage 3 — N_crit from first-error survival

Models answer verifiable arithmetic tasks (truth recomputed
independently); the first wrong number gives a token-level collapse time,
never-erroring trajectories are right-censored. A Weibull survival fit
gives `τ, γ`; with an external `--mu-eff` (e.g. the stage-5 KL proxy) the
fit also returns β — pass `--mu-eff-from results/stage5/<pair>` to take
the proxy from a stage-5 run instead of copying the number by hand.
Several models then feed the scaling test
`log τ ~ log μ_eff` with the pre-registered target slope 2 (β = 0.5).

```bash
python3 neural_lab/stages/stage3_ncrit/run_stage3.py \
    --model EleutherAI/pythia-70m --n-tasks 60 \
    --out results/stage3/pythia70m --mu-eff 0.42

# after >= 3 models (stage-3 fit.json AND stage-5 pair dirs both work):
python3 neural_lab/stages/stage3_ncrit/run_stage3.py \
    --scaling-in "results/stage3/*" "results/stage5/*" --out results/stage3/scaling
```

## Stage 4 — the interpretability bridge

Logit lens, activation patching, heavy-tail α (Martin & Mahoney) and —
optionally — a Gemma Scope SAE audit of the BBP-spike window:

```bash
python3 neural_lab/stages/stage4_interpretability/run_stage4.py \
    --model EleutherAI/pythia-70m --out results/stage4/pythia70m
# SAE audit (needs sae_lens + Gemma access + GPU):
#   ... --sae --sae-release gemma-scope-2b-pt-res-canonical --device cuda
```

The report aligns each mechanism profile with the spectral gap profile
and prints the alignment r with a layer-label permutation p.

## Stage 5 — base vs tuned pairs

```bash
python3 neural_lab/stages/stage5_rlhf/run_stage5.py --list
python3 neural_lab/stages/stage5_rlhf/run_stage5.py \
    --pair qwen2.5-0.5b --out results/stage5
python3 neural_lab/stages/stage5_rlhf/run_stage5.py --all-open \
    --out results/stage5 --dtype float16 --device cuda   # for the 7B pairs
```

Each pair gets paired spectra, Δτ from a shared arithmetic task set (with
a paired bootstrap CI), the KL-based μ_eff proxy, and a pre-registered
verdict (supported / refuted / not resolved / inconclusive — all four are
reported outcomes). Each pair also writes a stage-3-compatible
`fit.json`, which is what makes the `--scaling-in` aggregation above work
on stage-5 outputs directly.

## Tests

```bash
make test-stages   # offline suite: tiny random GPT-2 + synthetic data
```

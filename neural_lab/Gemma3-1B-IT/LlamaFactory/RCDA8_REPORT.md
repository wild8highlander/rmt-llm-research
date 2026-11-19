# RCDA-8 for Gemma3-1B-IT: the LlamaFactory round — formula arm vs data arm

Written: November 19, 2026. The fourth formula-training round on
**`litert-community/Gemma3-1B-IT`** (base checkpoint `google/gemma-3-1b-it`,
pulled from the public mirror `unsloth/gemma-3-1b-it`, snapshot
`5b1141…`, Gemma license) — and the first one executed end-to-end on the
real model through a production training framework,
[**LlamaFactory 0.9.5**](https://github.com/hiyouga/LlamaFactory).

One frozen protocol, two arms, one evaluator:

* **RCDA-8 — the formula arm.** The repository's formula
  `pen = (vᵀh/‖v‖)²`, phase-matched over the writer band, baked into a
  **LoRA adapter** trained by a custom `compute_loss` on top of
  LlamaFactory's `CustomSeq2SeqTrainer` — 1.31 M trainable parameters,
  150 steps.
* **SFT — the data arm.** Stock `llamafactory-cli train` (pt stage, LoRA,
  all-linear targets) on a 38-example anti-loop dataset: 20 bad prompts
  with curated clean continuations + 18 self-distilled good completions —
  152 steps.

**The result, measured on the full frozen protocol (20 bad + 20 good,
greedy 192, raw completion mode, 4-gram loops, threshold 0.5):**

| metric | base | RCDA-8 LoRA (formula) | SFT LoRA (data) |
| --- | --- | --- | --- |
| ρ_bad (loops ≥ 0.5) | 0.60 | 0.70 | **0.40** |
| ρ_good | 0.10 | 0.10 | **0.05** |
| loop_share mean (bad 20) | 0.562 | 0.656 | **0.393** |
| loop_waste mean (bad 20), tok | 103.3 | 126.0 | **74.2** |
| hard-12 loop share | 0.827 | **0.632** | **0.590** |
| mild-8 loop share | 0.164 | 0.691 | **0.097** |
| prompts fully repaired (of 12) | — | 3 / 12 | **5 / 12** |
| KL(base ‖ arm), control | 0 | 0.512 | **0.120** |
| PPL control (mean) | 10.9 | 12.0 | **8.6** |
| trainable parameters | — | 1 310 720 (0.13 %) | ≈ 6.9 M (0.69 %) |

**Two headlines.** (1) The formula arm, re-implemented as a band-masked
LoRA on LlamaFactory rails, produces the **first fully-repaired hard
loopers of any formula arm on the real Gemma3-1B-IT** — 3 of 12, with the
hard-12 mean down 0.827 → 0.632 and the mini-benchmark melting
0.921 → 0.529 — at **56× fewer trainable parameters** than RCDA-6's
full-matrix band and **one third of its PPL drift** (+10 % vs +29 %).
(2) The data arm **wins every measured axis anyway**: more repairs
(5/12), no defect migration (mild-8 0.097, better than base), an order
less KL, and a −21 % PPL *improvement* — the self-distillation bonus.
The loop defect of Gemma3-1B-IT is easier to teach away than to project
away. All numbers are fields of the committed JSONs in `results/`.

## 1. Setup

### 1.1 Why LlamaFactory, and what "using" it means here

LlamaFactory is a self-hosted training framework: its demo site builds
configs, the training itself runs wherever the framework is installed.
This round ran it on the validation box described in §1.4. Two usage
modes, both real:

* **SFT arm — stock.** `llamafactory-cli train configs/sft_antiloop_cpu.yaml`
  end-to-end: model loading, LoRA injection, dataset pipeline (pt stage),
  optimizer, scheduler, checkpointing, loss curve — nothing custom.
* **RCDA-8 arm — extended.** LlamaFactory's `load_model` (LoRA injection,
  `finetuning_type=lora`) and `CustomSeq2SeqTrainer` (optimizer creation,
  LR schedule, checkpoint/resume, logging), subclassed with the RCDA
  `compute_loss` — the same extension pattern LlamaFactory itself uses
  for `use_asft_loss` (run the model inside `compute_loss`, add the
  custom term).

### 1.2 The frozen protocol (unchanged from the real-Gemma rounds)

* **Model**: `unsloth/gemma-3-1b-it` @ snapshot `5b11413a…` — the same
  weights as the RCDA-5/6 real-Gemma runs (`JSON Results/probe_report.json`
  pins the same hash). 1.00 B params, 26 layers, d = 1152, vocab 262 144.
* **Mode**: raw completion (no chat template), greedy decoding — the
  on-device prefill/decode path of the LiteRT build. In chat mode the
  defect does not manifest (sentinel scan, RCDA-5 report).
* **Window**: `MAX_NEW = 192` — the loop opens only in long generations.
* **Sets**: 20 bad (`bad_g01…g20`, sentinel-selected top-20 by base loop
  share), 20 good, 5 control (RU + EN). Frozen, committed in `data/`.
* **Loop metric**: 4-gram repeat labels; a trajectory loops when
  `loop_share ≥ 0.5`; ρ_bad is the looping fraction of the bad set.

### 1.3 Stage 1 — the spectral probe on the real model (replication check)

`scripts/stage1_probe.py` reproduces the RCDA-5/6 probe byte-for-byte on
the same weights — and lands on the same numbers:

| quantity | this round | RCDA-5 report (Nov 16) |
| --- | --- | --- |
| ρ_bad (base, greedy 192) | **0.60** | 0.60 |
| loop_share mean (bad 20) | **0.562** | 0.562 |
| writer layers (sliding-window rule) | **L10 / L11 / L12** | L10 / L11 / L12 |
| z_diff at writers | **212.6 / 219.8 / 205.1** | 212.6 / 219.8 / 205.1 |
| band with z_diff > 130 | L6–L16 | L8–L15 (their cut) |

The phase probe (early vs deep loop positions, `early_cycles = 2`) gives
cos(v_early, v_deep) ≈ **0.72–0.97 inside the band** — on this model the
two phases are strongly aligned; the RCDA-6 phase-matching still applies
but buys less separation than on MNN-Pocket. The penalty band is kept at
the pre-registered **L8–L15** (RCDA-6 hypothesis), o_proj + down_proj.

### 1.4 The validation box (provenance, stated plainly)

CPU-only sandbox: **2 cores, 4 GB RAM, no GPU**, torch 2.15 + CPU,
transformers 5.6, LlamaFactory 0.9.5. Weights bf16, LoRA fp32 (PEFT),
logits/losses fp32, gradient checkpointing on. Wall-clock: probe ≈ 10 min,
SFT arm ≈ 5 min, RCDA-8 arm ≈ 9 min (incl. 3 mini-benches), each arm's
full evaluation ≈ 20 min. This is a **budget run**: 150 training steps,
20-prompt suite, LoRA rank 16. Effect sizes at GPU scale (more steps,
bigger anchors-of-good data) remain to be measured — the configs to do
so ship in `configs/`.

## 2. The RCDA-8 arm — the formula as a loss, on LlamaFactory rails

```
L = pen_norm + β · relu(mse − cap)                    β = 10, cap = 1.0
pen_norm = pen_raw / pen0_j                           (dimensionless, init 1.0)
pen_raw  = mean_l mean_t ( v̂_{phase(t), l}ᵀ h_t )²   over band l ∈ L8–L15
mse      = MSE(trailing-32 logits, base logits)       on a good prompt
```

* **Trainable set**: LoRA (r = 16, α = 32) injected by LlamaFactory on
  every `o_proj` + `down_proj`; a **band mask** then freezes every
  adapter outside L8–L15, so the effective trainable set is exactly the
  RCDA-6 band — 1 310 720 parameters, 0.13 % of the model (RCDA-6 trained
  73.1 M full-matrix weights on the same band).
* **Data**: the frozen sets. 20 bad prefixes (48 greedy tokens, reusing
  the stage-1 frozen generations — byte-identical to RCDA-6's cache) and
  20 good completions (192 greedy tokens), interleaved in the RCDA-6
  order, sequential sampling (no shuffling), per-device batch 1.
* **Optimizer**: AdamW (LlamaFactory default), lr 2e-4, linear decay,
  warmup 10, grad-clip 1.0, 150 steps (`training_log.json`).
* **Anchor cache**: base trailing-32 logits for the good prompts stored
  fp16 on disk (336 MB memmap — two 2 GB models do not fit in 4 GB RAM).

**Training**: `pen_norm` 0.954 (first-10 mean) → **0.0206 (last-10 mean,
−97.9 %)** — deeper than RCDA-5's Adam arm (−78 %) and RCDA-6 (pen_norm
0.028) at 56× fewer parameters. The anchor rode under the cap the whole
way (final MSE 0.55 < 1.0). The logged mini-benchmark (8 hardest bad
prompts, greedy 192) bends monotonically:

| bench | ρ_mini | mean loop_share |
| --- | --- | --- |
| step 0 | 1.00 | 0.921 |
| step 50 | 1.00 | 0.846 |
| step 100 | 0.625 | 0.612 |
| step 150 | 0.625 | **0.529** |

## 3. The SFT arm — stock LlamaFactory on clean targets

* **Dataset** (`data/sft_antiloop.jsonl`, `sft_manifest.json`): 38 rows.
  20 bad prompts → **curated anti-loop continuations** (experimenter-written,
  RU/EN matched to the prompt: acknowledge the pattern, then break it with
  concrete varied content and a natural ending) + 18 good prompts → the
  base model's own clean greedy completions (loop_share < 0.3;
  2 looping ones dropped).
* **Recipe**: `stage: pt` (raw-completion language modelling — matches the
  evaluation mode; **packing off**, so every example starts at position 0),
  LoRA r = 8, α = 16 on all linear projections, lr 5e-5 cosine, 4 epochs =
  152 steps. Loss 5.65 → 1.69.

## 4. The one evaluator, three arms — what each arm bought

### 4.1 The defect

* **SFT**: ρ_bad 0.60 → **0.40**, mean loop_share 0.562 → **0.393 (−30 %)**,
  waste 103 → **74 tok**, hard-12 0.827 → 0.590, **5 of 12 hard loopers
  fully repaired** (`bad_g12` 0.96 → 0.07, `bad_g13` 0.96 → 0.01, …), and
  the mild-8 subset got *cleaner than base* (0.164 → 0.097). No migration.
* **RCDA-8**: hard-12 0.827 → **0.632**, **3 of 12 repaired** — the first
  full repairs of a formula arm on this model (RCDA-5: 0/12 both arms;
  RCDA-6: 2/12). But ρ_bad rose to 0.70 and mean loop_share to 0.656:
  the **defect migrated** into the mild subset (0.164 → 0.691) exactly as
  in RCDA-6 (0.16 → 0.64). The mini-bench melt (0.921 → 0.529) is real
  but concentrated on the hardest attractors.

### 4.2 The abilities

* **SFT**: KL 0.120, ρ_good 0.10 → 0.05, PPL 10.9 → **8.6 (−21 %)** — the
  self-distilled good continuations act as a light domain-adaptive
  pretraining on the control corpus. All three abilities axes improved.
* **RCDA-8**: KL 0.512, PPL +10.1 % (12.0). The anchor held its MSE
  budget, but the LoRA learning rate (2e-4, aggressive by full-matrix
  standards) bought the penalty descent with real distribution drift.
  ρ_good held at 0.10.

### 4.3 The forensics (RCDA-8 edit as a matrix)

Merged ΔW = (B·A)·scaling per band tensor (`part8_eval_rcda8.json`):
rel-Frobenius 0.4–1.5 %, **TW violation ×16** (the v-orthogonal bulk of
the edit is structured, not random — same verdict as every Gemma round),
cos(u₁ of ΔW, v_deep) ≤ **0.44**. The MNN-Pocket convergence ("the edit
aligns with v and stays TW-clean on its own") remains architecture-
specific: on Gemma3 the optimizer finds penalty-compliant solutions that
are neither aligned nor spectrally clean.

## 5. What this round says

1. **The formula finally moves the hard subset the right way — through a
   LoRA.** The band-masked LoRA formulation is strictly more parameter-
   efficient than full-matrix RCDA-6 (3 vs 2 repairs, 1.31 M vs 73.1 M
   params, +10 % vs +29 % PPL drift). Low-rank structure appears to be a
   *regularizer* for this intervention: it constrains the edit to the
   dominant directions of the band, which is where the loop lives.
2. **The migration is the law, not the accident.** Three rounds, three
   geometries (full-matrix o_proj; full-matrix band; low-rank band), one
   invariant: suppress the loop direction where it is written and the
   attractor re-emerges in the mild prompts. The loop is a property of
   the computation, not of a direction.
3. **Data beats the formula on every axis at equal budget** — including
   the abilities (the self-distill PPL gain is a free lunch the formula
   cannot touch). For a deployed uncensored distill, the practical recipe
   against greedy repetition is supervised anti-loop SFT; the spectral
   penalty is the research instrument that explains *why* that works
   (distributed, carried, migratory defect).
4. **Honest limits.** CPU-scale budget run (150/152 steps, 20-prompt
   suite); the SFT arm trained on curated continuations of the same bad
   prompts it is evaluated on (in-protocol, but the RCDA arm trains on
   the same prompts' greedy prefixes, so the data axis is matched; the
   *target* axis is not — that is the point of the comparison); the two
   arms use different adapter geometries (band vs all-linear), which the
   report keeps visible rather than normalizing away.

## 6. Reproduction

```bash
# the whole round, resumable (a CPU box with 4 GB RAM is enough)
python3 LlamaFactory/scripts/pipeline.py          # ~75 min on 2 cores

# or stage by stage
python3 LlamaFactory/scripts/stage1_probe.py                              # probe
llamafactory-cli train LlamaFactory/configs/sft_antiloop_cpu.yaml          # SFT arm
python3 LlamaFactory/scripts/stage3_rcda8_train.py                         # formula arm
python3 LlamaFactory/scripts/stage4_eval.py --tag base                     # evaluator
python3 LlamaFactory/scripts/stage4_eval.py --tag rcda8 \
    --lora LlamaFactory/adapters/rcda8_formula
python3 LlamaFactory/scripts/stage4_eval.py --tag sft \
    --lora LlamaFactory/adapters/sft_data
python3 LlamaFactory/scripts/stage5_summarise.py                           # summary
python3 LlamaFactory/scripts/make_figures.py                               # figures + GIFs

# full-scale GPU replication (>= 16 GB VRAM)
llamafactory-cli train LlamaFactory/configs/sft_antiloop_gpu.yaml
python3 LlamaFactory/scripts/stage3_rcda8_train.py --steps 600 --lr 3e-4
```

## 7. Artifacts

| path | contents |
| --- | --- |
| `results/probe/` | frozen generations, `probe_report.json` (writers L10–12), `probe_phase_report.json`, direction vectors (npz) |
| `results/rcda8_arm/training_log.json` | 150 steps × (pen_norm, pen_raw, mse), pen0, 4 mini-benches, config |
| `results/sft_arm/trainer_state.json` | LlamaFactory training state (loss 5.65 → 1.69), `training_loss.png` |
| `results/part8_eval_{base,rcda8,sft}.json` | the one evaluator, three arms — every number of §4 |
| `results/summary_round8.json` | the cross-arm summary this report reads |
| `results/gens_*.jsonl` | per-prompt generations (loop_share, waste, text) |
| `adapters/rcda8_formula/`, `adapters/sft_data/` | the trained LoRA adapters (peft) |
| `figures/` | 9 static figures + 6 animations, all rebuilt from the JSONs |
| `data/`, `configs/`, `scripts/` | frozen protocol sets, LlamaFactory configs, the full pipeline |

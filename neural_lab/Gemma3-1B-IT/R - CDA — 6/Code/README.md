# Code/ — the Gemma3-1B-IT RCDA-6 toolkit

Everything needed to re-run the expanded-tensor experiment on CPU. Python
≥ 3.10; `pip install -r requirements.txt` (torch CPU, transformers,
safetensors, numpy, matplotlib). The base model goes through `--model` (any
local dir with `model.safetensors` + tokenizer, or an HF id;
`google/gemma-3-1b-it` requires Gemma license acceptance,
`unsloth/gemma-3-1b-it` is the public mirror of the same weights).

RCDA-6 = RCDA-5's kit with two pre-registered changes: the trainable set is
**o_proj + down_proj over the band L8–L15** (73.1 M params, 7.31 %), and the
penalty direction is **conditioned on the loop phase** (early / deep), not
only on the collapse point. The loss formula itself is unchanged.

| File | Role |
|------|------|
| `gemma_lab.py` | shared runtime (RCDA-6 copy): adds `down_proj_key()`, `cycle_period()` — smallest token period of the loop tail — and `phase_labels()` (pre / early / deep per position) used by both the probe and the penalty |
| `gemma_probe.py` | the RCDA-5 collapse probe (kept for reuse: it produces the frozen generations the phase probe consumes) |
| `gemma_probe_phase.py` | stage 1 — the PHASE probe: reuses the frozen RCDA-5 generations, splits every looping trajectory into pre/early/deep, extracts `v_early_l` / `v_deep_l` per layer + `z_early/z_deep`/cos profiles; period-irregular trajectories are excluded from the direction pools |
| `gemma_rcds6_train.py` | stage 2 — the formula as a loss over the expanded set: `L = pen_norm + β·relu(mse − cap)` with the phase-matched penalty on o_proj+down_proj of L8–L15, fp32 masters + bf16 compute, per-prompt init normalization, atomic resumable checkpoints (`ckpt_masters.pt` + `ckpt_opt.pt`, every 10 steps), final TW/cos forensics with incremental persistence, `trained_state.safetensors` (16-tensor delta) |
| `gemma_benchmark.py` | stage 3 — the evaluator (RCDA-6 copy): 20 bad + 20 good greedy 192, ρ_bad/ρ_good/loop_waste, KL(base‖arm), TW control per edited tensor, cos(u₁, v) — with lazy per-key base weights (a 4 GB box cannot hold the model and a materialized 2 GB checkpoint at once) and a resumable TW cache |
| `marchenko_pastur.py` | MP law + Johnstone TW centering + BBP thresholds + `spectral_tw_control` (unchanged from RCDA-5) |

Conventions fixed in `gemma_lab.py`: raw completion mode, `MAX_NEW = 192`,
`NGRAM = 4`, looping = loop share ≥ 0.5, `hidden_states[l + 1]` = residual
stream after decoder layer `l`.

Typical chain:

```bash
python3 gemma_probe.py        --model <model-dir>          # once (RCDA-5 kit has it)
python3 gemma_probe_phase.py  --model <model-dir> \
                              --probe <work>/probe --band 8 9 10 11 12 13 14 15
python3 gemma_rcds6_train.py  --model <model-dir> --steps 150 --lr 3e-3 --optim sgd
python3 Research/RCDA/tools/rcda6_minibench.py --tag step150
python3 gemma_benchmark.py    --model <model-dir> \
                              --edited-state <work>/rcda6/trained_state.safetensors \
                              --tag rcda6_trained --out part6 --probe <work>/probe_phase
```

Engineering notes from the recorded run (2 cores / 4 GB RAM):

* checkpoints are ATOMIC (`tmp` + `os.replace`) and split into a masters
  file (mmap-loaded read-only) and an optimizer file (plain load — a mapped
  storage must never end up inside the optimizer: the first in-place
  momentum update on it stalls the process);
* the anchor logits live in a fp16 `np.memmap`, not in RAM;
* generation inside the training process slows ~6–10× under the training
  allocator — run benches/evals in fresh processes
  (`rcda6_minibench.py`, `gemma_benchmark.py` are separate invocations);
* `use_cache=False` is passed explicitly for the training forwards.

# Code/ — the Gemma3-1B-IT RCDA-5 toolkit

Everything needed to re-run the experiment on CPU. Python ≥ 3.10;
`pip install -r requirements.txt` (torch CPU, transformers, safetensors,
numpy, matplotlib). The base model goes through `--model` (any local dir with
`model.safetensors` + tokenizer, or an HF id; `google/gemma-3-1b-it` requires
Gemma license acceptance, `unsloth/gemma-3-1b-it` is the public mirror of the
same weights).

| File | Role |
|------|------|
| `gemma_lab.py` | shared runtime: path resolution (`$GEMMA_WORK`, default `<lab>/rcda_work`), raw-completion generation, 4-gram loop labeling/collapse points, o_proj plumbing, fp32-master SGD wrapper |
| `gemma_probe.py` | stage 1 — behavioral + spectral probe: greedy 192 per bad prompt, loop/pre-loop pooling per layer, `Δh`/`z_diff`/β profile, sliding-window writer selection, `v_L{l}.npy` per writer layer, `probe_report.json` |
| `gemma_rcds_train.py` | stage 2 — the formula as a loss: `L = pen_norm + β·relu(mse − cap)` on o_proj of the writer layers, fp32 masters + bf16 compute, per-prompt init normalization, resume via `ckpt.pt` (every 10 steps), benches at 0/75/150, final TW/cos analysis, `trained_state.safetensors` (delta) |
| `gemma_benchmark.py` | stage 3 — the evaluator (port of the MNN kit's `rcda4_eval_b.py`): 20 bad + 20 good greedy 192, ρ_bad/ρ_good/loop_waste, KL(base‖arm) on control texts, TW control per edited tensor, cos(u₁, v); row-level resume |
| `marchenko_pastur.py` | MP law (bounds/density/CDF/Stieltjes/sampling), Johnstone TW centering, z-scores in TW units, BBP spike threshold, and `spectral_tw_control` — the spectral purity gate of a weight edit along a direction `v` |

Conventions fixed in `gemma_lab.py`: prompts are fed in **raw completion mode**
(no chat template; the MNN-Pocket analog), `MAX_NEW = 192`, `NGRAM = 4`,
looping = loop share ≥ 0.5, and `hidden_states[l + 1]` is the residual stream
after decoder layer `l` (so the penalty projects the stream that layer `l`'s
o_proj writes into).

Typical chain:

```bash
python3 gemma_probe.py       --model <model-dir>
python3 gemma_rcds_train.py  --model <model-dir> --steps 150 --lr 3e-3 \
                             --optim sgd --bench-at 75 150
python3 gemma_benchmark.py   --model <model-dir> --edited-state none \
                             --tag base --out part5 --reuse-gens
python3 gemma_benchmark.py   --model <model-dir> \
                             --edited-state <work>/rcda5_final/trained_state.safetensors \
                             --tag trained --out part5
```

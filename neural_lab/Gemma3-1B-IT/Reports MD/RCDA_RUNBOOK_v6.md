# RCDA RUNBOOK — Gemma3-1B-IT port (how to re-run the whole experiment)

Everything runs on CPU (the recorded run: 2 cores / 4 GB RAM, ~2.5 h wall
clock total). Set `GEMMA_WORK` to a scratch dir; results JSONs land there and
are copied into `Research/RCDA/results/`. `MODEL` below is any directory (or
HF id) holding the gemma-3-1b-it weights; for the public mirror use
`unsloth/gemma-3-1b-it`, for the canonical card `google/gemma-3-1b-it`
(Gemma license acceptance required).

```bash
export GEMMA_WORK="$HOME/gemma_rcda5_work"
export MODEL="unsloth/gemma-3-1b-it"

# 0. environment

pip install -r Code/requirements.txt     # torch CPU wheel is enough

# 1. sentinel scan (optional — the frozen selection ships in results/)
#    candidates -> greedy 192 -> 4-gram loop share -> top-20 bad set
#    (the recorded pool scan: 50 candidates, selection in
#     Research/RCDA/results/sentinel_selection.json)

# 2. probe: writers + loop directions  (~7 min)

python3 Code/gemma_probe.py --model "$MODEL"
#   -> $GEMMA_WORK/probe/{probe_report.json, gens.jsonl, v_L10..12.npy}
#   expected: writers [10, 11, 12], z_diff ~205–220, rho_mini_strict 0.60

# 3. training — the user-spec arm (SGD-150, resumable via ckpt.pt, ~15 min)

python3 Code/gemma_rcds_train.py --model "$MODEL" --steps 150 --lr 3e-3 \

```text
--optim sgd --momentum 0.9 --bench-at 75 150 --bench-n 8 \
--out "$GEMMA_WORK/rcda5_final"
```text
#   if the session is cut: rerun the exact same command — it resumes
#   from the last 10-step checkpoint
#   -> training_log.json, trained_state.safetensors (delta)

# 3b. training — the MNN-recipe control arm (Adam-150, ~15 min)

python3 Code/gemma_rcds_train.py --model "$MODEL" --steps 150 --lr 3e-5 \

```text
--optim adam --bench-at 75 150 --bench-n 8 --out "$GEMMA_WORK/rcda5_adam"
```text

# 4. the evaluator: three arms (base arm reuses the probe/cache generations)

python3 Code/gemma_benchmark.py --model "$MODEL" --edited-state none \

```text
--tag base --out "$GEMMA_WORK/rcda5_final/part5" --reuse-gens
```text
python3 Code/gemma_benchmark.py --model "$MODEL" \

```text
--edited-state "$GEMMA_WORK/rcda5_final/trained_state.safetensors" \
--tag trained --out "$GEMMA_WORK/rcda5_final/part5"
```text
python3 Code/gemma_benchmark.py --model "$MODEL" \

```text
--edited-state "$GEMMA_WORK/rcda5_adam/trained_state.safetensors" \
--tag adam --out "$GEMMA_WORK/rcda5_final/part5"
```text
#   -> part5_eval_{base,trained,adam}.json + part5_rows_*.jsonl

# 5. abilities PPL parity (~3 min)

python3 Research/RCDA/tools/rcda5_ppl.py --model "$MODEL" \

```text
--trained-state "$GEMMA_WORK/rcda5_final/trained_state.safetensors" \
--out "$GEMMA_WORK/ppl_sgd.json"
```text
python3 Research/RCDA/tools/rcda5_ppl.py --model "$MODEL" \

```text
--trained-state "$GEMMA_WORK/rcda5_adam/trained_state.safetensors" \
--out "$GEMMA_WORK/ppl_adam.json"
```text
#   merge -> ppl_check.json (base / trained / adam)

# 6. forensics + figures

python3 Research/RCDA/tools/rcda5_analyze.py --model "$MODEL" \

```text
--trained-state "$GEMMA_WORK/rcda5_final/trained_state.safetensors" \
--probe "$GEMMA_WORK/probe" --out "$GEMMA_WORK/rcda5_final"
```text
python3 Research/RCDA/tools/rcda5_figures.py --work "$GEMMA_WORK" \

```text
--results "$GEMMA_WORK/rcda5_final" --graphic Graphic/
```text

# 7. build the full trained checkpoint (base + the 3 edited tensors)

python3 - <<'EOF'
from safetensors.torch import load_file, save_file
import sys; sys.path.insert(0, "Code"); import gemma_lab as L
base = load_file(f"{L.resolve_lab().parent.parent}/../<base-dir>/model.safetensors")
```

For step 7 see the packaging helper used for the shipped kit: base safetensors
plus `trained_state.safetensors` merged (3 tensors replaced), then
`zip -r kit.zip gemma3-rcds-interventions/ && split -b 95000000 -d kit.zip gemma3-rcds-interventions.zip.`
— the exact procedure that produced [`../Model ZIP/`](../Model%20ZIP/), with
SHA-256 manifests at every step.

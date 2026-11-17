# RCDA RUNBOOK — RCDA-6, the expanded trainable set (how to re-run)

Everything runs on CPU (the recorded run: 2 cores / 4 GB RAM; training
~5–8 s/step, total wall clock ≈ 1.5 h with the chunked sessions). Set
`GEMMA_WORK` to a scratch dir; results JSONs land there and are copied into
`Research/RCDA/results/`. `MODEL` is any directory (or HF id) holding the
gemma-3-1b-it weights — `unsloth/gemma-3-1b-it` (public mirror) or
`google/gemma-3-1b-it` (Gemma license acceptance required). The RCDA-5 probe
generations are reused, so run the RCDA-5 probe first (or copy its
`$GEMMA_WORK/probe/` over).

```bash
export GEMMA_WORK="$HOME/gemma_rcda6_work"
export MODEL="unsloth/gemma-3-1b-it"

# 0. environment
pip install -r Code/requirements.txt

# 1. the RCDA-5 collapse probe (frozen; provides the generations + writer band)
python3 ../R\ CDA\ -\ 5/Code/gemma_probe.py --model "$MODEL"
#   expected: writers [10, 11, 12]; pool of 14 looping trajectories

# 2. the phase probe (RCDA-6 stage 1; ~1 min — teacher-forced forwards only)
python3 Code/gemma_probe_phase.py --model "$MODEL" --band 8 9 10 11 12 13 14 15
#   -> $GEMMA_WORK/probe_phase/{probe_phase_report.json, v_early_L*.npy, v_deep_L*.npy}
#   expected: z_early 75-118, z_deep 247-393 on the band; cos(v_e,v_d) 0.48-0.89

# 3. training (150 steps, resumable via atomic checkpoints, ~25 min total)
python3 Code/gemma_rcds6_train.py --model "$MODEL" --steps 150 --lr 3e-3 \
    --optim sgd --bench-at 999
#   if the session is cut: rerun the exact same command — it resumes from the
#   last 10-step checkpoint (ckpt_masters.pt + ckpt_opt.pt are atomic)
#   -> training_log.json, trained_state.safetensors (16-tensor delta),
#      analysis_partial.json (the final TW/cos forensics, resumable per tensor)

# 4. the mini-bench at step 150 (standalone: generation is ~6-10x faster in a
#    fresh process than inside the training allocator)
python3 Research/RCDA/tools/rcda6_minibench.py --model "$MODEL" \
    --work "$GEMMA_WORK/rcda6" --tag step150

# 5. the evaluator (row-level + TW-cache resume; run in chunks if needed)
python3 Code/gemma_benchmark.py --model "$MODEL" \
    --edited-state "$GEMMA_WORK/rcda6/trained_state.safetensors" \
    --tag rcda6_trained --out "$GEMMA_WORK/rcda6_final/part6" \
    --probe "$GEMMA_WORK/probe_phase"
#   -> part6_eval_rcda6_trained.json + part6_rows_*.jsonl + part6_tw_*.json

# 6. abilities PPL parity (~3 min)
python3 Research/RCDA/tools/rcda6_ppl.py --model "$MODEL" \
    --trained-state "$GEMMA_WORK/rcda6/trained_state.safetensors" \
    --out "$GEMMA_WORK/rcda6_final/ppl_rcda6.json"

# 7. digest + vectors + figures
python3 Research/RCDA/tools/rcda6_analyze.py --work "$GEMMA_WORK/rcda6" \
    --probe-phase "$GEMMA_WORK/probe_phase" --out "$GEMMA_WORK/rcda6_final"
python3 Research/RCDA/tools/rcda6_figures.py \
    --results "$GEMMA_WORK/rcda6_final" --graphic Graphic/

# 8. packaging: the delta kit (see Model ZIP/README.md)
#    zip the trained delta + logs + provenance, split -b 95000000, SHA-256
```

Measured expectations for a faithful re-run (see `RCDA6_REPORT.md`):
pen_norm 1.0 → ~0.03; bench ρ_mini 1.00 → 0.875; full benchmark ρ_bad 0.85
(base 0.60), loop_share mean 0.627, KL 0.307, PPL 14.1, TW violation ×16,
hard-12 loop share 0.827 → 0.620 with `bad_g02`/`bad_g11` fully repaired.

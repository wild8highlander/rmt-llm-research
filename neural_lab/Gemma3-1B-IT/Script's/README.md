# 🔧 Script's — the real-Gemma working scripts

<p align="center">
  <img src="https://img.shields.io/badge/scripts-19-00D4FF" alt="scripts badge">
  <img src="https://img.shields.io/badge/rounds-RCDA--5%20%C2%B7%20RCDA--6%20(real%20model)-B026FF" alt="rounds badge">
  <img src="https://img.shields.io/badge/runtime-gemma__lab%20%C2%B7%20raw%20mode%20%C2%B7%20MAX__NEW%20192-38E1C0" alt="runtime badge">
  <a href="../LlamaFactory/scripts/"><img src="https://img.shields.io/badge/successor-LlamaFactory%20pipeline-FFB020" alt="successor badge"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The scripts that actually produced the real-model record in
[`../JSON Results/`](../JSON%20Results/README.md) — kept exactly as they
ran, suffixes and all (`*_5.py` are the RCDA-5 generation, unsuffixed the
RCDA-6 generation). Their successor — the resumable LlamaFactory pipeline
that reproduces the probe byte-exact and trains both arms — lives in
[`../LlamaFactory/scripts/`](../LlamaFactory/README.md).

| Script | What it does |
|--------|--------------|
| [`gemma_lab.py`](gemma_lab.py) | the shared runtime: raw-completion encoder, greedy generation, 4-gram loop labels, collapse point, cycle period + phase labels, writer-tensor plumbing, RSS logging — **the protocol constants live here** (MAX_NEW = 192, BAD_PREF = 48, LOOP_THRESHOLD = 0.5) |
| [`gemma_probe.py`](gemma_probe.py) / [`gemma_probe_5.py`](gemma_probe_5.py) | the behavioral probe (z_diff via diagonal Mahalanobis, Rayleigh spike strength vs the MP edge, 5-layer sliding-window writer selection) |
| [`gemma_probe_phase.py`](gemma_probe_phase.py) | the phase probe: v_early / v_deep per layer from the frozen generations |
| [`gemma_rcds_train.py`](gemma_rcds_train.py) / [`gemma_rcds6_train.py`](gemma_rcds6_train.py) | the formula as a loss on the real model: RCDA-5 (o_proj L10–12) and RCDA-6 (band L8–L15, phase-matched penalty, fp32 masters over bf16 compute, resume-safe checkpoints) |
| [`gemma_rcda5.yaml`](gemma_rcda5.yaml) / [`gemma_rcda6.yaml`](gemma_rcda6.yaml) | the frozen run configs (LR ladder, anchor settings) |
| [`gemma_benchmark.py`](gemma_benchmark.py) / [`gemma_benchmark_5.py`](gemma_benchmark_5.py) | the one evaluator, real-model protocol (greedy 192, hard/mild split, repaired count) |
| [`marchenko_pastur.py`](marchenko_pastur.py) / [`marchenko_pastur_5.py`](marchenko_pastur_5.py) | the RMT runtime (identical math to [`../Code/marchenko_pastur.py`](../Code/marchenko_pastur.py)) |
| [`rcda5_analyze.py`](rcda5_analyze.py) / [`rcda6_analyze.py`](rcda6_analyze.py) | spectral digests: ΔW SVD, u₁ vs v alignment, TW control |
| [`rcda5_ppl.py`](rcda5_ppl.py) / [`rcda6_ppl.py`](rcda6_ppl.py) | abilities PPL parity checks |
| [`rcda5_figures.py`](rcda5_figures.py) / [`rcda6_figures.py`](rcda6_figures.py) | figure renderers feeding [`../Graphics/`](../Graphics/README.md) |
| [`rcda6_minibench.py`](rcda6_minibench.py) | the mini-benchmark harness (8 hardest prompts, greedy 192) |
| [`gemma_lab_5.py`](gemma_lab_5.py) | the RCDA-5 generation of the shared runtime |

## Protocol notes that only exist here

- **Raw completion mode** — no chat template; the on-device LiteRT
  prefill/decode path. In chat mode the defect does not manifest
  (sentinel scan: loop_share ≤ 0.09 at 96 tokens).
- **The 192-token window** — Gemma3-1B-IT is repetition-resistant; the
  loop opens only in long generations (96 → no defect, 192 → hard loops).
- **Phase labels** — `pre / early / deep` per generated position, with the
  cycle period fitted from the token tail; irregular tails count as deep.
- **RAM discipline** — anchor logits cached fp16 on disk (memmap), atomic
  split checkpoints (masters / optimizer state), incremental analysis
  persistence: the rounds ran on a 4 GB CPU box.

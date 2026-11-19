# 📜 Reports MD — the real-Gemma scientific record

<p align="center">
  <img src="https://img.shields.io/badge/documents-7-00D4FF" alt="docs badge">
  <img src="https://img.shields.io/badge/rounds-RCDA--5%20%C2%B7%206%20(real%20model)-B026FF" alt="rounds badge">
  <img src="https://img.shields.io/badge/written-November%202026-38E1C0" alt="date badge">
  <a href="../LlamaFactory/RCDA8_REPORT.md"><img src="https://img.shields.io/badge/continuation-RCDA--8%20report-FFB020" alt="rcda8 badge"></a>
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The paper trail of the real-model rounds on
`unsloth/gemma-3-1b-it` (the ungated mirror of the patient): plans
pre-registered before the runs, reports written from the committed JSONs
of [`../JSON Results/`](../JSON%20Results/README.md), runbooks that
document the protocol adaptations, and the cross-round summary. The
negative results are recorded exactly as measured — that is the point of
this folder.

| Document | What it is |
|----------|------------|
| [`RCDA5_PLAN.md`](RCDA5_PLAN.md) | the pre-registered plan of the transfer test: sentinel selection, probe protocol, the two optimizer arms, the evaluator |
| [`RCDA5_REPORT.md`](RCDA5_REPORT.md) | **the honest negative**: penalty collapses (−99 %) while ρ_bad goes 0.60 → 1.00 / 0.95 — the penalized direction is a correlate, not the cause; the MNN convergence does not transfer |
| [`RCDA6_PLAN.md`](RCDA6_PLAN.md) | the pre-registered band hypothesis (o_proj + down_proj, L8–L15, phase-matched directions) |
| [`RCDA6_REPORT.md`](RCDA6_REPORT.md) | the partial repair: hard-12 0.827 → 0.620, 2/12 repaired — and the first measured **defect migration** (mild-8 0.16 → 0.64), PPL +29 % |
| [`RCDA_RUNBOOK.md`](RCDA_RUNBOOK.md) / [`RCDA_RUNBOOK_v6.md`](RCDA_RUNBOOK_v6.md) | the operating procedures: raw-completion mode, MAX_NEW = 192 (the window where the defect opens), cache management, RAM discipline on the 4 GB box |
| [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) | the cross-round table: four states, one evaluator (base / RCDA-5 SGD / RCDA-5 Adam / RCDA-6 SGD) and the scaling-law reading of the pair |
| [`UNPACK_MODEL.md`](UNPACK_MODEL.md) | archive procedures (mirror of [`../Docs/`](../Docs/README.md)) |
| `requirements.txt` / `SHA256SUMS.txt` | the frozen stack + integrity pins |

## The three-line version

1. **RCDA-5**: the MNN-Pocket cure does **not** transfer to Gemma3 — the
   formula trains, the anchor holds, the loops get *worse*.
2. **RCDA-6**: aiming phase-pure directions at both writer types buys the
   first real repair (2/12) at the price of migration and PPL drift.
3. **RCDA-8** (next, [`../LlamaFactory/RCDA8_REPORT.md`](../LlamaFactory/RCDA8_REPORT.md)):
   the same formula as a band-masked LoRA on LlamaFactory — 3/12 repaired
   at 56× fewer parameters — and a data-trained control arm that wins
   every axis. The migration signature persists in all formula geometries.

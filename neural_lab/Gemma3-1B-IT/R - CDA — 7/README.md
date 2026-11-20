<p align="center">
  <img src="../assets/logos/gemma-rcda-avatar.svg" width="48" alt="R CDA - 7 avatar">
</p>

# <sub>🎯</sub> R CDA — 7 · deep-phase-only penalty + tightened anchor

<p align="center">
  <img src="https://img.shields.io/badge/snapshot-round%207-B026FF" alt="snapshot badge">
  <img src="https://img.shields.io/badge/layer-GPT--2%20stand--in%20validation-FF3B6B" alt="layer badge">
  <img src="https://img.shields.io/badge/pen%20point-deep%20phase%20L8%20only-B026FF" alt="pen badge">
  <img src="https://img.shields.io/badge/pen_norm-−80%25%20(best)-38E1C0" alt="descent badge">
  <img src="https://img.shields.io/badge/anchor-β%3D20%20%C2%B7%20cap%3D0.5-FF3B6B" alt="anchor badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The next leverage point named in the program: **concentrate the penalty where
the loop is read out, and make the anchor twice as strict.** The trainable set
stays RCDA-6's (`o_proj + down_proj` × L6/7/8), but the penalty is enforced at
a *single deep-phase point* — the L8 state, where the decomposition showed the
MLP channel carrying 83% of the write energy — and the abilities anchor is
tightened from β=10/cap=1.0 to **β=20/cap=0.5** (half the tolerated drift).

**Result.** The most efficient descent of the program: **pen_norm −80%**
(0.953 → 0.190) at one penalty point under the hardest anchor. Behaviour lands
between RCDA-5 and RCDA-6 on the strict full protocol (loop_share 0.8776,
waste 84.2, ρ_bad 0.9375) with KL 0.011 and PPL parity; relative TW pass ×6;
the deep edit aligns with v (|cos(u₁, v)| up to 0.86). Reading the three
rounds together ([`fig14`](../Graphics/fig14_leverage_chart.png)): RCDA-7 buys
its descent at the lowest KL per point of enforcement — the lever to keep when
the anchor budget matters, and the configuration the Gemma-scale run should
inherit.

> **Layer note.** Validation layer (GPT-2 stand-in). The real-model rounds
> live in [`../Reports MD/`](../Reports%20MD/README.md) (RCDA-5/6) and
> [`../LlamaFactory/`](../LlamaFactory/README.md) (RCDA-8 — the LlamaFactory
> round that inherited this lever design: deep-phase directions inside the
> band penalty, anchor β=10/cap=1.0 at LoRA-appropriate lr).

## What is inside

| Path | Contents |
|------|----------|
| [`Research/RCDA/RCDA7_REPORT.md`](Research/RCDA/RCDA7_REPORT.md) | the round report |
| [`Research/RCDA/RCDA7_PLAN.md`](Research/RCDA/RCDA7_PLAN.md) | the pre-registered plan |
| [`Research/RCDA/results/`](Research/RCDA/results) | `rcda7_training_log.json` · `rcda7_trained_state.pt` · `rcda7_deltas.npz` · `part7_eval_trained.json` · `ppl_check.json` · `tw_relative.json` |
| [`Research/RCDA/figures/`](Research/RCDA/figures) | the round's figure copies |

Run: `python ../Code/gemma_rcds_train.py --variant rcda7 --model google/gemma-3-1b-it --steps 150 --lr 2e-5 --anchor_scope good --refresh_every 32`

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> · <a href="Research/RCDA/RCDA7_REPORT.md">the round-7 report →</a> · <a href="../LlamaFactory/README.md">the RCDA-8 LlamaFactory round →</a></sub>
</p>

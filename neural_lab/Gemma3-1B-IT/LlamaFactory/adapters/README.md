# 🔌 adapters — the trained LoRA artifacts of the RCDA-8 round

<p align="center">
  <img src="https://img.shields.io/badge/base-unsloth%2Fgemma--3--1b--it%20%40%205b1141-00D4FF" alt="base badge">
  <img src="https://img.shields.io/badge/format-peft%20LoRA%20(safetensors)-38E1C0" alt="format badge">
  <img src="https://img.shields.io/badge/license-Gemma-yellow" alt="license badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-RCDA--8%20round-FFB020" alt="parent"></a>
</p>

The two trained arms of the round, shipped the modern way — a few MB of
adapters instead of multi-GB model dumps or split archives. Apply to any
pristine copy of the base checkpoint:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "adapters/rcda8_formula")
# or
model = PeftModel.from_pretrained(base_model, "adapters/sft_data")
```

| Adapter | Arm | Rank / targets | Trainable | Measured outcome |
|---------|-----|----------------|-----------|------------------|
| [`rcda8_formula/`](rcda8_formula) | **the formula** — `pen = (vᵀh/‖v‖)²` phase-matched over the band, trained through LlamaFactory's `CustomSeq2SeqTrainer` subclass | r=16, α=32, o_proj+down_proj, band-masked **L8–L15** (adapters outside the band are frozen at zero) | 1 310 720 (0.13 %) | pen **−97.9 %** · mini-bench 0.921 → **0.529** · hard-12 0.827 → 0.632 · **3/12 fully repaired** · KL 0.51 |
| [`sft_data/`](sft_data) | **the data** — stock `llamafactory-cli train`, pt stage, 38 anti-loop examples (20 curated + 18 self-distilled) | r=8, α=16, all linear projections | ≈ 6.9 M (0.69 %) | ρ_bad 0.60 → **0.40** · hard-12 0.827 → 0.590 · **5/12 fully repaired** · mild-8 0.097 · KL 0.12 · PPL 10.9 → **8.6** |

Both adapters were trained and evaluated on the same frozen protocol
(`../results/`); the adapters outside the RCDA-8 band ship as zero-B
no-ops, so the effective edit is exactly the band. The round's full
record: [`../RCDA8_REPORT.md`](../RCDA8_REPORT.md).

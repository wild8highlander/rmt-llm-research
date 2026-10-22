<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 📊 Laboratory Results

<p align="center">
  <em>Generated artifacts of the multi-language research pipeline</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/output-machine--generated-f472b6" alt="artifacts">
  <img src="https://img.shields.io/badge/auto--generated-8ca3c7" alt="status">
</p>

Everything here is **machine-generated** by the research pipelines
(`research_3d.py` and its cross-language twins) — do not edit by hand.
Each run writes a timestamped bundle `YYYYMMDD_HHMMSS_<experiment-id>`
into the matching subfolder, so runs never overwrite each other and any
figure in a paper can be traced back to the exact experiment that
produced it.

| Subfolder | Content |
|---|---|
| `charts/` | 2D chart bundles — PNG / SVG / PDF triplets per figure |
| `charts_3d/` | 3D scenes — MP density surfaces, BBP landscapes, NHSE collapse |
| `models/` | trained weights (`tiny_gpt_trained.npz`), BPE vocab (`tiny_gpt_bpe.json`), `training_history.json` |
| `reports/` | per-run reports — JSON · CSV · YAML · Markdown · HTML · TeX |

> Files ending in `.pdf.txt`, `.xlsx.txt`, … are textual stubs produced
> when an optional export backend was unavailable in the run environment;
> the canonical machine-readable formats are `.json`, `.csv`, `.yaml`.

## Regenerate

```bash
cd laboratory/python/lab_en
python main.py    # menu item "Research 3D"
```

Repository-level mirrors: [`results/`](../../results/README.md) and the
[chart gallery](../../assets/charts/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a>
  </sub>
</p>

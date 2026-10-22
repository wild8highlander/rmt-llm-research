<p align="center">
  <img src="../../../assets/logos/lab_reports.banner.svg" width="760" alt="laboratory reports — animated logo: stacked documents with sliding pages">
</p>

# 📄 laboratory/results/reports — Experiment Reports

<p align="center">
  <em>The written record of laboratory runs: yaml data + md analysis</em><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-fb923c" alt="laboratory"></a>
  <img src="https://img.shields.io/badge/181%20files-22d3ee" alt="181 files">
</p>

The report archive of the cross-language laboratory: for every
meaningful run, a machine-readable YAML file (parameters, seeds, raw
metrics) and, where a run warranted narration, a Markdown analysis
alongside it. Together with [`../charts/`](../charts/README.md) and
[`../charts_3d/`](../charts_3d/README.md) these reports are the full
audit trail of what the laboratories did, in what configuration, and
what came out.

## Conventions

- Naming mirrors the chart folders: `YYYYMMDD_HHMMSS_<EXPERIMENT>` with
  matching `.yaml` / `.md` extensions — a report and its chart set
  share the timestamp prefix.
- YAML carries the primary data (it is what aggregators parse); the
  Markdown files are narrative views and may reference the charts by
  their shared prefix.
- The archive is append-only: corrections arrive as new runs with new
  timestamps, not as edits of historical reports.

## Where the science went next

The lessons from these early laboratory runs — especially about honest
measurement and the failure modes of self-reported metrics — fed
directly into the predictive science core:
[`neural_lab/science/`](../../../neural_lab/science/README.md) and its
pre-registered H1 protocol are the disciplined continuation of what
this archive documents empirically.

For the current-generation experiment results (H1, S/M/L cells), see
[`neural_lab/science/results/`](../../../neural_lab/science/results/README.md)
— that is where live research happens now.

---

<p align="center">
  <img src="../../../assets/logos/lab_reports.avatar.svg" width="56" alt="laboratory reports animated avatar"><br>
  <sub>
    <a href="../README.md">← laboratory results</a> · <a href="../../README.md">← laboratory</a> · <a href="../../../neural_lab/science/results/README.md">science results →</a>
  </sub>
</p>

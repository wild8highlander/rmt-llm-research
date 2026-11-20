# Docs/ — how to unpack the model and re-run the experiment

| File | Contents |
|------|----------|
| [`UNPACK_MODEL.md`](../../Docs/UNPACK_MODEL.md) | reassembling the split archive in `Model ZIP/` into a working model directory (checksums verified) |
| [`RCDA_RUNBOOK.md`](RCDA_RUNBOOK.md) | the full re-run runbook: probe → train → eval → figures, with the exact commands and expected outputs |

The model itself ships in [`../Model ZIP/`](../../Model%20ZIP) as 17 checksummed
volumes (`gemma3-rcds-interventions.zip.01 … .17`); after reassembly the kit is
fully offline-ready — a plain transformers checkpoint you can point any
script at.

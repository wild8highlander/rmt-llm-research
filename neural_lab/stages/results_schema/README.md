# 🧾 results_schema — the artifact contracts

JSON Schema (draft 2020-12) definitions for every artifact the stage
pipeline writes. Validation happens at write time via
`stages.common.schemas` — a malformed report fails the run at the
boundary, not three stages later.

| Path | Contract for |
|------|--------------|
| [`v1/report.schema.json`](v1/report.schema.json) | `report.json` — every stage's run report |
| [`v1/fit.schema.json`](v1/fit.schema.json) | `fit.json` — spectral fits |
| [`v1/record.schema.json`](v1/record.schema.json) | `record.json` — per-run records |
| [`v1/scaling.schema.json`](v1/scaling.schema.json) | `scaling.json` — scaling studies |

Validate a results tree against the contracts:

```bash
make validate-results    # from the repository root
```

The full pipeline story: [`../README.md`](../README.md).

# 🧭 laboratory (nested) — Go lab internals

The Go laboratory keeps its package sources under a nested `laboratory/`
directory — a known quirk of the layout, kept for import-path stability
(`go.mod` module path ends in `/laboratory`).

| Path | Role |
|------|------|
| [`results/`](results/) | the Go lab's committed result reports |
| [`README.md`](README.md) | this file |

Build, test and compare against the reference implementation from the
parent directory:

```bash
cd laboratory/go/lab_en && go test ./...
```

See the parent [`../README.md`](../README.md) for the full Go laboratory
guide (benchmarks, cross-implementation checks, the Russian twin lab).

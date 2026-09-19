# Go Laboratory

Go port of the lab pipeline (`lab_en/` English, `lab_ru/` Russian).
Dependency-free implementation (stdlib only) of the 3D research engine,
TinyGPT inference, and the scenario runner.

```bash
cd laboratory/go/lab_en
go test ./...            # unit tests (research_3d_test.go)
go run .                 # interactive CLI
go vet ./...             # static analysis
```

Run outputs use the shared schema (`laboratory/shared/schema.json`) so they
can be validated and compared with the Python reference results.

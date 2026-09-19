# Shared Laboratory Contracts

Language-agnostic contracts used by all 8 laboratory implementations:

| File | Content |
|---|---|
| `schema.json` | the JSON schema every experiment report must satisfy (`experiment`, `results`, `timestamp`, ...) — validated in the CI cross-implementation job |
| `scenarios.json` | the adversarial verification scenarios (prompt sets, expected spectral behaviour) shared verbatim across languages |
| `model_registry.json` | registry of downloadable reference models with checksums |

Keeping these files as the single source of truth is what makes
cross-implementation consistency checks meaningful: each port reads the same
scenarios and writes reports under the same schema, so results from Python,
Julia, Rust, Go, C++, Java, R and the JavaScript web app
(`laboratory/webapp/`) can be compared 1:1.

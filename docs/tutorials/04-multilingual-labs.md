# Tutorial 4: Multilingual Labs

This tutorial compares the **same algorithm** implemented across 8 programming
languages. You'll see how RMT math translates between Python, Julia, Java,
Rust, Go, C++, R, and TypeScript.

The full runnable version is at
[`notebooks/04_multilingual_labs.ipynb`](https://github.com/wild8highlander/rmt-llm-research/blob/main/notebooks/04_multilingual_labs.ipynb).

---

## The 8-language contract

Every lab in `laboratory/<lang>/lab_en/` implements the **same interactive menu**
and emits the **same JSON schema**. The contract is enforced by:

1. **Shared schema**: [`laboratory/shared/schema.json`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/shared/schema.json)
2. **Shared scenarios**: [`laboratory/shared/scenarios.json`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/shared/scenarios.json)
3. **Cross-lang test**: [`src/rmt_llm/tests/test_cross_lang_json.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/tests/test_cross_lang_json.py)

---

## 1. Run the same scenario in every language

### Python

```bash
cd laboratory/python/lab_en
python main.py    # choose menu item 1 (SCEN-LIE-01)
```

### Julia

```bash
cd laboratory/julia/lab_en
julia --project=. -e 'using Main; main()'
# choose menu item 1
```

### Java

```bash
cd laboratory/java/lab_en
./gradlew run
# choose menu item 1
```

### Rust

```bash
cd laboratory/rust/lab_en
cargo run
# choose menu item 1
```

### Go

```bash
cd laboratory/go/lab_en
go run .
# choose menu item 1
```

### C++

```bash
cd laboratory/cpp/lab_en
make && ./lab
# choose menu item 1
```

### R

```bash
cd laboratory/r/lab_en
Rscript main.R
# choose menu item 1
```

### Web app (TypeScript + React)

```bash
cd laboratory/webapp
npm install && npm start
# open http://localhost:5173
```

---

## 2. Compare the JSON outputs

After running scenario `SCEN-LIE-01` in each language, you'll have 8 JSON
files in `laboratory/<lang>/lab_en/results/`. Compare them:

```python
import json
from pathlib import Path

labs = ["python", "julia", "java", "rust", "go", "cpp", "r", "webapp"]
outputs = {}

for lang in labs:
    p = Path(f"laboratory/{lang}/lab_en/results/scen_lie_01.json")
    if p.exists():
        outputs[lang] = json.loads(p.read_text())

# Compare a single field across all languages
key = "scenario_result.collapse_detected"
print(f"Field: {key}")
for lang, data in outputs.items():
    value = data
    for k in key.split("."):
        value = value.get(k, None) if isinstance(value, dict) else None
    print(f"  {lang:8s}: {value}")
```

All 8 should produce the **same** boolean value (within float tolerance for
numeric fields).

---

## 3. Cross-language consistency test

```bash
pytest src/rmt_llm/tests/test_cross_lang_json.py -v
```

This test:

1. Loads all 16 JSON output files (8 languages × 2 locales)
2. Validates each against `laboratory/shared/schema.json`
3. Compares equivalent fields across languages
4. Asserts float fields agree to within `1e-6` relative tolerance

---

## 4. Rosetta Stone: the same function in 8 languages

Here's the Marchenko-Pastur density function in all 8 languages. Notice that
the **math is identical** — only the syntax differs.

### Python

```python
def mp_density(lam, q, sigma=1.0):
    lam_minus = sigma**2 * (1 - q**0.5)**2
    lam_plus  = sigma**2 * (1 + q**0.5)**2
    if lam < lam_minus or lam > lam_plus:
        return 0.0
    return (q / (2 * np.pi * sigma**2 * lam)) * \
           np.sqrt((lam_plus - lam) * (lam - lam_minus))
```

### Julia

```julia
function mp_density(lam::Real, q::Real, sigma::Real=1.0)
    lam_minus = sigma^2 * (1 - sqrt(q))^2
    lam_plus  = sigma^2 * (1 + sqrt(q))^2
    if lam < lam_minus || lam > lam_plus
        return 0.0
    end
    return (q / (2 * pi * sigma^2 * lam)) *
           sqrt((lam_plus - lam) * (lam - lam_minus))
end
```

### Rust

```rust
pub fn mp_density(lam: f64, q: f64, sigma: f64) -> f64 {
    let lam_minus = sigma.powi(2) * (1.0 - q.sqrt()).powi(2);
    let lam_plus  = sigma.powi(2) * (1.0 + q.sqrt()).powi(2);
    if lam < lam_minus || lam > lam_plus {
        return 0.0;
    }
    (q / (2.0 * std::f64::consts::PI * sigma.powi(2) * lam))
        * ((lam_plus - lam) * (lam - lam_minus)).sqrt()
}
```

### Go

```go
func MPDensity(lam, q, sigma float64) float64 {
    lamMinus := math.Pow(sigma, 2) * math.Pow(1-math.Sqrt(q), 2)
    lamPlus  := math.Pow(sigma, 2) * math.Pow(1+math.Sqrt(q), 2)
    if lam < lamMinus || lam > lamPlus {
        return 0.0
    }
    return (q / (2 * math.Pi * math.Pow(sigma, 2) * lam)) *
           math.Sqrt((lamPlus-lam)*(lam-lamMinus))
}
```

### C++

```cpp
double mp_density(double lam, double q, double sigma=1.0) {
    double lam_minus = sigma*sigma * std::pow(1 - std::sqrt(q), 2);
    double lam_plus  = sigma*sigma * std::pow(1 + std::sqrt(q), 2);
    if (lam < lam_minus || lam > lam_plus) return 0.0;
    return (q / (2 * M_PI * sigma*sigma * lam)) *
           std::sqrt((lam_plus - lam) * (lam - lam_minus));
}
```

### Java

```java
public static double mpDensity(double lam, double q, double sigma) {
    double lamMinus = sigma*sigma * Math.pow(1 - Math.sqrt(q), 2);
    double lamPlus  = sigma*sigma * Math.pow(1 + Math.sqrt(q), 2);
    if (lam < lamMinus || lam > lamPlus) return 0.0;
    return (q / (2 * Math.PI * sigma*sigma * lam)) *
           Math.sqrt((lamPlus - lam) * (lam - lamMinus));
}
```

### R

```r
mp_density <- function(lam, q, sigma=1.0) {
    lam_minus <- sigma^2 * (1 - sqrt(q))^2
    lam_plus  <- sigma^2 * (1 + sqrt(q))^2
    if (lam < lam_minus || lam > lam_plus) return(0.0)
    (q / (2 * pi * sigma^2 * lam)) *
      sqrt((lam_plus - lam) * (lam - lam_minus))
}
```

### TypeScript

```typescript
export function mpDensity(lam: number, q: number, sigma: number = 1.0): number {
    const lamMinus = sigma**2 * (1 - Math.sqrt(q))**2;
    const lamPlus  = sigma**2 * (1 + Math.sqrt(q))**2;
    if (lam < lamMinus || lam > lamPlus) return 0.0;
    return (q / (2 * Math.PI * sigma**2 * lam)) *
           Math.sqrt((lamPlus - lam) * (lam - lamMinus));
}
```

---

## 5. Performance comparison

Each implementation runs `mp_density` 1M times. Lower is better.

| Language | Time (ms) | vs. Python | Notes |
|----------|-----------|------------|-------|
| Python   | 2,400     | 1.0×       | Baseline |
| Julia    | 180       | 13×        | JIT-compiled |
| Rust     | 95        | 25×        | Compiled, no GC |
| C++      | 100       | 24×        | Compiled |
| Go       | 130       | 18×        | Compiled, GC |
| Java     | 145       | 16×        | JIT after warmup |
| R        | 2,800     | 0.85×      | Slower than Python here |
| TypeScript | 4,500   | 0.53×      | Node.js overhead |

**Note:** these numbers are for the pure-language implementations. NumPy's
vectorized version (called once on a 1M array) runs in ~12 ms — 200× faster
than the Python scalar loop above.

---

## 6. Adding a new language

See [CONTRIBUTING.md → Adding a new language port](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md#adding-a-new-language-port).

The basic recipe:

1. Pick a language with linear algebra support
2. Mirror `laboratory/python/lab_en/` — same menu structure, same scenarios
3. Emit `laboratory/shared/schema.json`-compliant JSON
4. Add tests under `laboratory/<lang>/lab_en/tests/`
5. Add the language to `.github/workflows/ci.yml` matrix
6. Add a CODEOWNERS entry
7. Open a PR

---

## 7. EN ↔ RU synchronization

Every lab has `_en` and `_ru` siblings. The Russian versions are kept in
lock-step by `scripts/sync_ru_trainer.py` — do not hand-edit one without
syncing the other.

```bash
python scripts/sync_ru_trainer.py
```

This script:

1. Reads `laboratory/python/lab_en/tiny_gpt_trainer.py`
2. Translates docstrings, comments, and print statements to Russian
3. Writes `laboratory/python/lab_ru/tiny_gpt_trainer.py`
4. Verifies both files have identical function signatures and class hierarchies

---

## What's next?

- [Architecture: Design Principles](../architecture/principles.md)
- [API Reference](../api/index.md)
- [Contributing](https://github.com/wild8highlander/rmt-llm-research/blob/main/CONTRIBUTING.md)

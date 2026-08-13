# Installation

This page covers installing `rmt-llm-research` from scratch on Linux, macOS,
and Windows. If you just want the 60-second version, see
[Quick Start](quick-start.md).

---

## Prerequisites

| Tool | Version | Required for | Notes |
|------|---------|--------------|-------|
| Python | 3.10, 3.11, or 3.12 | All Python features | PyPy 3.10 also works |
| Julia  | 1.9, 1.10, or 1.11   | Julia lab + Julia viz | Optional |
| Java   | 21+                  | JavaFX viz + Java lab | Optional, JDK or JRE |
| Node.js| 18+                  | React webapp          | Optional |
| Rust   | 1.70+                | Rust lab              | Optional |
| Go     | 1.21+                | Go lab                | Optional |
| C++    | C++17 compiler       | C++ lab               | Optional |

You only need **Python 3.10+** to use the core package and TinyGPT.
Other languages are optional and only needed for the cross-language
laboratory ports.

---

## Option 1: Pip install (recommended for users)

```bash
# Clone the repo
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell

# Install in editable mode with dev deps
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs:

- `rmt_llm` package (the theoretical core)
- `pytest`, `ruff`, `mypy`, `pre-commit` (dev tooling)
- `matplotlib`, `jupyter` (notebook support)

### Verify

```bash
pytest src/rmt_llm/tests/ -v        # 70+ tests should pass
python -c "from rmt_llm.marchenko_pastur import mp_density; print(mp_density(1.0, 1.0))"
```

---

## Option 2: Install only what you need

The `pyproject.toml` defines optional dependency groups so you can install
exactly what you need:

```bash
pip install -e "."                   # core only (just numpy)
pip install -e ".[test]"             # + pytest, pytest-cov, pytest-xdist
pip install -e ".[dev]"              # + ruff, mypy, pre-commit, jupyter
pip install -e ".[lab]"              # + matplotlib, plotly, pandas, reportlab
pip install -e ".[docs]"             # + mkdocs-material, mkdocstrings
pip install -e ".[bench]"            # + hypothesis, pytest-benchmark
pip install -e ".[all]"              # everything
```

---

## Option 3: Conda / Mamba

```bash
conda create -n rmt-llm python=3.12
conda activate rmt-llm
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research
pip install -e ".[dev]"
```

---

## Option 4: Docker

The official image ships with Python 3.12 + Julia 1.10 + Java 21 pre-installed:

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/wild8highlander/rmt-llm-research:latest

# Run interactively
docker run --rm -it ghcr.io/wild8highlander/rmt-llm-research:latest bash

# Or build locally
make docker
make docker-run
```

See [Docker](../community/contributing.md#docker) for `docker-compose.yml`
services (`lab`, `tests`, `tinygpt-train`, `docs`, `jupyter`, `shell`).

---

## Option 5: Julia verification package

```bash
cd julia/RMTLLMVerify
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

---

## Option 6: Java visualization

```bash
cd java/rmt-llm-viz
./gradlew run          # Launch JavaFX GUI
./gradlew verify       # Headless verification (14 checks)
```

Requires JDK 21+ (Eclipse Temurin recommended).

---

## Troubleshooting

### `numpy` fails to install

Make sure you have a C compiler and Python headers:

```bash
# Debian/Ubuntu
sudo apt install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Use the official Python installer from python.org — wheels include binaries
```

### `pip install` is slow

Use a mirror or set `PIP_DEFAULT_TIMEOUT=120`:

```bash
export PIP_DEFAULT_TIMEOUT=120
pip install -e ".[dev]" --prefer-binary
```

### Julia `Pkg.instantiate()` fails

Make sure you're using Julia 1.10+ (LTS). The `Project.toml` is pinned to
compatible versions; if it's still broken, try:

```bash
julia --project=. -e 'using Pkg; Pkg.update(); Pkg.resolve()'
```

### `pre-commit install` fails

You may need to install pre-commit separately:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

### Tests fail on Windows

Most tests pass on Windows, but a few filesystem-sensitive tests may skip.
Run with:

```bash
pytest -v -p no:cacheprovider
```

---

## Next steps

- [Quick Start](quick-start.md) — get running in 60 seconds
- [First Run](first-run.md) — train your first TinyGPT
- [Tutorial: Quickstart Notebook](../tutorials/01-quickstart.md) — full walkthrough in Jupyter

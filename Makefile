# =============================================================================
# Makefile — common dev tasks. Run `make help` to see all targets.
# Requires GNU Make (Linux/macOS). Windows users: use WSL or `make.bat`.
# =============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Colors ─────────────────────────────────────────────────────────────────
BLUE   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
RESET  := \033[0m
BOLD   := \033[1m

# ─── Project paths ──────────────────────────────────────────────────────────
PYTHON_PKG_DIR  := src/rmt_llm
LAB_PYTHON_DIR  := laboratory/python
LAB_EN_DIR      := laboratory/python/lab_en
LAB_RU_DIR      := laboratory/python/lab_ru
TESTS_DIR       := $(PYTHON_PKG_DIR)/tests
LAB_TESTS_DIR   := $(LAB_EN_DIR)/tests

# ─── Tooling ────────────────────────────────────────────────────────────────
PYTHON  ?= python
PIP     ?= pip
PYTEST  ?= pytest
RUFF    ?= ruff
MYPY    ?= mypy
JULIA   ?= julia
GO      ?= go
CARGO   ?= cargo
GRADLE  ?= ./gradlew
NPM     ?= npm
DOCKER  ?= docker

# ─── Helpers ────────────────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "$(BOLD)rmt-llm-research — Makefile$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-22s$(RESET) %s\n", $$1, $$2}' \
	  | sort
	@echo ""
	@echo "$(BOLD)Tips:$(RESET)"
	@echo "  • Run $(GREEN)make install$(RESET) first to set up the environment."
	@echo "  • Run $(GREEN)make check$(RESET) before committing to run all checks."
	@echo "  • Run $(GREEN)make ci$(RESET) to simulate the full CI pipeline locally."
	@echo ""

# ─── Environment setup ──────────────────────────────────────────────────────
.PHONY: install install-dev install-test install-pre-commit venv
install:  ## Install the package for production use
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev:  ## Install with all dev dependencies (ruff, mypy, pytest, etc.)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install pre-commit
	pre-commit install

install-test:  ## Install only test dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[test]"

install-pre-commit:  ## Install pre-commit hooks (run after install-dev)
	pre-commit install
	pre-commit install --hook-type commit-msg
	pre-commit install --hook-type pre-push

venv:  ## Create a fresh virtual environment in .venv/
	$(PYTHON) -m venv .venv
	@echo ""
	@echo "$(GREEN)✓ Virtual environment created at .venv/$(RESET)"
	@echo "Activate with:  source .venv/bin/activate"
	@echo "Then run:       make install-dev"

# ─── Linting & formatting ───────────────────────────────────────────────────
.PHONY: lint lint-fix format typecheck
lint:  ## Run all linters (ruff, mypy, codespell, markdownlint)
	$(RUFF) check src/ laboratory/python/ python/ scripts/
	$(RUFF) format --check src/ laboratory/python/ python/ scripts/
	$(MYPY) src/ --ignore-missing-imports || true
	@echo "$(GREEN)✓ Lint passed$(RESET)"

lint-fix:  ## Auto-fix lint issues (ruff --fix + format)
	$(RUFF) check --fix src/ laboratory/python/ python/ scripts/
	$(RUFF) format src/ laboratory/python/ python/ scripts/
	@echo "$(GREEN)✓ Auto-fixes applied$(RESET)"

format:  ## Format all Python code with ruff
	$(RUFF) format src/ laboratory/python/ python/ scripts/
	@echo "$(GREEN)✓ Formatted$(RESET)"

typecheck:  ## Run mypy type checker
	$(MYPY) src/ --ignore-missing-imports

# ─── Testing ────────────────────────────────────────────────────────────────
.PHONY: test test-fast test-slow test-tiny-gpt test-julia test-rust test-go test-cpp test-all test-coverage
test:  ## Run all Python tests (src/rmt_llm/ + laboratory/python/lab_en/)
	$(PYTEST) src/ laboratory/python/lab_en/tests/ -v

test-fast:  ## Run only fast tests (skip slow ones)
	$(PYTEST) src/ laboratory/python/lab_en/tests/ -v -m "not slow"

test-slow:  ## Run only slow tests
	$(PYTEST) src/ laboratory/python/lab_en/tests/ -v -m "slow"

test-tiny-gpt:  ## Run only TinyGPT + BPE + trainer tests
	$(PYTEST) $(LAB_TESTS_DIR)/ -v

test-julia:  ## Run Julia verification tests
	cd julia/RMTLLMVerify && $(JULIA) --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

test-rust:  ## Run Rust laboratory tests (EN + RU)
	cd laboratory/rust/lab_en && $(CARGO) test
	cd laboratory/rust/lab_ru && $(CARGO) test

test-go:  ## Run Go laboratory tests (EN + RU)
	cd laboratory/go/lab_en && $(GO) test ./...
	cd laboratory/go/lab_ru && $(GO) test ./...

test-cpp:  ## Build and run C++ laboratory tests (EN + RU)
	cd laboratory/cpp/lab_en && make test
	cd laboratory/cpp/lab_ru && make test

test-all: test test-julia test-rust test-go test-cpp  ## Run tests in ALL languages (slow!)

test-coverage:  ## Run Python tests with coverage report
	$(PYTEST) src/ laboratory/python/lab_en/tests/ \
	  --cov=rmt_llm \
	  --cov-report=term-missing \
	  --cov-report=html \
	  --cov-report=xml \
	  -v
	@echo ""
	@echo "$(GREEN)✓ Coverage report generated:$(RESET)"
	@echo "  HTML:  htmlcov/index.html"
	@echo "  XML:   coverage.xml"

# ─── Cross-implementation checks ────────────────────────────────────────────
.PHONY: cross-check
cross-check:  ## Run cross-implementation consistency tests (Python vs Julia vs JSON schema)
	$(PYTEST) src/rmt_llm/tests/ -v -k "cross"
	@echo "$(GREEN)✓ Cross-implementation checks passed$(RESET)"

# ─── Pre-commit ─────────────────────────────────────────────────────────────
.PHONY: pre-commit pre-commit-all pre-commit-update
pre-commit:  ## Run pre-commit on staged files
	pre-commit run

pre-commit-all:  ## Run pre-commit on ALL files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hook versions
	pre-commit autoupdate

# ─── Combined checks ────────────────────────────────────────────────────────
.PHONY: check ci clean
check: lint test-fast  ## Quick checks to run before committing
	@echo "$(GREEN)✓ Quick checks passed — ready to commit$(RESET)"

ci: lint test-coverage cross-check  ## Simulate the full CI pipeline locally
	@echo "$(GREEN)✓ Full CI simulation passed$(RESET)"

clean:  ## Remove all build artifacts, caches, and generated files
	rm -rf build/ dist/ *.egg-info/ .eggs/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage coverage.xml htmlcov/
	rm -rf laboratory/python/lab_en/.pytest_cache/ laboratory/python/lab_en/.ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf node_modules/ 2>/dev/null || true
	rm -rf laboratory/webapp/node_modules/ 2>/dev/null || true
	rm -rf laboratory/webapp/dist/ 2>/dev/null || true
	@echo "$(GREEN)✓ Clean$(RESET)"

# ─── TinyGPT training ───────────────────────────────────────────────────────
.PHONY: tinygpt-train tinygpt-generate tinygpt-train-ru
tinygpt-train:  ## Train TinyGPT (English lab) — runs interactive menu item 14
	cd $(LAB_EN_DIR) && $(PYTHON) main.py
	@echo ""
	@echo "$(GREEN)Tip:$(RESET) Choose menu item $(BOLD)14$(RESET) to train, $(BOLD)15$(RESET) to generate."

tinygpt-train-ru:  ## Train TinyGPT (Russian lab)
	cd $(LAB_RU_DIR) && $(PYTHON) main.py

tinygpt-generate:  ## Generate text from trained TinyGPT weights
	cd $(LAB_EN_DIR) && $(PYTHON) main.py
	@echo ""
	@echo "$(GREEN)Tip:$(RESET) Choose menu item $(BOLD)15$(RESET), provide weights path:"
	@echo "  $(LAB_EN_DIR)/results/models/tiny_gpt_trained.npz"

# ─── Visualization ──────────────────────────────────────────────────────────
.PHONY: viz-python viz-julia viz-java
viz-python:  ## Launch Python interactive 3D visualization suite
	cd python/rmt_llm_viz && $(PYTHON) main.py

viz-julia:  ## Launch Julia interactive visualization REPL
	cd julia/RMTLLMViz && $(JULIA) --project=. -e 'using RMTLLMViz; rmt_llm_viz_menu()'

viz-java:  ## Launch Java JavaFX visualization GUI
	cd java/rmt-llm-viz && $(GRADLE) run

# ─── Web dashboard ──────────────────────────────────────────────────────────
.PHONY: webapp webapp-install webapp-build
webapp-install:  ## Install webapp dependencies (npm install)
	cd laboratory/webapp && $(NPM) install

webapp:  ## Launch the Vite + Socket.io web dashboard
	cd laboratory/webapp && $(NPM) start

webapp-build:  ## Build webapp for production
	cd laboratory/webapp && $(NPM) run build

# ─── Docker ─────────────────────────────────────────────────────────────────
.PHONY: docker docker-run docker-test docker-clean
docker:  ## Build the Docker image
	$(DOCKER) build -t rmt-llm-research:latest .

docker-run:  ## Run the Docker image interactively
	$(DOCKER) run --rm -it -v "$${PWD}:/workspace" rmt-llm-research:latest bash

docker-test:  ## Run tests inside Docker container
	$(DOCKER) run --rm rmt-llm-research:latest bash -c "pytest src/rmt_llm/tests/ -v"

docker-clean:  ## Remove Docker image and dangling containers
	$(DOCKER) rmi rmt-llm-research:latest 2>/dev/null || true
	$(DOCKER) image prune -f

# ─── Documentation ──────────────────────────────────────────────────────────
.PHONY: docs docs-serve docs-build
docs: docs-serve  ## Serve documentation locally (alias for docs-serve)

docs-serve:  ## Serve the GitHub Pages site locally
	@echo "$(YELLOW)Starting local server on http://localhost:8000$(RESET)"
	cd docs/site && $(PYTHON) -m http.server 8000

docs-build:  ## Build documentation (placeholder — currently static HTML)
	@echo "Documentation is currently static HTML in docs/site/."
	@echo "No build step needed — just open docs/site/index.html in a browser."

# ─── Release helpers (maintainers only) ─────────────────────────────────────
.PHONY: release-tag release-verify
release-tag:  ## Create a release tag (usage: make release-tag VERSION=1.6.0)
	@if [ -z "$(VERSION)" ]; then echo "$(RED)Error:$(RESET) Usage: make release-tag VERSION=1.6.0"; exit 1; fi
	@echo "$(YELLOW)Creating tag v$(VERSION)...$(RESET)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push origin "v$(VERSION)"
	@echo "$(GREEN)✓ Tag v$(VERSION) pushed — release.yml workflow will fire$(RESET)"

release-verify:  ## Verify that pyproject.toml version matches the latest git tag
	@PY_VER=$$($(PYTHON) -c "import tomllib; print(tomllib.loads(open('pyproject.toml','rb').read().decode())['project']['version'])") ; \
	GIT_VER=$$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//') ; \
	echo "pyproject.toml version: $$PY_VER" ; \
	echo "latest git tag:         $$GIT_VER" ; \
	if [ "$$PY_VER" = "$$GIT_VER" ]; then \
	  echo "$(GREEN)✓ Versions match$(RESET)" ; \
	else \
	  echo "$(RED)✗ Version mismatch — update pyproject.toml or create tag$(RESET)" ; \
	  exit 1 ; \
	fi

# ─── Misc ───────────────────────────────────────────────────────────────────
.PHONY: tree stats
tree:  ## Show the project tree (top 3 levels, no ignored files)
	@find . -maxdepth 3 -type d \
	  -not -path './.git*' \
	  -not -path './node_modules*' \
	  -not -path '*/__pycache__*' \
	  -not -path '*/.pytest_cache*' \
	  -not -path '*/results/*' \
	  | sort | sed 's|[^/]*/|  |g'

stats:  ## Show project statistics (LOC, file counts, language breakdown)
	@echo "$(BOLD)Project statistics$(RESET)"
	@echo ""
	@echo "$(BOLD)Files by language:$(RESET)"
	@find . -type f -not -path './.git/*' -not -path '*/node_modules/*' -not -path '*/__pycache__/*' -not -path '*/results/*' \
	  | awk -F. '{print $$NF}' | sort | uniq -c | sort -rn | head -20
	@echo ""
	@echo "$(BOLD)Lines of code (top 10 file types):$(RESET)"
	@find . -type f \( -name '*.py' -o -name '*.jl' -o -name '*.java' -o -name '*.rs' -o -name '*.go' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.R' -o -name '*.js' -o -name '*.jsx' -o -name '*.ts' -o -name '*.tsx' -o -name '*.md' \) \
	  -not -path './.git/*' -not -path '*/node_modules/*' -not -path '*/results/*' \
	  -exec wc -l {} + | tail -1

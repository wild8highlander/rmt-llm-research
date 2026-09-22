#syntax=docker/dockerfile:1.7
# =============================================================================
# Dockerfile — reproducible research environment for rmt-llm-research.
# Builds a self-contained image with Python, Julia, Java, and Node.js so that
# all 8 laboratory language ports + 3 visualization suites can run.
# =============================================================================

# ─── Stage 1: Python base ───────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS python-base

LABEL org.opencontainers.image.title="rmt-llm-research"
LABEL org.opencontainers.image.description="Random Matrix Theory meets Large Language Models — spectral verification + TinyGPT trainer + 8-language laboratory"
LABEL org.opencontainers.image.authors="Iskhak Hamzatovich Isaev <aslan08_05@mail.ru>"
LABEL org.opencontainers.image.url="https://github.com/wild8highlander/rmt-llm-research"
LABEL org.opencontainers.image.documentation="https://wild8highlander.github.io/rmt-llm-research"
LABEL org.opencontainers.image.source="https://github.com/wild8highlander/rmt-llm-research"
LABEL org.opencontainers.image.licenses="Proprietary"
LABEL org.opencontainers.image.base.name="docker.io/library/python:3.12-slim-bookworm"

# Prevent Python from writing .pyc files + force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system deps needed by numpy, matplotlib, reportlab, java, julia
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        unzip \
        pkg-config \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libgomp1 \
        fonts-dejavu \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 rmtuser

WORKDIR /workspace

# ─── Stage 2: Install Python package ────────────────────────────────────────
FROM python-base AS python-deps

COPY pyproject.toml README.md LICENSE /workspace/
COPY src/ /workspace/src/

RUN pip install --upgrade pip setuptools wheel \
 && pip install -e ".[dev,lab]"

# ─── Stage 3: Add Julia ─────────────────────────────────────────────────────
FROM python-deps AS julia-deps

# Julia 1.10 LTS — install from official binary tarball
ARG JULIA_VERSION=1.10.4
RUN wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
        -O /tmp/julia.tar.gz \
 && mkdir -p /opt/julia \
 && tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1 \
 && rm /tmp/julia.tar.gz \
 && ln -s /opt/julia/bin/julia /usr/local/bin/julia

ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    JULIA_NUM_THREADS=auto

# Pre-instantiate Julia packages to cache them in the image
COPY julia/ /workspace/julia/
RUN cd /workspace/julia/RMTLLMVerify \
 && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
 && cd /workspace/julia/RMTLLMViz \
 && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ─── Stage 4: Final image ───────────────────────────────────────────────────
FROM julia-deps AS final

# Copy the rest of the repo
COPY --chown=rmtuser:rmtuser . /workspace/

# Make sure scripts are executable
RUN chmod +x /workspace/scripts/*.py 2>/dev/null || true

# Switch to non-root user
USER rmtuser

# Default: run the verification test suite
CMD ["pytest", "src/rmt_llm/tests/", "-v", "--tb=short"]

# Health check — verify the Python import works
HEALTHCHECK --interval=5m --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "from rmt_llm.marchenko_pastur import mp_bounds; print('healthy')" || exit 1

# Expose port for the webapp (when launched via docker run --rm -p 5173:5173 ...)
EXPOSE 5173 8000

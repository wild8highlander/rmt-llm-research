"""
probe_real_gpt2.py — Real GPT-2 activation probe for RMT-LLM laboratory.

Loads GPT-2 small (124M parameters) via HuggingFace ``transformers``
and extracts hidden-state activations from Wikitext prompts. The
covariance spectra of these activations are then compared against
Marchenko-Pastur bounds to detect the BBP transition across layers.

This module is **optional**: it imports ``transformers`` and ``torch``
only when needed. The rest of the RMT-LLM framework is NumPy-only
(ADR-001), but this probe is explicitly opt-in.

Usage::

    from probe_real_gpt2 import GPT2Probe
    probe = GPT2Probe()
    results = probe.analyze_prompts(["The capital of France is", ...])
    probe.report(results)

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Optional dependency: transformers + torch
# ---------------------------------------------------------------------------
_IMPORT_ERROR: str | None = None
try:
    import torch  # type: ignore
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
    _HAS_TF = True
except ImportError as e:  # pragma: no cover
    _HAS_TF = False
    _IMPORT_ERROR = str(e)


# ---------------------------------------------------------------------------
# Local RMT imports (always available)
# ---------------------------------------------------------------------------
def _mp_bounds_general(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP bounds valid for any q > 0 (handles the H > T case).

    For q > 1 there is additionally a point mass at 0, but the
    continuous part of the support is still ``σ²(1 ± √q)²``.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    sq = np.sqrt(q)
    return sigma2 * (1 - sq) ** 2, sigma2 * (1 + sq) ** 2


try:
    from rmt_llm.bbp_transition import bbp_critical_theta, bbp_is_supercritical
    _HAS_RMT = True
except ImportError:
    _HAS_RMT = False

    def bbp_critical_theta(q: float) -> float:
        return np.sqrt(q)

    def bbp_is_supercritical(theta: float, q: float) -> bool:
        return theta > bbp_critical_theta(q)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LayerSpectrum:
    """Spectral analysis of one layer's hidden-state covariance.

    Attributes:
        layer: Layer index (0-based).
        n_tokens: Number of tokens in the prompt.
        hidden_dim: Hidden dimension of the layer.
        lambda_max: Largest covariance eigenvalue.
        lambda_min: Smallest positive covariance eigenvalue.
        lambda_mean: Mean eigenvalue (proxy for σ²).
        mp_upper: Marchenko-Pastur upper bulk edge.
        mp_lower: Marchenko-Pastur lower bulk edge.
        q: Aspect ratio ``H / T``.
        signal_detected: True if ``λ_max > 1.05 · λ_+`` (BBP supercritical).
        bbp_theta: Effective signal strength ``λ_max / σ² - 1``.
        spectral_gap: ``λ_max - λ_min``.
        n_bulk_eigvals: Count of eigenvalues inside MP support.
        n_outlier_eigvals: Count of eigenvalues above MP upper edge.
    """
    layer: int
    n_tokens: int
    hidden_dim: int
    lambda_max: float
    lambda_min: float
    lambda_mean: float
    mp_upper: float
    mp_lower: float
    q: float
    signal_detected: bool
    bbp_theta: float
    spectral_gap: float
    n_bulk_eigvals: int
    n_outlier_eigvals: int


@dataclass
class PromptResult:
    """Spectral analysis of one prompt across all layers.

    Attributes:
        prompt: The input text.
        n_tokens: Number of tokens after tokenization.
        layers: List of :class:`LayerSpectrum`, one per layer.
        bbp_transition_layer: Index of the first layer where
            ``signal_detected`` is True, or ``None`` if no transition.
    """
    prompt: str
    n_tokens: int
    layers: list[LayerSpectrum]
    bbp_transition_layer: int | None = None


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------
class GPT2Probe:
    """Probe GPT-2 hidden states and apply RMT spectral diagnostics.

    Args:
        model_name: HuggingFace model name (default ``"gpt2"`` = 124M).
        device: ``"cpu"`` or ``"cuda"``.
        cache_dir: Optional HuggingFace cache directory.

    Raises:
        ImportError: If ``transformers`` / ``torch`` not installed.
    """

    def __init__(
        self, model_name: str = "gpt2",
        device: str = "cpu", cache_dir: str | None = None,
    ) -> None:
        if not _HAS_TF:
            raise ImportError(
                "probe_real_gpt2 requires `transformers` and `torch`. "
                f"Install with: pip install transformers torch. Error: {_IMPORT_ERROR}"
            )
        self.model_name = model_name
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, cache_dir=cache_dir, output_hidden_states=True,
        ).to(device)
        self.model.eval()
        self.n_layers = self.model.config.n_layer
        self.hidden_dim = self.model.config.n_embd

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------
    def extract_hidden_states(
        self, prompt: str, max_length: int = 256,
    ) -> tuple[np.ndarray, int]:
        """Run GPT-2 on a prompt and extract per-layer hidden states.

        Args:
            prompt: Input text.
            max_length: Maximum token length (truncates longer prompts).

        Returns:
            Tuple ``(hidden_states, n_tokens)`` where ``hidden_states``
            is an array of shape ``(n_layers, n_tokens, hidden_dim)``.
        """
        tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=max_length,
        ).to(self.device)
        n_tokens = tokens["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model(**tokens)
        # outputs.hidden_states is a tuple of length n_layers+1
        # (embeddings + each layer output). Each is (1, T, H).
        hs = torch.stack(outputs.hidden_states[1:], dim=0).squeeze(1)  # (n_layers, T, H)
        hs = hs.cpu().numpy().astype(np.float64)
        return hs, n_tokens

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_layer(
        hidden: np.ndarray, layer_idx: int,
    ) -> LayerSpectrum:
        """Compute the RMT spectral diagnostics for one layer.

        Args:
            hidden: Hidden states of shape ``(T, H)``.
            layer_idx: Layer index for the report.

        Returns:
            :class:`LayerSpectrum` with all spectral statistics.
        """
        T, H = hidden.shape
        if T < 2:
            return LayerSpectrum(
                layer=layer_idx, n_tokens=T, hidden_dim=H,
                lambda_max=0.0, lambda_min=0.0, lambda_mean=0.0,
                mp_upper=0.0, mp_lower=0.0, q=float(H) / max(T, 1),
                signal_detected=False, bbp_theta=0.0, spectral_gap=0.0,
                n_bulk_eigvals=0, n_outlier_eigvals=0,
            )
        cov = np.cov(hidden.T)  # (H, H)
        all_eigvals = np.linalg.eigvalsh(cov)
        eigvals = all_eigvals[all_eigvals > 1e-12]
        if len(eigvals) == 0:
            return LayerSpectrum(
                layer=layer_idx, n_tokens=T, hidden_dim=H,
                lambda_max=0.0, lambda_min=0.0, lambda_mean=0.0,
                mp_upper=0.0, mp_lower=0.0, q=float(H) / max(T, 1),
                signal_detected=False, bbp_theta=0.0, spectral_gap=0.0,
                n_bulk_eigvals=0, n_outlier_eigvals=0,
            )
        lam_max = float(eigvals.max())
        lam_min = float(eigvals.min())
        lam_mean_nonzero = float(eigvals.mean())
        # σ² estimator: trace(cov)/H = mean of ALL eigenvalues (incl. zeros).
        # For q > 1 this is the correct MP variance; using the mean of
        # non-zero eigenvalues would inflate σ² by a factor of q.
        sigma2 = float(np.mean(all_eigvals))
        if sigma2 <= 0:
            sigma2 = lam_mean_nonzero
        q = H / max(T, 1)
        mp_lower, mp_upper = _mp_bounds_general(q, sigma2)
        signal_detected = lam_max > mp_upper * 1.05
        bbp_theta = lam_max / sigma2 - 1.0
        n_bulk = int(np.sum((eigvals >= mp_lower) & (eigvals <= mp_upper)))
        n_outlier = int(np.sum(eigvals > mp_upper))
        return LayerSpectrum(
            layer=layer_idx, n_tokens=T, hidden_dim=H,
            lambda_max=lam_max, lambda_min=lam_min, lambda_mean=lam_mean_nonzero,
            mp_upper=mp_upper, mp_lower=mp_lower, q=q,
            signal_detected=signal_detected, bbp_theta=bbp_theta,
            spectral_gap=lam_max - lam_min,
            n_bulk_eigvals=n_bulk, n_outlier_eigvals=n_outlier,
        )

    def analyze_prompt(self, prompt: str) -> PromptResult:
        """Run a full spectral analysis on one prompt.

        Args:
            prompt: Input text.

        Returns:
            :class:`PromptResult` with per-layer spectra.
        """
        hidden_states, n_tokens = self.extract_hidden_states(prompt)
        layers: list[LayerSpectrum] = []
        for li in range(hidden_states.shape[0]):
            spec = self.analyze_layer(hidden_states[li], li)
            layers.append(spec)
        bbp_layer: int | None = None
        for spec in layers:
            if spec.signal_detected:
                bbp_layer = spec.layer
                break
        return PromptResult(
            prompt=prompt, n_tokens=n_tokens, layers=layers,
            bbp_transition_layer=bbp_layer,
        )

    def analyze_prompts(
        self, prompts: list[str],
    ) -> list[PromptResult]:
        """Analyze a list of prompts.

        Args:
            prompts: List of input texts.

        Returns:
            List of :class:`PromptResult`.
        """
        return [self.analyze_prompt(p) for p in prompts]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def report(result: PromptResult) -> str:
        """Format a :class:`PromptResult` as a human-readable string."""
        lines = [
            f"Prompt: {result.prompt[:80]!r}...",
            f"Tokens: {result.n_tokens}",
            f"BBP transition layer: {result.bbp_transition_layer}",
            "",
            f"{'Layer':>5}  {'T':>4}  {'H':>4}  {'λ_max':>10}  "
            f"{'λ_MP+':>10}  {'q':>6}  {'signal':>6}  {'outliers':>8}",
            "-" * 70,
        ]
        for spec in result.layers:
            lines.append(
                f"{spec.layer:>5}  {spec.n_tokens:>4}  {spec.hidden_dim:>4}  "
                f"{spec.lambda_max:>10.4f}  {spec.mp_upper:>10.4f}  "
                f"{spec.q:>6.3f}  {'YES' if spec.signal_detected else 'no':>6}  "
                f"{spec.n_outlier_eigvals:>8}"
            )
        return "\n".join(lines)

    @staticmethod
    def to_json(results: list[PromptResult]) -> str:
        """Serialize results to JSON.

        Converts numpy bools/floats to native Python types for
        JSON compatibility.
        """
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        return json.dumps(_convert([asdict(r) for r in results]), indent=2)

    @staticmethod
    def save_results(
        results: list[PromptResult], path: str,
    ) -> None:
        """Save results to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(GPT2Probe.to_json(results))


# ---------------------------------------------------------------------------
# Fallback: synthetic probe (when transformers is not available)
# ---------------------------------------------------------------------------
class SyntheticProbe:
    """A NumPy-only stand-in for :class:`GPT2Probe`.

    Generates synthetic hidden states from a spiked-covariance model
    so the RMT diagnostics can be exercised without the
    ``transformers`` dependency. Useful for CI and unit tests.
    """

    def __init__(
        self, n_layers: int = 12, hidden_dim: int = 768,
        seed: int = 42,
    ) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

    def extract_hidden_states(
        self, prompt: str, max_length: int = 256,
    ) -> tuple[np.ndarray, int]:
        """Generate synthetic hidden states for a prompt.

        The synthetic model uses a rank-1 spiked covariance whose
        spike strength grows with layer depth, simulating the BBP
        transition.
        """
        n_tokens = min(len(prompt.split()), max_length)
        n_tokens = max(n_tokens, 8)
        H = self.hidden_dim
        hidden = np.zeros((self.n_layers, n_tokens, H), dtype=np.float64)
        for li in range(self.n_layers):
            # Background noise.
            X = self.rng.normal(0, 1, (H, n_tokens))
            # Spike strength grows with layer. Scale so later layers
            # cross the BBP threshold θ_c = √q.
            q = H / n_tokens
            theta_c = np.sqrt(q)
            theta = theta_c * (0.5 + 1.5 * li / max(self.n_layers - 1, 1))
            v = self.rng.normal(0, 1, H)
            v /= np.linalg.norm(v)
            signal = theta * np.outer(v, np.ones(n_tokens))
            X += signal
            hidden[li] = X.T
        return hidden, n_tokens

    def analyze_prompt(self, prompt: str) -> PromptResult:
        """Analyze a synthetic prompt."""
        hidden_states, n_tokens = self.extract_hidden_states(prompt)
        layers: list[LayerSpectrum] = []
        for li in range(hidden_states.shape[0]):
            layers.append(GPT2Probe.analyze_layer(hidden_states[li], li))
        bbp_layer: int | None = None
        for spec in layers:
            if spec.signal_detected:
                bbp_layer = spec.layer
                break
        return PromptResult(
            prompt=prompt, n_tokens=n_tokens, layers=layers,
            bbp_transition_layer=bbp_layer,
        )

    def analyze_prompts(self, prompts: list[str]) -> list[PromptResult]:
        return [self.analyze_prompt(p) for p in prompts]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("GPT-2 Activation Probe — RMT Spectral Analysis")
    print("=" * 70)
    if _HAS_TF:
        print("\ntransformers + torch detected. Running real GPT-2 probe...")
        probe: Any = GPT2Probe(model_name="gpt2")
    else:
        print(f"\ntransformers/torch not available ({_IMPORT_ERROR}).")
        print("Running synthetic probe (NumPy-only fallback)...")
        probe = SyntheticProbe(n_layers=12, hidden_dim=128, seed=42)

    prompts = [
        "The capital of France is Paris.",
        "Random Matrix Theory studies the eigenvalue distribution of large random matrices.",
        "In machine learning, attention mechanisms allow models to focus on relevant parts of the input.",
    ]
    results = probe.analyze_prompts(prompts)
    for r in results:
        print()
        print(GPT2Probe.report(r))
    # Save JSON
    out_path = "/tmp/gpt2_probe_results.json"
    GPT2Probe.save_results(results, out_path)
    print(f"\nResults saved to {out_path}")

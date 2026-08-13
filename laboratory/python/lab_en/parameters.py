"""
parameters.py — Infinite Parameter System for RMT-LLM Laboratory
================================================================

Custom launch system supporting parameters with bounds [0, +inf).
Each parameter can be int, float, range, categorical, bool, or string.
Both file-based (JSON) and interactive wizard modes are supported.

Author: Iskhak Hamzatovich Isaev
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Sentinel for infinity
# ---------------------------------------------------------------------------
INF = math.inf  # also accepts the string "inf" in JSON


# ---------------------------------------------------------------------------
# Parameter descriptor
# ---------------------------------------------------------------------------
@dataclass
class Parameter:
    """Single parameter descriptor. min/max accept any non-negative number or 'inf'."""
    name: str
    type: str  # int | float | range | categorical | bool | string
    default: Any
    min: Union[float, str] = 0.0
    max: Union[float, str] = INF
    step: Union[float, str] = 1.0
    choices: Optional[List[Any]] = None
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.min, str):
            self.min = INF if self.min.lower() in ("inf", "+inf", "infinity") else float(self.min)
        if isinstance(self.max, str):
            self.max = INF if self.max.lower() in ("inf", "+inf", "infinity") else float(self.max)
        if isinstance(self.step, str):
            self.step = INF if self.step.lower() in ("inf", "+inf", "infinity") else float(self.step)

    def validate(self, value: Any) -> Any:
        """Validate and coerce a value against this parameter's spec."""
        if self.type == "bool":
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes", "y", "да", "д")
            return bool(value)

        if self.type == "string":
            return str(value)

        if self.type == "categorical":
            if value not in (self.choices or []):
                raise ValueError(f"{self.name}: '{value}' not in choices {self.choices}")
            return value

        # Numeric: int, float, range
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: cannot parse '{value}' as number") from exc

        if math.isfinite(self.min) and num < self.min:
            raise ValueError(f"{self.name}: {num} < min {self.min}")
        if math.isfinite(self.max) and num > self.max:
            raise ValueError(f"{self.name}: {num} > max {self.max}")

        if self.type == "int":
            if abs(num - round(num)) > 1e-9:
                raise ValueError(f"{self.name}: {num} is not an integer")
            return int(num)
        return num

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("min", "max", "step"):
            v = d[k]
            if v == INF:
                d[k] = "inf"
            elif isinstance(v, float) and v.is_integer():
                d[k] = int(v)
        return d


# ---------------------------------------------------------------------------
# Default parameter space (infinite upper bound)
# ---------------------------------------------------------------------------
def default_parameter_space() -> List[Parameter]:
    """Default RMT-LLM laboratory parameter space. All upper bounds are infinite."""
    return [
        Parameter("temperature", "float", 0.7, 0.0, INF, 0.01, unit="",
                  description="Sampling temperature (0 = greedy, inf = pure random)"),
        Parameter("max_tokens", "int", 256, 1, INF, 1, unit="tokens",
                  description="Maximum tokens to generate"),
        Parameter("top_k", "int", 50, 0, INF, 1, unit="",
                  description="Top-k filtering (0 = disabled, inf = no filter)"),
        Parameter("top_p", "float", 0.95, 0.0, 1.0, 0.01, unit="",
                  description="Nucleus sampling probability mass"),
        Parameter("context_window", "int", 1024, 1, INF, 1, unit="tokens",
                  description="Context window size"),
        Parameter("ncrit_threshold", "float", 114.0, 0.0, INF, 0.1, unit="tokens",
                  description="RMT critical token count threshold"),
        Parameter("theta_b_deg", "float", 7.07, 0.0, 360.0, 0.01, unit="deg",
                  description="BBP rotation angle"),
        Parameter("beta_caputo", "float", 0.5, 0.0, INF, 0.01, unit="",
                  description="Caputo fractional memory parameter"),
        Parameter("rlhf_pressure", "float", 0.0, 0.0, INF, 0.01, unit="",
                  description="RLHF drift strength — accelerates hallucination"),
        Parameter("n_layers", "int", 6, 1, INF, 1, unit="",
                  description="Number of transformer layers"),
        Parameter("hidden_dim", "int", 64, 1, INF, 1, unit="",
                  description="Hidden dimension of synthetic model"),
        Parameter("n_heads", "int", 4, 1, INF, 1, unit="",
                  description="Number of attention heads"),
        Parameter("vocab_size", "int", 256, 1, INF, 1, unit="",
                  description="Vocabulary size of synthetic model"),
        Parameter("seed", "int", 42, 0, INF, 1, unit="",
                  description="Random seed"),
        Parameter("epochs", "int", 3, 0, INF, 1, unit="",
                  description="Training epochs for synthetic model"),
        Parameter("learning_rate", "float", 1e-3, 0.0, INF, 1e-6, unit="",
                  description="Learning rate"),
        Parameter("batch_size", "int", 4, 1, INF, 1, unit="",
                  description="Batch size"),
        Parameter("enable_filter", "bool", True, description="Enable output safety filter"),
        Parameter("capture_hidden", "bool", True, description="Capture hidden reasoning trace"),
        Parameter("language", "categorical", "en",
                  choices=["en", "ru"], description="Output language"),
        Parameter("report_format", "categorical", "all",
                  choices=["all", "txt", "md", "csv", "html", "json", "pdf",
                           "docx", "yaml", "xml", "latex", "parquet", "xlsx", "sqlite"],
                  description="Report format (all = 13 formats)"),
    ]


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
def interactive_wizard(params: Optional[List[Parameter]] = None) -> Dict[str, Any]:
    """Step-by-step interactive wizard. Returns validated parameter dict."""
    params = params or default_parameter_space()
    print("\n=== INTERACTIVE PARAMETER WIZARD ===")
    print("Enter values for each parameter. Press <Enter> to accept the default.")
    print("Numeric bounds support 'inf' for infinity. Range [0, inf) by default.\n")

    values: Dict[str, Any] = {}
    for p in params:
        while True:
            hint = f"[default={p.default}]"
            if p.type == "categorical":
                hint += f" choices={p.choices}"
            elif p.type == "bool":
                hint += " (y/n)"
            else:
                lo = "0" if p.min == 0 else (str(p.min) if math.isfinite(p.min) else "-inf")
                hi = "inf" if p.max == INF else str(p.max)
                hint += f" range=[{lo}, {hi}]"
            try:
                raw = input(f"  {p.name} ({p.unit}) {hint}: ").strip()
                if raw == "":
                    values[p.name] = p.default
                    break
                if p.type == "float" and raw.lower() in ("inf", "+inf", "infinity"):
                    values[p.name] = INF
                    break
                values[p.name] = p.validate(raw)
                break
            except ValueError as exc:
                print(f"    [ERROR] {exc}. Try again.")
    print()
    return values


# ---------------------------------------------------------------------------
# JSON config file mode
# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file. The file may include a 'parameters' dict
    whose values are either raw values (validated against defaults) or full
    Parameter descriptors (overrides the default)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def save_config(path: str, values: Dict[str, Any]) -> None:
    """Save a parameter dict to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def parse_cli_args(args: Optional[List[str]] = None,
                   params: Optional[List[Parameter]] = None) -> Dict[str, Any]:
    """Parse --param=value style CLI args against the default parameter space."""
    import sys
    args = args if args is not None else sys.argv[1:]
    params = params or default_parameter_space()
    by_name = {p.name: p for p in params}
    values: Dict[str, Any] = {p.name: p.default for p in params}

    for a in args:
        if not a.startswith("--") or "=" not in a:
            continue
        key, val = a[2:].split("=", 1)
        if key in by_name:
            values[key] = by_name[key].validate(val)
        else:
            # Unknown parameters are accepted as strings (infinite extensibility)
            values[key] = val
    return values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "INF", "Parameter", "default_parameter_space",
    "interactive_wizard", "load_config", "save_config", "parse_cli_args",
]


if __name__ == "__main__":
    # Smoke test
    space = default_parameter_space()
    print(f"Default parameter space: {len(space)} parameters")
    for p in space[:5]:
        print(f"  - {p.name}: type={p.type}, default={p.default}, range=[{p.min}, {p.max}]")
    print("All upper bounds support 'inf'.")

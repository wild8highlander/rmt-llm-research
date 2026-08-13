"""
parameters.py — Система бесконечных параметров для лаборатории RMT-LLM (RU)
==========================================================================

Кастомная система запуска с параметрами в диапазоне [0, +inf).
Поддерживаются типы: int, float, range, categorical, bool, string.
Два режима: JSON-конфиг и интерактивный мастер.

Автор: Исхак Хамзатович Исаев
Лицензия: Проприетарная — Все права защищены.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

INF = math.inf


@dataclass
class Parameter:
    """Дескриптор одного параметра. min/max принимают любое число или 'inf'."""
    name: str
    type: str
    default: Any
    min: Union[float, str] = 0.0
    max: Union[float, str] = INF
    step: Union[float, str] = 1.0
    choices: Optional[List[Any]] = None
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.min, str):
            self.min = INF if self.min.lower() in ("inf", "+inf", "infinity", "беск") else float(self.min)
        if isinstance(self.max, str):
            self.max = INF if self.max.lower() in ("inf", "+inf", "infinity", "беск") else float(self.max)
        if isinstance(self.step, str):
            self.step = INF if self.step.lower() in ("inf", "+inf", "infinity", "беск") else float(self.step)

    def validate(self, value: Any) -> Any:
        if self.type == "bool":
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes", "y", "да", "д")
            return bool(value)
        if self.type == "string":
            return str(value)
        if self.type == "categorical":
            if value not in (self.choices or []):
                raise ValueError(f"{self.name}: '{value}' не входит в варианты {self.choices}")
            return value
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: нельзя разобрать '{value}' как число") from exc
        if math.isfinite(self.min) and num < self.min:
            raise ValueError(f"{self.name}: {num} < минимума {self.min}")
        if math.isfinite(self.max) and num > self.max:
            raise ValueError(f"{self.name}: {num} > максимума {self.max}")
        if self.type == "int":
            if abs(num - round(num)) > 1e-9:
                raise ValueError(f"{self.name}: {num} не целое")
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


def default_parameter_space() -> List[Parameter]:
    """Пространство параметров по умолчанию. Все верхние границы бесконечны."""
    return [
        Parameter("temperature", "float", 0.7, 0.0, INF, 0.01, unit="",
                  description="Температура сэмплирования (0 = жадно, inf = чистый случай)"),
        Parameter("max_tokens", "int", 256, 1, INF, 1, unit="токены",
                  description="Максимум генерируемых токенов"),
        Parameter("top_k", "int", 50, 0, INF, 1, unit="",
                  description="Top-k фильтрация (0 = выкл, inf = без фильтра)"),
        Parameter("top_p", "float", 0.95, 0.0, 1.0, 0.01, unit="",
                  description="Ядерная сэмплирующая масса"),
        Parameter("context_window", "int", 1024, 1, INF, 1, unit="токены",
                  description="Размер контекстного окна"),
        Parameter("ncrit_threshold", "float", 114.0, 0.0, INF, 0.1, unit="токены",
                  description="Порог критического числа токенов N_crit (RMT)"),
        Parameter("theta_b_deg", "float", 7.07, 0.0, 360.0, 0.01, unit="град",
                  description="Угол вращения BBP"),
        Parameter("beta_caputo", "float", 0.5, 0.0, INF, 0.01, unit="",
                  description="Параметр дробной памяти Капуто"),
        Parameter("rlhf_pressure", "float", 0.0, 0.0, INF, 0.01, unit="",
                  description="Сила RLHF-дрейфа — ускоряет галлюцинации"),
        Parameter("n_layers", "int", 6, 1, INF, 1, unit="",
                  description="Число слоёв трансформера"),
        Parameter("hidden_dim", "int", 64, 1, INF, 1, unit="",
                  description="Скрытая размерность синтетической модели"),
        Parameter("n_heads", "int", 4, 1, INF, 1, unit="",
                  description="Число голов внимания"),
        Parameter("vocab_size", "int", 256, 1, INF, 1, unit="",
                  description="Размер словаря синтетической модели"),
        Parameter("seed", "int", 42, 0, INF, 1, unit="",
                  description="Случайное зерно"),
        Parameter("epochs", "int", 3, 0, INF, 1, unit="",
                  description="Эпохи обучения синтетической модели"),
        Parameter("learning_rate", "float", 1e-3, 0.0, INF, 1e-6, unit="",
                  description="Скорость обучения"),
        Parameter("batch_size", "int", 4, 1, INF, 1, unit="",
                  description="Размер батча"),
        Parameter("enable_filter", "bool", True, description="Включить выходной защитный фильтр"),
        Parameter("capture_hidden", "bool", True, description="Захватывать скрытую цепочку рассуждений"),
        Parameter("language", "categorical", "ru",
                  choices=["en", "ru"], description="Язык вывода"),
        Parameter("report_format", "categorical", "all",
                  choices=["all", "txt", "md", "csv", "html", "json", "pdf",
                           "docx", "yaml", "xml", "latex", "parquet", "xlsx", "sqlite"],
                  description="Формат отчёта (all = все 13 форматов)"),
    ]


def interactive_wizard(params: Optional[List[Parameter]] = None) -> Dict[str, Any]:
    """Пошаговый интерактивный мастер. Возвращает словарь провалидированных параметров."""
    params = params or default_parameter_space()
    print("\n=== ИНТЕРАКТИВНЫЙ МАСТЕР ПАРАМЕТРОВ ===")
    print("Вводите значения для каждого параметра. <Enter> = значение по умолчанию.")
    print("Числовые границы поддерживают 'inf' для бесконечности. Диапазон [0, inf) по умолчанию.\n")

    values: Dict[str, Any] = {}
    for p in params:
        while True:
            hint = f"[по умолчанию={p.default}]"
            if p.type == "categorical":
                hint += f" варианты={p.choices}"
            elif p.type == "bool":
                hint += " (да/нет)"
            else:
                lo = "0" if p.min == 0 else (str(p.min) if math.isfinite(p.min) else "-inf")
                hi = "inf" if p.max == INF else str(p.max)
                hint += f" диапазон=[{lo}, {hi}]"
            try:
                raw = input(f"  {p.name} ({p.unit}) {hint}: ").strip()
                if raw == "":
                    values[p.name] = p.default
                    break
                if p.type == "float" and raw.lower() in ("inf", "+inf", "infinity", "беск"):
                    values[p.name] = INF
                    break
                values[p.name] = p.validate(raw)
                break
            except ValueError as exc:
                print(f"    [ОШИБКА] {exc}. Повторите.")
    print()
    return values


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(path: str, values: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2, ensure_ascii=False)


def parse_cli_args(args: Optional[List[str]] = None,
                   params: Optional[List[Parameter]] = None) -> Dict[str, Any]:
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
            values[key] = val
    return values


__all__ = [
    "INF", "Parameter", "default_parameter_space",
    "interactive_wizard", "load_config", "save_config", "parse_cli_args",
]

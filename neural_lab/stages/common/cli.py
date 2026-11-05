"""common.cli — shared StageRunner plumbing for stages 2–5 (P2.2).

The four stage mains duplicated the same ~80 lines: parse args → load
model → build passport → run → report. This module factors that out:

    class Stage2Runner(StageRunner):
        def add_args(self, parser): ...
        def run(self, cfg, model, tok): return sections, extra

    if __name__ == "__main__":
        main(Stage2Runner())

Bought by the abstraction (all stages get it for free):

  --config FILE   YAML run config; CLI flags override it (P2.3)
  --resume        skip items/tasks already recorded in the incremental
                  output files (P2.4; the runner decides what "done" is)
  --only a,b      run a subset of the stage's steps (P2.12)
  --dry-run       print the merged config and exit before any model load
  --verbose/--quiet/--log-json FILE   structured logging + progress (P2.5)

run() returns (sections, extra) — main() then writes report.json /
REPORT.md (schema-validated via common.schemas). A runner may instead
return None after writing its own artifacts (stage-3 scaling, stage-5
pair records).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from abc import ABC

log = logging.getLogger("stages")  # 'print' in runners is fine; log.* is parseable


# --- logging (P2.5) ------------------------------------------------------------


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(verbose: bool = False, quiet: bool = False, log_json: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    root = logging.getLogger("stages")
    root.handlers.clear()
    root.setLevel(level)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream)
    if log_json:
        os.makedirs(os.path.dirname(os.path.abspath(log_json)), exist_ok=True)
        fh = logging.FileHandler(log_json, encoding="utf-8")
        fh.setFormatter(JsonLogFormatter())
        root.addHandler(fh)


def progress(iterable, cfg: dict, desc: str = "", total: int | None = None):
    """tqdm on the outer loop when available; a plain counter otherwise.

    Quiet mode (or a non-tty run under --quiet) returns the iterable bare."""
    if cfg.get("quiet"):
        return iterable
    try:
        from tqdm import tqdm  # optional dependency

        return tqdm(iterable, desc=desc, total=total, unit="item")
    except ImportError:
        n_total = total if total is not None else len(iterable)  # type: ignore[arg-type]
        every = max(1, n_total // 10) if n_total else 1
        count = 0

        def _gen():
            nonlocal count
            for x in iterable:
                count += 1
                if count % every == 0 or count == n_total:
                    log.info(f"  {desc} {count}/{n_total}")
                yield x

        return _gen()


# --- incremental output / resume (P2.4) -----------------------------------------


def load_done_ids(path: str, id_key: str = "item_id") -> set:
    """Ids already recorded in an incremental JSONL artifact (resume)."""
    return {r[id_key] for r, _ in _read_jsonl_rows(path, id_key)}


def load_done_rows(path: str, id_key: str = "item_id") -> tuple[list[dict], set]:
    """(rows, done_ids) from an incremental JSONL artifact (resume)."""
    rows = _read_jsonl_rows(path, id_key)
    return [r for r, _ in rows], {i for _, i in rows}


def _read_jsonl_rows(path: str, id_key: str) -> list[tuple[dict, object]]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated by a crash mid-write: that item reruns
            if id_key in row:
                out.append((row, row[id_key]))
    return out


class JsonlAppender:
    """Append-only JSONL writer; one flush per row so a crash loses at
    most the item in flight, never the whole run."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- steps (--only, P2.12) -------------------------------------------------------


def step_requested(cfg: dict, name: str) -> bool:
    only = cfg.get("only")
    return only is None or name in only


def duration_note(name: str, t0: float) -> str:
    return f"step '{name}' took {time.time() - t0:.1f}s"


# --- config merging (P2.3) --------------------------------------------------------


def _snapshot_defaults(parser: argparse.ArgumentParser) -> dict:
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


def _normalize_key(k: str) -> str:
    return k.replace("-", "_")


def parse_config(runner: "StageRunner", argv: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    """Parse argv and resolve the effective config: defaults <- YAML <- CLI.

    Returns (cfg, args). cfg holds every dest (underscores); YAML keys are
    normalized to underscores. A CLI flag beats YAML only when the flag was
    actually passed (value differs from its parser default)."""
    parser = build_parser(runner)
    defaults = _snapshot_defaults(parser)
    args = parser.parse_args(argv)
    explicit = {d for d, v in vars(args).items() if d in defaults and v != defaults[d]}

    cfg = dict(runner.default_config())
    cfg.update({k: v for k, v in vars(args).items()})
    if args.config:
        import yaml

        with open(args.config, encoding="utf-8") as f:
            ycfg = yaml.safe_load(f) or {}
        if not isinstance(ycfg, dict):
            raise SystemExit(f"--config: {args.config} must contain a YAML mapping")
        for k, v in ycfg.items():
            k = _normalize_key(k)
            if k in explicit:
                continue  # explicit CLI flag wins over the file
            cfg[k] = v
    if args.only:
        cfg["only"] = frozenset(s.strip() for s in args.only.split(",") if s.strip())
        unknown = cfg["only"] - set(runner.steps)
        if unknown:
            raise SystemExit(f"--only: unknown steps {sorted(unknown)}; available: {list(runner.steps)}")
    else:
        cfg["only"] = None
    return cfg, args


def build_parser(runner: "StageRunner") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{runner.name}: {runner.title}")
    # model flags are common, but not required here: some runners have
    # model-less modes (stage-3 --scaling-in, stage-5 --list); the runner
    # validates via needs_model(cfg) inside main()
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None,
                    help="output dir (required unless the mode writes nothing, e.g. --list)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="YAML run config; CLI flags override it")
    ap.add_argument("--resume", action="store_true",
                    help="skip items/tasks already recorded in the incremental output")
    ap.add_argument("--only", default=None, metavar="a,b",
                    help=f"run a subset of steps (available: {', '.join(runner.steps) or 'n/a'})")
    ap.add_argument("--dry-run", action="store_true", help="print the merged config and exit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-json", default=None, metavar="FILE",
                    help="also write structured JSON logs to FILE")
    runner.add_args(ap)
    return ap


# --- the runner -------------------------------------------------------------------


class StageRunner(ABC):
    """One research stage. Subclasses implement add_args() and run()."""

    name: str = "stage"
    title: str = "Stage"
    experiment_id: str = "stage_v1"
    hypothesis_id: str = "S?"
    steps: tuple[str, ...] = ()  # valid --only names
    requires_out: bool = True  # False for runners with out-less modes (--list)

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        """Stage-specific CLI flags."""

    def default_config(self) -> dict:
        """Stage constants that are not CLI flags (part of config_hash)."""
        return {}

    def needs_model(self, cfg: dict) -> bool:
        return True

    def run(self, cfg: dict, model, tok, passport: dict):
        """Execute the stage. `passport` may be enriched in place (e.g.
        data_hash of the loaded data) before the report is written.

        Returns (sections, extra) and main() writes the report; or None
        when the runner wrote its own artifacts (multi-output stages)."""
        raise NotImplementedError


def main(runner: StageRunner, argv: list[str] | None = None) -> None:
    cfg, args = parse_config(runner, argv)
    setup_logging(args.verbose, args.quiet, args.log_json)
    if args.dry_run:
        print(json.dumps({k: (sorted(v) if isinstance(v, (set, frozenset)) else v)
                          for k, v in cfg.items()}, indent=2, ensure_ascii=False, default=str))
        return
    if runner.requires_out and not args.out:
        raise SystemExit(f"{runner.name}: --out is required for a run")

    from stages.common.hf_model import load_causal_lm, model_passport
    from stages.common.passports import build_passport

    model = tok = None
    model_fields: dict = {}
    if runner.needs_model(cfg):
        if not args.model:
            raise SystemExit(f"{runner.name}: --model is required for this mode")
        log.info(f"[{runner.name}] loading {args.model} …")
        model, tok = load_causal_lm(args.model, revision=getattr(args, "revision", None),
                                    dtype=args.dtype, device=args.device)
        model_fields = model_passport(model, tok)

    passport = build_passport(
        experiment_id=runner.experiment_id,
        hypothesis_id=runner.hypothesis_id,
        model_fields=model_fields,
        config=cfg,
    )
    result = runner.run(cfg, model, tok, passport)
    if result is None:
        return  # the runner wrote its own artifacts
    sections, extra = result
    from stages.common.reporting import write_report

    path = write_report(args.out, runner.title, passport, sections, extra=extra)
    log.info(f"[{runner.name}] report: {path}")

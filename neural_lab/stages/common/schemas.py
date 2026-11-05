"""common.schemas — JSON-Schema validation of stage result artifacts.

Every artifact the stages write carries "schema_version": "1.0" and is
validated against neural_lab/stages/results_schema/v1/ at write time —
a malformed report fails the run at the boundary, not three stages later
when some consumer trips over it. Schemas use JSON Schema draft 2020-12.

CLI (also wired into `make validate-results`):

    python -m stages.common.schemas results/            # validate a tree
    python -m stages.common.schemas a/report.json ...   # validate files
"""

from __future__ import annotations

import json
import os
import sys

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_schema")

# artifact filename -> schema name (a run dir may contain several)
ARTIFACT_SCHEMAS = {
    "report.json": "report",
    "fit.json": "fit",
    "record.json": "record",
    "scaling.json": "scaling",
}

SCHEMA_VERSION = "1.0"

_CACHE: dict[str, dict] = {}


def load_schema(name: str) -> dict:
    if name not in _CACHE:
        path = os.path.join(SCHEMA_DIR, "v1", f"{name}.schema.json")
        with open(path, encoding="utf-8") as f:
            _CACHE[name] = json.load(f)
    return _CACHE[name]


def validate(artifact: dict, name: str) -> None:
    """Validate one artifact dict against the named v1 schema.

    Raises jsonschema.ValidationError with a human-readable path on
    failure; returns silently on success. The artifact carries its
    schema_version and is validated WITH it (the schemas pin the const)."""
    import jsonschema

    jsonschema.validate(artifact, load_schema(name))


def validate_report(payload: dict) -> None:
    validate(payload, "report")


def validate_fit(payload: dict) -> None:
    validate(payload, "fit")


def validate_record(payload: dict) -> None:
    validate(payload, "record")


def validate_scaling(payload: dict) -> None:
    validate(payload, "scaling")


def validate_tree(root: str) -> list[tuple[str, str]]:
    """Validate every known artifact under `root`; returns [(path, schema)].

    Unknown JSON files are ignored. Raises on the first invalid artifact."""
    checked: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            schema_name = ARTIFACT_SCHEMAS.get(fn)
            if schema_name is None:
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            validate(payload, schema_name)
            checked.append((path, schema_name))
    return checked


def main(argv: list[str] | None = None) -> int:
    ap = argparse_validate()
    args = ap.parse_args(argv)
    try:
        if len(args.paths) == 1 and os.path.isdir(args.paths[0]):
            checked = validate_tree(args.paths[0])
        else:
            checked = []
            for p in args.paths:
                schema_name = ARTIFACT_SCHEMAS.get(os.path.basename(p))
                if schema_name is None:
                    print(f"[schemas] SKIP (no schema for {os.path.basename(p)}): {p}")
                    continue
                with open(p, encoding="utf-8") as f:
                    validate(json.load(f), schema_name)
                checked.append((p, schema_name))
    except Exception as e:  # validation error or unreadable file
        print(f"[schemas] INVALID: {e}")
        return 1
    for p, name in checked:
        print(f"[schemas] OK  ({name})  {p}")
    print(f"[schemas] {len(checked)} artifact(s) valid")
    return 0


def argparse_validate():
    import argparse

    ap = argparse.ArgumentParser(description="Validate stage result artifacts against results_schema/v1")
    ap.add_argument("paths", nargs="+", help="a results tree OR explicit artifact files")
    return ap


if __name__ == "__main__":
    sys.exit(main())

"""common.index — a queryable registry of stage results (P2.7).

Globbing results/** to find "all stage-2 runs on 7B" does not scale; the
runs live in one sqlite database instead:

    python -m stages.common.index --rebuild results/              # scan + index
    python -m stages.common.index --query "model_name LIKE 'Qwen%'"  # ask

Tables: runs (one row per report.json / record.json / scaling.json), fits
(one row per fit.json / stage-5 sidecar). The index is DERIVED data —
rebuild any time; the artifacts remain the source of truth.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

from stages.common.schemas import ARTIFACT_SCHEMAS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    path TEXT PRIMARY KEY,
    kind TEXT,
    stage TEXT,
    experiment_id TEXT,
    hypothesis_id TEXT,
    protocol_version TEXT,
    model_name TEXT,
    git_commit TEXT,
    n_sections INTEGER,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS fits (
    path TEXT PRIMARY KEY,
    model TEXT,
    tau REAL,
    gamma REAL,
    mu_eff REAL,
    source TEXT,
    updated_at TEXT
);
"""


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rebuild(roots: list[str], db_path: str) -> dict:
    """Scan the roots for stage artifacts and (re)build the index."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_SCHEMA)
    counts = {"runs": 0, "fits": 0, "skipped": 0}
    con.execute("DELETE FROM runs")
    con.execute("DELETE FROM fits")
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, encoding="utf-8") as f:
                        payload = json.load(f)
                except (OSError, ValueError):
                    counts["skipped"] += 1  # unreadable/binary (e.g. the index db itself)
                    continue
                kind = ARTIFACT_SCHEMAS.get(fn)
                if kind == "report":
                    _index_report(con, path, payload, counts)
                elif kind == "record":
                    _index_record(con, path, payload, counts)
                elif kind == "scaling":
                    _index_scaling(con, path, payload, counts)
                elif kind == "fit":
                    _index_fit(con, path, payload, counts)
                else:
                    counts["skipped"] += 1
    con.commit()
    con.close()
    return counts


def _index_report(con: sqlite3.Connection, path: str, payload: dict, counts: dict) -> None:
    passport = payload.get("passport") or {}
    code = passport.get("code") or {}
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (path, "report",
         passport.get("experiment_id", "").split("_")[0],
         passport.get("experiment_id"),
         passport.get("hypothesis_id"),
         passport.get("protocol_version"),
         (passport.get("model") or {}).get("model_name"),
         code.get("git_commit"),
         len(payload.get("sections") or []),
         _utcnow()))
    counts["runs"] += 1


def _index_record(con: sqlite3.Connection, path: str, payload: dict, counts: dict) -> None:
    passport = payload.get("passport") or {}
    code = passport.get("code") or {}
    st = payload.get("survival_tuned") or {}
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (path, "record", "stage5", payload.get("pair_id"), "S5",
         passport.get("protocol_version"), payload.get("pair_id"),
         code.get("git_commit"), 0, _utcnow()))
    con.execute(
        "INSERT OR REPLACE INTO fits VALUES (?,?,?,?,?,?,?)",
        (path, payload.get("pair_id"), st.get("tau"), st.get("gamma"),
         payload.get("mu_eff_proxy_kl_ab"), "stage5_record", _utcnow()))
    counts["runs"] += 1
    counts["fits"] += 1


def _index_scaling(con: sqlite3.Connection, path: str, payload: dict, counts: dict) -> None:
    sc = payload.get("scaling") or {}
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (path, "scaling", "stage3", "stage3_scaling_v1", "S3",
         payload.get("protocol_version"),
         f"{sc.get('n_models')} models",
         None, 0, _utcnow()))
    counts["runs"] += 1


def _index_fit(con: sqlite3.Connection, path: str, payload: dict, counts: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO fits VALUES (?,?,?,?,?,?,?)",
        (path, payload.get("model"), payload.get("tau"), payload.get("gamma"),
         payload.get("mu_eff"), payload.get("source", "stage3_fit"), _utcnow()))
    counts["fits"] += 1


def query(db_path: str, sql: str) -> list[tuple]:
    """Run a read-only query against the index."""
    if not os.path.isfile(db_path):
        raise SystemExit(f"no index at {db_path}; run --rebuild first")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="sqlite index of stage results (P2.7)")
    ap.add_argument("--rebuild", nargs="*", default=None, metavar="ROOT",
                    help="scan these result roots (default: results/) and rebuild the index")
    ap.add_argument("--db", default=None, help="index path (default: <first root>/index.sqlite)")
    ap.add_argument("--query", default=None, help="read-only SQL against the index")
    ap.add_argument("--schema", action="store_true", help="print the table schemas")
    args = ap.parse_args(argv)

    if args.schema:
        print(_SCHEMA)
        return 0
    if args.query:
        rows = query(args.db or "results/index.sqlite", args.query)
        for r in rows:
            print(" | ".join(str(c) for c in r))
        print(f"[index] {len(rows)} row(s)")
        return 0
    roots = args.rebuild if args.rebuild is not None else []
    if not roots:
        ap.error("--rebuild ROOT [...] or --query SQL or --schema")
    db = args.db or os.path.join(roots[0], "index.sqlite")
    counts = rebuild(roots, db)
    print(f"[index] {db}: {counts['runs']} runs, {counts['fits']} fits, "
          f"{counts['skipped']} skipped files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

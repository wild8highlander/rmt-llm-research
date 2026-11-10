#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_kernel.py — persistent Python kernel for the RCDA-4 WSP agent harness.

MIT exp-3/exp-6-style: the agent gets a persistent Python session over the
artifacts (numpy analyses, state-dict edits, benchmark calls). The kernel:

  * runs a bootstrap script (loads the workspace: matrices / state_dict / model),
  * then reads cells from stdin:  "<<<RUN>>>\\n<code>\\n<<<EOF>>>",
  * executes each cell in ONE persistent namespace (state survives between cells),
  * returns the captured stdout tail (<= OUT_MAX chars) between <<<OUT>>>...<<<DONE>>>.

Line protocol, one process, no shell — the agent never touches the OS directly.
"""
import io
import sys
import traceback
import contextlib

BOOT = sys.argv[1]
OUT_MAX = 3000

NS = {"__name__": "kernel"}


def run_cell(code: str) -> str:
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                res = eval(compile(code, "<cell>", "eval"), NS)
                if res is not None:
                    print(repr(res)[:400])
            except SyntaxError:
                exec(compile(code, "<cell>", "exec"), NS)
    except Exception:
        err = traceback.format_exc(limit=4)
    out = buf.getvalue()
    if err:
        out = (out + "\n[ERROR]\n" + err)[-OUT_MAX:]
    else:
        out = out[-OUT_MAX:]
    return out


def main() -> None:
    try:
        src = open(BOOT, encoding="utf-8").read()
        exec(compile(src, BOOT, "exec"), NS)
        sys.stdout.write("[kernel ready]\n")
    except Exception:
        sys.stdout.write("[kernel BOOT FAILED]\n" + traceback.format_exc(limit=6) + "\n")
        sys.stdout.flush()
        return
    sys.stdout.flush()
    for line in sys.stdin:
        if line.rstrip("\n") != "<<<RUN>>>":
            continue
        code_lines = []
        for line2 in sys.stdin:
            if line2.rstrip("\n") == "<<<EOF>>>":
                break
            code_lines.append(line2)
        out = run_cell("".join(code_lines))
        sys.stdout.write("<<<OUT>>>\n" + out + "\n<<<DONE>>>\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

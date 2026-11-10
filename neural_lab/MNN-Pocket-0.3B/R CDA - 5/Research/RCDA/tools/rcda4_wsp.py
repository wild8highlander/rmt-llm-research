#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rcda4_wsp.py — agent driver for RCDA-4 (Weight-Space Programming).

Replaces the human in the RCDA loop with an LLM programmer (GLM via z-ai CLI),
exactly following the MIT exp-3/exp-6/exp-8 harness pattern:

    agent (GLM) ⇄ driver (this script) ⇄ persistent kernel (rcda4_kernel.py)
                                            ├── numpy analyses     (Part C)
                                            ├── state_dict edits   (Part B)
                                            └── benchmark() budget (Part B)

Each turn: transcript -> z-ai chat -> parse ```python cells -> execute in the
kernel -> append bounded outputs -> repeat until FINAL_JSON or budget exhausted.

Usage:
  python3 rcda4_wsp.py --system-file SYS --task-file TASK --boot BOOT.py \
      --cwd DIR --max-turns 12 --max-cells 8 --cell-timeout 120 --out PREFIX
"""
import argparse
import json
import re
import subprocess
import sys

FINAL = "FINAL_JSON"
CELL_RE = re.compile(r"```python\n([\s\S]*?)```", re.S)


def chat(system: str, prompt: str, out_json: str) -> str:
    r = subprocess.run(
        ["z-ai", "chat", "-s", system, "-p", prompt, "-o", out_json],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"z-ai chat failed: {r.stderr[-300:]}")
    d = json.load(open(out_json))
    # CLI may wrap the answer in different shapes; find the text
    if isinstance(d, dict):
        for k in ("content", "response", "message", "text"):
            if k in d and isinstance(d[k], str):
                return d[k]
        if "choices" in d:
            return d["choices"][0]["message"]["content"]
    return str(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system-file", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--boot", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--kernel", default="/home/z/my-project/scripts/rcda4_kernel.py")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-cells", type=int, default=8)
    ap.add_argument("--cell-timeout", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    system = open(args.system_file, encoding="utf-8").read()
    task = open(args.task_file, encoding="utf-8").read()

    import signal

    kernel = subprocess.Popen(
        [sys.executable, "-u", args.kernel, args.boot],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, cwd=args.cwd,
    )
    # wait for "[kernel ready]"
    ready = kernel.stdout.readline()
    print("kernel:", ready.strip(), flush=True)

    def run_cell(code: str) -> str:
        kernel.stdin.write("<<<RUN>>>\n" + code + "\n<<<EOF>>>\n")
        kernel.stdin.flush()
        out_lines, done = [], False
        import threading

        def reader():
            nonlocal done
            for line in kernel.stdout:
                out_lines.append(line)
                if "<<<DONE>>>" in line:
                    done = True
                    break

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=args.cell_timeout)
        text = "".join(out_lines)
        text = text.split("<<<OUT>>>\n")[-1].split("<<<DONE>>>")[0]
        return text.strip()[:3000] or "(no output)"

    msgs = []  # linear transcript (user role = tool outputs / driver notes)
    transcript = []
    final = None
    cells = 0
    for turn in range(1, args.max_turns + 1):
        prompt = task + "\n\n--- Conversation so far ---\n" + "\n".join(msgs[-40:])
        reply = chat(system, prompt, args.out + f"_turn{turn}.json")
        transcript.append({"turn": turn, "reply": reply})
        print(f"--- turn {turn}: {len(reply)} chars", flush=True)
        blocks = CELL_RE.findall(reply)
        if FINAL in reply and not blocks:
            tail = reply.rsplit(FINAL, 1)[1].strip()
            tail = tail.split("```")[0].strip()
            try:
                m = re.search(r"\{[\s\S]*\}", tail)
                json.loads(m.group(0) if m else tail)
                final = m.group(0) if m else tail
                transcript.append({"turn": turn, "final": final})
                break
            except Exception:
                msgs.append("[driver] FINAL_JSON is not valid JSON. Continue working "
                            "with ```python cells, then finish with a single valid "
                            "FINAL_JSON line.")
                continue
        if not blocks:
            msgs.append("[driver] No ```python block found. Send code in ```python "
                        "blocks, or finish with FINAL_JSON {...}")
            continue
        for b in blocks:
            if cells >= args.max_cells:
                msgs.append("[driver] Cell budget exhausted. Finish now: reply with "
                            "FINAL_JSON {...} only.")
                break
            cells += 1
            out = run_cell(b)
            transcript.append({"turn": turn, "cell": cells, "out": out})
            # incremental save: the session may be cut at any moment
            json.dump({"cfg": vars(args), "transcript": transcript,
                       "final": None, "cells_used": cells},
                      open(args.out + "_transcript.json", "w"),
                      ensure_ascii=False, indent=1)
            msgs.append(f"[cell {cells} stdout]\n{out}")
            print(f"    cell {cells}: {len(out)} chars", flush=True)

    kernel.kill()
    json.dump(
        {"cfg": vars(args), "transcript": transcript, "final": final,
         "cells_used": cells},
        open(args.out + "_transcript.json", "w"), ensure_ascii=False, indent=1,
    )
    print(f"DONE cells={cells} final={'yes' if final else 'NO'}", flush=True)


if __name__ == "__main__":
    main()

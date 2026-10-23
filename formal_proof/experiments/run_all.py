#!/usr/bin/env python3
"""Run the full formal_proof verification suite and regenerate all assets."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "verify_mp_law.py",
    "verify_bbp_transition.py",
    "verify_tracy_widom.py",
    "verify_shrinkage.py",
    "verify_entropy.py",
    "verify_layer_ops.py",
]


def main() -> None:
    total = time.time()
    for name in SCRIPTS:
        print(f"\n=== {name} " + "=" * 56)
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(HERE / name)])
        if rc != 0:
            sys.exit(f"{name} failed with exit code {rc}")
        print(f"--- {name} done in {time.time() - t0:.1f}s")
    print(f"\nAll verification experiments finished in {time.time() - total:.1f}s")


if __name__ == "__main__":
    main()

"""pytest configuration for the science tests: paths to neural_lab and src/rmt_llm."""

import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
_NEURAL_LAB = os.path.dirname(os.path.dirname(_HERE))  # neural_lab/
_REPO = os.path.dirname(_NEURAL_LAB)  # repo root

for p in (
    _NEURAL_LAB,  # import science.*
    os.path.join(_REPO, "src", "rmt_llm"),  # mp/bbp/cap/tw for verify_truth
    os.path.join(_REPO, "laboratory", "python", "lab_en"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

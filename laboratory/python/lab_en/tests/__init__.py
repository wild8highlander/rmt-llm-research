"""
Test suite for TinyGPT, BPE tokenizer, and trainer.

Run with:
    cd laboratory/python/lab_en
    pytest tests/ -v
    pytest tests/ -v -k "tiny_gpt"        # only model tests
    pytest tests/ -v -k "bpe"             # only BPE tests
    pytest tests/ -v -k "trainer"         # only trainer tests
    pytest tests/ -v -m "slow"            # only slow tests
    pytest tests/ -v -m "not slow"        # skip slow tests
"""

# Make lab_en importable as `import tiny_gpt` etc.
import os
import sys

_LAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LAB_DIR not in sys.path:
    sys.path.insert(0, _LAB_DIR)

# Re-export submodules for convenience
from tests.test_tiny_gpt import *  # noqa: F401,F403,E402
from tests.test_bpe import *  # noqa: F401,F403,E402
from tests.test_trainer import *  # noqa: F401,F403,E402

__all__ = []

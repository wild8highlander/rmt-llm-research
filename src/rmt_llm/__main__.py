"""``python -m rmt_llm`` entry point (also used by the console script)."""
import sys

from rmt_llm.cli import main


if __name__ == "__main__":
    sys.exit(main())

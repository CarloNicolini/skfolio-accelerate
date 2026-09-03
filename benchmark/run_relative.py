"""Entry point: ``python benchmark/run_relative.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.cli import app  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "relative", *sys.argv[1:]]
    app()

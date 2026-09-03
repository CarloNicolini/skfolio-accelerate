"""Entry point: ``python benchmark/run_benchmark.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer  # noqa: E402

from benchmark.cli import run  # noqa: E402

if __name__ == "__main__":
    typer.run(run)

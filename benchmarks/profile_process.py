"""Isolated-process wall-time and process-tree peak-RSS helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil


def _process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil.Error:
        pass
    total = 0
    for child in processes:
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total


def measure_call(function: Callable[[], Any]) -> tuple[Any, float, int]:
    """Measure a call's wall time and peak aggregate RSS, including workers."""
    process = psutil.Process()
    peak_rss = _process_tree_rss(process)
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak_rss
        while not stop.wait(0.002):
            peak_rss = max(peak_rss, _process_tree_rss(process))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        result = function()
    finally:
        wall_s = time.perf_counter() - started
        peak_rss = max(peak_rss, _process_tree_rss(process))
        stop.set()
        sampler.join()
    return result, wall_s, peak_rss


def run_worker(script: Path, arguments: list[str]) -> dict[str, Any]:
    """Run one benchmark case in a fresh interpreter and parse its JSON result."""
    completed = subprocess.run(
        [sys.executable, str(script), "--worker", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"worker produced no result: {completed.stderr}")
    return json.loads(lines[-1])

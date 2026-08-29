"""Git, OS, CPU, BLAS, and package-version metadata for a benchmark run."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from benchmark.config import SCHEMA_VERSION, BenchmarkConfig


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def git_metadata() -> dict[str, str | None]:
    return {
        "git_sha": _run_git(["rev-parse", "HEAD"]),
        "git_sha_short": _run_git(["rev-parse", "--short", "HEAD"]),
        "git_branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": _run_git(["status", "--porcelain"]),
    }


def _cpu_model() -> str | None:
    path = "/proc/cpuinfo"
    if not os.path.exists(path):
        return platform.processor() or None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _core_counts() -> dict[str, int | None]:
    logical = os.cpu_count()
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except ImportError:
        pass
    return {"physical_cores": physical, "logical_cores": logical}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _numpy_blas() -> dict[str, Any]:
    import numpy as np

    info: dict[str, Any] = {"numpy_version": np.__version__}
    try:
        config = np.show_config(mode="dicts")
    except TypeError:
        config = None
    if isinstance(config, dict):
        build = (
            config.get("Build Dependencies") or config.get("build_dependencies") or {}
        )
        blas = build.get("blas") or {}
        lapack = build.get("lapack") or {}
        info["blas_name"] = blas.get("name")
        info["blas_version"] = blas.get("version")
        info["openblas_configuration"] = blas.get("openblas configuration") or blas.get(
            "openblas_configuration"
        )
        info["lapack_name"] = lapack.get("name")
        info["lapack_version"] = lapack.get("version")
    return info


def collect_environment(config: BenchmarkConfig) -> dict[str, Any]:
    """Record host, git, and dependency versions (including skfolio)."""
    git = git_metadata()
    cores = _core_counts()
    packages = {
        "skfolio": _package_version("skfolio"),
        "skfolio-accelerate": _package_version("skfolio-accelerate"),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "cvxpy": _package_version("cvxpy"),
        "clarabel": _package_version("clarabel"),
        "osqp": _package_version("osqp"),
        "highspy": _package_version("highspy"),
        "scs": _package_version("scs"),
        "scikit-learn": _package_version("scikit-learn"),
        "plotly": _package_version("plotly"),
        "kaleido": _package_version("kaleido"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **git,
        "python": sys.version,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        **cores,
        "configured_workers": config.workers,
        "thread_limit": config.thread_limit,
        "n_jobs": config.n_jobs,
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "numpy_blas": _numpy_blas(),
        "packages": packages,
        "config": config.to_dict(),
        "speedup_definition": "native_time / accelerated_time",
        "reported_time": "median of raw repetitions (warm-ups excluded)",
    }

"""Write historical CSV/JSON/summary artifacts without overwriting prior runs."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.metrics import SCHEMA_FIELDS, attach_native_comparisons, format_raw_times

SUMMARY_COLUMNS = [
    ("dataset", "Dataset"),
    ("estimator", "Estimator"),
    ("method", "Method"),
    ("time_s", "Time (s)"),
    ("delta_time_s", "Δ Time (s)"),
    ("relative_time", "Relative Time"),
    ("speedup", "Speed-up"),
    ("mean_sharpe", "Mean Sharpe"),
    ("delta_sharpe", "Δ Sharpe"),
    ("relative_sharpe_error", "Relative Sharpe Error"),
]


def run_directory(root: Path, git_sha_short: str | None) -> Path:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    sha = git_sha_short or "unknown"
    base = root / f"{day}_{sha}"
    path = base
    suffix = 2
    while path.exists():
        path = root / f"{day}_{sha}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cell_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("dataset"),
        row.get("cv"),
        row.get("estimator"),
        row.get("objective"),
        row.get("risk"),
        row.get("extra"),
    )


def apply_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    natives = {_cell_key(row): row for row in rows if row.get("method") == "native"}
    return [
        attach_native_comparisons(
            row, natives.get(_cell_key(row)) if row.get("method") != "native" else row
        )
        for row in rows
    ]


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return format_raw_times([float(v) for v in value])
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            if "raw_times" in payload and "raw_times_s" not in payload:
                payload["raw_times_s"] = payload["raw_times"]
            writer.writerow(
                {key: _csv_value(payload.get(key)) for key in SCHEMA_FIELDS}
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return ""
    return f"{number:.{digits}f}"


def write_summary_md(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    packages = environment.get("packages") or {}
    lines = [
        "# cross_val_predict benchmark summary",
        "",
        f"- timestamp: `{environment.get('timestamp')}`",
        f"- git: `{environment.get('git_sha')}` (`{environment.get('git_branch')}`)",
        f"- skfolio: `{packages.get('skfolio')}`",
        f"- skfolio-accelerate: `{packages.get('skfolio-accelerate')}`",
        f"- python: `{environment.get('python_version')}`",
        f"- cpu: `{environment.get('cpu_model')}`",
        (
            f"- workers / threads: `{environment.get('configured_workers')}`"
            f" / `{environment.get('thread_limit')}`"
        ),
        "- speed-up: `native_time / accelerated_time` (median, no warm-ups)",
        "",
        "| " + " | ".join(title for _, title in SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        cells = []
        for key, _title in SUMMARY_COLUMNS:
            value = row.get(key)
            if key in {
                "time_s",
                "delta_time_s",
                "relative_time",
                "speedup",
                "mean_sharpe",
                "delta_sharpe",
                "relative_sharpe_error",
            }:
                cells.append(_fmt(value, 4 if key != "speedup" else 3))
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_historical_runs(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        return []
    runs = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "results.csv").is_file()
    ]
    return sorted(runs, key=lambda path: path.name)

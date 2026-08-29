"""In-run relative comparison: same host, base ref then head ref."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from benchmark.io import write_json
from benchmark.metrics import _as_float, delta_sharpe, relative_delta_pct

RELATIVE_FIELDS = [
    "dataset",
    "cv",
    "estimator",
    "method",
    "base_time_s",
    "head_time_s",
    "delta_time_s",
    "delta_pct",
    "base_speedup",
    "head_speedup",
    "base_mean_sharpe",
    "head_mean_sharpe",
    "delta_sharpe",
    "base_status",
    "head_status",
]


def row_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("dataset") or ""),
        str(row.get("cv") or ""),
        str(row.get("estimator") or ""),
        str(row.get("method") or ""),
    )


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {row_key(row): row for row in rows}


def compare_in_run_rows(
    base_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair cells from two back-to-back runs and compute ``delta_pct``.

    ``delta_pct = 100 * (head_time - base_time) / base_time``.
    Positive means the head commit is slower than the in-run base.
    """
    base_map = index_rows(base_rows)
    head_map = index_rows(head_rows)
    keys = sorted(set(base_map) | set(head_map))
    compared: list[dict[str, Any]] = []
    for key in keys:
        base = base_map.get(key, {})
        head = head_map.get(key, {})
        base_t = _as_float(base.get("time_s"))
        head_t = _as_float(head.get("time_s"))
        base_s = _as_float(base.get("mean_sharpe"))
        head_s = _as_float(head.get("mean_sharpe"))
        compared.append(
            {
                "dataset": key[0],
                "cv": key[1],
                "estimator": key[2],
                "method": key[3],
                "base_time_s": base_t,
                "head_time_s": head_t,
                "delta_time_s": (
                    head_t - base_t
                    if math.isfinite(base_t) and math.isfinite(head_t)
                    else float("nan")
                ),
                "delta_pct": relative_delta_pct(base_t, head_t),
                "base_speedup": _as_float(base.get("speedup")),
                "head_speedup": _as_float(head.get("speedup")),
                "base_mean_sharpe": base_s,
                "head_mean_sharpe": head_s,
                "delta_sharpe": delta_sharpe(base_s, head_s),
                "base_status": base.get("status") or "missing",
                "head_status": head.get("status") or "missing",
            }
        )
    return compared


def _cell(value: Any, digits: int) -> str:
    number = _as_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def write_relative_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    base_ref: str,
    head_ref: str,
    base_sha: str | None,
    head_sha: str | None,
) -> None:
    lines = [
        "# In-run relative benchmark",
        "",
        f"- base (installed and timed first): `{base_ref}` (`{base_sha}`)",
        f"- head (installed and timed second): `{head_ref}` (`{head_sha}`)",
        "- same host, same Python environment, identical CLI flags",
        "- Δ% = `100 * (head_time - base_time) / base_time` (positive = head slower)",
        "- Do not compare these times to a saved historical CSV from another run.",
        "",
        "| Dataset | Estimator | Method | Base (s) | Head (s) | Δ Time (s) | Δ% |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {estimator} | {method} | {base} | {head} | "
            "{delta} | {pct} |".format(
                dataset=row["dataset"],
                estimator=row["estimator"],
                method=row["method"],
                base=_cell(row["base_time_s"], 4),
                head=_cell(row["head_time_s"], 4),
                delta=_cell(row["delta_time_s"], 4),
                pct=_cell(row["delta_pct"], 2),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_relative_artifacts(
    out_dir: Path,
    *,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    base_ref: str,
    head_ref: str,
    base_sha: str | None,
    head_sha: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RELATIVE_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "delta.json", payload)
    write_relative_summary(
        out_dir / "summary.md",
        rows=rows,
        base_ref=base_ref,
        head_ref=head_ref,
        base_sha=base_sha,
        head_sha=head_sha,
    )

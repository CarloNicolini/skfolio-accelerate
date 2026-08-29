"""Deterministic Plotly figures for the canonical benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.io import list_historical_runs, parse_results_csv

FIGURE_WIDTH = 1200
FIGURE_HEIGHT = 640
FONT_FAMILY = "Inter, system-ui, sans-serif"
FONT_SIZE = 13
MARGIN = dict(t=72, r=32, b=96, l=72)
NATIVE_COLOR = "#4C78A8"
ACCEL_COLOR = "#F58518"
TEMPLATE = "plotly_white"
METHOD_ORDER = ("native", "accelerated")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("status", "")).startswith("ok")
        or row.get("status") == "ok"
        or str(row.get("status", "")) == "ok_with_fallback"
    ]


def _layout(fig, *, title: str, y_title: str, height: int = FIGURE_HEIGHT) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family=FONT_FAMILY)),
        template=TEMPLATE,
        width=FIGURE_WIDTH,
        height=height,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE, color="#172033"),
        margin=MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=y_title,
        xaxis_title="",
        bargap=0.25,
    )
    fig.update_xaxes(tickangle=-40, categoryorder="array")
    fig.update_yaxes(zeroline=True, zerolinecolor="#cbd5e1")


def _grouped_bar(rows: list[dict[str, Any]], *, y_key: str, title: str, y_title: str):
    import plotly.graph_objects as go

    estimators = sorted({str(row["estimator"]) for row in rows})
    datasets = sorted({str(row["dataset"]) for row in rows})
    fig = go.Figure()
    for dataset in datasets:
        for method in METHOD_ORDER:
            ys = []
            for estimator in estimators:
                match = next(
                    (
                        row
                        for row in rows
                        if row.get("dataset") == dataset
                        and row.get("estimator") == estimator
                        and row.get("method") == method
                    ),
                    None,
                )
                ys.append(None if match is None else _finite_float(match.get(y_key)))
            color = NATIVE_COLOR if method == "native" else ACCEL_COLOR
            fig.add_bar(
                name=f"{dataset} / {method}",
                x=estimators,
                y=ys,
                marker_color=color,
                opacity=0.95 if dataset == datasets[0] else 0.55,
            )
    _layout(
        fig,
        title=title,
        y_title=y_title,
        height=max(FIGURE_HEIGHT, 24 * len(estimators) + 200),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=estimators)
    return fig


def figure_execution_time(rows: list[dict[str, Any]]):
    return _grouped_bar(
        _ok_rows(rows),
        y_key="time_s",
        title="Execution time by dataset, estimator, and method (median, s)",
        y_title="Time (s)",
    )


def figure_speedup(rows: list[dict[str, Any]]):
    import plotly.graph_objects as go

    accelerated = [
        row
        for row in _ok_rows(rows)
        if row.get("method") == "accelerated"
        and _finite_float(row.get("speedup")) is not None
    ]
    estimators = sorted({str(row["estimator"]) for row in accelerated})
    datasets = sorted({str(row["dataset"]) for row in accelerated})
    fig = go.Figure()
    palette = ["#0f766e", "#c2410c", "#2563eb"]
    for color, dataset in zip(palette, datasets, strict=False):
        ys = []
        for estimator in estimators:
            match = next(
                (
                    row
                    for row in accelerated
                    if row.get("dataset") == dataset
                    and row.get("estimator") == estimator
                ),
                None,
            )
            ys.append(None if match is None else _finite_float(match.get("speedup")))
        fig.add_bar(name=dataset, x=estimators, y=ys, marker_color=color)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b", annotation_text="1×")
    _layout(
        fig,
        title="Speed-up versus native (native_time / accelerated_time)",
        y_title="Speed-up",
        height=max(FIGURE_HEIGHT, 24 * len(estimators) + 200),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=estimators)
    return fig


def figure_mean_sharpe(rows: list[dict[str, Any]]):
    return _grouped_bar(
        _ok_rows(rows),
        y_key="mean_sharpe",
        title="Mean path Sharpe by dataset, estimator, and method",
        y_title="Mean Sharpe",
    )


def figure_sharpe_difference(rows: list[dict[str, Any]]):
    import plotly.graph_objects as go

    accelerated = [
        row
        for row in _ok_rows(rows)
        if row.get("method") == "accelerated"
        and _finite_float(row.get("delta_sharpe")) is not None
    ]
    estimators = sorted({str(row["estimator"]) for row in accelerated})
    datasets = sorted({str(row["dataset"]) for row in accelerated})
    fig = go.Figure()
    palette = ["#0f766e", "#c2410c"]
    for color, dataset in zip(palette, datasets, strict=False):
        ys = []
        for estimator in estimators:
            match = next(
                (
                    row
                    for row in accelerated
                    if row.get("dataset") == dataset
                    and row.get("estimator") == estimator
                ),
                None,
            )
            ys.append(
                None if match is None else _finite_float(match.get("delta_sharpe"))
            )
        fig.add_bar(name=dataset, x=estimators, y=ys, marker_color=color)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b")
    _layout(
        fig,
        title="Δ Sharpe (accelerated − native)",
        y_title="Δ Sharpe",
        height=max(FIGURE_HEIGHT, 24 * len(estimators) + 200),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=estimators)
    return fig


def figure_historical(results_root: Path, current_rows: list[dict[str, Any]]):
    """Compare speed-ups across previous ``results.csv`` directories when present."""
    import plotly.graph_objects as go

    runs = list_historical_runs(results_root)
    traces: list[tuple[str, list[str], list[float | None]]] = []
    estimators: list[str] = []
    for run in runs[-8:]:
        rows = parse_results_csv(run / "results.csv")
        accelerated = [
            row
            for row in rows
            if row.get("method") == "accelerated"
            and row.get("dataset") == "synthetic"
            and _finite_float(row.get("speedup")) is not None
        ]
        if not accelerated:
            continue
        names = sorted({str(row["estimator"]) for row in accelerated})
        if not estimators:
            estimators = names
        ys = []
        for name in estimators:
            match = next((row for row in accelerated if row["estimator"] == name), None)
            ys.append(None if match is None else _finite_float(match.get("speedup")))
        traces.append((run.name, estimators, ys))
    if current_rows:
        accelerated = [
            row
            for row in current_rows
            if row.get("method") == "accelerated"
            and row.get("dataset") == "synthetic"
            and _finite_float(row.get("speedup")) is not None
        ]
        names = sorted({str(row["estimator"]) for row in accelerated})
        if names:
            estimators = names
            ys = [
                next(
                    (
                        _finite_float(row.get("speedup"))
                        for row in accelerated
                        if row["estimator"] == name
                    ),
                    None,
                )
                for name in estimators
            ]
            traces.append(("current", estimators, ys))
    fig = go.Figure()
    if not traces:
        fig.add_annotation(text="No historical speed-up rows yet", showarrow=False)
        _layout(fig, title="Historical synthetic speed-ups", y_title="Speed-up")
        return fig
    for name, cats, ys in traces:
        fig.add_bar(name=name, x=cats, y=ys)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b")
    _layout(
        fig,
        title="Historical synthetic speed-ups (native_time / accelerated_time)",
        y_title="Speed-up",
        height=max(FIGURE_HEIGHT, 24 * len(estimators) + 220),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=estimators)
    return fig


def write_figure(fig, path: Path) -> list[Path]:
    """Write HTML + Plotly JSON always; SVG/PNG when Kaleido is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    html_path = path.with_suffix(".html")
    json_path = path.with_suffix(".json")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    fig.write_json(json_path)
    written.extend([html_path, json_path])
    for suffix in (".svg", ".png"):
        try:
            fig.write_image(
                path.with_suffix(suffix),
                width=FIGURE_WIDTH,
                height=fig.layout.height or FIGURE_HEIGHT,
            )
        except Exception:
            continue
        written.append(path.with_suffix(suffix))
    return written


def generate_all_figures(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    results_root: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("execution_time", figure_execution_time(rows)),
        ("speedup", figure_speedup(rows)),
        ("mean_sharpe", figure_mean_sharpe(rows)),
        ("sharpe_difference", figure_sharpe_difference(rows)),
        ("historical_speedup", figure_historical(results_root, rows)),
    ]
    written: list[Path] = []
    for name, fig in specs:
        written.extend(write_figure(fig, output_dir / name))
    return written

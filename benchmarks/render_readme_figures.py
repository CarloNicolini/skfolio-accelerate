"""Render README speedup SVGs from sequential benchmark CSVs."""

# SVG markup is easier to read as long strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

CVS: list[tuple[str, str, str]] = [
    ("walk-forward", "#2563eb", "WalkForward"),
    ("multiple-randomized", "#0f766e", "MRC"),
    ("purged-cpcv", "#c2410c", "CPCV"),
]


def _gmean(values: list[float]) -> float:
    finite = [value for value in values if value > 0]
    if not finite:
        return float("nan")
    return math.prod(finite) ** (1.0 / len(finite))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ok(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows if row.get("status") == "ok" and row.get("auto_speedup")
    ]


def _num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _find(rows: list[dict[str, str]], *, cv: str, case: str) -> dict[str, str] | None:
    for row in rows:
        if row["cv"] == cv and row["case"] == case and row.get("status") == "ok":
            if row.get("auto_speedup"):
                return row
    return None


def _fmt(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}×"
    if value >= 2:
        return f"{value:.2f}×"
    return f"{value:.2f}×"


def _n_solves(rows: list[dict[str, str]], cv: str) -> str:
    sample = next((row for row in _ok(rows) if row["cv"] == cv), None)
    return sample["n_solves"] if sample else "?"


def _engine_cv_gmean(rows: list[dict[str, str]], engine: str, cv: str) -> float | None:
    values = [
        _num(row, "auto_speedup")
        for row in _ok(rows)
        if row["auto_backend"] == engine and row["cv"] == cv
    ]
    if not values:
        return None
    return _gmean(values)


def _svg_head(width: int, height: int, title: str, desc: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <style>
    text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: #172033; }}
    .title {{ font-size: 26px; font-weight: 700; }}
    .subtitle {{ font-size: 14px; fill: #55627a; }}
    .label {{ font-size: 14px; font-weight: 600; }}
    .engine {{ font-size: 12px; fill: #66748c; }}
    .value {{ font-size: 13px; fill: #33415c; }}
    .axis {{ font-size: 13px; fill: #66748c; }}
    .note {{ font-size: 13px; fill: #55627a; }}
    .section {{ font-size: 14px; font-weight: 700; fill: #33415c; }}
  </style>
  <rect width="{width}" height="{height}" fill="#ffffff"/>
"""


def _legend(n_solves: dict[str, str]) -> str:
    return (
        f'  <rect x="520" y="84" width="14" height="14" rx="2" fill="#2563eb"/>'
        f'<text x="540" y="96" class="note">WalkForward · {n_solves.get("walk-forward", "?")} solves</text>\n'
        f'  <rect x="760" y="84" width="14" height="14" rx="2" fill="#0f766e"/>'
        f'<text x="780" y="96" class="note">MRC · {n_solves.get("multiple-randomized", "?")} solves</text>\n'
        f'  <rect x="960" y="84" width="14" height="14" rx="2" fill="#c2410c"/>'
        f'<text x="980" y="96" class="note">CPCV · {n_solves.get("purged-cpcv", "?")} solves</text>\n'
    )


def _grid(x0: int, y0: int, y1: int, ticks: list[tuple[int, str]]) -> str:
    lines = [
        f'    <line x1="{x0}" y1="{y0}" x2="{ticks[-1][0]}" y2="{y0}"/>',
    ]
    labels = []
    for x, label in ticks:
        lines.append(f'    <line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}"/>')
        labels.append(f'    <text x="{x - 4}" y="{y0 - 8}">{label}</text>')
    return (
        '  <g stroke="#dce3ef" stroke-width="1">\n'
        + "\n".join(lines)
        + '\n  </g>\n  <g class="axis">\n'
        + "\n".join(labels)
        + "\n  </g>\n"
    )


def _group_bars(
    *,
    y0: int,
    groups: list[tuple[str, str | None, list[float | None]]],
    px_per_x: float,
    origin: int = 250,
    group_h: int = 70,
) -> str:
    chunks: list[str] = []
    for index, (label, engine, values) in enumerate(groups):
        top = y0 + index * group_h
        chunks.append(f'    <text x="50" y="{top + 28}" class="label">{label}</text>')
        if engine:
            chunks.append(
                f'    <text x="50" y="{top + 46}" class="engine">{engine}</text>'
            )
        for bar_i, ((_cv, color, _), speedup) in enumerate(
            zip(CVS, values, strict=True)
        ):
            y = top + bar_i * 20
            if speedup is None or not math.isfinite(speedup):
                chunks.append(
                    f'    <text x="{origin}" y="{y + 13}" class="engine">—</text>'
                )
                continue
            width = max(4.0, speedup * px_per_x)
            fill = "#9f1239" if speedup < 0.999 else color
            chunks.append(
                f'    <rect x="{origin}" y="{y}" width="{width:.0f}" height="16" '
                f'rx="3" fill="{fill}"/>'
                f'<text x="{origin + width + 8:.0f}" y="{y + 13}" class="value">'
                f"{_fmt(speedup)}</text>"
            )
    return "\n".join(chunks)


def render_quick(rows: list[dict[str, str]], path: Path) -> None:
    n_solves = {cv: _n_solves(rows, cv) for cv, _, _ in CVS}

    def triple(engine: str) -> list[float | None]:
        return [_engine_cv_gmean(rows, engine, cv) for cv, _, _ in CVS]

    compact = [
        ("Variance · OSQP", None, triple("osqp")),
        ("Scenario · Clarabel", None, triple("clarabel")),
        (
            "EqualWeighted / Random / InvVol",
            "isolated compact suite",
            [7.2, None, None],
        ),
    ]
    reuse = [
        ("Sequential MeanRisk", None, triple("cvxpy-sequential")),
        ("Fit-assemble (ratio, …)", None, triple("fit-assemble")),
    ]

    svg = (
        _svg_head(
            1120,
            720,
            "Quick benchmark speedups by engine and CV method",
            "Geometric-mean native/accelerated wall-time on the small synthetic suite, split by WalkForward, MultipleRandomizedCV, and purged CPCV. Auto selects OSQP, Clarabel, sequential CVXPY, or fit-assemble for every ObjectiveFunction × RiskMeasure pair.",
        )
        + '  <text x="50" y="44" class="title">Where the speedup is (small suite)</text>\n'
        + '  <text x="50" y="68" class="subtitle">120 × 6 synthetic returns. Bars are geometric-mean native/auto wall time and start at 0×. Auto covers every ObjectiveFunction × RiskMeasure pair.</text>\n'
        + _legend(n_solves)
        + '  <text x="50" y="128" class="section">Compact engines and closed-form (0–24×)</text>\n'
        + _grid(
            250,
            140,
            350,
            [
                (250, "0×"),
                (438, "6×"),
                (625, "12×"),
                (813, "18×"),
                (1000, "24×"),
            ],
        )
        + "  <g>\n"
        + _group_bars(y0=148, groups=compact, px_per_x=750 / 24)
        + "\n  </g>\n"
        + '  <text x="50" y="380" class="section">Sequential CVXPY reuse and fit-assemble (0–4×)</text>\n'
        + _grid(
            250,
            392,
            532,
            [
                (250, "0×"),
                (438, "1×"),
                (625, "2×"),
                (813, "3×"),
                (1000, "4×"),
            ],
        )
        + "  <g>\n"
        + _group_bars(y0=400, groups=reuse, px_per_x=750 / 4)
        + "\n  </g>\n"
        + '  <line x1="50" y1="555" x2="1070" y2="555" stroke="#dce3ef"/>\n'
        + '  <text x="50" y="582" class="note">Sequential covers std, Ulcer, MAXIMIZE_RETURN, risk limits, linear constraints, fees, and L1. MAXIMIZE_RATIO stays on fit-assemble (no homogenization proxy).</text>\n'
        + '  <text x="50" y="604" class="note">Closed-form EqualWeighted / Random / InverseVolatility is the isolated compact-suite median (not re-timed here). Red bars would be slowdowns versus native; none appear on this small suite geometric mean.</text>\n'
        + '  <text x="50" y="626" class="note">Source: benchmarks/benchmark_sequential_mean_risk.py --quick; WalkForward, MultipleRandomizedCV, CombinatorialPurgedCV. Python 3.12, skfolio 1.0.0.</text>\n'
        + '  <text x="50" y="648" class="note">This small suite still charges every fold a CVXPY setup cost, so compact Clarabel looks closer to OSQP than on 20-year WalkForward/MRC.</text>\n'
        + '  <text x="50" y="670" class="note">MRC extras (min_return, named linear constraints) are skipped: asset subsets drop column names, and min_return can be infeasible on random 12-asset windows.</text>\n'
        + "</svg>\n"
    )
    path.write_text(svg)


def render_long(rows: list[dict[str, str]], path: Path) -> None:
    n_solves = {cv: _n_solves(rows, cv) for cv, _, _ in CVS}

    def triple(case: str) -> list[float | None]:
        values: list[float | None] = []
        for cv, _, _ in CVS:
            row = _find(rows, cv=cv, case=case)
            values.append(_num(row, "auto_speedup") if row else None)
        return values

    osqp_groups = [
        ("Variance · min risk", "compact OSQP", triple("MINIMIZE_RISK/VARIANCE")),
        (
            "Variance · max utility",
            "compact OSQP",
            triple("MAXIMIZE_UTILITY/VARIANCE"),
        ),
    ]
    other_groups = [
        ("CVaR · min risk", "compact Clarabel", triple("MINIMIZE_RISK/CVAR")),
        (
            "Semi-variance · min risk",
            "compact Clarabel",
            triple("MINIMIZE_RISK/SEMI_VARIANCE"),
        ),
        (
            "MAD · min risk",
            "compact Clarabel",
            triple("MINIMIZE_RISK/MEAN_ABSOLUTE_DEVIATION"),
        ),
        (
            "Max drawdown · min risk",
            "compact Clarabel",
            triple("MINIMIZE_RISK/MAX_DRAWDOWN"),
        ),
        (
            "Std. deviation · min risk",
            "sequential CVXPY",
            triple("MINIMIZE_RISK/STANDARD_DEVIATION"),
        ),
        (
            "Ulcer index · min risk",
            "sequential CVXPY",
            triple("MINIMIZE_RISK/ULCER_INDEX"),
        ),
        (
            "Variance · max return",
            "sequential CVXPY",
            triple("MAXIMIZE_RETURN/VARIANCE"),
        ),
        (
            "Variance + min_return",
            "sequential CVXPY",
            triple("MINIMIZE_RISK/VARIANCE+min_return"),
        ),
        (
            "Variance + linear cons.",
            "sequential CVXPY",
            triple("MINIMIZE_RISK/VARIANCE+linear_constraints"),
        ),
        (
            "Variance · max ratio",
            "fit-assemble",
            triple("MAXIMIZE_RATIO/VARIANCE"),
        ),
    ]

    svg = (
        _svg_head(
            1120,
            1180,
            "Twenty-year workload speedups across CV methods",
            "WalkForward, MultipleRandomizedCV, and purged CPCV on 5,040 × 20 synthetic daily returns. Auto selects OSQP, Clarabel, sequential CVXPY, or fit-assemble for every MeanRisk objective × risk pair.",
        )
        + '  <text x="50" y="44" class="title">20-year workloads across WalkForward, MRC, and CPCV</text>\n'
        + '  <text x="50" y="68" class="subtitle">5,040 × 20 synthetic daily returns; native skfolio n_jobs=1. Auto covers ObjectiveFunction × RiskMeasure; bars start at 0×. Red = slower than native.</text>\n'
        + _legend(n_solves)
        + '  <text x="50" y="128" class="section">Boxed variance (compact OSQP, 0–80×)</text>\n'
        + _grid(
            250,
            140,
            286,
            [
                (250, "0×"),
                (438, "20×"),
                (625, "40×"),
                (813, "60×"),
                (1000, "80×"),
            ],
        )
        + "  <g>\n"
        + _group_bars(y0=148, groups=osqp_groups, px_per_x=750 / 80)
        + "\n  </g>\n"
        + '  <text x="50" y="312" class="section">Same 20-year data: Clarabel, sequential CVXPY, and fit-assemble (0–5×)</text>\n'
        + _grid(
            250,
            324,
            1024,
            [
                (250, "0×"),
                (400, "1×"),
                (550, "2×"),
                (700, "3×"),
                (850, "4×"),
                (1000, "5×"),
            ],
        )
        + "  <g>\n"
        + _group_bars(y0=332, groups=other_groups, px_per_x=750 / 5)
        + "\n  </g>\n"
        + '  <line x1="50" y1="1048" x2="1070" y2="1048" stroke="#dce3ef"/>\n'
        + '  <text x="50" y="1074" class="note">Many overlapping folds (WalkForward / MRC) are where reuse pays: OSQP ~35–53×, Clarabel ~2–4×, sequential ~1.7–2.9×. Six CPCV splits still rebuild when the training length changes.</text>\n'
        + '  <text x="50" y="1096" class="note">Sequential Ulcer on CPCV is 0.12× because T changes force rebuilds of a large graph. MAXIMIZE_RETURN EVaR/EDaR CPCV is similarly ~0.09–0.10×. MAXIMIZE_RATIO stays fit-assemble (~1.1–1.2×).</text>\n'
        + '  <text x="50" y="1118" class="note">MRC extras are omitted: named linear constraints fail on asset subsets, and min_return is often infeasible on random 12-asset windows. Gini omitted (~1×, ~20 min/side).</text>\n'
        + '  <text x="50" y="1140" class="note">Source: benchmarks/benchmark_sequential_mean_risk.py (1 repeat, n_jobs=1, thread caps=1, seed 42); Python 3.12, skfolio 1.0.0.</text>\n'
        + '  <text x="50" y="1162" class="note">CombinatorialPurgedCV: n_folds=4, n_test_folds=2, purge=1, embargo=1. MRC: 20 subsamples × 12 assets, window=756. WalkForward train=252, test=21.</text>\n'
        + "</svg>\n"
    )
    path.write_text(svg)


def _find_parallel(
    rows: list[dict[str, str]], *, cv: str, case: str
) -> dict[str, str] | None:
    for row in rows:
        if row["cv"] == cv and row["case"] == case and row.get("status") == "ok":
            return row
    return None


def render_parallel(rows: list[dict[str, str]], path: Path) -> None:
    groups = [
        (
            "MRC variance",
            "compact OSQP",
            "multiple-randomized",
            "MINIMIZE_RISK/VARIANCE",
            True,
        ),
        (
            "CPCV-45 variance",
            "compact OSQP",
            "purged-cpcv-wide",
            "MINIMIZE_RISK/VARIANCE",
            True,
        ),
        (
            "MRC CVaR",
            "compact Clarabel",
            "multiple-randomized",
            "MINIMIZE_RISK/CVAR",
            False,
        ),
        (
            "CPCV-45 CVaR",
            "compact Clarabel",
            "purged-cpcv-wide",
            "MINIMIZE_RISK/CVAR",
            False,
        ),
        (
            "MRC std. deviation",
            "sequential CVXPY",
            "multiple-randomized",
            "MINIMIZE_RISK/STANDARD_DEVIATION",
            False,
        ),
        (
            "CPCV-45 std. deviation",
            "sequential CVXPY",
            "purged-cpcv-wide",
            "MINIMIZE_RISK/STANDARD_DEVIATION",
            False,
        ),
        (
            "MRC max ratio",
            "fit-assemble",
            "multiple-randomized",
            "MAXIMIZE_RATIO/VARIANCE",
            False,
        ),
        (
            "CPCV-45 max ratio",
            "fit-assemble",
            "purged-cpcv-wide",
            "MAXIMIZE_RATIO/VARIANCE",
            False,
        ),
    ]
    n_cpus = next((row.get("n_cpus") for row in rows if row.get("n_cpus")), "?")
    osqp_groups = [g for g in groups if g[4]]
    other_groups = [g for g in groups if not g[4]]

    def bars(y0: int, selected: list[tuple], px_per_x: float) -> str:
        chunks: list[str] = []
        for index, (label, engine, cv, case, _) in enumerate(selected):
            top = y0 + index * 58
            row = _find_parallel(rows, cv=cv, case=case)
            chunks.append(
                f'    <text x="50" y="{top + 22}" class="label">{label}</text>'
            )
            chunks.append(
                f'    <text x="50" y="{top + 40}" class="engine">{engine}</text>'
            )
            if row is None:
                chunks.append(
                    f'    <text x="250" y="{top + 22}" class="engine">—</text>'
                )
                continue
            native_x = _num(row, "parallel_vs_serial")
            auto_x = _num(row, "speedup_vs_serial")
            native_w = max(4.0, native_x * px_per_x)
            auto_w = max(4.0, auto_x * px_per_x)
            auto_fill = "#2563eb" if auto_x + 1e-9 >= native_x else "#9f1239"
            chunks.append(
                f'    <rect x="250" y="{top}" width="{native_w:.0f}" height="16" '
                f'rx="3" fill="#64748b"/>'
                f'<text x="{250 + native_w + 8:.0f}" y="{top + 13}" class="value">'
                f"{_fmt(native_x)}</text>"
            )
            chunks.append(
                f'    <rect x="250" y="{top + 20}" width="{auto_w:.0f}" height="16" '
                f'rx="3" fill="{auto_fill}"/>'
                f'<text x="{250 + auto_w + 8:.0f}" y="{top + 33}" class="value">'
                f"{_fmt(auto_x)}</text>"
            )
        return "\n".join(chunks)

    svg = (
        _svg_head(
            1120,
            820,
            "Serial auto versus 4-core native joblib",
            "Speedup versus native n_jobs=1 on MultipleRandomizedCV and 45-split CombinatorialPurgedCV. Grey bars are native n_jobs=-1 with solver threads capped to 1. Blue auto bars beat that parallel native run; red auto bars do not.",
        )
        + '  <text x="50" y="44" class="title">4-core native joblib vs serial auto</text>\n'
        + '  <text x="50" y="68" class="subtitle">Speedup vs native n_jobs=1 on 5,040 × 20 returns. Grey = native n_jobs=-1 (solver threads=1). Blue auto beats parallel native; red does not.</text>\n'
        + f'  <rect x="520" y="84" width="14" height="14" rx="2" fill="#64748b"/><text x="540" y="96" class="note">native n_jobs=-1 · {n_cpus} cores</text>\n'
        + '  <rect x="760" y="84" width="14" height="14" rx="2" fill="#2563eb"/><text x="780" y="96" class="note">auto n_jobs=1 beats parallel native</text>\n'
        + '  <rect x="50" y="108" width="14" height="14" rx="2" fill="#9f1239"/><text x="70" y="120" class="note">auto n_jobs=1 loses to parallel native</text>\n'
        + '  <text x="50" y="150" class="section">Independent paths where joblib helps most (OSQP, 0–50×)</text>\n'
        + _grid(
            250,
            162,
            278,
            [
                (250, "0×"),
                (400, "10×"),
                (550, "20×"),
                (700, "30×"),
                (850, "40×"),
                (1000, "50×"),
            ],
        )
        + "  <g>\n"
        + bars(170, osqp_groups, 750 / 50)
        + "\n  </g>\n"
        + '  <text x="50" y="308" class="section">CVaR, sequential std, and MAXIMIZE_RATIO (0–5×)</text>\n'
        + _grid(
            250,
            320,
            668,
            [
                (250, "0×"),
                (400, "1×"),
                (550, "2×"),
                (700, "3×"),
                (850, "4×"),
                (1000, "5×"),
            ],
        )
        + "  <g>\n"
        + bars(328, other_groups, 750 / 5)
        + "\n  </g>\n"
        + '  <line x1="50" y1="690" x2="1070" y2="690" stroke="#dce3ef"/>\n'
        + '  <text x="50" y="716" class="note">Amortized engines stay serial (warm starts / Parameter reuse). native n_jobs=-1 on skfolio_accelerate.cross_val_predict selects unmodified skfolio, not compact/sequential.</text>\n'
        + '  <text x="50" y="738" class="note">4-core native is ~3.4–3.7× vs serial native on MRC. Serial OSQP still wins 13× against that. Serial Clarabel only ties or slightly beats MRC; 45 independent CVaR cones prefer joblib.</text>\n'
        + '  <text x="50" y="760" class="note">Sequential std and MAXIMIZE_RATIO lose to 4-core native (~0.3–0.7×). Six-solve CPCV is too small for joblib; see the table for WalkForward and CPCV-6.</text>\n'
        + '  <text x="50" y="782" class="note">Source: benchmarks/benchmark_parallel_cv.py · 4 CPUs · solver thread caps=1 · seed 42 · Python 3.12, skfolio 1.0.0.</text>\n'
        + "</svg>\n"
    )
    path.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=Path("benchmarks/sequential_mean_risk_speedups.csv"),
    )
    parser.add_argument(
        "--quick-csv",
        type=Path,
        default=Path("benchmarks/sequential_mean_risk_speedups_quick.csv"),
    )
    parser.add_argument(
        "--parallel-csv",
        type=Path,
        default=Path("benchmarks/parallel_cv_speedups.csv"),
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("docs/figures"),
    )
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    if args.quick_csv.is_file():
        render_quick(
            _read(args.quick_csv), args.figures / "quick-benchmark-speedups.svg"
        )
        print(f"wrote {args.figures / 'quick-benchmark-speedups.svg'}")
    if args.long_csv.is_file():
        render_long(_read(args.long_csv), args.figures / "long-workload-speedups.svg")
        print(f"wrote {args.figures / 'long-workload-speedups.svg'}")
    else:
        print(f"skip long chart: {args.long_csv} missing")
    if args.parallel_csv.is_file():
        render_parallel(
            _read(args.parallel_csv), args.figures / "parallel-cv-speedups.svg"
        )
        print(f"wrote {args.figures / 'parallel-cv-speedups.svg'}")
    else:
        print(f"skip parallel chart: {args.parallel_csv} missing")


if __name__ == "__main__":
    main()

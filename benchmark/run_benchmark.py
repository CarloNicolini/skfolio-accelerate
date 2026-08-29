"""Canonical native vs accelerated ``cross_val_predict`` benchmark.

Configuration
-------------
All numeric defaults live in :data:`benchmark.config.CONFIG`. Override them
here only through the CLI.

Examples
--------
::

    python benchmark/run_relative.py --base origin/main --quick --workers 1
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --dataset synthetic
    python benchmark/run_benchmark.py --dataset sp500
    python benchmark/run_benchmark.py --method native
    python benchmark/run_benchmark.py --method accelerated
    python benchmark/run_benchmark.py --repetitions 5
    python benchmark/run_benchmark.py --workers 1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.config import (  # noqa: E402
    DATASETS,
    METHODS,
    SCHEMA_VERSION,
    build_config,
)
from benchmark.datasets import load_dataset  # noqa: E402
from benchmark.environment import collect_environment, git_metadata  # noqa: E402
from benchmark.estimators import mean_risk_specs  # noqa: E402
from benchmark.figures import generate_all_figures  # noqa: E402
from benchmark.io import (  # noqa: E402
    apply_comparisons,
    run_directory,
    write_csv,
    write_json,
    write_summary_md,
)
from benchmark.metrics import format_raw_times  # noqa: E402
from benchmark.protocol import (  # noqa: E402
    apply_thread_limits,
    empty_row_base,
    fold_index_fingerprint,
    make_cv,
    run_method_cell,
    skip_extra_for_cv,
)

BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = BENCHMARK_ROOT / "results"
FIGURES_ROOT = BENCHMARK_ROOT / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical skfolio vs skfolio-accelerate cross_val_predict benchmark"
        )
    )
    parser.add_argument("--dataset", action="append", choices=DATASETS, default=[])
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument(
        "--cv",
        action="append",
        dest="cv_kinds",
        default=[],
        help="walk-forward, multiple-randomized, purged-cpcv (repeatable)",
    )
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--thread-limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None, dest="timeout_s")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--include-gini", action="store_true")
    parser.add_argument("--include-annualized", action="store_true")
    parser.add_argument("--skip-extras", action="store_true")
    parser.add_argument("--skip-lp-l2-zero", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="write this run into DIR instead of a dated results folder",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace):
    if args.quick and args.full:
        raise SystemExit("choose at most one of --quick or --full")
    preset = "quick" if args.quick else ("full" if args.full else None)
    overrides = {
        "repetitions": args.repetitions,
        "workers": args.workers,
        "warmups": args.warmups,
        "thread_limit": args.thread_limit
        if args.thread_limit is not None
        else args.workers,
        "timeout_s": args.timeout_s,
        "n_jobs": args.n_jobs,
        "include_gini": True if args.include_gini else None,
        "include_annualized": True if args.include_annualized else None,
        "include_extras": False if args.skip_extras else None,
        "include_lp_l2_zero": False if args.skip_lp_l2_zero else None,
    }
    datasets = tuple(args.dataset) if args.dataset else DATASETS
    methods = tuple(args.method) if args.method else METHODS
    cv_kinds = tuple(args.cv_kinds) if args.cv_kinds else None
    return build_config(
        overrides,
        datasets=datasets,
        methods=methods,
        cv_kinds=cv_kinds,
        preset=preset,
    )


def _flatten_cell(result: dict[str, Any]) -> dict[str, Any]:
    raw = list(result.get("raw_times") or [])
    return {
        "method": result.get("method"),
        "backend": result.get("backend"),
        "reason": result.get("reason"),
        "fallback_reason": result.get("fallback_reason"),
        "n_solves": result.get("n_solves"),
        "n_warm_starts": result.get("n_warm_starts"),
        "n_rebuilds": result.get("n_rebuilds"),
        "n_prior_fits": result.get("n_prior_fits"),
        "n_prior_updates": result.get("n_prior_updates"),
        "time_s": result.get("time_s"),
        "time_s_mean": result.get("time_s_mean"),
        "time_s_std": result.get("time_s_std"),
        "time_s_min": result.get("time_s_min"),
        "time_s_max": result.get("time_s_max"),
        "n_repetitions": result.get("n_repetitions"),
        "raw_times_s": format_raw_times(raw),
        "raw_times": raw,
        "mean_sharpe": result.get("mean_sharpe"),
        "max_abs_sharpe_diff": result.get("max_abs_sharpe_diff"),
        "n_failed_folds": result.get("n_failed_folds"),
        "n_invalid_outputs": result.get("n_invalid_outputs"),
        "n_nonfinite_weights": result.get("n_nonfinite_weights"),
        "max_abs_weight_diff": result.get("max_abs_weight_diff"),
        "solver_status": result.get("solver_status"),
        "validation_ok": result.get("validation_ok"),
        "cache_warning": result.get("cache_warning"),
        "status": result.get("status"),
        "error": result.get("error"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    apply_thread_limits(config)

    from skfolio_accelerate.predict import AccelerationWarning

    warnings.filterwarnings("default", category=AccelerationWarning)

    git = git_metadata()
    environment = collect_environment(config)
    specs = mean_risk_specs(config)
    datasets = [load_dataset(name, config) for name in config.datasets]

    print(
        f"skfolio={environment['packages']['skfolio']}  "
        f"datasets={list(config.datasets)}  methods={list(config.methods)}  "
        f"cv={list(config.cv_kinds)}  estimators={len(specs)}  "
        f"reps={config.repetitions} warmups={config.warmups}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    timestamp = environment["timestamp"]
    for dataset in datasets:
        for cv_kind in config.cv_kinds:
            fingerprint = fold_index_fingerprint(dataset.X, make_cv(cv_kind, config))
            n_folds = len(fingerprint)
            print(
                f"== {dataset.name} {dataset.X.shape[0]}×{dataset.X.shape[1]} "
                f"{cv_kind} folds={n_folds} ==",
                flush=True,
            )
            for spec in specs:
                if skip_extra_for_cv(spec, cv_kind):
                    continue
                asset_names = [str(col) for col in dataset.X.columns]
                estimator = spec.factory(asset_names=asset_names)
                native_pred = None
                methods = tuple(name for name in METHODS if name in config.methods)
                for method in methods:
                    cell = run_method_cell(
                        method=method,
                        estimator=estimator,
                        X=dataset.X,
                        cv_factory=lambda kind=cv_kind: make_cv(kind, config),
                        config=config,
                    )
                    if method == "native":
                        native_pred = cell.get("prediction")
                    elif native_pred is not None and cell.get("prediction") is not None:
                        from benchmark.protocol import validate_prediction

                        extra = validate_prediction(
                            cell["prediction"],
                            report=cell.get("report"),
                            reference_prediction=native_pred,
                        )
                        cell.update(extra)
                    row = empty_row_base(
                        timestamp=timestamp,
                        git_sha=git.get("git_sha"),
                        git_branch=git.get("git_branch"),
                        dataset=dataset,
                        cv_kind=cv_kind,
                        n_folds=n_folds,
                        spec=spec,
                        config=config,
                    )
                    row.update(_flatten_cell(cell))
                    rows.append(row)
                    time_s = cell.get("time_s")
                    try:
                        time_txt = f"{float(time_s):.4f}s"
                    except (TypeError, ValueError):
                        time_txt = "nan"
                    print(
                        f"{dataset.name:<10} {cv_kind:<22} {spec.name:<44} "
                        f"{method:<12} {cell.get('status')} "
                        f"{time_txt} "
                        f"{cell.get('backend') or ''}",
                        flush=True,
                    )

    rows = apply_comparisons(rows)
    if args.output_dir is not None:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = run_directory(RESULTS_ROOT, git.get("git_sha_short"))
    write_csv(out_dir / "results.csv", rows)
    write_json(
        out_dir / "results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "environment": environment,
            "config": config.to_dict(),
            "rows": rows,
        },
    )
    write_json(out_dir / "environment.json", environment)
    write_summary_md(out_dir / "summary.md", rows=rows, environment=environment)
    shutil.copy(BENCHMARK_ROOT / "ARCHITECTURE.md", out_dir / "ARCHITECTURE.md")

    figure_paths: list[Path] = []
    if not args.no_figures:
        try:
            figure_paths = generate_all_figures(
                rows, output_dir=out_dir / "figures", results_root=RESULTS_ROOT
            )
            if args.output_dir is None:
                FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
                for path in figure_paths:
                    shutil.copy(path, FIGURES_ROOT / path.name)
        except ImportError as error:
            print(f"Skipping figures (install plotly): {error}", flush=True)

    print(f"Wrote {out_dir} ({len(rows)} rows)", flush=True)
    for path in figure_paths:
        print(f"  figure {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

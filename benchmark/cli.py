"""Typer CLI: `run` (one commit) and `relative` (base then head)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from benchmark.config import (
    DATASETS,
    METHODS,
    SCHEMA_VERSION,
    build_config,
    load_dataset,
    make_estimator,
    mean_risk_specs,
)
from benchmark.harness import (
    apply_comparisons,
    apply_thread_limits,
    as_float,
    collect_environment,
    compare_in_run_rows,
    fold_index_fingerprint,
    format_raw_times,
    git_metadata,
    make_cv,
    parse_results_csv,
    run_method_cell,
    unique_run_dir,
    validate_prediction,
    write_csv,
    write_json,
    write_relative_artifacts,
    write_summary_md,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = BENCHMARK_ROOT / "results"
RELATIVE_ROOT = RESULTS_ROOT / "relative"
SKIP_CELL = {"prediction", "report", "expected_backend"}

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _build(
    dataset: list[str] | None,
    method: list[str] | None,
    cv: list[str] | None,
    repetitions: int | None,
    workers: int | None,
    warmups: int | None,
    thread_limit: int | None,
    timeout: float | None,
    n_jobs: int | None,
    quick: bool,
    full: bool,
    include_gini: bool,
    include_annualized: bool,
    skip_extras: bool,
    skip_lp_l2_zero: bool,
):
    if quick and full:
        raise typer.BadParameter("choose at most one of --quick or --full")
    return build_config(
        {
            "repetitions": repetitions,
            "workers": workers,
            "warmups": warmups,
            "thread_limit": thread_limit if thread_limit is not None else workers,
            "timeout_s": timeout,
            "n_jobs": n_jobs,
            "include_gini": True if include_gini else None,
            "include_annualized": True if include_annualized else None,
            "include_extras": False if skip_extras else None,
            "include_lp_l2_zero": False if skip_lp_l2_zero else None,
        },
        datasets=tuple(dataset) if dataset else DATASETS,
        methods=tuple(method) if method else METHODS,
        cv_kinds=tuple(cv) if cv else None,
        preset="quick" if quick else ("full" if full else None),
    )


def run_sweep(config, output_dir: Path | None) -> Path:
    apply_thread_limits(config)
    from skfolio_accelerate.predict import AccelerationWarning

    warnings.filterwarnings("default", category=AccelerationWarning)
    git = git_metadata()
    environment = collect_environment(config)
    specs = mean_risk_specs(config)
    print(
        f"skfolio={environment['packages']['skfolio']}  "
        f"datasets={list(config.datasets)}  methods={list(config.methods)}  "
        f"cv={list(config.cv_kinds)}  estimators={len(specs)}  "
        f"reps={config.repetitions} warmups={config.warmups}",
        flush=True,
    )
    rows = []
    timestamp = environment["timestamp"]
    for dataset in [load_dataset(name, config) for name in config.datasets]:
        n_obs = dataset.X.shape[0]
        for cv_kind in config.cv_kinds:
            n_folds = len(
                fold_index_fingerprint(dataset.X, make_cv(cv_kind, config, n_obs))
            )
            print(
                f"== {dataset.name} {dataset.X.shape[0]}×{dataset.X.shape[1]} "
                f"{cv_kind} folds={n_folds} ==",
                flush=True,
            )
            for spec in specs:
                if (
                    spec.extra
                    and spec.extra != "l2_0"
                    and cv_kind == "multiple-randomized"
                ):
                    continue
                estimator = make_estimator(
                    spec, [str(col) for col in dataset.X.columns]
                )
                native_pred = None
                for method in (name for name in METHODS if name in config.methods):
                    cell = run_method_cell(
                        method=method,
                        estimator=estimator,
                        X=dataset.X,
                        cv_kind=cv_kind,
                        config=config,
                    )
                    if method == "native":
                        native_pred = cell.get("prediction")
                    elif native_pred is not None and cell.get("prediction") is not None:
                        cell.update(
                            validate_prediction(
                                cell["prediction"],
                                report=cell.get("report"),
                                reference_prediction=native_pred,
                            )
                        )
                    raw = list(cell.get("raw_times") or [])
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "timestamp": timestamp,
                        "git_sha": git.get("git_sha"),
                        "git_branch": git.get("git_branch"),
                        "dataset": dataset.name,
                        "n_observations": int(dataset.X.shape[0]),
                        "n_assets": int(dataset.X.shape[1]),
                        "cv": cv_kind,
                        "n_folds": n_folds,
                        "estimator": spec.name,
                        "objective": spec.objective,
                        "risk": spec.risk,
                        "extra": spec.extra,
                        "solver": config.solver,
                        "workers": config.workers,
                        "thread_limit": config.thread_limit,
                        "n_jobs": config.n_jobs,
                        **{k: v for k, v in cell.items() if k not in SKIP_CELL},
                        "raw_times_s": format_raw_times(raw),
                        "raw_times": raw,
                    }
                    rows.append(row)
                    try:
                        time_txt = f"{float(cell.get('time_s')):.4f}s"
                    except (TypeError, ValueError):
                        time_txt = "nan"
                    print(
                        f"{dataset.name:<10} {cv_kind:<22} {spec.name:<44} "
                        f"{method:<12} {cell.get('status')} {time_txt} "
                        f"{cell.get('backend') or ''}",
                        flush=True,
                    )
    rows = apply_comparisons(rows)
    if output_dir is not None:
        out_dir = output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = unique_run_dir(RESULTS_ROOT, git.get("git_sha_short"))
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
    print(f"Wrote {out_dir} ({len(rows)} rows)", flush=True)
    return out_dir


def run(
    dataset: Annotated[
        list[str] | None, typer.Option(help="Repeatable: synthetic, sp500")
    ] = None,
    method: Annotated[
        list[str] | None, typer.Option(help="Repeatable: native, accelerated")
    ] = None,
    cv: Annotated[list[str] | None, typer.Option(help="Repeatable CV kind")] = None,
    repetitions: int | None = None,
    workers: int | None = None,
    warmups: int | None = None,
    thread_limit: int | None = None,
    timeout: float | None = None,
    n_jobs: int | None = None,
    quick: bool = False,
    full: bool = False,
    include_gini: bool = False,
    include_annualized: bool = False,
    skip_extras: bool = False,
    skip_lp_l2_zero: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Native vs accelerated cross_val_predict on this checkout."""
    run_sweep(
        _build(
            dataset,
            method,
            cv,
            repetitions,
            workers,
            warmups,
            thread_limit,
            timeout,
            n_jobs,
            quick,
            full,
            include_gini,
            include_annualized,
            skip_extras,
            skip_lp_l2_zero,
        ),
        output_dir,
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), check=True, capture_output=True, text=True
    ).stdout.strip()


def relative(
    ctx: typer.Context,
    base: str = "origin/main",
    head: str = "HEAD",
    fail_on_slow_pct: float | None = None,
) -> None:
    """Install and time base, then HEAD. Extra flags are forwarded to `run`."""
    if "--output-dir" in ctx.args:
        raise typer.BadParameter("--output-dir is reserved for relative internals")

    def resolve(ref: str) -> str:
        try:
            return _git("rev-parse", "--verify", ref)
        except subprocess.CalledProcessError:
            _git("fetch", "--depth", "1", "origin", ref.removeprefix("origin/"))
            return _git("rev-parse", "--verify", ref)

    head_sha = resolve(head)
    try:
        base_sha = resolve(base)
    except subprocess.CalledProcessError as error:
        raise typer.Exit(
            f"cannot resolve base ref {base!r}; fetch main first"
        ) from error
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_root = RELATIVE_ROOT / f"{stamp}_{head_sha[:7]}"
    suffix = 2
    while out_root.exists():
        out_root = RELATIVE_ROOT / f"{stamp}_{head_sha[:7]}_{suffix}"
        suffix += 1
    worktree = Path(tempfile.gettempdir()) / f"skfolio-base-{base_sha[:12]}"
    runner = BENCHMARK_ROOT / "run_benchmark.py"

    def install(src: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(src),
                "--no-deps",
                "--quiet",
            ],
            check=True,
        )

    def remove_worktree() -> None:
        shutil.rmtree(worktree / ".git", ignore_errors=True)
        try:
            _git("worktree", "remove", "--force", str(worktree))
        except subprocess.CalledProcessError:
            shutil.rmtree(worktree, ignore_errors=True)

    if worktree.exists():
        remove_worktree()
    try:
        _git("worktree", "add", "--detach", str(worktree), base_sha)
        print(f"== base {base} {base_sha} ==", flush=True)
        install(worktree)
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--output-dir",
                str(out_root / "base"),
                *ctx.args,
            ],
            check=True,
            cwd=str(ROOT),
        )
        print(f"== head {head} {head_sha} ==", flush=True)
        install(ROOT)
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--output-dir",
                str(out_root / "head"),
                *ctx.args,
            ],
            check=True,
            cwd=str(ROOT),
        )
    finally:
        remove_worktree()
        install(ROOT)

    delta_rows = compare_in_run_rows(
        parse_results_csv(out_root / "base" / "results.csv"),
        parse_results_csv(out_root / "head" / "results.csv"),
    )
    write_relative_artifacts(
        out_root,
        delta_rows,
        {
            "kind": "in-run-relative",
            "base_ref": base,
            "head_ref": head,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "delta_pct_definition": "100 * (head_time - base_time) / base_time",
            "forwarded_flags": ctx.args,
            "rows": delta_rows,
        },
        {
            "base_ref": base,
            "head_ref": head,
            "base_sha": base_sha,
            "head_sha": head_sha,
        },
    )
    print(f"Wrote {out_root}", flush=True)
    if fail_on_slow_pct is None:
        return
    slow = [
        row
        for row in delta_rows
        if row.get("base_status") == "ok"
        and row.get("head_status") == "ok"
        and as_float(row.get("delta_pct")) > fail_on_slow_pct
    ]
    if slow:
        print(
            f"{len(slow)} cells slower than {fail_on_slow_pct}% vs in-run base",
            flush=True,
        )
        raise typer.Exit(code=2)


app.command("run")(run)
app.command(
    "relative",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(relative)


if __name__ == "__main__":
    app()

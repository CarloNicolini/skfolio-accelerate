"""COSMO.rs persistence experiment (not the canonical auto-vs-native suite).

This driver answers whether a persistent COSMO.rs workspace accelerates
MeanRisk walk-forward relative to:

* native skfolio + Clarabel
* ``backend="auto"`` (OSQP / HiGHS / Clarabel)
* a cold COSMO.rs solve on every fold

It is **not** a PR-vs-main relative benchmark. Do not paste these seconds
against ``benchmark/results/YYYY-MM-DD_*`` from another machine. See
``AGENTS.md``.

Outputs land in ``benchmark/results/cosmo/<date>_<sha>_<cosmo-sha>/``
and are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from skfolio import RiskMeasure  # noqa: E402
from skfolio.model_selection import (  # noqa: E402
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import (  # noqa: E402
    cross_val_predict as skfolio_cv_predict,
)
from skfolio.optimization import MeanRisk  # noqa: E402

from benchmark.environment import git_metadata  # noqa: E402
from skfolio_accelerate import cross_val_predict, path_sharpes  # noqa: E402
from skfolio_accelerate._cosmo import cosmo_available, make_cosmo_engine  # noqa: E402
from skfolio_accelerate.compact import estimator_spec  # noqa: E402
from skfolio_accelerate.flagship import factor_returns  # noqa: E402
from skfolio_accelerate.formulations import (  # noqa: E402
    formulation_record,
    to_markdown,
)
from skfolio_accelerate.moments import empirical_from_window  # noqa: E402

RISKS = (
    RiskMeasure.VARIANCE,
    RiskMeasure.SEMI_DEVIATION,
    RiskMeasure.CVAR,
    RiskMeasure.MAX_DRAWDOWN,
)
MODES = ("cold", "warm_x", "warm_xy", "persist_factor", "persist_full")


def _cosmo_rs_git() -> dict[str, str | None]:
    """Best-effort COSMO.rs revision from the installed package checkout."""
    try:
        import cosmo_rs
    except ImportError:
        return {"cosmo_rs_git_sha": None, "cosmo_rs_git_sha_short": None}
    path = Path(cosmo_rs.__file__).resolve()
    for parent in path.parents:
        git_dir = parent / ".git"
        if git_dir.exists():
            try:
                sha = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                ).strip()
            except Exception:
                break
            return {
                "cosmo_rs_git_sha": sha,
                "cosmo_rs_git_sha_short": sha[:7],
            }
    return {"cosmo_rs_git_sha": None, "cosmo_rs_git_sha_short": None}


def _output_dir(root: Path) -> Path:
    meta = git_metadata()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha = meta["git_sha_short"] or "unknown"
    cosmo = _cosmo_rs_git().get("cosmo_rs_git_sha_short") or "cosmo"
    path = root / "benchmark" / "results" / "cosmo" / f"{stamp}_{sha}_{cosmo}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _time_call(fn, *, repeats: int, warmups: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "total_time_s": float(np.median(arr)),
        "mean_time_s": float(np.mean(arr)),
        "min_time_s": float(np.min(arr)),
        "max_time_s": float(np.max(arr)),
    }


def _walk_forward(*, train: int, test: int, expanding: bool) -> WalkForward:
    if expanding:
        return WalkForward(train_size=train, test_size=test, expand_train=True)
    return WalkForward(train_size=train, test_size=test)


def profile_fold_breakdown(
    X,
    *,
    risk: RiskMeasure,
    persist_mode: str,
    train_size: int,
    test_size: int,
) -> dict[str, float]:
    """Single-trajectory timing: moments vs matrix bind vs COSMO solve."""
    estimator = MeanRisk(risk_measure=risk, l2_coef=1e-5)
    spec = replace(estimator_spec(estimator), solver="COSMO")
    cv = WalkForward(train_size=train_size, test_size=test_size)
    x_arr = np.ascontiguousarray(X, dtype=np.float64)
    n_assets = x_arr.shape[1]
    engine = make_cosmo_engine(
        spec,
        n_assets=n_assets,
        n_observations=None if risk is RiskMeasure.VARIANCE else train_size,
        persist_mode=persist_mode,  # type: ignore[arg-type]
    )
    moments_s = bind_s = solve_s = 0.0
    n_folds = 0
    for i, (train, _test) in enumerate(cv.split(x_arr)):
        t0 = time.perf_counter()
        moments = empirical_from_window(
            x_arr[train], keep_returns=risk is not RiskMeasure.VARIANCE
        )
        moments_s += time.perf_counter() - t0
        t1 = time.perf_counter()
        engine.solve(moments, warm=i > 0)
        elapsed = time.perf_counter() - t1
        solve_s += elapsed
        n_folds += 1
    traces = engine._workspace.traces
    factor_s = sum(t.factor_time for t in traces)
    iter_s = sum(t.iter_time for t in traces)
    setup_s = sum(t.setup_time for t in traces)
    iters = [t.iterations for t in traces]
    return {
        "n_folds": float(n_folds),
        "moments_s": moments_s,
        "solve_s": solve_s,
        "factor_s": factor_s,
        "iter_s": iter_s,
        "setup_s": setup_s,
        "bind_s": bind_s,
        "mean_iterations": float(np.mean(iters)) if iters else float("nan"),
        "median_iterations": float(np.median(iters)) if iters else float("nan"),
        "max_iterations": float(np.max(iters)) if iters else float("nan"),
        "solver_share": solve_s / (moments_s + solve_s)
        if solve_s + moments_s
        else float("nan"),
    }


def run_end_to_end_cell(
    X,
    cv,
    estimator: MeanRisk,
    *,
    method: str,
    repeats: int,
    warmups: int,
) -> dict:
    def native():
        return skfolio_cv_predict(estimator, X, cv=cv, n_jobs=1)

    def auto():
        return cross_val_predict(estimator, X, cv=cv, n_jobs=1, return_report=True)

    def cosmo():
        return cross_val_predict(
            estimator, X, cv=cv, n_jobs=1, backend="cosmo", return_report=True
        )

    ref = native()
    ref_sharpe = path_sharpes(ref)
    row: dict = {
        "method": method,
        "n_folds": int(getattr(cv, "get_n_splits", lambda X=None: 0)(X) or 0),
    }
    if method == "native":
        pred = ref
        report = None
        timing = _time_call(native, repeats=repeats, warmups=warmups)
    elif method == "auto":
        pred, report = auto()
        timing = _time_call(lambda: auto(), repeats=repeats, warmups=warmups)
    else:
        pred, report = cosmo()
        timing = _time_call(lambda: cosmo(), repeats=repeats, warmups=warmups)
    obs = path_sharpes(pred)
    row.update(timing)
    row["mean_sharpe"] = float(np.mean(obs))
    row["median_sharpe"] = float(np.median(obs))
    row["max_sharpe_error"] = _max_abs_diff(obs, ref_sharpe)
    row["backend"] = "sklearn" if report is None else report.backend
    row["n_warm_starts"] = 0 if report is None else report.n_warm_starts
    row["solve_s"] = 0.0 if report is None else report.solve_s
    row["moments_s"] = 0.0 if report is None else report.moments_s
    row["failure_count"] = 0
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    args = parser.parse_args(argv)

    if not cosmo_available():
        raise SystemExit("cosmo_rs is not installed; cannot run the COSMO experiment")

    n_obs, n_assets, train, test = (80, 6, 40, 10) if args.quick else (252, 12, 84, 21)
    repeats = args.repeats if args.repeats is not None else (1 if args.quick else 2)
    warmups = args.warmups if args.warmups is not None else 1
    X = factor_returns(n_obs, n_assets, seed=42)
    out = _output_dir(Path(__file__).resolve().parents[1])
    (out / "formulations.md").write_text(to_markdown(), encoding="utf-8")
    env = {
        **git_metadata(),
        "python": __import__("sys").version,
        "cosmo_rs": True,
        "quick": args.quick,
        "n_obs": n_obs,
        "n_assets": n_assets,
        "train_size": train,
        "test_size": test,
        **_cosmo_rs_git(),
    }
    try:
        from importlib.metadata import version

        for pkg in (
            "skfolio",
            "cvxpy",
            "cvxpy-base",
            "clarabel",
            "osqp",
            "highspy",
            "cosmo-rs",
            "numpy",
        ):
            try:
                env[f"{pkg}_version"] = version(pkg)
            except Exception:
                env[f"{pkg}_version"] = None
        if env.get("cvxpy_version") is None:
            env["cvxpy_version"] = env.get("cvxpy-base_version")
    except Exception:
        pass
    rustc = shutil.which("rustc")
    if rustc:
        try:
            env["rustc"] = subprocess.check_output(
                [rustc, "--version"], text=True
            ).strip()
        except Exception:
            env["rustc"] = None
    (out / "environment.json").write_text(
        json.dumps(env, indent=2, default=str), encoding="utf-8"
    )

    rows: list[dict] = []
    cv = WalkForward(train_size=train, test_size=test)
    for risk in RISKS:
        estimator = MeanRisk(risk_measure=risk, l2_coef=1e-5)
        rec = formulation_record(risk)
        for method in ("native", "auto", "cosmo"):
            try:
                cell = run_end_to_end_cell(
                    X,
                    cv,
                    estimator,
                    method=method,
                    repeats=repeats,
                    warmups=warmups,
                )
            except Exception as error:
                cell = {
                    "method": method,
                    "total_time_s": float("nan"),
                    "failure_count": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            cell.update(
                {
                    "risk": risk.name,
                    "cone_class": rec.cone_class,
                    "persist_class": rec.persist_class_fixed_t,
                    "cv_method": "walk-forward",
                    "n_assets": n_assets,
                    "n_samples": n_obs,
                    "train_size": train,
                    "test_size": test,
                    "quick": args.quick,
                }
            )
            rows.append(cell)
            print(
                f"{risk.name:20} {method:8} "
                f"{cell.get('total_time_s', float('nan')):8.3f}s  "
                f"backend={cell.get('backend', '?')}"
            )

        try:
            br = profile_fold_breakdown(
                X,
                risk=risk,
                persist_mode=(
                    "persist_full" if risk is RiskMeasure.VARIANCE else "persist_factor"
                ),
                train_size=train,
                test_size=test,
            )
            br.update(
                {
                    "risk": risk.name,
                    "method": "cosmo-profile",
                    "persist_mode": (
                        "persist_full"
                        if risk is RiskMeasure.VARIANCE
                        else "persist_factor"
                    ),
                }
            )
            rows.append(br)
            print(
                f"{risk.name:20} profile  solver_share={br['solver_share']:.2%}  "
                f"mean_iter={br['mean_iterations']:.1f}"
            )
        except Exception as error:
            rows.append(
                {
                    "risk": risk.name,
                    "method": "cosmo-profile",
                    "error": f"{type(error).__name__}: {error}",
                    "failure_count": 1,
                }
            )

        x_arr = np.ascontiguousarray(X, dtype=np.float64)
        spec = replace(estimator_spec(estimator), solver="COSMO")
        windows = [
            x_arr[i : i + train] for i in range(0, n_obs - train - test + 1, test)
        ]
        for mode in MODES:
            engine = make_cosmo_engine(
                spec,
                n_assets=n_assets,
                n_observations=None if risk is RiskMeasure.VARIANCE else train,
                persist_mode=mode,  # type: ignore[arg-type]
            )
            started = time.perf_counter()
            failed = 0
            iters = []
            try:
                for i, window in enumerate(windows):
                    moments = empirical_from_window(
                        window, keep_returns=risk is not RiskMeasure.VARIANCE
                    )
                    engine.solve(moments, warm=i > 0)
                    if engine.last_trace is not None:
                        iters.append(engine.last_trace.iterations)
            except Exception as error:
                failed = 1
                error_s = f"{type(error).__name__}: {error}"
            else:
                error_s = None
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "risk": risk.name,
                    "method": f"cosmo-{mode}",
                    "persist_mode": mode,
                    "total_time_s": elapsed,
                    "n_folds": len(windows),
                    "mean_iterations": float(np.mean(iters)) if iters else float("nan"),
                    "median_iterations": float(np.median(iters))
                    if iters
                    else float("nan"),
                    "max_iterations": float(np.max(iters)) if iters else float("nan"),
                    "n_warm_starts": engine.n_warm_starts,
                    "n_rebuilds": engine._workspace.n_rebuilds,
                    "failure_count": failed,
                    "cv_method": "walk-forward-windows",
                    "error": error_s,
                }
            )
            print(
                f"{risk.name:20} {mode:16} {elapsed:8.3f}s  "
                f"mean_iter={np.mean(iters) if iters else float('nan'):.1f}"
            )

    # Rolling vs expanding windows, and n_jobs vs sequential COSMO (variance).
    estimator_var = MeanRisk(l2_coef=1e-5)
    for expanding, label in ((False, "rolling"), (True, "expanding")):
        wf = _walk_forward(train=train, test=test, expanding=expanding)
        for method in ("native", "auto", "cosmo"):
            try:
                cell = run_end_to_end_cell(
                    X,
                    wf,
                    estimator_var,
                    method=method,
                    repeats=repeats,
                    warmups=warmups,
                )
            except Exception as error:
                cell = {
                    "method": method,
                    "error": f"{type(error).__name__}: {error}",
                    "failure_count": 1,
                    "total_time_s": float("nan"),
                }
            cell.update(
                {
                    "risk": "VARIANCE",
                    "cv_method": f"walk-forward-{label}",
                    "n_assets": n_assets,
                    "n_samples": n_obs,
                    "train_size": train,
                    "test_size": test,
                    "quick": args.quick,
                }
            )
            rows.append(cell)
            print(
                f"VARIANCE {label:10} {method:8} "
                f"{cell.get('total_time_s', float('nan')):8.3f}s"
            )

    for n_jobs, label in ((1, "native-n_jobs=1"), (2, "native-n_jobs=2")):
        cv = WalkForward(train_size=train, test_size=test)

        def _native(n_jobs=n_jobs, cv=cv):
            return skfolio_cv_predict(estimator_var, X, cv=cv, n_jobs=n_jobs)

        try:
            timing = _time_call(_native, repeats=repeats, warmups=warmups)
            pred = _native()
            rows.append(
                {
                    "risk": "VARIANCE",
                    "method": label,
                    "cv_method": "walk-forward-parallel",
                    "n_jobs": n_jobs,
                    "mean_sharpe": float(np.mean(path_sharpes(pred))),
                    "n_folds": int(cv.get_n_splits(X)),
                    "failure_count": 0,
                    **timing,
                }
            )
            print(f"VARIANCE {label:20} {timing['total_time_s']:8.3f}s")
        except Exception as error:
            rows.append(
                {
                    "risk": "VARIANCE",
                    "method": label,
                    "cv_method": "walk-forward-parallel",
                    "n_jobs": n_jobs,
                    "failure_count": 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if not args.quick:
        X_mrc = factor_returns(180, 10, seed=7)
        mrc = MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=60, test_size=20),
            n_subsamples=3,
            asset_subset_size=6,
            window_size=140,
            random_state=3,
        )
        for method in ("native", "auto", "cosmo"):
            try:
                cell = run_end_to_end_cell(
                    X_mrc,
                    mrc,
                    MeanRisk(l2_coef=1e-5),
                    method=method,
                    repeats=1,
                    warmups=0,
                )
            except Exception as error:
                cell = {"method": method, "error": str(error), "failure_count": 1}
            cell.update({"risk": "VARIANCE", "cv_method": "multiple-randomized"})
            rows.append(cell)
        cpcv = CombinatorialPurgedCV(
            n_folds=4, n_test_folds=2, purged_size=1, embargo_size=1
        )
        for method in ("native", "auto", "cosmo"):
            try:
                cell = run_end_to_end_cell(
                    factor_returns(120, 8, seed=8),
                    cpcv,
                    MeanRisk(l2_coef=1e-5),
                    method=method,
                    repeats=1,
                    warmups=0,
                )
            except Exception as error:
                cell = {"method": method, "error": str(error), "failure_count": 1}
            cell.update({"risk": "VARIANCE", "cv_method": "purged-cpcv"})
            rows.append(cell)

    csv_path = out / "results.csv"
    keys = sorted({k for row in rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "results.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# COSMO.rs persistence experiment",
        "",
        f"Output: `{csv_path}`",
        "",
        f"Panel: {n_obs} × {n_assets}, train={train}, test={test}, quick={args.quick}",
        "",
        "| Risk | Method | Time (s) | Backend | Mean Sharpe | "
        "Sharpe |Δ| | Mean iter | Failures |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("method") not in {"native", "auto", "cosmo"}:
            continue
        lines.append(
            "| {risk} | {method} | {total_time_s:.3f} | {backend} | "
            "{mean_sharpe} | {max_sharpe_error} | {mean_iterations} | "
            "{failure_count} |".format(
                risk=row.get("risk", ""),
                method=row.get("method", ""),
                total_time_s=float(row.get("total_time_s") or float("nan")),
                backend=row.get("backend", ""),
                mean_sharpe=row.get("mean_sharpe", ""),
                max_sharpe_error=row.get("max_sharpe_error", ""),
                mean_iterations=row.get("mean_iterations", ""),
                failure_count=row.get("failure_count", 0),
            )
        )
    lines += ["", "## Persist-mode ablations", ""]
    lines.append("| Risk | Mode | Time (s) | Mean iter | Rebuilds | Failures |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        if not str(row.get("method", "")).startswith("cosmo-"):
            continue
        if row.get("method") == "cosmo-profile":
            continue
        lines.append(
            "| {risk} | {mode} | {total_time_s:.3f} | {mean_iterations} | "
            "{n_rebuilds} | {failure_count} |".format(
                risk=row.get("risk", ""),
                mode=row.get("persist_mode", row.get("method")),
                total_time_s=float(row.get("total_time_s") or float("nan")),
                mean_iterations=row.get("mean_iterations", ""),
                n_rebuilds=row.get("n_rebuilds", ""),
                failure_count=row.get("failure_count", 0),
            )
        )
    lines += [
        "",
        "## Solver share of COSMO compact time",
        "",
        "| Risk | Solver share | Mean iter | Moments (s) | Solve (s) | Factor (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row.get("method") != "cosmo-profile":
            continue
        if row.get("error"):
            lines.append(
                f"| {row.get('risk', '')} | error | {row.get('error')} | | | |"
            )
            continue
        lines.append(
            "| {risk} | {share} | {mean_iterations} | {moments_s} | "
            "{solve_s} | {factor_s} |".format(
                risk=row.get("risk", ""),
                share=row.get("solver_share", ""),
                mean_iterations=row.get("mean_iterations", ""),
                moments_s=row.get("moments_s", ""),
                solve_s=row.get("solve_s", ""),
                factor_s=row.get("factor_s", ""),
            )
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

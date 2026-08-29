"""CVXPY Moreau coverage and batched-CV timings for MeanRisk.

Not the canonical PR-vs-main harness. On one host, compare Clarabel,
CVXPY Moreau, backend=\"auto\", and (for boxed variance) CompiledSolver
batches of WalkForward / MRC / CPCV folds.

Δ% = 100 * (head_time - base_time) / base_time. Positive means slower.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MOREAU_SOLVER_PARAMS = {"device": "cpu", "verbose": False}


def moreau_available() -> tuple[bool, str]:
    try:
        import moreau  # noqa: F401
    except ImportError as exc:
        return False, f"import moreau failed: {exc}"
    return True, ""


@lru_cache(maxsize=1)
def cvxpy_moreau_available() -> tuple[bool, str]:
    try:
        import cvxpy as cp
    except ImportError as exc:
        return False, f"import cvxpy failed: {exc}"
    if not hasattr(cp, "MOREAU"):
        return False, f"cvxpy {cp.__version__} has no solver MOREAU (need >= 1.8.2)"
    from skfolio.optimization import MeanRisk

    rng = np.random.default_rng(0)
    panel = rng.normal(scale=0.01, size=(24, 3))
    estimator = MeanRisk(
        l2_coef=1e-5,
        solver="MOREAU",
        solver_params={"device": "cpu", "verbose": False},
    )
    try:
        estimator.fit(panel)
    except Exception as exc:
        if "license" in str(exc).lower():
            return (
                False,
                "MOREAU_LICENSE_KEY is not set (CVXPY Moreau / MeanRisk requires a license)",
            )
        return False, str(exc).split("\n")[0][:240]
    return True, ""


def delta_pct(head: float, base: float) -> float:
    if base <= 0:
        return float("nan")
    return 100.0 * (head - base) / base


def _classify_error(message: str) -> str:
    text = message.lower()
    if "license" in text:
        return "license"
    if "moreau" in text and ("not" in text or "unknown" in text or "install" in text):
        return "missing_solver"
    if "cone" in text or "exponential" in text or "expcone" in text:
        return "unsupported_cone"
    if "infeasible" in text or "unbounded" in text:
        return "infeasible"
    return "error"


def run_coverage(x, specs, timeout_note: str) -> list[dict[str, object]]:
    from sklearn.base import clone

    rows: list[dict[str, object]] = []
    for spec in specs:
        row: dict[str, object] = {
            "name": spec.name,
            "objective": spec.objective,
            "risk": spec.risk,
            "extra": spec.extra,
            "clarabel_ok": False,
            "moreau_ok": False,
            "max_abs_dw": "",
            "error_class": "",
            "error": "",
            "note": timeout_note,
        }
        clarabel = spec.factory()
        clarabel.set_params(solver="CLARABEL")
        try:
            clarabel.fit(x)
            w_ref = np.asarray(clarabel.weights_, dtype=np.float64)
            row["clarabel_ok"] = True
        except Exception as exc:
            row["error_class"] = _classify_error(str(exc))
            row["error"] = f"clarabel: {exc}"
            rows.append(row)
            continue
        moreau_est = clone(clarabel)
        moreau_est.set_params(solver="MOREAU", solver_params=dict(MOREAU_SOLVER_PARAMS))
        try:
            moreau_est.fit(x)
            w_m = np.asarray(moreau_est.weights_, dtype=np.float64)
            row["moreau_ok"] = True
            row["max_abs_dw"] = float(np.max(np.abs(w_m - w_ref)))
        except Exception as exc:
            row["error_class"] = _classify_error(str(exc))
            row["error"] = str(exc).split("\n")[0][:240]
        rows.append(row)
    return rows


def _time_call(fn: Callable[[], object], *, warmups: int, repetitions: int) -> tuple[float, object]:
    result = None
    for _ in range(warmups):
        result = fn()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], result


def run_cv_timings(x, config, specs) -> list[dict[str, object]]:
    from sklearn.base import clone
    from skfolio.model_selection import cross_val_predict as skfolio_cv

    from benchmark.protocol import make_cv
    from skfolio_accelerate import cross_val_predict
    from skfolio_accelerate._arrays import as_float_2d
    from skfolio_accelerate.compact import estimator_spec
    from skfolio_accelerate.cv_plan import compile_cv_plan
    from skfolio_accelerate.predict import compact_blocked_reason

    from experiments.moreau_batch import batched_weights_for_plan, moments_for_plan
    from skfolio_accelerate.compact import estimator_spec, make_compact_engine

    rows: list[dict[str, object]] = []
    x_np = as_float_2d(np.asarray(x, dtype=np.float64))
    cv_kinds = config.cv_kinds
    for cv_kind in cv_kinds:
        cv = make_cv(cv_kind, config)
        plan = compile_cv_plan(cv, x)
        for spec in specs:
            base_est = spec.factory()
            base_est.set_params(solver="CLARABEL")
            moreau_est = clone(base_est)
            moreau_est.set_params(
                solver="MOREAU", solver_params=dict(MOREAU_SOLVER_PARAMS)
            )

            def native_clarabel(estimator=base_est, splitter=cv):
                return skfolio_cv(clone(estimator), x, cv=splitter, n_jobs=1)

            def native_moreau(estimator=moreau_est, splitter=cv):
                return skfolio_cv(clone(estimator), x, cv=splitter, n_jobs=1)

            def auto_backend(estimator=base_est, splitter=cv):
                return cross_val_predict(clone(estimator), x, cv=splitter, n_jobs=1)

            timings: dict[str, float] = {}
            errors: dict[str, str] = {}
            cvx_ok, cvx_err = cvxpy_moreau_available()
            legs: list[tuple[str, Callable[[], object]]] = [
                ("native_clarabel", native_clarabel),
                ("auto", auto_backend),
            ]
            if cvx_ok:
                legs.insert(1, ("native_moreau", native_moreau))
            else:
                errors["native_moreau"] = cvx_err
            for name, fn in legs:
                try:
                    elapsed, _ = _time_call(
                        fn, warmups=config.warmups, repetitions=config.repetitions
                    )
                    timings[name] = elapsed
                except Exception as exc:
                    errors[name] = str(exc).split("\n")[0][:200]

            blocked = compact_blocked_reason(base_est)
            if blocked is None and spec.risk == "VARIANCE" and not spec.extra:
                compact_spec = estimator_spec(base_est)

                def moreau_batch(spec_=compact_spec, matrix=x_np, cv_plan=plan):
                    return batched_weights_for_plan(spec_, matrix, cv_plan)

                def osqp_loop(spec_=compact_spec, matrix=x_np, cv_plan=plan):
                    pairs = moments_for_plan(matrix, cv_plan, spec_)
                    engine = None
                    weights = []
                    for fold, moments in pairs:
                        n = int(moments.covariance.shape[0])
                        if engine is None or engine.n_assets != n:
                            engine = make_compact_engine(
                                spec_, n_assets=n, n_observations=moments.n_observations
                            )
                        weights.append(engine.solve(moments))
                    return weights

                for name, fn in (
                    ("moreau_batch", moreau_batch),
                    ("osqp_folds", osqp_loop),
                ):
                    try:
                        elapsed, _ = _time_call(
                            fn, warmups=config.warmups, repetitions=config.repetitions
                        )
                        timings[name] = elapsed
                    except Exception as exc:
                        errors[name] = str(exc).split("\n")[0][:200]

            clarabel_s = timings.get("native_clarabel")

            def _fmt(value) -> str:
                if not isinstance(value, float):
                    return value if value is not None else ""
                return f"{value:.6f}"

            row = {
                "cv_kind": cv_kind,
                "name": spec.name,
                "n_folds": plan.n_splits,
                "native_clarabel_s": _fmt(clarabel_s),
                "native_moreau_s": _fmt(timings.get("native_moreau")),
                "auto_s": _fmt(timings.get("auto")),
                "osqp_folds_s": _fmt(timings.get("osqp_folds")),
                "moreau_batch_s": _fmt(timings.get("moreau_batch")),
                "delta_pct_moreau_vs_clarabel": _fmt(
                    delta_pct(timings["native_moreau"], clarabel_s)
                    if "native_moreau" in timings and clarabel_s
                    else None
                ),
                "delta_pct_batch_vs_osqp": _fmt(
                    delta_pct(timings["moreau_batch"], timings["osqp_folds"])
                    if "moreau_batch" in timings and "osqp_folds" in timings
                    else None
                ),
                "delta_pct_batch_vs_auto": _fmt(
                    delta_pct(timings["moreau_batch"], timings["auto"])
                    if "moreau_batch" in timings and "auto" in timings
                    else None
                ),
                "errors": "; ".join(f"{k}: {v}" for k, v in errors.items()),
            }
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, object]], columns: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*columns))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*(str(row.get(col, "")) for col in columns)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--coverage", action="store_true", default=True)
    parser.add_argument("--no-coverage", action="store_false", dest="coverage")
    parser.add_argument("--timings", action="store_true", default=True)
    parser.add_argument("--no-timings", action="store_false", dest="timings")
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "experiments" / "results",
    )
    args = parser.parse_args(argv)

    ok, err = moreau_available()
    if not ok:
        print(err)
        return 1
    cvx_ok, cvx_err = cvxpy_moreau_available()
    import moreau

    print(f"moreau {getattr(moreau, '__version__', '?')}")
    print(f"devices: {moreau.available_devices()}")
    print(f"cuda: {moreau.device_available('cuda')}")
    print(f"cvxpy MOREAU: {cvx_ok} {cvx_err}")

    from benchmark.config import build_config
    from benchmark.datasets import make_synthetic
    from benchmark.environment import collect_environment
    from benchmark.estimators import mean_risk_specs
    from benchmark.protocol import apply_thread_limits

    preset = "quick" if args.quick else None
    config = build_config(
        preset=preset,
        datasets=("synthetic",),
        methods=("native",),
        cv_kinds=("walk-forward", "multiple-randomized", "purged-cpcv"),
    )
    apply_thread_limits(config)
    env = collect_environment(config)
    packages = env.get("packages", {})
    print(f"  python: {env.get('python_version')}")
    print(f"  cpu: {env.get('cpu_model')}")
    print(f"  workers={config.workers} thread_limit={config.thread_limit}")
    print(f"  skfolio={packages.get('skfolio')} cvxpy={packages.get('cvxpy')}")
    loaded = make_synthetic(config)
    x = loaded.X
    print(f"X shape: {x.shape}  quick={args.quick}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.coverage:
        if not cvx_ok:
            print("skip coverage: ", cvx_err)
        else:
            specs = mean_risk_specs(config)
            rows = run_coverage(x, specs, timeout_note="fit")
            _write_csv(args.out / "moreau_coverage.csv", rows)
            n_ok = sum(1 for row in rows if row["moreau_ok"])
            print(f"coverage: {n_ok}/{len(rows)} Moreau fits succeeded")
            _print_table(
                rows,
                ["name", "clarabel_ok", "moreau_ok", "max_abs_dw", "error_class", "error"],
            )

    if args.timings:
        if not cvx_ok:
            print("skip CVXPY timings: ", cvx_err)
        timing_names = {
            "MINIMIZE_RISK/VARIANCE",
            "MINIMIZE_RISK/CVAR",
            "MINIMIZE_RISK/VARIANCE+min_return",
        }
        specs = [s for s in mean_risk_specs(config) if s.name in timing_names]
        rows = run_cv_timings(x, config, specs)
        _write_csv(args.out / "moreau_cv_timings.csv", rows)
        print("Δ% = 100 * (head - base) / base; positive means slower")
        _print_table(
            rows,
            [
                "cv_kind",
                "name",
                "n_folds",
                "native_clarabel_s",
                "auto_s",
                "osqp_folds_s",
                "moreau_batch_s",
                "delta_pct_batch_vs_osqp",
                "delta_pct_batch_vs_auto",
                "errors",
            ],
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise

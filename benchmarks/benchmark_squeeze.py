"""Larger-frame timings vs sklearn GridSearchCV / naive MeanRisk.fit.

Run with::

    PYTHONPATH=src python benchmarks/benchmark_squeeze.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV
from skfolio_accelerate.backends.rust_clarabel import rust_is_available


def _cap_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def factor_returns(n_obs: int, n_assets: int, n_factors: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(n_obs, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n_assets))
    idio = rng.normal(0.0, 0.005, size=(n_obs, n_assets))
    return pd.DataFrame(factors @ loadings + idio)


def _l2_grid(n: int) -> list[float]:
    return list(np.logspace(-5, -1, n))


def main() -> None:
    _cap_threads()
    n_jobs = os.cpu_count() or 1
    print(
        "cpu_count",
        n_jobs,
        "nproc",
        os.cpu_count(),
        "rust",
        rust_is_available(),
    )

    # --- KFold 2520 x 200, 10 folds x 24 l2 ---
    X = factor_returns(2520, 200, seed=1)
    params = {"l2_coef": _l2_grid(24)}
    cv = KFold(n_splits=10, shuffle=False)

    t0 = time.perf_counter()
    sk = GridSearchCV(MeanRisk(), params, cv=cv, error_score="raise").fit(X)
    sklearn_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    py = MassiveGridSearchCV(
        MeanRisk(), params, cv=cv, backend="python", refit=False, n_jobs=1
    ).fit(X)
    python_s = time.perf_counter() - t0

    rust_s = float("nan")
    rust = None
    if rust_is_available():
        t0 = time.perf_counter()
        rust = MassiveGridSearchCV(
            MeanRisk(),
            params,
            cv=cv,
            backend="rust",
            refit=False,
            n_jobs=n_jobs,
            solver_threads=1,
        ).fit(X)
        rust_s = time.perf_counter() - t0

    print("KFold 2520x200 10x24")
    print(f"  sklearn {sklearn_s:.4f}s  python {python_s:.4f}s  rust {rust_s:.4f}s")
    print(
        "  best_params match",
        py.best_params_ == sk.best_params_,
        "score Δ",
        abs(py.best_score_ - sk.best_score_),
    )
    if rust is not None:
        print(
            "  rust best_params match",
            rust.best_params_ == sk.best_params_,
            "score Δ",
            abs(rust.best_score_ - sk.best_score_),
        )
        print("  rust report")
        r = rust.acceleration_report_
        print(
            f"    instantiate {r.instantiate_s:.4f}s solve {r.solve_s:.4f}s "
            f"eval {r.eval_s:.4f}s compile {r.compile_s:.4f}s "
            f"priors {r.n_prior_fits}"
        )
    print("  python report")
    r = py.acceleration_report_
    print(
        f"    instantiate {r.instantiate_s:.4f}s solve {r.solve_s:.4f}s "
        f"eval {r.eval_s:.4f}s compile {r.compile_s:.4f}s "
        f"priors {r.n_prior_fits}"
    )

    # --- CPCV 1008 x 120, 10c2 x 16 l2 ---
    X2 = factor_returns(1008, 120, seed=2)
    params2 = {"l2_coef": _l2_grid(16)}
    cv2 = CombinatorialPurgedCV(n_folds=10, n_test_folds=2)
    n_splits = cv2.get_n_splits()

    t0 = time.perf_counter()
    n_naive = 0
    for train, _tests in cv2.split(X2):
        for l2 in params2["l2_coef"][:1]:
            MeanRisk(l2_coef=float(l2)).fit(X2.iloc[train])
            n_naive += 1
            break
        break
    # time a 30-fit sample then scale? Better time compiled vs a subset.
    # Full naive: n_splits * 16 fits. Time compiled fully; naive on 30 fits * scale.
    t0 = time.perf_counter()
    sample = 0
    for train, _tests in cv2.split(X2):
        MeanRisk(l2_coef=params2["l2_coef"][0]).fit(X2.iloc[train])
        sample += 1
        if sample >= 8:
            break
    sample_s = time.perf_counter() - t0
    naive_est = sample_s / sample * (n_splits * len(params2["l2_coef"]))

    t0 = time.perf_counter()
    cpcv = MassiveGridSearchCV(
        MeanRisk(),
        params2,
        cv=cv2,
        backend="rust" if rust_is_available() else "python",
        refit=False,
        n_jobs=n_jobs,
        solver_threads=1,
    ).fit(X2)
    cpcv_s = time.perf_counter() - t0
    print("CPCV 1008x120 10c2 x 16")
    print(
        f"  naive-est {naive_est:.4f}s ({n_splits * 16} fits)  "
        f"compiled {cpcv_s:.4f}s  priors {cpcv.acceleration_report_.n_prior_fits}"
    )

    # --- small KFold parity ---
    X3 = factor_returns(180, 25, seed=3)
    params3 = {"l2_coef": [1e-4, 1e-3, 1e-2, 1e-1]}
    cv3 = KFold(5, shuffle=False)
    t0 = time.perf_counter()
    GridSearchCV(MeanRisk(), params3, cv=cv3).fit(X3)
    s_sk = time.perf_counter() - t0
    t0 = time.perf_counter()
    MassiveGridSearchCV(
        MeanRisk(), params3, cv=cv3, backend="python", refit=False
    ).fit(X3)
    s_py = time.perf_counter() - t0
    print(f"KFold 180x25 5x4  sklearn {s_sk:.4f}s  python {s_py:.4f}s")


if __name__ == "__main__":
    main()

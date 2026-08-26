"""Compare skfolio and skfolio-accelerate across public optimization cases.

This is a compatibility and timing matrix, not a claim that every optimizer has
the same acceleration opportunity.  Unsupported compact problems deliberately
run through skfolio's original implementation and should remain numerically
identical, while structured MeanRisk cases may use a compact solver.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import (
    BenchmarkTracker,
    DistributionallyRobustCVaR,
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    MaximumDiversification,
    MeanRisk,
    NestedClustersOptimization,
    Random,
    RiskBudgeting,
    SchurComplementary,
    StackingOptimization,
)

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.flagship import factor_returns


@dataclass(frozen=True)
class Case:
    name: str
    factory: Callable[[], object]
    needs_target: bool = False


def _cv_cases(quick: bool) -> list[tuple[str, Callable[[], object]]]:
    train_size = 40 if quick else 252
    test_size = 20 if quick else 21
    n_subsamples = 3 if quick else 20
    window_size = 100 if quick else 756
    return [
        (
            "walk-forward",
            lambda: WalkForward(train_size=train_size, test_size=test_size),
        ),
        (
            "purged-cpcv",
            lambda: CombinatorialPurgedCV(
                n_folds=4,
                n_test_folds=2,
                purged_size=1,
                embargo_size=1,
            ),
        ),
        (
            "multiple-randomized",
            lambda: MultipleRandomizedCV(
                walk_forward=WalkForward(train_size=train_size, test_size=test_size),
                n_subsamples=n_subsamples,
                asset_subset_size=4 if quick else 12,
                window_size=window_size,
                random_state=43,
            ),
        ),
    ]


def _cases() -> list[Case]:
    mean_risk = [
        Case(f"MeanRisk[{risk.name}]", lambda risk=risk: MeanRisk(risk_measure=risk))
        for risk in RiskMeasure
    ]
    other = [
        Case("EqualWeighted", EqualWeighted),
        Case("InverseVolatility", InverseVolatility),
        Case("Random", Random),
        Case("HierarchicalRiskParity", HierarchicalRiskParity),
        Case("HierarchicalEqualRiskContribution", HierarchicalEqualRiskContribution),
        Case("NestedClustersOptimization", NestedClustersOptimization),
        Case("MaximumDiversification", MaximumDiversification),
        Case("RiskBudgeting", RiskBudgeting),
        Case("DistributionallyRobustCVaR", DistributionallyRobustCVaR),
        Case("SchurComplementary", SchurComplementary),
        Case(
            "StackingOptimization",
            lambda: StackingOptimization([("equal", EqualWeighted())]),
        ),
        Case("BenchmarkTracker", BenchmarkTracker, needs_target=True),
    ]
    return mean_risk + other


def _predict(function, case: Case, X, cv):
    np.random.seed(44)
    started = time.perf_counter()
    prediction = function(
        case.factory(),
        X,
        y=X.iloc[:, 0] if case.needs_target else None,
        cv=cv,
        n_jobs=1,
    )
    return prediction, time.perf_counter() - started


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0]}".replace(",", ";")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_observations = 120 if args.quick else 20 * 252
    n_assets = 6 if args.quick else 20
    X = factor_returns(n_observations, n_assets, seed=42)

    print("case,cv,status,backend,skfolio_s,accelerated_s,speedup,max_sharpe_delta")
    for case in _cases():
        for cv_name, cv_factory in _cv_cases(args.quick):
            try:
                reference, baseline_s = _predict(
                    skfolio_cross_val_predict, case, X, cv_factory()
                )
            except Exception as exc:
                print(f"{case.name},{cv_name},native-error,,{_error(exc)},,,,")
                continue
            try:
                observed, accelerated_s = _predict(
                    cross_val_predict, case, X, cv_factory()
                )
                _, report = cross_val_predict(
                    case.factory(),
                    X,
                    y=X.iloc[:, 0] if case.needs_target else None,
                    cv=cv_factory(),
                    n_jobs=1,
                    return_report=True,
                )
            except Exception as exc:
                print(
                    f"{case.name},{cv_name},accelerated-error,,{baseline_s:.4f},"
                    f"{_error(exc)},,,"
                )
                continue
            delta = np.max(np.abs(path_sharpes(observed) - path_sharpes(reference)))
            print(
                f"{case.name},{cv_name},ok,{report.backend},{baseline_s:.4f},"
                f"{accelerated_s:.4f},{baseline_s / accelerated_s:.2f},{delta:.3e}"
            )


if __name__ == "__main__":
    main()

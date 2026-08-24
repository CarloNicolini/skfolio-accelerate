"""Clarabel Python update vs CVXPY problem.solve on a MeanRisk VARIANCE twin."""

from __future__ import annotations

import time

import numpy as np

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.backends.python_clarabel import PythonClarabelEngine
from skfolio_accelerate.compile import extract_problem_template, instantiate
from skfolio_accelerate.estimators.mean_risk_twin import (
    bind_twin_values,
    build_mean_risk_twin,
)
from skfolio_accelerate.moments import fit_prior


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0005, scale=0.01, size=(120, 15))
    estimator = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    moments = fit_prior(estimator, X)
    twin = build_mean_risk_twin(
        X.shape[1],
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
    )
    template = extract_problem_template(twin, "var")
    engine = PythonClarabelEngine(template)
    l2_grid = np.logspace(-5, -1, 20)

    t0 = time.perf_counter()
    for l2 in l2_grid:
        bind_twin_values(twin, moments, l2_coef=float(l2))
        engine.solve(instantiate(template))
    update_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for l2 in l2_grid:
        MeanRisk(risk_measure=RiskMeasure.VARIANCE, l2_coef=float(l2)).fit(X)
    fit_s = time.perf_counter() - t0
    print("clarabel update", update_s, "MeanRisk.fit", fit_s, "speedup", fit_s / update_s)


if __name__ == "__main__":
    main()

"""MeanRisk complexity ladder including TimeSeriesFactorModel parity."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import TimeSeriesFactorModel
from sklearn import set_config

from skfolio_accelerate import cross_val_predict, path_sharpes
from skfolio_accelerate.predict import classify_call
from tests.helpers import synthetic_returns


@pytest.fixture(scope="module")
def factor_panel():
    """Aligned asset / factor returns for factor-prior CV tests."""
    set_config(enable_metadata_routing=True)
    X, factors = prices_to_returns(load_sp500_dataset(), load_factors_dataset())
    return X.iloc[:120, :8], factors.iloc[:120]


def _walk_forward():
    return WalkForward(train_size=60, test_size=20)


def _assert_path_parity(observed, reference, *, rtol=3e-3, atol=2e-4):
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=rtol,
        atol=atol,
    )


def test_boxed_empirical_stays_compact():
    X = synthetic_returns(96, 6, seed=41)
    observed, report = cross_val_predict(
        MeanRisk(l2_coef=1e-5),
        X,
        cv=_walk_forward(),
        n_jobs=1,
        return_report=True,
    )
    reference = skfolio_cv_predict(
        MeanRisk(l2_coef=1e-5), X, cv=_walk_forward(), n_jobs=1
    )
    _assert_path_parity(observed, reference)
    assert report.backend == "osqp"


def test_asset_linear_constraints_use_osqp():
    import pandas as pd

    raw = synthetic_returns(96, 4, seed=42)
    X = pd.DataFrame(raw, columns=["A0", "A1", "A2", "A3"])
    estimator = MeanRisk(linear_constraints=["A0 <= 0.45"], l2_coef=1e-5)
    reference = skfolio_cv_predict(estimator, X, cv=_walk_forward(), n_jobs=1)
    observed, report = cross_val_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, return_report=True
    )
    _assert_path_parity(observed, reference)
    assert report.backend == "osqp"


def test_timeseries_factor_model_matches_native(factor_panel):
    set_config(enable_metadata_routing=True)
    X, factors = factor_panel
    estimator = MeanRisk(prior_estimator=TimeSeriesFactorModel(), l2_coef=1e-5)
    params = {"factors": factors}
    caps = classify_call(estimator, params=params)
    assert caps.can_sequential
    assert not caps.can_compact
    assert caps.compact_reason == "fit params use skfolio cross_val_predict"

    reference = skfolio_cv_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, params=params
    )
    observed, report = cross_val_predict(
        estimator,
        X,
        cv=_walk_forward(),
        n_jobs=1,
        params=params,
        return_report=True,
    )
    _assert_path_parity(observed, reference)
    assert report.backend == "cvxpy-sequential"
    assert report.n_rebuilds == 1
    assert report.n_warm_starts >= 1


def test_factor_exposure_constraints_rebuild_but_match(factor_panel):
    set_config(enable_metadata_routing=True)
    X, factors = factor_panel
    estimator = MeanRisk(
        prior_estimator=TimeSeriesFactorModel(),
        linear_constraints=["SIZE >= 0.0"],
        l2_coef=1e-5,
    )
    params = {"factors": factors}
    reference = skfolio_cv_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, params=params
    )
    observed, report = cross_val_predict(
        estimator,
        X,
        cv=_walk_forward(),
        n_jobs=1,
        params=params,
        return_report=True,
    )
    _assert_path_parity(observed, reference)
    assert report.backend == "cvxpy-sequential"
    # Loadings are baked into constraint rows; every fold rebuilds.
    assert report.n_warm_starts == 0
    assert report.n_rebuilds == len(observed)


def test_maximize_ratio_factor_model_fit_assemble(factor_panel):
    set_config(enable_metadata_routing=True)
    X, factors = factor_panel
    estimator = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
        prior_estimator=TimeSeriesFactorModel(),
        l2_coef=1e-5,
    )
    params = {"factors": factors}
    caps = classify_call(estimator, params=params)
    assert caps.auto_backend(estimator) == "fit-assemble"

    reference = skfolio_cv_predict(
        estimator, X, cv=_walk_forward(), n_jobs=1, params=params
    )
    observed, report = cross_val_predict(
        estimator,
        X,
        cv=_walk_forward(),
        n_jobs=1,
        params=params,
        return_report=True,
    )
    _assert_path_parity(observed, reference, rtol=5e-3, atol=5e-4)
    assert report.backend == "fit-assemble"


def test_characteristics_params_stay_on_sklearn():
    estimator = MeanRisk(prior_estimator=TimeSeriesFactorModel())
    caps = classify_call(estimator, params={"characteristics": object()})
    assert caps.auto_backend(estimator) == "sklearn"
    assert caps.assemble_reason == "fit params use skfolio cross_val_predict"

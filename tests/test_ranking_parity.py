"""Numerical and ranking parity across skfolio optimizers and CV methods."""

from __future__ import annotations

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)
from skfolio.model_selection import cross_val_predict as skfolio_cv_predict
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    RiskBudgeting,
)

from skfolio_accelerate import (
    cross_val_predict,
    path_sharpes,
    ranking_precision_at_k,
    spearman_rank_correlation,
)
from tests.helpers import synthetic_returns


def _walk_forward():
    return WalkForward(train_size=40, test_size=20)


def _cpcv():
    return CombinatorialPurgedCV(
        n_folds=4,
        n_test_folds=2,
        purged_size=1,
        embargo_size=1,
    )


def _mrc():
    return MultipleRandomizedCV(
        walk_forward=WalkForward(train_size=40, test_size=20),
        n_subsamples=4,
        asset_subset_size=4,
        window_size=100,
        random_state=31,
    )


CV_CASES = [
    pytest.param(_walk_forward, id="walk-forward"),
    pytest.param(_cpcv, id="purged-cpcv"),
    pytest.param(_mrc, id="multiple-randomized"),
]

ESTIMATOR_CASES = [
    pytest.param(lambda: MeanRisk(), id="mean-risk-variance"),
    pytest.param(
        lambda: MeanRisk(risk_measure=RiskMeasure.CVAR),
        id="mean-risk-cvar",
    ),
    pytest.param(EqualWeighted, id="equal-weighted"),
    pytest.param(InverseVolatility, id="inverse-volatility"),
    pytest.param(HierarchicalRiskParity, id="hierarchical-risk-parity"),
    pytest.param(RiskBudgeting, id="risk-budgeting"),
]


@pytest.mark.parametrize("cv_factory", CV_CASES)
@pytest.mark.parametrize("estimator_factory", ESTIMATOR_CASES)
def test_optimizer_and_cv_matrix_matches_skfolio(cv_factory, estimator_factory):
    X = synthetic_returns(120, 6, seed=30)
    reference = skfolio_cv_predict(estimator_factory(), X, cv=cv_factory(), n_jobs=1)
    observed = cross_val_predict(estimator_factory(), X, cv=cv_factory(), n_jobs=1)
    np.testing.assert_allclose(
        path_sharpes(observed),
        path_sharpes(reference),
        rtol=2e-3,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "cv_factory",
    [
        pytest.param(_cpcv, id="purged-cpcv"),
        pytest.param(_mrc, id="multiple-randomized"),
    ],
)
@pytest.mark.parametrize(
    "estimator_factory",
    [
        pytest.param(lambda: MeanRisk(), id="variance"),
        pytest.param(
            lambda: MeanRisk(risk_measure=RiskMeasure.CVAR),
            id="cvar",
        ),
    ],
)
def test_path_ranking_matches_skfolio(cv_factory, estimator_factory):
    X = synthetic_returns(120, 6, seed=32)
    reference = path_sharpes(
        skfolio_cv_predict(estimator_factory(), X, cv=cv_factory(), n_jobs=1)
    )
    observed = path_sharpes(
        cross_val_predict(estimator_factory(), X, cv=cv_factory(), n_jobs=1)
    )
    k = max(1, reference.size // 2)
    assert ranking_precision_at_k(reference, observed, k=k) == 1.0
    assert spearman_rank_correlation(reference, observed) >= 0.95


@pytest.mark.parametrize("cv_factory", CV_CASES)
def test_mean_risk_configuration_ranking_matches_skfolio(cv_factory):
    X = synthetic_returns(120, 6, seed=33)
    l2_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    reference = []
    observed = []
    for l2_coef in l2_values:
        reference.append(
            np.mean(
                path_sharpes(
                    skfolio_cv_predict(
                        MeanRisk(l2_coef=l2_coef),
                        X,
                        cv=cv_factory(),
                        n_jobs=1,
                    )
                )
            )
        )
        observed.append(
            np.mean(
                path_sharpes(
                    cross_val_predict(
                        MeanRisk(l2_coef=l2_coef),
                        X,
                        cv=cv_factory(),
                        n_jobs=1,
                    )
                )
            )
        )

    assert ranking_precision_at_k(reference, observed, k=3) == 1.0
    assert spearman_rank_correlation(reference, observed) >= 0.95


def test_ranking_metrics_have_expected_values():
    reference = [0.9, 0.8, 0.7, 0.6]
    observed = [0.9, 0.6, 0.8, 0.7]
    assert ranking_precision_at_k(reference, observed, k=2) == 0.5
    assert spearman_rank_correlation(reference, reference) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="between 1 and 4"):
        ranking_precision_at_k(reference, observed, k=0)

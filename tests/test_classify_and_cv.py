"""Unit tests for classification, CV plans, and eligibility."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk

from skfolio_accelerate import MassiveGridSearchCV
from skfolio_accelerate.backends.sklearn_fallback import acceleration_blocked_reason
from skfolio_accelerate.classify import classify_param_grid
from skfolio_accelerate.cv_plan import compile_cv_plan
from skfolio_accelerate.ir import ParameterClass


def test_classify_l2_is_numerical():
    classes = classify_param_grid(MeanRisk(), {"l2_coef": [1e-3, 1e-2]})
    assert classes["l2_coef"] is ParameterClass.NUMERICAL


def test_kfold_plan_has_one_test_segment():
    X = np.random.default_rng(0).normal(size=(30, 4))
    plan = compile_cv_plan(KFold(3, shuffle=False), X)
    assert plan.n_splits == 3
    assert not plan.combinatorial
    assert len(plan.folds[0].test_segments) == 1


def test_cpcv_plan_records_path_ids():
    X = np.random.default_rng(0).normal(size=(24, 3))
    cv = CombinatorialPurgedCV(n_folds=4, n_test_folds=2)
    plan = compile_cv_plan(cv, X)
    assert plan.combinatorial
    assert plan.n_paths == cv.n_test_paths
    assert len(plan.folds) == cv.get_n_splits()
    assert len(plan.folds[0].test_segments) == 2


def test_sklearn_backend_rejects_cpcv():
    X = np.random.default_rng(0).normal(size=(24, 3))
    search = MassiveGridSearchCV(
        MeanRisk(),
        {"l2_coef": [1e-3]},
        cv=CombinatorialPurgedCV(n_folds=4, n_test_folds=2),
        backend="sklearn",
    )
    try:
        search.fit(X)
    except TypeError as exc:
        assert "sklearn" in str(exc).lower() or "Combinatorial" in str(exc)
    else:
        raise AssertionError("expected TypeError for CPCV + sklearn backend")


def test_mip_is_blocked():
    reason = acceleration_blocked_reason(
        MeanRisk(cardinality=3),
        {"l2_coef": [1e-3]},
        cv=3,
        scoring=None,
    )
    assert reason is not None
    assert "cardinality" in reason

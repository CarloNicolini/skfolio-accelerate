"""skfolio-accelerate: drop-in amortized ``cross_val_predict`` for skfolio.

This package accelerates large skfolio backtests without changing the portfolio
problem. The public entry point is :func:`cross_val_predict`, a drop-in
replacement for :func:`skfolio.model_selection.cross_val_predict`.

A call is compiled once into a :class:`~skfolio_accelerate.cv_plan.CVPlan`, then
executed. Compact OSQP / HiGHS / Clarabel engines and Parameterized MeanRisk
reuse amortize the *solver* for a subset of
:class:`~skfolio.optimization.MeanRisk`. Every other serial
:class:`~skfolio.optimization.BaseOptimization` still gets the same compiled
plan, contiguous slices, and assembly from ``weights_`` (skipping joblib,
train/test copies, and ``predict()``). A few estimators have trivial weights
and skip ``fit``; that is the same bookkeeping path, not a second optimizer.
Unmodified skfolio is used when the call options require it.

See the user guide for eligibility rules, backend reports, and hyperparameter
search with :func:`grid_search`.

Examples
--------
>>> from skfolio.model_selection import WalkForward
>>> from skfolio.optimization import MeanRisk
>>> from skfolio_accelerate import cross_val_predict
>>> # prediction = cross_val_predict(MeanRisk(), X, cv=WalkForward(...))
"""

from skfolio_accelerate.predict import (
    AccelerationReport,
    AccelerationWarning,
    cross_val_predict,
)
from skfolio_accelerate.scoring import (
    path_sharpes,
    ranking_precision_at_k,
    spearman_rank_correlation,
)
from skfolio_accelerate.search import GridSearchResult, grid_search

__all__ = [
    "AccelerationReport",
    "AccelerationWarning",
    "GridSearchResult",
    "cross_val_predict",
    "grid_search",
    "path_sharpes",
    "ranking_precision_at_k",
    "spearman_rank_correlation",
]

"""skfolio-accelerate: drop-in amortized ``cross_val_predict`` for skfolio.

This package accelerates large skfolio backtests without changing the portfolio
problem. The public entry point is :func:`cross_val_predict`, a drop-in
replacement for :func:`skfolio.model_selection.cross_val_predict`.

A call is compiled once into a :class:`~skfolio_accelerate.cv_plan.CVPlan`, then
executed by one of several backends:

* compact OSQP / Clarabel engines for a subset of
  :class:`~skfolio.optimization.MeanRisk`,
* closed-form weights for default
  :class:`~skfolio.optimization.EqualWeighted`,
  :class:`~skfolio.optimization.Random`, and
  :class:`~skfolio.optimization.InverseVolatility`,
* native ``fit`` plus weight assembly for other serial optimizers,
* unmodified skfolio when the call options require it.

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
    cross_val_predict,
    massive_cross_val_predict,
)
from skfolio_accelerate.scoring import (
    path_sharpes,
    ranking_precision_at_k,
    spearman_rank_correlation,
)
from skfolio_accelerate.search import GridSearchResult, grid_search

__all__ = [
    "AccelerationReport",
    "GridSearchResult",
    "cross_val_predict",
    "grid_search",
    "massive_cross_val_predict",
    "path_sharpes",
    "ranking_precision_at_k",
    "spearman_rank_correlation",
]

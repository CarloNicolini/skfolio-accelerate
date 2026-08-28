.. _grid_search_guide:

*********************
Hyperparameter search
*********************

.. currentmodule:: skfolio_accelerate

:func:`grid_search` evaluates a MeanRisk parameter grid with one shared
CV plan. Compact-eligible candidates reuse OSQP / Clarabel and empirical
moments. Other MeanRisk candidates (ratio objectives, ``max_cvar``,
``linear_constraints``, ...) reuse Parameterized CVXPY problems. Candidates
are scored by mean out-of-sample path Sharpe computed from fold weights.
Only the winning parameter set is materialized into Portfolio objects.

.. code-block:: python

    import numpy as np
    from skfolio.optimization import MeanRisk
    from skfolio_accelerate import grid_search

    result = grid_search(
        MeanRisk(),
        X,
        {"l2_coef": np.logspace(-5, -1, 16)},
        cv=cv,
    )
    print(result.best_params_)
    print(result.best_score_)
    prediction = result.best_prediction_

Every compact-eligible candidate stays on that engine. MeanRisk grids that
leave the boxed subset use the sequential CVXPY backend instead. For
non-MeanRisk estimators use skfolio's ``OnlineGridSearch`` or sklearn's
``GridSearchCV``.

The returned :class:`GridSearchResult` also carries
``acceleration_report_`` with backend ``"compact-grid"`` or
``"sequential-grid"``.

.. _grid_search_guide:

*********************
Hyperparameter search
*********************

.. currentmodule:: skfolio_accelerate

:func:`grid_search` evaluates a compact MeanRisk parameter grid with one shared
CV plan and one shared moment pass. Candidates are scored by mean out-of-sample
path Sharpe computed from fold weights. Only the winning parameter set is
materialized into Portfolio objects.

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

Every candidate must be compact-eligible. For general estimators use skfolio's
``OnlineGridSearch`` or sklearn's ``GridSearchCV``.

If you stay on sklearn ``GridSearchCV`` / ``cross_val_score``, set
``n_jobs=-1`` and pin solver threads to 1 (see :ref:`backends`). For
exploratory native MeanRisk search, Clarabel
``solver_params={"tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4}`` cut a 252-day
CVaR ``fit`` from 18.4 ms to 13.0 ms with unchanged weights; on 3-year and
20-year windows the same change did not move wall time because CVXPY
construction dominates. Compact :func:`grid_search` already shares one
compiled OSQP / Clarabel problem at the tight default tolerances: eight
``l2_coef`` candidates on a 20-year WalkForward were 34× faster than a
native ``ParameterGrid`` with ``n_jobs=-1``.

The returned :class:`GridSearchResult` also carries
``acceleration_report_`` with backend ``"compact-grid"``.

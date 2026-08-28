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

The returned :class:`GridSearchResult` also carries
``acceleration_report_`` with backend ``"compact-grid"``.

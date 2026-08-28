.. _quickstart:

***********
Quick start
***********

.. currentmodule:: skfolio_accelerate

.. warning::

   **Experimental library.** Validate results against native skfolio. See
   :ref:`methods` for the mathematics and assumptions behind the speedups.

Replace skfolio's ``cross_val_predict`` import and keep the rest of the backtest
unchanged:

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.model_selection import WalkForward
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns
    from skfolio_accelerate import cross_val_predict

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    cv = WalkForward(train_size=252, test_size=21)
    prediction = cross_val_predict(MeanRisk(), X, cv=cv)

The prediction type matches skfolio: a
:class:`~skfolio.portfolio.MultiPeriodPortfolio` for single-path CV, or a
:class:`~skfolio.population.Population` for combinatorial / multi-path CV.

Inspect the selected backend
****************************

Pass ``return_report=True`` to learn which path ran:

.. code-block:: python

    prediction, report = cross_val_predict(
        MeanRisk(),
        X,
        cv=cv,
        return_report=True,
    )
    print(report.backend)  # "osqp", "clarabel", "closed-form", ...
    print(report)

See :ref:`backends` for the full list of backends and eligibility rules.

What is accelerated
*******************

The fast path applies to:

* :class:`~skfolio.optimization.MeanRisk` with the default empirical prior,
* minimize-risk or maximize-utility objectives,
* a fixed equality budget, ordinary weight bounds, and optional L2
  regularization,
* variance (OSQP) and the supported scenario risks (Clarabel),
* default :class:`~skfolio.optimization.EqualWeighted`,
  :class:`~skfolio.optimization.Random`, and default-empirical
  :class:`~skfolio.optimization.InverseVolatility`.

Other serial :class:`~skfolio.optimization.BaseOptimization` estimators still
call native ``fit``, then assemble portfolios from ``weights_``. Pipelines,
sequential ``previous_weights``, ``raise_on_failure=False``, parallel
``n_jobs``, and ``entry_rebalancing_params`` use unmodified skfolio.

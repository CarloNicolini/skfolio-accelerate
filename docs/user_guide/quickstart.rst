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
    print(report.backend)  # "osqp", "clarabel", "cvxpy-sequential", ...
    print(report.reason)
    print(report)

See :ref:`backends` for the full list of backends and eligibility rules.

What is accelerated
*******************

``backend="auto"`` covers every :class:`~skfolio.optimization.MeanRisk`
``ObjectiveFunction`` × ``RiskMeasure`` pair, including WalkForward,
MultipleRandomizedCV, and CombinatorialPurgedCV:

* boxed variance uses compact OSQP; boxed scenario risks use compact Clarabel;
* other MeanRisk configurations (standard deviation, Ulcer,
  ``MAXIMIZE_RETURN``, risk limits, linear constraints, fees, L1, …) reuse
  skfolio's CVXPY problem when the training shape is fixed;
* ``MAXIMIZE_RATIO``, transaction costs, custom CVXPY hooks, and MeanRisk
  subclasses stay on native ``fit`` plus assembly;
* default :class:`~skfolio.optimization.EqualWeighted`,
  :class:`~skfolio.optimization.Random`, and default-empirical
  :class:`~skfolio.optimization.InverseVolatility` use closed-form weights.

Other serial :class:`~skfolio.optimization.BaseOptimization` estimators still
call native ``fit``, then assemble portfolios from ``weights_``. Pipelines,
``raise_on_failure=False``, parallel ``n_jobs``, and
``entry_rebalancing_params`` use unmodified skfolio.

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

* boxed ``MAXIMIZE_RETURN`` uses an analytic L2-regularized projection;
  boxed variance uses compact OSQP; boxed scenario LPs (MAD, CVaR, …) on
  WalkForward / MultipleRandomizedCV use persistent HiGHS; CombinatorialPurgedCV
  MAD/FLPM fall back to native skfolio with :class:`~skfolio_accelerate.AccelerationWarning`;
  remaining boxed scenario cones use compact Clarabel;
* other MeanRisk configurations (standard deviation, Ulcer, risk limits,
  linear constraints, fees, L1, …) reuse
  skfolio's CVXPY problem when the training shape is fixed;
* ``MAXIMIZE_RATIO``, transaction costs, custom CVXPY hooks, MeanRisk
  subclasses, and every other serial
  :class:`~skfolio.optimization.BaseOptimization` still call native ``fit``
  (when there is something to fit) and then use that same compiled plan and
  weight assembly. Cheap closed-form weights skip ``fit``; the speedup is the
  shared CV bookkeeping, not a solver.

Pipelines, ``raise_on_failure=False``, parallel ``n_jobs``, and
``entry_rebalancing_params`` use unmodified skfolio.

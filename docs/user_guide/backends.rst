.. _backends:

********************
Backends and reports
********************

.. currentmodule:: skfolio_accelerate

:func:`cross_val_predict` classifies each call once, then selects a backend.

Backend names
*************

======================  ============================================================
``backend``             Meaning
======================  ============================================================
``osqp``                Compact mean-variance QP
``clarabel``            Compact scenario LP / QP / SOCP / exponential cone
``cvxpy-sequential``    MeanRisk CVXPY graph reused via Parameters (full constraints)
``closed-form``         EqualWeighted, Random, or InverseVolatility weights
``fit-assemble``        Native ``fit`` + assembly from ``weights_``
``sklearn``             Unmodified skfolio ``cross_val_predict``
``compact-grid``        Shared compact path inside :func:`grid_search`
``sequential-grid``     Parameterized MeanRisk grid inside :func:`grid_search`
======================  ============================================================

Force a policy with the keyword-only ``backend`` argument:

* ``"auto"`` (default) — choose the best eligible path,
* ``"compact"`` — require compact / closed-form; raise if ineligible,
* ``"cvxpy-sequential"`` — require Parameterized MeanRisk reuse; raise if
  ineligible,
* ``"sklearn"`` — always call native skfolio.

AccelerationReport
******************

:class:`AccelerationReport` records timing and reuse counters:

.. code-block:: python

    prediction, report = cross_val_predict(
        MeanRisk(), X, cv=cv, return_report=True
    )
    print(report.n_prior_fits, report.n_prior_updates, report.n_warm_starts)
    print(report.fallback_reason)

When a compact numerical solve fails, the package retries with native ``fit``
and the assembled path when allowed, rather than returning an accelerator-only
failure. ``fallback_reason`` explains that transition.

Eligibility helpers
*******************

Inspect gates without running a backtest:

.. code-block:: python

    from skfolio_accelerate.predict import classify_call

    caps = classify_call(MeanRisk(), cv=cv)
    assert caps.can_compact
    assert caps.can_assemble

Compact and assemble are independent. A MeanRisk configuration may be
ineligible for the cone engines yet still eligible for Parameterized CVXPY
reuse (``cvxpy-sequential``) or serial fit-assemble.

``cvxpy-sequential`` keeps skfolio's own constraint and risk construction. Fold
data (``mu``, returns, covariance square root, previous weights) is injected as
``cp.Parameter`` objects so WalkForward windows with a fixed training length
reuse one compiled problem. Topology changes (expanding ``TimeSeriesSplit``,
a different number of assets) rebuild the graph. Custom ``add_constraints`` /
``add_objective`` hooks stay on fit-assemble so a callable that closes over a
particular window is not frozen into the first fold.

.. danger::

    Do not reuse mutable estimator or solver state across unrelated calls
    without proving equivalence. All reuse in this package is local to one
    :func:`cross_val_predict` / :func:`grid_search` invocation.

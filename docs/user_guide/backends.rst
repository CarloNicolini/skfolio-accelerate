.. _backends:

********************
Backends and reports
********************

.. currentmodule:: skfolio_accelerate

:func:`cross_val_predict` classifies each call once, then selects a backend.

Backend names
*************

=================  ============================================================
``backend``        Meaning
=================  ============================================================
``osqp``           Compact mean-variance QP
``clarabel``       Compact scenario LP / QP / SOCP / exponential cone
``cosmo``          Optional COSMO.jl ADMM engine (variance or scenario)
``closed-form``    EqualWeighted, Random, or InverseVolatility weights
``fit-assemble``   Native ``fit`` + assembly from ``weights_``
``sklearn``        Unmodified skfolio ``cross_val_predict``
``compact-grid``   Shared compact path inside :func:`grid_search`
=================  ============================================================

Force a policy with the keyword-only ``backend`` argument:

* ``"auto"`` (default) — choose the best eligible path,
* ``"compact"`` — require compact / closed-form; raise if ineligible,
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
ineligible for the cone engines yet still eligible for serial fit-assemble.

Optional COSMO backend
**********************

Pass ``MeanRisk(solver="COSMO")`` to use COSMO.jl instead of OSQP / Clarabel
on the compact subset. The Julia runtime is started once per process and the
ADMM workspace is reused across folds of one :func:`cross_val_predict` call.
Default ``MeanRisk()`` is unchanged.

See :ref:`install` for the optional ``[cosmo]`` extra. If COSMO is requested
but juliacall / COSMO.jl are missing, compaction is skipped rather than
failing at import time.

COSMO is an ADMM solver. Variance, CVaR, and second-order cones typically
match the Clarabel / OSQP weights to the compact test gates. Scenario LPs
(MAD) and exponential cones (EVaR) use COSMO's default ``1e-5`` residuals
and can differ from Clarabel at the ``1e-3`` to ``1e-2`` level; they stay
opt-in so those results can be measured rather than made the default.

.. danger::

    Do not reuse mutable estimator or solver state across unrelated calls
    without proving equivalence. All reuse in this package is local to one
    :func:`cross_val_predict` / :func:`grid_search` invocation.

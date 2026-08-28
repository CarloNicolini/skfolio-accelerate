.. _backends:

********************
Backends and reports
********************

.. currentmodule:: skfolio_accelerate

:func:`cross_val_predict` classifies each call once. Leave ``backend`` at
``"auto"``; the library picks an engine and records it on
:class:`AccelerationReport` (``report.backend`` and ``report.reason``).

``backend="auto"`` selects the first eligible engine:

1. compact OSQP / Clarabel, or closed-form weights,
2. Parameterized CVXPY reuse for other MeanRisk configurations with a fixed
   training shape (``mu``, returns, and covariance square-root are
   ``cp.Parameter``; skfolio still builds every constraint),
3. native ``fit`` plus assembly from ``weights_``,
4. unmodified skfolio.

Ratio homogenization, transaction costs, custom CVXPY hooks, and MeanRisk
subclasses stay on fit-assemble or native skfolio. You do not pass an engine
name in application code.

Backend names
*************

======================  ============================================================
``backend``             Meaning
======================  ============================================================
``osqp``                Compact mean-variance QP
``clarabel``            Compact scenario LP / QP / SOCP / exponential cone
``cvxpy-sequential``    Reuse skfolio's MeanRisk CVXPY problem across folds
``closed-form``         EqualWeighted, Random, or InverseVolatility weights
``fit-assemble``        Native ``fit`` + assembly from ``weights_``
``sklearn``             Unmodified skfolio ``cross_val_predict``
``compact-grid``        Shared compact path inside :func:`grid_search`
======================  ============================================================

Force a policy with the keyword-only ``backend`` argument:

* ``"auto"`` (default) — the order above,
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

Compact, sequential, and assemble are independent. A MeanRisk configuration
may be ineligible for the cone engines yet still eligible for Parameterized
CVXPY reuse or serial fit-assemble.

.. danger::

    Do not reuse mutable estimator or solver state across unrelated calls
    without proving equivalence. All reuse in this package is local to one
    :func:`cross_val_predict` / :func:`grid_search` invocation.

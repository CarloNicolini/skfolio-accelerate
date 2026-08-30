.. _backends:

********************
Backends and reports
********************

.. currentmodule:: skfolio_accelerate

.. warning::

   **Experimental.** Backend selection and fallback behaviour may evolve.
   Inspect :class:`AccelerationReport` and compare against native skfolio
   before trusting a new configuration. Mathematical assumptions are listed
   in :ref:`methods`.

:func:`cross_val_predict` classifies each call once. Leave ``backend`` at
``"auto"``; the library picks an engine and records it on
:class:`AccelerationReport` (``report.backend`` and ``report.reason``).

``backend="auto"`` selects the first eligible engine:

1. analytic maximum-return or compact OSQP / HiGHS / Clarabel for boxed MeanRisk,
2. Parameterized CVXPY reuse for other MeanRisk configurations with a fixed
   training shape (``mu``, returns, and covariance square-root are
   ``cp.Parameter``; skfolio still builds every constraint),
3. serial assembly from ``weights_`` (native ``fit``, unless the weights are
   a trivial closed-form formula),
4. unmodified skfolio.

The compiled CV plan, contiguous slices, and portfolio assembly in (3) are
shared by every serial estimator. They are not an EqualWeighted-specific
optimization. Ratio homogenization, transaction costs, custom CVXPY hooks,
and MeanRisk subclasses stay on that assembly path or on native skfolio.
Boxed MAD and FLPM on
:class:`~skfolio.model_selection.CombinatorialPurgedCV` also use native
skfolio: a persistent simplex basis does not speed up non-rolling long
training windows. The call emits :class:`~skfolio_accelerate.AccelerationWarning`.
You do not pass an engine name in application code.

Backend names
*************

======================  ============================================================
``backend``             Meaning
======================  ============================================================
``max-return``          Analytic box-and-budget maximum return with L2 regularization
``osqp``                Compact mean-variance QP
``highs``               Compact scenario LP with persistent HiGHS simplex
``clarabel``            Compact standard-deviation / scenario cone problem
``cvxpy-sequential``    Reuse skfolio's MeanRisk CVXPY problem across folds
``closed-form``         Trivial weights on the shared serial assembly path
``fit-assemble``        Native ``fit`` + the same assembly from ``weights_``
``sklearn``             Unmodified skfolio ``cross_val_predict``
``compact-grid``        Shared compact path inside :func:`grid_search`
======================  ============================================================

Force a policy with the keyword-only ``backend`` argument:

* ``"auto"`` (default) — the order above,
* ``"compact"`` — require compact MeanRisk / trivial-weight assembly; raise if ineligible,
* ``"cvxpy-sequential"`` — require Parameterized MeanRisk reuse; raise if ineligible,
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

Parallel folds and solver threads
*********************************

Amortized backends require ``n_jobs in {None, 1}``. MRC paths and CPCV
combinations are independent, so native skfolio can use joblib: pass
``n_jobs=-1`` and this package forwards the call to unmodified skfolio.

When you use that native path (or sklearn ``GridSearchCV`` /
``cross_val_score``), cap solver-internal threads to 1 so workers do not
oversubscribe cores:

.. code-block:: python

    import os
    from skfolio.optimization import MeanRisk

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")

    estimator = MeanRisk(solver_params={"max_threads": 1})

:func:`cross_val_predict` already sets those environment variables, and
compact Clarabel uses ``max_threads=1``. On a 4-core 20-year MRC, native
joblib is about 3.5× versus serial native. Serial compact OSQP still beats
that parallel run by ~13×; serial compact CVaR only ties or slightly wins
on MRC, and 45 independent CVaR cones prefer joblib. Sequential std and
``MAXIMIZE_RATIO`` lose to ``n_jobs=-1``.

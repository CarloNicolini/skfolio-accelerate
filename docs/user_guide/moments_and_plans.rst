.. _moments_and_plans:

********************
Moments and CV plans
********************

.. currentmodule:: skfolio_accelerate

This page is the operational companion to :ref:`methods`. It shows how to
inspect compiled plans and moment caches; the mathematical derivations and
assumptions live on the methods page.

CV plans
********

:func:`~skfolio_accelerate.cv_plan.compile_cv_plan` is the only place that
iterates ``splitter.split``. It produces an immutable
:class:`~skfolio_accelerate.cv_plan.CVPlan` consumed by every backend:

.. code-block:: python

    from skfolio_accelerate.cv_plan import compile_cv_plan

    plan = compile_cv_plan(cv, X)
    print(plan.kind, plan.n_splits, plan.n_paths)
    for batch in plan.path_batches():
        print(len(batch))

* WalkForward / KFold — single path, ``kind="walk_forward"`` or ``"kfold"``.
* CombinatorialPurgedCV — multi-path with fold blocks for moment reuse.
* MultipleRandomizedCV — one batch per randomized asset subset.

Mutable splitters that advance RNG state inside ``split`` are consumed exactly
once. Native fallback keeps a deepcopy of the original splitter taken *before*
compilation.

Empirical moments
*****************

The moment cache stores sufficient statistics

.. math::

    n,\quad s = \sum_t x_t,\quad G = X^\top X

and forms the unbiased sample covariance only when needed:

.. math::

    \mu = s / n,\qquad
    \Sigma = \bigl(G - s s^\top / n\bigr) / (n - 1)

This matches ``numpy.cov(..., ddof=1)`` and skfolio's default empirical
covariance *before* the optional nearest-PD projection.

When a WalkForward window slides from ``[a, b)`` to ``[a', b')``, dropped and
added blocks ``D`` and ``A`` update the cache exactly:

.. math::

    s' = s - \sum_{x \in D} x + \sum_{x \in A} x,\qquad
    G' = G - D^\top D + A^\top A.

CPCV fold blocks are precomputed once and summed (with purge / embargo rows
subtracted). Scenario risks additionally keep the training window (as a view
when rows are contiguous).

.. note::

   Floating-point cancellation after many slides can in principle erode
   positive-definiteness. The OSQP engine retries with a small diagonal jitter;
   the statistical model is unchanged. See :ref:`methods` for the full list of
   assumptions (default empirical prior, PD regime, fixed constraints, …).

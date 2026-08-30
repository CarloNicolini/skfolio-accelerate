.. _user_guide:

==========
User Guide
==========

.. warning::

   **Experimental library.** ``skfolio-accelerate`` is research software.
   Prefer native skfolio whenever a workload falls outside the documented
   eligibility rules, and compare results before relying on rankings or
   production signals. Details are in :ref:`methods`.

`skfolio-accelerate` is a small companion library for
`skfolio <https://skfolio.org>`_. It accelerates serial
:func:`~skfolio.model_selection.cross_val_predict` workloads by removing
repeated work that does not change the portfolio problem:

* the CV plan is compiled once; test portfolios are assembled from fold
  weights (contiguous slices, no joblib, no per-fold ``predict()``) — this
  applies to every serial :class:`~skfolio.optimization.BaseOptimization`,
  not only MeanRisk,
* overlapping empirical moments are updated from sufficient statistics,
* compact OSQP / HiGHS / Clarabel engines reuse a fixed topology across folds,
* MeanRisk's own CVXPY graph is reused for the rest of the objective × risk
  surface when the training shape is fixed.

The public API is intentionally narrow. Prefer the user guide pages below for
eligibility rules and the mathematics behind the speedups, then the
:ref:`API reference <api>` for parameter details.

.. toctree::
    :maxdepth: 2

    Installation <install>
    Quick start <quickstart>
    Methods, mathematics, and assumptions <methods>
    Backends and reports <backends>
    Moments and CV plans <moments_and_plans>
    Hyperparameter search <grid_search>
    Ranking checks <ranking>

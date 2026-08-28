.. _user_guide:

==========
User Guide
==========

`skfolio-accelerate` is a small companion library for
`skfolio <https://skfolio.org>`_. It accelerates serial
:func:`~skfolio.model_selection.cross_val_predict` workloads by removing
repeated work that does not change the portfolio problem:

* overlapping empirical moments are updated from sufficient statistics,
* compact OSQP / Clarabel engines reuse a fixed cone topology across folds,
* MeanRisk's own CVXPY graph is reused when extra constraints keep a fixed
  training shape,
* closed-form estimators skip ``fit`` entirely,
* other serial optimizers still call native ``fit``, then assemble test
  portfolios from ``weights_``.

The public API is intentionally narrow. Prefer the user guide pages below for
eligibility rules, then the :ref:`API reference <api>` for parameter details.

.. toctree::
    :maxdepth: 2

    Installation <install>
    Quick start <quickstart>
    Backends and reports <backends>
    Moments and CV plans <moments_and_plans>
    Hyperparameter search <grid_search>
    Ranking checks <ranking>

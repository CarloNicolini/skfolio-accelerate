.. _user_guide:

==========
User Guide
==========

.. warning::

   **Experimental library.** ``skfolio-accelerate`` is research software.
   Prefer native skfolio whenever a workload falls outside the documented
   compact subset, and compare results before relying on rankings or
   production signals. Details are in :ref:`methods`.

`skfolio-accelerate` is a small companion library for
`skfolio <https://skfolio.org>`_. It accelerates serial
:func:`~skfolio.model_selection.cross_val_predict` workloads by removing
repeated work that does not change the portfolio problem:

* overlapping empirical moments are updated from sufficient statistics,
* compact OSQP / Clarabel engines reuse a fixed cone topology across folds,
* closed-form estimators skip ``fit`` entirely,
* other serial optimizers still call native ``fit``, then assemble test
  portfolios from ``weights_``.

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

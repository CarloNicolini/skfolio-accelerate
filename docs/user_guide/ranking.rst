.. _ranking:

**************
Ranking checks
**************

.. currentmodule:: skfolio_accelerate

When validating a new workload against native skfolio, numerical closeness of
weights or Sharpes does not automatically mean that a ranking is unchanged.

.. code-block:: python

    from skfolio_accelerate import (
        path_sharpes,
        ranking_precision_at_k,
        spearman_rank_correlation,
    )

    reference = path_sharpes(native_prediction)
    observed = path_sharpes(accelerated_prediction)

    precision = ranking_precision_at_k(reference, observed, k=5)
    correlation = spearman_rank_correlation(reference, observed)

    # Treat score gaps below the numerical tolerance as ties.
    tie_aware = ranking_precision_at_k(
        reference,
        observed,
        k=5,
        score_tolerance=1e-6,
    )

* :func:`ranking_precision_at_k` checks whether the native best set remains in
  the accelerated best set.
* :func:`spearman_rank_correlation` compares the full ordering and is ``nan``
  when every score is the same after tie grouping.

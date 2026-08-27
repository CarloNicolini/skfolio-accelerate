.. skfolio-accelerate documentation master file

==================
skfolio-accelerate
==================

`skfolio-accelerate` makes large skfolio backtests less repetitive. It provides
a drop-in replacement for :func:`skfolio.model_selection.cross_val_predict`, so
an existing backtest usually needs one import change:

.. code-block:: python

    from skfolio.model_selection import WalkForward
    from skfolio.optimization import MeanRisk
    from skfolio_accelerate import cross_val_predict

    cv = WalkForward(train_size=2 * 252, test_size=21)
    prediction = cross_val_predict(MeanRisk(), X, cv=cv)

The result is still a skfolio :class:`~skfolio.portfolio.MultiPeriodPortfolio`
or :class:`~skfolio.population.Population`.

Internally a call is compiled once into a CV plan, then executed: overlapping
training moments are updated from sufficient statistics, a compact OSQP or
Clarabel engine reuses a fixed problem shape across folds, and test portfolios
are assembled from ``weights_``.

.. toctree::
   :maxdepth: 2
   :hidden:

   User guide <user_guide/index>
   Examples <auto_examples/index>
   API Reference <api>

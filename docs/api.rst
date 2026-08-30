:og:description: API reference for skfolio-accelerate: amortized cross_val_predict, grid search, scoring, and CV plans.

.. meta::
    :description: API reference for skfolio-accelerate: amortized cross_val_predict,
                  grid search, scoring, and CV plans.

.. _api:

=============
API Reference
=============

This is the class and function reference of `skfolio-accelerate`. Please refer to
the :ref:`user guide <user_guide>` for eligibility rules and usage patterns.

The supported application surface is :func:`cross_val_predict`,
:class:`AccelerationReport`, :func:`grid_search`, and the ranking helpers.
Capability and CV-plan helpers below are documented because the user guide
refers to them; solver engines are not a public API.

.. _predict_ref:

:mod:`skfolio_accelerate.predict`: Cross-validated prediction
=============================================================

.. automodule:: skfolio_accelerate.predict
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: skfolio_accelerate

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    AccelerationReport
    AccelerationWarning

.. currentmodule:: skfolio_accelerate.predict

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    CallCapabilities

Functions
---------
.. currentmodule:: skfolio_accelerate

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cross_val_predict

.. currentmodule:: skfolio_accelerate.predict

.. autosummary::
    :toctree: generated/
    :template: function.rst

    classify_call

.. _search_ref:

:mod:`skfolio_accelerate.search`: Hyperparameter search
=======================================================

.. automodule:: skfolio_accelerate.search
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    GridSearchResult

.. autosummary::
    :toctree: generated/
    :template: function.rst

    grid_search

.. _scoring_ref:

:mod:`skfolio_accelerate.scoring`: Assembly and ranking
=======================================================

.. automodule:: skfolio_accelerate.scoring
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate

.. autosummary::
    :toctree: generated/
    :template: function.rst

    path_sharpes
    ranking_precision_at_k
    spearman_rank_correlation

.. _cv_plan_ref:

:mod:`skfolio_accelerate.cv_plan`: Compiled CV plans
====================================================

.. automodule:: skfolio_accelerate.cv_plan
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate.cv_plan

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    CVPlan

.. autosummary::
    :toctree: generated/
    :template: function.rst

    compile_cv_plan

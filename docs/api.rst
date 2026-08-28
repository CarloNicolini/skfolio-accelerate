:og:description: API reference for skfolio-accelerate: amortized cross_val_predict, grid search, scoring, and compact MeanRisk engines.

.. meta::
    :description: API reference for skfolio-accelerate: amortized cross_val_predict,
                  grid search, scoring, and compact MeanRisk engines.

.. _api:

=============
API Reference
=============

This is the class and function reference of `skfolio-accelerate`. Please refer to
the :ref:`user guide <user_guide>` for eligibility rules and usage patterns.

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

.. currentmodule:: skfolio_accelerate.predict

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    CallCapabilities
    FoldBatchResult

Functions
---------
.. currentmodule:: skfolio_accelerate

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cross_val_predict
    classify_call

.. currentmodule:: skfolio_accelerate.predict

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resolve_backend
    compact_engine_name
    blocked_reason
    compact_blocked_reason
    sequential_blocked_reason
    assemble_blocked_reason
    solve_compact_folds
    solve_sequential_folds
    closed_form_weights
    fit_native_weights
    merge_batch_results

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

.. currentmodule:: skfolio_accelerate.scoring

.. autosummary::
    :toctree: generated/
    :template: function.rst

    assemble_prediction
    make_segment_portfolio
    path_sharpes_from_weights
    window_view

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
    FoldSpec

.. autosummary::
    :toctree: generated/
    :template: function.rst

    compile_cv_plan
    cpcv_fold_blocks
    chains_previous_weights

.. _moments_ref:

:mod:`skfolio_accelerate.moments`: Empirical moments
====================================================

.. automodule:: skfolio_accelerate.moments
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate.moments

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    FoldMoments
    OverlapMomentCache
    PathMomentSession

.. autosummary::
    :toctree: generated/
    :template: function.rst

    path_moment_session
    empirical_from_window
    empirical_from_stats
    is_default_empirical

.. _compact_ref:

:mod:`skfolio_accelerate.compact`: Compact MeanRisk engines
===========================================================

.. automodule:: skfolio_accelerate.compact
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate.compact

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    MeanRiskSpec
    EngineCache
    MinVarianceOSQP
    CVaRClarabel
    ScenarioClarabel

.. autosummary::
    :toctree: generated/
    :template: function.rst

    estimator_spec
    make_compact_engine

.. _mean_risk_problem_ref:

:mod:`skfolio_accelerate.mean_risk_problem`: Sequential MeanRisk
================================================================

.. automodule:: skfolio_accelerate.mean_risk_problem
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate.mean_risk_problem

.. autosummary::
    :nosignatures:
    :toctree: generated/
    :template: class.rst

    ParametricMeanRisk
    SequentialProblemCache
    ProblemTopology
    CompiledProblem

.. autosummary::
    :toctree: generated/
    :template: function.rst

    as_parametric
    problem_topology
    needs_observation_dimension
    can_reuse_distribution

.. _flagship_ref:

:mod:`skfolio_accelerate.flagship`: Benchmark workloads
=======================================================

.. automodule:: skfolio_accelerate.flagship
    :no-members:
    :no-inherited-members:

.. currentmodule:: skfolio_accelerate.flagship

.. autosummary::
    :toctree: generated/
    :template: function.rst

    factor_returns
    make_mrc
    make_cpcv

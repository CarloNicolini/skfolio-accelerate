.. _methods:

****************************************
Methods, mathematics, and assumptions
****************************************

.. currentmodule:: skfolio_accelerate

.. warning::

   **Experimental library.** ``skfolio-accelerate`` is research software.
   Formulations and eligibility gates may change between releases. Treat every
   accelerated result as provisional until you have compared it to native
   skfolio on the same workload. Prefer the native path whenever the problem
   falls outside the documented eligibility rules. Compact OSQP / HiGHS /
   Clarabel mathematics is on this page; Parameterized MeanRisk reuse is
   described in :ref:`backends`.

This page explains *why* the library is faster, *what* mathematics it reuses,
and *which* assumptions that reuse relies on. It does not change the investment
problem: every compact path is meant to solve the same MeanRisk program that
skfolio builds with CVXPY.

What is amortized
*****************

A serial ``cross_val_predict`` call spends most of its time on work that is
identical, or nearly identical, across folds:

1. **CV bookkeeping** — cloning estimators, slicing DataFrames, wrapping
   ``n_jobs=1`` in joblib, and constructing ``Portfolio`` objects.
2. **Empirical moments** — recomputing ``μ`` and ``Σ`` on overlapping training
   windows that share almost all rows with the previous fold.
3. **Cone / QP construction** — rebuilding a CVXPY graph (and its Clarabel /
   OSQP backend) whose *topology* does not change when only ``(μ, Σ)`` or the
   scenario matrix changes.

``skfolio-accelerate`` attacks each of these once per call:

* :func:`~skfolio_accelerate.cv_plan.compile_cv_plan` consumes
  ``splitter.split`` exactly once and stores immutable fold indices.
* Sufficient statistics ``(n, s, G)`` update overlapping moments with exact
  rank-k add/drop operations.
* Compact OSQP / HiGHS / Clarabel engines keep a fixed topology and warm-start
  across folds while ``(n_assets, T)`` stay constant. HiGHS additionally
  restores the previous simplex basis after incremental scenario updates.
* Test portfolios are assembled from ``weights_``, skipping native
  ``predict()`` construction on the serial path.

All reuse is **local to one call**. The package does not keep process-wide
caches of returns, estimators, or solver workspaces.

Compiled CV plans
*****************

:class:`~skfolio_accelerate.cv_plan.CVPlan` records every
``(train_idx, test_idx)`` pair, plus CPCV fold-block identifiers and MRC path
batches when applicable. Downstream backends never call ``split`` again.

Mutable splitters that advance RNG state inside ``split`` (notably
:class:`~skfolio.model_selection.MultipleRandomizedCV`) are therefore consumed
exactly once. The native fallback keeps a deepcopy of the original splitter
taken *before* compilation so a retry still sees a fresh RNG.

Empirical moments from sufficient statistics
********************************************

For a training matrix ``X ∈ ℝ^{T×n}`` the cache stores

.. math::

   n = T,\qquad
   s = \sum_{t=1}^{T} x_t ∈ ℝ^{n},\qquad
   G = X^\top X ∈ ℝ^{n×n}.

The unbiased sample estimators used by the compact engines are

.. math::

   \mu = \frac{s}{n},\qquad
   \Sigma = \frac{1}{n - 1}\Bigl(G - \frac{s s^\top}{n}\Bigr).

These match ``numpy.cov(..., ddof=1)`` and skfolio's default empirical
covariance *before* the optional nearest-positive-definite projection.

Rank-k sliding updates
======================

When a WalkForward window slides from ``[a, b)`` to ``[a', b')``, only the
symmetric difference of rows changes. Let ``D`` be the dropped block and
``A`` the added block. Then

.. math::

   s' = s - \sum_{x ∈ D} x + \sum_{x ∈ A} x,

.. math::

   G' = G - D^\top D + A^\top A,

with ``n' = n - |D| + |A|``. The update is algebraically exact; it is not a
different statistical estimator. Floating-point cancellation can in principle
erode positive-definiteness after many slides; the OSQP engine then retries
with a small diagonal jitter rather than changing the model.

CPCV fold blocks
================

:class:`~skfolio.model_selection.CombinatorialPurgedCV` reuses contiguous fold
blocks. The library precomputes ``(n, s, G)`` once per block, sums the blocks
that form a training set, and subtracts purge / embargo rows. Scenario risks
additionally keep the training window (as a view when rows are contiguous).

Compact mean-variance (OSQP)
****************************

For variance MeanRisk the portfolio problem is the boxed QP

.. math::

   \begin{aligned}
   \underset{w}{\mathrm{minimize}}
     &\quad w^\top \Sigma w + \ell_2 \|w\|_2^2
        && \text{(minimize risk)} \\
   \underset{w}{\mathrm{minimize}}
     &\quad \lambda\, w^\top \Sigma w + \ell_2 \|w\|_2^2 - \mu^\top w
        && \text{(maximize utility)} \\
   \mathrm{subject\ to}
     &\quad \mathbf{1}^\top w = b,\quad
            \ell \le w \le u.
   \end{aligned}

skfolio implements variance as the square of a second-order cone of a
covariance square-root. When ``Σ`` is positive definite that cone is
equivalent to the quadratic above. OSQP uses the standard form
``½ xᵀ P x + qᵀ x`` with

.. math::

   P = 2\,\mathrm{scale}\,\Sigma + 2\,\ell_2\, I,
   \qquad
   q =
   \begin{cases}
   0 & \text{minimize risk}, \\
   -\mu & \text{maximize utility},
   \end{cases}

where ``scale = 1`` or ``risk_aversion`` respectively. The constraint matrix
``A`` (budget row plus bound identities) is built once. Only the upper
triangle of ``P`` and the linear term ``q`` are updated each fold, and the
previous primal/dual iterate is warm-started when allowed.

Compact scenario LPs (HiGHS)
*********************************

MAD, first lower partial moment, CVaR, and worst realization are linear
programs when ``l2_coef = 0``. Baking ``R − μ`` into every scenario coefficient
makes adjacent WalkForward folds look unrelated: a previous simplex basis then
needs as many pivots as a cold start, or more.

The compact HiGHS engine keeps an auxiliary portfolio-mean variable (MAD /
FLPM) or stores raw ``r_t`` (CVaR / worst realization). Overlapping
observations keep the same constraint rows and slack variables. A rolling step
of ``s`` overwrites ``s`` scenario rows plus the mean equality, restores the
previous optimal basis, and reoptimizes. Later WalkForward / MRC folds
therefore do ``T_{\mathrm{fold}}^{(k+1)} \ll T_{\mathrm{fold}}^{(1)}`` in
simplex iterations.

CombinatorialPurgedCV is different. Training sets are unions of blocks, not a
slide of ``s`` rows, so the previous MAD/FLPM basis is not a nearby vertex.
On 5,040 × 20 synthetic returns that persistent simplex was **0.51×** versus
native Clarabel. ``backend="auto"`` therefore emits
:class:`~skfolio_accelerate.AccelerationWarning` and uses unmodified skfolio
for boxed MAD and FLPM on CombinatorialPurgedCV. CVaR and worst realization
stay on HiGHS (they were not slower than native in the same study).

A diagonal ``l2_coef`` term makes the same measures QPs; those stay on
Clarabel.

Compact scenario cones (Clarabel)
*********************************

Semi-variance, drawdown measures, exponential-cone risks, and LP measures with
a quadratic ``l2_coef`` keep skfolio's QP / SOCP / exponential-cone
formulations:

* downside measures use skfolio's minimum acceptable return (asset mean when
  unset);
* drawdown uses the ordered, non-compounded recurrence
  ``v₀ = 0``, ``vₜ ≥ vₜ₋₁ − rₜ``, ``vₜ ≥ 0``;
* cone *types* and sparsity pattern are reused while ``(n_assets, T)`` are
  fixed; numeric returns are the fold-varying data.

Clarabel workspaces are updated in place when the API allows it; otherwise a
new solver is constructed with the same cone list. Compact eligibility already
forbids transaction costs, management fees, and a non-zero risk-free rate, so
those CVXPY terms that appear in skfolio are identically zero here.

Closed-form and fit-assemble paths
**********************************

Default :class:`~skfolio.optimization.EqualWeighted`,
:class:`~skfolio.optimization.Random`, and default-empirical
:class:`~skfolio.optimization.InverseVolatility` skip optimization entirely.
Their speedup is the removal of native CV bookkeeping (joblib, copies,
``predict()``), not a hidden solver trick.

Other serial optimizers (HRP, risk budgeting, ratio objectives, …) still call
native ``fit``, then assemble test portfolios from ``weights_``. That cuts the
same per-fold overhead without claiming a compact cone equivalence.

Assumptions and non-goals
*************************

The compact path is deliberately narrow. It assumes:

* **Default empirical prior.** Sample mean and ``ddof=1`` covariance. Custom
  priors, log-normal projection, investment horizons, rolling ``window_size``,
  ``assume_centered``, or non-default ``ddof`` force a fallback.
* **Nearest-PD projection may be skipped.** Compact ``Σ`` matches
  ``numpy.cov`` before skfolio's optional nearest-PD step. The two coincide
  when the sample covariance is already PD — the intended regime
  (typically ``T > n_assets``).
* **Fixed linear constraints.** Equality budget, ordinary scalar or per-asset
  weight bounds, optional L2 regularisation. No transaction costs, turnover
  constraints, management fees, or risk-free cash.
* **Minimize-risk or maximize-utility** MeanRisk objectives for the cone
  engines. Ratio objectives and risk limits use fit-assemble or native
  skfolio.
* **Constant problem shape across folds.** Warm starts and cone reuse require
  fixed ``(n_assets, T)`` inside one call. Changing scenario length forces a
  rebuild.
* **Serial execution.** Parallel ``n_jobs``, pipelines, sequential
  ``previous_weights``, ``raise_on_failure=False``, and
  ``entry_rebalancing_params`` stay on unmodified skfolio.

These boundaries are intentional. Reusing mutable estimator or solver state
without proving equivalence could silently solve a different investment
problem.

When a compact numerical solve fails, the package retries with native ``fit``
plus assembly when allowed, rather than returning an accelerator-only failure.
Use :class:`~skfolio_accelerate.AccelerationReport` (``return_report=True``)
to see which backend ran and why a fallback occurred.

Expected speedups
*****************

There is no single universal speedup factor. Measured results depend on data
shape and how many overlapping training windows you run:

* **Variance MeanRisk** on long WalkForward / MRC workloads is typically the
  largest win (tens of times faster) because the OSQP QP is reused.
* **Scenario risks** still rebuild or update a cone each fold; the same
  workloads are often only a few times faster, and short CPCV grids can be
  ~1×.
* **Closed-form** estimators look fast on tiny problems because native CV
  overhead dominates milliseconds of work; the saving is roughly constant per
  fold, not a multiplicative floor under a heavy CVXPY solve.

See the gallery examples after the usage tutorials for Plotly figures of
live and published speedup comparisons, and the project README for the full
benchmark tables.

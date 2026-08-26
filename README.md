# skfolio-accelerate

`skfolio-accelerate` makes large skfolio backtests less repetitive. It provides
a drop-in replacement for `skfolio.model_selection.cross_val_predict`, so an
existing backtest usually needs one import change:

```python
from skfolio.model_selection import WalkForward
from skfolio.optimization import MeanRisk
from skfolio_accelerate import cross_val_predict

cv = WalkForward(train_size=2 * 252, test_size=21)
prediction = cross_val_predict(MeanRisk(), X, cv=cv)
```

The result is still a skfolio `MultiPeriodPortfolio` or `Population`.

## What is being saved

Long backtests fit almost the same model many times. A monthly walk-forward
over twenty years contains hundreds of overlapping training windows, while a
randomized multi-path backtest can contain tens of thousands.

For compact-compatible MeanRisk and simple closed-form cases, this package
avoids doing all of the setup again:

- Empirical means and covariances are updated as a window moves instead of
  being recomputed from scratch. KFold reuses the observations shared by
  consecutive splits. CPCV builds each fold's sufficient statistics once,
  then adds and subtracts blocks; purged and embargoed rows are corrected
  exactly.
- Variance is sent directly to OSQP. Scenario LPs, QPs, second-order cones, and
  exponential cones are sent directly to Clarabel. Their fixed-dimension
  workspaces are updated as the window changes.
- Randomized paths with the same number of assets share one solver workspace.
- Hyperparameter candidates share the CV splits and empirical moments.
- Contiguous test periods are passed to skfolio as NumPy views, avoiding a
  copy for every portfolio segment.

Small, immutable pieces of solver structure are cached with `lru_cache`.
Returns, covariance matrices, fitted estimators, and CV plans are deliberately
not kept in a global cache: they are large, depend on mutable inputs, and can
silently become stale. Those values are reused only inside one call.

## Compatibility

The function accepts the same arguments as skfolio, including `y`, `method`,
`params`, `column_indices`, and `entry_rebalancing_params`. It works with
KFold, TimeSeriesSplit, WalkForward, CombinatorialPurgedCV,
MultipleRandomizedCV, integer `cv`, pipelines, and skfolio optimization
estimators.

The compact solver is used only for `MeanRisk` with minimize-risk or
maximize-utility objectives, the default empirical prior, and ordinary box
plus equality-budget constraints. The supported risks and exact formulations
are:

| Risk | Compact formulation | Solver |
|---|---|---|
| Variance | sample-covariance QP | OSQP |
| Semi-variance | `sum(u²)/(T-1)`, `u >= -(R-MAR)w`, `u >= 0` | Clarabel QP |
| Semi-deviation | same downside variables with `norm(u)/sqrt(T-1)` | Clarabel SOCP |
| MAD | `2 sum(u)/T`, `u >= -(R-MAR)w`, `u >= 0` | Clarabel LP/QP |
| First lower partial moment | `sum(u)/T` with the same downside epigraph | Clarabel LP/QP |
| Worst realization | `z >= -Rw` | Clarabel LP/QP |
| CVaR | `alpha + sum(u)/(T(1-beta))` | Clarabel LP/QP |
| Max/average drawdown and CDaR | ordered linear drawdown recurrence | Clarabel LP/QP |
| EVaR and EDaR | skfolio's perspective exponential-cone model | Clarabel |

`MAR` has skfolio's exact meaning: when no minimum acceptable return is given,
it is the fitted empirical mean, so the scenario matrix is `(R - mean(R))`.
Drawdown optimization uses skfolio's ordered, non-compounded recurrence.
Scenario rows, empirical moments, sparse topology, and solver iterates are
reused only inside the current call and only while dimensions are equivalent.

Default `EqualWeighted` and default-empirical `InverseVolatility` also use
closed-form paths. Inverse volatility shares the same per-call covariance
updates.

Everything else is passed to skfolio unchanged. This includes standard
deviation, ulcer index, Gini mean difference, ratio/return objectives,
mixed-integer/group/linear constraints, risk limits, transaction costs,
management fees, turnover, previous or target weights, custom priors,
uncertainty sets, factor data, custom solver parameters/scaling, pipelines,
and other optimizers. A compact numerical failure is also retried through
native skfolio and reported as fallback. Native errors are not relabeled as
accelerator errors.

This is an intentional correctness boundary, not a general acceleration claim.
For an arbitrary skfolio estimator, safely caching a fitted prior or solver
would require knowing which inputs, constraints, and state it owns. Reusing one
without that knowledge can silently solve a different portfolio problem.

Pass `return_report=True` if you want to see which path was selected:

```python
prediction, report = cross_val_predict(
    MeanRisk(),
    X,
    cv=cv,
    return_report=True,
)
print(report.backend)  # "osqp", "clarabel", "closed-form", or "sklearn"
```

## Checking portfolio rankings

Numerical closeness does not guarantee that portfolio selection is unchanged.
The package therefore exposes two small comparison helpers. Treat skfolio's
native scores as the reference:

```python
from skfolio_accelerate import (
    ranking_precision_at_k,
    spearman_rank_correlation,
)

precision = ranking_precision_at_k(reference_scores, accelerated_scores, k=5)
correlation = spearman_rank_correlation(reference_scores, accelerated_scores)

# Treat score gaps below the numerical tolerance as ties.
tie_aware = ranking_precision_at_k(
    reference_scores,
    accelerated_scores,
    k=5,
    score_tolerance=1e-6,
)
```

Precision@k measures how much of skfolio's top-k set is retained. With a score
tolerance, candidates tied with skfolio's kth score are accepted instead of
being penalized for an arbitrary numerical ordering. Spearman correlation
compares the complete ordering and can use the same tolerance to form tie
groups. Spearman is `nan` when either set is constant because no ranking exists.

## Parameter search

`grid_search` is intended for large grids that stay inside the compact
MeanRisk subset. It scores each candidate by the mean out-of-sample Sharpe
ratio across paths.

```python
import numpy as np
from skfolio_accelerate import grid_search

result = grid_search(
    MeanRisk(),
    X,
    {"l2_coef": np.logspace(-5, -1, 16)},
    cv=cv,
)

print(result.best_params_)
print(result.best_score_)
prediction = result.best_prediction_
```

Use skfolio's `OnlineGridSearch`, `OnlineRandomizedSearch`, or sklearn's
`GridSearchCV` for general estimator searches. The specialized function above
is faster because all candidates see one compiled CV plan and one stream of
moment updates.

## Benchmarks

`benchmark_matrix.py` runs each case in an isolated process. A sampling thread
records peak RSS for the process and all worker descendants, so parallel native
runs are included. It reports median wall time, median absolute deviation,
peak RSS, solve/moment/warm-start counts, maximum Sharpe difference, tie-aware
precision@k and Spearman correlation, backend, and fallback reason.

The quick matrix covers every `RiskMeasure`, directly constructible public
optimizers, and WalkForward, purged CPCV, and MultipleRandomizedCV:

```bash
PYTHONPATH=src python benchmarks/benchmark_matrix.py --quick --repeats 3
```

On this cloud VM (Python 3.12, skfolio 1.0.0), the repeated quick results were:

- Structured variance: 19.9–29.6× faster.
- Structured semi-variance/deviation: 5.8–8.9× faster.
- MAD, first lower partial moment, worst realization, and drawdown LPs:
  7.0–12.7× faster.
- CVaR: 8.6–15.2× faster.
- EVaR: 5.5–7.8× when the compact cone solved; one MRC case retried native
  after Clarabel reported insufficient progress.
- EqualWeighted and InverseVolatility: 4.5–14.2× faster.
- Generic fallback: approximately 1×, as expected.

All solvable cases retained tie-aware precision@k of 1.0. Native EDaR failed
on two of the three quick synthetic cases; those are reported as native solver
limitations. Structured peak-RSS ratios were about 0.97–0.98, while fallback
was about 1.0. Absolute RSS is dominated by the Python/skfolio import baseline,
so the ratio should not be interpreted as solver workspace size alone.

For representative 5,040-day workloads, native skfolio used `n_jobs=1`.
WalkForward had 228 solves and MRC had 480 solves:

| Risk | WalkForward speedup | MRC speedup | max path Sharpe difference |
|---|---:|---:|---:|
| Variance | 62.1× | 75.3× | `4.3e-5` |
| Semi-variance | 2.3× | 3.0× | `1.1e-16` |
| MAD | 2.2× | 3.1× | `2.1e-6` |
| CVaR | 3.3× | 4.2× | `7.4e-6` |
| Max drawdown | 2.1× | 2.6× | `1.6e-7` |

Peak-RSS ratios for these runs were 0.96–0.99. Small six-solve CPCV cases gave
only 0.98–1.12× for the scenario engines because fixed setup dominates; they
are not included in a headline average. Semi-variance uses direct Clarabel:
OSQP was mathematically valid but measured at only 0.69× on the 20-asset
WalkForward workload.

Reproduce a focused 20-year run with:

```bash
PYTHONPATH=src python benchmarks/benchmark_matrix.py --repeats 3 \
  --only 'MeanRisk[VARIANCE]' \
  --only 'MeanRisk[SEMI_VARIANCE]' \
  --only 'MeanRisk[MEAN_ABSOLUTE_DEVIATION]' \
  --only 'MeanRisk[CVAR]' \
  --only 'MeanRisk[MAX_DRAWDOWN]'
```

Use `--native-n-jobs=-1` as a separate parallel timing comparison. The
correctness baseline remains native `n_jobs=1`. Structured and fallback rows
are deliberately reported separately.

## Installation and tests

The package targets skfolio 1.x. Its only additional runtime dependency is
OSQP; NumPy, SciPy, Clarabel, and scikit-learn come from skfolio's own runtime
stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Automation and releases

Every pull request and push to `main` runs Ruff, the complete test suite on
Python 3.10 through 3.14, a coverage report, and a wheel/sdist build with an
installation smoke test. Dependabot checks Python and GitHub Actions
dependencies weekly.

Publishing a non-prerelease GitHub Release automatically verifies the source,
checks that a tag such as `v0.1.0` matches the version in `pyproject.toml`,
builds the distributions, and publishes them to PyPI. PyPI Trusted Publishing
must be configured once for this repository, the `release.yml` workflow, and
the `pypi` GitHub environment; no API token is stored in GitHub.

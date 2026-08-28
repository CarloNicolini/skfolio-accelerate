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

Internally a call is compiled once (`cv_plan`) then executed.
`backend="auto"` covers every `ObjectiveFunction` × `RiskMeasure` pair on
WalkForward, MultipleRandomizedCV, and CombinatorialPurgedCV:

- overlapping training moments are updated from sufficient statistics;
- boxed variance uses a compact OSQP QP reused across folds;
- boxed scenario risks use a compact Clarabel cone;
- other MeanRisk configurations reuse skfolio's own CVXPY problem
  (`mu`, scenario returns, and covariance square-root are `cp.Parameter`);
- test portfolios are assembled from `weights_`.

Estimators that cannot reuse a compiled problem still call native `fit`, then
use that same assembly path unless the call needs sequential
`previous_weights`, a pipeline, parallel `n_jobs`, or another option that
changes how `predict` is called.

## What it does

Backtests repeatedly fit nearly identical portfolios. This package recognises
the cases where that repetition can be removed without changing the portfolio
problem:

- It updates empirical moments as a training window moves.
- It assembles CPCV training moments from fold blocks, including purge and
  embargo exclusions.
- It reuses a compact OSQP or Clarabel problem shape for boxed MeanRisk.
- It reuses skfolio's MeanRisk CVXPY graph across folds when extra options
  keep a fixed training length.
- It scores compact hyperparameter candidates from weights before constructing
  the final portfolio objects.
- It compiles the CV plan once and builds test portfolios from `weights_`, so
  serial calls skip joblib, `safe_split` copies, and `predict()` construction.

All reuse is local to one call. The package does not keep global caches of
returns, estimators, fitted priors, or portfolios.

The EqualWeighted speedup is this last step, not a hidden solver trick. Native
`cross_val_predict` still clones the estimator, validates a DataFrame/array
slice, wraps `n_jobs=1` in joblib, copies the test fold, and constructs a
`Portfolio` for every split. EqualWeighted has no optimisation, so removing
that CV tax is the whole gain. It is a roughly constant saving per fold, not a
multiplicative floor: a cheap estimator shows a large ratio; a CVXPY solve
that already dominates the fold only shrinks by that same overhead.

## When it helps

Leave `backend="auto"`. The policy picks the first eligible engine:

1. compact OSQP (boxed variance) / Clarabel (boxed scenario) / closed-form
   EqualWeighted, Random, or InverseVolatility;
2. Parameterized CVXPY reuse (`cvxpy-sequential`) for other MeanRisk
   configurations with a fixed training shape;
3. native `fit` plus assembly from `weights_`;
4. unmodified skfolio.

Sequential reuse covers standard deviation, Ulcer, `MAXIMIZE_RETURN`, risk
limits (`min_return`, …), linear constraints, management fees, and L1. The
formulation stays skfolio's: this package only Parameterizes expected return,
scenario returns, covariance square-root, and the default MAR used by
downside measures.

Not sequential (fit-assemble or native skfolio):

- `MAXIMIZE_RATIO` (no Charnes–Cooper homogenization proxy);
- transaction costs / turnover (`needs_previous_weights`);
- `add_constraints` / `add_objective` / `overwrite_expected_return`;
- uncertainty sets, `max_tracking_error`;
- MeanRisk subclasses;
- pipelines, `n_jobs != 1`, and other options that change `predict`.

Gini mean-difference is sequential-eligible but omitted from the default
benchmark: a year-long training window is a ~20-minute LP per side and stays
~1×.

This boundary is intentional. Reusing mutable estimator or solver state without
proving equivalence could silently solve a different investment problem.

Pass `return_report=True` if you want to see which engine `backend="auto"`
selected and why:

```python
prediction, report = cross_val_predict(
    MeanRisk(),
    X,
    cv=cv,
    return_report=True,
)
print(report.backend, report.reason)
```

You do not pass an engine name in application code. `"osqp"`, `"clarabel"`,
and `"cvxpy-sequential"` are the policy's choices, not a user setting.
`"sklearn"` means native skfolio was used; `report.reason` explains why. If a
compact or sequential numerical solve cannot finish, the package retries with
native `fit` and the assembled path rather than returning an
accelerator-only failure.

## Checking a result

Use native skfolio as the reference when validating a new workload. Numerical
closeness does not automatically mean that a ranking is unchanged:

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

Precision@k checks whether skfolio's best candidates remain in the best set.
The optional tolerance treats near-equal scores as ties. Spearman compares the
full ordering. It is `nan` when every score is the same.

## Parameter search

`grid_search` evaluates a compact MeanRisk parameter grid with one shared CV
plan. Every candidate must be compact-eligible (boxed OSQP / Clarabel). Ratio
objectives, risk limits, linear constraints, and other sequential MeanRisk
options are not searched here — use skfolio or sklearn search for those, and
for non-MeanRisk estimators.

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

## Benchmarks

`backend="auto"` is measured on every non-annualized `ObjectiveFunction` ×
`RiskMeasure` pair, plus a few extra MeanRisk options, across three CV
protocols. The large multiplicative win is still **boxed variance with many
overlapping OSQP folds**. Sequential reuse is about **2×** on those same
WalkForward / MRC windows. A six-solve CPCV on the same 20-year sample is
near **1×**, and sequential Ulcer / exponential-cone graphs can be slower than
native when the training length changes.

![Representative 20-year workload speedups](docs/figures/long-workload-speedups.svg)

The large test is 5,040 × 20 synthetic daily returns, native `n_jobs=1`, one
isolated process (Python 3.12, skfolio 1.0.0, seed 42). Geometric means over
ok cells:

| Engine | WalkForward (228) | MRC (480) | CPCV (6) |
|---|---:|---:|---:|
| OSQP | 50.0× (46.7–53.4, n=2) | 41.5× (35.7–48.2, n=2) | 11.0× (10.8–11.3, n=2) |
| Clarabel | 2.32× (1.74–3.51, n=18) | 3.05× (2.28–4.54, n=18) | 0.95× (0.54–1.12, n=20) |
| Sequential | 2.35× (1.74–2.94, n=23) | 2.19× (1.74–2.59, n=18) | 0.82× (0.09–2.82, n=23) |
| Fit-assemble | 1.04× (0.71–1.20, n=14) | 1.12× (1.08–1.18, n=12) | 1.02× (0.94–1.16, n=13) |
| All ok cells | 2.14× (n=57) | 2.36× (n=50) | 0.99× (n=58) |

Minimize-risk, same 20-year sample:

| Risk | WalkForward (228) | MRC (480) | CPCV (6) | Engine |
|---|---:|---:|---:|---|
| Variance | 46.7× | 48.2× | 10.8× | OSQP |
| CVaR | 3.38× | 4.33× | 1.05× | Clarabel |
| Worst realization | 2.54× | 3.51× | 0.91× | Clarabel |
| MAD | 2.40× | 3.23× | 0.85× | Clarabel |
| First lower partial moment | 2.35× | 3.31× | 0.96× | Clarabel |
| Semi-variance | 2.29× | 3.05× | 0.97× | Clarabel |
| Max drawdown | 2.11× | 2.62× | 1.07× | Clarabel |
| CDaR | 2.07× | 2.46× | 0.94× | Clarabel |
| Semi-deviation | 2.05× | 2.64× | 1.01× | Clarabel |
| Average drawdown | 1.74× | 2.28× | 0.54× | Clarabel |
| Standard deviation | 2.58× | 2.44× | 1.59× | Sequential |
| Ulcer | 1.74× | 1.74× | 0.12× | Sequential |
| EVaR | 0.71× | fail | 1.12× | Compact Clarabel retried native on WalkForward |
| EDaR | fail | fail | fail | Native Clarabel `SolverError` |

Sequential extras (WalkForward / CPCV; MRC skipped because named constraints
fail on asset subsets and `min_return` can be infeasible on random windows):

| Extra | WalkForward (228) | CPCV (6) |
|---|---:|---:|
| Variance + `min_return` | 2.94× | 1.73× |
| Variance + linear constraints | 2.82× | 1.68× |
| Variance + L1 | 2.73× | 1.65× |
| Variance + management fees | 2.68× | 1.60× |
| CVaR + `min_return` | 2.12× | 0.97× |
| Variance `MAXIMIZE_RETURN` | 2.74× | 1.65× |
| Variance `MAXIMIZE_RATIO` | 1.15× | 1.16× |

`MAXIMIZE_RATIO` is fit-assemble on every risk that finished (~1.05–1.20×).
Sequential CPCV rebuilds (5 of 6 folds) of Ulcer and of
`MAXIMIZE_RETURN` EVaR / EDaR are **0.09–0.14×** versus native: the compiled
graph is large, and a changing training length pays the construction cost
again.

On the small 120 × 6 suite every fold still pays CVXPY setup, so compact
scenario risks look closer to variance. That ratio does not survive once the
cone solve dominates. Closed-form EqualWeighted / Random / InverseVolatility
skip native CV machinery (about 5–13× on this tiny problem). Serial estimators
that still call native `fit` (HRP, …) are 1.05–2.1× — the same overhead cut,
not a compact solver. Pipelines stay on native skfolio (~1×). Peak RSS is
typically similar to native because importing Python and skfolio dominates
these processes.

![Quick benchmark speedups by engine](docs/figures/quick-benchmark-speedups.svg)

![EqualWeighted native CV overhead](docs/figures/cv-overhead-breakdown.svg)

The project does not claim one universal speedup number. Measured results
depend on data shape and how many overlapping training windows you actually
run.

Compare native skfolio to `backend="auto"` across every
`ObjectiveFunction` × non-annualized `RiskMeasure` on WalkForward,
MultipleRandomizedCV, and CombinatorialPurgedCV (the library picks OSQP,
Clarabel, sequential CVXPY, or fit-assemble; you do not pass an engine name):

```bash
PYTHONPATH=src python benchmarks/benchmark_sequential_mean_risk.py --repeats 1
PYTHONPATH=src python benchmarks/benchmark_sequential_mean_risk.py --quick
PYTHONPATH=src python benchmarks/render_readme_figures.py
```

The compact-only matrix (five boxed risks, median of three isolated runs) is
still available:

```bash
PYTHONPATH=src python benchmarks/benchmark_matrix.py --quick --repeats 3
PYTHONPATH=src python benchmarks/benchmark_matrix.py --repeats 3 \
  --only 'MeanRisk[VARIANCE]' \
  --only 'MeanRisk[SEMI_VARIANCE]' \
  --only 'MeanRisk[MEAN_ABSOLUTE_DEVIATION]' \
  --only 'MeanRisk[CVAR]' \
  --only 'MeanRisk[MAX_DRAWDOWN]'
```

Use `--native-n-jobs=-1` separately when comparing parallel throughput. Native
`n_jobs=1` is the correctness baseline. The matrix CSV keeps compact and
fallback rows separate and includes timing spread, peak RSS, numerical
difference, rankings, solver counts, and fallback reasons.

## Documentation

API reference, user guide, and gallery examples are built with Sphinx (numpydoc
+ pydata-sphinx-theme, matching skfolio's documentation style):

```bash
pip install -e ".[docs]"
cd docs && make html
```

Open `docs/_build/html/index.html`. Continuous integration builds the docs on
every pull request and publishes them to GitHub Pages from `main`.

## Installation and tests

The package targets skfolio 1.x. Its only additional runtime dependency is
OSQP; NumPy, SciPy, Clarabel, and scikit-learn come from skfolio's own runtime
stack.

```bash
uv sync --extra dev
source .venv/bin/activate
pytest
```

This uses the project's `.python-version` pin (Python 3.12). `uv` installs that
interpreter if needed, creates `.venv`, and installs the package in editable
mode with the `dev` extras.

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

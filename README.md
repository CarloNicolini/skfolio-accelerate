# skfolio-accelerate

A small add-on for [skfolio](https://github.com/skfolio/skfolio). It is a
drop-in for `skfolio.model_selection.cross_val_predict`: same estimators,
parameters, splitters, and return types (`MultiPeriodPortfolio` /
`Population`). Compact solvers speed up overlapping multi-path backtests
when they are equivalent to `MeanRisk`.

```python
from skfolio.optimization import MeanRisk
from skfolio.model_selection import WalkForward

from skfolio_accelerate import cross_val_predict

pred = cross_val_predict(MeanRisk(), X, cv=WalkForward(train_size=252, test_size=21))
```

The return type is the same `MultiPeriodPortfolio` / `Population` as skfolio.

## How it is faster

Thousands of CV splits share almost the same train window. This library does
not rebuild a CVXPY `MeanRisk` problem on every split. It:

1. **Reuses empirical moments.** Sliding Gram updates on WalkForward / MRC, and
   fold-block sufficient stats on CPCV, so `n_prior_fits` ≪ `n_solves`.
2. **Solves a compact QP/LP** with warm starts along time: OSQP for
   `VARIANCE` (dense n-asset QP), Clarabel for `CVAR` (LP in weights, VaR, and
   residuals).
3. **Assembles paths** from NumPy views into skfolio `Portfolio` objects.

This is a drop-in: the call signature matches skfolio (any estimator, any
options, any splitter). Compact OSQP/Clarabel is used only when it is
equivalent to `MeanRisk`; otherwise the original skfolio `fit` / CVXPY
problem runs unchanged.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from skfolio.optimization import MeanRisk
from skfolio_accelerate import cross_val_predict
from skfolio_accelerate.flagship import SMOKE_MRC, make_mrc

X, cv = make_mrc(SMOKE_MRC)
pred, report = cross_val_predict(MeanRisk(), X, cv=cv, return_report=True)
print(report)
print(pred.summary())
```

`return_report=True` adds an `AccelerationReport` with solve counts and phase
times. `n_jobs` and the rest of the skfolio kwargs (`method`, `params`,
`column_indices`, `entry_rebalancing_params`, …) are forwarded whenever the
compact engine does not apply. The compact engine itself is sequential.

## Flagship vs skfolio `cross_val_predict(n_jobs=-1)`

Hardware: cloud VM, Python 3.12, skfolio 1.0.0, OSQP 1.1.3, Clarabel 0.11.1,
`os.cpu_count()=4`, BLAS threads capped at 1. Factor-model synthetic dailies.

```bash
PYTHONPATH=src python benchmarks/benchmark_multipath.py
PYTHONPATH=src python benchmarks/benchmark_multipath.py --quick
```

The frozen MRC workload is `FLAGSHIP_MRC` in
[`src/skfolio_accelerate/flagship.py`](src/skfolio_accelerate/flagship.py):
10y-scale daily frame (2520×80), Palomar-style **500 subsamples** of 25 assets
on 3-year windows, monthly `WalkForward(train=252, test=21)` → **12 000**
MeanRisk fits.

| Workload | skfolio `cross_val_predict` | Compact engine | Speedup |
|---|---|---|---|
| **FLAGSHIP MRC VARIANCE** (12 000 solves) | **29.25 s** (`n_jobs=-1`) | **2.55 s** (sequential) | **11.5×** |
| CPCV smoke (15 combinations) | 0.093 s (`n_jobs=-1`) | 0.004 s | **21×** |
| WalkForward CVaR (24 steps, sequential) | 0.360 s (`n_jobs=1`) | 0.127 s | **2.8×** |

FLAGSHIP MRC VARIANCE phase report:

- Baseline sample `MeanRisk.fit` on a 252×25 window: **8.1 ms** (empirical prior **1.4 ms**, QP **6.7 ms**).
- Compact: moments **0.31 s**, OSQP (warm-started) **1.29 s**, path assembly **0.45 s**, wall **2.55 s**.
- **500** prior cold starts vs **12 000** solves; **11 500** sliding Gram updates and solver warm starts.
- Path Sharpe vs skfolio: max \|Δ\| **2.1×10⁻⁴**.

CVaR uses a compact Clarabel LP (correct to ~1e-7 in weights). It is the right
kernel when `T` is large; on a short WalkForward it is only a modest win
because each LP is already cheap.

## Tests

```bash
source .venv/bin/activate
pytest
```

## Coverage

`cross_val_predict` accepts the same estimators, parameters, and splitters as
skfolio: `MeanRisk` (all risk measures and objective functions), naive and
hierarchical estimators, pipelines, `KFold`, `TimeSeriesSplit`, `WalkForward`,
`CombinatorialPurgedCV`, `MultipleRandomizedCV`, integer `cv`, and skfolio
kwargs such as `entry_rebalancing_params`.

**Compact (faster, equivalent subset):** `MeanRisk` with
`MINIMIZE_RISK` / `MAXIMIZE_UTILITY`, `VARIANCE` (OSQP) or `CVAR` (Clarabel LP),
default `EmpiricalPrior`, and simple box/budget constraints.

**Skfolio path (full CVXPY / estimator `fit`):** every other risk measure
(semi-variance, CDaR, EVaR, Gini, …), `MAXIMIZE_RATIO`, MIP, costs, turnover,
custom priors, uncertainty sets, extra constraints, factor `y`, pipelines, and
non-`MeanRisk` estimators. Those keep skfolio’s own problem, solver, and
parameters.

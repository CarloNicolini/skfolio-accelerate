# skfolio-accelerate

Amortized multi-path backtests for [skfolio](https://github.com/skfolio/skfolio), plus
an optional compiled grid-search engine.

The workload that matters is **evaluation**, not a 15-second `l2` GridSearch.
`WalkForward`, `CombinatorialPurgedCV`, and `MultipleRandomizedCV` produce
thousands of overlapping train windows. This library:

1. Reuses empirical moments on sliding windows and CPCV fold blocks
   (`n_prior_fits` ≪ `n_solves`).
2. Solves a **compact** MeanRisk QP/LP (OSQP for VARIANCE, Clarabel for CVaR)
   with **warm starts along time**, not a CVXPY rebuild per split.
3. Returns the same `MultiPeriodPortfolio` / `Population` types as
   `skfolio.model_selection.cross_val_predict`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The optional Rust Clarabel engine (used only by `MassiveGridSearchCV`) is built
by maturin when a Rust toolchain is available:

```bash
pip install -e ".[dev,native]"
```

## Usage: multi-path predict

```python
from skfolio.optimization import MeanRisk
from skfolio.model_selection import WalkForward, MultipleRandomizedCV

from skfolio_accelerate import massive_cross_val_predict
from skfolio_accelerate.flagship import SMOKE_MRC, make_mrc

X, cv = make_mrc(SMOKE_MRC)
pred, report = massive_cross_val_predict(
    MeanRisk(), X, cv=cv, n_jobs=-1, return_report=True
)
print(report)
print(pred.summary())
```

`backend="auto"` uses the compact engine for `MeanRisk` with
`MINIMIZE_RISK` / `MAXIMIZE_UTILITY` and `VARIANCE` / `CVAR`. Anything else
falls back to skfolio `cross_val_predict`.

## Flagship vs skfolio `cross_val_predict(n_jobs=-1)`

Hardware: cloud VM, Python 3.12, skfolio 1.0.0, `os.cpu_count()=4`, BLAS
threads capped at 1. Returns are factor-model synthetic dailies.

Reproducible run:

```bash
PYTHONPATH=src python benchmarks/benchmark_multipath.py
PYTHONPATH=src python benchmarks/benchmark_multipath.py --quick
```

The frozen MRC workload is `FLAGSHIP_MRC` in
[`src/skfolio_accelerate/flagship.py`](src/skfolio_accelerate/flagship.py):
10y-scale daily frame (2520×80), Palomar-style **500 subsamples** of 25 assets
on 3-year windows, monthly `WalkForward(train=252, test=21)` → **12 000**
MeanRisk fits.

Hardware: cloud VM, Python 3.12, skfolio 1.0.0, OSQP 1.1.3, Clarabel 0.11.1,
`os.cpu_count()=4`, BLAS threads capped at 1. Factor-model synthetic dailies.

| Workload | skfolio `cross_val_predict` | Compact engine | Speedup |
|---|---|---|---|
| **FLAGSHIP MRC VARIANCE** (12 000 solves) | **28.99 s** (`n_jobs=-1`) | **2.59 s** (`n_jobs=1`) | **11.2×** |
| CPCV smoke (15 combinations) | 0.087 s (`n_jobs=-1`) | 0.004 s | **20×** |
| WalkForward CVaR (24 steps, sequential) | 0.360 s (`n_jobs=1`) | 0.127 s | **2.8×** |

FLAGSHIP MRC VARIANCE phase report:

- Baseline sample `MeanRisk.fit` on a 252×25 window: **8.2 ms** (empirical prior **1.4 ms**, QP **6.8 ms**). That kernel is essentially **all** of the 29 s baseline (joblib-parallelized).
- Compact: moments **0.40 s**, OSQP (warm-started) **1.29 s**, path assembly **0.41 s**, wall **2.59 s** (81% in those three phases).
- **500** prior cold starts vs **12 000** solves; **11 500** sliding Gram updates and solver warm starts.
- Path Sharpe vs skfolio: max \|Δ\| **2.1×10⁻⁴**.

The pass bar was ≥10× wall-clock with the accelerated kernels ≥70% of the baseline job. Both hold for the VARIANCE MRC flagship.

CVaR uses a compact Clarabel LP (correct to ~1e-7 in weights). It is the right kernel when `T` is large; on a 24-step WalkForward it is only a modest win against sequential skfolio because each LP is already cheap.

## Usage: compiled grid search (secondary)

```python
from skfolio_accelerate import MassiveGridSearchCV
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk
from skfolio import RiskMeasure

search = MassiveGridSearchCV(
    estimator=MeanRisk(),
    param_grid={
        "risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR],
        "l2_coef": [1e-5, 1e-4, 1e-3, 1e-2],
    },
    cv=CombinatorialPurgedCV(n_folds=10, n_test_folds=2),
    backend="auto",  # "rust" | "python" | "sklearn" | "auto"
)
search.fit(X)
print(search.acceleration_report_)
```

Grid search still amortizes priors across an `l2` grid. That loop is **not**
the product: on the 2520×200 KFold squeeze, instantiate is ~1% of wall time
and Clarabel is ~93%. See the notes at the end of this file.

## Tests

```bash
source .venv/bin/activate
pytest
```

## What the compact path engine covers

- Estimator: `MeanRisk` with `MINIMIZE_RISK` / `MAXIMIZE_UTILITY`
- Risk: `VARIANCE` (OSQP), `CVAR` (Clarabel LP)
- CV: `KFold`, `WalkForward`, `CombinatorialPurgedCV`, `MultipleRandomizedCV`
- Default `EmpiricalPrior` (sample mean, sample covariance `ddof=1`)

Out of scope for the compact engine: MIP, transaction costs, `MAXIMIZE_RATIO`,
`min_return`, `l1_coef`, dict weight bounds, custom priors (those still use
the compact **solver** but refit the prior each window).

## Grid-search engine (v0.1) — measured vs original skfolio

These numbers are kept for regression of `MassiveGridSearchCV`. They are **not**
the success criterion for this repository.

Hardware for these runs: cloud VM, Python 3.12, skfolio 1.0.0, CVXPY 1.9.2,
Clarabel 0.11.1, `os.cpu_count()=4`, BLAS threads capped at 1.

| Workload | Original | Accelerate | Speedup |
|---|---|---|---|
| KFold **2520×200**, 10 folds × 24 `l2` (240 evals) | sklearn **15.26 s** | **Rust n_jobs=4: 5.23 s** | **2.9×** |
| CPCV **1008×120**, 10c2 × 16 `l2` (720 solves) | naive ~14.1 s | **Rust 4.06 s** | **~3.5×** |

Phase split on the 2520×200 KFold: instantiate **0.05 s (~1%)**, solve **4.87 s (~93%)**.
Further DPP / instantiate work cannot move this job. The multi-path predictor
exists because that 5% was the wrong target.

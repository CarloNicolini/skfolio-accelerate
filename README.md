# skfolio-accelerate

Compiled massive cross-validation and hyperparameter search for
[skfolio](https://github.com/skfolio/skfolio).

CVXPY is used as a **compiler**, not a runtime. A DPP-parameterized MeanRisk twin is
canonicalized once per problem structure. Clarabel then `update`s numeric `P, q, A, b`
across folds × grid points. Priors are fit **once per fold**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The optional Rust engine (Clarabel 0.11.1 + Rayon) is built by maturin when a Rust
toolchain is available:

```bash
pip install -e ".[dev,native]"
```

## Usage

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
search.best_params_
search.best_score_
search.cv_results_
search.best_estimator_
print(search.acceleration_report_)
```

`backend="auto"` uses the Rust engine when the extension imported, otherwise the
Python Clarabel `update` loop. If the estimator/CV/grid is not accelerable, it falls
back to sklearn `GridSearchCV`. CombinatorialPurgedCV is never passed to sklearn.

## What is accelerated (v0.1)

- Estimator: `MeanRisk` with `MINIMIZE_RISK` / `MAXIMIZE_UTILITY`
- Risk: `VARIANCE`, `CVAR`
- Numerical params: `l1_coef`, `l2_coef`, `risk_aversion`, `min_return`, `cvar_beta`
- CV: `KFold`, `WalkForward`, `CombinatorialPurgedCV`
- Solver: Clarabel, updates with presolve / chordal decomposition / dropped zeros off

Out of scope: MIP (`cardinality`, thresholds), transaction costs, `MAXIMIZE_RATIO`,
arbitrary CVXPY in Rust, Polars.

## Combinatorial CV scoring

For `CombinatorialPurgedCV`, test segments are reassembled into paths with
`get_path_ids()` (same as `cross_val_predict`). `mean_test_score` is the mean of
path Sharpes, not a concatenation of overlapping test indices.

## Tests

```bash
source .venv/bin/activate
pytest
```

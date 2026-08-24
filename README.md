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

For the fastest path on a numerical grid (`l2_coef`, …): `backend="rust"`,
`n_jobs` = process CPU count, `solver_threads=1`.

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

## Measured results vs original skfolio

Hardware for these runs: cloud VM, Python 3.12, skfolio 1.0.0, CVXPY 1.9.2,
Clarabel 0.11.1, `os.cpu_count()=4`, BLAS threads capped at 1. Returns are
factor-model synthetic dailies in a pandas DataFrame.

Reproducible squeeze: `PYTHONPATH=src python benchmarks/benchmark_squeeze.py`.

### Correctness (unit suite + large frames)

All **14** tests passed, including VARIANCE/CVaR twins vs `MeanRisk.fit`, KFold
parity vs sklearn `GridSearchCV`, CPCV path Sharpes vs `cross_val_predict`,
Rust vs Python Clarabel, and numerical instantiate reuse (shared `A`, matching `P`).

| Check | Frame / grid | vs skfolio |
|---|---|---|
| VARIANCE twin weights | 60×6, several `l2` | max \|Δw\| **4.6×10⁻¹⁴** |
| CVaR twin weights | 40×5 | max \|Δw\| **4.4×10⁻¹⁴** |
| KFold `mean_test_score` | 90×8, 3 folds × 2 `l2` | max \|Δ\| **1.3×10⁻⁶**, same `best_params_` |
| KFold (wide) | **2520×200**, 10 folds × 24 `l2` | same `best_params_`, \|Δ best_score\| **4.7×10⁻¹⁰** |
| Mixed VARIANCE+CVaR | 504×50, 5 folds × 16 combos | same `best_params_`, \|Δ score\| **4.1×10⁻⁹** |
| CPCV path Sharpe | 48×6, 4c2 | \|Δ mean path Sharpe\| **7.6×10⁻⁶** |
| Prior reuse | KFold 2520×200 | **10** prior fits vs **240** sklearn clone+fits |

### Timing (wall-clock)

Small isolated pieces:

| Workload | skfolio / naive | Accelerate | Speedup |
|---|---|---|---|
| DPP QP vs rebuild, n=40 × 40 solves | (see `benchmarks/benchmark_dpp.py`) | DPP `problem.solve` | >1× |
| 20 × VARIANCE `l2` on 120×15 | 0.139 s `MeanRisk.fit` | 0.021 s Python update | **6.6×** |
| KFold 180×25, 4 `l2`, 5 folds | sklearn 0.178–0.208 s | Python **0.045–0.047 s** | **3.9–4.5×** |

Longer and wider frames (the squeeze runs):

| Workload | Original | Accelerate | Speedup |
|---|---|---|---|
| 80 × `MeanRisk.fit` on 2520×200 | ~4.92 s (20-fit ×4) | instantiate 0.007 s + Rust `solve_many` **1.60 s** | **~3.1×** |
| KFold 2016×120, 8 folds × 20 `l2` (160 evals) | sklearn 3.18 s | Python 1.78 s / **Rust n_jobs=4: 0.80 s** | 1.8× / **4.0×** |
| KFold **2520×200**, 10 folds × 24 `l2` (240 evals) | sklearn **15.26 s** | Python 7.16 s / **Rust n_jobs=4: 5.23 s** | 2.1× / **2.9×** |
| Mixed VARIANCE+CVaR 504×50, 5×16 | sklearn 1.73 s | Python **0.83 s** (3 templates, 5 priors) | **2.1×** |
| CPCV **1008×120**, 10c2 × 16 `l2` (720 solves) | naive ~14.1 s (8-fit extrapolate) | **Rust 4.06 s** (45 priors vs 720) | **~3.5×** |

On these sizes sequential Rust (`n_jobs=1`) is **not** faster than Python Clarabel
(16.2 s vs 7.2 s on the 2520×200 KFold): the FFI path plus one-solver-at-a-time
loses to CPython `clarabel.DefaultSolver.update`. The Rust win is **batched
`solve_many` + one persistent solver per Rayon worker**.

Phase split on the 2520×200 KFold:

| Phase | Previous (full `apply_parameters` × 240) | Now |
|---|---|---|
| compile | 0.09 s | 0.07 s |
| instantiate | **0.87 s** | **0.05 s** |
| solve (Rust n_jobs=4) | 4.20 s | 4.87 s |
| eval | 0.18 s | 0.09 s |
| moment fits | 10 | 10 |

Instantiate dropped by about **17×** because `l2` no longer rebuilds CSC `A`.

## Where time and memory go (hot path)

Each accelerated `fit` does, per fold, then per `(risk_measure, train length)`
template:

1. **Slice train rows.** Contiguous KFold test blocks (and the first/last train
   block) use `iloc[start:stop]` / ndarray views instead of fancy-index copies.
2. **Fit prior once** → `mu` (`n`), Cholesky `L` (`n × n`). Scenario returns `R`
   (`T × n`) are stored only if some grid point is CVaR.
3. **Bind Parameters.** Fold data (`L`, `R`, `mu`) is written **once per fold**
   into the existing CVXPY buffers (`np.copyto`). Numerical scalars (`l2`, …)
   are set every grid point.
4. **One `apply_parameters` per fold** to materialize CSC `A, b` (these depend
   on `L` / `R`, not on `l2`). Further `l2` values are a saxpy on `P.data`
   from cached DPP tensor columns: `P = P_const + l2 * P_l2` (201 nnz at
   n=200). `A` is reused by object identity.
5. **Stack `P` only** into a C-contiguous `(n_grid, nnz_P)` batch. Shared `A`,
   `q`, `b` are passed as a **single row**.
6. **Clarabel `update_P` + `solve`.** After the first point of a fold, `A` is
   not rewritten into the KKT. Rust keeps one `WorkerSolver` per Rayon worker
   for the whole `fit`.
7. **Copy the weight slice** out of `x` (full primal is not retained).
8. **Score** the test window (`X @ w` via `to_numpy(copy=False)`, or
   `estimator.score`).

KFold on 2520×200 with 24 `l2` values is **240 Clarabel solves** but only
**10 prior fits**, **10 `apply_parameters`**, and **1 template**. sklearn
`GridSearchCV` rebuilds a CVXPY graph 240 times and refits the prior 240 times.

Empirically, at n=200 VARIANCE:

- `reduced_A` is ~40 803×40 002 with 40 803 nnz, **0 nnz in the `l2` column**
- `reduced_P` is 201×40 002 with 201 nnz, **all in `l2` + the constant column**
- `q` is identically 0
- packing `param_vec` + matvec is ~0.003 s / 24 `l2`; full
  `apply_parameters` was ~0.078 s / 24 because it reconstructed `A` every time

### Layout / copies removed

- CSC **index arrays live on the template**, not on every `NumericInstance`.
- Fold cache **drops the dense covariance** after Cholesky; VARIANCE grids
  **drop `R`** (`T × n` × n_folds).
- Data Parameters (`L`, `R`) are rebound **once per fold**, in-place.
- Numerical `l2` updates **do not call `apply_parameters`**.
- Python Clarabel **reuses one `P, q, A, b` buffer**; after the first point of
  a fold it calls `solver.update(P=P)` only.
- Rust `update_P` on a reused `Vec` buffer; `A`/`b` only on the first point
  of each worker chunk (new fold).
- Rust FFI reads NumPy as one C-contiguous slice; shared `A` is one row.
- Persistent Rayon **worker solvers** across folds (no `CscMatrix::new` per fold).
- Native return is **weights only**.
- KFold **does not keep per-eval weight vectors** after scoring; contiguous
  test slices; one scratch estimator.

### Further timing / footprint work (not done)

Ranked by likely impact now that instantiate is ~1% of the 2520×200 job
(solve is ~93%):

1. **Clarabel itself.** Interior-point on a 200-asset SOC is the wall. Warm-start
   `x` along the sorted `l2` grid, or a first-order / OSQP path for VARIANCE-only
   QP, would move this more than any more host-side copies. Measure iterations
   before relying on warm-start.
2. **Fold-constant `A` into the DPP map without `apply_parameters`.** The
   remaining 0.05 s is 10 full applies (one per fold) to scatter `L` into `A`.
   `reduced_A @ param_vec` is cheap; CSC reconstruction + `keep_zeros` is not.
   Serializing the reduced `A` tensor and writing `.data` in place would drop
   that too.
3. **Score in the native engine.** `X_test @ w` is already a view-matmul.
   CPCV still builds many `Portfolio` objects and copies each test segment.
   Native Sharpe on views of `X` would drop the second pass in `_path_means`.
4. **Column-major `X`.** Scoring and CVaR `R @ w` are `T × n` @ `n`. Fortran
   layout matches that access; pandas frames are C-contiguous by row.
5. **Range folds for middle KFold trains.** First/last folds are one slice;
   middle trains are two blocks, so fancy-index still copies into the prior.
   Fitting with 0/1 sample weights on full `X` would avoid that copy if it
   matches EmpiricalPrior numerically.
6. **Do not keep `search_plan_.templates` CVXPY graphs** after `fit` unless
   requested — each template holds the DPP tape.
7. **f32 moments, f64 KKT.** Priors and test returns do not need 64-bit for
   scoring; Clarabel 0.11.1 is f64.
8. **Joblib the bind+apply stage** across VARIANCE vs CVaR templates. GIL still
   serializes CVXPY unless (2) moves the map out of Python.

Peak RAM is dominated by (a) pandas `X`, (b) cached `L` per fold (`n × n`; `R`
only for CVaR), (c) CVXPY twin + DPP tape, (d) one Clarabel workspace per
worker. The old `(n_grid, nnz_A)` batch is gone when `A` is fold-constant.

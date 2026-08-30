# Historical architecture notes (current at benchmark time)

These notes describe the accelerated `cross_val_predict` stack **as of the
git SHA recorded with each run**. They are not a permanent architecture
document. Re-read `src/skfolio_accelerate/predict.py` if the SHA is old.

## What is measured

Native reference: `skfolio.model_selection.cross_val_predict`.

Accelerated method: `skfolio_accelerate.cross_val_predict(..., backend="auto")`.
The library chooses an engine; the benchmark does not pass engine names.

## Engines (policy order)

1. Compact **OSQP** — boxed MeanRisk variance (`l2_coef` allowed).
2. Compact **HiGHS** — boxed scenario LPs with `l2_coef=0` (MAD, FLPM, CVaR,
   worst realization). Persistent simplex basis across rolling folds.
   CombinatorialPurgedCV + MAD/FLPM is **not** HiGHS: `backend="auto"` emits
   `AccelerationWarning` and uses unmodified skfolio (measured ~0.5× on
   20-year windows).
3. Compact **Clarabel** — other boxed scenario cones (semi-variance, CDaR, …).
4. **cvxpy-sequential** — Parameterized reuse of skfolio's MeanRisk CVXPY
   graph (`mu`, scenario returns, covariance square-root as `cp.Parameter`)
   for configurations outside the compact subset (risk limits, linear
   constraints, fees, L1, standard deviation, Ulcer, `MAXIMIZE_RETURN`, …).
5. **fit-assemble** — native `fit` then the shared serial assembly from
   `weights_` (compiled plan, views, portfolios). Cheap closed-form
   estimators are not in this MeanRisk suite.
6. **sklearn** — unmodified skfolio (`n_jobs != 1`, pipelines, sequential
   `previous_weights`, custom hooks, …).

## Solver and tolerances

Native MeanRisk default solver is **CLARABEL** with skfolio's default
tolerances. The suite does **not** pass `solver_params`: that option makes
the estimator compact-ineligible. Compact OSQP / HiGHS / Clarabel use the
package's own compiled problems and tolerances.

## Workers, processes, threads

Canonical comparison is **serial**: `n_jobs=1` on both sides.
`skfolio_accelerate.cross_val_predict` with `n_jobs != 1` selects unmodified
skfolio, so joblib is not an accelerated engine.

`--workers` / `--thread-limit` set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` in the
benchmark process to avoid BLAS/OpenMP oversubscription. Compact Clarabel
also uses `max_threads=1` inside the library.

There is no cross-call global cache of returns, estimators, or portfolios.
Reuse (warm starts, matrix / moment updates, persistent HiGHS basis,
Parameterized CVXPY) is **local to one call**. Each timed repetition clones
the estimator and invokes a full `cross_val_predict`. Warm-up calls are
untimed. Validation runs before timed repetitions.

## Matrix reuse and warm starts

Overlapping WalkForward / MRC training windows update empirical moments from
sufficient statistics (`n_prior_updates` in the acceleration report). OSQP,
HiGHS, and Clarabel warm-start fold solutions (`n_warm_starts`). Sequential
CVXPY rebuilds when the training length changes (`n_rebuilds`).

## Native / Rust / Cython / NumPy components

* NumPy (OpenBLAS in typical wheels) for moments and array views.
* SciPy where skfolio uses it.
* CVXPY + Clarabel for native MeanRisk and sequential reuse.
* OSQP (Python `osqp` package) for compact variance QPs.
* HiGHS via `highspy` for compact LPs.
* No first-party Rust or Cython extension in skfolio-accelerate at the time
  of this note; the speedup is problem reuse, not a compiled kernel rewrite.

## Gini and EVaR / EDaR

Gini mean-difference is sequential-eligible but omitted from the default
estimator grid (year-long windows are ~20-minute LPs). EVaR / EDaR cells may
fail or fall back to native on some windows; failures are recorded, not
hidden.

# Canonical `cross_val_predict` benchmark

This suite is the **canonical performance and numerical-regression benchmark**
for accelerated MeanRisk cross-validation in skfolio-accelerate. It compares
`skfolio.model_selection.cross_val_predict` (native) with
`skfolio_accelerate.cross_val_predict` (`backend="auto"`).

Exploratory scripts under `benchmarks/` remain available; do not treat them as
the source of truth for regressions.

## Reference implementation

* Native: `skfolio.model_selection.cross_val_predict`
* Accelerated: `skfolio_accelerate.cross_val_predict` with `backend="auto"`
  (OSQP / HiGHS / Clarabel / sequential CVXPY / fit-assemble / sklearn as
  selected by the library)

MeanRisk configurations are the full grid used by
`benchmarks/benchmark_sequential_mean_risk.py`: every `ObjectiveFunction` ×
every non-annualized `RiskMeasure` (Gini omitted by default), plus the
sequential extras, plus boxed LP rows with `l2_coef=0` from
`benchmarks/benchmark_lp_cv.py`. Annualized measures and Gini are opt-in.

## Datasets

1. **synthetic** — deterministic factor-model panel from
   `skfolio_accelerate.flagship.factor_returns`:
   `factors ~ N(0, 0.01)`, `loadings ~ N(0, 1)`, `idio ~ N(0, 0.005)`,
   `X = factors @ loadings + idio`. Defaults: 504 × 12, 8 factors, seed 42.
   Columns are named `A0`… so linear-constraint extras match the sequential
   script.
2. **sp500** — `skfolio.datasets.load_sp500_dataset()` prices converted with
   `skfolio.preprocessing.prices_to_returns` (20 assets). Record the skfolio
   version in `environment.json`; dataset contents can change across releases.

## Speed-up and correctness

* Reported time is the **median** of timed repetitions (warm-ups excluded).
  Mean, sample standard deviation, min, max, and raw repetitions are stored.
* **Speed-up** = `native_time / accelerated_time`. Values greater than 1 mean
  the accelerator is faster. Relative time = `accelerated_time / native_time`.
  Δ time = `accelerated_time - native_time`.
* Mean Sharpe is the mean of path Sharpes from `skfolio_accelerate.scoring.path_sharpes`
  (skfolio’s default Sharpe on each path).
* Δ Sharpe = `accelerated - native`. Relative Sharpe error =
  `(accelerated - native) / |native|`.
* Validation runs **before** timed repetitions. Failed cells, NaNs/infs,
  empty outputs, solver/backend fallbacks, optional weight disagreement, and
  near-zero timings (possible accidental cache hits) are recorded. Both
  methods use the same dataset, CV factory, estimator kwargs, `n_jobs=1`,
  seed, and native Clarabel defaults (no custom `solver_params`, which would
  block compact engines).

Thread caps (`OMP_NUM_THREADS` and related) default to 1. `--workers N` sets
that cap; it does **not** enable joblib on the accelerator (`n_jobs != 1`
selects unmodified skfolio).

## Reproduction

Use the existing environment tooling. Do not upgrade dependencies for a
comparison run.

```bash
uv sync --extra dev --extra docs
source .venv/bin/activate

python benchmark/run_relative.py --base origin/main --workers 1
python benchmark/run_relative.py --base origin/main --quick --workers 1 --no-figures
python benchmark/run_benchmark.py
python benchmark/run_benchmark.py --dataset synthetic
python benchmark/run_benchmark.py --dataset sp500
python benchmark/run_benchmark.py --method native
python benchmark/run_benchmark.py --method accelerated
python benchmark/run_benchmark.py --repetitions 5
python benchmark/run_benchmark.py --workers 1
```

Useful flags: `--quick` (smoke sizes), `--full` (20-year sequential panel),
`--cv walk-forward` (repeatable), `--include-gini`, `--include-annualized`,
`--timeout SECONDS`, `--no-figures`, `--output-dir DIR`.

PR vs `main` timing uses **in-run relative** benchmarking on one host (see
`AGENTS.md` and `benchmark/run_relative.py`). Do not compare a PR's seconds to
a CSV saved from an earlier job or laptop.

Plotly is required to write figures (`docs` extra, or `pip install plotly kaleido`).
Kaleido/Chrome is optional for SVG/PNG; HTML and Plotly JSON are always written.

Configuration defaults live in one place: `benchmark/config.py` (`CONFIG`).

## Result locations

Each single-commit run writes a new directory (never overwrites):

```
benchmark/results/YYYY-MM-DD_<git-short-sha>/
    results.csv
    results.json
    environment.json
    summary.md
    ARCHITECTURE.md
    figures/
```

In-run PR vs main pairs land under:

```
benchmark/results/relative/YYYY-MM-DD_<head-sha>/
    base/
    head/
    delta.csv
    delta.json
    summary.md
```

Δ% = `100 * (head_time - base_time) / base_time` (positive = head slower).

Dated folders under `benchmark/results/` are archives of one commit, not the
PR timing baseline. Coding agents must use `run_relative.py`; see `AGENTS.md`.

Latest Plotly HTML/JSON copies are also placed in `benchmark/figures/`.

The summary table columns are:

Dataset | Estimator | Method | Time (s) | Δ Time (s) | Relative Time | Speed-up | Mean Sharpe | Δ Sharpe | Relative Sharpe Error

## Generated figures

Produced programmatically with Plotly (fixed width, fonts, margins, category
order, labels):

* `execution_time` — median seconds by dataset, estimator, and method
* `speedup` — native / accelerated
* `mean_sharpe` — mean path Sharpe
* `sharpe_difference` — accelerated − native
* `historical_speedup` — synthetic speed-ups vs prior `results.csv` runs

Engine, solver, caching, and warm-start details for the SHA of a run are in
that run’s `ARCHITECTURE.md` (historical / current-at-benchmark-time).

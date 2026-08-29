# Agent instructions

This repository has a **canonical** performance and numerical-regression
benchmark for accelerated MeanRisk `cross_val_predict`. Coding agents must
use it instead of ad-hoc timers, notebooks, or the exploratory scripts under
`benchmarks/`.

## Always use this tool

For any change that could affect runtime, numerical results, CV backends,
MeanRisk engines, moments, assembly, or claimed speed-ups / Sharpe errors:

1. Run `python benchmark/run_benchmark.py` (see commands below).
2. Read the new `benchmark/results/YYYY-MM-DD_<git-short-sha>/` directory.
3. Compare against the official baseline pointer
   `benchmark/results/baseline.json`.
4. Report numbers **from those files**, not from memory or hand-timed snippets.

Do **not**:

- Invent or estimate timings.
- Time `cross_val_predict` in a one-off Python snippet and treat that as the
  result.
- Use `benchmarks/benchmark_*.py` as the source of truth (those scripts are
  exploratory / README-figure helpers).
- Upgrade or pin new dependency versions just to make a run look faster.
- Commit over an existing results directory; the runner always creates a new
  dated folder.

Ordinary `pytest` must **not** execute the full sweep. Lightweight checks live
in `tests/test_benchmark_suite.py`.

## Commands

Official baseline (default `CONFIG` in `benchmark/config.py`: synthetic 504×12,
full S&P 500 returns, WalkForward 126/21, 3 timed repetitions, 1 warm-up,
full MeanRisk grid):

```bash
python benchmark/run_benchmark.py --baseline --workers 1
```

Smoke check only (must be labeled `--quick`, never called a baseline):

```bash
python benchmark/run_benchmark.py --quick --workers 1
```

Filtered debugging (not a baseline; do not overwrite `baseline.json`):

```bash
python benchmark/run_benchmark.py --dataset synthetic --method accelerated
python benchmark/run_benchmark.py --repetitions 3 --workers 1
```

Setup: `uv sync --extra dev --extra docs` (or `pip install -e ".[dev,benchmark]"`).
Do not silently upgrade packages. Plotly is required for figures.

## How to report results

Quote the speed-up definition exactly: **speed-up = native_time / accelerated_time**
(median wall time; warm-ups excluded).

From the run `summary.md` / `results.csv`, include at least:

| Dataset | Estimator | Method | Time (s) | Δ Time (s) | Relative Time | Speed-up | Mean Sharpe | Δ Sharpe | Relative Sharpe Error |

Also report:

- path to the new run directory and to `benchmark/results/baseline.json`;
- `environment.json` fields: git SHA, skfolio version, Python, CPU, workers,
  thread caps, BLAS backend;
- failed / invalid cells (`status` not `ok`, NaNs, solver fallbacks);
- whether the comparison used `--quick`, `--baseline`, or `--full`.

When claiming a regression or improvement versus the committed baseline, diff
the same estimator × dataset × method rows. If the protocol differs (window
size, repetitions, threads), say so and do not treat the numbers as a
like-for-like baseline comparison.

## Pointers

- Protocol and defaults: `benchmark/README.md`, `benchmark/config.py`
- Historical engine notes (SHA-specific): `benchmark/ARCHITECTURE.md`
- Current baseline: `benchmark/results/baseline.json` → that run’s
  `summary.md`, `results.csv`, `environment.json`

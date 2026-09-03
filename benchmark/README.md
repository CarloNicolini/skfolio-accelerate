# How to run the `cross_val_predict` benchmark

This harness times **native** `skfolio.model_selection.cross_val_predict`
against **`skfolio_accelerate.cross_val_predict(..., backend="auto")`**.
Use it for every speed claim. Do not time `cross_val_predict` in a notebook
or a one-off snippet and treat that as the result.

There are two commands:

| Command | What it answers |
| --- | --- |
| `run` | On **this commit**, is accelerated faster than native? |
| `relative` | On **this machine**, is HEAD slower or faster than `origin/main`? |

`run` is not a PR baseline. `relative` is.

## Setup

From the repo root, in Python 3.12:

```bash
uv sync --extra dev --extra benchmark
source .venv/bin/activate
```

Typer is required (`dev` / `benchmark` extras). Do not upgrade other
packages to make a run look faster.

Equivalent entry points:

```bash
python -m benchmark run …
python benchmark/run_benchmark.py …

python -m benchmark relative …
python benchmark/run_relative.py …
```

`python -m benchmark` with no arguments prints command help.

## 1. Native vs accelerated on this commit

```bash
python benchmark/run_benchmark.py --workers 1
```

That runs the full default sweep: both datasets, both methods, all three
CV splitters, one warm-up, three timed repetitions, `n_jobs=1`.

Smoke run (4 folds, 1 repetition, shorter panels):

```bash
python benchmark/run_benchmark.py --quick --workers 1
```

Longer synthetic panel (20 years × 20 assets), still 15 folds:

```bash
python benchmark/run_benchmark.py --full --workers 1
```

Restrict the cartesian product with repeatable flags:

```bash
python benchmark/run_benchmark.py --dataset sp500 --cv walk-forward
python benchmark/run_benchmark.py --dataset synthetic --method accelerated
python benchmark/run_benchmark.py --cv walk-forward --cv purged-cpcv
```

Write into a chosen directory (otherwise a dated folder is created):

```bash
python benchmark/run_benchmark.py --quick --output-dir /tmp/bench-out
```

`--quick` and `--full` cannot be combined.

## 2. PR vs `main` (same host)

On **one** machine, time `origin/main` first, then HEAD, with the **same**
flags. The driver installs base from a git worktree, times it, reinstalls
this checkout, times it again, and writes Δ%.

```bash
python benchmark/run_relative.py --base origin/main --workers 1
python benchmark/run_relative.py --base origin/main --quick --workers 1
```

Flags after `--base` / `--head` are forwarded to `run`:

```bash
python benchmark/run_relative.py --base origin/main --head HEAD \
    --quick --workers 1 --dataset synthetic
```

Optional gate (exit 2 if any ok cell is slower than the threshold):

```bash
python benchmark/run_relative.py --base origin/main --quick \
    --fail-on-slow-pct 10
```

`--output-dir` is reserved for the two legs (`base/` and `head/`). Do not
pass it yourself.

Do **not** compare a PR's seconds to a `results.csv` from another day,
CI job, or laptop.

## What is timed

Each cell is one MeanRisk configuration × one dataset × one CV splitter ×
one method (`native` or `accelerated`).

**Estimators.** Every `ObjectiveFunction` × every non-annualized
`RiskMeasure` except Gini, plus extras (`min_return`, linear constraints,
management fees, `l1_coef`) and boxed LP rows with `l2_coef=0`. Gini and
annualized aliases are off unless you pass `--include-gini` /
`--include-annualized`. Skip extras or LP rows with `--skip-extras` /
`--skip-lp-l2-zero`. Linear-constraint extras rewrite `A0` to the first
column of the dataset.

**Datasets.**

- `synthetic` — `factor_returns` panel (default 504 × 12, seed 42).
- `sp500` — `load_sp500_dataset()` prices → `prices_to_returns` (20 assets).
  `--quick` keeps the last 252 return rows.

**CV.** Default is all three, with parameters chosen so they produce the
same number of folds on both panels:

| Splitter | Flag | Folds (default / `--quick`) |
| --- | --- | --- |
| `WalkForward` | `--cv walk-forward` | 15 / 4 |
| `MultipleRandomizedCV` | `--cv multiple-randomized` | 15 / 4 |
| `CombinatorialPurgedCV` | `--cv purged-cpcv` | 15 / 4 |

Walk-forward `train_size` is `n_obs - target_folds * test_size`. MRC uses
its own window and inner walk-forward. CPCV uses `C(n_folds, n_test_folds)`.
Extras other than `l2_0` are skipped on MultipleRandomizedCV.

**Timing protocol.** Untimed warm-ups, then an untimed validation call, then
`--repetitions` isolated calls (each clones the estimator). Reported time is
the **median** of those repetitions. `--workers` / `--thread-limit` set
OpenMP/BLAS thread caps in this process; they do not turn on joblib.
`n_jobs != 1` selects unmodified skfolio, so leave `--n-jobs` at 1.

`--timeout SECONDS` aborts a single `cross_val_predict` call.

## How to read the numbers

On one commit:

**speed-up = `native_time / accelerated_time`**

Values greater than 1 mean accelerated is faster. Relative time is the
reciprocal. Δ time is `accelerated_time - native_time`.

PR vs main, same runner:

**Δ% = `100 * (head_time - base_time) / base_time`**

Positive Δ% means HEAD is slower. Report a table:

| Dataset | Estimator | Method | Base (s) | Head (s) | Δ Time (s) | Δ% |

Also record both SHAs, skfolio version, Python, CPU, workers, thread caps,
failed cells, and whether `--quick` was used.

Mean Sharpe is the mean of path Sharpes from
`skfolio_accelerate.scoring.path_sharpes`. Δ Sharpe is accelerated − native.

## Where files go

Single-commit `run` (never overwrites):

```
benchmark/results/YYYY-MM-DD_<git-short-sha>/
    results.csv
    results.json
    environment.json
    summary.md
    ARCHITECTURE.md
```

In-run `relative`:

```
benchmark/results/relative/YYYY-MM-DD_<head-sha>/
    base/          # full `run` artifacts for origin/main
    head/          # full `run` artifacts for HEAD
    delta.csv
    delta.json
    summary.md
```

`summary.md` is the table to paste. `ARCHITECTURE.md` is a snapshot of the
engine stack at that SHA, not a living design doc.

## Tests

```bash
source .venv/bin/activate
pytest tests/test_benchmark_suite.py
```

That checks harness math and CLI help only. It does **not** run the sweep.

Numeric defaults live in `benchmark/config.py` (`CONFIG`, `QUICK_PRESET`,
`FULL_PRESET`).

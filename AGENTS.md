# Agent instructions

This repository has a **canonical** MeanRisk `cross_val_predict` benchmark.
Coding agents must use it instead of ad-hoc timers or notebooks.

## Relative benchmarking (mandatory)

Do **not** compare a pull request's run time to a saved historical CSV, a
previous CI job, or `benchmark/results/YYYY-MM-DD_*` from another machine.

For any performance claim (faster, slower, Δ%, regression, speed-up change):

1. On **one host / one CI runner**, time the **base branch** (`main` /
   `origin/main`) first.
2. Then, **without leaving that environment**, install the **PR commit** and
   time it with the **same CLI flags**.
3. Report **Δ%** from that pair: `100 * (head_time - base_time) / base_time`.
   Positive Δ% means the PR is slower.

Use the in-run driver (it installs base from a git worktree, then reinstalls
HEAD, and always uses this checkout's `benchmark/` harness):

```bash
python benchmark/run_relative.py --base origin/main --workers 1
python benchmark/run_relative.py --base origin/main --quick --workers 1 --no-figures
```

A single-SHA native vs accelerated sweep is still:

```bash
python benchmark/run_benchmark.py --workers 1
python benchmark/run_benchmark.py --quick --workers 1
```

That measures **native skfolio vs `backend="auto"` on one commit**. It is not
a PR-vs-main timing baseline. Do not paste those seconds against an older
`results.csv`.

## Do not

- Invent or estimate timings.
- Time `cross_val_predict` in a one-off snippet and treat that as the result.
- Treat exploratory notebooks or one-off timers as the source of truth.
- Upgrade dependencies to make a run look faster.
- Treat Plotly `historical_speedup` figures as a PR gate.

Ordinary `pytest` must **not** execute the full sweep.
`tests/test_benchmark_suite.py` covers harness math only.

## How to report

**In-run Δ%** (PR vs main, same runner):

| Dataset | Estimator | Method | Base (s) | Head (s) | Δ Time (s) | Δ% |

Quote: **Δ% = `100 * (head_time - base_time) / base_time`**.

**Within one commit**, native vs accelerated:

**speed-up = `native_time / accelerated_time`** (median wall time; warm-ups
excluded).

Also report git SHAs for both legs, skfolio version, Python, CPU, workers,
thread caps, BLAS, failed cells, and whether `--quick` was used.

## Pointers

- Driver: `benchmark/run_relative.py`, `benchmark/run_benchmark.py`
- Protocol: `benchmark/README.md`, `benchmark/config.py`
- In-run outputs: `benchmark/results/relative/<date>_<head-sha>/`
  (`base/`, `head/`, `delta.csv`, `summary.md`)

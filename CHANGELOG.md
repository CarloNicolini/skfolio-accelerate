# Changelog

## Unreleased

- Docs and examples treat serial CV assembly (compiled plan, views, portfolios
  from `weights_`) as a general `cross_val_predict` saving. Closed-form
  estimators are no longer presented as an optimizer speedup.

## 0.1.0

First public release: a drop-in `cross_val_predict` for skfolio that compiles
the CV plan once and amortizes overlapping moments, compact OSQP / HiGHS /
Clarabel solves, Parameterized MeanRisk reuse, and serial assembly from
`weights_`.

- Public API: `cross_val_predict`, `AccelerationReport`, `grid_search`, and
  ranking helpers (`path_sharpes`, `ranking_precision_at_k`,
  `spearman_rank_correlation`).
- Canonical timings live in `benchmark/` (`run_relative.py`,
  `run_benchmark.py`). Do not compare a PR to a saved CSV from another machine.

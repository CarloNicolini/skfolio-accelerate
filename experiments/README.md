# LP continuation experiments

These scripts are the measurements behind the boxed MeanRisk HiGHS path.
They are not part of the public API.

## What we tried

MAD is already an LP. Sequential linearization (SLP) is the wrong name.
Warm-starting a **rebuilt** \((R-\mu)\) LP with the previous basis does **not**
cut pivots: adjacent WalkForward folds change most of \(A\) because \(\mu\) is
baked into every scenario row.

The encoding that *does* share a basis is circular scenario slots plus an
auxiliary portfolio mean \(\mu_p=\mu^\top w\) (or raw \(r_t\) for CVaR). A
roll of \(s\) observations overwrites \(s\) rows and reoptimizes from the
previous simplex basis.

`parametric_lp_cv.py` compares Clarabel, HiGHS cold starts, basis import on a
rebuilt \((R-\mu)\) LP, and that circular formulation.

## What we kept

WalkForward and MultipleRandomizedCV boxed LPs with `l2_coef=0`:

- MAD, first lower partial moment, CVaR, worst realization → persistent HiGHS

On 5,040 × 20 synthetic returns vs native skfolio (`benchmark_lp_cv.py`):

| Risk | WalkForward | MRC | CPCV |
|---|---:|---:|---:|
| MAD | 6.5× | 6.8× | 0.51× |
| FLPM | 6.5× | 6.9× | 0.52× |
| CVaR | 11.7× | 11.4× | 1.3× |
| Worst realization | 12.6× | 13.5× | 3.6× |

Mean path Sharpe matched native.

## What we did not keep

CombinatorialPurgedCV **MAD and FLPM**: training sets are block unions, not
slides. The previous basis is not a nearby vertex. HiGHS simplex with
presolve off lost to native Clarabel on long \(T\). `backend="auto"` warns
(`AccelerationWarning`) and uses unmodified skfolio.

CVaR and worst realization on CPCV stay on HiGHS (not slower than native).

A projected-subgradient MAD solver was cheaper per iteration but not accurate
enough to replace the LP.

CSV copies: `../benchmarks/lp_cv_speedups.csv` (5-year),
`../benchmarks/lp_cv_speedups_20y.csv` (20-year).

# Moreau (CVXPY + batched portfolios)

CPU-only probe of [Moreau](https://docs.moreau.so/) as a MeanRisk solver.
This is **not** the canonical PR-vs-main harness (`benchmark/run_relative.py`).
Do not paste these seconds against an older `results.csv`.

Install the extra, then run on **this** host:

```bash
pip install -e '.[moreau]'
python -m moreau check
python experiments/moreau_mean_risk.py --quick
```

Force CPU even if CUDA is present (`device="cpu"` in the script).

The native `moreau.Solver` / `CompiledSolver` CPU wheels run unlicensed for
the boxed QP used in batch timings. **CVXPY `solver=cp.MOREAU`** (skfolio
`MeanRisk(solver="MOREAU")`) currently requires `MOREAU_LICENSE_KEY` or
`~/.moreau/key`. Without a key, coverage still records that as `license`
rather than an unsupported cone. Obtain a key from
https://license.moreau.so if you need the full MeanRisk graph.

**Coverage** uses skfolio's CVXPY graph with `solver="MOREAU"` versus Clarabel
on `benchmark.estimators.mean_risk_specs` (including extras). That is the
[CVXPY example](https://docs.moreau.so/examples/cvxpy.html) path.

**Timings** compare, on WalkForward, MultipleRandomizedCV, and CPCV:

1. native skfolio Clarabel
2. native skfolio Moreau (CVXPY)
3. `backend="auto"` (OSQP / HiGHS / Clarabel reuse)
4. boxed variance: Moreau `CompiledSolver` over folds that share `n_assets`
   versus a compact OSQP loop on the **same** moments (`osqp_folds_s`). That
   isolates solver throughput from `cross_val_predict` assembly. `auto_s` is
   still reported as the end-to-end accelerator.

Quote **Δ% = `100 * (head_time - base_time) / base_time`**. Positive Δ% means
the Moreau leg is slower. Moreau has to beat `backend="auto"` on boxed
problems to be interesting; beating only native Clarabel is not enough.

Outputs: `experiments/results/moreau_coverage.csv` and
`experiments/results/moreau_cv_timings.csv`.

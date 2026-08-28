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

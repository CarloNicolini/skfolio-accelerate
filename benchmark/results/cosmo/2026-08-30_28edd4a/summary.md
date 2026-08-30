# COSMO.rs persistence experiment

Output: `/workspace/benchmark/results/cosmo/2026-08-30_28edd4a/results.csv`

Panel: 80 × 6, train=40, test=10, quick=True

| Risk | Method | Time (s) | Backend | Mean Sharpe | Sharpe |Δ| | Mean iter | Failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| VARIANCE | native | 0.034 | sklearn | -0.15669545699568418 | 0.0 |  | 0 |
| VARIANCE | auto | 0.002 | osqp | -0.1567012223966073 | 5.765400923118946e-06 |  | 0 |
| VARIANCE | cosmo | 0.001 | cosmo | -0.15670122023439803 | 5.763238713846208e-06 |  | 0 |
| SEMI_DEVIATION | native | 0.038 | sklearn | -0.08382005989226522 | 0.0 |  | 0 |
| SEMI_DEVIATION | auto | 0.004 | clarabel | -0.08382005989234433 | 7.91033905045424e-14 |  | 0 |
| SEMI_DEVIATION | cosmo | 0.017 | cosmo | -0.08381943350633718 | 6.263859280430939e-07 |  | 0 |
| CVAR | native | 0.037 | sklearn | -0.1240178675951987 | 0.0 |  | 0 |
| CVAR | auto | 0.002 | clarabel | -0.12401791761638115 | 5.002118244612497e-08 |  | 0 |
| CVAR | cosmo | 0.056 | cosmo | -0.12401568858527505 | 2.179009923650166e-06 |  | 0 |
| MAX_DRAWDOWN | native | 0.035 | sklearn | -0.07658271193403166 | 0.0 |  | 0 |
| MAX_DRAWDOWN | auto | 0.003 | clarabel | -0.07658270889089935 | 3.043132301705498e-09 |  | 0 |
| MAX_DRAWDOWN | cosmo | nan |  |  |  |  | 1 |
| VARIANCE | native | 0.033 | sklearn | -0.15669545699568418 | 0.0 |  | 0 |
| VARIANCE | auto | 0.002 | osqp | -0.1567012223966073 | 5.765400923118946e-06 |  | 0 |
| VARIANCE | cosmo | 0.001 | cosmo | -0.15670122023439803 | 5.763238713846208e-06 |  | 0 |
| VARIANCE | native | 0.033 | sklearn | -0.16938732919790056 | 0.0 |  | 0 |
| VARIANCE | auto | 0.002 | osqp | -0.1694322666285034 | 4.493743060282607e-05 |  | 0 |
| VARIANCE | cosmo | 0.001 | cosmo | -0.16943226541768802 | 4.493621978746143e-05 |  | 0 |

## Persist-mode ablations

| Risk | Mode | Time (s) | Mean iter | Rebuilds | Failures |
| --- | --- | ---: | ---: | ---: | ---: |
| VARIANCE | cold | 0.001 | 68.5 | 4 | 0 |
| VARIANCE | warm_x | 0.001 | 61.0 | 4 | 0 |
| VARIANCE | warm_xy | 0.001 | 56.25 | 4 | 0 |
| VARIANCE | persist_factor | 0.001 | 33.5 | 1 | 0 |
| VARIANCE | persist_full | 0.001 | 33.25 | 1 | 0 |
| SEMI_DEVIATION | cold | 0.008 | 355.5 | 4 | 0 |
| SEMI_DEVIATION | warm_x | 0.007 | 325.5 | 4 | 0 |
| SEMI_DEVIATION | warm_xy | 0.012 | 575.75 | 4 | 0 |
| SEMI_DEVIATION | persist_factor | 0.014 | 702.0 | 1 | 0 |
| SEMI_DEVIATION | persist_full | 0.013 | 627.75 | 1 | 0 |
| CVAR | cold | 0.040 | 2846.75 | 4 | 0 |
| CVAR | warm_x | 0.055 | 3961.75 | 4 | 0 |
| CVAR | warm_xy | 0.050 | 3542.25 | 4 | 0 |
| CVAR | persist_factor | 0.054 | 3852.75 | 1 | 0 |
| CVAR | persist_full | 0.054 | 3878.5 | 1 | 0 |
| MAX_DRAWDOWN | cold | 0.040 | nan | 1 | 1 |
| MAX_DRAWDOWN | warm_x | 0.040 | nan | 1 | 1 |
| MAX_DRAWDOWN | warm_xy | 0.040 | nan | 1 | 1 |
| MAX_DRAWDOWN | persist_factor | 0.140 | 2675.0 | 1 | 0 |
| MAX_DRAWDOWN | persist_full | 0.047 | 7.0 | 1 | 0 |

## Solver share of COSMO compact time

| Risk | Solver share | Mean iter | Moments (s) | Solve (s) | Factor (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| VARIANCE | 0.6644872014290679 | 33.25 | 0.0003641260009317193 | 0.0007211559986899374 | 2.3966e-05 |
| SEMI_DEVIATION | 0.9790546594190199 | 702.0 | 0.0003087330005655531 | 0.014431203997446573 | 0.000257151 |
| CVAR | 0.9952594539591213 | 3852.75 | 0.0002577500017650891 | 0.05411362400081998 | 0.000262902 |
| MAX_DRAWDOWN | 0.9980109150910814 | 2675.0 | 0.0002787809989968082 | 0.13987662299950898 | 8.7209e-05 |

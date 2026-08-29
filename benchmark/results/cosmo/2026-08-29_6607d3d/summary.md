# COSMO.rs persistence experiment

Output: `/workspace/benchmark/results/cosmo/2026-08-29_6607d3d/results.csv`

Panel: 80 × 6, train=40, test=10, quick=True

| Risk | Method | Time (s) | Backend | Mean Sharpe | Sharpe |Δ| | Mean iter | Failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| VARIANCE | native | 0.033 | sklearn | -0.15669545699568418 | 0.0 |  | 0 |
| VARIANCE | auto | 0.002 | osqp | -0.1567012223966073 | 5.765400923118946e-06 |  | 0 |
| VARIANCE | cosmo | 0.001 | cosmo | -0.15670122023439803 | 5.763238713846208e-06 |  | 0 |
| SEMI_DEVIATION | native | 0.035 | sklearn | -0.08382005989226522 | 0.0 |  | 0 |
| SEMI_DEVIATION | auto | 0.004 | clarabel | -0.08382005989234433 | 7.91033905045424e-14 |  | 0 |
| SEMI_DEVIATION | cosmo | 0.016 | cosmo | -0.08381943350633718 | 6.263859280430939e-07 |  | 0 |
| CVAR | native | 0.035 | sklearn | -0.1240178675951987 | 0.0 |  | 0 |
| CVAR | auto | 0.002 | clarabel | -0.12401791761638115 | 5.002118244612497e-08 |  | 0 |
| CVAR | cosmo | 0.055 | cosmo | -0.12401568858527505 | 2.179009923650166e-06 |  | 0 |
| MAX_DRAWDOWN | native | 0.035 | sklearn | -0.07658271193403166 | 0.0 |  | 0 |
| MAX_DRAWDOWN | auto | 0.003 | clarabel | -0.07658270889089935 | 3.043132301705498e-09 |  | 0 |
| MAX_DRAWDOWN | cosmo | 0.140 | cosmo | -0.2421366512570071 | 0.16555393932297544 |  | 0 |
| VARIANCE | native | 0.033 | sklearn | -0.15669545699568418 | 0.0 |  | 0 |
| VARIANCE | auto | 0.002 | osqp | -0.1567012223966073 | 5.765400923118946e-06 |  | 0 |
| VARIANCE | cosmo | 0.001 | cosmo | -0.15670122023439803 | 5.763238713846208e-06 |  | 0 |
| VARIANCE | native | 0.033 | sklearn | -0.16938732919790056 | 0.0 |  | 0 |
| VARIANCE | auto | 0.001 | osqp | -0.1694322666285034 | 4.493743060282607e-05 |  | 0 |
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
| SEMI_DEVIATION | persist_full | 0.012 | 627.75 | 1 | 0 |
| CVAR | cold | 0.040 | 2846.75 | 4 | 0 |
| CVAR | warm_x | 0.055 | 3961.75 | 4 | 0 |
| CVAR | warm_xy | 0.050 | 3542.25 | 4 | 0 |
| CVAR | persist_factor | 0.054 | 3852.75 | 1 | 0 |
| CVAR | persist_full | 0.055 | 3878.5 | 1 | 0 |
| MAX_DRAWDOWN | cold | 0.041 | nan | 1 | 1 |
| MAX_DRAWDOWN | warm_x | 0.041 | nan | 1 | 1 |
| MAX_DRAWDOWN | warm_xy | 0.041 | nan | 1 | 1 |
| MAX_DRAWDOWN | persist_factor | 0.140 | 2675.0 | 1 | 0 |
| MAX_DRAWDOWN | persist_full | 0.042 | 7.0 | 1 | 0 |

## Solver share of COSMO compact time

| Risk | Solver share | Mean iter | Moments (s) | Solve (s) | Factor (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| VARIANCE | 0.6757677180968177 | 33.25 | 0.0002051690003099793 | 0.0004276149998077017 | 1.5206e-05 |
| SEMI_DEVIATION | 0.9804521062246658 | 702.0 | 0.0002706409998154413 | 0.013574379999681696 | 0.000254903 |
| CVAR | 0.9957178610917524 | 3852.75 | 0.00023208700008581218 | 0.05396676200007278 | 0.000252953 |
| MAX_DRAWDOWN | 0.9979395575559011 | 2675.0 | 0.0002886120000766823 | 0.1397842160001801 | 8.953700000000001e-05 |

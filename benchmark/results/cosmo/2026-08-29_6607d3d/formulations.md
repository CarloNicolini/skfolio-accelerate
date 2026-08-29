| Risk | Cone | Auto engine | Fixed-T class | Expanding class | Variables | COSMO | Sparsity constant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VARIANCE | QP | osqp | C | C | n | compact | yes |
| ANNUALIZED_VARIANCE | QP | none | C | C | n | sequential | yes |
| STANDARD_DEVIATION | SOCP | sequential | C | C | n + 1 (radius) | sequential | yes |
| ANNUALIZED_STANDARD_DEVIATION | SOCP | none | C | C | n + 1 | sequential | yes |
| SEMI_VARIANCE | QP | clarabel | B | E | n + T | compact | yes |
| ANNUALIZED_SEMI_VARIANCE | QP | none | B | E | n + T | sequential | yes |
| SEMI_DEVIATION | SOCP | clarabel | B | E | n + T + 1 | compact | yes |
| ANNUALIZED_SEMI_DEVIATION | SOCP | none | B | E | n + T + 1 | sequential | yes |
| MEAN_ABSOLUTE_DEVIATION | LP | highs | B | E | n + T | compact | yes |
| FIRST_LOWER_PARTIAL_MOMENT | LP | highs | B | E | n + T | compact | yes |
| WORST_REALIZATION | LP | highs | B | E | n + 1 | compact | yes |
| CVAR | LP | highs | B | E | n + 1 + T | compact | yes |
| EVAR | ExpCone | clarabel | B | E | n + 2 + T | compact | yes |
| MAX_DRAWDOWN | LP | clarabel | B | E | n + (T+1) + 1 | compact | yes |
| AVERAGE_DRAWDOWN | LP | clarabel | B | E | n + T + 1 | compact | yes |
| CDAR | LP | clarabel | B | E | n + (T+1) + 1 + T | compact | yes |
| EDAR | ExpCone | clarabel | B | E | n + (T+1) + 2 + T | compact | yes |
| ULCER_INDEX | SOCP | sequential | B | E | n + T + 1 | sequential | yes |
| GINI_MEAN_DIFFERENCE | LP | sequential | B | E | n + 3T | sequential | yes |

### Objectives

* **MINIMIZE_RISK** — Linear objective on the risk epigraph / quadratic term.
* **MAXIMIZE_UTILITY** — Adds −μᵀw; q on the weight block changes each fold (still B/C).
* **MAXIMIZE_RETURN** — Risk is dropped from the objective; risk constraints may remain. Boxed compact engines do not implement this; sequential CVXPY does.
* **MAXIMIZE_RATIO** — Charnes-Cooper / Schaible homogenization adds a free factor variable and a return equality. Sequential path refuses it; fit-assemble / native.

### Constraints

* **weight box + budget** — class A, COSMO `compact`. Structure constant; values constant in compact engines.
* **l2_coef** — class C, COSMO `compact`. Adds a constant diagonal to P.
* **l1_coef** — class B, COSMO `sequential`. Auxiliary abs variables; not compacted.
* **min_return** — class A, COSMO `sequential`. Linear μᵀw ≥ r; μ in A or as a Parameter.
* **linear_constraints / groups** — class B, COSMO `sequential`. Extra A rows; names need DataFrame columns.
* **transaction_costs / management_fees** — class B, COSMO `sequential`. Affine terms in return / risk.
* **max_turnover / previous_weights** — class F, COSMO `no`. Sequential previous_weights; compact and sequential refuse.
* **max_tracking_error** — class C, COSMO `sequential`. SOC of (Rw − y); sequential refuses Parameterization.
* **mu uncertainty set** — class C, COSMO `sequential`. Extra SOC; sequential refuses.
* **covariance uncertainty set (generic)** — class F, COSMO `no`. Lifted SDP (S₊). COSMO.rs has no PSD cone.
* **covariance uncertainty set (compact)** — class C, COSMO `sequential`. Extra quadratic residual; still sequential.
* **add_constraints / add_objective / custom** — class F, COSMO `sequential`. Unknown cone class; fit-assemble.
* **efficient_frontier_size** — class F, COSMO `no`. Multiple solves; assemble refuses.

# Persistent COSMO.rs for MeanRisk walk-forward CV

**Status:** experimental, opt-in, **not** selected by `backend="auto"`.

**Question:** can a native persistent COSMO.rs workspace exploit the temporal
and structural similarity of MeanRisk problems inside long walk-forward (and
other massive) cross-validation, and if so where and by how much?

**Short measured answer:** persistence can cut ADMM iterations on boxed
**variance** (class C, `update_p`). It does **not** beat compact OSQP. On
class-B scenario problems (`update_a` always drops the KKT factorisation)
stale ADMM state often **increases** iterations. End-to-end, existing
`backend="auto"` (OSQP / HiGHS / Clarabel) remains faster. COSMO.rs is not
a first-class auto backend on this evidence.

This document separates **measured** results from **hypotheses**. Timings are
in-run on one host. They are **not** a PR-vs-`main` relative benchmark; do
not paste them against historical `benchmark/results/` CSVs. See `AGENTS.md`.

Reproduce:

```bash
python benchmark/run_cosmo.py --quick
python benchmark/run_cosmo.py
```

Outputs: `benchmark/results/cosmo/<date>_<sha>/`.

---

## 0. Execution map

Native `skfolio.model_selection.cross_val_predict(MeanRisk, X, cv=WalkForward)`:

1. sklearn clones the estimator per fold
2. slices train/test
3. `MeanRisk.fit` estimates μ and Σ (or scenario returns)
4. builds a new CVXPY problem
5. canonicalizes
6. instantiates Clarabel (default)
7. solves
8. writes `weights_`, then `predict` on the test slice

`skfolio_accelerate.cross_val_predict` with `backend="auto"`:

1. `classify_call` → compact / sequential / assemble / sklearn
2. `compile_cv_plan` (splits once)
3. overlapping moments (`path_moment_session`)
4. one compact engine (OSQP / HiGHS / Clarabel) reused across folds
5. `assemble_prediction` from fold weights

`backend="cosmo"` (this work) swaps step 4 onto `PersistentCosmo` with the
**same** compact `P, q, A, b, cones` as Clarabel/OSQP. It does not rewrite
skfolio's CVXPY graph (Option C for all of MeanRisk was not justified).

Estimator clones still exist on the native path. The compact path does not
clone MeanRisk per fold.

---

## 1. Which MeanRisk formulations can COSMO.rs solve?

COSMO.rs cones (from source): zero, nonnegative, SOC, exp, dual-exp, power.
**No SDP / PSD.**

| Support | Formulations |
| --- | --- |
| Compact COSMO path | VARIANCE, SEMI_VARIANCE, SEMI_DEVIATION, MAD, FLPM, WORST_REALIZATION, CVAR, EVAR, MAX_DRAWDOWN, AVERAGE_DRAWDOWN, CDAR, EDAR — boxed, `MINIMIZE_RISK` / `MAXIMIZE_UTILITY`, optional `l2_coef` |
| Sequential only (not compacted) | annualized aliases, STANDARD_DEVIATION, ULCER_INDEX, GINI_MEAN_DIFFERENCE, L1, groups, costs, tracking error, μ-uncertainty |
| Cannot | generic covariance uncertainty (lifted SDP), MIP cardinality, `MAXIMIZE_RATIO` on compact path |

Full table: `docs/cosmo_meanrisk_formulations.md` (generated from
`skfolio_accelerate.formulations`).

## 2. Which cannot?

* SDP covariance uncertainty sets
* Combinations the compact engines already refuse (ratio, turnover /
  `previous_weights`, custom CVXPY hooks, efficient frontier)
* ND power cones if skfolio ever emitted them (it does not in the inspected
  MeanRisk graph)

## 3. Cone structure per supported formulation

See the formulation table. Dominant classes:

* VARIANCE (compact): **QP**, Zero(1) + Nonneg(2n)
* SEMI_VARIANCE: **QP** with T auxiliaries
* SEMI_DEVIATION: **SOCP**
* MAD / FLPM / CVaR / worst / drawdowns / CDaR: **LP** (tiny ℓ₂ diagonal if `l2_coef>0`)
* EVaR / EDaR: **ExpCone**

skfolio's own variance graph is SOC of √Σ; the compact OSQP/COSMO engines
use the equivalent QP `½ wᵀ (2Σ) w`.

## 4. What stays fixed across walk-forward folds?

Fixed training length `T` (rolling WalkForward):

* VARIANCE: `n`, `m`, cones, `A` (budget + box), `P` sparsity. Class **C**.
* Scenario compact risks: `n`, `T`, cone dims, sparsity of `A` and `P`.
  Numeric `A` (returns) and often `b` change. Class **B**.

## 5. Which numerical data change?

* VARIANCE: `P ← 2Σ` (and `q ← −μ` if utility)
* Scenario: `A` rows that hold `R`; `b` for some epigraphs; `q` if utility
* Expanding `T` or CPCV with changing train length: cone dimensions change
  (class **E**). The workspace is dropped (`solver = None`).

## 6. What COSMO internal state can actually be reused?

Verified in COSMO.rs `src/solver/mod.rs`, not inferred from docs.

| Update | Invalidated |
| --- | --- |
| `update_q` / `update_b` | nothing in KKT; scaled `q`/`b` rewritten |
| `update_p` same CSC pattern | numerical KKT refactor (`kkt.rebuild`) |
| `update_p` pattern change | `kkt = None` (full rebuild) |
| `update_a` | **always** `kkt = None` |
| sparsity / dimensions change | new `CosmoSolver` |

After the first solve, Ruiz scaling (`D`, `E`, `c`) is kept and applied to
later updates. `warm_start(x, y)` maps unscaled primal/dual into the scaled
workspace. Adaptive `ρ` and Anderson history live in the workspace until
`reset`.

GitHub `CarloNicolini/COSMO.rs` **Python** bindings on `main` expose only
`update_q`, `update_b`, `warm_start`. Rust already has `update_p` /
`update_a` / `reset`. This experiment used a local maturin build that
exports those methods. Without them, persist modes reconstruct each fold.

## 7. Does persistent state reduce ADMM iterations?

**Measured** (quick panel, 80×6, train=40, test=10, 4 windows):

| Risk | cold mean iter | persist_full mean iter |
| --- | ---: | ---: |
| VARIANCE | 68.5 | 33.25 |
| SEMI_DEVIATION | 355.5 | 627.75 |
| CVAR | 2846.75 | 3878.5 |
| MAX_DRAWDOWN | failed (max_iter) | 7.0 (see §23) |

So: **yes for variance**, **no for the measured SOC/LP scenario problems**.
Carrying more ADMM state is not uniformly better.

## 8. Does factorization reuse reduce solve time?

On variance, `persist_factor` and `persist_full` both rebuilt once
(`n_rebuilds=1`) vs 4 cold rebuilds, with ~half the iterations. Wall time
on this tiny QP is ~1 ms; the difference is not a CV-scale win versus OSQP
(also ~1–2 ms end-to-end for the same cell).

On class B, `update_a` drops KKT, so “persistent factorization” does not
survive the update that walk-forward actually performs.

## 9. How much does CVXPY canonicalization cost?

On the **compact** COSMO path there is no CVXPY. A `cosmo-profile` row
(after fixing an `UnboundLocalError` in the first driver) reports
moments vs solve share. On the 80×6 panel, compact solve time dominates
moments; Python/CV orchestration still dominates native skfolio.

Native skfolio rebuilds CVXPY every `fit`. That is why `backend="auto"`
already beats native by ~20× on this panel **without COSMO**. COSMO cannot
claim that speedup; OSQP/Clarabel/HiGHS already removed canonicalization.

Option A (CVXPY `COSMO_RUST` interface) was **not** timed. COSMO.rs ships
`cosmo_rs.cvxpy_interface`; CVXPY typically constructs a new solver object
per `problem.solve()`, so Option A is a poor persistence vehicle.

## 10. Can the solver persist through CVXPY, or is a lower-level path required?

A lower-level path is required for true persistence. This integration is
Option B/C on the **existing compact engines**: same matrices, persistent
`CosmoSolver`. Option C as a from-scratch MeanRisk canonicalizer was not
implemented; profiling did not show compact matrix bind as the bottleneck
relative to OSQP/Clarabel.

## 11. Speedup versus native skfolio + Clarabel

Quick panel, same host, `n_jobs=1`, median of 1 timed repeat after 1 warmup.
Δ% here is `100 * (cosmo - native) / native` (negative = COSMO faster).
These are **not** `run_relative.py` PR-vs-main numbers.

| Risk | Native (s) | COSMO persist (s) | Δ% vs native | Sharpe \|Δ\| |
| --- | ---: | ---: | ---: | ---: |
| VARIANCE | 0.041 | 0.001 | −97% | 6e-6 |
| SEMI_DEVIATION | 0.041 | 0.014 | −66% | 4e-7 |
| CVAR | 0.038 | 0.056 | **+47%** | 6e-7 |
| MAX_DRAWDOWN | 0.035 | 0.045 | **+29%** | 2.4e-3 |

COSMO vs native on variance is real but **smaller than auto-vs-native**,
and is the compact-engine effect, not a COSMO-specific miracle.

## 12. Speedup versus current `backend="auto"`

| Risk | auto (s, engine) | COSMO (s) | COSMO vs auto |
| --- | ---: | ---: | ---: |
| VARIANCE | 0.002 OSQP | 0.001 | comparable (noise at 1 ms) |
| SEMI_DEVIATION | 0.005 Clarabel | 0.014 | COSMO **slower** |
| CVAR (`l2=1e-5` → Clarabel not HiGHS) | 0.002 | 0.056 | COSMO **much slower** |
| MAX_DRAWDOWN | 0.003 Clarabel | 0.045 | COSMO **much slower** |

## 13. Speedup versus cold-start COSMO

Variance: persist_full 0.001 s vs cold 0.001 s (iteration cut 68.5 → 33;
wall clock drowned in Python). Scenario: persist_full **slower** and more
iterations than cold.

## 14. Which formulations benefit most?

Boxed **variance** (class C). That is already OSQP’s home turf.

## 15. Which benefit least?

CVaR and drawdown LPs. ADMM is a poor simplex substitute; HiGHS already
wins on `l2_coef=0` boxed LPs. EVaR/CDaR may fail to converge at
Clarabel-style `1e-8`; the compact COSMO settings use `1e-5` and disable
Anderson on the slow LP family.

## 16. Long walk-forward sequences?

The measured sequence is **4 folds**. Hypothesis that thousands of folds
would amortize setup better is **unmeasured**. Variance iteration reduction
would still have to beat OSQP, which already warm-starts and updates `P`.

## 17. Rolling vs expanding windows

Driver cells `walk-forward-rolling` and `walk-forward-expanding` use
`WalkForward(..., expand_train=True)` for expanding. Expanding scenario
risks are class **E** (growing `T`); the COSMO workspace is discarded when
`n_observations` changes. Variance stays class **C**. Numbers: rerun
`benchmark/run_cosmo.py`.

## 18. MultipleRandomizedCV?

Non-`--quick` driver includes a small MRC cell. Trajectories are
independent subsample windows; a **single** global persistent solver is
the wrong model. The compact engine already keys the cache on
`(n_assets, T)`. Clustering of similar canonical problems was **not**
implemented (Phase 16): the benchmark did not justify it.

## 19. CombinatorialPurgedCV?

Same: non-`--quick` cell. Changing train lengths are class E.
`backend="auto"` already refuses persistent HiGHS continuation on CPCV
MAD/FLPM (native fallback). COSMO persistence has the same structural
issue.

## 20. Parallelism (`n_jobs`) vs state reuse

Amortized backends already require `n_jobs in {None, 1}`. Persistent COSMO
is sequential per trajectory. Driver cells `native-n_jobs=1` and
`native-n_jobs=2` time native WalkForward. Break-even: if parallel native
wall-clock on `F` folds is `T_native / min(F, workers)` plus overhead, a
sequential persistent COSMO trajectory must be faster than that **and**
faster than sequential auto. On the 4-fold toy panel, auto sequential
already beats native; adding COSMO state does not change the parallel
story. **Hypothesis (unmeasured at scale):** many-core native Clarabel
could beat sequential persistent COSMO even if per-fold COSMO were
faster—because auto is already faster per fold.

## 21. Break-even sequential stateful vs parallel cold

Not identified as a COSMO-vs-Clarabel number: auto sequential compact is
the incumbent. COSMO would need to beat **OSQP/HiGHS/Clarabel**, not only
native `n_jobs=-1`.

## 22. Numerical differences versus Clarabel / OSQP

Tests allow `atol=5e-3` weights ( `2e-2` for slow LPs). Quick-panel Sharpe
errors vs native: ~1e-6 for variance / semi-deviation / CVaR; ~2e-3 for
max drawdown (COSMO ADMM gap vs IPM). Not bitwise equality. Feasibility:
budget and bounds checked in tests.

## 23. Failure modes

* CDaR / some utility drawdowns / EVaR: COSMO.rs may hit `max_iter`
  (tests skip).
* Tight `1e-8` on LPs: routinely exhausts iterations; settings use `1e-5`
  for the slow LP family. Tolerances are **documented and different** from
  Clarabel defaults.
* `persist_full` on MAX_DRAWDOWN reported mean 7 iterations and success
  after cold failed on the same windows. Treat that as a **stale-state
  false solve**, not a speedup. Default scenario persist mode is therefore
  `persist_factor` (drop ADMM iterates; do not carry a possibly infeasible
  warm start). Restart policy `status` rebuilds on non-Solved.
* GitHub Python API without `update_p`: persist modes reconstruct (warning).

## 24. Restart policies

Implemented: `never`, `status`, `iter_threshold` (default threshold 8000).
Measured justification: do not blindly reuse full ADMM state on class B.
Variance keeps `persist_full` + `status`. Scenario default `persist_factor`.
Adaptive covariance-change restarts were **not** implemented; iteration
count already captures “this warm start is not helping.”

## 25. Is COSMO.rs worth a first-class auto backend?

**No**, on current measurements. Keep it opt-in (`backend="cosmo"` or
`MeanRisk(solver="COSMO")`). Auto order stays OSQP → HiGHS → Clarabel →
sequential CVXPY → assemble → sklearn.

## 26. Production-quality follow-up

1. Upstream Python `update_p` / `update_a` / `reset` on COSMO.rs.
2. Larger panels: 50–250 assets, hundreds of folds, rolling and expanding,
   MRC/CPCV, `n_jobs` break-even on a many-core host.
3. Do not put COSMO on `auto` unless it beats Clarabel on SEMI_DEVIATION /
   EVaR SOC-exp problems **and** HiGHS on LPs at equal tolerances.
4. Incremental μ/Σ (Phase 22): **not** done. Moments are a small share of
   compact time; auto already incremental-updates overlapping windows.
5. Option A CVXPY timing: only if someone needs a drop-in `solver="COSMO"`
   inside native `fit` (no persistence).

---

## Architecture choice

| Option | Verdict |
| --- | --- |
| A — CVXPY solver replacement | Easy, weak persistence. Not wired. |
| B — reuse compact canonical matrices | **Implemented.** Necessary for persistence. |
| C — new MeanRisk canonicalizer in Rust | Not justified; bind time is not the auto bottleneck. |

## Environment (quick run that produced the tables above)

* skfolio-accelerate base SHA before this branch: `d42f946`
* skfolio 1.0.1
* Python 3.12.3
* clarabel 0.11.1, osqp 1.1.3, highspy 1.15.1, numpy 2.5.2, cosmo-rs 0.1.0
  (local `/tmp/COSMO.rs` maturin build)
* cvxpy-base (no full `cvxpy` extra)
* Panel: 80 observations × 6 assets, train=40, test=10, 4 folds, `--quick`
* Workers: 1

Later runs live under `benchmark/results/cosmo/` and are not overwritten.

## Success criteria

| Level | Result |
| --- | --- |
| 1 Correctness | Compact COSMO matches OSQP/Clarabel within documented tols on the tested boxed set; some LPs/exp cones skip |
| 2 Persistence | Variance: one rebuild, correct weights vs cold. Scenario: `update_a` works; full ADMM reuse is harmful |
| 3 Solver speed | Variance iterations drop; wall clock vs OSQP is a tie at 1 ms. Scenario COSMO slower than Clarabel |
| 4 End-to-end CV | Auto already provides the large native→compact speedup. COSMO does not improve on auto |
| 5 Broad coverage | Table in `formulations.py`; not every constraint combination is compacted |
| 6 Robustness | `persist_full` on drawdown is a documented hazard; defaults avoid it |
| 7 Scalability | 10–1000 asset sweep **not** measured in the quick panel |

## Hypothesis vs measurement

**Hypothesis:** walk-forward MeanRisk is a highly correlated ADMM trajectory
and full COSMO state continuation will dominate Clarabel/OSQP.

**Measurement:** the correlation is real for variance QPs (fewer iterations
with `update_p`), but (1) OSQP already does that, (2) scenario `update_a`
destroys the factorisation COSMO would like to keep, (3) ADMM state can go
stale, (4) CVXPY canonicalization is already gone on the winning auto path.
The bottleneck of native CV is **not** a missing COSMO workspace; it is
repeated CVXPY + cold Clarabel, which `backend="auto"` already removed.

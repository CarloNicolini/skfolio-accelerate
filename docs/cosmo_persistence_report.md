# Persistent COSMO.rs for MeanRisk walk-forward CV

**Status:** experimental, opt-in, **not** selected by `backend="auto"`.

**Central question:** Can a persistent COSMO solver, carrying primal/dual
state and reusing as much linear-algebra structure as mathematically valid
between consecutive walk-forward folds, substantially reduce total
walk-forward optimization time while remaining numerically equivalent to
native skfolio?

**Short measured answer:** On boxed **variance** (class C), persistence cuts
ADMM iterations roughly in half (`68.5 → 33`). It does **not** beat compact
OSQP end-to-end. On class-B scenario problems, `update_a` drops the KKT object
in COSMO.rs `main`, and stale ADMM state often **increases** iterations.
`backend="auto"` (OSQP / HiGHS / Clarabel) remains faster. COSMO.rs is not a
first-class auto backend on this evidence.

This document separates **measured** results from architecture notes. Timings
are in-run on one host. They are **not** a PR-vs-`main` relative benchmark; do
not paste them against historical `benchmark/results/` CSVs. See `AGENTS.md`.

Reproduce:

```bash
python benchmark/run_cosmo.py --quick
python benchmark/run_cosmo.py
```

Outputs: `benchmark/results/cosmo/<date>_<sha>_<cosmo-sha>/`.

---

## Persist modes (A–E)

| Mode | Name | Behaviour |
| --- | --- | --- |
| A | `cold` | New solver every fold; default ADMM state |
| B | `warm_x` / `warm_primal` | New solver; warm-start primal `x` |
| C | `warm_xy` / `warm_primal_dual` | New solver; warm-start `x` and dual `y` (COSMO.rs has no public slack `s` inject; `s` is recovered from `Ax + s = b`) |
| D | `persist_factor` / `persist_structure` | Keep workspace; update data; reset ADMM. Same-sparsity `update_p` numerically refactors (QDLDL symbolic analysis reused via `rebuild`). `update_a` always drops KKT |
| E | `persist_full` / `persist_numerical` | Keep ADMM warm state when allowed, but **claim numerical-factor reuse only after verifying `P` and `A` are numerically identical**. Changing covariance therefore forbids Mode-E factor reuse |

Defaults: variance → Mode E (`persist_full`); scenario risks → Mode D
(`persist_factor`).

---

## What changes between folds?

Verified from the compact `P, q, A, b, cones` that the engines actually emit
(not from documentation alone).

| Quantity | Variance (fixed `n`) | Scenario risks (fixed `n`, `T`) |
| --- | --- | --- |
| Cone structure | invariant | invariant |
| Sparsity of `A` | invariant | invariant |
| Sparsity of `P` | invariant (dense upper Δ) | invariant (tiny ℓ₂ diagonal) |
| Numeric `P` | **changes** (`2Σ`) | usually fixed |
| Numeric `A` | fixed (budget + box) | **changes** (returns / deviations) |
| `q` / `b` | `q` may change (utility) | often change |
| Symbolic KKT | reusable under `update_p` | **dropped** by `update_a` |
| Numeric KKT | refactor after `update_p` | rebuild after `update_a` |
| Persist class | **C** | **B** |

Expanding `T` or CPCV with changing train length → class **E** (workspace
dropped).

COSMO.rs `update_a` always sets `kkt = None`. Same-sparsity symbolic reuse
across scenario folds is **not** available in the current backend. Mode E
therefore almost never reuses a numerical factor on variance walk-forward
(`P_t ≠ P_{t+1}`); the mode still warm-starts ADMM after a valid `update_p`.

---

## Measured panel

* skfolio-accelerate SHA `5aa2c6a`
* COSMO.rs `ec6268c`
* skfolio 1.0.2, Python 3.12.3, Clarabel 0.11.1, OSQP 1.1.3, HiGHS 1.15.1
* Panel: 80 × 6, train=40, test=10, 4 folds, `--quick`, `n_jobs=1`
* Tolerances: COSMO `1e-8` (tight) except slow LP family settings in ablations

Machine-readable copy: `benchmark/results/cosmo/2026-08-30_5aa2c6a_ec6268c/`.

---

## Question 1 — COSMO vs native Clarabel (single CV trajectory)

End-to-end `cross_val_predict` wall time (4 folds), same data / folds /
estimator / hardware:

| Risk | Native (s) | COSMO (s) | speedup_vs_native | Sharpe ‖Δ‖ |
| --- | ---: | ---: | ---: | ---: |
| VARIANCE | 0.033 | 0.002 | **16.5×** | 6e-6 |
| SEMI_DEVIATION | 0.036 | 0.015 | **2.4×** | 6e-7 |
| CVAR | 0.034 | 0.051 | **0.67×** (slower) | 2e-6 |
| MAX_DRAWDOWN | 0.036 | *refused on CV path* | — | — |

The variance win is largely the **compact engine** effect (no CVXPY), not a
COSMO-specific miracle — see Question 7 / auto comparison.

---

## Question 2 — Warm-started vs cold COSMO

| Risk | cold (s) | persist (s) | speedup_vs_cosmo_cold | cold mean iter | warm mean iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| VARIANCE | 0.001 | 0.001 (`persist_full`) | ~1× at 1 ms | 68.5 | 33.2 |
| SEMI_DEVIATION | 0.007 | 0.013 (`persist_factor`) | **0.54×** | 355.5 | 702.0 |
| CVAR | 0.038 | 0.051 (`persist_full`) | **0.75×** | 2846.8 | 3878.5 |

Warm starting helps iterations on variance; on scenario problems it often
hurts. Wall-clock on the tiny variance QP is drowned in Python overhead.

---

## Question 3 — Iterations: `x` only vs `x + y`

Variance (4 folds):

| Mode | Mean iterations |
| --- | ---: |
| A cold | 68.5 |
| B warm_x | 61.0 |
| C warm_xy | 56.2 |
| D persist_factor | 33.5 |
| E persist_full | 33.2 |

Carrying duals (C) saves a few iterations beyond primal-only (B). Persistent
structure (D/E) saves far more on variance. On SEMI_DEVIATION / CVaR, Mode C
increased iterations vs cold.

---

## Question 4 — How much solver structure can be reused?

| Mechanism | Variance | Scenario |
| --- | --- | --- |
| Workspace object | yes (D/E) | yes (D/E) |
| Ruiz scaling after first solve | yes | yes |
| Same-sparsity `update_p` numeric refactor | yes | n/a (P usually fixed) |
| Symbolic analysis across `update_a` | n/a (`A` fixed) | **no** (COSMO.rs drops KKT) |
| Mode E numerical factor without update | only when `P,A` identical (rare for Σ) | only when `A` identical (rare) |
| ADMM `x,y` continuation | helpful | often harmful |

---

## Question 5 — When `P` changes, can symbolic / sparse structure still be reused?

**Yes, for variance:** `update_p` with unchanged CSC pattern calls QDLDL
`rebuild` (symbolic analysis retained; numeric values refreshed). Blind reuse
of an old numerical LDLᵀ without refactor is **not** done — Mode E verifies
identity and refuses to claim factor reuse when `P` changed.

**No, for scenario `A` updates:** `update_a` always `kkt = None`.

---

## Question 6 — Which MeanRisk estimators benefit?

| Category | Estimators |
| --- | --- |
| Strong iteration improvement (still not > OSQP) | boxed VARIANCE (class C) |
| Moderate / neutral vs native, slower than Clarabel | SEMI_DEVIATION |
| Regression vs Clarabel / auto | CVAR (ADMM LP) |
| Unsupported / refused on CV path | MAD, FLPM, MAX_DRAWDOWN, AVERAGE_DRAWDOWN, CDAR |
| Skip / fragile | EVAR (may hit `max_iter`) |

---

## Question 7 — End-to-end walk-forward speedup

| Risk | Native | auto | COSMO | winner |
| --- | ---: | ---: | ---: | --- |
| VARIANCE | 0.033 s | 0.002 s OSQP | 0.002 s | auto / COSMO tie at 1–2 ms; auto is the product path |
| SEMI_DEVIATION | 0.036 s | 0.003 s Clarabel | 0.015 s | **auto** |
| CVAR | 0.034 s | 0.002 s Clarabel | 0.051 s | **auto** |
| MAX_DRAWDOWN | 0.036 s | 0.003 s Clarabel | refused | **auto** |

Moments vs solve share on the compact COSMO path: variance ~73% solve, scenario
risks ≥98% solve. Python model construction is **not** the COSMO bottleneck once
the compact matrices exist; ADMM iteration count is.

Rolling vs expanding variance: both ~0.033 native / 0.002 auto / 0.002 COSMO.
Native `n_jobs=2` (0.022 s) does not catch sequential compact OSQP/COSMO on
this 4-fold panel.

---

## Question 8 — Justify a persistent COSMO path in skfolio-accelerate?

**As an opt-in research backend: yes** (this PR).

**As a `backend="auto"` replacement: no.** Auto already removed CVXPY
canonicalization and uses OSQP / HiGHS / Clarabel with their own persistence.
COSMO does not beat those engines at equal portfolio accuracy on the measured
panel. The desired architecture

```text
fold t → state_t → fold t+1 warm-started from state_t
```

is real for variance ADMM iterations, but the end-to-end product win was
already achieved by compact OSQP.

---

## Numerical correctness

* Unit tests: COSMO weights vs OSQP/Clarabel within `atol=5e-3` (tighter LP
  family `2e-2`), budget and box constraints checked.
* CV Sharpe ‖Δ‖ vs native on the quick panel: ~1e-6 for variance /
  semi-deviation / CVaR.
* Mode E identity test: changing covariance → `numerical_factor_reused=False`;
  identical rematch → `True`.
* Drawdown `persist_full` reported mean 7 iterations after cold failures —
  treated as stale-state hazard; those LPs are refused on `cross_val_predict`.

---

## Adaptive stopping (Pareto)

Not loosened in product code. Ablation settings for slow LPs use `1e-5` only
inside `make_cosmo_engine` experiments; CV entry refuses those LPs. A full
tolerance Pareto on large panels remains future work; this quick panel already
shows Clarabel wins at tight tolerances for SOC/LP scenarios.

---

## Architecture choice

| Option | Verdict |
| --- | --- |
| A — CVXPY `COSMO_RUST` per `solve()` | Weak persistence. Not wired. |
| B — reuse compact canonical matrices | **Implemented.** |
| C — new MeanRisk canonicalizer in Rust | Not justified; bind time ≪ ADMM on scenario cells. |

---

## Conclusion

Walk-forward MeanRisk **does** provide a correlated ADMM trajectory for boxed
variance, and Modes D/E cut iterations. That is not enough to displace OSQP
or Clarabel. Persistent COSMO stays experimental and opt-in. The evidence-based
answer to the research question is: **not a meaningful end-to-end improvement
over `backend="auto"` for the actual skfolio walk-forward workloads measured
here.**

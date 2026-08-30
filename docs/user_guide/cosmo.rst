.. _cosmo:

****************************************
Persistent COSMO.rs (experimental)
****************************************

.. currentmodule:: skfolio_accelerate

.. warning::

   COSMO.rs is **not** the default engine. ``backend="auto"`` still selects
   OSQP, HiGHS, or Clarabel. Enable COSMO only for the persistence experiment
   described here. Compare every result to native skfolio.

`COSMO.rs <https://github.com/CarloNicolini/COSMO.rs>`_ is a native Rust
ADMM / operator-splitting conic solver. This package can reuse one COSMO
workspace across WalkForward folds of boxed MeanRisk.

It does **not** replace Clarabel. On boxed variance OSQP already updates
``P`` and warm-starts; on boxed LPs HiGHS already reuses a simplex basis.
COSMO is interesting only where those engines do not apply, or as a
controlled ablation of ADMM state reuse. ``cross_val_predict`` refuses
MAD / FLPM / max-drawdown / average-drawdown / CDaR: ADMM is not a
reliable LP engine there (measured Sharpe errors vs Clarabel). Use
``make_cosmo_engine`` only for the persistence ablation.

Enabling the backend
********************

Install the optional extra (build COSMO.rs with maturin), then::

    from skfolio.optimization import MeanRisk
    from skfolio_accelerate import cross_val_predict

    prediction, report = cross_val_predict(
        MeanRisk(), X, cv=cv, backend="cosmo", return_report=True
    )
    assert report.backend == "cosmo"

``MeanRisk(solver="COSMO")`` with ``backend="auto"`` selects the same
compact path. If COSMO fails, the call retries native Clarabel
(``solver`` is rewritten for the fallback).

What is reused
**************

Verified from COSMO.rs ``main`` (PR #3 Python API, PR #4 ADMM hot paths):

* ``update_q`` / ``update_b`` — no KKT refactor
* ``update_p`` — numerical refactor when the sparsity pattern is unchanged;
  full rebuild if the pattern changes
* ``update_a`` — drops the KKT object; the next ``solve()`` rebuilds it
  (scenario-return walk-forward is still class B)
* ``reset("cold"|"factor")`` — drop ADMM iterates; ``factor`` keeps ρ and
  a still-valid factorisation (not valid after ``update_a``)
* ``warm_start(x, y)`` — unscaled primal / dual
* After the first ``solve``, Ruiz scaling is kept
* Adaptive ``ρ`` and Anderson history stay until ``reset`` or a reconstruct
  persist mode

Walk-forward with fixed ``T`` is class **C** for variance (``P`` changes)
and class **B** for scenario risks (``A`` coefficients change). Expanding
windows and CPCV with changing ``T`` are class **E**.

Default persist mode is ``persist_full`` for variance and
``persist_factor`` for scenario risks. Full ADMM-state reuse *increased*
iteration count on class-B problems in the measured panel.

See the formulation table in ``docs/cosmo_meanrisk_formulations.md`` and
the investigation report in ``docs/cosmo_persistence_report.md``.

What COSMO cannot do
********************

* SDP / generic covariance uncertainty sets
* ND power cones
* Beating OSQP on boxed variance or HiGHS on boxed LPs (measured; do not
  assume otherwise)
* Parallel ``n_jobs != 1`` with a shared workspace — amortized backends
  already require serial execution

Reproduce the experiment
************************

::

    python benchmark/run_cosmo.py --quick
    python benchmark/run_cosmo.py

Outputs: ``benchmark/results/cosmo/<date>_<sha>/``. This is **not** the
canonical native-vs-auto suite (see ``AGENTS.md``).

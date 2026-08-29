"""MeanRisk cone classes and walk-forward persistability.

Inspected from skfolio 1.0 ``ConvexOptimization`` risk methods and from the
compact engines in :mod:`skfolio_accelerate.compact`. This table is the
source of truth for the COSMO experiment: it records the *generated*
canonical class, not the documentation name of the risk measure.

Persist classes (walk-forward with fixed training length ``T``):

* **A** — only ``q`` / ``b`` change
* **B** — ``q`` / ``b`` and numerical coefficients in ``A`` change
* **C** — ``P`` changes; sparsity pattern of ``P`` and ``A`` stay constant
* **D** — ``A`` and/or ``P`` sparsity pattern changes
* **E** — cone structure or dimensions change
* **F** — the complete canonical problem changes

Expanding windows and CombinatorialPurgedCV training sets with changing
``T`` are class **E** even when a rolling window of the same estimator is
class **B** or **C**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

ConeClass = Literal["LP", "QP", "SOCP", "ExpCone", "SDP", "mixed"]
PersistClass = Literal["A", "B", "C", "D", "E", "F"]
CosmoSupport = Literal["yes", "compact", "sequential", "no"]


@dataclass(frozen=True, slots=True)
class FormulationRecord:
    """One MeanRisk risk measure in canonical form.

    Attributes
    ----------
    risk : str
        ``RiskMeasure`` name.

    cone_class : {"LP", "QP", "SOCP", "ExpCone", "SDP", "mixed"}
        Dominant cone class of skfolio's CVXPY graph (variance is SOC in
        skfolio and a QP in the compact OSQP engine).

    compact_engine : str
        Engine ``backend="auto"`` would use for boxed minimize-risk /
        maximize-utility, or ``sequential`` / ``none``.

    persist_class_fixed_t : PersistClass
        Walk-forward class when ``(n_assets, T)`` are constant.

    persist_class_expanding : PersistClass
        Expanding-window class (``T`` grows).

    n_vars : str
        Decision-variable count as a function of assets ``n`` and scenarios
        ``T``.

    n_eq, n_ineq, cone_blocks : str
        Constraint / cone layout for the boxed compact problem, or the
        sequential CVXPY graph when there is no compact engine.

    p_structure, q_structure, a_structure, b_structure : str
        Which canonical blocks are structural vs numeric.

    fold_constant, fold_numeric : str
        What stays structurally constant vs what is rebound each fold.

    sparsity_constant : bool
        ``True`` when the CSC pattern of ``P`` and ``A`` is invariant for
        fixed ``(n, T)``.

    cosmo_support : CosmoSupport
        ``compact`` if the compact COSMO engine can represent it; ``sequential``
        if only a CVXPY ``COSMO_RUST`` path could; ``no`` if COSMO.rs cannot
        represent the cones (SDP).

    notes : str
        Extra caveats.
    """

    risk: str
    cone_class: ConeClass
    compact_engine: str
    persist_class_fixed_t: PersistClass
    persist_class_expanding: PersistClass
    n_vars: str
    n_eq: str
    n_ineq: str
    cone_blocks: str
    p_structure: str
    q_structure: str
    a_structure: str
    b_structure: str
    fold_constant: str
    fold_numeric: str
    sparsity_constant: bool
    cosmo_support: CosmoSupport
    notes: str


_RECORDS: tuple[FormulationRecord, ...] = (
    FormulationRecord(
        risk="VARIANCE",
        cone_class="QP",
        compact_engine="osqp",
        persist_class_fixed_t="C",
        persist_class_expanding="C",
        n_vars="n",
        n_eq="1 (budget)",
        n_ineq="2n (box)",
        cone_blocks="Zero(1) + Nonneg(2n)",
        p_structure="dense SPD 2Σ + 2 ℓ₂ I (upper CSC, nnz fixed)",
        q_structure="0 or −μ (utility)",
        a_structure="budget row + ±I; constant",
        b_structure="budget and box bounds; constant",
        fold_constant="n, m, cones, A sparsity, P sparsity, variable order",
        fold_numeric="P (covariance); q if MAXIMIZE_UTILITY",
        sparsity_constant=True,
        cosmo_support="compact",
        notes=(
            "skfolio implements variance as SOC(√Σ w) then squares the epigraph "
            "variable. Compact OSQP uses the equivalent QP. COSMO update_p "
            "numerically refactors; update_a is not needed. Expanding windows "
            "do not change n, so this stays class C."
        ),
    ),
    FormulationRecord(
        risk="ANNUALIZED_VARIANCE",
        cone_class="QP",
        compact_engine="none",
        persist_class_fixed_t="C",
        persist_class_expanding="C",
        n_vars="n",
        n_eq="1",
        n_ineq="2n",
        cone_blocks="same as VARIANCE with an annualization scale",
        p_structure="scaled variance QP / SOC",
        q_structure="0 or −μ",
        a_structure="budget + bounds",
        b_structure="constant",
        fold_constant="same as VARIANCE",
        fold_numeric="P",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Annualized alias; compact engines refuse it. Sequential CVXPY.",
    ),
    FormulationRecord(
        risk="STANDARD_DEVIATION",
        cone_class="SOCP",
        compact_engine="sequential",
        persist_class_fixed_t="C",
        persist_class_expanding="C",
        n_vars="n + 1 (radius)",
        n_eq="1",
        n_ineq="2n",
        cone_blocks="Zero + Nonneg + SOC(n+1) of covariance square-root",
        p_structure="ℓ₂ diagonal only",
        q_structure="e_radius or −μ on weights",
        a_structure="SOC rows hold √Σ (numeric, dense pattern)",
        b_structure="bounds; SOC rhs 0",
        fold_constant="n, cone dims, sparsity of SOC rows if √Σ is dense",
        fold_numeric="A SOC block (√Σ); q if utility",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Not in the compact subset. Parameterized CVXPY reuses √Σ.",
    ),
    FormulationRecord(
        risk="ANNUALIZED_STANDARD_DEVIATION",
        cone_class="SOCP",
        compact_engine="none",
        persist_class_fixed_t="C",
        persist_class_expanding="C",
        n_vars="n + 1",
        n_eq="1",
        n_ineq="2n",
        cone_blocks="same as STANDARD_DEVIATION",
        p_structure="ℓ₂ diagonal",
        q_structure="scaled radius",
        a_structure="√Σ",
        b_structure="constant",
        fold_constant="same as STANDARD_DEVIATION",
        fold_numeric="A SOC block",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Annualized alias.",
    ),
    FormulationRecord(
        risk="SEMI_VARIANCE",
        cone_class="QP",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="Zero(1) + Nonneg(2n+2T)",
        p_structure="diag(2ℓ₂ on w, 2λ/(T-1) on u)",
        q_structure="0 or −μ",
        a_structure="−(R−MAR) on u-rows; pattern dense in assets × T",
        b_structure="bounds; u≥0 rhs 0",
        fold_constant="n, T, cones, P, A sparsity",
        fold_numeric="A scenario coefficients; q if utility",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="COSMO update_a currently rebuilds the KKT system.",
    ),
    FormulationRecord(
        risk="ANNUALIZED_SEMI_VARIANCE",
        cone_class="QP",
        compact_engine="none",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="same as SEMI_VARIANCE",
        p_structure="scaled semi-variance diagonal",
        q_structure="0 or −μ",
        a_structure="R−MAR",
        b_structure="bounds",
        fold_constant="fixed T",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Annualized alias.",
    ),
    FormulationRecord(
        risk="SEMI_DEVIATION",
        cone_class="SOCP",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T + 1",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="Zero(1) + Nonneg(2n+2T) + SOC(T+1)",
        p_structure="ℓ₂ diagonal on w",
        q_structure="λ/√(T-1) on radius; −μ if utility",
        a_structure="R−MAR in LP rows; −I in SOC",
        b_structure="bounds",
        fold_constant="n, T, cone dims, sparsity",
        fold_numeric="A LP block (deviations); q if utility",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Primary COSMO-vs-Clarabel candidate: SOC projection is cheap.",
    ),
    FormulationRecord(
        risk="ANNUALIZED_SEMI_DEVIATION",
        cone_class="SOCP",
        compact_engine="none",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T + 1",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="same as SEMI_DEVIATION",
        p_structure="ℓ₂ diagonal",
        q_structure="scaled radius",
        a_structure="R−MAR",
        b_structure="bounds",
        fold_constant="fixed T",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Annualized alias.",
    ),
    FormulationRecord(
        risk="MEAN_ABSOLUTE_DEVIATION",
        cone_class="LP",
        compact_engine="highs",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="Zero(1) + Nonneg(2n+2T)",
        p_structure="0 (or 2ℓ₂ I if l2_coef>0 → QP, Clarabel)",
        q_structure="2λ/T on u; −μ if utility",
        a_structure="−(R−μ) on residual rows",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A; q if utility",
        sparsity_constant=True,
        cosmo_support="compact",
        notes=(
            "HiGHS simplex continuation already wins on rolling LPs. COSMO ADMM "
            "is not expected to beat it. CPCV MAD uses native skfolio."
        ),
    ),
    FormulationRecord(
        risk="FIRST_LOWER_PARTIAL_MOMENT",
        cone_class="LP",
        compact_engine="highs",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="Zero(1) + Nonneg(2n+2T)",
        p_structure="0 unless ℓ₂",
        q_structure="λ/T on u",
        a_structure="R−MAR",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Same persist story as MAD.",
    ),
    FormulationRecord(
        risk="WORST_REALIZATION",
        cone_class="LP",
        compact_engine="highs",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + 1",
        n_eq="1",
        n_ineq="2n + T",
        cone_blocks="Zero(1) + Nonneg(2n+T)",
        p_structure="0 unless ℓ₂",
        q_structure="λ on t",
        a_structure="−R on epigraph rows",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="HiGHS is the auto engine.",
    ),
    FormulationRecord(
        risk="CVAR",
        cone_class="LP",
        compact_engine="highs",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + 1 + T",
        n_eq="1",
        n_ineq="2n + 2T",
        cone_blocks="Zero(1) + Nonneg(2n+2T)",
        p_structure="0 unless ℓ₂",
        q_structure="λ on α and λ/(T(1-β)) on u",
        a_structure="−R on CVaR rows; pattern reserved with explicit zeros",
        b_structure="bounds; u and CVaR rhs 0",
        fold_constant="n, T, cones, P, q (min-risk), b, A sparsity",
        fold_numeric="A (returns); q if utility",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Rockafellar-Uryasev LP. HiGHS auto; COSMO is experimental.",
    ),
    FormulationRecord(
        risk="EVAR",
        cone_class="ExpCone",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + 2 + T",
        n_eq="1",
        n_ineq="2n + 1",
        cone_blocks="Zero(1) + Nonneg(2n+1) + T × ExpCone",
        p_structure="ℓ₂ diagonal on w",
        q_structure="λ on x, λ log(1/(T(1-β))) on y",
        a_structure="R in exp-cone x-slice",
        b_structure="bounds; exp rhs 0",
        fold_constant="n, T, cone product, sparsity",
        fold_numeric="A exp-cone rows; q if utility",
        sparsity_constant=True,
        cosmo_support="compact",
        notes=(
            "Exp-cone projections are relatively expensive per ADMM iteration. "
            "Native Clarabel sometimes fails EVaR on long windows."
        ),
    ),
    FormulationRecord(
        risk="MAX_DRAWDOWN",
        cone_class="LP",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + (T+1) + 1",
        n_eq="2 (budget + v0=0)",
        n_ineq="2n + 3T",
        cone_blocks="Zero(2) + Nonneg(2n+3T)",
        p_structure="ℓ₂ on w",
        q_structure="λ on epigraph",
        a_structure="drawdown recurrence uses R",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A (returns in v_t ≥ v_{t-1} − r_t w)",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Non-compounded drawdown recurrence from skfolio.",
    ),
    FormulationRecord(
        risk="AVERAGE_DRAWDOWN",
        cone_class="LP",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T + 1",
        n_eq="2",
        n_ineq="2n + 2T",
        cone_blocks="Zero(2) + Nonneg(2n+2T)",
        p_structure="ℓ₂ on w",
        q_structure="λ/T on drawdown states",
        a_structure="R in recurrence",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Same recurrence as max drawdown without an epigraph.",
    ),
    FormulationRecord(
        risk="CDAR",
        cone_class="LP",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + (T+1) + 1 + T",
        n_eq="2",
        n_ineq="2n + 4T",
        cone_blocks="Zero(2) + Nonneg(2n+4T)",
        p_structure="ℓ₂ on w",
        q_structure="λ on α and λ/(T(1-β)) on z",
        a_structure="R in drawdown; CVaR of drawdowns",
        b_structure="bounds",
        fold_constant="n, T, cones, sparsity",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="CVaR of the drawdown path.",
    ),
    FormulationRecord(
        risk="EDAR",
        cone_class="ExpCone",
        compact_engine="clarabel",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + (T+1) + 2 + T",
        n_eq="2",
        n_ineq="2n + 2T + 1",
        cone_blocks="Zero(2) + Nonneg(...) + T × ExpCone",
        p_structure="ℓ₂ on w",
        q_structure="EVaR-style (x, y)",
        a_structure="R in drawdown; exp cones on drawdown − x",
        b_structure="bounds",
        fold_constant="n, T, cone product",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="compact",
        notes="Native Clarabel often raises SolverError on EDaR.",
    ),
    FormulationRecord(
        risk="ULCER_INDEX",
        cone_class="SOCP",
        compact_engine="sequential",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + T + 1",
        n_eq="2",
        n_ineq="2n + 2T",
        cone_blocks="Zero + Nonneg + SOC(T+1) on drawdowns",
        p_structure="ℓ₂ on w",
        q_structure="λ/√T on radius",
        a_structure="R in drawdown; −I in SOC",
        b_structure="bounds",
        fold_constant="fixed T",
        fold_numeric="A",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="‖v[1:]‖₂ / √T. Sequential CVXPY; not compacted.",
    ),
    FormulationRecord(
        risk="GINI_MEAN_DIFFERENCE",
        cone_class="LP",
        compact_engine="sequential",
        persist_class_fixed_t="B",
        persist_class_expanding="E",
        n_vars="n + 3T",
        n_eq="1 + T (z = R w − costs)",
        n_ineq="2n + T² (OWA)",
        cone_blocks="Zero + Nonneg; OWA is a dense LP",
        p_structure="ℓ₂ on w",
        q_structure="2 on x and y",
        a_structure="R in z-equalities; OWA matrix constant",
        b_structure="bounds; OWA rhs 0",
        fold_constant="fixed T (OWA weights depend only on T)",
        fold_numeric="A equalities from returns",
        sparsity_constant=True,
        cosmo_support="sequential",
        notes="Cajas OWA LP, O(T²) inequalities. Too large for compact engines.",
    ),
)


_OBJECTIVE_NOTES: dict[str, str] = {
    "MINIMIZE_RISK": "Linear objective on the risk epigraph / quadratic term.",
    "MAXIMIZE_UTILITY": (
        "Adds −μᵀw; q on the weight block changes each fold (still B/C)."
    ),
    "MAXIMIZE_RETURN": (
        "Risk is dropped from the objective; risk constraints may remain. "
        "Boxed compact engines do not implement this; sequential CVXPY does."
    ),
    "MAXIMIZE_RATIO": (
        "Charnes-Cooper / Schaible homogenization adds a free factor variable "
        "and a return equality. Sequential path refuses it; fit-assemble / native."
    ),
}


_CONSTRAINT_NOTES: tuple[tuple[str, PersistClass, CosmoSupport, str], ...] = (
    (
        "weight box + budget",
        "A",
        "compact",
        "Structure constant; values constant in compact engines.",
    ),
    ("l2_coef", "C", "compact", "Adds a constant diagonal to P."),
    ("l1_coef", "B", "sequential", "Auxiliary abs variables; not compacted."),
    ("min_return", "A", "sequential", "Linear μᵀw ≥ r; μ in A or as a Parameter."),
    (
        "linear_constraints / groups",
        "B",
        "sequential",
        "Extra A rows; names need DataFrame columns.",
    ),
    (
        "transaction_costs / management_fees",
        "B",
        "sequential",
        "Affine terms in return / risk.",
    ),
    (
        "max_turnover / previous_weights",
        "F",
        "no",
        "Sequential previous_weights; compact and sequential refuse.",
    ),
    (
        "max_tracking_error",
        "C",
        "sequential",
        "SOC of (Rw − y); sequential refuses Parameterization.",
    ),
    ("mu uncertainty set", "C", "sequential", "Extra SOC; sequential refuses."),
    (
        "covariance uncertainty set (generic)",
        "F",
        "no",
        "Lifted SDP (S₊). COSMO.rs has no PSD cone.",
    ),
    (
        "covariance uncertainty set (compact)",
        "C",
        "sequential",
        "Extra quadratic residual; still sequential.",
    ),
    (
        "add_constraints / add_objective / custom",
        "F",
        "sequential",
        "Unknown cone class; fit-assemble.",
    ),
    ("efficient_frontier_size", "F", "no", "Multiple solves; assemble refuses."),
)


def formulation_table() -> tuple[FormulationRecord, ...]:
    """Return the frozen MeanRisk formulation records."""
    return _RECORDS


def formulation_record(risk: str | RiskMeasure) -> FormulationRecord:
    """Look up one risk measure.

    Parameters
    ----------
    risk : str or RiskMeasure
        Risk measure name or enum.

    Returns
    -------
    record : FormulationRecord
        Matching row.

    Raises
    ------
    KeyError
        If ``risk`` is not in the table.
    """
    name = risk.name if isinstance(risk, RiskMeasure) else str(risk)
    for row in _RECORDS:
        if row.risk == name:
            return row
    raise KeyError(name)


def persist_class_for(
    risk: str | RiskMeasure,
    *,
    expanding: bool = False,
    n_jobs: int | None = 1,
) -> PersistClass:
    """Walk-forward persist class, or ``F`` when folds are independent.

    Parameters
    ----------
    risk : str or RiskMeasure
        Risk measure.

    expanding : bool, default=False
        If ``True``, use the expanding-window class.

    n_jobs : int or None, default=1
        Parallel native folds cannot share a persistent workspace.

    Returns
    -------
    persist_class : PersistClass
        ``F`` when ``n_jobs`` is not ``None`` or ``1``.
    """
    if n_jobs not in (None, 1):
        return "F"
    row = formulation_record(risk)
    return row.persist_class_expanding if expanding else row.persist_class_fixed_t


def objective_note(objective: str | ObjectiveFunction) -> str:
    """Short note on how an objective changes the canonical problem."""
    name = (
        objective.name if isinstance(objective, ObjectiveFunction) else str(objective)
    )
    return _OBJECTIVE_NOTES[name]


def constraint_notes() -> tuple[tuple[str, PersistClass, CosmoSupport, str], ...]:
    """Constraint families and whether COSMO can keep them persistent."""
    return _CONSTRAINT_NOTES


def to_markdown() -> str:
    """Render the formulation table as GitHub-flavoured markdown."""
    lines = [
        "| Risk | Cone | Auto engine | Fixed-T class | Expanding class | "
        "Variables | COSMO | Sparsity constant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _RECORDS:
        lines.append(
            f"| {row.risk} | {row.cone_class} | {row.compact_engine} | "
            f"{row.persist_class_fixed_t} | {row.persist_class_expanding} | "
            f"{row.n_vars} | {row.cosmo_support} | "
            f"{'yes' if row.sparsity_constant else 'no'} |"
        )
    lines += ["", "### Objectives", ""]
    for name, note in _OBJECTIVE_NOTES.items():
        lines.append(f"* **{name}** — {note}")
    lines += ["", "### Constraints", ""]
    for name, klass, support, note in _CONSTRAINT_NOTES:
        lines.append(f"* **{name}** — class {klass}, COSMO `{support}`. {note}")
    return "\n".join(lines) + "\n"

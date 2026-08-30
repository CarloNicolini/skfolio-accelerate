"""MeanRisk configurations used by the native ``cross_val_predict`` benchmarks.

Authoritative set
-----------------
``mean_risk_specs`` times native
``skfolio.model_selection.cross_val_predict`` against ``backend="auto"`` for:

* every :class:`~skfolio.optimization.ObjectiveFunction` × every
  **non-annualized** :class:`~skfolio.RiskMeasure` (Gini omitted by default);
* extras: variance ``min_return``, named ``linear_constraints``,
  ``management_fees``, ``l1_coef``, and CVaR ``min_return``.
* boxed LPs with ``l2_coef=0`` (MAD, FLPM, CVaR, worst realization) so HiGHS
  is eligible (``include_lp_l2_zero``).

Annualized aliases are optional via ``--include-annualized``.

This suite includes the sequential grid + extras by default, the LP ``l2=0``
rows, and optional annualized / Gini flags. It does
not subsample “representative” risks from that grid.

Ambiguity
---------
Annualized measures are opt-in (``--include-annualized``).
Gini is a ~20-minute LP per fold on year-long windows; enable with
``--include-gini``. Linear-constraint extras use the first asset column name
at runtime so they are valid on S&P 500 tickers as well as synthetic ``A0``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

from benchmark.config import BenchmarkConfig

# Same skip as the default MeanRisk grid (Gini is opt-in).
SKIP_GINI = {RiskMeasure.GINI_MEAN_DIFFERENCE}

LP_L2_ZERO_RISKS = (
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.CVAR,
    RiskMeasure.WORST_REALIZATION,
)


@dataclass(frozen=True)
class EstimatorSpec:
    """One MeanRisk configuration, constructed only when the cell runs."""

    name: str
    objective: str
    risk: str
    extra: str
    kwargs: dict[str, Any]
    factory: Callable[..., MeanRisk]


def _mean_risk_factory(kwargs: dict[str, Any]) -> Callable[..., MeanRisk]:
    def factory(*, asset_names: list[str] | None = None) -> MeanRisk:
        params = dict(kwargs)
        constraints = params.get("linear_constraints")
        if constraints and asset_names:
            first = asset_names[0]
            params["linear_constraints"] = [
                item.replace("A0", first) if isinstance(item, str) else item
                for item in constraints
            ]
        return MeanRisk(**params)

    return factory


def mean_risk_specs(config: BenchmarkConfig) -> list[EstimatorSpec]:
    """Return the full MeanRisk grid selected by ``config`` flags."""
    risks = [
        risk
        for risk in RiskMeasure
        if (config.include_annualized or not risk.is_annualized)
        and (config.include_gini or risk not in SKIP_GINI)
    ]
    specs: list[EstimatorSpec] = []
    l2 = config.l2_coef
    solver_kwargs: dict[str, Any] = {"solver": config.solver}
    if config.solver_params is not None:
        solver_kwargs["solver_params"] = dict(config.solver_params)

    for objective in ObjectiveFunction:
        for risk in risks:
            kwargs = {
                "objective_function": objective,
                "risk_measure": risk,
                "l2_coef": l2,
                **solver_kwargs,
            }
            name = f"{objective.name}/{risk.name}"
            specs.append(
                EstimatorSpec(
                    name=name,
                    objective=objective.name,
                    risk=risk.name,
                    extra="",
                    kwargs=kwargs,
                    factory=_mean_risk_factory(kwargs),
                )
            )

    if config.include_extras:
        extras: list[tuple[str, str, str, dict[str, Any]]] = [
            (
                "MINIMIZE_RISK/VARIANCE+min_return",
                "MINIMIZE_RISK",
                "VARIANCE",
                {"min_return": 1e-5, "l2_coef": l2, **solver_kwargs},
            ),
            (
                "MINIMIZE_RISK/VARIANCE+linear_constraints",
                "MINIMIZE_RISK",
                "VARIANCE",
                {
                    "linear_constraints": ["A0 <= 0.45"],
                    "l2_coef": l2,
                    **solver_kwargs,
                },
            ),
            (
                "MINIMIZE_RISK/VARIANCE+management_fees",
                "MINIMIZE_RISK",
                "VARIANCE",
                {"management_fees": 1e-4, "l2_coef": l2, **solver_kwargs},
            ),
            (
                "MINIMIZE_RISK/VARIANCE+l1_coef",
                "MINIMIZE_RISK",
                "VARIANCE",
                {"l1_coef": 1e-3, "l2_coef": l2, **solver_kwargs},
            ),
            (
                "MINIMIZE_RISK/CVAR+min_return",
                "MINIMIZE_RISK",
                "CVAR",
                {
                    "risk_measure": RiskMeasure.CVAR,
                    "min_return": 1e-6,
                    "l2_coef": l2,
                    **solver_kwargs,
                },
            ),
        ]
        for name, objective, risk, kwargs in extras:
            extra = name.split("+", 1)[1]
            specs.append(
                EstimatorSpec(
                    name=name,
                    objective=objective,
                    risk=risk,
                    extra=extra,
                    kwargs=kwargs,
                    factory=_mean_risk_factory(kwargs),
                )
            )

    if config.include_lp_l2_zero:
        for risk in LP_L2_ZERO_RISKS:
            kwargs = {
                "objective_function": ObjectiveFunction.MINIMIZE_RISK,
                "risk_measure": risk,
                "l2_coef": 0.0,
                **solver_kwargs,
            }
            name = f"MINIMIZE_RISK/{risk.name}+l2_0"
            specs.append(
                EstimatorSpec(
                    name=name,
                    objective="MINIMIZE_RISK",
                    risk=risk.name,
                    extra="l2_0",
                    kwargs=kwargs,
                    factory=_mean_risk_factory(kwargs),
                )
            )
    return specs

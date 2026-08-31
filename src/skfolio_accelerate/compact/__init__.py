"""Direct MeanRisk QP / LP / SOCP engines (OSQP, HiGHS, Clarabel)."""

from __future__ import annotations

from dataclasses import dataclass

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.compact.cones import (
    MaxReturnBox,
    MinVarianceOSQP,
    ScenarioClarabel,
    StandardDeviationClarabel,
)
from skfolio_accelerate.compact._util import (
    SCENARIO_RISKS,
    MeanRiskSpec,
    as_bounds,
    estimator_spec,
)

# Re-export the historical private name used by linear_lp.
_as_bounds = as_bounds


def make_compact_engine(spec: MeanRiskSpec, *, n_assets: int, n_observations: int | None):
    risk = spec.risk_measure
    if spec.objective is ObjectiveFunction.MAXIMIZE_RETURN:
        return MaxReturnBox(spec, n_assets)
    if risk is RiskMeasure.VARIANCE:
        return MinVarianceOSQP(spec, n_assets)
    if risk is RiskMeasure.STANDARD_DEVIATION:
        return StandardDeviationClarabel(spec, n_assets)
    if risk not in SCENARIO_RISKS:
        raise ValueError(f"Unsupported risk_measure {risk}")
    if n_observations is None:
        raise ValueError(f"{risk.name} engine requires n_observations")
    from skfolio_accelerate.linear_lp import LinearHighs, is_highs_lp_risk

    if is_highs_lp_risk(spec):
        return LinearHighs(spec, n_assets, n_observations)
    return ScenarioClarabel(spec, n_assets, n_observations)


@dataclass
class EngineCache:
    spec: MeanRiskSpec
    engine: object | None = None
    n_assets: int = -1
    n_observations: int | None = None

    def get(self, n_assets: int, n_observations: int | None):
        need_new = self.engine is None or n_assets != self.n_assets
        if self.spec.needs_returns():
            need_new = need_new or n_observations != self.n_observations
        if need_new:
            import skfolio_accelerate.compact as compact

            self.engine = compact.make_compact_engine(
                self.spec, n_assets=n_assets, n_observations=n_observations
            )
            self.n_assets = n_assets
            self.n_observations = n_observations
        return self.engine

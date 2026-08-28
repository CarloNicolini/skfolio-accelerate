"""Reuse skfolio MeanRisk CVXPY problems across sequential CV folds.

skfolio 1.0 ``MeanRisk._fit`` rebuilds a new ``cp.Problem`` on every call and
bakes ``mu``, scenario returns, and the covariance square root in as numpy
constants. This adapter keeps that construction (all constraints, all
``ObjectiveFunction`` / ``RiskMeasure`` combinations) but injects fold-varying
data as ``cp.Parameter`` objects.

When the cone topology is unchanged, a later ``fit`` updates Parameter values
and warm-starts the existing problem instead of reassembling the graph.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import skfolio.optimization.convex._mean_risk as _mean_risk_mod
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from skfolio import RiskMeasure
from skfolio._constants import _TRANSACTION_COSTS
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex._base import ConvexOptimization
from skfolio.prior import ReturnDistribution
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.tools import _call_estimator
from sklearn.base import clone

_HOMOG_SENTINEL = object()

_OBSERVATION_SHAPED_RISKS = frozenset(
    {
        RiskMeasure.SEMI_VARIANCE,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.CVAR,
        RiskMeasure.EVAR,
        RiskMeasure.MAX_DRAWDOWN,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.CDAR,
        RiskMeasure.EDAR,
        RiskMeasure.ULCER_INDEX,
        RiskMeasure.GINI_MEAN_DIFFERENCE,
    }
)


@dataclass(frozen=True, slots=True)
class ProblemTopology:
    """Cone / variable shape that must match for CVXPY problem reuse.

    ``n_observations`` is ``0`` when the compiled graph does not allocate
    per-period auxiliaries (pure variance / standard-deviation without
    scenario risk limits or tracking error).
    """

    n_assets: int
    n_observations: int
    sqrt_layout: tuple[tuple[int, int], ...]
    sqrt_has_diag: bool
    sample_weight: bool
    factor_model: bool
    tracking: bool


@dataclass
class CompiledProblem:
    """One compiled MeanRisk ``cp.Problem`` plus the Parameters that feed it."""

    topology: ProblemTopology
    problem: cp.Problem
    w: cp.Variable
    factor: Any
    expressions: dict[str, cp.Expression]
    parameters_values: list
    mu_param: cp.Parameter | None = None
    returns_param: cp.Parameter | None = None
    sqrt_params: list[cp.Parameter] = field(default_factory=list)
    sqrt_diag_param: cp.Parameter | None = None
    prev_w_param: cp.Parameter | None = None
    homog_param: cp.Parameter | None = None
    y_param: cp.Parameter | None = None
    mar_param: cp.Parameter | None = None


@dataclass
class _ParamState:
    compiled: dict[ProblemTopology, CompiledProblem] = field(default_factory=dict)
    mu_param: cp.Parameter | None = None
    returns_param: cp.Parameter | None = None
    sqrt_params: list[cp.Parameter] = field(default_factory=list)
    sqrt_diag_param: cp.Parameter | None = None
    prev_w_param: cp.Parameter | None = None
    homog_param: cp.Parameter | None = None
    y_param: cp.Parameter | None = None
    mar_param: cp.Parameter | None = None
    build_distribution: ReturnDistribution | None = None
    fit_y: Any = None
    active_topo: ProblemTopology | None = None
    last_problem: cp.Problem | None = None
    n_warm_starts: int = 0
    n_rebuilds: int = 0
    is_dpp: bool | None = None
    is_dcp: bool | None = None


def needs_observation_dimension(estimator) -> bool:
    """True when the MeanRisk graph allocates auxiliaries sized by ``T``."""
    risk = getattr(estimator, "risk_measure", RiskMeasure.VARIANCE)
    if risk in _OBSERVATION_SHAPED_RISKS:
        return True
    if getattr(estimator, "max_tracking_error", None) is not None:
        return True
    for measure in _OBSERVATION_SHAPED_RISKS:
        if getattr(estimator, f"max_{measure.value}", None) is not None:
            return True
    return False


def can_reuse_distribution(return_distribution: ReturnDistribution) -> bool:
    """False when fold data is baked into constraints we do not Parameterize."""
    if return_distribution.sample_weight is not None:
        return False
    if return_distribution.factor_model is not None:
        return False
    return True


def problem_topology(
    estimator,
    return_distribution: ReturnDistribution,
    y=None,
) -> ProblemTopology:
    """Shape key for a fitted investable ``ReturnDistribution``."""
    n_assets = int(return_distribution.returns.shape[1])
    n_obs = (
        int(return_distribution.returns.shape[0])
        if needs_observation_dimension(estimator)
        else 0
    )
    cov_sqrt = return_distribution.covariance_sqrt
    layout = tuple(
        tuple(int(s) for s in component.shape) for component in cov_sqrt.components
    )
    return ProblemTopology(
        n_assets=n_assets,
        n_observations=n_obs,
        sqrt_layout=layout,
        sqrt_has_diag=cov_sqrt.diagonal is not None,
        sample_weight=return_distribution.sample_weight is not None,
        factor_model=return_distribution.factor_model is not None,
        tracking=getattr(estimator, "max_tracking_error", None) is not None,
    )


def as_parametric(estimator: MeanRisk) -> ParametricMeanRisk:
    """Copy MeanRisk parameters onto a :class:`ParametricMeanRisk` adapter."""
    if isinstance(estimator, ParametricMeanRisk):
        return estimator
    adapted = ParametricMeanRisk()
    adapted.set_params(**estimator.get_params(deep=True))
    return adapted


class ParametricMeanRisk(MeanRisk):
    """MeanRisk adapter that Parameterizes fold data and reuses ``cp.Problem``.

    Construction still runs through ``MeanRisk._fit``, so every constraint and
    objective skfolio would add is present. Overrides wrap ``mu``, scenario
    returns, the covariance square root, previous weights, tracking-error
    benchmarks, and the ratio homogenization factor as CVXPY Parameters.
    """

    def _param_state(self) -> _ParamState:
        state = getattr(self, "_skacc_state", None)
        if state is None:
            state = _ParamState()
            self._skacc_state = state
        return state

    @property
    def n_warm_starts(self) -> int:
        """Successful Parameter updates that reused a compiled problem."""
        return int(self._param_state().n_warm_starts)

    @property
    def n_rebuilds(self) -> int:
        """Number of times a new CVXPY graph was compiled."""
        return int(self._param_state().n_rebuilds)

    @property
    def last_problem(self) -> cp.Problem | None:
        """Most recently solved ``cp.Problem``, compiled or reused."""
        return self._param_state().last_problem

    @property
    def is_dpp_(self) -> bool | None:
        """DPP flag of the last compiled problem, if CVXPY could answer."""
        return self._param_state().is_dpp

    @property
    def is_dcp_(self) -> bool | None:
        """DCP flag of the last compiled problem."""
        return self._param_state().is_dcp

    def _clear_active_params(self) -> None:
        state = self._param_state()
        state.mu_param = None
        state.returns_param = None
        state.sqrt_params = []
        state.sqrt_diag_param = None
        state.prev_w_param = None
        state.homog_param = None
        state.y_param = None
        state.mar_param = None
        state.build_distribution = None
        self._clear_models_cache()

    def _ensure_parameter(
        self,
        current: cp.Parameter | None,
        shape: tuple[int, ...] | int,
        *,
        name: str,
        nonneg: bool = False,
    ) -> cp.Parameter:
        expected = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        if current is not None and tuple(int(s) for s in current.shape) == expected:
            return current
        kwargs: dict[str, Any] = {"name": name}
        if nonneg:
            kwargs["nonneg"] = True
        return cp.Parameter(expected, **kwargs)

    def _cvx_expected_return(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        state = self._param_state()
        state.build_distribution = return_distribution
        if self.overwrite_expected_return is not None:
            return ConvexOptimization._cvx_expected_return(self, return_distribution, w)
        mu = np.ascontiguousarray(return_distribution.mu, dtype=float)
        state.mu_param = self._ensure_parameter(state.mu_param, mu.shape, name="mu")
        state.mu_param.value = mu
        return state.mu_param @ w

    def _cvx_returns(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        state = self._param_state()
        state.build_distribution = return_distribution
        returns = np.ascontiguousarray(return_distribution.returns, dtype=float)
        state.returns_param = self._ensure_parameter(
            state.returns_param, returns.shape, name="returns"
        )
        state.returns_param.value = returns
        return state.returns_param @ w

    def _cvx_min_acceptable_return(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return=None,
    ) -> cp.Expression:
        returns_expr = self._cvx_returns(return_distribution, w)
        if min_acceptable_return is None:
            return returns_expr - self._cvx_expected_return(return_distribution, w)
        if np.isscalar(min_acceptable_return):
            return returns_expr - float(min_acceptable_return) * cp.sum(w)
        mar = np.asarray(min_acceptable_return, dtype=float).reshape(-1)
        state = self._param_state()
        state.mar_param = self._ensure_parameter(state.mar_param, mar.shape, name="mar")
        state.mar_param.value = mar
        return returns_expr - state.mar_param @ w

    def _standard_deviation_risk(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ):
        state = self._param_state()
        state.build_distribution = return_distribution
        risk = cp.Variable(nonneg=True)
        scale = self._scale_constraints
        covariance_sqrt = return_distribution.covariance_sqrt
        terms = []
        params: list[cp.Parameter] = []
        for index, component in enumerate(covariance_sqrt.components):
            array = np.ascontiguousarray(component, dtype=float)
            existing = (
                state.sqrt_params[index] if index < len(state.sqrt_params) else None
            )
            param = self._ensure_parameter(
                existing, array.shape, name=f"cov_sqrt_{index}"
            )
            param.value = array
            params.append(param)
            terms.append(param.T @ w)
        state.sqrt_params = params
        if covariance_sqrt.diagonal is not None:
            diagonal = np.ascontiguousarray(covariance_sqrt.diagonal, dtype=float)
            state.sqrt_diag_param = self._ensure_parameter(
                state.sqrt_diag_param, diagonal.shape, name="cov_sqrt_diag"
            )
            state.sqrt_diag_param.value = diagonal
            terms.append(cp.multiply(state.sqrt_diag_param, w))
        else:
            state.sqrt_diag_param = None
        constraints = [cp.SOC(risk * scale, cp.hstack(terms) * scale)]
        return risk, constraints

    def _cvx_transaction_cost(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor,
    ) -> cp.Expression:
        n_assets = return_distribution.returns.shape[1]
        transaction_costs = self._clean_input(
            self.transaction_costs,
            n_assets=n_assets,
            fill_value=0,
            name=_TRANSACTION_COSTS,
        )
        if np.all(transaction_costs == 0):
            return cp.Constant(0)
        prev = self._previous_weights_param(n_assets)
        if np.isscalar(transaction_costs):
            return transaction_costs * cp.norm(prev * factor - w, 1)
        return cp.norm(
            cp.multiply(transaction_costs, (prev * factor - w)),
            1,
        )

    def _turnover(self, n_assets: int, w: cp.Variable, factor) -> cp.Expression:
        prev = self._previous_weights_param(n_assets)
        return cp.abs(w - prev * factor)

    def _previous_weights_param(self, n_assets: int) -> cp.Parameter:
        state = self._param_state()
        state.prev_w_param = self._ensure_parameter(
            state.prev_w_param, n_assets, name="previous_weights"
        )
        state.prev_w_param.value = np.ascontiguousarray(
            self._clean_previous_weights(n_assets=n_assets), dtype=float
        )
        return state.prev_w_param

    def _tracking_error(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        y: FloatArray,
        factor,
    ) -> cp.Expression:
        n_observations = return_distribution.returns.shape[0]
        ptf_returns = self._cvx_returns(return_distribution, w)
        benchmark = np.ascontiguousarray(y, dtype=float).reshape(-1)
        state = self._param_state()
        state.y_param = self._ensure_parameter(
            state.y_param, benchmark.shape, name="benchmark"
        )
        state.y_param.value = benchmark
        return cp.norm(ptf_returns - state.y_param * factor, "fro") / np.sqrt(
            n_observations - 1
        )

    @contextmanager
    def _homogenization_parameters(self):
        """Expose the Charnes–Cooper homogenization factor as a Parameter."""
        state = self._param_state()
        original_homog = _mean_risk_mod._optimal_homogenization_factor
        original_constant = _mean_risk_mod.cp.Constant

        def hooked_homog(mu):
            value = float(original_homog(mu))
            if state.homog_param is None:
                state.homog_param = cp.Parameter(nonneg=True, name="homog")
            state.homog_param.value = value
            return _HOMOG_SENTINEL

        def hooked_constant(val, *args, **kwargs):
            if val is _HOMOG_SENTINEL:
                return state.homog_param
            return original_constant(val, *args, **kwargs)

        _mean_risk_mod._optimal_homogenization_factor = hooked_homog
        _mean_risk_mod.cp.Constant = hooked_constant
        try:
            yield
        finally:
            _mean_risk_mod._optimal_homogenization_factor = original_homog
            _mean_risk_mod.cp.Constant = original_constant

    def _ensure_warm_start(self) -> None:
        params = dict(getattr(self, "_solver_params", None) or {})
        params.setdefault("warm_start", True)
        self._solver_params = params

    def _fit_prior_distribution(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        method: str,
        **fit_params,
    ) -> ReturnDistribution:
        """Fit the prior the same way ``MeanRisk._fit`` does, then stop."""
        routed_params = skm.process_routing(self, method, **fit_params)
        first_call = not hasattr(self, "weights_")
        _ = skv.validate_data(
            self,
            X,
            skip_check_array=True,
            reset=first_call,
            ensure_all_finite="allow-nan",
        )
        if first_call:
            self._validate_params(method=method)
            self._initialize()
        if method == "partial_fit":
            self._validate_partial_fit_fallback()
            self._validate_partial_fit_estimators()
        _call_estimator(
            self.prior_estimator_,
            method,
            X,
            y,
            routed_params=routed_params.prior_estimator,
        )
        return self._prepare_investable_distribution(
            self.prior_estimator_.return_distribution_, slim=True
        )

    def _bind_compiled(
        self,
        bundle: CompiledProblem,
        return_distribution: ReturnDistribution,
        y=None,
    ) -> None:
        mu = np.ascontiguousarray(return_distribution.mu, dtype=float)
        if bundle.mu_param is not None:
            bundle.mu_param.value = mu
        if bundle.returns_param is not None:
            bundle.returns_param.value = np.ascontiguousarray(
                return_distribution.returns, dtype=float
            )
        cov_sqrt = return_distribution.covariance_sqrt
        if bundle.sqrt_params:
            for param, component in zip(
                bundle.sqrt_params, cov_sqrt.components, strict=True
            ):
                param.value = np.ascontiguousarray(component, dtype=float)
        if bundle.sqrt_diag_param is not None:
            if cov_sqrt.diagonal is None:
                raise RuntimeError("compiled covariance square root lost its diagonal")
            bundle.sqrt_diag_param.value = np.ascontiguousarray(
                cov_sqrt.diagonal, dtype=float
            )
        n_assets = int(return_distribution.returns.shape[1])
        if bundle.prev_w_param is not None:
            bundle.prev_w_param.value = np.ascontiguousarray(
                self._clean_previous_weights(n_assets=n_assets), dtype=float
            )
        if bundle.homog_param is not None:
            bundle.homog_param.value = float(
                _mean_risk_mod._optimal_homogenization_factor(mu=mu)
            )
        if bundle.y_param is not None:
            if y is None:
                raise ValueError(
                    "If `max_tracking_error` is provided, `y` must also be provided"
                )
            benchmark = np.ascontiguousarray(y, dtype=float)
            if benchmark.ndim > 1:
                if benchmark.shape[1] == 1:
                    benchmark = benchmark[:, 0]
                else:
                    benchmark = benchmark.reshape(-1)
            bundle.y_param.value = benchmark.reshape(-1)
        if bundle.mar_param is not None and self.min_acceptable_return is not None:
            bundle.mar_param.value = np.asarray(
                self.min_acceptable_return, dtype=float
            ).reshape(-1)

    def _snapshot(
        self,
        topology: ProblemTopology,
        problem: cp.Problem,
        w: cp.Variable,
        factor,
        parameters_values,
        expressions,
    ) -> CompiledProblem:
        state = self._param_state()
        return CompiledProblem(
            topology=topology,
            problem=problem,
            w=w,
            factor=factor,
            expressions=expressions or {},
            parameters_values=parameters_values or [],
            mu_param=state.mu_param,
            returns_param=state.returns_param,
            sqrt_params=list(state.sqrt_params),
            sqrt_diag_param=state.sqrt_diag_param,
            prev_w_param=state.prev_w_param,
            homog_param=state.homog_param,
            y_param=state.y_param,
            mar_param=state.mar_param,
        )

    def _record_problem_flags(self, problem: cp.Problem) -> None:
        state = self._param_state()
        try:
            state.is_dcp = bool(problem.is_dcp())
        except Exception:
            state.is_dcp = None
        try:
            state.is_dpp = bool(problem.is_dpp())
        except Exception:
            state.is_dpp = None
        state.last_problem = problem

    def _replay(
        self,
        bundle: CompiledProblem,
        return_distribution: ReturnDistribution,
        y=None,
    ) -> None:
        self._bind_compiled(bundle, return_distribution, y)
        self._ensure_warm_start()
        state = self._param_state()
        state.n_warm_starts += 1
        ConvexOptimization._solve_problem(
            self,
            problem=bundle.problem,
            w=bundle.w,
            factor=bundle.factor,
            parameters_values=bundle.parameters_values,
            expressions=bundle.expressions,
        )
        self._record_problem_flags(bundle.problem)

    def _solve_problem(
        self,
        problem: cp.Problem,
        w: cp.Variable,
        factor,
        parameters_values=None,
        expressions=None,
    ) -> None:
        self._ensure_warm_start()
        ConvexOptimization._solve_problem(
            self,
            problem=problem,
            w=w,
            factor=factor,
            parameters_values=parameters_values,
            expressions=expressions,
        )
        state = self._param_state()
        rd = state.build_distribution
        if rd is not None and can_reuse_distribution(rd):
            topology = state.active_topo or problem_topology(self, rd, state.fit_y)
            if topology not in state.compiled:
                state.compiled[topology] = self._snapshot(
                    topology,
                    problem,
                    w,
                    factor,
                    parameters_values,
                    expressions,
                )
                state.n_rebuilds += 1
            state.active_topo = topology
        self._record_problem_flags(problem)

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        method: str = "fit",
        **fit_params,
    ) -> ParametricMeanRisk:
        state = self._param_state()
        state.fit_y = y
        if state.compiled:
            return_distribution = self._fit_prior_distribution(
                X, y, method, **fit_params
            )
            topology = problem_topology(self, return_distribution, y)
            bundle = state.compiled.get(topology)
            if bundle is not None and can_reuse_distribution(return_distribution):
                self._replay(bundle, return_distribution, y)
                return self
            state.active_topo = topology
            self._clear_active_params()
            with self._homogenization_parameters():
                return super()._fit(X, y, method=method, **fit_params)
        self._clear_active_params()
        with self._homogenization_parameters():
            return super()._fit(X, y, method=method, **fit_params)


class SequentialProblemCache:
    """One :class:`ParametricMeanRisk` adapter per MRC / CV path batch."""

    def __init__(self, estimator: MeanRisk) -> None:
        self._template = as_parametric(estimator)
        self._adapters: dict[int, ParametricMeanRisk] = {}

    def get(self, path_id: int = 0) -> ParametricMeanRisk:
        adapter = self._adapters.get(path_id)
        if adapter is None:
            adapter = clone(self._template)
            self._adapters[path_id] = adapter
        return adapter

    @property
    def n_warm_starts(self) -> int:
        return sum(adapter.n_warm_starts for adapter in self._adapters.values())

    @property
    def n_rebuilds(self) -> int:
        return sum(adapter.n_rebuilds for adapter in self._adapters.values())

    @property
    def is_dpp(self) -> bool | None:
        flags = [adapter.is_dpp_ for adapter in self._adapters.values()]
        flags = [flag for flag in flags if flag is not None]
        if not flags:
            return None
        return all(flags)

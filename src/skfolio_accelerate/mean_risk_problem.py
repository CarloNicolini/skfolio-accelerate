"""Reuse one MeanRisk CVXPY problem; Parameters are updated per fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex._base import ConvexOptimization
from skfolio.prior import ReturnDistribution
from skfolio.utils.tools import _call_estimator
from sklearn.base import clone

from skfolio_accelerate.moments import FoldMoments


def as_parametric(estimator: MeanRisk) -> ParametricMeanRisk:
    if isinstance(estimator, ParametricMeanRisk):
        return estimator
    adapted = ParametricMeanRisk()
    adapted.set_params(**estimator.get_params(deep=True))
    return adapted


@dataclass
class _State:
    problem: cp.Problem | None = None
    w: cp.Variable | None = None
    factor: Any = None
    parameters_values: list | None = None
    expressions: dict | None = None
    mu: cp.Parameter | None = None
    returns: cp.Parameter | None = None
    sqrt: list[cp.Parameter] = field(default_factory=list)
    sqrt_diag: cp.Parameter | None = None
    n_warm_starts: int = 0
    n_rebuilds: int = 0
    is_dpp: bool | None = None


class ParametricMeanRisk(MeanRisk):
    """MeanRisk that Parameterizes fold data and reuses ``cp.Problem``."""

    def _state(self) -> _State:
        state = getattr(self, "_skacc_state", None)
        if state is None:
            state = _State()
            self._skacc_state = state
        return state

    @property
    def n_warm_starts(self) -> int:
        return int(self._state().n_warm_starts)

    @property
    def n_rebuilds(self) -> int:
        return int(self._state().n_rebuilds)

    @property
    def last_problem(self) -> cp.Problem | None:
        return self._state().problem

    @property
    def is_dpp_(self) -> bool | None:
        return self._state().is_dpp

    def _parameter(self, current, shape, name: str) -> cp.Parameter:
        expected = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        if current is not None and tuple(int(s) for s in current.shape) == expected:
            return current
        return cp.Parameter(expected, name=name)

    def _cvx_expected_return(self, return_distribution, w):
        mu = np.ascontiguousarray(return_distribution.mu, dtype=float)
        state = self._state()
        state.mu = self._parameter(state.mu, mu.shape, "mu")
        state.mu.value = mu
        return state.mu @ w

    def _cvx_returns(self, return_distribution, w):
        returns = np.ascontiguousarray(return_distribution.returns, dtype=float)
        state = self._state()
        state.returns = self._parameter(state.returns, returns.shape, "returns")
        state.returns.value = returns
        return state.returns @ w

    def _cvx_min_acceptable_return(
        self, return_distribution, w, min_acceptable_return=None
    ):
        ptf_returns = self._cvx_returns(return_distribution, w)
        if min_acceptable_return is None:
            return ptf_returns - self._cvx_expected_return(return_distribution, w)
        if np.isscalar(min_acceptable_return):
            return ptf_returns - float(min_acceptable_return) * cp.sum(w)
        mar = np.ascontiguousarray(min_acceptable_return, dtype=float).reshape(-1)
        return ptf_returns - mar @ w

    def _standard_deviation_risk(self, return_distribution, w):
        state = self._state()
        risk = cp.Variable(nonneg=True)
        scale = self._scale_constraints
        cov_sqrt = return_distribution.covariance_sqrt
        terms, params = [], []
        for index, component in enumerate(cov_sqrt.components):
            array = np.ascontiguousarray(component, dtype=float)
            existing = state.sqrt[index] if index < len(state.sqrt) else None
            param = self._parameter(existing, array.shape, f"cov_sqrt_{index}")
            param.value = array
            params.append(param)
            terms.append(param.T @ w)
        state.sqrt = params
        if cov_sqrt.diagonal is not None:
            diagonal = np.ascontiguousarray(cov_sqrt.diagonal, dtype=float)
            state.sqrt_diag = self._parameter(
                state.sqrt_diag, diagonal.shape, "cov_diag"
            )
            state.sqrt_diag.value = diagonal
            terms.append(cp.multiply(state.sqrt_diag, w))
        else:
            state.sqrt_diag = None
        return risk, [cp.SOC(risk * scale, cp.hstack(terms) * scale)]

    def _shapes_match(self, return_distribution) -> bool:
        state = self._state()
        if state.problem is None:
            return False
        mu = np.asarray(return_distribution.mu)
        if state.mu is not None and tuple(state.mu.shape) != tuple(mu.shape):
            return False
        returns = np.asarray(return_distribution.returns)
        if state.returns is not None and tuple(state.returns.shape) != tuple(
            returns.shape
        ):
            return False
        if state.sqrt:
            components = return_distribution.covariance_sqrt.components
            if len(state.sqrt) != len(components):
                return False
            return all(
                tuple(param.shape) == tuple(np.asarray(component).shape)
                for param, component in zip(state.sqrt, components, strict=True)
            )
        return True

    def _bind(self, return_distribution) -> None:
        state = self._state()
        if state.mu is not None:
            state.mu.value = np.ascontiguousarray(return_distribution.mu, dtype=float)
        if state.returns is not None:
            state.returns.value = np.ascontiguousarray(
                return_distribution.returns, dtype=float
            )
        if not state.sqrt:
            return
        cov_sqrt = return_distribution.covariance_sqrt
        for param, component in zip(state.sqrt, cov_sqrt.components, strict=True):
            param.value = np.ascontiguousarray(component, dtype=float)
        if state.sqrt_diag is not None:
            if cov_sqrt.diagonal is None:
                raise RuntimeError("compiled covariance square-root lost its diagonal")
            state.sqrt_diag.value = np.ascontiguousarray(cov_sqrt.diagonal, dtype=float)

    def _solve_cached(self, state: _State) -> None:
        ConvexOptimization._solve_problem(
            self,
            problem=state.problem,
            w=state.w,
            factor=state.factor,
            parameters_values=state.parameters_values,
            expressions=state.expressions,
        )

    def fit_from_moments(self, moments: FoldMoments) -> bool:
        distribution = ReturnDistribution(
            mu=np.asarray(moments.mu, dtype=float),
            covariance=np.asarray(moments.covariance, dtype=float),
            returns=np.asarray(moments.returns, dtype=float),
        )
        state = self._state()
        if not self._shapes_match(distribution):
            return False
        self._reset()
        self._bind(distribution)
        state.n_warm_starts += 1
        self._solve_cached(state)
        return True

    def _fit_prior(self, X, y, method: str, **fit_params) -> ReturnDistribution:
        routed = skm.process_routing(self, method, **fit_params)
        first = not hasattr(self, "prior_estimator_")
        _ = skv.validate_data(
            self, X, skip_check_array=True, reset=first, ensure_all_finite="allow-nan"
        )
        if first:
            self._validate_params(method=method)
            self._initialize()
        _call_estimator(
            self.prior_estimator_, method, X, y, routed_params=routed.prior_estimator
        )
        return self._prepare_investable_distribution(
            self.prior_estimator_.return_distribution_, slim=True
        )

    def _solve_problem(
        self, problem, w, factor, parameters_values=None, expressions=None
    ) -> None:
        params = dict(getattr(self, "_solver_params", None) or {})
        params.setdefault("warm_start", True)
        self._solver_params = params
        ConvexOptimization._solve_problem(
            self,
            problem=problem,
            w=w,
            factor=factor,
            parameters_values=parameters_values,
            expressions=expressions,
        )
        state = self._state()
        if state.problem is not problem:
            state.n_rebuilds += 1
        state.problem, state.w, state.factor = problem, w, factor
        state.parameters_values, state.expressions = parameters_values, expressions
        try:
            state.is_dpp = bool(problem.is_dpp())
        except Exception:
            state.is_dpp = None

    def _fit(self, X, y=None, method: str = "fit", **fit_params):
        state = self._state()
        if state.problem is not None:
            distribution = self._fit_prior(X, y, method, **fit_params)
            loadings_block = (
                distribution.factor_model is not None
                and self.linear_constraints is not None
            )
            if (
                distribution.sample_weight is None
                and self._shapes_match(distribution)
                and not loadings_block
            ):
                self._bind(distribution)
                state.n_warm_starts += 1
                self._solve_cached(state)
                return self
            state.problem = None
            state.mu = state.returns = state.sqrt_diag = None
            state.sqrt = []
            self._clear_models_cache()
        return super()._fit(X, y, method=method, **fit_params)


class SequentialProblemCache:
    def __init__(self, estimator: MeanRisk) -> None:
        self._template = as_parametric(estimator)
        self._adapters: dict[int, ParametricMeanRisk] = {}

    def get(self, path_id: int = 0) -> ParametricMeanRisk:
        adapter = self._adapters.get(path_id)
        if adapter is None:
            adapter = clone(self._template)
            self._adapters[path_id] = adapter
        return adapter

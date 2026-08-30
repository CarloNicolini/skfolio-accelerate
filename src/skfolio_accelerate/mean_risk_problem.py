"""Reuse one MeanRisk CVXPY problem across WalkForward folds.

skfolio rebuilds a new ``cp.Problem`` on every ``fit`` and bakes fold data in
as numpy constants. This adapter still calls ``MeanRisk._fit`` so every
constraint and risk expression is skfolio's. It only Parameterizes the
fold-varying injections:

* expected return (``mu``)
* scenario returns
* covariance square-root (variance / standard deviation), including the
  low-rank factor form ``CovarianceSqrt(B @ chol(F), diag=sqrt(D))``
* default minimum acceptable return (``returns - mu``)

The prior (empirical or :class:`~skfolio.prior.TimeSeriesFactorModel`) is
still fitted on every fold. Later folds with the same shapes update Parameter
values and warm-start. Factor-named ``linear_constraints`` bake loadings as
constants, so those configurations rebuild. Shape changes rebuild. Ratio
homogenization, transaction costs, custom CVXPY hooks, and MeanRisk subclasses
are not handled here.
"""

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
from skfolio.typing import ArrayLike
from skfolio.utils.tools import _call_estimator
from sklearn.base import clone

from skfolio_accelerate.moments import FoldMoments


def as_parametric(estimator: MeanRisk) -> ParametricMeanRisk:
    """Copy MeanRisk parameters onto a :class:`ParametricMeanRisk`."""
    if isinstance(estimator, ParametricMeanRisk):
        return estimator
    adapted = ParametricMeanRisk()
    adapted.set_params(**estimator.get_params(deep=True))
    return adapted


@dataclass
class _State:
    """Compiled problem plus the Parameters that feed it.

    Stored on ``_skacc_state`` so ``fit()``'s ``_reset()`` (which deletes
    ``weights_``) does not drop the cache, and ``sklearn.clone`` does not copy it.
    """

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

    def _parameter(
        self, current: cp.Parameter | None, shape, name: str
    ) -> cp.Parameter:
        expected = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        if current is not None and tuple(int(s) for s in current.shape) == expected:
            return current
        return cp.Parameter(expected, name=name)

    def _cvx_expected_return(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        mu = np.ascontiguousarray(return_distribution.mu, dtype=float)
        state = self._state()
        state.mu = self._parameter(state.mu, mu.shape, "mu")
        state.mu.value = mu
        return state.mu @ w

    def _cvx_returns(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        returns = np.ascontiguousarray(return_distribution.returns, dtype=float)
        state = self._state()
        state.returns = self._parameter(state.returns, returns.shape, "returns")
        state.returns.value = returns
        return state.returns @ w

    def _cvx_min_acceptable_return(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return=None,
    ) -> cp.Expression:
        """Same as skfolio, but default MAR uses Parameterized ``mu`` / returns."""
        ptf_returns = self._cvx_returns(return_distribution, w)
        if min_acceptable_return is None:
            return ptf_returns - self._cvx_expected_return(return_distribution, w)
        if np.isscalar(min_acceptable_return):
            return ptf_returns - float(min_acceptable_return) * cp.sum(w)
        mar = np.ascontiguousarray(min_acceptable_return, dtype=float).reshape(-1)
        return ptf_returns - mar @ w

    def _standard_deviation_risk(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ):
        """Same SOC as skfolio, with Parameterized covariance square-root."""
        state = self._state()
        risk = cp.Variable(nonneg=True)
        scale = self._scale_constraints
        cov_sqrt = return_distribution.covariance_sqrt
        terms = []
        params: list[cp.Parameter] = []
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

    def _shapes_match(self, return_distribution: ReturnDistribution) -> bool:
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

    def _bind(self, return_distribution: ReturnDistribution) -> None:
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

    def fit_from_moments(self, moments: FoldMoments) -> bool:
        """Bind default empirical moments to an existing fixed-shape problem."""
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
        ConvexOptimization._solve_problem(
            self,
            problem=state.problem,
            w=state.w,
            factor=state.factor,
            parameters_values=state.parameters_values,
            expressions=state.expressions,
        )
        return True

    def _fit_prior(
        self, X: ArrayLike, y: ArrayLike | None, method: str, **fit_params
    ) -> ReturnDistribution:
        routed = skm.process_routing(self, method, **fit_params)
        first = not hasattr(self, "prior_estimator_")
        _ = skv.validate_data(
            self, X, skip_check_array=True, reset=first, ensure_all_finite="allow-nan"
        )
        if first:
            self._validate_params(method=method)
            self._initialize()
        _call_estimator(
            self.prior_estimator_,
            method,
            X,
            y,
            routed_params=routed.prior_estimator,
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
        state.problem = problem
        state.w = w
        state.factor = factor
        state.parameters_values = parameters_values
        state.expressions = expressions
        try:
            state.is_dpp = bool(problem.is_dpp())
        except Exception:
            state.is_dpp = None

    def _loading_constraints_block_warm_start(
        self, distribution: ReturnDistribution
    ) -> bool:
        """True when linear constraints may bake fold-varying loadings as constants.

        Factor-named exposure constraints inject ``loading_matrix`` into equality /
        inequality rows. Those rows are not Parameterized today, so a warm start
        would keep stale exposures. Asset-only constraints are also treated as
        unsafe whenever a factor model is present (skfolio still passes ``B`` into
        ``equations_to_matrix``).
        """
        return (
            distribution.factor_model is not None and self.linear_constraints is not None
        )

    def _fit(self, X, y=None, method: str = "fit", **fit_params):
        state = self._state()
        if state.problem is not None:
            distribution = self._fit_prior(X, y, method, **fit_params)
            if (
                distribution.sample_weight is None
                and self._shapes_match(distribution)
                and not self._loading_constraints_block_warm_start(distribution)
            ):
                self._bind(distribution)
                state.n_warm_starts += 1
                ConvexOptimization._solve_problem(
                    self,
                    problem=state.problem,
                    w=state.w,
                    factor=state.factor,
                    parameters_values=state.parameters_values,
                    expressions=state.expressions,
                )
                return self
            state.problem = None
            state.mu = state.returns = state.sqrt_diag = None
            state.sqrt = []
            self._clear_models_cache()
        return super()._fit(X, y, method=method, **fit_params)


class SequentialProblemCache:
    """One adapter per MRC path (different asset subsets)."""

    def __init__(self, estimator: MeanRisk) -> None:
        self._template = as_parametric(estimator)
        self._adapters: dict[int, ParametricMeanRisk] = {}

    def get(self, path_id: int = 0) -> ParametricMeanRisk:
        adapter = self._adapters.get(path_id)
        if adapter is None:
            adapter = clone(self._template)
            self._adapters[path_id] = adapter
        return adapter

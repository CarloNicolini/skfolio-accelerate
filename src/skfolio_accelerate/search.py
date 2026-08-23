"""MassiveGridSearchCV: DPP + solver-update search over skfolio estimators."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid

from skfolio.model_selection import BaseCombinatorialCV

from skfolio_accelerate.backends.python_clarabel import PythonClarabelEngine
from skfolio_accelerate.backends.rust_clarabel import (
    RustClarabelEngine,
    rust_is_available,
)
from skfolio_accelerate.backends.sklearn_fallback import (
    acceleration_blocked_reason,
    sklearn_grid_search,
)
from skfolio_accelerate.classify import classify_param_grid, data_fingerprint
from skfolio_accelerate.compile import extract_problem_template, instantiate
from skfolio_accelerate.cv_plan import compile_cv_plan, slice_rows
from skfolio_accelerate.estimators.mean_risk_twin import (
    bind_from_estimator,
    build_twin_from_estimator,
    structure_key,
)
from skfolio_accelerate.ir import Evaluation, SearchPlan, SolveResult
from skfolio_accelerate.moments import FoldCache
from skfolio_accelerate.profile import acceleration_report
from skfolio_accelerate.scoring import (
    path_portfolios,
    score_multi_period,
    score_with_estimator,
)


def rust_engine_available() -> bool:
    return rust_is_available()


def _cap_native_threads() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _solved(status: str) -> bool:
    text = str(status).lower()
    return "solved" in text or "optimal" in text


class MassiveGridSearchCV(BaseEstimator):
    """sklearn-compatible massive grid search with a compiled CV engine."""

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit: bool = True,
        cv=None,
        verbose: int = 0,
        pre_dispatch: str | int = "2*n_jobs",
        error_score=np.nan,
        return_train_score: bool = False,
        backend: str = "auto",
        solver_threads: int = 1,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.backend = backend
        self.solver_threads = solver_threads

    def fit(self, X, y=None, **fit_params):
        del fit_params
        _cap_native_threads()
        t0 = time.perf_counter()
        estimator = clone(self.estimator)
        cv = self.cv
        n_jobs = self.n_jobs if self.n_jobs not in (None, 0) else 1
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1

        classification = classify_param_grid(estimator, self.param_grid)
        blocked = acceleration_blocked_reason(
            estimator, self.param_grid, cv, self.scoring
        )
        requested = self.backend
        if requested not in {"auto", "python", "rust", "sklearn"}:
            raise ValueError(f"Unknown backend {requested!r}")

        combinatorial = isinstance(cv, BaseCombinatorialCV)
        if requested == "sklearn" or (requested == "auto" and blocked is not None):
            if combinatorial:
                raise TypeError(
                    "sklearn GridSearchCV cannot consume CombinatorialPurgedCV splits"
                    + (f": {blocked}" if blocked else ".")
                )
            return self._fit_sklearn(
                estimator, X, y, t0, blocked or "backend=sklearn"
            )

        if requested == "rust" and not rust_is_available():
            raise RuntimeError(
                "backend='rust' but the compiled extension is not installed"
            )
        if requested in {"python", "rust"} and blocked is not None:
            raise ValueError(
                f"backend={requested!r} cannot accelerate this search: {blocked}"
            )

        cv_plan = compile_cv_plan(cv, X, y)
        param_list = list(ParameterGrid(self.param_grid))
        use_rust = requested == "rust" or (
            requested == "auto" and rust_is_available()
        )
        backend_used = "rust" if use_rust else "python"

        self.search_plan_ = SearchPlan(
            cv_plan=cv_plan,
            param_grid_list=param_list,
            backend=backend_used,
            native_scoring=self.scoring is None,
            estimator_name=type(estimator).__name__,
            classification=classification,
            scoring=self.scoring,
            n_jobs=n_jobs,
        )

        cache = FoldCache()
        evaluations: list[Evaluation] = []
        n_templates = 0
        n_updates = 0
        compile_s = 0.0
        instantiate_s = 0.0
        solve_s = 0.0
        eval_s = 0.0
        engines: dict[str, Any] = {}
        templates: dict[str, Any] = {}
        twins: dict[str, Any] = {}
        template_ids: dict[str, int] = {}

        schedule = sorted(
            range(len(param_list)),
            key=lambda i: (
                str(param_list[i].get("risk_measure", "")),
                float(param_list[i].get("l2_coef", 0.0) or 0.0),
            ),
        )

        for fold in cv_plan.folds:
            X_train = slice_rows(X, fold.train_idx)
            y_train = None if y is None else slice_rows(y, fold.train_idx)
            batches: dict[str, list[tuple[int, Any]]] = {}

            for param_id in schedule:
                params = param_list[param_id]
                data_key = data_fingerprint(params)
                fold_estimator = clone(estimator)
                fold_estimator.set_params(
                    **{
                        key: value
                        for key, value in params.items()
                        if key == "prior_estimator"
                        or key.startswith("prior_estimator__")
                    }
                )
                moments = cache.get(
                    fold.fold_id,
                    fold_estimator,
                    X_train,
                    y_train,
                    data_key=data_key,
                )
                n_obs = int(moments.returns.shape[0])
                n_assets = int(moments.mu.size)

                t_c = time.perf_counter()
                key = structure_key(
                    estimator,
                    params,
                    n_observations=n_obs,
                    n_assets=n_assets,
                )
                if key not in twins:
                    twins[key] = build_twin_from_estimator(
                        estimator,
                        params,
                        n_observations=n_obs,
                        n_assets=n_assets,
                    )
                    templates[key] = extract_problem_template(twins[key], key)
                    if use_rust:
                        engines[key] = RustClarabelEngine(
                            templates[key],
                            n_jobs=n_jobs,
                            solver_threads=self.solver_threads,
                        )
                    else:
                        engines[key] = PythonClarabelEngine(
                            templates[key],
                            solver_threads=self.solver_threads,
                        )
                    template_ids[key] = n_templates
                    n_templates += 1
                compile_s += time.perf_counter() - t_c

                t_i = time.perf_counter()
                bind_from_estimator(twins[key], moments, estimator, params)
                instance = instantiate(templates[key])
                instantiate_s += time.perf_counter() - t_i
                batches.setdefault(key, []).append((param_id, instance))

            for key, items in batches.items():
                instances = [instance for _, instance in items]
                t_s = time.perf_counter()
                if use_rust:
                    results = engines[key].solve_many(instances)
                else:
                    results = [engines[key].solve(instance) for instance in instances]
                solve_s += time.perf_counter() - t_s
                n_updates += len(results)

                t_e = time.perf_counter()
                n_test = int(fold.test_idx.size)
                for (param_id, _), result in zip(items, results, strict=True):
                    params = param_list[param_id]
                    weights = result.weights
                    if not _solved(result.status):
                        score = (
                            float(self.error_score)
                            if np.isscalar(self.error_score)
                            else float("nan")
                        )
                    elif n_test == 0 or cv_plan.combinatorial:
                        score = float("nan")
                    else:
                        X_test = slice_rows(X, fold.test_idx)
                        score = score_with_estimator(
                            estimator,
                            params,
                            weights,
                            X_test,
                            scoring=self.scoring,
                        )
                    evaluations.append(
                        Evaluation(
                            template_id=template_ids[key],
                            fold_id=fold.fold_id,
                            param_id=param_id,
                            params=params,
                            weights=weights,
                            score=score,
                            n_test=n_test,
                            path_ids=list(fold.path_ids),
                            status=result.status,
                        )
                    )
                eval_s += time.perf_counter() - t_e

        self.search_plan_.templates = list(templates.values())
        self.search_plan_.evaluations = evaluations

        cv_results, best_idx, best_params, best_score = self._aggregate(
            param_list, evaluations, cv_plan, X
        )
        self.cv_results_ = cv_results
        self.best_index_ = best_idx
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.n_splits_ = cv_plan.n_splits
        self.multimetric_ = False

        if self.refit:
            best_est = clone(estimator)
            best_est.set_params(**best_params)
            best_est.fit(X, y)
            self.best_estimator_ = best_est
            self.refit_time_ = time.perf_counter() - t0

        self.acceleration_report_ = acceleration_report(
            backend=backend_used,
            n_folds=cv_plan.n_splits,
            n_params=len(param_list),
            n_solves=n_templates,
            n_updates=n_updates,
            n_prior_fits=cache.n_fits,
            compile_s=compile_s,
            instantiate_s=instantiate_s,
            solve_s=solve_s,
            eval_s=eval_s,
            wall_s=time.perf_counter() - t0,
            fallback_reason=None,
            n_templates=n_templates,
        )
        return self

    def _fit_sklearn(self, estimator, X, y, t0, reason: str):
        gs = sklearn_grid_search(
            estimator,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        gs.fit(X, y)
        self.best_estimator_ = gs.best_estimator_ if self.refit else None
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.best_index_ = gs.best_index_
        self.cv_results_ = gs.cv_results_
        self.n_splits_ = gs.n_splits_
        self.multimetric_ = gs.multimetric_
        self.acceleration_report_ = acceleration_report(
            backend="sklearn",
            n_folds=int(getattr(gs, "n_splits_", 0)),
            n_params=len(list(ParameterGrid(self.param_grid))),
            n_solves=0,
            n_updates=0,
            n_prior_fits=0,
            compile_s=0.0,
            instantiate_s=0.0,
            solve_s=0.0,
            eval_s=0.0,
            wall_s=time.perf_counter() - t0,
            fallback_reason=reason,
            n_templates=0,
        )
        return self

    def _aggregate(self, param_list, evaluations: list[Evaluation], cv_plan, X):
        n_params = len(param_list)
        n_splits = cv_plan.n_splits
        split_scores = np.full((n_splits, n_params), np.nan)

        if cv_plan.combinatorial:
            mean_test, std_test, split_scores = self._path_means(
                param_list, evaluations, cv_plan, X
            )
        else:
            for ev in evaluations:
                split_scores[ev.fold_id, ev.param_id] = ev.score
            mean_test = np.nanmean(split_scores, axis=0)
            std_test = (
                np.nanstd(split_scores, axis=0, ddof=1)
                if n_splits > 1
                else np.zeros(n_params)
            )

        results: dict[str, Any] = {
            "mean_test_score": mean_test,
            "std_test_score": std_test,
            "rank_test_score": _rank_desc(mean_test),
            "params": param_list,
        }
        for i, params in enumerate(param_list):
            for key, value in params.items():
                column = results.setdefault(f"param_{key}", [None] * n_params)
                column[i] = value
        for split_id in range(n_splits):
            results[f"split{split_id}_test_score"] = split_scores[split_id]

        best_idx = int(np.nanargmax(mean_test))
        return results, best_idx, param_list[best_idx], float(mean_test[best_idx])

    def _path_means(self, param_list, evaluations, cv_plan, X):
        n_params = len(param_list)
        n_paths = cv_plan.n_paths
        n_splits = cv_plan.n_splits
        split_scores = np.full((n_splits, n_params), np.nan)
        path_lists: list[list[list]] = [
            [[] for _ in range(n_paths)] for _ in range(n_params)
        ]

        for ev in evaluations:
            fold = cv_plan.folds[ev.fold_id]
            if ev.weights is None:
                continue
            portfolios = path_portfolios(X, ev.weights, fold.test_segments)
            split_scores[ev.fold_id, ev.param_id] = score_multi_period(
                portfolios, scoring=self.scoring
            )
            for portfolio, path_id in zip(portfolios, fold.path_ids, strict=False):
                if 0 <= path_id < n_paths:
                    path_lists[ev.param_id][path_id].append(portfolio)

        path_scores = np.full((n_paths, n_params), np.nan)
        for param_id in range(n_params):
            for path_id in range(n_paths):
                path_scores[path_id, param_id] = score_multi_period(
                    path_lists[param_id][path_id], scoring=self.scoring
                )
        mean_test = np.nanmean(path_scores, axis=0)
        std_test = (
            np.nanstd(path_scores, axis=0, ddof=1)
            if n_paths > 1
            else np.zeros(n_params)
        )
        return mean_test, std_test, split_scores

    def score(self, X, y=None):
        del y
        return self.best_estimator_.score(X)

    def predict(self, X):
        return self.best_estimator_.predict(X)




def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.nan_to_num(values, nan=-np.inf))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks

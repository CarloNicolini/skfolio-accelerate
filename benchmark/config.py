"""Defaults, datasets, and MeanRisk grid for the cross_val_predict benchmark."""

from types import SimpleNamespace

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

from skfolio_accelerate.flagship import factor_returns

CONFIG = {
    "synthetic_n_observations": 504,
    "synthetic_n_assets": 12,
    "synthetic_n_factors": 8,
    "synthetic_seed": 42,
    "target_folds": 15,
    "test_size": 21,
    "min_train_size": 147,
    "mrc_n_subsamples": 3,
    "mrc_asset_subset_size": 8,
    "mrc_window_size": 252,
    "mrc_train_size": 147,
    "mrc_random_state": 43,
    "cpcv_n_folds": 6,
    "cpcv_n_test_folds": 2,
    "cpcv_purged_size": 1,
    "cpcv_embargo_size": 1,
    "repetitions": 3,
    "warmups": 1,
    "workers": 1,
    "thread_limit": 1,
    "n_jobs": 1,
    "solver": "CLARABEL",
    "solver_params": None,
    "l2_coef": 1e-5,
    "timeout_s": None,
    "include_gini": False,
    "include_annualized": False,
    "include_extras": True,
    "include_lp_l2_zero": True,
    "random_seed": 42,
    "sp500_tail_observations": None,
}
FULL_PRESET = {
    "synthetic_n_observations": 20 * 252,
    "synthetic_n_assets": 20,
    "min_train_size": 252,
    "mrc_asset_subset_size": 12,
    "mrc_window_size": 756,
    "mrc_train_size": 651,
}
QUICK_PRESET = {
    "synthetic_n_observations": 120,
    "synthetic_n_assets": 6,
    "target_folds": 4,
    "test_size": 20,
    "min_train_size": 40,
    "mrc_n_subsamples": 2,
    "mrc_asset_subset_size": 4,
    "mrc_window_size": 80,
    "mrc_train_size": 40,
    "cpcv_n_folds": 4,
    "cpcv_n_test_folds": 3,
    "repetitions": 1,
    "sp500_tail_observations": 252,
}
DATASETS = ("synthetic", "sp500")
METHODS = ("native", "accelerated")
CV_KINDS = ("walk-forward", "multiple-randomized", "purged-cpcv")
PRESETS = {"full": FULL_PRESET, "quick": QUICK_PRESET}
SCHEMA_VERSION = 1
LP_L2_ZERO_RISKS = (
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasure.CVAR,
    RiskMeasure.WORST_REALIZATION,
)


class BenchmarkConfig(SimpleNamespace):
    def to_dict(self) -> dict:
        data = vars(self).copy()
        for key in ("datasets", "methods", "cv_kinds"):
            data[key] = list(getattr(self, key))
        return data


def build_config(
    overrides: dict | None = None,
    *,
    datasets=DATASETS,
    methods=METHODS,
    cv_kinds=None,
    preset: str | None = None,
) -> BenchmarkConfig:
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    raw = {**CONFIG, **(PRESETS[preset] if preset else {})}
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})
    if int(raw["repetitions"]) < 1:
        raise ValueError(
            f"repetitions must be an integer >= 1, got {raw['repetitions']!r}"
        )
    if raw["timeout_s"] is not None and float(raw["timeout_s"]) <= 0:
        raise ValueError("timeout_s must be None or a positive number")
    dataset_tuple, method_tuple = tuple(datasets), tuple(methods)
    cv_tuple = tuple(cv_kinds) if cv_kinds is not None else CV_KINDS
    if unknown := [n for n in dataset_tuple if n not in DATASETS]:
        raise ValueError(f"unknown datasets: {unknown}")
    if unknown := [n for n in method_tuple if n not in METHODS]:
        raise ValueError(f"unknown methods: {unknown}")
    if unknown := [n for n in cv_tuple if n not in CV_KINDS]:
        raise ValueError(f"unknown cv kinds: {unknown}")
    if not dataset_tuple or not method_tuple or not cv_tuple:
        raise ValueError("datasets, methods, and cv_kinds must be non-empty")
    return BenchmarkConfig(
        **raw, datasets=dataset_tuple, methods=method_tuple, cv_kinds=cv_tuple
    )


def make_synthetic(config: BenchmarkConfig) -> SimpleNamespace:
    X = factor_returns(
        config.synthetic_n_observations,
        config.synthetic_n_assets,
        n_factors=config.synthetic_n_factors,
        seed=config.synthetic_seed,
    )
    X.columns = [f"A{i}" for i in range(config.synthetic_n_assets)]
    return SimpleNamespace(name="synthetic", X=X)


def make_sp500(config: BenchmarkConfig) -> SimpleNamespace:
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    X = prices_to_returns(load_sp500_dataset())
    if config.sp500_tail_observations is not None:
        X = X.iloc[-int(config.sp500_tail_observations) :].copy()
    return SimpleNamespace(name="sp500", X=X)


def load_dataset(name: str, config: BenchmarkConfig) -> SimpleNamespace:
    if name == "synthetic":
        return make_synthetic(config)
    if name == "sp500":
        return make_sp500(config)
    raise ValueError(f"unknown dataset {name!r}")


def make_estimator(spec, asset_names: list[str]) -> MeanRisk:
    params = dict(spec.kwargs)
    if params.get("linear_constraints"):
        first = asset_names[0]
        params["linear_constraints"] = [
            item.replace("A0", first) for item in params["linear_constraints"]
        ]
    return MeanRisk(**params)


def mean_risk_specs(config: BenchmarkConfig) -> list:
    risks = [
        risk
        for risk in RiskMeasure
        if (config.include_annualized or not risk.is_annualized)
        and (config.include_gini or risk != RiskMeasure.GINI_MEAN_DIFFERENCE)
    ]
    solver = {"solver": config.solver}
    if config.solver_params is not None:
        solver["solver_params"] = dict(config.solver_params)
    l2 = config.l2_coef
    specs = []
    for objective in ObjectiveFunction:
        for risk in risks:
            kwargs = {
                "objective_function": objective,
                "risk_measure": risk,
                "l2_coef": l2,
                **solver,
            }
            specs.append(
                SimpleNamespace(
                    name=f"{objective.name}/{risk.name}",
                    objective=objective.name,
                    risk=risk.name,
                    extra="",
                    kwargs=kwargs,
                )
            )
    if config.include_extras:
        extras = [
            ("min_return", {"min_return": 1e-5, "l2_coef": l2, **solver}, "VARIANCE"),
            (
                "linear_constraints",
                {"linear_constraints": ["A0 <= 0.45"], "l2_coef": l2, **solver},
                "VARIANCE",
            ),
            (
                "management_fees",
                {"management_fees": 1e-4, "l2_coef": l2, **solver},
                "VARIANCE",
            ),
            ("l1_coef", {"l1_coef": 1e-3, "l2_coef": l2, **solver}, "VARIANCE"),
            (
                "min_return",
                {
                    "risk_measure": RiskMeasure.CVAR,
                    "min_return": 1e-6,
                    "l2_coef": l2,
                    **solver,
                },
                "CVAR",
            ),
        ]
        for extra, kwargs, risk in extras:
            specs.append(
                SimpleNamespace(
                    name=f"MINIMIZE_RISK/{risk}+{extra}",
                    objective="MINIMIZE_RISK",
                    risk=risk,
                    extra=extra,
                    kwargs=kwargs,
                )
            )
    if config.include_lp_l2_zero:
        for risk in LP_L2_ZERO_RISKS:
            kwargs = {
                "objective_function": ObjectiveFunction.MINIMIZE_RISK,
                "risk_measure": risk,
                "l2_coef": 0.0,
                **solver,
            }
            specs.append(
                SimpleNamespace(
                    name=f"MINIMIZE_RISK/{risk.name}+l2_0",
                    objective="MINIMIZE_RISK",
                    risk=risk.name,
                    extra="l2_0",
                    kwargs=kwargs,
                )
            )
    return specs

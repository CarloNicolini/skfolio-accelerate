"""Frozen multi-path workloads used by tests and the publishable benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
    optimal_folds_number,
)


def factor_returns(
    n_obs: int,
    n_assets: int,
    n_factors: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.01, size=(n_obs, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n_assets))
    idio = rng.normal(0.0, 0.005, size=(n_obs, n_assets))
    return pd.DataFrame(factors @ loadings + idio)


# Published-shaped MRC (Palomar-style). Baseline is minutes on one VM;
# the compact engine should be ≥10× vs cross_val_predict(n_jobs=-1).
FLAGSHIP_MRC = {
    "n_obs": 2520,
    "n_assets": 80,
    "n_subsamples": 80,
    "asset_subset_size": 25,
    "window_size": 3 * 252,
    "train_size": 252,
    "test_size": 21,
    "seed": 1,
}

# CI / unit-test scale.
SMOKE_MRC = {
    "n_obs": 420,
    "n_assets": 12,
    "n_subsamples": 4,
    "asset_subset_size": 6,
    "window_size": 320,
    "train_size": 80,
    "test_size": 20,
    "seed": 2,
}

SMOKE_CPCV = {
    "n_obs": 240,
    "n_assets": 8,
    "target_n_test_paths": 12,
    "target_train_size": 80,
    "seed": 3,
}


def make_mrc(spec: dict) -> tuple[pd.DataFrame, MultipleRandomizedCV]:
    X = factor_returns(spec["n_obs"], spec["n_assets"], seed=spec["seed"])
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(
            test_size=spec["test_size"], train_size=spec["train_size"]
        ),
        n_subsamples=spec["n_subsamples"],
        asset_subset_size=spec["asset_subset_size"],
        window_size=spec["window_size"],
        random_state=spec["seed"],
    )
    return X, cv


def make_cpcv(spec: dict) -> tuple[pd.DataFrame, CombinatorialPurgedCV]:
    X = factor_returns(spec["n_obs"], spec["n_assets"], seed=spec["seed"])
    n_folds, n_test = optimal_folds_number(
        n_observations=spec["n_obs"],
        target_train_size=spec["target_train_size"],
        target_n_test_paths=spec["target_n_test_paths"],
    )
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_test)
    return X, cv

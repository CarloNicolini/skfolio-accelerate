"""Shared synthetic returns for tests."""

from __future__ import annotations

import numpy as np


def synthetic_returns(n_observations: int = 120, n_assets: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0005, scale=0.01, size=(n_observations, n_assets))

"""Benchmark datasets: synthetic factor panel and skfolio S&P 500 prices."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from benchmark.config import BenchmarkConfig
from skfolio_accelerate.flagship import factor_returns


@dataclass(frozen=True)
class LoadedDataset:
    """Returns matrix plus generation metadata recorded in results."""

    name: str
    X: pd.DataFrame
    procedure: str
    seed: int | None
    source: str


def make_synthetic(config: BenchmarkConfig) -> LoadedDataset:
    """Deterministic factor-model returns matching :func:`factor_returns`.

    Generation (seeded ``numpy.random.default_rng``):

    * factors ~ Normal(0, 0.01) of shape ``(n_obs, n_factors)``
    * loadings ~ Normal(0, 1) of shape ``(n_factors, n_assets)``
    * idiosyncratic ~ Normal(0, 0.005) of shape ``(n_obs, n_assets)``
    * returns = ``factors @ loadings + idiosyncratic``

    Columns are named ``A0`` … so MeanRisk ``linear_constraints`` extras that
    reference ``A0`` match ``benchmarks/benchmark_sequential_mean_risk.py``.
    """
    X = factor_returns(
        config.synthetic_n_observations,
        config.synthetic_n_assets,
        n_factors=config.synthetic_n_factors,
        seed=config.synthetic_seed,
    )
    X.columns = [f"A{i}" for i in range(config.synthetic_n_assets)]
    procedure = (
        f"factor_returns(n_obs={config.synthetic_n_observations}, "
        f"n_assets={config.synthetic_n_assets}, "
        f"n_factors={config.synthetic_n_factors}, seed={config.synthetic_seed}); "
        "factors~N(0,0.01), loadings~N(0,1), idio~N(0,0.005); "
        "X = factors @ loadings + idio"
    )
    return LoadedDataset(
        name="synthetic",
        X=X,
        procedure=procedure,
        seed=config.synthetic_seed,
        source="skfolio_accelerate.flagship.factor_returns",
    )


def make_sp500(config: BenchmarkConfig) -> LoadedDataset:
    """Simple returns from :func:`skfolio.datasets.load_sp500_dataset`.

    The published file is **prices** (20 assets, 1990-01-02 … 2022-12-28).
    Native skfolio ``cross_val_predict`` expects **returns**, so this loader
    applies :func:`skfolio.preprocessing.prices_to_returns` on the full 20-asset
    panel. Optional ``sp500_tail_observations`` keeps the most recent *return*
    rows after conversion (still the same dataset, a trailing window).
    """
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    tail = config.sp500_tail_observations
    if tail is not None:
        X = X.iloc[-int(tail) :].copy()
    procedure = "load_sp500_dataset() prices -> prices_to_returns(full 20 assets)" + (
        f"; tail {tail} return rows" if tail is not None else "; all return rows"
    )
    return LoadedDataset(
        name="sp500",
        X=X,
        procedure=procedure,
        seed=None,
        source="skfolio.datasets.load_sp500_dataset",
    )


def load_dataset(name: str, config: BenchmarkConfig) -> LoadedDataset:
    if name == "synthetic":
        return make_synthetic(config)
    if name == "sp500":
        return make_sp500(config)
    raise ValueError(f"unknown dataset {name!r}")

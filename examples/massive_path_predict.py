"""Amortized multi-path backtest with massive_cross_val_predict."""

from __future__ import annotations

from skfolio.optimization import MeanRisk

from skfolio_accelerate import massive_cross_val_predict
from skfolio_accelerate.flagship import SMOKE_MRC, make_mrc


def main() -> None:
    X, cv = make_mrc(SMOKE_MRC)
    pred, report = massive_cross_val_predict(
        MeanRisk(), X, cv=cv, n_jobs=1, return_report=True
    )
    print(report)
    print("n_paths", len(pred))
    print("mean path Sharpe", float(sum(p.sharpe_ratio for p in pred) / len(pred)))


if __name__ == "__main__":
    main()

"""
=========================
Walk-forward MeanRisk CV
=========================

This example replaces skfolio's ``cross_val_predict`` with the amortized
drop-in from :mod:`skfolio_accelerate` on a short WalkForward backtest.
"""

# %%
# Load returns and define a WalkForward splitter
# ----------------------------------------------
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

from skfolio_accelerate import cross_val_predict

prices = load_sp500_dataset()
X = prices_to_returns(prices[["AAPL", "MSFT", "JPM", "XOM", "GE"]].dropna())
cv = WalkForward(train_size=252, test_size=63)

# %%
# Run amortized cross_val_predict
# -------------------------------
prediction, report = cross_val_predict(
    MeanRisk(),
    X,
    cv=cv,
    return_report=True,
)
print(report)
print("cumulative returns shape:", prediction.cumulative_returns_df.shape)

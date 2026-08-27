"""
============================
Acceleration report backends
============================

Compare the backend selected for MeanRisk variance versus EqualWeighted on the
same WalkForward plan.
"""

# %%
# Setup
# -----
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward
from skfolio.optimization import EqualWeighted, MeanRisk
from skfolio.preprocessing import prices_to_returns

from skfolio_accelerate import cross_val_predict

prices = load_sp500_dataset()
X = prices_to_returns(prices[["AAPL", "MSFT", "JPM", "XOM"]].dropna())
cv = WalkForward(train_size=252, test_size=63)

# %%
# MeanRisk uses the compact OSQP path
# -----------------------------------
_, mean_risk_report = cross_val_predict(
    MeanRisk(),
    X,
    cv=cv,
    return_report=True,
)
print("MeanRisk backend:", mean_risk_report.backend)

# %%
# EqualWeighted uses closed-form weights
# --------------------------------------
_, equal_report = cross_val_predict(
    EqualWeighted(),
    X,
    cv=cv,
    return_report=True,
)
print("EqualWeighted backend:", equal_report.backend)
print(equal_report)

"""
============================
Acceleration report backends
============================

The same compiled CV plan can run a compact MeanRisk solver or a serial
estimator that still calls native ``fit``. The second path is shared
bookkeeping (slices, assembly from ``weights_``), not an optimizer.
"""

# %%
# Setup
# -----
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward
from skfolio.optimization import HierarchicalRiskParity, MeanRisk
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
# HierarchicalRiskParity still fits, then shares assembly
# -------------------------------------------------------
_, hrp_report = cross_val_predict(
    HierarchicalRiskParity(),
    X,
    cv=cv,
    return_report=True,
)
print("HierarchicalRiskParity backend:", hrp_report.backend)
print(hrp_report)

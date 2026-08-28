"""
=======================================
Speedups by risk measure (Plotly)
=======================================

Two Plotly views of accelerator speedups:

1. A **live** micro-benchmark across variance and a few scenario risks on a
   short WalkForward plan (executed in CI).
2. The **published** 20-year WalkForward numbers from the project README, so
   the gallery also shows the long-workload regime where OSQP reuse dominates.

.. warning::

   ``skfolio-accelerate`` is **experimental**. Live timings vary with machine
   load; published factors are representative medians, not a guarantee.
"""

# %%
# Live micro-benchmark on synthetic returns
# -----------------------------------------
from time import perf_counter

import plotly.graph_objects as go
import plotly.io as pio
from plotly.io import show
from plotly.subplots import make_subplots
from skfolio import RiskMeasure
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import MeanRisk

from skfolio_accelerate import cross_val_predict
from skfolio_accelerate.flagship import factor_returns

X = factor_returns(n_obs=378, n_assets=6, seed=11)
cv = WalkForward(train_size=126, test_size=21)
risks = [
    ("Variance", RiskMeasure.VARIANCE),
    ("Semi-variance", RiskMeasure.SEMI_VARIANCE),
    ("CVaR", RiskMeasure.CVAR),
    ("Max drawdown", RiskMeasure.MAX_DRAWDOWN),
]


def _median_seconds(fn, repeats: int = 2) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


live_labels: list[str] = []
live_native: list[float] = []
live_accel: list[float] = []
live_speedup: list[float] = []
for label, risk in risks:
    estimator = MeanRisk(risk_measure=risk)
    native_s = _median_seconds(
        lambda est=estimator: skfolio_cross_val_predict(est, X, cv=cv, n_jobs=1)
    )
    accel_s = _median_seconds(
        lambda est=estimator: cross_val_predict(est, X, cv=cv, n_jobs=1)
    )
    live_labels.append(label)
    live_native.append(native_s)
    live_accel.append(accel_s)
    live_speedup.append(native_s / accel_s if accel_s > 0 else float("nan"))
    print(
        f"{label}: native={native_s:.4f}s  accel={accel_s:.4f}s  "
        f"×{live_speedup[-1]:.2f}"
    )

# %%
# Published long-workload WalkForward factors
# -------------------------------------------
# one isolated process on 5,040 × 20 synthetic daily returns
# (Python 3.12, skfolio 1.0.0, n_jobs=1), as reported in the project README.
published_labels = ["Variance", "Semi-variance", "MAD", "CVaR", "Max drawdown"]
published_walk_forward = [46.7, 2.29, 2.40, 3.38, 2.11]
published_mrc = [48.2, 3.05, 3.23, 4.33, 2.62]
published_cpcv = [10.8, 0.97, 0.85, 1.05, 1.07]

# %%
# Plotly: live wall times and published speedup factors
# -----------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Live micro-benchmark wall time (WalkForward, synthetic)",
        "Published 20-year speedup factors (native / accelerated)",
    ),
    vertical_spacing=0.16,
)

fig.add_trace(
    go.Bar(
        name="native skfolio",
        x=live_labels,
        y=live_native,
        marker_color="#4C78A8",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="skfolio-accelerate",
        x=live_labels,
        y=live_accel,
        marker_color="#F58518",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        name="WalkForward (228)",
        x=published_labels,
        y=published_walk_forward,
        marker_color="#54A24B",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="MRC (480)",
        x=published_labels,
        y=published_mrc,
        marker_color="#B279A2",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="CPCV (6)",
        x=published_labels,
        y=published_cpcv,
        marker_color="#FF9DA6",
    ),
    row=2,
    col=1,
)

fig.update_yaxes(title_text="seconds", row=1, col=1)
fig.update_yaxes(title_text="speedup ×", row=2, col=1)
fig.update_layout(
    title="MeanRisk speedups — live micro-benchmark and published long workloads",
    barmode="group",
    template="plotly_white",
    height=720,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=100, r=20, b=40, l=60),
)
# ``show`` writes HTML+PNG for Sphinx-Gallery; skip interactive backends locally.
if "sphinx_gallery" in str(pio.renderers.default):
    show(fig)

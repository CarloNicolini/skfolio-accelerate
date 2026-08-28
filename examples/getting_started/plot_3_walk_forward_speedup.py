"""
=================================
Walk-forward speedup (Plotly)
=================================

Measure wall-clock time for native skfolio ``cross_val_predict`` versus the
amortized drop-in on a compact WalkForward MeanRisk backtest, then plot the
result with Plotly.

.. warning::

   This library is **experimental**. Timings below are a smoke-scale demo for
   the docs gallery, not a substitute for the longer README benchmarks.
"""

# %%
# Synthetic returns and WalkForward plan
# --------------------------------------
# Use a modest factor panel so the example stays cheap enough for continuous
# integration while still exercising overlapping moment updates and OSQP reuse.
from time import perf_counter

import plotly.graph_objects as go
from skfolio.model_selection import WalkForward
from skfolio.model_selection import cross_val_predict as skfolio_cross_val_predict
from skfolio.optimization import MeanRisk

from skfolio_accelerate import cross_val_predict
from skfolio_accelerate.flagship import factor_returns

X = factor_returns(n_observations=504, n_assets=8, seed=7)
cv = WalkForward(train_size=126, test_size=21)

# %%
# Time native skfolio and the accelerator
# ---------------------------------------
# Both calls use ``n_jobs=1``. The accelerated path selects the compact OSQP
# backend for default variance MeanRisk.


def _median_seconds(fn, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


native_s = _median_seconds(
    lambda: skfolio_cross_val_predict(MeanRisk(), X, cv=cv, n_jobs=1)
)
accelerated_s = _median_seconds(
    lambda: cross_val_predict(MeanRisk(), X, cv=cv, n_jobs=1)
)
speedup = native_s / accelerated_s if accelerated_s > 0 else float("nan")
print(f"native={native_s:.4f}s  accelerated={accelerated_s:.4f}s  speedup={speedup:.2f}x")

# %%
# Plotly bar chart of the measured wall times
# -------------------------------------------
fig = go.Figure(
    data=[
        go.Bar(
            name="wall time",
            x=["native skfolio", "skfolio-accelerate"],
            y=[native_s, accelerated_s],
            text=[f"{native_s:.3f}s", f"{accelerated_s:.3f}s"],
            textposition="outside",
            marker_color=["#4C78A8", "#F58518"],
        )
    ]
)
fig.update_layout(
    title=(
        f"WalkForward MeanRisk variance — median of 3 runs "
        f"({speedup:.1f}× speedup)"
    ),
    yaxis_title="seconds",
    template="plotly_white",
    showlegend=False,
    margin=dict(t=60, r=20, b=40, l=60),
    height=420,
)
fig

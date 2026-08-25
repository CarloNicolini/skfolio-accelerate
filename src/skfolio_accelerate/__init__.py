"""skfolio-accelerate: compiled massive CV and amortized multi-path backtests."""

from skfolio_accelerate.ir import AccelerationReport
from skfolio_accelerate.predict import massive_cross_val_predict, path_sharpes
from skfolio_accelerate.search import MassiveGridSearchCV, rust_engine_available

__all__ = [
    "AccelerationReport",
    "MassiveGridSearchCV",
    "massive_cross_val_predict",
    "path_sharpes",
    "rust_engine_available",
]

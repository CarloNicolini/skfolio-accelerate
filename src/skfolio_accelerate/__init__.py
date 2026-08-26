"""skfolio-accelerate: drop-in amortized ``cross_val_predict`` for skfolio."""

from skfolio_accelerate.predict import (
    AccelerationReport,
    cross_val_predict,
    massive_cross_val_predict,
)
from skfolio_accelerate.scoring import path_sharpes
from skfolio_accelerate.search import GridSearchResult, grid_search

__all__ = [
    "AccelerationReport",
    "GridSearchResult",
    "cross_val_predict",
    "grid_search",
    "massive_cross_val_predict",
    "path_sharpes",
]

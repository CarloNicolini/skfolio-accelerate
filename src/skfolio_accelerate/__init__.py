"""skfolio-accelerate: drop-in amortized ``cross_val_predict`` for skfolio."""

from skfolio_accelerate.predict import (
    AccelerationReport,
    AccelerationWarning,
    cross_val_predict,
)
from skfolio_accelerate.scoring import (
    path_sharpes,
    ranking_precision_at_k,
    spearman_rank_correlation,
)
from skfolio_accelerate.search import GridSearchResult, grid_search

__all__ = [
    "AccelerationReport",
    "AccelerationWarning",
    "GridSearchResult",
    "cross_val_predict",
    "grid_search",
    "path_sharpes",
    "ranking_precision_at_k",
    "spearman_rank_correlation",
]

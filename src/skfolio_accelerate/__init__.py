"""skfolio-accelerate: compiled massive CV for skfolio."""

from skfolio_accelerate.ir import AccelerationReport
from skfolio_accelerate.search import MassiveGridSearchCV, rust_engine_available

MassiveGridSearchCV = MassiveGridSearchCV

__all__ = [
    "AccelerationReport",
    "MassiveGridSearchCV",
    "MassiveGridSearchCV",
    "rust_engine_available",
]

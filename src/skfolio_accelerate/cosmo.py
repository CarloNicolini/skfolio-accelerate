"""Public COSMO.rs helpers for the optional persistent compact backend."""

from skfolio_accelerate._cosmo import (
    CosmoFoldTrace,
    PersistMode,
    RestartPolicy,
    cosmo_available,
    cosmo_persistence_api_available,
    default_cosmo_settings,
    default_persist_mode,
    make_cosmo_engine,
    normalize_persist_mode,
    uses_cosmo_solver,
)
from skfolio_accelerate.formulations import (
    FormulationRecord,
    formulation_record,
    formulation_table,
    persist_class_for,
    to_markdown,
)

__all__ = [
    "CosmoFoldTrace",
    "FormulationRecord",
    "PersistMode",
    "RestartPolicy",
    "cosmo_available",
    "cosmo_persistence_api_available",
    "default_cosmo_settings",
    "default_persist_mode",
    "formulation_record",
    "formulation_table",
    "make_cosmo_engine",
    "normalize_persist_mode",
    "persist_class_for",
    "to_markdown",
    "uses_cosmo_solver",
]

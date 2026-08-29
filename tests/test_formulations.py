"""Formulation table completeness."""

from __future__ import annotations

from pathlib import Path

from skfolio import RiskMeasure
from skfolio.optimization.convex import ObjectiveFunction

from skfolio_accelerate.formulations import (
    formulation_record,
    formulation_table,
    persist_class_for,
    to_markdown,
)

_FORMULATIONS_MD = (
    Path(__file__).resolve().parents[1] / "docs" / "cosmo_meanrisk_formulations.md"
)


def test_every_risk_measure_is_classified():
    names = {row.risk for row in formulation_table()}
    assert names == {risk.name for risk in RiskMeasure}


def test_persist_class_parallel_is_f():
    assert persist_class_for(RiskMeasure.VARIANCE, n_jobs=-1) == "F"
    assert persist_class_for(RiskMeasure.VARIANCE, n_jobs=1) == "C"
    assert persist_class_for(RiskMeasure.CVAR, expanding=True) == "E"


def test_docs_table_matches_generated_markdown():
    assert _FORMULATIONS_MD.read_text(encoding="utf-8") == to_markdown()
    text = to_markdown()
    assert "VARIANCE" in text
    assert "GINI_MEAN_DIFFERENCE" in text
    row = formulation_record(RiskMeasure.VARIANCE)
    assert row.cosmo_support == "compact"
    assert ObjectiveFunction.MAXIMIZE_RATIO.name in text

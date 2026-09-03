"""Lightweight tests for the canonical cross_val_predict benchmark suite."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from benchmark.cli import app
from benchmark.config import (
    CV_KINDS,
    SCHEMA_VERSION,
    build_config,
    make_synthetic,
    mean_risk_specs,
)
from benchmark.harness import (
    SCHEMA_FIELDS,
    attach_native_comparisons,
    compare_in_run_rows,
    delta_sharpe,
    delta_time,
    fold_index_fingerprint,
    make_cv,
    parse_raw_times,
    parse_results_csv,
    relative_delta_pct,
    relative_sharpe_error,
    relative_time,
    speedup,
    timing_summary,
    write_csv,
    write_json,
    write_summary_md,
)


def test_synthetic_data_is_deterministic():
    config = build_config()
    first = make_synthetic(config).X
    second = make_synthetic(config).X
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    assert first.shape == (
        config.synthetic_n_observations,
        config.synthetic_n_assets,
    )
    assert list(first.columns) == [f"A{i}" for i in range(config.synthetic_n_assets)]


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError, match="repetitions"):
        build_config({"repetitions": 0})
    with pytest.raises(ValueError, match="unknown datasets"):
        build_config(datasets=("nope",))
    with pytest.raises(ValueError, match="unknown methods"):
        build_config(methods=("osqp",))
    with pytest.raises(ValueError, match="unknown cv"):
        build_config(cv_kinds=("kfold",))
    with pytest.raises(ValueError, match="timeout"):
        build_config({"timeout_s": 0})


def test_mean_risk_grid_covers_sequential_objectives_and_risks():
    from skfolio import RiskMeasure
    from skfolio.optimization import ObjectiveFunction

    specs = mean_risk_specs(build_config())
    names = {spec.name for spec in specs}
    non_annualized = [risk for risk in RiskMeasure if not risk.is_annualized]
    for objective in ObjectiveFunction:
        for risk in non_annualized:
            if risk.name == "GINI_MEAN_DIFFERENCE":
                assert f"{objective.name}/{risk.name}" not in names
            else:
                assert f"{objective.name}/{risk.name}" in names
    assert "MINIMIZE_RISK/VARIANCE+min_return" in names
    assert "MINIMIZE_RISK/CVAR+l2_0" in names
    gini = mean_risk_specs(build_config({"include_gini": True}))
    assert any(spec.risk == "GINI_MEAN_DIFFERENCE" for spec in gini)


def test_result_schema_roundtrip(tmp_path: Path):
    row = {key: "" for key in SCHEMA_FIELDS}
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "synthetic",
            "estimator": "MINIMIZE_RISK/VARIANCE",
            "method": "native",
            "time_s": 1.5,
            "raw_times_s": "1.4|1.5|1.6",
            "status": "ok",
        }
    )
    path = tmp_path / "results.csv"
    write_csv(path, [row])
    parsed = parse_results_csv(path)
    assert parsed[0]["dataset"] == "synthetic"
    assert parsed[0]["schema_version"] == str(SCHEMA_VERSION)
    assert set(SCHEMA_FIELDS) <= set(parsed[0])
    times = parse_raw_times(parsed[0]["raw_times_s"])
    assert times == pytest.approx([1.4, 1.5, 1.6])


def test_identical_walk_forward_folds():
    config = build_config()
    X = make_synthetic(config).X
    first = fold_index_fingerprint(X, make_cv("walk-forward", config, len(X)))
    second = fold_index_fingerprint(X, make_cv("walk-forward", config, len(X)))
    assert first == second
    assert len(first) == config.target_folds
    train, test, assets = first[0]
    assert train and test
    assert set(train).isdisjoint(set(test))
    assert assets == ()


def test_cv_fold_counts_are_aligned():
    config = build_config()
    assert config.cv_kinds == CV_KINDS
    X = make_synthetic(config).X
    counts = [
        len(fold_index_fingerprint(X, make_cv(kind, config, len(X))))
        for kind in CV_KINDS
    ]
    assert counts == [config.target_folds] * 3


def test_identical_mrc_folds_with_fixed_seed():
    config = build_config(cv_kinds=("multiple-randomized",))
    X = make_synthetic(config).X
    first = fold_index_fingerprint(X, make_cv("multiple-randomized", config, len(X)))
    second = fold_index_fingerprint(X, make_cv("multiple-randomized", config, len(X)))
    assert first == second


def test_timing_and_sharpe_comparisons():
    summary = timing_summary([2.0, 1.0, 3.0])
    assert summary["time_s"] == 2.0
    assert summary["time_s_mean"] == 2.0
    assert summary["n_repetitions"] == 3
    assert summary["time_s_min"] == 1.0
    assert math.isclose(summary["time_s_std"], 1.0)
    assert speedup(10.0, 2.0) == 5.0
    assert delta_time(10.0, 2.0) == -8.0
    assert relative_time(10.0, 2.0) == 0.2
    assert delta_sharpe(1.0, 1.1) == pytest.approx(0.1)
    assert relative_sharpe_error(2.0, 2.2) == pytest.approx(0.1)
    assert math.isnan(speedup(1.0, 0.0))
    native = {"method": "native", "time_s": 4.0, "mean_sharpe": 1.0}
    acc = {"method": "accelerated", "time_s": 1.0, "mean_sharpe": 1.05}
    compared = attach_native_comparisons(acc, native)
    assert compared["speedup"] == 4.0
    assert compared["delta_time_s"] == -3.0
    failed_native = {
        "method": "native",
        "time_s": None,
        "mean_sharpe": None,
        "status": "SolverError",
    }
    failed = attach_native_comparisons(failed_native, failed_native)
    assert math.isnan(failed["speedup"])
    assert relative_delta_pct(10.0, 12.0) == pytest.approx(20.0)
    assert math.isnan(relative_delta_pct(0.0, 1.0))


def test_parse_raw_times_formats():
    assert parse_raw_times("1|2|3") == [1.0, 2.0, 3.0]
    assert parse_raw_times("[1.0, 2.0]") == [1.0, 2.0]
    assert parse_raw_times([]) == []
    assert parse_raw_times("") == []


def test_summary_writer(tmp_path: Path):
    rows = [
        {
            "dataset": "synthetic",
            "estimator": "MINIMIZE_RISK/VARIANCE",
            "method": "native",
            "time_s": 2.0,
            "delta_time_s": 0.0,
            "relative_time": 1.0,
            "speedup": 1.0,
            "mean_sharpe": 0.2,
            "delta_sharpe": 0.0,
            "relative_sharpe_error": 0.0,
        }
    ]
    path = tmp_path / "summary.md"
    write_summary_md(path, rows=rows, environment={"timestamp": "t", "packages": {}})
    text = path.read_text()
    assert "Speed-up" in text
    assert "MINIMIZE_RISK/VARIANCE" in text


def test_json_writer_roundtrip(tmp_path: Path):
    path = tmp_path / "results.json"
    write_json(path, {"schema_version": SCHEMA_VERSION, "rows": []})
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_cli_help():
    runner = CliRunner()
    run_help = runner.invoke(app, ["run", "--help"])
    assert run_help.exit_code == 0
    assert "--quick" in run_help.output
    assert "--output-dir" in run_help.output
    rel_help = runner.invoke(app, ["relative", "--help"])
    assert rel_help.exit_code == 0
    assert "--base" in rel_help.output


def test_in_run_relative_delta_rows():
    base = [
        {
            "dataset": "synthetic",
            "cv": "walk-forward",
            "estimator": "MINIMIZE_RISK/VARIANCE",
            "method": "accelerated",
            "time_s": 2.0,
            "speedup": 4.0,
            "mean_sharpe": 0.1,
            "status": "ok",
        }
    ]
    head = [{**base[0], "time_s": 2.5, "speedup": 3.2}]
    rows = compare_in_run_rows(base, head)
    assert rows[0]["delta_pct"] == pytest.approx(25.0)
    assert rows[0]["delta_time_s"] == pytest.approx(0.5)


def test_agents_md_requires_in_run_relative_benchmark():
    text = Path(__file__).resolve().parents[1].joinpath("AGENTS.md").read_text()
    assert "python benchmark/run_relative.py" in text
    assert "python benchmark/run_benchmark.py" in text
    assert "100 * (head_time - base_time) / base_time" in text
    assert "native_time / accelerated_time" in text
    assert "one-off timers" in text
    assert "baseline.json" not in text


def test_benchmark_runner_does_not_shadow_relative_install():
    runner = (
        Path(__file__).resolve().parents[1] / "benchmark" / "run_benchmark.py"
    ).read_text()
    assert 'ROOT / "src"' not in runner

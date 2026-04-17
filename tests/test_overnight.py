"""Tests for overnight orchestration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from porchbench.overnight import (
    classify_suite,
    build_plan,
    discover_suites,
    estimate_duration,
    format_estimate,
    OvernightTask,
    OvernightResult,
)
from porchbench.schemas import (
    Message,
    ModelOptions,
    Prompt,
    Strategy,
    Suite,
    SuiteDefaults,
    SuiteMetadata,
)


def _make_suite(name: str, n_prompts: int, strategies: dict | None = None) -> Suite:
    """Create a minimal Suite for testing."""
    prompts = [
        Prompt(
            id=f"p-{i}",
            category="coding",
            difficulty="easy",
            messages=[Message(role="user", content=f"prompt {i}")],
        )
        for i in range(n_prompts)
    ]

    return Suite(
        suite=SuiteMetadata(name=name, version="1.0", description="test"),
        defaults=SuiteDefaults(options=ModelOptions()),
        prompts=prompts,
        strategies=strategies or {},
    )


class TestClassifySuite:
    def test_standard_suite_no_strategies(self):
        suite = _make_suite("basic", 10)
        assert classify_suite(suite) == "standard"

    def test_discovery_suite_with_strategies(self):
        suite = _make_suite("routing", 10, strategies={
            "universal": Strategy(),
            "brevity": Strategy(system_message="Be brief."),
        })
        assert classify_suite(suite) == "discovery"

    def test_empty_strategies_is_standard(self):
        suite = _make_suite("basic", 10, strategies={})
        assert classify_suite(suite) == "standard"


class TestDiscoverSuites:
    def test_finds_yaml_files(self, tmp_path):
        (tmp_path / "a.yaml").write_text("suite: {}")
        (tmp_path / "b.yaml").write_text("suite: {}")
        (tmp_path / "c.txt").write_text("not a suite")
        paths = discover_suites(tmp_path)
        assert len(paths) == 2
        assert all(p.suffix == ".yaml" for p in paths)

    def test_sorted_by_name(self, tmp_path):
        (tmp_path / "z-suite.yaml").write_text("")
        (tmp_path / "a-suite.yaml").write_text("")
        paths = discover_suites(tmp_path)
        assert paths[0].name == "a-suite.yaml"
        assert paths[1].name == "z-suite.yaml"

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            discover_suites(Path("/nonexistent/path"))

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No .yaml"):
            discover_suites(tmp_path)


class TestBuildPlan:
    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_standard_suite_plan(self, mock_ref, mock_load):
        suite = _make_suite("coding", 10)
        mock_load.return_value = suite
        mock_ref.return_value = MagicMock()

        plan = build_plan([Path("suites/coding-basics.yaml")], ["model-a", "model-b"], repeats=3)

        assert len(plan) == 1
        task = plan[0]
        assert task.dispatch_type == "standard"
        assert task.repeats == 3
        assert task.run_count == 10 * 2 * 3  # prompts * models * repeats

    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_discovery_suite_plan(self, mock_ref, mock_load):
        suite = _make_suite("routing", 20, strategies={
            "universal": Strategy(),
            "brevity": Strategy(system_message="Brief."),
            "cot": Strategy(system_message="Think step by step."),
        })
        mock_load.return_value = suite
        mock_ref.return_value = MagicMock()

        plan = build_plan([Path("suites/routing-discovery.yaml")], ["model-a"], repeats=3)

        assert len(plan) == 1
        task = plan[0]
        assert task.dispatch_type == "discovery"
        assert task.repeats == 1  # discovery ignores repeats
        assert task.strategy_count == 3
        assert task.run_count == 20 * 3 * 1  # prompts * strategies * models

    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_mixed_plan(self, mock_ref, mock_load):
        standard = _make_suite("coding", 10)
        discovery = _make_suite("routing", 5, strategies={"a": Strategy(), "b": Strategy()})

        mock_load.side_effect = [standard, discovery]
        mock_ref.return_value = MagicMock()

        plan = build_plan(
            [Path("suites/coding-basics.yaml"), Path("suites/routing-discovery.yaml")],
            ["m1", "m2"],
            repeats=2,
        )

        assert len(plan) == 2
        assert plan[0].dispatch_type == "standard"
        assert plan[1].dispatch_type == "discovery"


class TestEstimateDuration:
    def test_basic_estimate(self):
        task = MagicMock(run_count=100)
        assert estimate_duration([task], seconds_per_prompt=10.0) == 1000.0

    def test_multiple_tasks(self):
        t1 = MagicMock(run_count=50)
        t2 = MagicMock(run_count=30)
        assert estimate_duration([t1, t2], seconds_per_prompt=20.0) == 1600.0

    def test_empty_plan(self):
        assert estimate_duration([]) == 0.0


class TestFormatEstimate:
    def test_minutes_only(self):
        assert format_estimate(300) == "~5m"

    def test_hours_and_minutes(self):
        assert format_estimate(5400) == "~1h 30m"

    def test_zero(self):
        assert format_estimate(0) == "~0m"

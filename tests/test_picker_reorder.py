"""Integration tests for the suite-first picker reorder.

The unit tests cover `_format_model_label` and `required_capabilities_for_suite`
in isolation, but nothing else asserts that the cli actually wires the suite's
required capabilities into the model picker. A regression where
`required_capabilities_for_suite(suite)` got swapped for `[]` would pass the
unit suite but break the user-facing UX silently.

Each test patches `select_suite`/`select_suites` and `select_models` at the
source module (the cli imports them lazily, so the patched attribute is what
the lazy `from porchbench.interactive import ...` resolves to). The
`select_models` stub captures kwargs and raises `typer.Exit(99)` to short-
circuit the rest of the CLI flow — keeps tests fast and isolated to the
picker-wiring assertion.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from porchbench.assets import find_suite
from porchbench.cli import app

runner = CliRunner()


def _capture_models_kwargs(captured: dict):
    """Build a select_models stub that records its kwargs and exits."""
    def _stub(backend, required_capabilities=None):
        captured["backend"] = backend
        captured["required_capabilities"] = required_capabilities
        raise typer.Exit(99)
    return _stub


# ---------------------------------------------------------------------------
# `porchbench run`
# ---------------------------------------------------------------------------


class TestRunPickerReorder:
    def test_tool_use_suite_passes_tools_requirement(self):
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suite",
                return_value=find_suite("tool-use"),
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["run"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == ["tools"]

    def test_text_suite_passes_empty_requirement(self):
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suite",
                return_value=find_suite("coding-basics"),
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["run"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == []


# ---------------------------------------------------------------------------
# `porchbench routes discover`
# ---------------------------------------------------------------------------


class TestRoutesDiscoverPickerReorder:
    def test_tool_use_strategies_suite_passes_tools_requirement(self):
        # tool-use.yaml carries both `strategies:` and `mode: tool-use` prompts,
        # so it satisfies the routes-discover suite filter and triggers the
        # capability requirement at the same time.
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suite",
                return_value=find_suite("tool-use"),
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["routes", "discover"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == ["tools"]

    def test_text_strategies_suite_passes_empty_requirement(self):
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suite",
                return_value=find_suite("routing-discovery"),
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["routes", "discover"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == []


# ---------------------------------------------------------------------------
# `porchbench overnight`
# ---------------------------------------------------------------------------


class TestOvernightPickerReorder:
    def test_mixed_suites_union_includes_tools(self):
        # Mix one tool-use suite with one text-only suite — the union should
        # still demand `tools` because at least one suite needs it.
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[
                    find_suite("tool-use"),
                    find_suite("coding-basics"),
                ],
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["overnight"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == ["tools"]

    def test_only_text_suites_yields_empty_requirement(self):
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[
                    find_suite("coding-basics"),
                    find_suite("cross-domain"),
                ],
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["overnight"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == []

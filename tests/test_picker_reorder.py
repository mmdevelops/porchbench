"""Integration tests for the suite-first picker reorder + unified run command.

The unit tests cover `_format_model_label` and `required_capabilities_for_suite`
in isolation, but nothing else asserts that the cli actually wires the suite's
required capabilities into the model picker. A regression where
`required_capabilities_for_suite(suite)` got swapped for `[]` would pass the
unit suite but break the user-facing UX silently.

Each test patches `select_suites` and `select_models` at the source module
(the cli imports them lazily, so the patched attribute is what the lazy
`from porchbench.interactive import ...` resolves to). The `select_models`
stub captures kwargs and raises `typer.Exit(99)` to short-circuit the rest
of the CLI flow — keeps tests fast and isolated to the picker-wiring
assertion.
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
# `porchbench run` — single-suite picker required-cap derivation
# ---------------------------------------------------------------------------


class TestRunSingleSuitePickerReorder:
    def test_tool_use_suite_passes_tools_requirement(self):
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[find_suite("tool-use")],
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
                "porchbench.interactive.select_suites",
                return_value=[find_suite("coding-basics")],
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
# `porchbench run --strategies` strategy-matrix path
# (replaces the deleted `routes discover` and `overnight --strategies` paths)
# ---------------------------------------------------------------------------


class TestRunStrategiesPath:
    def test_strategies_flag_with_strategies_suite_passes_through(self):
        # tool-use carries both `strategies:` and `mode: tool-use` prompts,
        # so --strategies + this suite reaches the model picker (no early
        # exit) and the picker still gets the tools requirement.
        captured: dict = {}
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[find_suite("tool-use")],
            ),
            patch(
                "porchbench.interactive.select_models",
                side_effect=_capture_models_kwargs(captured),
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["run", "--strategies"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == ["tools"]

    def test_strategies_flag_with_no_strategies_single_suite_hard_fails(self):
        # Single-suite + --strategies + suite has no strategies block:
        # unambiguously wrong intent. CLI hard-fails before reaching the
        # model picker, with a clear message naming the suite.
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[find_suite("coding-basics")],
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(app, ["run", "--strategies"])

        assert result.exit_code == 1, result.output
        assert "no `strategies:` block" in result.output
        assert "coding-basics.yaml" in result.output

    def test_strategies_plus_prompt_id_filter_rejected(self):
        # --strategies dispatches through run_discovery which doesn't honor
        # the per-prompt-id filter. Reject the combination explicitly so
        # users don't get a silently-ignored -p flag. Tracked for v0.2.
        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[find_suite("tool-use")],
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
        ):
            result = runner.invoke(
                app, ["run", "--strategies", "-p", "t1-read-file"],
            )

        assert result.exit_code == 1, result.output
        assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# `porchbench run` multi-suite — required-caps union, mixed-strategies plan
# ---------------------------------------------------------------------------


class TestRunMultiSuitePickerReorder:
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
            result = runner.invoke(app, ["run"])

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
            result = runner.invoke(app, ["run"])

        assert result.exit_code == 99, result.output
        assert captured["required_capabilities"] == []


# ---------------------------------------------------------------------------
# `porchbench overnight` migration shim
# ---------------------------------------------------------------------------


class TestOvernightMigrationShim:
    def test_overnight_invocation_prints_breadcrumb_and_exits(self):
        result = runner.invoke(app, ["overnight"])

        assert result.exit_code == 2, result.output
        assert "consolidated in v0.1" in result.output
        assert "porchbench run" in result.output

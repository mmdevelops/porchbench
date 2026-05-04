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

    def test_overnight_with_options_prints_breadcrumb(self):
        """Users running their old `overnight --strategies -s X -m Y`
        invocations must hit the migration breadcrumb, not a Typer
        "No such option" parser error. The shim accepts arbitrary
        extra args via context_settings and ignores them.
        """
        result = runner.invoke(
            app, ["overnight", "--strategies", "-s", "tool-use", "-m", "qwen3:8b"],
        )

        assert result.exit_code == 2, result.output
        assert "consolidated in v0.1" in result.output
        assert "porchbench run --strategies" in result.output

    def test_overnight_with_positional_args_prints_breadcrumb(self):
        result = runner.invoke(app, ["overnight", "foo", "bar", "--unknown=qux"])

        assert result.exit_code == 2, result.output
        assert "consolidated in v0.1" in result.output


# ---------------------------------------------------------------------------
# Re-pick judge toggle implies --evaluate
# ---------------------------------------------------------------------------


class TestRepickImpliesEvaluate:
    """Re-pick judge alone is meaningless without post-phase evaluation.
    Picking a different judge for nothing-to-judge would silently no-op,
    leaving the user wondering why the model picker never appeared (the
    bug behind this test). The CLI promotes `do_evaluate=True` when the
    user tickets Re-pick alone, and surfaces a [Note] line so the
    auto-promotion is visible.
    """

    def test_repick_alone_promotes_evaluate_and_prints_note(self):
        captured: dict = {}

        def _resolve_stub(backend_name, explicit, backend, *, interactive, force_pick=False):
            captured["force_pick"] = force_pick
            captured["called"] = True
            raise typer.Exit(99)

        opts_returned = {
            "repeats": 1,
            "verbose": False,
            "resume": False,
            "profile_vram": False,
            "profile": False,
            "evaluate": False,        # user did NOT tick Evaluate
            "repick_judge": True,     # user DID tick Re-pick
            "strategies": False,
        }

        with (
            patch(
                "porchbench.interactive.select_suites",
                return_value=[find_suite("coding-basics")],
            ),
            patch(
                "porchbench.interactive.select_models",
                return_value=["fake-model:1b"],
            ),
            patch(
                "porchbench.interactive.select_run_options",
                return_value=opts_returned,
            ),
            patch("porchbench.cli.construct_backend", return_value=MagicMock()),
            patch("porchbench.cli.check_server_or_exit"),
            patch("porchbench.cli.check_models_or_exit"),
            patch("porchbench.cli.check_tool_support_or_exit"),
            patch("porchbench.cli.resolve_eval_model_or_exit", side_effect=_resolve_stub),
        ):
            result = runner.invoke(app, ["run"])

        # The auto-promotion message surfaces so the user knows what
        # happened.
        assert "Re-pick judge implies --evaluate" in result.output, result.output
        # Crucially: resolve_eval_model_or_exit was reached at all,
        # which proves do_evaluate was flipped to True. Also confirms
        # force_pick=True propagated from the toggle.
        assert captured.get("called") is True, "resolve was never called — eval branch skipped"
        assert captured.get("force_pick") is True, "force_pick should propagate from repick_judge"
        assert result.exit_code == 99, result.output

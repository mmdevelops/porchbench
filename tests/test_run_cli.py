"""Tests for the `porchbench run` CLI — particularly the post-phase --evaluate hook."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from porchbench.cli import app
from porchbench.schemas import (
    Message,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    RunMetadata,
    RunResult,
    RunSummary,
    SuiteReference,
)

runner = CliRunner()


def _fake_run_result(run_id: str, model: str = "qwen2.5:3b") -> RunResult:
    return RunResult(
        run=RunMetadata(
            id=run_id,
            suite=SuiteReference(
                name="Coding Basics", version="1.0",
                file="<bundled>/coding-basics.yaml", sha256="x", rubric="coding",
            ),
            model=ModelInfo(name=model),
        ),
        results=[
            PromptResult(
                prompt_id="p1",
                category="coding",
                difficulty="easy",
                options_used=ModelOptions(),
                request=RequestData(messages=[Message(role="user", content="q")]),
                response=ResponseData(message=ResponseMessage(content="a")),
                metrics=PromptMetrics(),
            )
        ],
        summary=RunSummary(total_prompts=1, completed=1, failed=0, total_duration_s=1.0),
    )


@pytest.fixture
def mocked_run_environment(tmp_path: Path):
    """Stub the inference path so `porchbench run` exercises only its own orchestration."""
    fake_result = _fake_run_result("11111111-aaaa-bbbb-cccc-dddddddddddd")

    async def _fake_run_suite(*args, **kwargs):
        return fake_result

    with (
        patch("porchbench.cli.construct_backend", return_value=MagicMock()),
        patch("porchbench.cli.check_server_or_exit"),
        patch("porchbench.cli.check_models_or_exit"),
        patch("porchbench.cli.run_suite", new=_fake_run_suite),
    ):
        yield fake_result


def test_run_with_evaluate_triggers_post_phase_with_collected_paths(
    mocked_run_environment: RunResult, tmp_path: Path,
):
    """`run --evaluate` collects every result path and hands them to the post-phase batch."""
    fake_result = mocked_run_environment

    with patch("porchbench.cli._run_post_phase_evaluation") as post_phase:
        res = runner.invoke(
            app,
            [
                "run",
                "--suite", "coding-basics",
                "--model", "qwen2.5:3b",
                "--output-dir", str(tmp_path),
                "--evaluate",
                "--eval-backend", "ollama",
                "--eval-model", "gemma4:e4b",
            ],
        )

    assert res.exit_code == 0, res.output
    post_phase.assert_called_once()
    kwargs = post_phase.call_args.kwargs
    # The path we expect is the deterministic name run_suite would have written.
    from porchbench.runner import result_path_for
    expected = result_path_for(fake_result, tmp_path)
    assert kwargs["eval_paths"] == [expected]
    assert kwargs["eval_backend_name"] == "ollama"
    assert kwargs["eval_model"] == "gemma4:e4b"


def test_run_without_evaluate_skips_post_phase(
    mocked_run_environment: RunResult, tmp_path: Path,
):
    """Default (no --evaluate) leaves the post-phase batch entirely uninvoked."""
    with patch("porchbench.cli._run_post_phase_evaluation") as post_phase:
        res = runner.invoke(
            app,
            [
                "run",
                "--suite", "coding-basics",
                "--model", "qwen2.5:3b",
                "--output-dir", str(tmp_path),
            ],
        )

    assert res.exit_code == 0, res.output
    post_phase.assert_not_called()


# ---------------------------------------------------------------------------
# --set KEY=VALUE override surface
# ---------------------------------------------------------------------------


class TestParseSetOverrides:
    def test_empty_input_returns_empty_dict(self):
        from porchbench.cli import parse_set_overrides
        assert parse_set_overrides(None) == {}
        assert parse_set_overrides([]) == {}

    def test_yaml_typed_values_round_trip(self):
        from porchbench.cli import parse_set_overrides
        out = parse_set_overrides([
            "think=false", "num_ctx=8192", "temperature=0.7", "label=hello",
        ])
        assert out == {
            "think": False, "num_ctx": 8192, "temperature": 0.7, "label": "hello",
        }

    def test_null_value_parses_to_none(self):
        from porchbench.cli import parse_set_overrides
        out = parse_set_overrides(["think=null"])
        assert out == {"think": None}

    def test_value_with_equals_preserves_remainder(self):
        from porchbench.cli import parse_set_overrides
        # partition on first '=' so values can themselves contain '='
        out = parse_set_overrides(["custom=a=b=c"])
        assert out == {"custom": "a=b=c"}

    def test_missing_equals_raises(self):
        from porchbench.cli import parse_set_overrides
        import typer
        with pytest.raises(typer.BadParameter):
            parse_set_overrides(["thinkfalse"])

    def test_empty_key_raises(self):
        from porchbench.cli import parse_set_overrides
        import typer
        with pytest.raises(typer.BadParameter):
            parse_set_overrides(["=false"])


def test_run_with_set_override_lands_in_resolved_options(
    mocked_run_environment: RunResult, tmp_path: Path,
):
    """`--set think=false` must reach the suite's defaults.options before run_suite sees it."""
    captured: dict = {}

    async def _capture_suite(*args, **kwargs):
        # run_suite receives the overridden suite — capture its defaults to assert on
        captured["suite"] = kwargs.get("suite") or args[0]
        return mocked_run_environment

    with (
        patch("porchbench.cli.run_suite", new=_capture_suite),
        patch("porchbench.cli.construct_backend", return_value=MagicMock()),
        patch("porchbench.cli.check_server_or_exit"),
        patch("porchbench.cli.check_models_or_exit"),
    ):
        res = runner.invoke(
            app,
            [
                "run",
                "--suite", "coding-basics",
                "--model", "qwen2.5:3b",
                "--output-dir", str(tmp_path),
                "--set", "think=false",
                "--set", "num_ctx=8192",
            ],
        )

    assert res.exit_code == 0, res.output
    suite = captured["suite"]
    assert suite.defaults.options.think is False
    assert suite.defaults.options.num_ctx == 8192

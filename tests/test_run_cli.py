"""Tests for the `porchbench run` CLI — particularly the post-phase --evaluate hook."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# Evaluator-model resolution: explicit > env-default > picker (ollama)
# ---------------------------------------------------------------------------


class TestResolveEvalModelOrExit:
    def test_explicit_model_wins_without_touching_backend(self):
        from porchbench.cli import resolve_eval_model_or_exit
        out = resolve_eval_model_or_exit(
            "ollama", "user-pick", backend=None, interactive=True,
        )
        assert out == "user-pick"

    def test_api_backend_returns_stable_default(self):
        from porchbench.cli import resolve_eval_model_or_exit
        out = resolve_eval_model_or_exit(
            "api", None, backend=None, interactive=False,
        )
        assert out == "claude-sonnet-4-6"

    def test_claude_code_backend_returns_stable_default(self):
        from porchbench.cli import resolve_eval_model_or_exit
        out = resolve_eval_model_or_exit(
            "claude-code", None, backend=None, interactive=False,
        )
        assert out == "sonnet"

    def test_ollama_no_model_non_interactive_exits(self):
        """--yes (or any non-interactive flow) must hard-fail rather than try to prompt."""
        import typer

        from porchbench.cli import resolve_eval_model_or_exit
        with pytest.raises(typer.Exit) as exc_info:
            resolve_eval_model_or_exit(
                "ollama", None, backend=MagicMock(), interactive=False,
            )
        assert exc_info.value.exit_code == 1

    def test_ollama_interactive_persists_choice_when_confirmed(self, tmp_path: Path, monkeypatch):
        from porchbench.backend import OllamaBackend
        from porchbench.cli import resolve_eval_model_or_exit

        monkeypatch.chdir(tmp_path)
        fake_backend = MagicMock(spec=OllamaBackend)

        with (
            patch(
                "porchbench.interactive.select_evaluator_model",
                return_value="my-judge:8b",
            ),
            patch("typer.confirm", return_value=True),
        ):
            out = resolve_eval_model_or_exit(
                "ollama", None, backend=fake_backend, interactive=True,
            )

        assert out == "my-judge:8b"
        env_text = (tmp_path / ".env").read_text()
        assert "PORCHBENCH_EVAL_MODEL=my-judge:8b" in env_text

    def test_ollama_interactive_skips_persist_when_declined(self, tmp_path: Path, monkeypatch):
        from porchbench.backend import OllamaBackend
        from porchbench.cli import resolve_eval_model_or_exit

        monkeypatch.chdir(tmp_path)
        fake_backend = MagicMock(spec=OllamaBackend)

        with (
            patch(
                "porchbench.interactive.select_evaluator_model",
                return_value="my-judge:8b",
            ),
            patch("typer.confirm", return_value=False),
        ):
            out = resolve_eval_model_or_exit(
                "ollama", None, backend=fake_backend, interactive=True,
            )

        assert out == "my-judge:8b"
        # User declined — .env should not have been created.
        assert not (tmp_path / ".env").exists()

    def test_ollama_non_ollama_backend_with_no_model_fails(self):
        """If backend isn't OllamaBackend (e.g. openai-compat), the picker can't help."""
        import typer

        from porchbench.cli import resolve_eval_model_or_exit
        with pytest.raises(typer.Exit) as exc_info:
            resolve_eval_model_or_exit(
                "ollama", None, backend=MagicMock(), interactive=True,
            )
        assert exc_info.value.exit_code == 1


class TestDiscoverResultFiles:
    """Picker scans `results/` and must skip non-RunResult JSONs (profiles, analyses)."""

    def test_skips_system_profile_json(self, tmp_path: Path):
        import json

        from porchbench.interactive import _discover_result_files

        # A real run-result JSON
        run_result = tmp_path / "2026-04-28_coding_qwen.json"
        run_result.write_text(
            json.dumps({"run": {"model": {"name": "qwen"}, "suite": {"name": "Coding", "version": "1.0"}}}),
            encoding="utf-8",
        )
        # A system-profile JSON (no `run` block)
        profile = tmp_path / "2026-04-28_system-profile.json"
        profile.write_text(
            json.dumps({"timestamp": "2026-04-28", "models": {}, "gpu": {}}),
            encoding="utf-8",
        )
        # A routing-analysis JSON (also no `run` block)
        analysis = tmp_path / "2026-04-28_routing-analysis.json"
        analysis.write_text(
            json.dumps({"models_tested": ["qwen"], "headline": {}, "patterns": []}),
            encoding="utf-8",
        )

        entries = _discover_result_files(tmp_path)

        assert len(entries) == 1
        label, path = entries[0]
        assert path == run_result
        assert "qwen" in label
        assert "?" not in label

    def test_skips_files_missing_required_fields(self, tmp_path: Path):
        """Defense in depth: a parseable JSON with empty model/suite is also skipped."""
        import json

        from porchbench.interactive import _discover_result_files

        bad = tmp_path / "incomplete.json"
        bad.write_text(json.dumps({"run": {"model": {}, "suite": {}}}), encoding="utf-8")

        assert _discover_result_files(tmp_path) == []


class TestSelectModelsEmptyServer:
    def test_empty_model_list_exits_with_pull_hint(self):
        """Server up but no models pulled → exit, don't drop into manual entry."""
        import typer

        from porchbench.interactive import select_models

        backend = MagicMock()
        backend.list_available_models = AsyncMock(return_value=[])
        with pytest.raises(typer.Exit) as exc_info:
            select_models(backend)
        assert exc_info.value.exit_code == 1

    def test_listing_exception_falls_back_to_manual_entry(self):
        """Backend can't list (e.g. some openai-compat servers) → manual prompt path."""
        from porchbench.interactive import select_models

        backend = MagicMock()
        backend.list_available_models = AsyncMock(side_effect=RuntimeError("no listing"))
        with patch("porchbench.interactive._prompt_models_manually", return_value=["m"]):
            out = select_models(backend)
        assert out == ["m"]

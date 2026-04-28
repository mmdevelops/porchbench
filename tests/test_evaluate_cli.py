"""Tests for the `porchbench evaluate` CLI — batch flow, skip-scored, error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from porchbench.cli import app


@pytest.fixture(autouse=True)
def _stub_evaluator_resolution():
    """Skip the eval-model picker and ollama model preflight in CLI tests.

    These tests exercise the evaluate command's orchestration around scoring,
    not the model-resolution UX (which has its own dedicated tests). Without
    this stub, every test invocation would hang in the interactive picker
    because no Ollama server is running.
    """
    with (
        patch("porchbench.cli.resolve_eval_model_or_exit", return_value="stub-judge"),
        patch("porchbench.cli.check_models_or_exit"),
    ):
        yield
from porchbench.schemas import (
    AggregateScores,
    Criterion,
    EvaluationMetadata,
    Message,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    Rubric,
    RubricMetadata,
    RunMetadata,
    RunResult,
    RunSummary,
    Scorecard,
    SuiteReference,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run_result(run_id: str, model: str = "m", suite_rubric: str | None = "reasoning") -> RunResult:
    return RunResult(
        run=RunMetadata(
            id=run_id,
            suite=SuiteReference(
                name="T", version="1.0", file="s.yaml", sha256="x", rubric=suite_rubric,
            ),
            model=ModelInfo(name=model),
        ),
        results=[
            PromptResult(
                prompt_id="p1",
                category="reasoning",
                difficulty="easy",
                options_used=ModelOptions(),
                request=RequestData(messages=[Message(role="user", content="q")]),
                response=ResponseData(message=ResponseMessage(content="a")),
                metrics=PromptMetrics(),
            )
        ],
        summary=RunSummary(total_prompts=1, completed=1, failed=0, total_duration_s=1.0),
    )


def _make_scorecard(run_id: str) -> Scorecard:
    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_id,
            evaluator="ollama/gemma4:e4b",
            rubric="test-rubric v1.0",
            model_name="m",
            suite_name="T",
        ),
        scores=[],
        aggregate=AggregateScores(overall_weighted=3.5),
    )


def _make_fake_rubric() -> Rubric:
    return Rubric(
        rubric=RubricMetadata(name="fake", version="1.0"),
        criteria=[Criterion(name="correctness", weight=1.0, description="Is it right?")],
    )


def _write_result(path: Path, run_result: RunResult) -> None:
    path.write_text(run_result.model_dump_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Glob-pattern compatibility with write_scorecard's filename format
# ---------------------------------------------------------------------------


def test_skip_scored_glob_matches_write_scorecard_filename(tmp_path: Path):
    """The --skip-scored glob (*_<run_id[:8]>.json) must match real scorecard filenames."""
    from porchbench.evaluator import write_scorecard

    run_id = "abcdef1234567890-rest-of-uuid"
    sc = _make_scorecard(run_id)

    written = write_scorecard(sc, tmp_path)

    assert written.parent == tmp_path
    assert written.exists()
    matches = list(tmp_path.glob(f"*_{run_id[:8]}.json"))
    assert matches == [written], (
        f"glob '*_{run_id[:8]}.json' failed to match scorecard '{written.name}'. "
        "If write_scorecard's naming convention changed, --skip-scored detection is broken."
    )


def test_skip_scored_glob_does_not_match_different_run(tmp_path: Path):
    """Scorecards from other runs must not trigger the skip."""
    from porchbench.evaluator import write_scorecard

    write_scorecard(_make_scorecard("aaaaaaaa-other-run"), tmp_path)
    matches = list(tmp_path.glob("*_bbbbbbbb.json"))
    assert matches == []


# ---------------------------------------------------------------------------
# 2. CLI: single-result path preserves detailed output
# ---------------------------------------------------------------------------


def test_single_result_prints_detailed_aggregate_table(tmp_path: Path):
    result_path = tmp_path / "r.json"
    _write_result(result_path, _make_run_result("11111111-aaaa-bbbb-cccc-dddddddddddd"))

    scorecards_dir = tmp_path / "scorecards"

    with (
        patch("porchbench.evaluator.evaluate_run", new=AsyncMock(
            return_value=_make_scorecard("11111111-aaaa-bbbb-cccc-dddddddddddd")
        )),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        res = runner.invoke(app, [
            "evaluate",
            "-r", str(result_path),
            "--output-dir", str(scorecards_dir),
            "--backend", "ollama",
        ])

    assert res.exit_code == 0, res.output
    assert "Aggregate Scores" in res.output
    assert "Overall" in res.output
    assert "Scorecard written to" in res.output
    # Batch summary block should NOT appear for single-result invocations
    assert "Batch Evaluation Summary" not in res.output


# ---------------------------------------------------------------------------
# 3. CLI: batch flow — multiple results, shared setup, per-result status line
# ---------------------------------------------------------------------------


def test_batch_evaluate_scores_all_results(tmp_path: Path):
    run_ids = [
        "11111111-aaaa-bbbb-cccc-dddddddddddd",
        "22222222-aaaa-bbbb-cccc-dddddddddddd",
        "33333333-aaaa-bbbb-cccc-dddddddddddd",
    ]
    paths = []
    for rid in run_ids:
        p = tmp_path / f"{rid[:8]}.json"
        _write_result(p, _make_run_result(rid))
        paths.append(p)

    scorecards_dir = tmp_path / "scorecards"

    def _fake_eval(run_result, *args, **kwargs):
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=AsyncMock(side_effect=_fake_eval)),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        args = ["evaluate", "--output-dir", str(scorecards_dir), "--backend", "ollama"]
        for p in paths:
            args += ["-r", str(p)]
        res = runner.invoke(app, args)

    assert res.exit_code == 0, res.output
    assert "Batch Evaluation Summary" in res.output
    assert "3 scored, 0 skipped, 0 failed" in res.output
    # Each run should have produced a scorecard file
    assert len(list(scorecards_dir.glob("*.json"))) == 3


# ---------------------------------------------------------------------------
# 4. CLI: batch continues on per-result failure; exits nonzero
# ---------------------------------------------------------------------------


def test_batch_continues_on_failure_and_exits_nonzero(tmp_path: Path):
    good_id = "11111111-aaaa-bbbb-cccc-dddddddddddd"
    bad_id = "22222222-aaaa-bbbb-cccc-dddddddddddd"
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    _write_result(good, _make_run_result(good_id))
    _write_result(bad, _make_run_result(bad_id))

    scorecards_dir = tmp_path / "scorecards"

    async def _flaky_eval(run_result, *args, **kwargs):
        if run_result.run.id == bad_id:
            raise RuntimeError("evaluator unreachable")
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=_flaky_eval),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        res = runner.invoke(app, [
            "evaluate",
            "-r", str(good), "-r", str(bad),
            "--output-dir", str(scorecards_dir),
            "--backend", "ollama",
        ])

    assert res.exit_code == 1, res.output
    assert "evaluation failed" in res.output
    assert "1 scored, 0 skipped, 1 failed" in res.output
    # The good one still produced a scorecard
    assert list(scorecards_dir.glob(f"*_{good_id[:8]}.json"))
    assert not list(scorecards_dir.glob(f"*_{bad_id[:8]}.json"))


# ---------------------------------------------------------------------------
# 5. CLI: --skip-scored skips results with an existing scorecard
# ---------------------------------------------------------------------------


def test_skip_scored_flag_skips_existing(tmp_path: Path):
    from porchbench.evaluator import write_scorecard

    scored_id = "11111111-aaaa-bbbb-cccc-dddddddddddd"
    fresh_id = "22222222-aaaa-bbbb-cccc-dddddddddddd"
    scored_path = tmp_path / "scored.json"
    fresh_path = tmp_path / "fresh.json"
    _write_result(scored_path, _make_run_result(scored_id))
    _write_result(fresh_path, _make_run_result(fresh_id))

    scorecards_dir = tmp_path / "scorecards"
    # Pre-seed a scorecard for the "already scored" run
    write_scorecard(_make_scorecard(scored_id), scorecards_dir)

    call_count = {"n": 0}

    async def _counting_eval(run_result, *args, **kwargs):
        call_count["n"] += 1
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=_counting_eval),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        res = runner.invoke(app, [
            "evaluate",
            "-r", str(scored_path), "-r", str(fresh_path),
            "--output-dir", str(scorecards_dir),
            "--backend", "ollama",
            "--skip-scored",
        ])

    assert res.exit_code == 0, res.output
    assert call_count["n"] == 1, f"evaluate_run should have run only for the fresh result, got {call_count['n']} calls"
    assert "1 scored, 1 skipped, 0 failed" in res.output
    assert "skipped (scorecard exists" in res.output


# ---------------------------------------------------------------------------
# 6. CLI: rubric cache — same rubric path loaded only once across results
# ---------------------------------------------------------------------------


def test_rubric_cache_avoids_duplicate_loads(tmp_path: Path):
    run_ids = [
        "11111111-aaaa-bbbb-cccc-dddddddddddd",
        "22222222-aaaa-bbbb-cccc-dddddddddddd",
        "33333333-aaaa-bbbb-cccc-dddddddddddd",
    ]
    paths = []
    for rid in run_ids:
        p = tmp_path / f"{rid[:8]}.json"
        # All share the same suite rubric hint → same resolved rubric path
        _write_result(p, _make_run_result(rid, suite_rubric="reasoning"))
        paths.append(p)

    scorecards_dir = tmp_path / "scorecards"
    rubric_file = tmp_path / "rubric.yaml"

    load_rubric_mock = MagicMock(return_value=_make_fake_rubric())

    async def _fake_eval(run_result, *args, **kwargs):
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=_fake_eval),
        patch("porchbench.evaluator.load_rubric", new=load_rubric_mock),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=rubric_file),
    ):
        args = ["evaluate", "--output-dir", str(scorecards_dir), "--backend", "ollama"]
        for p in paths:
            args += ["-r", str(p)]
        res = runner.invoke(app, args)

    assert res.exit_code == 0, res.output
    assert load_rubric_mock.call_count == 1, (
        f"load_rubric should be called once across 3 shared-rubric results, "
        f"got {load_rubric_mock.call_count} calls"
    )


# ---------------------------------------------------------------------------
# 7. CLI: bad result file fails that result but doesn't abort the batch
# ---------------------------------------------------------------------------


def test_batch_survives_corrupt_result_file(tmp_path: Path):
    good_id = "11111111-aaaa-bbbb-cccc-dddddddddddd"
    good = tmp_path / "good.json"
    bad = tmp_path / "corrupt.json"
    _write_result(good, _make_run_result(good_id))
    bad.write_text("{ this is not valid json", encoding="utf-8")

    scorecards_dir = tmp_path / "scorecards"

    async def _fake_eval(run_result, *args, **kwargs):
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=_fake_eval),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        res = runner.invoke(app, [
            "evaluate",
            "-r", str(bad), "-r", str(good),
            "--output-dir", str(scorecards_dir),
            "--backend", "ollama",
        ])

    assert res.exit_code == 1, res.output
    assert "load failed" in res.output
    assert "1 scored, 0 skipped, 1 failed" in res.output
    assert list(scorecards_dir.glob(f"*_{good_id[:8]}.json"))


# ---------------------------------------------------------------------------
# 8. Positional path argument — `porchbench evaluate path1 path2 ...` without `-r`
# ---------------------------------------------------------------------------


def test_positional_paths_are_accepted(tmp_path: Path):
    """Positional paths should flow into the evaluation list just like -r flags.
    This is the shell-glob-friendly form (`porchbench evaluate results/*.json`)."""
    run_ids = [
        "11111111-aaaa-bbbb-cccc-dddddddddddd",
        "22222222-aaaa-bbbb-cccc-dddddddddddd",
    ]
    paths = []
    for rid in run_ids:
        p = tmp_path / f"{rid[:8]}.json"
        _write_result(p, _make_run_result(rid))
        paths.append(p)

    scorecards_dir = tmp_path / "scorecards"

    def _fake_eval(run_result, *args, **kwargs):
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=AsyncMock(side_effect=_fake_eval)),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        res = runner.invoke(app, [
            "evaluate",
            *(str(p) for p in paths),  # positional, no -r
            "--output-dir", str(scorecards_dir),
            "--backend", "ollama",
        ])

    assert res.exit_code == 0, res.output
    assert "2 scored, 0 skipped, 0 failed" in res.output


def test_positional_and_flag_paths_compose(tmp_path: Path):
    """Mixing `-r` and positional paths in one invocation merges both sets.
    Supports the hybrid case: some paths glob-expanded, one or two added explicitly."""
    run_ids = [
        "11111111-aaaa-bbbb-cccc-dddddddddddd",
        "22222222-aaaa-bbbb-cccc-dddddddddddd",
    ]
    positional, via_flag = [], []
    for rid in run_ids:
        p = tmp_path / f"{rid[:8]}.json"
        _write_result(p, _make_run_result(rid))
        if rid.startswith("11"):
            positional.append(p)
        else:
            via_flag.append(p)

    scorecards_dir = tmp_path / "scorecards"

    def _fake_eval(run_result, *args, **kwargs):
        return _make_scorecard(run_result.run.id)

    with (
        patch("porchbench.evaluator.evaluate_run", new=AsyncMock(side_effect=_fake_eval)),
        patch("porchbench.evaluator.load_rubric", return_value=_make_fake_rubric()),
        patch("porchbench.evaluator.load_calibration_examples", return_value={}),
        patch("porchbench.assets.find_rubric", return_value=tmp_path / "rubric.yaml"),
    ):
        args = ["evaluate"]
        args += [str(p) for p in positional]
        for p in via_flag:
            args += ["-r", str(p)]
        args += ["--output-dir", str(scorecards_dir), "--backend", "ollama"]
        res = runner.invoke(app, args)

    assert res.exit_code == 0, res.output
    assert "2 scored, 0 skipped, 0 failed" in res.output

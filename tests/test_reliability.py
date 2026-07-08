"""Tests for judge-reliability analysis (matrix building, report, CLI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from porchbench.cli import app
from porchbench.reliability import (
    analyze_matrix,
    matrix_from_samples,
    matrix_from_scorecards,
)
from porchbench.schemas import (
    AggregateScores,
    CriterionScore,
    EvaluationMetadata,
    JudgeSample,
    PromptScore,
    Scorecard,
)


def _make_scorecard(
    scores: list[PromptScore],
    run_id: str = "run-12345678",
) -> Scorecard:
    return Scorecard(
        evaluation=EvaluationMetadata(
            run_id=run_id,
            evaluator="ollama/test-judge",
            rubric="test v1",
            model_name="test-model",
        ),
        scores=scores,
        aggregate=AggregateScores(overall_weighted=4.0),
    )


def _sampled_score(
    prompt_id: str,
    weighted: list[float],
    correctness: list[float] | None = None,
) -> PromptScore:
    samples = []
    for i, w in enumerate(weighted):
        samples.append(
            JudgeSample(
                seed=42 + i,
                temperature=0.3,
                weighted_score=w,
                criteria_scores={"correctness": (correctness or weighted)[i]},
            )
        )
    return PromptScore(
        prompt_id=prompt_id,
        criteria={"correctness": CriterionScore(score=sum(weighted) / len(weighted), rationale="")},
        weighted_score=sum(weighted) / len(weighted),
        summary="",
        samples=samples,
    )


def _plain_score(prompt_id: str, weighted: float) -> PromptScore:
    return PromptScore(
        prompt_id=prompt_id,
        criteria={"correctness": CriterionScore(score=weighted, rationale="")},
        weighted_score=weighted,
        summary="",
    )


class TestMatrixFromSamples:
    def test_builds_matrix(self):
        sc = _make_scorecard(
            [
                _sampled_score("p1", [4.0, 4.5, 4.2]),
                _sampled_score("p2", [2.0, 2.5, 2.2]),
            ]
        )
        matrix, ids, excluded = matrix_from_samples(sc)
        assert ids == ["p1", "p2"]
        assert matrix == [[4.0, 4.5, 4.2], [2.0, 2.5, 2.2]]
        assert excluded == []

    def test_excludes_sampleless_prompts(self):
        sc = _make_scorecard(
            [
                _sampled_score("p1", [4.0, 4.5]),
                _plain_score("p2", 0.0),  # zero-scored truncation: no samples
            ]
        )
        matrix, ids, excluded = matrix_from_samples(sc)
        assert ids == ["p1"]
        assert excluded == ["p2"]

    def test_criterion_matrix(self):
        sc = _make_scorecard(
            [_sampled_score("p1", [4.0, 5.0], correctness=[3.0, 5.0])]
        )
        matrix, _, _ = matrix_from_samples(sc, criterion="correctness")
        assert matrix == [[3.0, 5.0]]


class TestMatrixFromScorecards:
    def test_each_scorecard_is_a_rater(self):
        a = _make_scorecard([_plain_score("p1", 4.0), _plain_score("p2", 3.0)])
        b = _make_scorecard([_plain_score("p1", 4.4), _plain_score("p2", 2.8)])
        matrix, ids, excluded = matrix_from_scorecards([a, b])
        assert ids == ["p1", "p2"]
        assert matrix == [[4.0, 4.4], [3.0, 2.8]]
        assert excluded == []

    def test_prompt_missing_from_one_pass_is_excluded(self):
        a = _make_scorecard([_plain_score("p1", 4.0), _plain_score("p2", 3.0)])
        b = _make_scorecard([_plain_score("p1", 4.4)])
        matrix, ids, excluded = matrix_from_scorecards([a, b])
        assert ids == ["p1"]
        assert excluded == ["p2"]


class TestAnalyzeMatrix:
    def test_gate_pass_on_reliable_spread_data(self):
        # Wide between-prompt spread, tight within-prompt agreement -> high ICC.
        matrix = [
            [1.0, 1.1, 0.9],
            [2.0, 2.1, 1.9],
            [3.0, 3.1, 2.9],
            [4.0, 4.1, 3.9],
            [5.0, 5.1, 4.9],
            [1.5, 1.6, 1.4],
            [3.5, 3.6, 3.4],
            [4.5, 4.6, 4.4],
        ]
        report = analyze_matrix(matrix, excluded=[])
        assert report.icc is not None
        assert not report.icc.degenerate
        assert report.icc.icc_single > 0.9
        assert report.gate_passed is True

    def test_gate_undecidable_on_degenerate(self):
        matrix = [[5.0, 5.0], [5.0, 5.0]]
        report = analyze_matrix(matrix, excluded=[])
        assert report.icc is not None
        assert report.icc.degenerate
        assert report.gate_passed is None

    def test_companions_computed(self):
        matrix = [[4.0, 4.5], [3.0, 3.2]]
        report = analyze_matrix(matrix, excluded=[])
        assert report.pct_within_tol == 1.0
        assert report.mae_across_samples == pytest.approx((0.25 + 0.25 + 0.1 + 0.1) / 4)
        assert report.between_prompt_sd is not None


class TestReliabilityCLI:
    def test_per_sample_scorecard_report(self, tmp_path: Path):
        sc = _make_scorecard(
            [
                _sampled_score("p1", [1.0, 1.1, 0.9]),
                _sampled_score("p2", [4.0, 4.1, 3.9]),
                _sampled_score("p3", [3.0, 3.1, 2.9]),
            ]
        )
        p = tmp_path / "scorecard.json"
        p.write_text(sc.model_dump_json(), encoding="utf-8")

        result = CliRunner().invoke(app, ["reliability", str(p)])
        assert result.exit_code == 0
        assert "ICC(A,1)" in result.output
        assert "Gate" in result.output

    def test_single_pass_scorecard_errors_helpfully(self, tmp_path: Path):
        sc = _make_scorecard([_plain_score("p1", 4.0), _plain_score("p2", 3.0)])
        p = tmp_path / "scorecard.json"
        p.write_text(sc.model_dump_json(), encoding="utf-8")

        result = CliRunner().invoke(app, ["reliability", str(p)])
        assert result.exit_code == 1
        assert "judge-samples" in result.output

    def test_cross_scorecard_mode(self, tmp_path: Path):
        a = _make_scorecard([_plain_score("p1", 4.0), _plain_score("p2", 3.0)])
        b = _make_scorecard([_plain_score("p1", 4.2), _plain_score("p2", 2.9)])
        pa, pb = tmp_path / "a.json", tmp_path / "b.json"
        pa.write_text(a.model_dump_json(), encoding="utf-8")
        pb.write_text(b.model_dump_json(), encoding="utf-8")

        result = CliRunner().invoke(app, ["reliability", str(pa), str(pb)])
        assert result.exit_code == 0
        assert "2 passes" in result.output

    def test_cross_scorecard_different_runs_refused(self, tmp_path: Path):
        a = _make_scorecard([_plain_score("p1", 4.0)], run_id="run-aaaa")
        b = _make_scorecard([_plain_score("p1", 4.2)], run_id="run-bbbb")
        pa, pb = tmp_path / "a.json", tmp_path / "b.json"
        pa.write_text(a.model_dump_json(), encoding="utf-8")
        pb.write_text(b.model_dump_json(), encoding="utf-8")

        result = CliRunner().invoke(app, ["reliability", str(pa), str(pb)])
        assert result.exit_code == 1
        assert "same run" in result.output

"""Tests for overnight orchestration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from porchbench.overnight import (
    OvernightTask,
    _seconds_per_prompt_from_history,
    build_plan,
    estimate_duration,
    estimate_duration_from_history,
    estimate_single_suite_duration_from_history,
    format_estimate,
)
from porchbench.schemas import (
    Message,
    ModelInfo,
    ModelOptions,
    Prompt,
    PromptMetrics,
    PromptResult,
    RequestData,
    ResponseData,
    ResponseMessage,
    RunMetadata,
    RunResult,
    RunSummary,
    Strategy,
    Suite,
    SuiteDefaults,
    SuiteMetadata,
    SuiteReference,
)
from porchbench.suite import discover_suites


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
    def test_baseline_suite_plan(self, mock_ref, mock_load):
        suite = _make_suite("coding", 10)
        mock_load.return_value = suite
        mock_ref.return_value = MagicMock()

        plan = build_plan([Path("suites/coding-basics.yaml")], ["model-a", "model-b"], repeats=3)

        assert len(plan) == 1
        task = plan[0]
        assert task.expand_strategies is False
        assert task.repeats == 3
        assert task.run_count == 10 * 2 * 3  # prompts * models * repeats

    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_strategies_suite_baseline_when_flag_off(self, mock_ref, mock_load):
        # Suite has strategies, but caller didn't pass expand_strategies=True.
        # Baseline by default — strategies stay dormant. Mirrors the v0.1
        # default-to-safe stance: expansion is opt-in.
        suite = _make_suite("routing", 20, strategies={
            "universal": Strategy(),
            "brevity": Strategy(system_message="Brief."),
            "cot": Strategy(system_message="Think step by step."),
        })
        mock_load.return_value = suite
        mock_ref.return_value = MagicMock()

        plan = build_plan([Path("suites/routing-discovery.yaml")], ["model-a"], repeats=3)

        assert plan[0].expand_strategies is False
        assert plan[0].run_count == 20 * 1 * 3  # baseline shape, not the matrix

    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_strategies_suite_matrix_when_flag_on(self, mock_ref, mock_load):
        suite = _make_suite("routing", 20, strategies={
            "universal": Strategy(),
            "brevity": Strategy(system_message="Brief."),
            "cot": Strategy(system_message="Think step by step."),
        })
        mock_load.return_value = suite
        mock_ref.return_value = MagicMock()

        plan = build_plan(
            [Path("suites/routing-discovery.yaml")], ["model-a"], repeats=3,
            expand_strategies=True,
        )

        task = plan[0]
        assert task.expand_strategies is True
        assert task.repeats == 1  # matrix expansion replaces repeats
        assert task.strategy_count == 3
        assert task.run_count == 20 * 3 * 1  # prompts * strategies * models

    @patch("porchbench.overnight.load_suite")
    @patch("porchbench.overnight.make_suite_reference")
    def test_mixed_plan_with_flag_on_only_strategies_suite_expands(
        self, mock_ref, mock_load,
    ):
        # When --strategies is set but a suite has no strategies block,
        # that suite quietly falls back to baseline; the strategies suite
        # still expands. Matches the multi-suite "warn + skip" semantics.
        baseline_suite = _make_suite("coding", 10)
        strategies_suite = _make_suite("routing", 5, strategies={
            "a": Strategy(), "b": Strategy(),
        })

        mock_load.side_effect = [baseline_suite, strategies_suite]
        mock_ref.return_value = MagicMock()

        plan = build_plan(
            [Path("suites/coding-basics.yaml"), Path("suites/routing-discovery.yaml")],
            ["m1", "m2"],
            repeats=2,
            expand_strategies=True,
        )

        assert len(plan) == 2
        assert plan[0].expand_strategies is False  # no strategies block
        assert plan[0].run_count == 10 * 2 * 2  # baseline: prompts * models * repeats
        assert plan[1].expand_strategies is True  # has strategies, flag honored
        assert plan[1].run_count == 5 * 2 * 2  # matrix: prompts * strategies * models


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


# ---------------------------------------------------------------------------
# History-aware duration estimator
# ---------------------------------------------------------------------------


def _write_fake_run(
    results_dir: Path, suite_name: str, model: str,
    per_prompt_seconds: list[float], timestamp: str = "2026-04-25T10-00-00",
) -> Path:
    """Write a RunResult JSON shaped to match the runner's filename convention."""
    suite_slug = suite_name.lower().replace(" ", "-")
    model_slug = model.replace(":", "-").replace("/", "-")
    path = results_dir / f"{timestamp}_{suite_slug}_{model_slug}.json"

    prompt_results = []
    for i, secs in enumerate(per_prompt_seconds):
        prompt_results.append(PromptResult(
            prompt_id=f"p{i}",
            category="coding",
            difficulty="easy",
            options_used=ModelOptions(),
            request=RequestData(messages=[Message(role="user", content="q")]),
            response=ResponseData(message=ResponseMessage(content="a")),
            metrics=PromptMetrics(total_duration=int(secs * 1e9)),
        ))

    rr = RunResult(
        run=RunMetadata(
            suite=SuiteReference(
                name=suite_name, version="1.0",
                file=f"{suite_slug}.yaml", sha256="x",
            ),
            model=ModelInfo(name=model),
        ),
        results=prompt_results,
        summary=RunSummary(
            total_prompts=len(prompt_results),
            completed=len(prompt_results),
            failed=0,
            total_duration_s=sum(per_prompt_seconds),
        ),
    )
    path.write_text(rr.model_dump_json(), encoding="utf-8")
    return path


def _make_task(suite_name: str, models: list[str], run_count: int) -> OvernightTask:
    suite = Suite(
        suite=SuiteMetadata(name=suite_name, version="1.0"),
        defaults=SuiteDefaults(options=ModelOptions()),
        prompts=[Prompt(
            id="p0", category="coding", difficulty="easy",
            messages=[Message(role="user", content="q")],
        )],
    )
    suite_ref = SuiteReference(
        name=suite_name, version="1.0",
        file=f"{suite_name}.yaml", sha256="x",
    )
    return OvernightTask(
        suite_path=Path(f"{suite_name}.yaml"),
        suite=suite,
        suite_ref=suite_ref,
        expand_strategies=False,
        models=models,
        repeats=1,
        prompt_count=run_count // len(models),
        strategy_count=1,
        run_count=run_count,
    )


class TestSecondsPerPromptFromHistory:
    def test_returns_none_when_results_dir_missing(self, tmp_path: Path):
        out = _seconds_per_prompt_from_history("qwen3:8b", "coding-basics", tmp_path / "missing")
        assert out is None

    def test_returns_none_when_no_matching_files(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Other Suite", "qwen3:8b", [10.0])
        out = _seconds_per_prompt_from_history("qwen3:8b", "coding-basics", tmp_path)
        assert out is None

    def test_returns_median_across_prompts(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [10.0, 20.0, 30.0])
        out = _seconds_per_prompt_from_history("qwen3:8b", "coding-basics", tmp_path)
        assert out == 20.0

    def test_aggregates_across_multiple_runs(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [10.0, 20.0],
                        timestamp="2026-04-20T10-00-00")
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [30.0, 40.0],
                        timestamp="2026-04-21T10-00-00")
        out = _seconds_per_prompt_from_history("qwen3:8b", "coding-basics", tmp_path)
        assert out == 25.0  # median of [10, 20, 30, 40]

    def test_falls_back_to_summary_when_per_prompt_missing(self, tmp_path: Path):
        """Tool-use runs leave metrics.total_duration empty.

        Without the fallback the estimator returns None for any past tool-use
        run and users see 'no prior runs' even when timing is available at
        the run-summary level.
        """
        # Build a fake run with empty per-prompt durations but a populated
        # summary.total_duration_s — mirrors what the tool-use harness writes.
        suite_slug = "tool-use-discovery"
        model = "gemma4:e2b"
        model_slug = model.replace(":", "-")
        path = tmp_path / f"2026-04-28T10-00-00_{suite_slug}_{model_slug}.json"
        run_dict = {
            "run": {
                "id": "abc123",
                "timestamp": "2026-04-28T10:00:00Z",
                "suite": {
                    "name": "Tool Use Discovery", "version": "1.0",
                    "file": "x", "sha256": "y",
                },
                "model": {"name": model},
            },
            "results": [
                {
                    "prompt_id": f"p-{i}",
                    "category": "tool-use", "difficulty": "easy",
                    "options_used": {},
                    "request": {"messages": [{"role": "user", "content": "q"}]},
                    "response": {"message": {"role": "assistant", "content": "a"}},
                    "metrics": {},  # all None — tool-use harness leaves these empty
                }
                for i in range(10)
            ],
            "summary": {
                "total_prompts": 10, "completed": 10, "failed": 0,
                "total_duration_s": 200.0,  # ~20s per prompt averaged
            },
        }
        import json

        path.write_text(json.dumps(run_dict), encoding="utf-8")

        rate = _seconds_per_prompt_from_history(model, suite_slug, tmp_path)
        assert rate == 20.0  # 200s / 10 prompts

    def test_skips_malformed_files(self, tmp_path: Path):
        # malformed file matching the glob is skipped; valid one still scored
        (tmp_path / "garbage_coding-basics_qwen3-8b.json").write_text("{not json", encoding="utf-8")
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [15.0])
        out = _seconds_per_prompt_from_history("qwen3:8b", "coding-basics", tmp_path)
        assert out == 15.0


class TestEstimateDurationFromHistory:
    def test_empty_plan_returns_zero_coverage(self, tmp_path: Path):
        total, with_hist, total_calls = estimate_duration_from_history([], tmp_path)
        assert (total, with_hist, total_calls) == (0.0, 0, 0)

    def test_no_history_returns_zero_seconds_with_full_call_count(self, tmp_path: Path):
        task = _make_task("Coding Basics", ["qwen3:8b"], run_count=10)
        total, with_hist, total_calls = estimate_duration_from_history([task], tmp_path)
        assert total == 0.0
        assert with_hist == 0
        assert total_calls == 10

    def test_full_history_sums_per_model_estimate(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [12.0, 12.0, 12.0])
        task = _make_task("Coding Basics", ["qwen3:8b"], run_count=5)
        total, with_hist, total_calls = estimate_duration_from_history([task], tmp_path)
        assert total == 60.0  # 5 calls * 12s median
        assert with_hist == 5
        assert total_calls == 5

    def test_partial_history_flags_uncovered_calls(self, tmp_path: Path):
        # qwen3:8b has history (10s/prompt), llama3.1:8b does not
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [10.0, 10.0])
        task = _make_task("Coding Basics", ["qwen3:8b", "llama3.1:8b"], run_count=10)
        total, with_hist, total_calls = estimate_duration_from_history([task], tmp_path)
        # Per-model split: 5 calls each. qwen has rate, llama doesn't.
        assert total == 50.0  # 5 * 10s for qwen only
        assert with_hist == 5
        assert total_calls == 10


class TestEstimateSingleSuiteDurationFromHistory:
    """`porchbench run`'s estimator — same partial-coverage semantics as overnight's."""

    def test_no_history_reports_zero_coverage_with_full_call_count(self, tmp_path: Path):
        total, with_hist, total_calls = estimate_single_suite_duration_from_history(
            models=["qwen3:8b"],
            suite_name="Coding Basics",
            prompt_count=5,
            repeats=2,
            results_dir=tmp_path,
        )
        assert total == 0.0
        assert with_hist == 0
        assert total_calls == 10  # 5 prompts * 2 repeats

    def test_full_history_uses_median(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [12.0, 12.0, 12.0])
        total, with_hist, total_calls = estimate_single_suite_duration_from_history(
            models=["qwen3:8b"],
            suite_name="Coding Basics",
            prompt_count=4,
            repeats=1,
            results_dir=tmp_path,
        )
        assert total == 48.0  # 4 calls * 12s
        assert with_hist == 4
        assert total_calls == 4

    def test_partial_coverage_excludes_no_history_models_from_seconds(self, tmp_path: Path):
        _write_fake_run(tmp_path, "Coding Basics", "qwen3:8b", [10.0, 10.0])
        total, with_hist, total_calls = estimate_single_suite_duration_from_history(
            models=["qwen3:8b", "llama3.1:8b"],
            suite_name="Coding Basics",
            prompt_count=3,
            repeats=1,
            results_dir=tmp_path,
        )
        # qwen has rate (3 calls × 10s); llama doesn't (3 calls excluded from seconds)
        assert total == 30.0
        assert with_hist == 3
        assert total_calls == 6


class TestCheckVramCofit:
    """Pre-flight cofit check warns before users discover swap-thrash mid-run."""

    @pytest.mark.asyncio
    async def test_cofit_fits_in_vram(self):
        """qwen2.5:3b (2 GB) + gemma4:e2b (2 GB) + 1.5 GB headroom < 16 GB."""
        import asyncio as _asyncio
        from porchbench.backend import OllamaBackend
        from porchbench.overnight import check_vram_cofit

        backend = MagicMock(spec=OllamaBackend)

        async def _fake_size(_backend, model):
            return {"qwen2.5:3b": 2 * 1024**3, "gemma4:e2b": 2 * 1024**3}[model]

        with (
            patch("porchbench.overnight.detect_gpu", return_value=("Test GPU", 16.0)),
            patch("porchbench.overnight._get_ollama_model_size_bytes", side_effect=_fake_size),
        ):
            ok, msg = await check_vram_cofit(backend, ["qwen2.5:3b"], "gemma4:e2b")

        assert ok is True
        assert "fit in VRAM" in msg

    @pytest.mark.asyncio
    async def test_cofit_fails_on_16gb_with_qwen_and_gemma(self):
        """qwen3:8b (6 GB) + gemma4:e4b (9 GB) + 1.5 GB headroom = 16.5 GB > 15.9 GB.
        Reproduces the exact scenario the user hit that motivated this check."""
        from porchbench.backend import OllamaBackend
        from porchbench.overnight import check_vram_cofit

        backend = MagicMock(spec=OllamaBackend)

        async def _fake_size(_backend, model):
            return {"qwen3:8b": 6 * 1024**3, "gemma4:e4b": 9 * 1024**3}[model]

        with (
            patch("porchbench.overnight.detect_gpu", return_value=("RX 9070 XT", 15.9)),
            patch("porchbench.overnight._get_ollama_model_size_bytes", side_effect=_fake_size),
        ):
            ok, msg = await check_vram_cofit(backend, ["qwen3:8b"], "gemma4:e4b")

        assert ok is False
        assert "don't cofit" in msg
        assert "claude-code" in msg or "api" in msg  # suggests an off-GPU mitigation

    @pytest.mark.asyncio
    async def test_picks_largest_target_for_worst_case(self):
        """With multiple target models, check uses the largest against eval."""
        from porchbench.backend import OllamaBackend
        from porchbench.overnight import check_vram_cofit

        backend = MagicMock(spec=OllamaBackend)

        async def _fake_size(_backend, model):
            return {
                "small:3b": 2 * 1024**3,
                "big:14b": 9 * 1024**3,
                "judge": 3 * 1024**3,
            }[model]

        with (
            patch("porchbench.overnight.detect_gpu", return_value=("Test GPU", 12.0)),
            patch("porchbench.overnight._get_ollama_model_size_bytes", side_effect=_fake_size),
        ):
            # big:14b (9) + judge (3) + 1.5 = 13.5 > 12.0 → fail
            ok, msg = await check_vram_cofit(backend, ["small:3b", "big:14b"], "judge")

        assert ok is False
        assert "big:14b" in msg  # worst-case model is the one cited

    @pytest.mark.asyncio
    async def test_skips_when_vram_unknown(self):
        from porchbench.backend import OllamaBackend
        from porchbench.overnight import check_vram_cofit

        backend = MagicMock(spec=OllamaBackend)

        with patch("porchbench.overnight.detect_gpu", return_value=("Test GPU", None)):
            ok, msg = await check_vram_cofit(backend, ["qwen2.5:3b"], "gemma4:e4b")

        assert ok is True
        assert "VRAM unknown" in msg

    @pytest.mark.asyncio
    async def test_skips_for_non_ollama_backend(self):
        from porchbench.overnight import check_vram_cofit
        from porchbench.backend import OpenAICompatBackend

        backend = MagicMock(spec=OpenAICompatBackend)
        ok, msg = await check_vram_cofit(backend, ["x"], "y")

        assert ok is True
        assert "not available" in msg

    @pytest.mark.asyncio
    async def test_headroom_scales_with_num_ctx(self):
        """Same models on tight VRAM: fits at 8K context, doesn't fit at 32K.

        Pins the linear KV-cache scaling: at 8K the headroom matches the prior
        fixed 1.5 GB; at 32K (current suite default) it's 4.5 GB, which can
        flip the verdict.
        """
        from porchbench.backend import OllamaBackend
        from porchbench.overnight import check_vram_cofit

        backend = MagicMock(spec=OllamaBackend)

        async def _fake_size(_backend, model):
            return {"target:8b": 2 * 1024**3, "judge": 2 * 1024**3}[model]

        with (
            patch("porchbench.overnight.detect_gpu", return_value=("Test GPU", 6.0)),
            patch("porchbench.overnight._get_ollama_model_size_bytes", side_effect=_fake_size),
        ):
            # 8K: 2 + 2 + 1.5 = 5.5 ≤ 6 → fits
            ok_8k, msg_8k = await check_vram_cofit(
                backend, ["target:8b"], "judge", num_ctx=8192,
            )
            # 32K: 2 + 2 + 4.5 = 8.5 > 6 → doesn't fit
            ok_32k, msg_32k = await check_vram_cofit(
                backend, ["target:8b"], "judge", num_ctx=32768,
            )

        assert ok_8k is True
        assert "num_ctx=8192" in msg_8k
        assert "1.5 GB headroom" in msg_8k

        assert ok_32k is False
        assert "num_ctx=32768" in msg_32k
        assert "4.5 GB headroom" in msg_32k

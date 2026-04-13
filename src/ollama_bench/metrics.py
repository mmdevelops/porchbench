"""Aggregation and summary statistics for benchmark run results.

Provides grouping (by category, difficulty, model) and descriptive stats
(mean, min, max, stdev, median) over the per-prompt metrics captured by
the runner. Used by compare.py and the CLI for analysis output.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from ollama_bench.schemas import PromptResult, RunResult


@dataclass(frozen=True)
class DescriptiveStats:
    """Standard descriptive statistics over a list of numeric values."""

    count: int
    mean: float
    median: float
    stdev: float | None  # None when count < 2
    min: float
    max: float

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "stdev": round(self.stdev, 2) if self.stdev is not None else None,
            "min": round(self.min, 2),
            "max": round(self.max, 2),
        }


def describe(values: list[float]) -> DescriptiveStats | None:
    """Compute descriptive stats for a list of values. Returns None if empty."""
    if not values:
        return None
    return DescriptiveStats(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) >= 2 else None,
        min=min(values),
        max=max(values),
    )


def extract_tokens_per_second(results: list[PromptResult]) -> list[float]:
    """Pull tokens_per_second from results that have it."""
    return [
        r.metrics.tokens_per_second
        for r in results
        if r.metrics.tokens_per_second is not None
    ]


def extract_time_to_first_token(results: list[PromptResult]) -> list[float]:
    """Pull time_to_first_token_ms from results that have it."""
    return [
        r.metrics.time_to_first_token_ms
        for r in results
        if r.metrics.time_to_first_token_ms is not None
    ]


def extract_total_tokens(results: list[PromptResult]) -> list[int]:
    """Pull eval_count (tokens generated) from results that have it."""
    return [
        r.metrics.eval_count
        for r in results
        if r.metrics.eval_count is not None
    ]


def extract_total_duration_s(results: list[PromptResult]) -> list[float]:
    """Pull total_duration converted to seconds from results that have it."""
    return [
        r.metrics.total_duration / 1e9
        for r in results
        if r.metrics.total_duration is not None
    ]


def group_by_category(results: list[PromptResult]) -> dict[str, list[PromptResult]]:
    """Group prompt results by their category field."""
    groups: dict[str, list[PromptResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    return dict(groups)


def group_by_difficulty(results: list[PromptResult]) -> dict[str, list[PromptResult]]:
    """Group prompt results by their difficulty field."""
    groups: dict[str, list[PromptResult]] = defaultdict(list)
    for r in results:
        groups[r.difficulty].append(r)
    return dict(groups)


def summarize_run(run: RunResult) -> dict:
    """Produce a detailed metrics summary for a single run.

    Returns a dict with overall stats and breakdowns by category and difficulty.
    """
    results = run.results

    overall_tps = describe(extract_tokens_per_second(results))
    overall_ttft = describe(extract_time_to_first_token(results))
    overall_tokens = describe([float(t) for t in extract_total_tokens(results)])
    overall_duration = describe(extract_total_duration_s(results))

    by_category = {}
    for cat, group in group_by_category(results).items():
        by_category[cat] = {
            "tokens_per_second": _stats_dict(describe(extract_tokens_per_second(group))),
            "time_to_first_token_ms": _stats_dict(describe(extract_time_to_first_token(group))),
            "count": len(group),
        }

    by_difficulty = {}
    for diff, group in group_by_difficulty(results).items():
        by_difficulty[diff] = {
            "tokens_per_second": _stats_dict(describe(extract_tokens_per_second(group))),
            "time_to_first_token_ms": _stats_dict(describe(extract_time_to_first_token(group))),
            "count": len(group),
        }

    return {
        "model": run.run.model.name,
        "quantization": run.run.model.details.quantization_level,
        "kv_cache_type": run.run.system.kv_cache_type,
        "overall": {
            "tokens_per_second": _stats_dict(overall_tps),
            "time_to_first_token_ms": _stats_dict(overall_ttft),
            "tokens_generated": _stats_dict(overall_tokens),
            "duration_s": _stats_dict(overall_duration),
            "prompts_completed": run.summary.completed,
            "prompts_failed": run.summary.failed,
        },
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }


def _stats_dict(stats: DescriptiveStats | None) -> dict | None:
    if stats is None:
        return None
    return stats.as_dict()

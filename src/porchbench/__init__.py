"""porchbench: reproducible quality benchmarking for local LLMs.

Runs prompt suites against Ollama and OpenAI-compatible backends, scores
responses via LLM-as-judge (local Ollama, Anthropic API, or Claude Code),
and produces paired statistical comparisons across models and quantization
levels.
"""

from porchbench.schemas import (
    Rubric,
    RoutingAnalysis,
    RunResult,
    Scorecard,
    Suite,
    SystemProfile,
)

__version__ = "0.1.0"

__all__ = [
    "Rubric",
    "RoutingAnalysis",
    "RunResult",
    "Scorecard",
    "Suite",
    "SystemProfile",
]

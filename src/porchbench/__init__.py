"""porchbench: reproducible quality benchmarking for local LLMs.

Runs prompt suites against Ollama and OpenAI-compatible backends, scores
responses via LLM-as-judge (local Ollama, Anthropic API, or Claude Code),
and produces paired statistical comparisons across models and quantization
levels.
"""

from porchbench.evaluator import (
    AnthropicEvalBackend,
    ClaudeCodeEvalBackend,
    EvalBackend,
    OllamaEvalBackend,
    batch_evaluate_results,
    batch_evaluate_results_sync,
    evaluate_single,
    evaluate_single_sync,
    make_backend,
)
from porchbench.schemas import (
    PromptScore,
    RoutingAnalysis,
    Rubric,
    RunResult,
    Scorecard,
    Suite,
    SystemProfile,
    slugify_suite_name,
)

__version__ = "0.1.0"

__all__ = [
    "AnthropicEvalBackend",
    "ClaudeCodeEvalBackend",
    "EvalBackend",
    "OllamaEvalBackend",
    "PromptScore",
    "Rubric",
    "RoutingAnalysis",
    "RunResult",
    "Scorecard",
    "Suite",
    "SystemProfile",
    "batch_evaluate_results",
    "batch_evaluate_results_sync",
    "evaluate_single",
    "evaluate_single_sync",
    "make_backend",
    "slugify_suite_name",
]

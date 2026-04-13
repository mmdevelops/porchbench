"""Pydantic models for suite definitions, run results, rubrics, and scorecards.

Covers the full data lifecycle: suite YAML → run result JSON → scorecard JSON.
Schema extension fields from DESIGN-ROUTING.md and DESIGN-SANDBOX.md are included
as optional fields so the validation layer accepts all suite types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message in the Ollama conversation format."""

    role: str  # user, assistant, system, tool
    content: str


class ModelOptions(BaseModel):
    """Ollama inference parameters. Extras are forwarded to the API as-is."""

    model_config = ConfigDict(extra="allow")

    temperature: float = 0
    seed: int = 42
    top_p: float = 1
    num_predict: int = 2048
    num_ctx: int = 4096


# ---------------------------------------------------------------------------
# Suite input schemas (loaded from YAML)
# ---------------------------------------------------------------------------


class SuiteMetadata(BaseModel):
    name: str
    version: str
    description: str = ""
    categories: list[str] = []


class SuiteDefaults(BaseModel):
    options: ModelOptions


class Strategy(BaseModel):
    """A prompt strategy template for routing discovery (DESIGN-ROUTING.md)."""

    system_message: str = ""


class Prompt(BaseModel):
    """A single benchmark prompt with optional extension fields."""

    id: str
    category: str  # coding, reasoning, cross-domain, tool-use, ...
    difficulty: str  # easy, medium, hard
    tags: list[str] = []
    messages: list[Message]
    options: ModelOptions | None = None

    # --- Routing extensions (DESIGN-ROUTING.md) ---
    answer_type: str | None = None  # factual, numeric, code, explanation, open-ended
    reasoning_depth: str | None = None  # shallow, medium, deep
    expected_answer: str | None = None

    # --- Sandbox extensions (DESIGN-SANDBOX.md) — models deferred, typed as Any ---
    mode: str = "text"  # text | tool-use
    tools: list[dict[str, Any]] | None = None
    sandbox: dict[str, Any] | None = None
    setup_files: list[dict[str, str]] | None = None
    expected_outcome: dict[str, Any] | None = None
    max_tool_calls: int | None = None


class Suite(BaseModel):
    """Top-level suite definition, validated from YAML."""

    suite: SuiteMetadata
    defaults: SuiteDefaults
    prompts: list[Prompt]
    strategies: dict[str, Strategy] = {}  # routing discovery extension

    @model_validator(mode="after")
    def validate_unique_prompt_ids(self) -> Suite:
        ids = [p.id for p in self.prompts]
        duplicates = [pid for pid in ids if ids.count(pid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate prompt IDs: {set(duplicates)}")
        return self


# ---------------------------------------------------------------------------
# Run result schemas (written as JSON)
# ---------------------------------------------------------------------------


class SuiteReference(BaseModel):
    """Identifies which suite produced this run, with content hash for reproducibility."""

    name: str
    version: str
    file: str
    sha256: str


class ModelDetails(BaseModel):
    """Model metadata from ollama.show(). All optional because availability varies."""

    format: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class ModelInfo(BaseModel):
    name: str
    details: ModelDetails = Field(default_factory=ModelDetails)


class SystemInfo(BaseModel):
    """Execution environment metadata for reproducibility."""

    ollama_version: str = ""
    gpu: str = ""
    vram_gb: float | None = None
    os: str = ""


class RunMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    suite: SuiteReference
    model: ModelInfo
    system: SystemInfo = Field(default_factory=SystemInfo)


class RequestData(BaseModel):
    messages: list[Message]


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class ResponseData(BaseModel):
    message: ResponseMessage
    done_reason: str | None = None


class PromptMetrics(BaseModel):
    """Raw Ollama timing fields plus computed derivatives.

    Raw fields are nanoseconds (Ollama convention — field names omit the _ns suffix).
    Computed fields are populated by compute_derived_metrics().
    """

    # Raw from Ollama response
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None

    # Computed
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None


def compute_derived_metrics(metrics: PromptMetrics) -> PromptMetrics:
    """Populate computed fields from raw Ollama timing data. Returns a new instance."""
    updates: dict[str, float] = {}

    if metrics.eval_count is not None and metrics.eval_duration and metrics.eval_duration > 0:
        updates["tokens_per_second"] = metrics.eval_count / (metrics.eval_duration / 1e9)

    if metrics.prompt_eval_duration is not None and metrics.prompt_eval_duration > 0:
        updates["time_to_first_token_ms"] = metrics.prompt_eval_duration / 1e6

    if updates:
        return metrics.model_copy(update=updates)
    return metrics


class PromptResult(BaseModel):
    """Result of running a single prompt against a model."""

    prompt_id: str
    category: str
    difficulty: str
    tags: list[str] = []
    options_used: ModelOptions
    request: RequestData
    response: ResponseData
    metrics: PromptMetrics = Field(default_factory=PromptMetrics)


class RunSummary(BaseModel):
    total_prompts: int
    completed: int
    failed: int
    total_duration_s: float
    avg_tokens_per_second: float | None = None


class RunResult(BaseModel):
    """Complete output of a benchmark run. Serialized to results/ as JSON."""

    run: RunMetadata
    results: list[PromptResult]
    summary: RunSummary


# ---------------------------------------------------------------------------
# Rubric schemas (loaded from YAML)
# ---------------------------------------------------------------------------


class RubricMetadata(BaseModel):
    name: str
    version: str


class Criterion(BaseModel):
    name: str
    weight: float
    description: str
    scale: str = "1-5"


class Rubric(BaseModel):
    rubric: RubricMetadata
    criteria: list[Criterion]

    @model_validator(mode="after")
    def validate_weights_sum(self) -> Rubric:
        total = sum(c.weight for c in self.criteria)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Criterion weights sum to {total:.3f}, expected ~1.0")
        return self


# ---------------------------------------------------------------------------
# Scorecard schemas (output of evaluation)
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    score: int
    rationale: str


class PromptScore(BaseModel):
    prompt_id: str
    criteria: dict[str, CriterionScore]
    weighted_score: float
    summary: str


class AggregateScores(BaseModel):
    overall_weighted: float
    by_category: dict[str, float] = {}
    by_difficulty: dict[str, float] = {}


class EvaluationMetadata(BaseModel):
    run_id: str
    evaluator: str
    rubric: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Scorecard(BaseModel):
    """Complete output of an evaluation pass. Serialized to scorecards/ as JSON."""

    evaluation: EvaluationMetadata
    scores: list[PromptScore]
    aggregate: AggregateScores

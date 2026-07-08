"""Pydantic models for suite definitions, run results, rubrics, and scorecards.

Covers the full data lifecycle: suite YAML → run result JSON → scorecard JSON.
Routing-discovery and tool-use/sandbox extension fields are included as optional
attributes so the validation layer accepts all suite types.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
    # Reasoning-mode toggle forwarded to Ollama as the top-level `think` field.
    # Default None leaves it unset, letting Ollama use the model's default
    # (qwen3, deepseek-r1, etc. default to on). Set False to disable thinking
    # when benchmarking — avoids paying for reasoning tokens you discard.
    think: bool | None = None


# ---------------------------------------------------------------------------
# Suite input schemas (loaded from YAML)
# ---------------------------------------------------------------------------


class SuiteMetadata(BaseModel):
    name: str
    version: str
    description: str = ""
    categories: list[str] = []
    rubric: str | None = None  # e.g. "coding", "cross-domain-science" → rubrics/{rubric}.yaml


class SuiteDefaults(BaseModel):
    options: ModelOptions


class Strategy(BaseModel):
    """A prompt strategy template for routing discovery."""

    system_message: str = ""


class Prompt(BaseModel):
    """A single benchmark prompt with optional extension fields."""

    id: str
    category: str  # coding, reasoning, cross-domain, tool-use, ...
    difficulty: str  # easy, medium, hard
    tags: list[str] = []
    messages: list[Message]
    options: ModelOptions | None = None

    # --- Methodology extension (METHODOLOGY.md) ---
    contamination_risk: str | None = None  # high, medium, low

    # --- Routing extensions ---
    answer_type: str | None = None  # factual, numeric, code, explanation, open-ended
    reasoning_depth: str | None = None  # shallow, medium, deep
    expected_answer: str | None = None

    # --- Sandbox extensions — sub-models deferred, typed as Any ---
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


def slugify_suite_name(name: str) -> str:
    """Map a human-readable suite name to its filesystem-safe identifier.

    Used for result-file names and history-lookup keys so callers can match
    runs across both the YAML form ("Tool Use Discovery") and the on-disk
    form ("tool-use-discovery"). Single source of truth — keep all consumers
    routed through this helper or `SuiteReference.slug`.
    """
    return name.lower().replace(" ", "-")


class SuiteReference(BaseModel):
    """Identifies which suite produced this run, with content hash for reproducibility."""

    name: str
    version: str
    file: str
    sha256: str
    rubric: str | None = None  # rubric hint from suite metadata, auto-resolves at eval time

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier — `name.lower().replace(" ", "-")`.

        Stable across the YAML display name and on-disk filename
        conventions; consumers filtering result paths by suite should
        match against `slug` rather than `name`.
        """
        return slugify_suite_name(self.name)


class ModelDetails(BaseModel):
    """Model metadata from ollama.show(). All optional because availability varies."""

    format: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class ModelInfo(BaseModel):
    name: str
    digest: str | None = None  # SHA256 from ollama.show(), detects silent model updates
    details: ModelDetails = Field(default_factory=ModelDetails)


class SystemInfo(BaseModel):
    """Execution environment metadata for reproducibility."""

    ollama_version: str = ""
    gpu: str = ""
    vram_gb: float | None = None
    os: str = ""
    kv_cache_type: str | None = None  # f16, q8_0, q4_0, tq3, tq4 — from OLLAMA_KV_CACHE_TYPE


class RunMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    suite: SuiteReference
    model: ModelInfo
    system: SystemInfo = Field(default_factory=SystemInfo)
    repeat_index: int | None = None  # 1-based repeat number when using --repeats
    total_repeats: int | None = None
    porchbench_version: str | None = None  # set at run time; None on pre-0.1 result files


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

    # VRAM profiling (opt-in via --profile-vram)
    peak_vram_bytes: int | None = None


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


class ToolUseMetricsData(BaseModel):
    """Serializable mirror of harness.ToolUseMetrics for run result JSON."""

    total_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = {}
    errors_encountered: int = 0
    self_corrections: int = 0
    conversation_turns: int = 0
    tool_calls_via_text: int = 0


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

    # --- Methodology extensions ---
    contamination_risk: str | None = None  # high, medium, low — copied from Prompt

    # --- Routing discovery extensions ---
    strategy: str | None = None  # strategy name when run via run --strategies
    correct: bool | None = None  # automated correctness check result
    expected_answer: str | None = None  # ground truth copied from prompt for analysis

    # --- Tool-use extensions ---
    validation_passed: bool | None = None  # sandbox validator result
    validation_reason: str | None = None
    stopped_reason: str | None = None  # done, max_tool_calls, max_turns, error
    tool_use_metrics: ToolUseMetricsData | None = None


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
    """Weighted scoring criteria for LLM-as-judge evaluation, loaded from rubrics/{name}.yaml."""

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
    # float since mean-of-k judge sampling: the aggregated criterion score is
    # the mean across samples (4.6), not a single integer draw. Old integer
    # scorecards validate unchanged.
    score: float
    rationale: str


class JudgeSample(BaseModel):
    """One judge sample's scores within a mean-of-k evaluation pass.

    Compact by design (no per-criterion rationales): the per-sample matrix
    is what `porchbench reliability` computes ICC from, so k samples x N
    prompts must stay cheap to store.
    """

    seed: int
    temperature: float
    weighted_score: float
    criteria_scores: dict[str, float]


class AuditItem(BaseModel):
    """One obligation from the two-phase judge audit (extract, then check)."""

    obligation: str
    met: bool


class PromptScore(BaseModel):
    prompt_id: str
    criteria: dict[str, CriterionScore]
    weighted_score: float
    summary: str
    # Per-sample scores when judged with --judge-samples k > 1 (absent on
    # single-pass scorecards, including all pre-v0.2 files).
    samples: list[JudgeSample] | None = None
    # Obligations audit from the two-phase judge protocol, first sample.
    audit: list[AuditItem] | None = None


class AggregateScores(BaseModel):
    overall_weighted: float
    by_category: dict[str, float] = {}
    by_difficulty: dict[str, float] = {}

    # --- Normalized scoring (METHODOLOGY.md) ---
    overall_normalized: float | None = None  # 0-100 scale where 1→0, 5→100
    by_difficulty_normalized: dict[str, float] = {}

    # --- Contamination-filtered aggregates ---
    overall_weighted_clean: float | None = None  # excludes high-contamination prompts
    by_category_clean: dict[str, float] = {}
    by_difficulty_clean: dict[str, float] = {}


class EvaluationMetadata(BaseModel):
    run_id: str
    evaluator: str
    rubric: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_name: str | None = None
    suite_name: str | None = None


class Scorecard(BaseModel):
    """Complete output of an evaluation pass. Serialized to scorecards/ as JSON."""

    evaluation: EvaluationMetadata
    scores: list[PromptScore]
    aggregate: AggregateScores


# ---------------------------------------------------------------------------
# Evaluation extraction schemas (compact projection of RunResult for scoring)
# ---------------------------------------------------------------------------


class EvalPromptSummary(BaseModel):
    """Compact representation of a prompt+response for evaluation.

    Strips metrics, options, and raw Ollama fields to reduce context
    consumption when reading responses during interactive evaluation.
    """

    prompt_id: str
    category: str
    difficulty: str
    done_reason: str | None = None
    contamination_risk: str | None = None
    prompt_text: str  # flattened from request.messages
    response_text: str  # from response.message.content
    expected_answer: str | None = None


class EvalRunHeader(BaseModel):
    """Run-level metadata extracted alongside prompt summaries."""

    run_id: str
    model_name: str
    suite_name: str
    suite_file: str
    total_prompts: int
    truncated_count: int
    categories: dict[str, int]
    difficulties: dict[str, int]


class EvalData(BaseModel):
    """Complete pre-extracted evaluation data: header + compact prompts."""

    header: EvalRunHeader
    prompts: list[EvalPromptSummary]


# ---------------------------------------------------------------------------
# Routing discovery schemas
# ---------------------------------------------------------------------------


class RoutingCell(BaseModel):
    """One (model, prompt, strategy) observation in the routing matrix."""

    model: str
    prompt_id: str
    strategy: str
    correct: bool | None = None
    tokens_generated: int | None = None
    latency_ms: float | None = None
    tokens_per_second: float | None = None


class DefaultComparison(BaseModel):
    """How a route compares to the default (largest model, universal strategy)."""

    default_model: str
    default_strategy: str = "universal"
    default_correct: bool | None = None
    default_tokens: int | None = None
    quality_delta_pp: float | None = None  # percentage points
    token_savings_pct: float | None = None


class BestRoute(BaseModel):
    """Best (model, strategy) for a single problem."""

    prompt_id: str
    best_model: str
    best_strategy: str
    correct: bool | None = None
    tokens: int | None = None
    vs_default: DefaultComparison | None = None


class RoutingPattern(BaseModel):
    """A discovered pattern across problem groups."""

    description: str
    affected_problems: list[str]
    recommended_route: dict[str, str]  # e.g. {"model_size": "<=7B", "strategy": "direct"}
    confidence: str  # high, medium, low
    evidence_count: int


class RoutingHeadline(BaseModel):
    """Top-level summary of routing discovery findings."""

    inverse_scaling_detected: bool
    inverse_scaling_rate: float  # fraction of problems showing the effect
    problems_where_routing_helps: int
    problems_total: int
    max_quality_gain_pp: float | None = None
    max_cost_reduction_pct: float | None = None
    routing_worthwhile: bool


class RoutingVerdict(BaseModel):
    routing_recommended: bool
    estimated_quality_improvement_pp: float | None = None
    estimated_token_savings_pct: float | None = None
    caveat: str = ""


class RoutingAnalysis(BaseModel):
    """Complete routing analysis output. Serialized to results/ as JSON."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    models_tested: list[str]
    strategies_tested: list[str]
    headline: RoutingHeadline
    best_route_per_problem: list[BestRoute]
    patterns: list[RoutingPattern]
    verdict: RoutingVerdict
    matrix: list[RoutingCell] = []  # raw data backing the analysis


# ---------------------------------------------------------------------------
# System profile schemas (profiler subsystem)
# ---------------------------------------------------------------------------


class ModelProfile(BaseModel):
    """Performance profile for a single model."""

    vram_bytes: int | None = None
    vram_gb: float | None = None
    load_time_s: float | None = None
    tokens_per_second: float | None = None


class SwapMeasurement(BaseModel):
    from_model: str
    to_model: str
    swap_time_s: float


class CoexistenceTest(BaseModel):
    models: list[str]
    fits: bool
    combined_vram_gb: float | None = None
    headroom_gb: float | None = None


class SystemProfile(BaseModel):
    """Complete system profile for routing cost estimation."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    gpu: str = ""
    vram_total_gb: float | None = None
    ollama_version: str = ""
    models: dict[str, ModelProfile] = {}
    swap_times: list[SwapMeasurement] = []
    coexistence: list[CoexistenceTest] = []
    recommended_hot_tier: list[str] = []
    cold_tier: list[str] = []

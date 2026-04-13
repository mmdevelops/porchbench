"""Thin async wrapper around the Ollama Python client.

Handles chat completions, model metadata, and server info. All Ollama
interaction is funneled through this module so the rest of the codebase
never imports ollama directly.
"""

from __future__ import annotations

import httpx
import ollama as _ollama
from ollama import AsyncClient, ChatResponse

from ollama_bench.schemas import ModelDetails, ModelInfo, ModelOptions, Message, PromptMetrics


async def chat(
    messages: list[Message],
    model: str,
    options: ModelOptions,
    host: str | None = None,
) -> ChatResponse:
    """Send a chat request to Ollama with the given options.

    Returns the raw ChatResponse so callers can extract both the message
    and timing metadata.
    """
    client = AsyncClient(host=host)

    # Separate known Ollama option fields from any extras the user passed through
    opts_dict = options.model_dump()

    return await client.chat(
        model=model,
        messages=[{"role": m.role, "content": m.content} for m in messages],
        options=opts_dict,
    )


def extract_metrics(response: ChatResponse) -> PromptMetrics:
    """Pull raw timing fields from a ChatResponse into our schema."""
    return PromptMetrics(
        prompt_eval_count=getattr(response, "prompt_eval_count", None),
        prompt_eval_duration=getattr(response, "prompt_eval_duration", None),
        eval_count=getattr(response, "eval_count", None),
        eval_duration=getattr(response, "eval_duration", None),
        total_duration=getattr(response, "total_duration", None),
        load_duration=getattr(response, "load_duration", None),
    )


async def get_model_info(model: str, host: str | None = None) -> ModelInfo:
    """Fetch model metadata (family, parameter size, quantization, digest) via Ollama API."""
    client = AsyncClient(host=host)
    info = await client.show(model)
    details = info.get("details", {}) or {}

    # Digest comes from list() (/api/tags), not show()
    digest = await _get_model_digest(model, client)

    return ModelInfo(
        name=model,
        digest=digest,
        details=ModelDetails(
            format=details.get("format"),
            family=details.get("family"),
            parameter_size=details.get("parameter_size"),
            quantization_level=details.get("quantization_level"),
        ),
    )


async def _get_model_digest(model: str, client: AsyncClient) -> str | None:
    """Look up a model's digest from the local model list."""
    try:
        listing = await client.list()
        for m in listing.models:
            m_name = getattr(m, "model", "") or ""
            if m_name == model or m_name.startswith(f"{model}:") or model.startswith(m_name):
                return getattr(m, "digest", None)
    except Exception:
        pass
    return None


async def get_server_version(host: str | None = None) -> str:
    """Fetch the Ollama server version via the REST API.

    The Python client doesn't expose this, so we call the endpoint directly.
    """
    base = host or "http://localhost:11434"
    base = base.rstrip("/")
    try:
        async with httpx.AsyncClient() as http:
            response = await http.get(f"{base}/api/version")
            response.raise_for_status()
            return response.json().get("version", "unknown")
    except (httpx.HTTPError, KeyError):
        return "unknown"


async def list_running_models(host: str | None = None) -> list[dict]:
    """List currently loaded models and their VRAM usage via ollama.ps()."""
    client = AsyncClient(host=host)
    ps_response = await client.ps()
    models = ps_response.get("models", []) or []
    return [
        {
            "name": m.get("name", ""),
            "size": m.get("size"),
            "size_vram": m.get("size_vram"),
            "expires_at": m.get("expires_at"),
        }
        for m in models
    ]

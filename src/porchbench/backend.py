"""Inference backend abstraction layer.

Defines the InferenceBackend protocol and provider implementations.
All provider-specific imports are contained here — the rest of the
codebase uses the protocol and ChatResult type exclusively.

OllamaBackend absorbs the logic previously in client.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx
from ollama import AsyncClient, ChatResponse

from porchbench.schemas import (
    ModelDetails,
    ModelInfo,
    ModelOptions,
    PromptMetrics,
)

# ---------------------------------------------------------------------------
# Provider-neutral types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation from a chat response."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    """Provider-neutral chat completion result."""

    content: str
    role: str
    done_reason: str | None
    metrics: PromptMetrics
    tool_calls: list[ToolCall] | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InferenceBackend(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: ModelOptions,
        tools: list[dict] | None = None,
    ) -> ChatResult: ...

    async def get_model_info(self, model: str) -> ModelInfo: ...

    async def get_server_health(self) -> tuple[bool, str]: ...

    async def list_available_models(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


class OllamaBackend:
    """Inference backend using a local Ollama server."""

    def __init__(self, host: str | None = None):
        self.host = host

    # --- Protocol methods ---

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: ModelOptions,
        tools: list[dict] | None = None,
    ) -> ChatResult:
        """Send a chat request to Ollama and return a provider-neutral result."""
        client = AsyncClient(host=self.host)
        opts_dict = options.model_dump()

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            options=opts_dict,
        )
        if tools is not None:
            kwargs["tools"] = tools

        response = await client.chat(**kwargs)
        return self._to_chat_result(response)

    async def get_model_info(self, model: str) -> ModelInfo:
        """Fetch model metadata via Ollama API."""
        client = AsyncClient(host=self.host)
        info = await client.show(model)
        details = info.get("details", {}) or {}

        digest = await self._get_model_digest(model, client)

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

    async def get_server_health(self) -> tuple[bool, str]:
        """Check Ollama server reachability and return version label."""
        version = await self.get_server_version()
        if version == "unknown":
            return False, "Ollama server not reachable"
        return True, f"Ollama v{version}"

    # --- Ollama-specific extensions (not on protocol) ---

    async def get_server_version(self) -> str:
        """Fetch the Ollama server version via the REST API."""
        base = self.host or "http://localhost:11434"
        base = base.rstrip("/")
        try:
            async with httpx.AsyncClient() as http:
                response = await http.get(f"{base}/api/version")
                response.raise_for_status()
                return response.json().get("version", "unknown")
        except (httpx.HTTPError, KeyError):
            return "unknown"

    async def list_running_models(self) -> list[dict]:
        """List currently loaded models and their VRAM usage via ollama.ps()."""
        client = AsyncClient(host=self.host)
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

    async def list_available_models(self) -> list[str]:
        """Return names of all models pulled locally."""
        client = AsyncClient(host=self.host)
        listing = await client.list()
        return sorted(
            {getattr(m, "model", "") or "" for m in listing.models} - {""}
        )

    # --- Internal helpers ---

    async def _get_model_digest(
        self, model: str, client: AsyncClient
    ) -> str | None:
        """Look up a model's digest from the local model list."""
        try:
            listing = await client.list()
            for m in listing.models:
                m_name = getattr(m, "model", "") or ""
                if (
                    m_name == model
                    or m_name.startswith(f"{model}:")
                    or model.startswith(m_name)
                ):
                    return getattr(m, "digest", None)
        except Exception:
            pass
        return None

    def _to_chat_result(self, response: ChatResponse) -> ChatResult:
        """Translate ollama.ChatResponse into provider-neutral ChatResult."""
        metrics = PromptMetrics(
            prompt_eval_count=getattr(response, "prompt_eval_count", None),
            prompt_eval_duration=getattr(response, "prompt_eval_duration", None),
            eval_count=getattr(response, "eval_count", None),
            eval_duration=getattr(response, "eval_duration", None),
            total_duration=getattr(response, "total_duration", None),
            load_duration=getattr(response, "load_duration", None),
        )

        raw_tool_calls = getattr(response.message, "tool_calls", None) or []
        tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=tc.function.arguments or {},
                )
                for tc in raw_tool_calls
            ]

        return ChatResult(
            content=response.message.content or "",
            role=response.message.role or "assistant",
            done_reason=getattr(response, "done_reason", None),
            metrics=metrics,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAICompatBackend:
    """Inference backend using any OpenAI-compatible API.

    Covers LM Studio, vLLM, llama.cpp server, TabbyAPI, Aphrodite,
    text-generation-webui, and any other server that speaks the OpenAI
    /v1/chat/completions format. Uses httpx directly — no openai SDK.
    """

    def __init__(self, base_url: str, api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # --- Protocol methods ---

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        options: ModelOptions,
        tools: list[dict] | None = None,
    ) -> ChatResult:
        """POST /v1/chat/completions and translate to ChatResult."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_tokens": options.num_predict,
            "seed": options.seed,
        }
        if tools is not None:
            payload["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        wall_start = time.monotonic()
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
        wall_elapsed = time.monotonic() - wall_start

        return self._to_chat_result(data, wall_elapsed)

    async def get_model_info(self, model: str) -> ModelInfo:
        """Best-effort model lookup via /v1/models/{id}.

        Returns ModelInfo on success. Raises LookupError if the server
        confirms the model doesn't exist (404). Returns a stub ModelInfo
        if the endpoint isn't supported (lets callers degrade gracefully).
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{self.base_url}/v1/models/{model}",
                    headers=headers,
                    timeout=10.0,
                )
                if resp.status_code == 404:
                    raise LookupError(f"Model not found: {model}")
                resp.raise_for_status()
                return ModelInfo(name=model)
        except LookupError:
            raise
        except Exception:
            # Server doesn't support model lookups — return stub
            return ModelInfo(name=model)

    async def get_server_health(self) -> tuple[bool, str]:
        """Check reachability by listing models at /v1/models."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{self.base_url}/v1/models",
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
            return True, f"OpenAI-compat @ {self.base_url}"
        except (httpx.HTTPError, Exception) as exc:
            return False, f"OpenAI-compat server not reachable: {exc}"

    async def list_available_models(self) -> list[str]:
        """Best-effort model listing via /v1/models. Returns empty on failure."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"{self.base_url}/v1/models",
                    headers=headers,
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                return sorted(m.get("id", "") for m in data if m.get("id"))
        except Exception:
            return []

    # --- Internal helpers ---

    def _to_chat_result(
        self, data: dict[str, Any], wall_elapsed: float
    ) -> ChatResult:
        """Translate OpenAI chat completion JSON into ChatResult."""
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        # Wall-clock total_duration in nanoseconds (consistent with Ollama convention)
        total_duration_ns = int(wall_elapsed * 1e9)

        metrics = PromptMetrics(
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration=None,
            eval_count=completion_tokens,
            eval_duration=None,
            total_duration=total_duration_ns,
            load_duration=None,
        )

        # Parse tool calls if present
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args_raw = func.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_raw
                tool_calls.append(ToolCall(name=func.get("name", ""), arguments=args))

        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")

        return ChatResult(
            content=content,
            role=message.get("role", "assistant"),
            done_reason=finish_reason,
            metrics=metrics,
            tool_calls=tool_calls,
        )

"""Tests for the inference backend abstraction layer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from porchbench.backend import ChatResult, InferenceBackend, OllamaBackend, OpenAICompatBackend, ToolCall
from porchbench.schemas import ModelOptions, PromptMetrics


# ---------------------------------------------------------------------------
# Helpers — fake Ollama response objects
# ---------------------------------------------------------------------------


def _make_ollama_response(
    content="Hello",
    role="assistant",
    done_reason="stop",
    tool_calls=None,
    prompt_eval_count=10,
    prompt_eval_duration=500_000_000,
    eval_count=20,
    eval_duration=1_000_000_000,
    total_duration=1_600_000_000,
    load_duration=100_000_000,
):
    """Build a fake object shaped like ollama.ChatResponse."""
    message = SimpleNamespace(
        content=content,
        role=role,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        message=message,
        done_reason=done_reason,
        prompt_eval_count=prompt_eval_count,
        prompt_eval_duration=prompt_eval_duration,
        eval_count=eval_count,
        eval_duration=eval_duration,
        total_duration=total_duration,
        load_duration=load_duration,
    )


def _make_tool_call(name="read_file", arguments=None):
    """Build a fake Ollama tool call object."""
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=arguments or {"path": "data.txt"},
        )
    )


# ---------------------------------------------------------------------------
# ChatResult translation
# ---------------------------------------------------------------------------


class TestToChatResult:
    def test_extracts_all_fields(self):
        response = _make_ollama_response()
        backend = OllamaBackend()
        result = backend._to_chat_result(response)

        assert result.content == "Hello"
        assert result.role == "assistant"
        assert result.done_reason == "stop"
        assert result.tool_calls is None

        m = result.metrics
        assert m.prompt_eval_count == 10
        assert m.prompt_eval_duration == 500_000_000
        assert m.eval_count == 20
        assert m.eval_duration == 1_000_000_000
        assert m.total_duration == 1_600_000_000
        assert m.load_duration == 100_000_000

    def test_handles_tool_calls(self):
        tc1 = _make_tool_call("read_file", {"path": "a.txt"})
        tc2 = _make_tool_call("write_file", {"path": "b.txt", "content": "hi"})
        response = _make_ollama_response(tool_calls=[tc1, tc2])

        backend = OllamaBackend()
        result = backend._to_chat_result(response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "a.txt"}
        assert result.tool_calls[1].name == "write_file"
        assert result.tool_calls[1].arguments == {"path": "b.txt", "content": "hi"}

    def test_no_tool_calls(self):
        response = _make_ollama_response(tool_calls=None)
        backend = OllamaBackend()
        result = backend._to_chat_result(response)

        assert result.tool_calls is None

    def test_empty_tool_calls_list(self):
        response = _make_ollama_response(tool_calls=[])
        backend = OllamaBackend()
        result = backend._to_chat_result(response)

        assert result.tool_calls is None

    def test_none_fields_default_gracefully(self):
        response = _make_ollama_response(
            content=None,
            role=None,
            done_reason=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None,
            total_duration=None,
            load_duration=None,
        )
        # Remove attributes to simulate missing fields
        del response.prompt_eval_count
        del response.prompt_eval_duration
        del response.eval_count
        del response.eval_duration
        del response.total_duration
        del response.load_duration
        del response.done_reason

        backend = OllamaBackend()
        result = backend._to_chat_result(response)

        assert result.content == ""
        assert result.role == "assistant"
        assert result.done_reason is None

        m = result.metrics
        assert m.prompt_eval_count is None
        assert m.prompt_eval_duration is None
        assert m.eval_count is None
        assert m.eval_duration is None
        assert m.total_duration is None
        assert m.load_duration is None


# ---------------------------------------------------------------------------
# Server health
# ---------------------------------------------------------------------------


class TestGetServerHealth:
    @pytest.mark.asyncio
    async def test_success(self):
        backend = OllamaBackend()

        with patch.object(backend, "get_server_version", new_callable=AsyncMock, return_value="0.6.2"):
            healthy, label = await backend.get_server_health()

        assert healthy is True
        assert label == "Ollama v0.6.2"

    @pytest.mark.asyncio
    async def test_unreachable(self):
        backend = OllamaBackend()

        with patch.object(backend, "get_server_version", new_callable=AsyncMock, return_value="unknown"):
            healthy, label = await backend.get_server_health()

        assert healthy is False
        assert "not reachable" in label


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_ollama_backend_satisfies_protocol(self):
        backend = OllamaBackend()
        assert isinstance(backend, InferenceBackend)

    def test_chat_result_is_dataclass(self):
        result = ChatResult(
            content="hi",
            role="assistant",
            done_reason="stop",
            metrics=PromptMetrics(),
        )
        assert result.content == "hi"
        assert result.tool_calls is None

    def test_tool_call_is_dataclass(self):
        tc = ToolCall(name="read_file", arguments={"path": "x"})
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "x"}

    def test_openai_compat_backend_satisfies_protocol(self):
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        assert isinstance(backend, InferenceBackend)


# ---------------------------------------------------------------------------
# OpenAI-compat backend
# ---------------------------------------------------------------------------


def _make_openai_response(
    content="Hello",
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=20,
    tool_calls=None,
):
    """Build a fake OpenAI /v1/chat/completions JSON response."""
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestOpenAICompatChatResult:
    def test_extracts_all_fields(self):
        data = _make_openai_response()
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=1.5)

        assert result.content == "Hello"
        assert result.role == "assistant"
        assert result.done_reason == "stop"
        assert result.tool_calls is None

        m = result.metrics
        assert m.prompt_eval_count == 10
        assert m.eval_count == 20
        assert m.total_duration == 1_500_000_000  # 1.5s in nanoseconds
        # OpenAI-compat doesn't provide internal timing
        assert m.prompt_eval_duration is None
        assert m.eval_duration is None
        assert m.load_duration is None

    def test_handles_tool_calls_with_string_arguments(self):
        """OpenAI returns tool_call arguments as JSON strings."""
        data = _make_openai_response(tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "data.txt"}',
                },
            },
        ])
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.5)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "data.txt"}

    def test_handles_tool_calls_with_dict_arguments(self):
        """Some servers return arguments already parsed as dicts."""
        data = _make_openai_response(tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "out.txt", "content": "hi"},
                },
            },
        ])
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.5)

        assert result.tool_calls[0].name == "write_file"
        assert result.tool_calls[0].arguments == {"path": "out.txt", "content": "hi"}

    def test_no_tool_calls(self):
        data = _make_openai_response(tool_calls=None)
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.5)

        assert result.tool_calls is None

    def test_empty_content_defaults(self):
        data = _make_openai_response(content=None)
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.5)

        assert result.content == ""

    def test_missing_usage(self):
        data = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.3)

        assert result.metrics.prompt_eval_count is None
        assert result.metrics.eval_count is None
        assert result.metrics.total_duration is not None

    def test_malformed_tool_call_arguments(self):
        """Gracefully handle unparseable argument strings."""
        data = _make_openai_response(tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bad_tool", "arguments": "not json{{{"},
            },
        ])
        backend = OpenAICompatBackend(base_url="http://localhost:1234")
        result = backend._to_chat_result(data, wall_elapsed=0.5)

        assert result.tool_calls[0].name == "bad_tool"
        assert result.tool_calls[0].arguments == {}


class TestOpenAICompatHealth:
    @pytest.mark.asyncio
    async def test_success(self):
        backend = OpenAICompatBackend(base_url="http://localhost:1234")

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            healthy, label = await backend.get_server_health()

        assert healthy is True
        assert "localhost:1234" in label

    @pytest.mark.asyncio
    async def test_unreachable(self):
        backend = OpenAICompatBackend(base_url="http://localhost:9999")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            healthy, label = await backend.get_server_health()

        assert healthy is False
        assert "not reachable" in label


class TestOpenAICompatChat:
    @pytest.mark.asyncio
    async def test_chat_sends_correct_payload(self):
        backend = OpenAICompatBackend(base_url="http://localhost:1234", api_key="test-key")

        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = _make_openai_response()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await backend.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="test-model",
                options=ModelOptions(temperature=0.5, num_predict=100),
            )

        # Verify the result
        assert result.content == "Hello"
        assert result.metrics.prompt_eval_count == 10
        assert result.metrics.eval_count == 20

        # Verify the POST payload
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100  # num_predict mapped
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

        # Verify auth header
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"

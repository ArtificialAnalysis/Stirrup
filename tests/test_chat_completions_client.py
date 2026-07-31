"""Token-budget behavior for ChatCompletionsClient."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from stirrup import ContextOverflowError, OutputTokenLimitError
from stirrup.clients import chat_completions_client as chat_completions_module
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import UserMessage


def _bad_request_error(code: str) -> BadRequestError:
    response = httpx.Response(
        400,
        json={"error": {"message": "boom", "type": "invalid_request_error", "code": code}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return BadRequestError("boom", response=response, body=response.json()["error"])


def _client_with_provider_call(monkeypatch: pytest.MonkeyPatch, provider_call: AsyncMock) -> ChatCompletionsClient:
    provider_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider_call)))
    monkeypatch.setattr(chat_completions_module, "AsyncOpenAI", lambda **_kwargs: provider_client)
    return ChatCompletionsClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)


def test_context_window_tokens_is_required() -> None:
    with pytest.raises(TypeError, match="context_window_tokens"):
        ChatCompletionsClient(model="gpt-4o", api_key="test-key", max_tokens=8_192)  # ty: ignore[missing-argument]


@pytest.mark.parametrize("context_window_tokens", [0, -1])
def test_context_window_must_be_positive(context_window_tokens: int) -> None:
    with pytest.raises(ValueError, match="context_window_tokens must be positive"):
        ChatCompletionsClient(
            model="gpt-4o",
            api_key="test-key",
            context_window_tokens=context_window_tokens,
        )


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_output_limit_is_forwarded_and_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    provider_call = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason)],
            usage=None,
        )
    )
    provider_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider_call)))
    monkeypatch.setattr(chat_completions_module, "AsyncOpenAI", lambda **_kwargs: provider_client)
    client = ChatCompletionsClient(
        model="gpt-4o",
        api_key="test-key",
        max_tokens=321,
        context_window_tokens=64_000,
    )

    with pytest.raises(OutputTokenLimitError, match=r"gpt-4o.*max_tokens=321.*Increase max_tokens"):
        await client.generate([UserMessage(content="hello")], {})

    provider_call.assert_awaited_once()
    provider_request = provider_call.await_args
    assert provider_request is not None
    assert provider_request.kwargs["max_completion_tokens"] == 321
    assert client.context_window_tokens == 64_000


async def test_context_length_rejection_surfaces_as_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_provider_call(
        monkeypatch, AsyncMock(side_effect=_bad_request_error("context_length_exceeded"))
    )

    with pytest.raises(ContextOverflowError):
        await client.generate([UserMessage(content="hello")], {})


async def test_unrelated_bad_request_is_not_mapped_to_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_provider_call(monkeypatch, AsyncMock(side_effect=_bad_request_error("invalid_value")))

    with pytest.raises(BadRequestError):
        await client.generate([UserMessage(content="hello")], {})

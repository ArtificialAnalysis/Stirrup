"""Token-budget behavior for ChatCompletionsClient."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from stirrup import OutputTokenLimitError
from stirrup.clients import chat_completions_client as chat_completions_module
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import UserMessage


def test_context_window_defaults_to_max_tokens() -> None:
    client = ChatCompletionsClient(model="gpt-4o", api_key="test-key", max_tokens=8_192)

    assert client.context_window_tokens == 8_192


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

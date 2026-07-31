"""Token-budget behavior for the optional LiteLLM client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("litellm")

from litellm.exceptions import ContextWindowExceededError

from stirrup.clients import litellm_client as litellm_module
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError


def test_context_window_tokens_is_required() -> None:
    with pytest.raises(TypeError, match="context_window_tokens"):
        LiteLLMClient(model="test/provider-model", max_tokens=8_192)  # ty: ignore[missing-argument]


@pytest.mark.parametrize("context_window_tokens", [0, -1])
def test_context_window_must_be_positive(context_window_tokens: int) -> None:
    with pytest.raises(ValueError, match="context_window_tokens must be positive"):
        LiteLLMClient(model="test/provider-model", context_window_tokens=context_window_tokens)


def test_max_tokens_must_fit_context_window() -> None:
    with pytest.raises(ValueError, match="must not exceed context_window_tokens"):
        LiteLLMClient(model="test/provider-model", max_tokens=128_000, context_window_tokens=8_192)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_output_limit_is_forwarded_and_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    provider_call = AsyncMock(return_value={"choices": [SimpleNamespace(finish_reason=finish_reason)]})
    monkeypatch.setattr(litellm_module, "acompletion", provider_call)
    client = LiteLLMClient(
        model="test/provider-model",
        max_tokens=789,
        context_window_tokens=76_543,
    )

    with pytest.raises(OutputTokenLimitError, match=r"test/provider-model.*max_tokens=789.*Increase max_tokens"):
        await client.generate([], {})

    provider_call.assert_awaited_once()
    provider_request = provider_call.await_args
    assert provider_request is not None
    assert provider_request.kwargs["max_tokens"] == 789
    assert client.context_window_tokens == 76_543


async def test_context_window_rejection_surfaces_as_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_call = AsyncMock(
        side_effect=ContextWindowExceededError(
            message="boom",
            model="test/provider-model",
            llm_provider="test",
        )
    )
    monkeypatch.setattr(litellm_module, "acompletion", provider_call)
    client = LiteLLMClient(model="test/provider-model", max_tokens=8_192, context_window_tokens=76_543)

    with pytest.raises(ContextOverflowError):
        await client.generate([], {})

"""Token-budget behavior for the optional LiteLLM client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("litellm")

from litellm.exceptions import ContextWindowExceededError

from stirrup.clients import litellm_client as litellm_module
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError


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

    with pytest.raises(OutputTokenLimitError) as exc_info:
        await client.generate([], {})

    assert exc_info.value.model_slug == "test/provider-model"
    assert exc_info.value.max_tokens == 789
    assert exc_info.value.provider_reason == finish_reason
    provider_call.assert_awaited_once()
    provider_request = provider_call.await_args
    assert provider_request is not None
    assert provider_request.kwargs["max_tokens"] == 789


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

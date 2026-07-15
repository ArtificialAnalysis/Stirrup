"""Tests for context_window separation from max_tokens on LLM clients."""

import pytest

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.open_responses_client import OpenResponsesClient
from stirrup.core.models import llm_client_context_window


class _LegacyClientStub:
    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens


class TestLlmClientContextWindowHelper:
    def test_falls_back_to_max_tokens_for_legacy_clients(self) -> None:
        client = _LegacyClientStub(max_tokens=128_000)
        assert llm_client_context_window(client) == 128_000

    def test_uses_context_window_when_present(self) -> None:
        client = ChatCompletionsClient(
            model="gpt-4o",
            max_tokens=4096,
            context_window=200_000,
            api_key="test-key",
        )
        assert llm_client_context_window(client) == 200_000


class TestContextWindowProperty:
    """Verify context_window defaults and independent configuration."""

    @pytest.mark.parametrize(
        "client_cls",
        [ChatCompletionsClient, OpenResponsesClient],
    )
    def test_context_window_defaults_to_max_tokens(self, client_cls: type) -> None:
        client = client_cls(model="gpt-4o", max_tokens=8192, api_key="test-key")
        assert client.max_tokens == 8192
        assert client.context_window == 8192

    @pytest.mark.parametrize(
        "client_cls",
        [ChatCompletionsClient, OpenResponsesClient],
    )
    def test_context_window_can_differ_from_max_tokens(self, client_cls: type) -> None:
        client = client_cls(
            model="gpt-4o",
            max_tokens=4096,
            context_window=128_000,
            api_key="test-key",
        )
        assert client.max_tokens == 4096
        assert client.context_window == 128_000

    def test_open_responses_client_properties(self) -> None:
        client = OpenResponsesClient(
            model="gpt-4o",
            max_tokens=50000,
            context_window=200_000,
            api_key="test-key",
        )
        assert client.model_slug == "gpt-4o"
        assert client.max_tokens == 50000
        assert client.context_window == 200_000


class TestLiteLLMContextWindowProperty:
    """LiteLLM-specific context_window tests (requires litellm extra)."""

    def test_context_window_defaults_to_max_tokens(self) -> None:
        pytest.importorskip("litellm")
        from stirrup.clients.litellm_client import LiteLLMClient

        client = LiteLLMClient(model="anthropic/claude-sonnet-4-5", max_tokens=8192)
        assert client.max_tokens == 8192
        assert client.context_window == 8192

    def test_context_window_can_differ_from_max_tokens(self) -> None:
        pytest.importorskip("litellm")
        from stirrup.clients.litellm_client import LiteLLMClient

        client = LiteLLMClient(
            model="anthropic/claude-sonnet-4-5",
            max_tokens=4096,
            context_window=128_000,
        )
        assert client.max_tokens == 4096
        assert client.context_window == 128_000

"""Naming an endpoint means naming its key; without a base_url the SDK decides."""

import pytest
from openai import OpenAIError

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.open_responses_client import OpenResponsesClient

# Both clients build their SDK client through the same factory, so both must enforce it.
pytestmark = pytest.mark.parametrize("client_cls", [ChatCompletionsClient, OpenResponsesClient])

OpenAIClient = type[ChatCompletionsClient] | type[OpenResponsesClient]


@pytest.fixture(autouse=True)
def _both_provider_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every provider key is set, so a refusal also proves neither one leaked."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-env-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


@pytest.mark.parametrize("api_key", [None, ""], ids=["omitted", "empty"])
def test_base_url_without_api_key_is_refused(client_cls: OpenAIClient, api_key: str | None) -> None:
    with pytest.raises(OpenAIError):
        client_cls(model="deepseek-chat", base_url="https://api.deepseek.com", api_key=api_key)


def test_base_url_sends_the_key_it_was_given(client_cls: OpenAIClient) -> None:
    client = client_cls(model="deepseek-chat", base_url="https://api.deepseek.com", api_key="deepseek-key")
    assert client._client.auth_headers == {"Authorization": "Bearer deepseek-key"}  # noqa: SLF001


def test_without_base_url_the_sdk_resolves_openai(client_cls: OpenAIClient) -> None:
    client = client_cls(model="gpt-5")
    assert str(client._client.base_url).startswith("https://api.openai.com")  # noqa: SLF001
    assert client._client.auth_headers == {"Authorization": "Bearer openai-env-key"}  # noqa: SLF001


def test_without_base_url_a_missing_openai_key_raises(
    client_cls: OpenAIClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(OpenAIError):
        client_cls(model="gpt-5")

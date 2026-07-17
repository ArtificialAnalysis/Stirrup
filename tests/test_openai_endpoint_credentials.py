"""Endpoint and credential selection tests for OpenAI-compatible clients."""

import httpx
import pytest
from openai import OpenAIError
from pytest import MonkeyPatch

from stirrup.clients.chat_completions_client import ChatCompletionsClient


def _set_provider_keys(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")


async def _assert_client_configuration(
    client: ChatCompletionsClient,
    expected_base_url: str,
    expected_api_key: str,
) -> None:
    try:
        assert str(client._client.base_url) == expected_base_url  # noqa: SLF001
        assert client._client.auth_headers == {"Authorization": f"Bearer {expected_api_key}"}  # noqa: SLF001
    finally:
        await client._client.close()  # noqa: SLF001


@pytest.mark.parametrize(
    ("base_url", "environment_base_url", "api_key", "expected_base_url", "expected_api_key"),
    [
        (None, None, None, "https://api.openai.com/v1/", "openai-key"),
        (
            "https://openrouter.ai/api/v1",
            None,
            None,
            "https://openrouter.ai/api/v1/",
            "openrouter-key",
        ),
        (None, "https://openrouter.ai/api/v1", None, "https://openrouter.ai/api/v1/", "openrouter-key"),
        (
            "https://OPENROUTER.AI.:443/api/v1",
            None,
            None,
            "https://openrouter.ai/api/v1/",
            "openrouter-key",
        ),
        (
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            None,
            "https://api.openai.com/v1/",
            "openai-key",
        ),
        (
            "https://gateway.example/v1",
            "https://openrouter.ai/api/v1",
            "explicit-key",
            "https://gateway.example/v1/",
            "explicit-key",
        ),
        ("http://localhost:8000/v1", None, "local-key", "http://localhost:8000/v1/", "local-key"),
    ],
    ids=[
        "default-openai",
        "openrouter",
        "environment-endpoint",
        "canonical-url",
        "explicit-endpoint",
        "explicit-key-custom-https",
        "explicit-key-custom-http",
    ],
)
async def test_effective_endpoint_selects_its_api_key(
    monkeypatch: MonkeyPatch,
    base_url: str | None,
    environment_base_url: str | None,
    api_key: str | None,
    expected_base_url: str,
    expected_api_key: str,
) -> None:
    _set_provider_keys(monkeypatch)
    if environment_base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", environment_base_url)

    client = ChatCompletionsClient(model="gpt-4o", base_url=base_url, api_key=api_key)

    await _assert_client_configuration(client, expected_base_url, expected_api_key)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.example/v1",
        "https://api.openrouter.ai/api/v1",
        "https://openrouter.ai.evil.example/api/v1",
    ],
    ids=["custom", "unrecognized-subdomain", "deceptive-host"],
)
def test_custom_https_endpoint_requires_explicit_api_key(
    monkeypatch: MonkeyPatch,
    base_url: str,
) -> None:
    _set_provider_keys(monkeypatch)

    with pytest.raises(OpenAIError, match="Custom endpoints require an explicit api_key"):
        ChatCompletionsClient(model="gpt-4o", base_url=base_url)


def test_http_endpoint_requires_explicit_api_key(monkeypatch: MonkeyPatch) -> None:
    _set_provider_keys(monkeypatch)

    with pytest.raises(OpenAIError, match="HTTP endpoints require an explicit api_key"):
        ChatCompletionsClient(model="gpt-4o", base_url="http://openrouter.ai/api/v1")


@pytest.mark.parametrize(
    ("base_url", "missing_variable", "other_variable"),
    [
        (None, "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    ],
    ids=["openai", "openrouter"],
)
def test_provider_key_does_not_fall_back_to_other_provider(
    monkeypatch: MonkeyPatch,
    base_url: str | None,
    missing_variable: str,
    other_variable: str,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv(missing_variable, raising=False)
    monkeypatch.setenv(other_variable, "other-provider-key")

    with pytest.raises(OpenAIError, match=missing_variable):
        ChatCompletionsClient(model="gpt-4o", base_url=base_url)


def test_base_url_with_userinfo_is_rejected() -> None:
    with pytest.raises(httpx.InvalidURL, match="userinfo"):
        ChatCompletionsClient(
            model="gpt-4o",
            base_url="https://user:password@openrouter.ai/api/v1",
            api_key="explicit-key",
        )


def test_non_http_base_url_is_rejected() -> None:
    with pytest.raises(httpx.InvalidURL, match=r"absolute HTTP\(S\) URL"):
        ChatCompletionsClient(
            model="gpt-4o",
            base_url="ftp://api.openai.com/v1",
            api_key="explicit-key",
        )

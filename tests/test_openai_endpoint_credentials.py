"""Endpoint and credential selection tests for OpenAI-compatible clients."""

from collections.abc import Mapping

import pytest
from openai import OpenAIError
from pytest import MonkeyPatch

from stirrup.clients.chat_completions_client import ChatCompletionsClient

_CUSTOM_ENDPOINT_MESSAGE = "Custom endpoints require an explicit api_key"


def _set_provider_keys(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")


def _apply_environment_overrides(monkeypatch: MonkeyPatch, overrides: Mapping[str, str | None]) -> None:
    """Set each variable to its value, or delete it when the value is ``None``."""
    for name, value in overrides.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


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
        (None, "", None, "https://api.openai.com/v1/", "openai-key"),
        ("https://openrouter.ai/api/v1", None, None, "https://openrouter.ai/api/v1/", "openrouter-key"),
        (None, "https://openrouter.ai/api/v1", None, "https://openrouter.ai/api/v1/", "openrouter-key"),
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
        (
            "https://openrouter.ai/api/v1",
            None,
            "explicit-key",
            "https://openrouter.ai/api/v1/",
            "explicit-key",
        ),
    ],
    ids=[
        "default-openai",
        "blank-environment-endpoint-falls-back-to-default",
        "openrouter",
        "environment-endpoint",
        "explicit-endpoint-overrides-environment",
        "explicit-key-custom-https",
        "explicit-key-custom-http",
        "explicit-key-overrides-provider-key",
    ],
)
async def test_accepted_configurations(
    monkeypatch: MonkeyPatch,
    base_url: str | None,
    environment_base_url: str | None,
    api_key: str | None,
    expected_base_url: str,
    expected_api_key: str,
) -> None:
    _set_provider_keys(monkeypatch)
    _apply_environment_overrides(monkeypatch, {"OPENAI_BASE_URL": environment_base_url})

    client = ChatCompletionsClient(model="gpt-4o", base_url=base_url, api_key=api_key)

    await _assert_client_configuration(client, expected_base_url, expected_api_key)


@pytest.mark.parametrize(
    ("base_url", "environment_base_url", "api_key", "environment_overrides", "expected_message"),
    [
        ("https://gateway.example/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://gateway.example/v1", None, "", {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://api.openrouter.ai/api/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://chat.openai.com/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://openrouter.ai.evil.example/api/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://api.openai.com.evil.example/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://evil-openai.com/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("https://api.openai.com:8443/v1", None, None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        (None, "https://litellm.mycorp.internal/v1", None, {}, _CUSTOM_ENDPOINT_MESSAGE),
        ("http://openrouter.ai/api/v1", None, None, {}, "HTTP endpoints require an explicit api_key"),
        (
            None,
            None,
            None,
            {"OPENAI_API_KEY": None, "OPENROUTER_API_KEY": "other-provider-key"},
            "OPENAI_API_KEY",
        ),
        (
            "https://openrouter.ai/api/v1",
            None,
            None,
            {"OPENROUTER_API_KEY": None, "OPENAI_API_KEY": "other-provider-key"},
            "OPENROUTER_API_KEY",
        ),
        (None, None, None, {"OPENAI_API_KEY": ""}, "OPENAI_API_KEY"),
        ("https://user:password@openrouter.ai/api/v1", None, "explicit-key", {}, "userinfo"),
        ("ftp://api.openai.com/v1", None, "explicit-key", {}, r"absolute HTTP\(S\) URL"),
    ],
    ids=[
        "custom-host",
        "blank-api-key-is-not-explicit",
        "unrecognized-openrouter-subdomain",
        "unrecognized-openai-subdomain",
        "deceptive-openrouter-host",
        "deceptive-openai-host",
        "openai-lookalike-host",
        "provider-host-on-other-port",
        "custom-environment-endpoint",
        "http-provider-host",
        "openai-key-does-not-fall-back-to-openrouter",
        "openrouter-key-does-not-fall-back-to-openai",
        "blank-provider-key-is-missing",
        "userinfo-in-base-url",
        "non-http-scheme",
    ],
)
def test_rejected_configurations(
    monkeypatch: MonkeyPatch,
    base_url: str | None,
    environment_base_url: str | None,
    api_key: str | None,
    environment_overrides: Mapping[str, str | None],
    expected_message: str,
) -> None:
    _set_provider_keys(monkeypatch)
    _apply_environment_overrides(monkeypatch, {"OPENAI_BASE_URL": environment_base_url, **environment_overrides})

    with pytest.raises(OpenAIError, match=expected_message):
        ChatCompletionsClient(model="gpt-4o", base_url=base_url, api_key=api_key)

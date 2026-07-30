"""Endpoint and credential selection tests for OpenAI-compatible clients."""

from collections.abc import Mapping

import pytest
from openai import OpenAIError
from pytest import MonkeyPatch

from stirrup.clients.chat_completions_client import ChatCompletionsClient

_EXPLICIT_KEY_MESSAGE = "requires an explicit api_key"
_SDK_MISSING_KEY_MESSAGE = "The api_key client option must be set"


def _set_openai_key(monkeypatch: MonkeyPatch) -> None:
    """Make ``OPENAI_API_KEY`` available, so a rejection also proves it was not sent."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")


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
        (
            "https://api.openai.com/v1",
            "https://gateway.example/v1",
            None,
            "https://api.openai.com/v1/",
            "openai-key",
        ),
        (
            "https://openrouter.ai/api/v1",
            None,
            "openrouter-key",
            "https://openrouter.ai/api/v1/",
            "openrouter-key",
        ),
        (
            "https://gateway.example/v1",
            "https://api.openai.com/v1",
            "gateway-key",
            "https://gateway.example/v1/",
            "gateway-key",
        ),
        ("http://localhost:8000/v1", None, "local-key", "http://localhost:8000/v1/", "local-key"),
        ("https://api.openai.com:8443/v1", None, "port-key", "https://api.openai.com:8443/v1/", "port-key"),
    ],
    ids=[
        "default-endpoint-defers-key-lookup-to-sdk",
        "blank-environment-endpoint-falls-back-to-default",
        "explicit-endpoint-overrides-environment",
        "explicit-key-provider-host",
        "explicit-key-custom-https",
        "explicit-key-custom-http",
        "explicit-key-openai-host-on-other-port",
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
    _set_openai_key(monkeypatch)
    _apply_environment_overrides(monkeypatch, {"OPENAI_BASE_URL": environment_base_url})

    client = ChatCompletionsClient(model="gpt-4o", base_url=base_url, api_key=api_key)

    await _assert_client_configuration(client, expected_base_url, expected_api_key)


def test_default_endpoint_without_openai_key_raises_the_sdk_error(monkeypatch: MonkeyPatch) -> None:
    """Stirrup sends no credential of its own, so the SDK reports the missing key itself."""
    _apply_environment_overrides(monkeypatch, {"OPENAI_BASE_URL": None, "OPENAI_API_KEY": None})

    with pytest.raises(OpenAIError, match=_SDK_MISSING_KEY_MESSAGE):
        ChatCompletionsClient(model="gpt-4o")


@pytest.mark.parametrize(
    ("base_url", "api_key", "environment_overrides", "expected_message"),
    [
        ("https://gateway.example/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("https://gateway.example/v1", "", {}, _EXPLICIT_KEY_MESSAGE),
        ("https://openrouter.ai/api/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("https://chat.openai.com/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("https://xapi.openai.com/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("https://api.openai.com.evil.example/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("https://api.openai.com:8443/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("http://api.openai.com/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        ("http://localhost:8000/v1", None, {}, _EXPLICIT_KEY_MESSAGE),
        (None, None, {"OPENAI_BASE_URL": "https://litellm.mycorp.internal/v1"}, _EXPLICIT_KEY_MESSAGE),
        (None, None, {"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"}, _EXPLICIT_KEY_MESSAGE),
        ("https://user:password@api.openai.com/v1", "explicit-key", {}, "userinfo"),
        ("ftp://api.openai.com/v1", "explicit-key", {}, r"absolute HTTP\(S\) URL"),
    ],
    ids=[
        "custom-host",
        "blank-api-key-is-not-explicit",
        "provider-host",
        "unrecognized-openai-subdomain",
        "openai-prefix-lookalike",
        "deceptive-openai-host",
        "openai-host-on-other-port",
        "openai-host-over-http",
        "custom-http-host",
        "custom-environment-endpoint",
        "provider-environment-endpoint",
        "userinfo-in-base-url",
        "non-http-scheme",
    ],
)
def test_rejected_configurations(
    monkeypatch: MonkeyPatch,
    base_url: str | None,
    api_key: str | None,
    environment_overrides: Mapping[str, str | None],
    expected_message: str,
) -> None:
    _set_openai_key(monkeypatch)
    # OPENAI_BASE_URL is absent unless a row asks for it.
    _apply_environment_overrides(monkeypatch, {"OPENAI_BASE_URL": None, **environment_overrides})

    with pytest.raises(OpenAIError, match=expected_message):
        ChatCompletionsClient(model="gpt-4o", base_url=base_url, api_key=api_key)

"""Tests for Brave web search."""

from collections.abc import Awaitable
from typing import Any, cast

import httpx
import pytest
from pytest import MonkeyPatch
from tenacity import wait_fixed, wait_none

from stirrup.core.models import Tool, ToolResult
from stirrup.tools import web
from stirrup.tools.web import WebSearchMetadata, WebSearchParams, WebToolProvider, _get_websearch_tool


async def run_web_search_tool(
    tool: Tool[WebSearchParams, WebSearchMetadata],
    query: str,
) -> ToolResult[WebSearchMetadata]:
    result = tool.executor(WebSearchParams(query=query))
    return await cast(Awaitable[ToolResult[WebSearchMetadata]], result)


async def test_web_search_retries_brave_429(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(web, "WEB_SEARCH_RETRY_WAIT", wait_fixed(0.01))
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "description": "Example result",
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_websearch_tool("test-key", client)
        result = await run_web_search_tool(tool, "example")

    assert attempts == 2
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.pages_returned == 1
    assert result.metadata.retry_idle_for == 0.01
    assert "https://example.com" in result.content


async def test_web_search_does_not_retry_non_429_http_errors(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(web, "WEB_SEARCH_RETRY_WAIT", wait_none())
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_websearch_tool("test-key", client)
        with pytest.raises(httpx.HTTPStatusError):
            await run_web_search_tool(tool, "example")

    assert attempts == 1


def brave_results(*urls: str) -> dict:
    return {"web": {"results": [{"title": url, "url": url, "description": ""} for url in urls]}}


async def test_web_search_without_exclude_domains_sends_no_goggles() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json=brave_results("https://example.com"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_websearch_tool("test-key", client)
        result = await run_web_search_tool(tool, "example")

    assert "goggles" not in requests[0].url.params
    assert result.metadata is not None
    assert result.metadata.results_excluded == 0


async def test_web_search_sends_exclude_domains_as_brave_discard_goggle() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json=brave_results("https://example.com"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_websearch_tool(
            "test-key",
            client,
            exclude_domains=["huggingface.co", "https://GitHub.com/some/repo", "*.kaggle.com", "huggingface.co"],
        )
        await run_web_search_tool(tool, "example")

    params = requests[0].url.params
    assert params["q"] == "example"
    assert params["count"] == "5"
    assert params.get_list("goggles") == [
        "$discard,site=huggingface.co\n$discard,site=github.com\n$discard,site=kaggle.com"
    ]


async def test_web_search_drops_results_from_excluded_domains_and_subdomains() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json=brave_results(
                "https://huggingface.co/datasets/leaked",
                "https://discuss.HuggingFace.co/t/answer",
                "https://nothuggingface.co/page",
                "https://example.com/huggingface.co",
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_websearch_tool("test-key", client, exclude_domains=["huggingface.co"])
        first = await run_web_search_tool(tool, "example")
        second = await run_web_search_tool(tool, "example")

    assert "https://huggingface.co/datasets/leaked" not in first.content
    assert "discuss.HuggingFace.co" not in first.content
    assert "https://nothuggingface.co/page" in first.content
    assert "https://example.com/huggingface.co" in first.content
    assert first.metadata is not None
    assert first.metadata.pages_returned == 2
    assert first.metadata.results_excluded == 2
    assert second.metadata is not None
    assert (first.metadata + second.metadata).results_excluded == 4


@pytest.mark.parametrize(
    "domain", ["*", "https://", "*.*.com", "a b.com", "a.com\n$discard", "a.com,$boost=3", "a..com"]
)
def test_web_search_rejects_invalid_exclude_domains(domain: str) -> None:
    with pytest.raises(ValueError, match="Invalid domain"):
        _get_websearch_tool("test-key", exclude_domains=[domain])


def test_web_search_rejects_exclude_domains_given_as_a_string() -> None:
    with pytest.raises(TypeError):
        _get_websearch_tool("test-key", exclude_domains="huggingface.co")


async def search_through_provider(monkeypatch: MonkeyPatch, provider: WebToolProvider) -> httpx.Request:
    """Run one search through ``provider`` and return the request it sent to Brave."""
    requests: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, json=brave_results("https://huggingface.co/x"))

    def mock_async_client(**kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler), **cast(Any, kwargs))

    monkeypatch.setattr(web.httpx, "AsyncClient", mock_async_client)

    async with provider as tools:
        search_tool = next(tool for tool in tools if tool.name == "web_search")
        result = await run_web_search_tool(search_tool, "example")

    assert result.metadata is not None
    expected_excluded = 1 if "goggles" in requests[0].url.params else 0
    assert result.metadata.results_excluded == expected_excluded
    assert result.metadata.pages_returned == 1 - expected_excluded
    return requests[0]


async def test_web_provider_passes_exclude_domains_to_search(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_EXCLUDE_DOMAINS", "ignored.com")
    provider = WebToolProvider(brave_api_key="test-key", exclude_domains=["huggingface.co"])

    request = await search_through_provider(monkeypatch, provider)

    assert request.url.params["goggles"] == "$discard,site=huggingface.co"


async def test_web_provider_reads_exclude_domains_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_EXCLUDE_DOMAINS", " huggingface.co, ,github.com ")

    request = await search_through_provider(monkeypatch, WebToolProvider(brave_api_key="test-key"))

    assert request.url.params["goggles"] == "$discard,site=huggingface.co\n$discard,site=github.com"


async def test_web_provider_empty_exclude_domains_overrides_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_EXCLUDE_DOMAINS", "huggingface.co")

    request = await search_through_provider(monkeypatch, WebToolProvider(brave_api_key="test-key", exclude_domains=[]))

    assert "goggles" not in request.url.params

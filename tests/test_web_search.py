from collections.abc import Awaitable
from typing import cast

import httpx
import pytest
from pytest import MonkeyPatch
from tenacity import wait_fixed, wait_none

from stirrup.core.models import Tool, ToolResult
from stirrup.tools import web
from stirrup.tools.web import (
    FetchWebPageParams,
    WebFetchMetadata,
    WebSearchMetadata,
    WebSearchParams,
    _get_fetch_web_page_tool,
    _get_websearch_tool,
)


async def run_web_search_tool(
    tool: Tool[WebSearchParams, WebSearchMetadata],
    query: str,
) -> ToolResult[WebSearchMetadata]:
    result = tool.executor(WebSearchParams(query=query))
    return await cast(Awaitable[ToolResult[WebSearchMetadata]], result)


async def run_web_fetch_tool(
    tool: Tool[FetchWebPageParams, WebFetchMetadata],
    url: str,
) -> ToolResult[WebFetchMetadata]:
    result = tool.executor(FetchWebPageParams(url=url))
    return await cast(Awaitable[ToolResult[WebFetchMetadata]], result)


async def test_web_fetch_returns_tool_error_for_local_file_path() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, text="<html>unused</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_fetch_web_page_tool(client)
        result = await run_web_fetch_tool(tool, "/home/user/stageplot/preview.png")

    assert requests == []
    assert result.success is False
    assert result.metadata is not None
    assert result.metadata.pages_fetched == ["/home/user/stageplot/preview.png"]
    assert "only supports absolute http:// or https:// URLs" in result.content


async def test_web_fetch_still_fetches_absolute_http_urls() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            text="<html><body><main><p>Example body</p></main></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_fetch_web_page_tool(client)
        result = await run_web_fetch_tool(tool, "https://example.com/page")

    assert [str(request.url) for request in requests] == ["https://example.com/page"]
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.pages_fetched == ["https://example.com/page"]


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

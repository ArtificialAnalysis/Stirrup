from collections.abc import Awaitable
from typing import cast

import httpx
import pytest
from pytest import MonkeyPatch
from tenacity import wait_fixed, wait_none

from stirrup.core.models import Tool, ToolResult
from stirrup.tools import web
from stirrup.tools.web import WebSearchMetadata, WebSearchParams, _get_websearch_tool


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

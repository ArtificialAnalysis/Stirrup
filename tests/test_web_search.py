import httpx
import pytest
from pytest import MonkeyPatch
from tenacity import wait_none

from stirrup.tools import web
from stirrup.tools.web import WebSearchParams, _get_websearch_tool


async def test_web_search_retries_brave_429(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(web, "WEB_SEARCH_RETRY_WAIT", wait_none())
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
        result = await tool.executor(WebSearchParams(query="example"))

    assert attempts == 2
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.pages_returned == 1
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
            await tool.executor(WebSearchParams(query="example"))

    assert attempts == 1

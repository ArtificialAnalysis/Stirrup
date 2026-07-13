"""Behavioral tests for secure web-page fetching."""

import socket
import threading
import time
from collections.abc import Awaitable
from typing import Any, cast

import httpx
import pytest
from pytest import MonkeyPatch
from tenacity import wait_none

from stirrup.core.models import Tool, ToolResult
from stirrup.tools import web
from stirrup.tools.web import (
    FetchWebPageParams,
    WebFetchMetadata,
    WebSearchMetadata,
    WebSearchParams,
    _get_fetch_web_page_tool,
)

PUBLIC_IPV4 = "93.184.216.34"
PUBLIC_IPV6 = "2606:2800:220:1:248:1893:25c8:1946"


async def run_web_fetch_tool(
    tool: Tool[FetchWebPageParams, WebFetchMetadata],
    url: str,
) -> ToolResult[WebFetchMetadata]:
    result = tool.executor(FetchWebPageParams(url=url))
    return await cast(Awaitable[ToolResult[WebFetchMetadata]], result)


def install_dns(monkeypatch: MonkeyPatch, *addresses: str) -> list[tuple[str, int]]:
    """Resolve every hostname to ``addresses`` and return the recorded calls."""
    calls: list[tuple[str, int]] = []

    def getaddrinfo(
        host: str,
        port: int,
        *,
        type: socket.SocketKind,
    ) -> list[tuple[Any, socket.SocketKind, int, str, tuple[Any, ...]]]:
        calls.append((host, port))
        records = []
        for address in addresses:
            if ":" in address:
                records.append((socket.AF_INET6, type, socket.IPPROTO_TCP, "", (address, port, 0, 0)))
            else:
                records.append((socket.AF_INET, type, socket.IPPROTO_TCP, "", (address, port)))
        return records

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return calls


def html_response(
    request: httpx.Request,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=request,
        headers=headers,
        text="<html><body><main><p>Example body</p></main></body></html>",
    )


@pytest.mark.parametrize(
    "url",
    [
        "/home/user/stageplot/preview.png",
        "https://example.com:not-a-port/page",
        "http://93.184.216.34:65536/page",
    ],
)
async def test_web_fetch_rejects_invalid_urls(url: str) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), url)

    assert result.success is False
    assert result.metadata is not None
    assert result.metadata.pages_fetched == [url]
    assert requests == []


@pytest.mark.parametrize(("scheme", "port"), [("http", 80), ("https", 443)])
async def test_web_fetch_connects_to_validated_ip_with_logical_host_and_sni(
    monkeypatch: MonkeyPatch,
    scheme: str,
    port: int,
) -> None:
    resolution_calls = install_dns(monkeypatch, PUBLIC_IPV4)
    requests: list[httpx.Request] = []
    logical_url = f"{scheme}://example.com/page"

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), logical_url)

    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.pages_fetched == [logical_url]
    assert resolution_calls == [("example.com", port)]
    assert len(requests) == 1
    assert str(requests[0].url) == f"{scheme}://{PUBLIC_IPV4}/page"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].headers["connection"] == "close"
    assert requests[0].extensions["sni_hostname"] == "example.com"


async def test_web_fetch_retries_validated_addresses_without_resolving_again(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(web, "WEB_FETCH_RETRY_WAIT", wait_none())
    second_address = "93.184.216.35"
    resolution_calls = install_dns(monkeypatch, PUBLIC_IPV4, second_address)
    requested_hosts: list[str] = []
    second_address_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal second_address_attempts
        requested_hosts.append(request.url.host)
        if request.url.host == PUBLIC_IPV4:
            raise httpx.ConnectError("first address unavailable", request=request)
        second_address_attempts += 1
        if second_address_attempts == 1:
            raise httpx.ReadTimeout("temporary read failure", request=request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://example.com/page")

    assert result.success is True
    assert resolution_calls == [("example.com", 443)]
    assert set(requested_hosts) == {PUBLIC_IPV4, second_address}
    assert second_address_attempts == 2


@pytest.mark.parametrize("address", [PUBLIC_IPV4, PUBLIC_IPV6])
async def test_web_fetch_does_not_send_or_retain_cookies(monkeypatch: MonkeyPatch, address: str) -> None:
    install_dns(monkeypatch, address)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"Set-Cookie": "session=untrusted; Path=/"} if len(requests) == 1 else None
        return html_response(request, headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = _get_fetch_web_page_tool(client)
        first_result = await run_web_fetch_tool(tool, "https://first.example/page")
        second_result = await run_web_fetch_tool(tool, "https://second.example/page")

        assert list(client.cookies.jar) == []

    assert first_result.success is True
    assert second_result.success is True
    assert [request.headers.get("cookie") for request in requests] == [None, None]


async def test_web_provider_isolates_fetch_from_proxies_and_search_state(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    install_dns(monkeypatch, PUBLIC_IPV4)
    client_configs: list[dict[str, object]] = []
    clients: list[httpx.AsyncClient] = []
    requests: list[tuple[str, httpx.Request]] = []
    real_async_client = httpx.AsyncClient

    def recording_async_client(**kwargs: object) -> httpx.AsyncClient:
        client_kind = "fetch" if kwargs.get("trust_env") is False else "search"

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append((client_kind, request))
            if client_kind == "search":
                return httpx.Response(200, request=request, json={"web": {"results": []}})
            return html_response(request)

        client_configs.append(kwargs)
        client = real_async_client(transport=httpx.MockTransport(handler), **cast(Any, kwargs))
        clients.append(client)
        return client

    monkeypatch.setattr(web.httpx, "AsyncClient", recording_async_client)

    async with web.WebToolProvider(brave_api_key="test-key") as tools:
        fetch_tool = next(tool for tool in tools if tool.name == "fetch_web_page")
        search_tool = next(tool for tool in tools if tool.name == "web_search")
        await run_web_fetch_tool(cast(Tool[FetchWebPageParams, WebFetchMetadata], fetch_tool), "https://example.com")
        search_result = search_tool.executor(WebSearchParams(query="example"))
        await cast(Awaitable[ToolResult[WebSearchMetadata]], search_result)

    fetch_configs = [config for config in client_configs if config.get("trust_env") is False]
    search_configs = [config for config in client_configs if "trust_env" not in config]
    assert len(fetch_configs) == 1
    assert len(search_configs) == 1
    assert fetch_configs[0]["follow_redirects"] is False
    assert search_configs[0]["follow_redirects"] is True
    assert [(kind, request.url.host) for kind, request in requests] == [
        ("fetch", PUBLIC_IPV4),
        ("search", "api.search.brave.com"),
    ]
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.88.99.2",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "[::1]",
        "[fd00::1]",
        "[fe80::1]",
        "[ff02::1]",
        "[2001:db8::1]",
        "[fec0::1]",
        "[3fff::1]",
        "[4000::1]",
        "[::]",
    ],
)
async def test_web_fetch_rejects_non_public_ip_literals(host: str) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), f"http://{host}/secret")

    assert result.success is False
    assert requests == []


@pytest.mark.parametrize(
    "addresses",
    [
        ("10.0.0.8",),
        (PUBLIC_IPV4, "10.0.0.8"),
        (PUBLIC_IPV4, "fec0::1"),
    ],
)
async def test_web_fetch_rejects_dns_with_any_non_public_address(
    monkeypatch: MonkeyPatch,
    addresses: tuple[str, ...],
) -> None:
    install_dns(monkeypatch, *addresses)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://mixed.example/secret")

    assert result.success is False
    assert requests == []


async def test_web_fetch_validates_redirect_destinations(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, request=request, headers={"Location": "http://127.0.0.1/secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://example.com/redirect")

    assert result.success is False
    assert len(requests) == 1
    assert requests[0].url.host == PUBLIC_IPV4


async def test_web_fetch_returns_tool_error_when_dns_resolution_fails(monkeypatch: MonkeyPatch) -> None:
    def getaddrinfo(host: str, port: int, *, type: socket.SocketKind) -> list[object]:
        del host, port, type
        raise socket.gaierror("host not found")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: html_response(request))) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://missing.example/page")

    assert result.success is False
    assert "Could not resolve destination host" in result.content


async def test_web_fetch_bounds_dns_resolution_by_provider_timeout(monkeypatch: MonkeyPatch) -> None:
    release_resolver = threading.Event()

    def getaddrinfo(
        host: str,
        port: int,
        *,
        type: socket.SocketKind,
    ) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]]:
        del host
        release_resolver.wait(timeout=0.4)
        return [(socket.AF_INET, type, socket.IPPROTO_TCP, "", (PUBLIC_IPV4, port))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    real_async_client = httpx.AsyncClient

    def mock_async_client(**kwargs: object) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(lambda request: html_response(request)),
            **cast(Any, kwargs),
        )

    monkeypatch.setattr(web.httpx, "AsyncClient", mock_async_client)
    started_at = time.perf_counter()

    try:
        async with web.WebToolProvider(timeout=0.05) as tools:
            tool = next(tool for tool in tools if tool.name == "fetch_web_page")
            result = await run_web_fetch_tool(tool, "https://example.com/page")
    finally:
        release_resolver.set()

    assert result.success is False
    assert time.perf_counter() - started_at < 0.2
    assert "Timed out resolving destination host" in result.content


async def test_web_fetch_limits_redirects(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        redirect_number = int(request.url.path.removeprefix("/redirect/"))
        return httpx.Response(
            302,
            request=request,
            headers={"Location": f"/redirect/{redirect_number + 1}"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://example.com/redirect/0")

    assert result.success is False
    assert len(requests) == web.WEB_FETCH_MAX_REDIRECTS + 1

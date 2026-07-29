"""Behavioral tests for secure web-page fetching."""

import socket
import threading
import time
from collections.abc import AsyncIterator, Awaitable
from typing import Any, cast

import anyio
import httpx
import pytest
from anyio.lowlevel import checkpoint
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
)

PUBLIC_IPV4 = "93.184.216.34"
PUBLIC_IPV6 = "2606:2800:220:1:248:1893:25c8:1946"
EXPECTED_MAX_BODY_BYTES = 1024 * 1024
BODY_CHUNK_BYTES = 64 * 1024


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

    fetch_client, search_client = clients
    assert fetch_client.trust_env is False
    assert fetch_client.follow_redirects is False
    assert search_client.follow_redirects is True
    assert [(kind, request.url.host) for kind, request in requests] == [
        ("fetch", PUBLIC_IPV4),
        ("search", "api.search.brave.com"),
    ]
    assert all(client.is_closed for client in clients)


async def test_web_provider_shields_partial_startup_cleanup_from_cancellation(monkeypatch: MonkeyPatch) -> None:
    class StartupClient:
        def __init__(self) -> None:
            self.is_closed = False
            clients.append(self)

        async def __aenter__(self) -> "StartupClient":
            if len(clients) == 2:
                cancel_scope.cancel()
                await checkpoint()
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: object,
        ) -> None:
            del exc_type, exc_val, exc_tb
            await checkpoint()
            self.is_closed = True

    clients: list[StartupClient] = []

    def startup_client(**kwargs: object) -> StartupClient:
        del kwargs
        return StartupClient()

    monkeypatch.setattr(web.httpx, "AsyncClient", startup_client)
    provider = web.WebToolProvider(brave_api_key="test-key")

    with anyio.CancelScope() as cancel_scope:
        with pytest.raises(anyio.get_cancelled_exc_class()):
            await provider.__aenter__()

    assert len(clients) == 2
    assert clients[0].is_closed is True


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.0.0.8",
        "192.0.0.9",
        "192.0.0.10",
        "192.0.0.170",
        "192.0.0.171",
        "192.88.99.2",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "[::1]",
        "[fd00::1]",
        "[fe80::1]",
        "[ff02::1]",
        "[2001:1::1]",
        "[2001:1::2]",
        "[2001:3::1]",
        "[2001:4:112::1]",
        "[2001:20::1]",
        "[2001:30::1]",
        "[2001:db8::1]",
        "[2002::1]",
        "[64:ff9b:1::1]",
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
    ("host", "is_allowed"),
    [
        ("[::ffff:127.0.0.1]", False),
        (f"[::ffff:{PUBLIC_IPV4}]", True),
    ],
)
async def test_web_fetch_validates_embedded_ipv4_in_mapped_ipv6(host: str, is_allowed: bool) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return html_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), f"https://{host}/page")

    assert result.success is is_allowed
    assert len(requests) == int(is_allowed)


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
    assert "Web fetch timed out" in result.content


async def test_web_fetch_timeout_is_a_total_deadline_across_retries(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    monkeypatch.setattr(web, "WEB_FETCH_RETRY_WAIT", wait_fixed(1))
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("unavailable", request=request)

    started_at = time.perf_counter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(
            _get_fetch_web_page_tool(client, timeout=0.05),
            "https://example.com/page",
        )

    assert result.success is False
    assert time.perf_counter() - started_at < 0.2
    assert len(requests) == 1
    assert "Web fetch timed out" in result.content


async def test_web_fetch_timeout_closes_response_and_clears_cookies(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)

    class BlockingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.is_closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"partial body"
            await anyio.sleep_forever()

        async def aclose(self) -> None:
            await checkpoint()
            self.is_closed = True

    stream = BlockingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"Set-Cookie": "session=untrusted; Path=/"},
            stream=stream,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(
            _get_fetch_web_page_tool(client, timeout=0.05),
            "https://example.com/page",
        )
        retained_cookies = list(client.cookies.jar)

    assert result.success is False
    assert "Web fetch timed out" in result.content
    assert stream.is_closed is True
    assert retained_cookies == []


async def test_web_fetch_extraction_uses_remaining_total_deadline(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    extraction_started = threading.Event()
    release_extraction = threading.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.15)
        return html_response(request)

    def extract(body: bytes, *, output_format: str) -> str:
        del body, output_format
        extraction_started.set()
        release_extraction.wait(timeout=1)
        return "late extraction"

    monkeypatch.setattr(web.trafilatura, "extract", extract)
    started_at = time.perf_counter()

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await run_web_fetch_tool(
                _get_fetch_web_page_tool(client, timeout=0.3),
                "https://example.com/page",
            )
    finally:
        release_extraction.set()

    assert result.success is False
    assert extraction_started.is_set()
    assert time.perf_counter() - started_at < 0.4
    assert "Web fetch timed out" in result.content


async def test_web_fetch_rejects_compressed_response_without_reading_it(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    requests: list[httpx.Request] = []

    class CompressedStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.was_read = False
            self.is_closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            self.was_read = True
            yield b"compressed bytes that must not be decoded"

        async def aclose(self) -> None:
            self.is_closed = True

    stream = CompressedStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request, headers={"Content-Encoding": "gzip"}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://example.com/page")

    assert result.success is False
    assert "Unsupported Content-Encoding" in result.content
    assert requests[0].headers["accept-encoding"] == "identity"
    assert stream.was_read is False
    assert stream.is_closed is True


async def test_web_fetch_caps_response_body_before_extraction(monkeypatch: MonkeyPatch) -> None:
    install_dns(monkeypatch, PUBLIC_IPV4)
    extracted_bodies: list[bytes] = []

    class RecordingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.bytes_sent = 0
            self.is_closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            chunk = b"x" * BODY_CHUNK_BYTES
            for _ in range(EXPECTED_MAX_BODY_BYTES // len(chunk) + 10):
                self.bytes_sent += len(chunk)
                yield chunk

        async def aclose(self) -> None:
            self.is_closed = True

    stream = RecordingStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, stream=stream)

    def extract(body: bytes, *, output_format: str) -> str:
        assert output_format == "markdown"
        extracted_bodies.append(body)
        return "bounded body"

    monkeypatch.setattr(web.trafilatura, "extract", extract)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_web_fetch_tool(_get_fetch_web_page_tool(client), "https://example.com/page")

    assert result.success is True
    assert len(extracted_bodies[0]) == EXPECTED_MAX_BODY_BYTES
    # Detecting truncation costs at most one chunk read past the cap.
    assert stream.bytes_sent <= EXPECTED_MAX_BODY_BYTES + BODY_CHUNK_BYTES
    assert stream.is_closed is True
    assert "truncated" in result.content


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

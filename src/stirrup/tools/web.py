"""Web tools for fetching pages and searching the web.

This module provides web_fetch and web_search tools with a WebToolProvider
that isolates fetch transport and state from search and safely cleans up resources.

Example usage:
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    # As part of DEFAULT_TOOLS in Agent
    client = ChatCompletionsClient(model="gpt-5")
    agent = Agent(
        client=client,
        name="assistant",
        tools=DEFAULT_TOOLS,  # Includes WebToolProvider
    )

    # Standalone usage
    async with WebToolProvider() as provider:
        tools = provider.get_tools()
"""

import ipaddress
import os
import socket
from contextlib import AsyncExitStack
from functools import partial
from html import escape
from types import TracebackType
from typing import Annotated, Any

import httpx
import trafilatura
from anyio import CancelScope, fail_after, to_thread
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from stirrup.core.models import Tool, ToolProvider, ToolResult
from stirrup.utils.text import truncate_msg

__all__ = ["WebToolProvider"]

# Constants
MAX_LENGTH_WEB_FETCH_HTML = 40000
MAX_LENGTH_WEB_SEARCH_RESULTS = 40000
DEFAULT_WEBFETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
}
WEB_FETCH_TIMEOUT = 60 * 3
WEB_FETCH_MAX_REDIRECTS = 10
WEB_FETCH_RETRY_WAIT = wait_exponential(multiplier=1, min=1, max=10)
_WEB_FETCH_MAX_BODY_BYTES = 1024 * 1024
_WEB_FETCH_TRUNCATION_NOTICE = (
    f"\n... This page has been truncated to its first {_WEB_FETCH_MAX_BODY_BYTES} bytes before extraction ...\n"
)
_WEB_FETCH_BODY_CHUNK_BYTES = 64 * 1024
# Keep IANA special-purpose policy stable across supported ``ipaddress`` versions.
_EXPLICITLY_NON_PUBLIC_NETWORKS = (
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),  # Deprecated 6to4 Relay Anycast
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/23"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("3fff::/20"),  # Documentation
)
WEB_SEARCH_TIMEOUT = 60 * 3
WEB_SEARCH_RETRY_WAIT = wait_random_exponential(multiplier=1, min=1, max=15)


# =============================================================================
# Web Fetch Tool
# =============================================================================


class FetchWebPageParams(BaseModel):
    """Parameters for web page fetch tool."""

    url: Annotated[str, Field(description="Full HTTP or HTTPS URL of the web page to fetch and extract")]


class WebFetchMetadata(BaseModel):
    """Metadata for web fetch tool tracking URLs fetched.

    Implements Addable protocol for aggregation across multiple fetches.
    """

    num_uses: int = 1
    pages_fetched: list[str] = Field(default_factory=list)

    def __add__(self, other: "WebFetchMetadata") -> "WebFetchMetadata":
        return WebFetchMetadata(
            num_uses=self.num_uses + other.num_uses,
            pages_fetched=self.pages_fetched + other.pages_fetched,
        )


class _WebFetchError(Exception):
    """An expected destination validation or redirect failure."""


def _ensure_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        _ensure_public_ip(address.ipv4_mapped)
        return

    # is_global rejects private, loopback, link-local, and unspecified addresses.
    # ipaddress still labels some reserved, multicast, and legacy IPv6 site-local
    # addresses as global, so reject those explicitly.
    is_ipv6_site_local = isinstance(address, ipaddress.IPv6Address) and address.is_site_local
    is_explicitly_non_public = any(address in network for network in _EXPLICITLY_NON_PUBLIC_NETWORKS)
    if (
        not address.is_global
        or address.is_reserved
        or address.is_multicast
        or is_ipv6_site_local
        or is_explicitly_non_public
    ):
        raise _WebFetchError(f"Destination resolves to a non-public IP address: {address}")


async def _resolve_public_addresses(url: httpx.URL) -> tuple[str, ...]:
    """Resolve a URL and reject the entire destination if any address is non-public."""
    if url.scheme not in {"http", "https"} or not url.host:
        raise _WebFetchError("fetch_web_page only supports absolute http:// or https:// URLs.")
    if url.port is not None and not 0 <= url.port <= 65535:
        raise _WebFetchError("Destination port must be between 0 and 65535.")

    host = url.host
    port = url.port or (443 if url.scheme == "https" else 80)

    try:
        literal_address = ipaddress.ip_address(host)
    except ValueError:
        try:
            address_info = await to_thread.run_sync(
                partial(
                    socket.getaddrinfo,
                    host,
                    port,
                    type=socket.SOCK_STREAM,
                ),
                abandon_on_cancel=True,
            )
        except OSError as exc:
            raise _WebFetchError(f"Could not resolve destination host {host}: {exc}") from exc

        if not address_info:
            raise _WebFetchError(f"Could not resolve destination host {host}.") from None

        public_addresses: list[str] = []
        for _family, _type, _protocol, _canonical_name, socket_address in address_info:
            try:
                resolved_address = ipaddress.ip_address(socket_address[0])
            except ValueError as exc:
                raise _WebFetchError(f"DNS returned an invalid IP address for {host}.") from exc
            _ensure_public_ip(resolved_address)
            public_addresses.append(str(resolved_address))
        return tuple(dict.fromkeys(public_addresses))
    else:
        _ensure_public_ip(literal_address)
        return (str(literal_address),)


def _get_fetch_web_page_tool(
    client: httpx.AsyncClient,
    *,
    timeout: float = WEB_FETCH_TIMEOUT,
) -> Tool[FetchWebPageParams, WebFetchMetadata]:
    """Create a web page fetching tool that extracts main content as markdown.

    Args:
        client: Dedicated fetch client managed by ``WebToolProvider``
        timeout: Total deadline in seconds for fetching and extracting the page

    Returns:
        Tool configured to fetch web pages and extract clean markdown content
    """

    async def _read_response_body(response: httpx.Response) -> tuple[bytes, bool]:
        """Read at most ``_WEB_FETCH_MAX_BODY_BYTES`` and report whether the body was truncated."""
        content_encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if content_encoding != "identity":
            # httpx's GZipDecoder expands a whole chunk with no max_length, so a compressed
            # body cannot be bounded by the byte cap below. Refuse rather than decompress.
            raise _WebFetchError(f"Unsupported Content-Encoding in web fetch response: {content_encoding}")

        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=_WEB_FETCH_BODY_CHUNK_BYTES):
            body.extend(chunk)
            if len(body) > _WEB_FETCH_MAX_BODY_BYTES:
                return bytes(body[:_WEB_FETCH_MAX_BODY_BYTES]), True
        return bytes(body), False

    async def _send_request(
        logical_url: httpx.URL,
        destination_address: str,
    ) -> tuple[httpx.Response, tuple[bytes, bool] | None]:
        connection_url = logical_url.copy_with(host=destination_address)
        request = client.build_request(
            "GET",
            connection_url,
            headers={
                **DEFAULT_WEBFETCH_HEADERS,
                # httpcore keys connection reuse on the pinned-IP origin and ignores
                # sni_hostname, so keep-alive would let a connection authenticated for
                # one hostname serve another.
                "Connection": "close",
                "Host": logical_url.netloc.decode("ascii"),
            },
            extensions={"sni_hostname": logical_url.raw_host.decode("ascii")},
        )
        request.headers.pop("cookie", None)
        response: httpx.Response | None = None
        try:
            response = await client.send(request, follow_redirects=False, stream=True)
            if response.has_redirect_location:
                return response, None
            response.raise_for_status()
            return response, await _read_response_body(response)
        finally:
            try:
                if response is not None:
                    with CancelScope(shield=True):
                        await response.aclose()
            finally:
                # Fetch never sends or retains cookies. The client is dedicated to
                # this tool, so clearing the whole jar also handles HTTPX's special
                # domain representation for IPv6 literals.
                client.cookies.clear()

    async def _request_hop(
        logical_url: httpx.URL,
        public_addresses: tuple[str, ...],
    ) -> tuple[httpx.Response, tuple[bytes, bool] | None]:
        """Try each validated address in turn, retrying the whole hop on network failure."""
        retryer = AsyncRetrying(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            stop=stop_after_attempt(3),
            wait=WEB_FETCH_RETRY_WAIT,
            reraise=True,
        )
        async for retry_attempt in retryer:
            with retry_attempt:
                last_error: httpx.TimeoutException | httpx.NetworkError | None = None
                for destination_address in public_addresses:
                    try:
                        return await _send_request(logical_url, destination_address)
                    except (httpx.TimeoutException, httpx.NetworkError) as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error

        raise RuntimeError("Web fetch retry loop completed without a response.")

    async def _fetch_with_redirects(url: str) -> tuple[bytes, bool]:
        current_url = httpx.URL(url).copy_with(fragment=None)
        redirects_followed = 0

        while True:
            public_addresses = await _resolve_public_addresses(current_url)
            response, body = await _request_hop(current_url, public_addresses)
            if not response.has_redirect_location:
                if body is None:
                    raise RuntimeError("Web fetch completed without a response body.")
                return body

            if redirects_followed >= WEB_FETCH_MAX_REDIRECTS:
                raise _WebFetchError(f"Too many redirects (maximum {WEB_FETCH_MAX_REDIRECTS}).")

            current_url = current_url.join(response.headers["location"])
            redirects_followed += 1

    async def fetch_web_page_executor(params: FetchWebPageParams) -> ToolResult[WebFetchMetadata]:
        """Fetch web page and extract main content as markdown using trafilatura."""
        try:
            with fail_after(timeout):
                response_body, is_truncated = await _fetch_with_redirects(params.url)
                body_md = (
                    await to_thread.run_sync(
                        partial(trafilatura.extract, response_body, output_format="markdown"),
                        abandon_on_cancel=True,
                    )
                    or ""
                )
                if is_truncated:
                    body_md += _WEB_FETCH_TRUNCATION_NOTICE
            return ToolResult(
                content=f"<web_fetch><url>{params.url}</url><body>"
                f"{truncate_msg(body_md, MAX_LENGTH_WEB_FETCH_HTML)}</body></web_fetch>",
                metadata=WebFetchMetadata(pages_fetched=[params.url]),
            )
        except TimeoutError:
            error_message = f"Web fetch timed out after {timeout:g} seconds."
        except (httpx.HTTPError, httpx.InvalidURL, _WebFetchError) as exc:
            error_message = str(exc)

        return ToolResult(
            content=f"<web_fetch><url>{params.url}</url><error>"
            f"{truncate_msg(error_message, MAX_LENGTH_WEB_FETCH_HTML)}</error></web_fetch>",
            success=False,
            metadata=WebFetchMetadata(pages_fetched=[params.url]),
        )

    return Tool[FetchWebPageParams, WebFetchMetadata](
        name="fetch_web_page",
        description=(
            "Fetch and extract the main content from a web page as markdown. Cannot reach private, loopback, "
            "or other non-public network addresses. Returns body text or error as XML."
        ),
        parameters=FetchWebPageParams,
        executor=fetch_web_page_executor,
    )


# =============================================================================
# Web Search Tool
# =============================================================================


class WebSearchParams(BaseModel):
    """Parameters for web search tool."""

    query: Annotated[
        str, Field(description="Natural language search query for Brave Search (similar to Google search syntax)")
    ]


class WebSearchMetadata(BaseModel):
    """Metadata for web search tool tracking search results.

    Implements Addable protocol for aggregation across multiple searches.
    """

    num_uses: int = 1
    pages_returned: int = 0
    retry_idle_for: float = 0.0

    def __add__(self, other: "WebSearchMetadata") -> "WebSearchMetadata":
        return WebSearchMetadata(
            num_uses=self.num_uses + other.num_uses,
            pages_returned=self.pages_returned + other.pages_returned,
            retry_idle_for=self.retry_idle_for + other.retry_idle_for,
        )


def _should_retry_web_search(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return False


def _get_websearch_tool(
    brave_api_key: str | None, client: httpx.AsyncClient | None = None
) -> Tool[WebSearchParams, WebSearchMetadata]:
    """Create a web search tool using Brave Search API.

    Args:
        brave_api_key: Brave Search API key, or None to use BRAVE_API_KEY environment variable
        client: Optional shared httpx.AsyncClient for connection pooling

    Returns:
        Tool configured to search the web and return top 5 results as XML

    Raises:
        RuntimeError: If no API key is provided or found in environment
    """
    if brave_api_key is None:
        brave_api_key = os.getenv("BRAVE_API_KEY")

    if brave_api_key is None:
        raise RuntimeError("No Brave Search API key provided.")

    async def _search_once(query: str, http_client: httpx.AsyncClient) -> dict:
        """Execute one Brave Search API request."""
        response = await http_client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": brave_api_key,
                "Accept": "application/json",
            },
            params={"q": query, "count": 5},
        )
        response.raise_for_status()
        return response.json()

    async def _search(query: str, http_client: httpx.AsyncClient) -> tuple[dict, float]:
        """Execute Brave Search API request with retries and return retry sleep time."""
        data: dict | None = None
        retryer = AsyncRetrying(
            retry=retry_if_exception(_should_retry_web_search),
            stop=stop_after_attempt(10),
            wait=WEB_SEARCH_RETRY_WAIT,
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                data = await _search_once(query, http_client)

        if data is None:
            raise RuntimeError("Brave search retry loop completed without a result.")
        return data, float(retryer.statistics.get("idle_for", 0.0))

    async def websearch_executor(params: WebSearchParams) -> ToolResult[WebSearchMetadata]:
        """Execute web search and format results as XML with title, URL, and description."""
        # Use provided client or create temporary one for backward compatibility
        if client is not None:
            data, retry_idle_for = await _search(params.query, client)
        else:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as temp_client:
                data, retry_idle_for = await _search(params.query, temp_client)

        results = data.get("web", {}).get("results", [])
        results_xml = (
            "<results>\n"
            + "\n".join(
                (
                    "<result>"
                    f"\n<title>{escape(result.get('title', '') or '')}</title>"
                    f"\n<url>{escape(result.get('url', '') or '')}</url>"
                    f"\n<description>{escape(result.get('description', '') or '')}</description>"
                    "\n</result>"
                )
                for result in results
            )
            + "\n</results>"
        )

        return ToolResult(
            content=truncate_msg(results_xml, MAX_LENGTH_WEB_SEARCH_RESULTS),
            metadata=WebSearchMetadata(pages_returned=len(results), retry_idle_for=retry_idle_for),
        )

    return Tool[WebSearchParams, WebSearchMetadata](
        name="web_search",
        description="Search the web using Brave Search API. Returns top 5 results with title, URL, and description as XML.",
        parameters=WebSearchParams,
        executor=websearch_executor,
    )


# =============================================================================
# WebToolProvider
# =============================================================================


class WebToolProvider(ToolProvider):
    """Provides web tools (web_fetch, web_search) with managed HTTP resources.

    WebToolProvider implements the Tool lifecycle protocol (has_lifecycle=True),
    so it can be used directly in Agent's tools list. Web fetch ignores environment
    proxies and does not send or retain cookies; its transport and state are isolated
    from web search. Provider resources are safely cleaned up after interrupted startup.

    Usage as Tool in Agent (preferred):
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[LocalCodeExecToolProvider(), WebToolProvider(), CALCULATOR_TOOL],
        )

        async with agent.session(output_dir="./output") as session:
            await session.run("Search the web and fetch a page")

    Standalone usage:
        async with WebToolProvider() as provider:
            tools = provider.get_tools()
    """

    def __init__(
        self,
        *,
        timeout: float = 60 * 3,
        brave_api_key: str | None = None,
    ) -> None:
        """Initialize WebToolProvider.

        Args:
            timeout: Total web fetch timeout and per-operation web search timeout in seconds (default: 180)
            brave_api_key: Brave Search API key for web_search tool.
                          If None, uses BRAVE_API_KEY environment variable.
                          Web search is only available if API key is provided.
        """
        self._timeout = timeout
        self._brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self._fetch_client: httpx.AsyncClient | None = None
        self._search_client: httpx.AsyncClient | None = None
        self._client_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        """Enter async context: create HTTP clients and return web tools.

        Returns:
            List of Tool objects (web_fetch, and web_search if API key available).
        """
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            self._fetch_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=False,
                    trust_env=False,
                )
            )
            if self._brave_api_key:
                self._search_client = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
                )
            tools = self.get_tools()
        except BaseException:
            self._fetch_client = None
            self._search_client = None
            with CancelScope(shield=True):
                await stack.aclose()
            raise
        self._client_stack = stack
        return tools

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: close HTTP clients."""
        stack = self._client_stack
        self._client_stack = None
        self._fetch_client = None
        self._search_client = None
        if stack is not None:
            await stack.__aexit__(exc_type, exc_val, exc_tb)

    def get_tools(self) -> list[Tool[Any, Any]]:
        """Get web tools configured with the managed HTTP clients.

        Returns:
            List containing web_fetch tool, and web_search tool if API key is available.

        Raises:
            RuntimeError: If called before entering context.
        """
        if self._fetch_client is None:
            raise RuntimeError("WebToolProvider not started. Use 'async with' first.")

        tools: list[Tool[Any, Any]] = [_get_fetch_web_page_tool(self._fetch_client, timeout=self._timeout)]

        # Only add web_search if API key is available
        if self._brave_api_key:
            if self._search_client is None:
                raise RuntimeError("Web search client not started. Use 'async with' first.")
            tools.append(_get_websearch_tool(self._brave_api_key, self._search_client))

        return tools

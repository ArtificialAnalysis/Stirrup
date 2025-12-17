"""Web tools for fetching pages and searching the web.

This module provides web_fetch and web_search tools with a WebToolProvider
class that manages the shared HTTP client lifecycle.

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

import os
from html import escape
from types import TracebackType
from typing import Annotated, Any

import httpx
import trafilatura
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
WEB_FETCH_TIMEOUT = 60 * 3
WEB_SEARCH_TIMEOUT = 60 * 3


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


def _get_fetch_web_page_tool(client: httpx.AsyncClient | None = None) -> Tool[FetchWebPageParams, WebFetchMetadata]:
    """Create a web page fetching tool that extracts main content as markdown.

    Args:
        client: Optional shared httpx.AsyncClient for connection pooling

    Returns:
        Tool configured to fetch web pages and extract clean markdown content
    """

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch(url: str, http_client: httpx.AsyncClient) -> httpx.Response:
        """Execute HTTP GET request with automatic retries on network errors."""
        response = await http_client.get(url, headers=DEFAULT_WEBFETCH_HEADERS)
        response.raise_for_status()
        return response

    async def fetch_web_page_executor(params: FetchWebPageParams) -> ToolResult[WebFetchMetadata]:
        """Fetch web page and extract main content as markdown using trafilatura."""
        try:
            # Use provided client or create temporary one for backward compatibility
            if client is not None:
                response = await _fetch(params.url, client)
            else:
                async with httpx.AsyncClient(
                    headers=DEFAULT_WEBFETCH_HEADERS,
                    follow_redirects=True,
                    timeout=WEB_FETCH_TIMEOUT,
                ) as temp_client:
                    response = await _fetch(params.url, temp_client)

            body_md = trafilatura.extract(response.text, output_format="markdown") or ""
            return ToolResult(
                content=f"<web_fetch><url>{params.url}</url><body>"
                f"{truncate_msg(body_md, MAX_LENGTH_WEB_FETCH_HTML)}</body></web_fetch>",
                metadata=WebFetchMetadata(pages_fetched=[params.url]),
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                content=f"<web_fetch><url>{params.url}</url><error>"
                f"{truncate_msg(str(exc), MAX_LENGTH_WEB_FETCH_HTML)}</error></web_fetch>",
                success=False,
                metadata=WebFetchMetadata(pages_fetched=[params.url]),
            )

    return Tool[FetchWebPageParams, WebFetchMetadata](
        name="fetch_web_page",
        description="Fetch and extract the main content from a web page as markdown. Returns body text or error as XML.",
        parameters=FetchWebPageParams,
        executor=fetch_web_page_executor,  # ty: ignore[invalid-argument-type]
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

    def __add__(self, other: "WebSearchMetadata") -> "WebSearchMetadata":
        return WebSearchMetadata(
            num_uses=self.num_uses + other.num_uses,
            pages_returned=self.pages_returned + other.pages_returned,
        )


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

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=4, min=1, max=3),
        reraise=True,
    )
    async def _search(query: str, http_client: httpx.AsyncClient) -> dict:
        """Execute Brave Search API request with automatic retries on network errors."""
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

    async def websearch_executor(params: WebSearchParams) -> ToolResult[WebSearchMetadata]:
        """Execute web search and format results as XML with title, URL, and description."""
        # Use provided client or create temporary one for backward compatibility
        if client is not None:
            data = await _search(params.query, client)
        else:
            async with httpx.AsyncClient(timeout=WEB_SEARCH_TIMEOUT) as temp_client:
                data = await _search(params.query, temp_client)

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
            metadata=WebSearchMetadata(pages_returned=len(results)),
        )

    return Tool[WebSearchParams, WebSearchMetadata](
        name="web_search",
        description="Search the web using Brave Search API. Returns top 5 results with title, URL, and description as XML.",
        parameters=WebSearchParams,
        executor=websearch_executor,  # ty: ignore[invalid-argument-type]
    )


# =============================================================================
# WebToolProvider
# =============================================================================


class WebToolProvider(ToolProvider):
    """Provides web tools (web_fetch, web_search) with managed HTTP client lifecycle.

    WebToolProvider implements the Tool lifecycle protocol (has_lifecycle=True),
    so it can be used directly in Agent's tools list. It creates an httpx.AsyncClient
    on __aenter__ and returns the web tools.

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
            timeout: HTTP timeout in seconds (default: 180)
            brave_api_key: Brave Search API key for web_search tool.
                          If None, uses BRAVE_API_KEY environment variable.
                          Web search is only available if API key is provided.
        """
        self._timeout = timeout
        self._brave_api_key = brave_api_key or os.getenv("BRAVE_API_KEY")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        """Enter async context: create HTTP client and return web tools.

        Returns:
            List of Tool objects (web_fetch, and web_search if API key available).
        """
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
        )
        await self._client.__aenter__()
        return self.get_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: close HTTP client."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def get_tools(self) -> list[Tool[Any, Any]]:
        """Get web tools configured with the managed HTTP client.

        Returns:
            List containing web_fetch tool, and web_search tool if API key is available.

        Raises:
            RuntimeError: If called before entering context.
        """
        if self._client is None:
            raise RuntimeError("WebToolProvider not started. Use 'async with' first.")

        tools: list[Tool[Any, Any]] = [_get_fetch_web_page_tool(self._client)]

        # Only add web_search if API key is available
        if self._brave_api_key:
            tools.append(_get_websearch_tool(self._brave_api_key, self._client))

        return tools

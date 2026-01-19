"""Browser automation tool provider using browser-use library.

This module provides BrowserUseToolProvider, a ToolProvider that manages a browser
session and exposes browser automation actions as individual Tool objects.

Example usage:
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools.browser_use import BrowserUseToolProvider

    client = ChatCompletionsClient(model="gpt-5")
    agent = Agent(
        client=client,
        name="browser_agent",
        tools=[*DEFAULT_TOOLS, BrowserUseToolProvider()],
    )

    async with agent.session() as session:
        await session.run("Go to google.com and search for 'AI agents'")

Requires browser-use dependency (`uv add 'stirrup[browser]'`).
"""

import asyncio
import os
import urllib.parse
from types import TracebackType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from stirrup.core.models import EmptyParams, ImageContentBlock, Tool, ToolProvider, ToolResult, ToolUseCountMetadata

try:
    from browser_use import BrowserSession
    from browser_use.browser.events import (
        ClickElementEvent,
        GoBackEvent,
        NavigateToUrlEvent,
        ScrollEvent,
        ScrollToTextEvent,
        SendKeysEvent,
        SwitchTabEvent,
        TypeTextEvent,
    )
except ImportError as e:
    raise ImportError(
        "Requires installation of the browser extra. Install with (for example): "
        "`uv pip install stirrup[browser]` or `uv add stirrup[browser]`",
    ) from e


__all__ = [
    "BrowserUseToolProvider",
    "InputTextMetadata",
    "NavigateMetadata",
    "SearchMetadata",
]


# =============================================================================
# Parameter Models
# =============================================================================


class SearchParams(BaseModel):
    """Parameters for web search."""

    query: Annotated[str, Field(description="Search query string")]
    engine: Annotated[
        Literal["google", "duckduckgo", "bing"],
        Field(default="google", description="Search engine to use"),
    ] = "google"


class NavigateParams(BaseModel):
    """Parameters for URL navigation."""

    url: Annotated[str, Field(description="URL to navigate to")]
    new_tab: Annotated[bool, Field(default=False, description="Open in new tab")] = False


class ClickParams(BaseModel):
    """Parameters for clicking an element."""

    index: Annotated[int, Field(description="Element index from the page snapshot")]


class InputTextParams(BaseModel):
    """Parameters for typing text into an element."""

    index: Annotated[int, Field(description="Element index from the page snapshot")]
    text: Annotated[str, Field(description="Text to input")]
    clear_first: Annotated[bool, Field(default=True, description="Clear existing text before typing")] = True


class ScrollParams(BaseModel):
    """Parameters for scrolling the page."""

    direction: Annotated[Literal["up", "down"], Field(description="Scroll direction")]
    amount: Annotated[int, Field(default=500, description="Pixels to scroll")] = 500


class FindTextParams(BaseModel):
    """Parameters for finding and scrolling to text."""

    text: Annotated[str, Field(description="Text to find on the page")]


class SendKeysParams(BaseModel):
    """Parameters for sending keyboard keys."""

    keys: Annotated[str, Field(description="Keys to send (e.g., 'Enter', 'Escape', 'Tab', 'ArrowDown')")]


class EvaluateJsParams(BaseModel):
    """Parameters for executing JavaScript."""

    script: Annotated[str, Field(description="JavaScript code to execute")]


class SwitchTabParams(BaseModel):
    """Parameters for switching browser tabs."""

    index: Annotated[int, Field(description="Tab index to switch to (0-based)")]


class WaitParams(BaseModel):
    """Parameters for waiting."""

    seconds: Annotated[int, Field(default=3, description="Seconds to wait (max 30)")] = 3


# =============================================================================
# Metadata
# =============================================================================


class NavigateMetadata(ToolUseCountMetadata):
    """Metadata tracking URLs visited."""

    urls: list[str] = Field(default_factory=list)

    def __add__(self, other: "NavigateMetadata") -> "NavigateMetadata":  # type: ignore[override]
        return NavigateMetadata(
            num_uses=self.num_uses + other.num_uses,
            urls=self.urls + other.urls,
        )


class SearchMetadata(ToolUseCountMetadata):
    """Metadata tracking search queries."""

    queries: list[str] = Field(default_factory=list)

    def __add__(self, other: "SearchMetadata") -> "SearchMetadata":  # type: ignore[override]
        return SearchMetadata(
            num_uses=self.num_uses + other.num_uses,
            queries=self.queries + other.queries,
        )


class InputTextMetadata(ToolUseCountMetadata):
    """Metadata tracking text inputs."""

    texts: list[str] = Field(default_factory=list)

    def __add__(self, other: "InputTextMetadata") -> "InputTextMetadata":  # type: ignore[override]
        return InputTextMetadata(
            num_uses=self.num_uses + other.num_uses,
            texts=self.texts + other.texts,
        )


# =============================================================================
# BrowserUseToolProvider
# =============================================================================


class BrowserUseToolProvider(ToolProvider):
    """Browser automation tool provider using browser-use library.

    Provides tools for:
    - Navigation: search, navigate, go_back, wait
    - Page Interaction: click, input_text, scroll, find_text, send_keys
    - JavaScript: evaluate_js
    - Tab Management: switch_tab
    - Content Extraction: snapshot, screenshot, get_url

    Example:
        from stirrup.tools.browser_use import BrowserUseToolProvider

        agent = Agent(
            client=client,
            name="browser_agent",
            tools=[BrowserUseToolProvider(headless=False)],
        )

        async with agent.session() as session:
            await session.run("Navigate to example.com and click the first link")

    """

    def __init__(
        self,
        *,
        headless: bool = True,
        disable_security: bool = False,
        executable_path: str | None = None,
        cdp_url: str | None = None,
        use_cloud: bool = False,
        tool_prefix: str = "browser",
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize BrowserUseToolProvider.

        Args:
            headless: Run browser in headless mode (default: True)
            disable_security: Disable browser security features (default: False)
            executable_path: Path to Chrome/Chromium executable
            cdp_url: Chrome DevTools Protocol URL for remote connection
            use_cloud: Use Browser Use cloud browser (requires BROWSER_USE_API_KEY env var)
            tool_prefix: Prefix for tool names (default: "browser")
            extra_args: Additional Chromium command line arguments

        """
        self._headless = headless
        self._disable_security = disable_security
        self._executable_path = executable_path
        self._cdp_url = cdp_url
        self._use_cloud = use_cloud
        self._tool_prefix = tool_prefix
        self._extra_args = extra_args

        self._session: BrowserSession | None = None

    def _tool_name(self, name: str) -> str:
        """Generate prefixed tool name."""
        return f"{self._tool_prefix}_{name}" if self._tool_prefix else name

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        """Enter async context: start browser and return tools."""
        if self._use_cloud and not os.environ.get("BROWSER_USE_API_KEY"):
            raise ValueError(
                "BROWSER_USE_API_KEY environment variable is required when use_cloud=True. "
                "Get your API key from https://cloud.browser-use.com"
            )
        self._session = BrowserSession(  # type: ignore[call-overload]
            headless=self._headless,
            disable_security=self._disable_security,
            executable_path=self._executable_path,
            cdp_url=self._cdp_url,
            use_cloud=self._use_cloud,
            args=self._extra_args,
        )
        await self._session.start()
        return self._build_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: close browser."""
        if self._session:
            await self._session.stop()
            self._session = None

    def _build_tools(self) -> list[Tool[Any, Any]]:
        """Build all browser tools."""
        session = self._session
        if session is None:
            raise RuntimeError("Browser session not initialized")

        tools: list[Tool[Any, Any]] = []

        # --- Navigation Tools ---

        async def search_executor(params: SearchParams) -> ToolResult[SearchMetadata]:
            """Search the web using specified search engine."""

            search_urls = {
                "google": f"https://www.google.com/search?q={urllib.parse.quote_plus(params.query)}&udm=14",
                "duckduckgo": f"https://duckduckgo.com/?q={urllib.parse.quote_plus(params.query)}",
                "bing": f"https://www.bing.com/search?q={urllib.parse.quote_plus(params.query)}",
            }
            url = search_urls[params.engine]
            event = session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=False))
            await event
            return ToolResult(
                content=f"Searched {params.engine} for: {params.query}",
                metadata=SearchMetadata(queries=[params.query]),
            )

        tools.append(
            Tool(
                name=self._tool_name("search"),
                description="Search the web using Google, DuckDuckGo, or Bing.",
                parameters=SearchParams,
                executor=search_executor,
            )
        )

        async def navigate_executor(params: NavigateParams) -> ToolResult[NavigateMetadata]:
            """Navigate to a URL."""
            event = session.event_bus.dispatch(NavigateToUrlEvent(url=params.url, new_tab=params.new_tab))
            await event
            return ToolResult(
                content=f"Navigated to: {params.url}" + (" (new tab)" if params.new_tab else ""),
                metadata=NavigateMetadata(urls=[params.url]),
            )

        tools.append(
            Tool(
                name=self._tool_name("navigate"),
                description="Navigate to a URL. Optionally open in a new tab.",
                parameters=NavigateParams,
                executor=navigate_executor,
            )
        )

        async def go_back_executor(_: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
            """Go back in browser history."""
            event = session.event_bus.dispatch(GoBackEvent())
            await event
            return ToolResult(
                content="Navigated back",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("go_back"),
                description="Go back to the previous page in browser history.",
                parameters=EmptyParams,
                executor=go_back_executor,
            )
        )

        async def wait_executor(params: WaitParams) -> ToolResult[ToolUseCountMetadata]:
            """Wait for specified seconds."""

            wait_time = min(max(params.seconds, 1), 30)
            await asyncio.sleep(wait_time)
            return ToolResult(
                content=f"Waited for {wait_time} seconds",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("wait"),
                description="Wait for a specified number of seconds (1-30).",
                parameters=WaitParams,
                executor=wait_executor,
            )
        )

        # --- Page Interaction Tools ---

        async def click_executor(params: ClickParams) -> ToolResult[ToolUseCountMetadata]:
            """Click an element by index."""
            node = await session.get_element_by_index(params.index)
            if node is None:
                return ToolResult(
                    content=f"Element with index {params.index} not found",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
            event = session.event_bus.dispatch(ClickElementEvent(node=node))
            await event
            return ToolResult(
                content=f"Clicked element at index {params.index}",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("click"),
                description="Click an element by its index from the page snapshot.",
                parameters=ClickParams,
                executor=click_executor,
            )
        )

        async def input_text_executor(params: InputTextParams) -> ToolResult[InputTextMetadata]:
            """Input text into an element."""
            node = await session.get_element_by_index(params.index)
            if node is None:
                return ToolResult(
                    content=f"Element with index {params.index} not found",
                    success=False,
                    metadata=InputTextMetadata(texts=[params.text]),
                )
            event = session.event_bus.dispatch(
                TypeTextEvent(
                    node=node,
                    text=params.text,
                    clear=params.clear_first,
                )
            )
            await event
            return ToolResult(
                content=f"Typed text into element at index {params.index}",
                metadata=InputTextMetadata(texts=[params.text]),
            )

        tools.append(
            Tool(
                name=self._tool_name("input_text"),
                description="Type text into a form field or input element.",
                parameters=InputTextParams,
                executor=input_text_executor,
            )
        )

        async def scroll_executor(params: ScrollParams) -> ToolResult[ToolUseCountMetadata]:
            """Scroll the page."""
            event = session.event_bus.dispatch(
                ScrollEvent(
                    direction=params.direction,
                    amount=params.amount,
                )
            )
            await event
            return ToolResult(
                content=f"Scrolled {params.direction} by {params.amount} pixels",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("scroll"),
                description="Scroll the page up or down by a specified amount.",
                parameters=ScrollParams,
                executor=scroll_executor,
            )
        )

        async def find_text_executor(params: FindTextParams) -> ToolResult[ToolUseCountMetadata]:
            """Find and scroll to text on the page."""
            event = session.event_bus.dispatch(ScrollToTextEvent(text=params.text))
            await event
            return ToolResult(
                content=f"Scrolled to text: {params.text}",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("find_text"),
                description="Find specific text on the page and scroll to it.",
                parameters=FindTextParams,
                executor=find_text_executor,
            )
        )

        async def send_keys_executor(params: SendKeysParams) -> ToolResult[ToolUseCountMetadata]:
            """Send keyboard keys."""
            event = session.event_bus.dispatch(SendKeysEvent(keys=params.keys))
            await event
            return ToolResult(
                content=f"Sent keys: {params.keys}",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("send_keys"),
                description="Send keyboard keys (e.g., 'Enter', 'Escape', 'Tab', 'ArrowDown').",
                parameters=SendKeysParams,
                executor=send_keys_executor,
            )
        )

        # --- JavaScript Execution ---

        async def evaluate_js_executor(params: EvaluateJsParams) -> ToolResult[ToolUseCountMetadata]:
            """Execute JavaScript on the page."""
            page = await session.must_get_current_page()
            script = params.script.strip()
            # browser-use requires arrow function format - wrap if needed
            if not script.startswith("("):
                script = f"() => {script}"
            try:
                result = await page.evaluate(script)
                return ToolResult(
                    content=f"JavaScript result: {result}",
                    metadata=ToolUseCountMetadata(),
                )
            except Exception as e:
                return ToolResult(
                    content=f"JavaScript error: {e}",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )

        tools.append(
            Tool(
                name=self._tool_name("evaluate_js"),
                description="Execute custom JavaScript code on the page. Code is auto-wrapped in arrow function.",
                parameters=EvaluateJsParams,
                executor=evaluate_js_executor,
            )
        )

        # --- Tab Management ---

        async def switch_tab_executor(params: SwitchTabParams) -> ToolResult[ToolUseCountMetadata]:
            """Switch to a different tab."""
            tabs = await session.get_tabs()
            if params.index < 0 or params.index >= len(tabs):
                return ToolResult(
                    content=f"Tab index {params.index} out of range (0-{len(tabs) - 1})",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
            target_id = tabs[params.index].target_id
            event = session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
            await event
            return ToolResult(
                content=f"Switched to tab {params.index}",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("switch_tab"),
                description="Switch to a different browser tab by index (0-based).",
                parameters=SwitchTabParams,
                executor=switch_tab_executor,
            )
        )

        # --- Content Extraction ---

        async def snapshot_executor(_: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
            """Get accessibility snapshot of the current page."""
            state_text = await session.get_state_as_text()
            return ToolResult(
                content=f"<page_snapshot>\n{state_text}\n</page_snapshot>",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("snapshot"),
                description="Get accessibility snapshot of current page showing interactive elements with indices.",
                parameters=EmptyParams,
                executor=snapshot_executor,
            )
        )

        async def screenshot_executor(_: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
            """Take a screenshot of the current page."""
            screenshot_bytes = await session.take_screenshot()
            return ToolResult(
                content=[
                    "Screenshot captured:",
                    ImageContentBlock(data=screenshot_bytes),
                ],
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("screenshot"),
                description="Take a screenshot of the current page for visual inspection.",
                parameters=EmptyParams,
                executor=screenshot_executor,
            )
        )

        async def get_url_executor(_: EmptyParams) -> ToolResult[ToolUseCountMetadata]:
            """Get the current page URL."""
            url = await session.get_current_page_url()
            return ToolResult(
                content=f"Current URL: {url}",
                metadata=ToolUseCountMetadata(),
            )

        tools.append(
            Tool(
                name=self._tool_name("get_url"),
                description="Get the current page URL.",
                parameters=EmptyParams,
                executor=get_url_executor,
            )
        )

        return tools

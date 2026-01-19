"""Tests for BrowserUseToolProvider."""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from stirrup.tools.browser_use import (
    BrowserUseToolProvider,
    EmptyParams,
    InputTextMetadata,
    NavigateMetadata,
    NavigateParams,
    SearchMetadata,
    SearchParams,
)


class TestBrowserUseToolProvider:
    """Tests for BrowserUseToolProvider initialization and configuration."""

    def test_default_configuration(self) -> None:
        """Test default configuration values."""
        provider = BrowserUseToolProvider()
        assert provider._headless is True  # noqa: SLF001
        assert provider._tool_prefix == "browser"  # noqa: SLF001
        assert provider._use_cloud is False  # noqa: SLF001

    def test_tool_name_with_prefix(self) -> None:
        """Test tool name generation."""
        provider = BrowserUseToolProvider(tool_prefix="browser")
        assert provider._tool_name("click") == "browser_click"  # noqa: SLF001

        provider_no_prefix = BrowserUseToolProvider(tool_prefix="")
        assert provider_no_prefix._tool_name("click") == "click"  # noqa: SLF001


class TestToolBuilding:
    """Tests for tool building."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock browser session."""
        session = MagicMock()
        session.event_bus = MagicMock()
        session.event_bus.dispatch = MagicMock(return_value=AsyncMock())
        session.get_element_by_index = AsyncMock(return_value=None)
        session.get_state_as_text = AsyncMock(return_value="[1] button 'Click me'")
        session.take_screenshot = AsyncMock(return_value=b"\x89PNG\r\n")
        session.get_current_page_url = AsyncMock(return_value="https://example.com")
        session.get_tabs = AsyncMock(return_value=[])
        session.must_get_current_page = AsyncMock()
        return session

    async def test_build_tools_returns_expected_tools(self, mock_session: MagicMock) -> None:
        """Test that _build_tools returns all expected tools."""
        provider = BrowserUseToolProvider()
        provider._session = mock_session  # noqa: SLF001

        tools = provider._build_tools()  # noqa: SLF001
        tool_names = [t.name for t in tools]

        expected = [
            "browser_search",
            "browser_navigate",
            "browser_go_back",
            "browser_wait",
            "browser_click",
            "browser_input_text",
            "browser_scroll",
            "browser_find_text",
            "browser_send_keys",
            "browser_evaluate_js",
            "browser_switch_tab",
            "browser_snapshot",
            "browser_screenshot",
            "browser_get_url",
        ]
        for name in expected:
            assert name in tool_names, f"Missing tool: {name}"

    async def test_build_tools_without_session_raises(self) -> None:
        """Test _build_tools raises when session is None."""
        provider = BrowserUseToolProvider()
        with pytest.raises(RuntimeError, match="Browser session not initialized"):
            provider._build_tools()  # noqa: SLF001


class TestMetadataAggregation:
    """Test metadata aggregation for run tracking."""

    def test_navigate_metadata_addition(self) -> None:
        """Test NavigateMetadata aggregates URLs correctly."""
        meta1 = NavigateMetadata(urls=["https://example.com"])
        meta2 = NavigateMetadata(urls=["https://google.com"])

        combined = meta1 + meta2
        assert combined.num_uses == 2
        assert combined.urls == ["https://example.com", "https://google.com"]

    def test_search_metadata_addition(self) -> None:
        """Test SearchMetadata aggregates queries correctly."""
        meta1 = SearchMetadata(queries=["first query"])
        meta2 = SearchMetadata(queries=["second query"])

        combined = meta1 + meta2
        assert combined.num_uses == 2
        assert combined.queries == ["first query", "second query"]

    def test_input_text_metadata_addition(self) -> None:
        """Test InputTextMetadata aggregates texts correctly."""
        meta1 = InputTextMetadata(texts=["hello"])
        meta2 = InputTextMetadata(texts=["world"])

        combined = meta1 + meta2
        assert combined.num_uses == 2
        assert combined.texts == ["hello", "world"]


class TestEmptyArgumentsHandling:
    """Tests for empty tool arguments handling (agent.py fix)."""

    def test_empty_params_from_empty_json(self) -> None:
        """Test EmptyParams can be created from empty JSON object."""
        params = EmptyParams.model_validate_json("{}")
        assert params.model_dump() == {}


# =============================================================================
# Integration Tests (real headless Chrome)
# =============================================================================


@pytest.mark.browser
class TestBrowserIntegration:
    """Integration tests using real headless Chrome.

    Run with: pytest -m browser tests/test_browser_use.py
    Skip with: pytest -m "not browser" tests/test_browser_use.py
    """

    async def _call_tool(self, tool, params):  # noqa: ANN001, ANN202
        """Call tool executor, handling both sync and async executors."""
        result = tool.executor(params)
        if inspect.iscoroutine(result):
            return await result
        return result

    async def test_provider_lifecycle(self) -> None:
        """Test provider starts and stops browser session."""
        provider = BrowserUseToolProvider(headless=True)

        assert provider._session is None  # noqa: SLF001

        async with provider as tools:
            assert provider._session is not None  # noqa: SLF001
            assert len(tools) > 0

        assert provider._session is None  # noqa: SLF001

    async def test_navigate_and_snapshot(self) -> None:
        """Test navigating to a URL and taking snapshot."""
        provider = BrowserUseToolProvider(headless=True)

        async with provider as tools:
            tools_dict = {t.name: t for t in tools}

            # Navigate
            result = await self._call_tool(
                tools_dict["browser_navigate"],
                NavigateParams(url="https://example.com"),
            )
            assert result.success

            # Snapshot
            result = await self._call_tool(tools_dict["browser_snapshot"], EmptyParams())
            assert result.success
            assert "<page_snapshot>" in str(result.content)

    async def test_screenshot_returns_image(self) -> None:
        """Test screenshot returns image content."""
        provider = BrowserUseToolProvider(headless=True)

        async with provider as tools:
            tools_dict = {t.name: t for t in tools}

            await self._call_tool(
                tools_dict["browser_navigate"],
                NavigateParams(url="https://example.com"),
            )

            result = await self._call_tool(tools_dict["browser_screenshot"], EmptyParams())
            assert result.success
            assert isinstance(result.content, list)
            assert len(result.content) == 2  # text + ImageContentBlock

    async def test_search(self) -> None:
        """Test search navigates to search engine."""
        provider = BrowserUseToolProvider(headless=True)

        async with provider as tools:
            tools_dict = {t.name: t for t in tools}

            result = await self._call_tool(
                tools_dict["browser_search"],
                SearchParams(query="test", engine="duckduckgo"),
            )
            assert result.success

            url_result = await self._call_tool(tools_dict["browser_get_url"], EmptyParams())
            assert "duckduckgo.com" in str(url_result.content)

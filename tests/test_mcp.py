"""Behavioral tests for the MCP tools host."""

import inspect
from collections.abc import Awaitable
from contextlib import AsyncExitStack
from typing import Any, cast

import pytest
from mcp import Client
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ListToolsResult,
    ResourceLink,
    ServerCapabilities,
    TextContent,
    TextResourceContents,
    ToolsCapability,
)
from mcp.types import (
    Tool as MCPTool,
)
from pydantic import BaseModel

from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.mcp import (
    MCPConfig,
    MCPServerConfig,
    MCPToolProvider,
    SseServerConfig,
    StdioServerConfig,
    StreamableHttpServerConfig,
)


class FakeClient:
    """Small typed stand-in for the SDK client."""

    server_capabilities = ServerCapabilities(tools=ToolsCapability())

    def __init__(
        self,
        pages: dict[str | None, ListToolsResult],
        results: dict[str, CallToolResult] | None = None,
    ) -> None:
        self.pages = pages
        self.results = results or {}
        self.list_cursors: list[str | None] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        self.list_cursors.append(cursor)
        return self.pages[cursor]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        **_: object,
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        return self.results[name]


class StubMCPToolProvider(MCPToolProvider):
    """Use an in-memory fake while exercising the provider lifecycle."""

    def __init__(self, client: FakeClient) -> None:
        super().__init__(
            MCPConfig.model_validate({"mcpServers": {"test": {"command": "unused"}}}),
        )
        self.client = client

    async def _open_client(self, stack: AsyncExitStack, config: MCPServerConfig) -> Client:
        del stack, config
        return cast(Client, self.client)


def _tool(
    name: str,
    *,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> MCPTool:
    return MCPTool(
        name=name,
        description=f"{name} description",
        input_schema=input_schema or {"type": "object", "properties": {}},
        output_schema=output_schema,
    )


def _stirrup_tool(tools: list[Tool[Any, ToolUseCountMetadata]], name: str) -> Tool[Any, ToolUseCountMetadata]:
    return next(tool for tool in tools if tool.name == f"test__{name}")


async def _execute(
    tool: Tool[Any, ToolUseCountMetadata],
    arguments: dict[str, Any],
) -> ToolResult[ToolUseCountMetadata]:
    params = cast(type[BaseModel], tool.parameters).model_validate(arguments)
    result = tool.executor(params)
    assert inspect.isawaitable(result)
    return await cast(Awaitable[ToolResult[ToolUseCountMetadata]], result)


def test_config_expands_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "secret")

    config = MCPConfig.model_validate(
        {
            "mcpServers": {
                "remote": {
                    "transport": "streamable-http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
                }
            }
        }
    )

    server = config.mcp_servers["remote"]
    assert isinstance(server, StreamableHttpServerConfig)
    assert server.headers == {"Authorization": "Bearer secret"}


def test_config_rejects_missing_environment_variable() -> None:
    with pytest.raises(ValueError, match="MISSING_MCP_TOKEN"):
        MCPConfig.model_validate(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer ${MISSING_MCP_TOKEN}"},
                    }
                }
            }
        )


@pytest.mark.parametrize(
    ("raw", "expected_type"),
    [
        ({"command": "python"}, StdioServerConfig),
        ({"url": "https://example.com/sse"}, SseServerConfig),
        ({"url": "https://example.com/mcp"}, StreamableHttpServerConfig),
        ({"type": "http", "url": "https://example.com/mcp"}, StreamableHttpServerConfig),
    ],
)
def test_config_infers_conventional_transports(
    raw: dict[str, Any],
    expected_type: type[MCPServerConfig],
) -> None:
    config = MCPConfig.model_validate({"mcpServers": {"server": raw}})
    assert isinstance(config.mcp_servers["server"], expected_type)


def test_config_rejects_nonstandard_websocket_transport() -> None:
    with pytest.raises(ValueError, match="WebSocket is not an MCP transport"):
        MCPConfig.model_validate({"mcpServers": {"server": {"url": "wss://example.com/mcp"}}})


async def test_tool_discovery_follows_pagination_and_preserves_input_schema() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
        "additionalProperties": False,
    }
    client = FakeClient(
        {
            None: ListToolsResult(tools=[_tool("first", input_schema=schema)], next_cursor="page-2"),
            "page-2": ListToolsResult(tools=[_tool("second")]),
        }
    )

    async with StubMCPToolProvider(client) as tools:
        assert [tool.name for tool in tools] == ["test__first", "test__second"]
        assert _stirrup_tool(tools, "first").parameter_schema == schema

    assert client.list_cursors == [None, "page-2"]


async def test_nested_connection_attempt_does_not_clear_the_active_connection() -> None:
    provider = StubMCPToolProvider(FakeClient({None: ListToolsResult(tools=[])}))

    async with provider.connect():
        with pytest.raises(RuntimeError, match="already connected"):
            async with provider.connect():
                pass
        assert provider.servers == ["test"]

    assert provider.servers == []


async def test_invalid_arguments_are_rejected_before_the_server_call() -> None:
    client = FakeClient(
        {
            None: ListToolsResult(
                tools=[
                    _tool(
                        "count",
                        input_schema={
                            "type": "object",
                            "properties": {"count": {"type": "integer"}},
                            "required": ["count"],
                            "additionalProperties": False,
                        },
                    )
                ]
            )
        },
        {"count": CallToolResult(content=[TextContent(text="unused")])},
    )

    async with StubMCPToolProvider(client) as tools:
        result = await _execute(_stirrup_tool(tools, "count"), {"count": "one"})

    assert result.success is False
    assert "not of type 'integer'" in cast(str, result.content)
    assert client.calls == []


async def test_tool_errors_and_structured_content_keep_their_semantics() -> None:
    output_schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    client = FakeClient(
        {
            None: ListToolsResult(
                tools=[
                    _tool("success", output_schema=output_schema),
                    _tool("failure"),
                ]
            )
        },
        {
            "success": CallToolResult(
                content=[TextContent(text="done")],
                structured_content={"answer": 42},
            ),
            "failure": CallToolResult(
                content=[TextContent(text="business rule failed")],
                is_error=True,
            ),
        },
    )

    async with StubMCPToolProvider(client) as tools:
        success = await _execute(_stirrup_tool(tools, "success"), {})
        failure = await _execute(_stirrup_tool(tools, "failure"), {})

    assert success.success is True
    assert isinstance(success.content, list)
    assert success.content[0] == "done"
    assert '"answer": 42' in cast(str, success.content[1])
    assert failure.success is False
    assert failure.content == "business rule failed"


async def test_invalid_structured_content_is_not_passed_to_the_model() -> None:
    client = FakeClient(
        {
            None: ListToolsResult(
                tools=[
                    _tool(
                        "invalid",
                        output_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    )
                ]
            )
        },
        {
            "invalid": CallToolResult(
                content=[TextContent(text="untrusted")],
                structured_content={"answer": "wrong"},
            )
        },
    )

    async with StubMCPToolProvider(client) as tools:
        result = await _execute(_stirrup_tool(tools, "invalid"), {})

    assert result.success is False
    assert "invalid structuredContent" in cast(str, result.content)
    assert "untrusted" not in cast(str, result.content)


async def test_resource_content_is_preserved_as_model_readable_json() -> None:
    client = FakeClient(
        {None: ListToolsResult(tools=[_tool("resources")])},
        {
            "resources": CallToolResult(
                content=[
                    ResourceLink(
                        name="guide",
                        uri="https://example.com/guide",
                        description="The guide",
                        mime_type="text/html",
                    ),
                    EmbeddedResource(
                        resource=TextResourceContents(
                            uri="file:///README.md",
                            mime_type="text/markdown",
                            text="# Read me",
                        )
                    ),
                ]
            )
        },
    )

    async with StubMCPToolProvider(client) as tools:
        result = await _execute(_stirrup_tool(tools, "resources"), {})

    assert isinstance(result.content, list)
    assert "https://example.com/guide" in cast(str, result.content[0])
    assert "# Read me" in cast(str, result.content[1])

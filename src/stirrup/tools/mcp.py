"""MCP (Model Context Protocol) tool provider for connecting to MCP servers.

This module provides MCPToolProvider, a ToolProvider that manages connections to
multiple MCP servers and exposes each MCP tool as a separate Tool object.

Example usage:
    ```python
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    # With Agent (preferred)
    client = ChatCompletionsClient(model="gpt-5")
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*DEFAULT_TOOLS, MCPToolProvider.from_config("mcp.json")],
    )
    async with agent.session() as session:
        await session.run("Use MCP tools")

    # Standalone usage
    provider = MCPToolProvider.from_config(Path("mcp.json"))
    async with provider as tools:
        # tools is a list of Tool objects
        pass
    ```

Requires the optional `mcp` dependency:
    pip install stirrup[mcp]
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from json_schema_to_pydantic import create_model
from pydantic import BaseModel, Field, model_validator

from stirrup.core.models import Tool, ToolProvider, ToolResult, ToolUseCountMetadata

# MCP imports (optional dependency)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError as e:
    raise ImportError(
        "Requires installation of the mcp extra. Install with (for example): `uv pip install stirrup[mcp]` or `uv add stirrup[mcp]`",
    ) from e

# WebSocket client requires additional 'websockets' package
try:
    from mcp.client.websocket import websocket_client
except ImportError:
    websocket_client = None  # type: ignore[assignment, misc]


__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "MCPToolProvider",
    "SseServerConfig",
    "StdioServerConfig",
    "StreamableHttpServerConfig",
    "WebSocketServerConfig",
]


# === Models ===


class StdioServerConfig(BaseModel):
    """Configuration for stdio-based MCP servers (local process)."""

    command: str
    """Command to run the MCP server (e.g., "npx", "python")."""

    args: list[str] = Field(default_factory=list)
    """Arguments to pass to the command."""

    env: dict[str, str] | None = None
    """Environment variables to set for the server process."""

    cwd: str | None = None
    """Working directory for the server process."""

    encoding: str = "utf-8"
    """Text encoding for messages."""


class SseServerConfig(BaseModel):
    """Configuration for SSE-based MCP servers (HTTP GET with Server-Sent Events)."""

    url: str
    """The SSE endpoint URL (must end with /sse)."""

    headers: dict[str, str] | None = None
    """Optional HTTP headers."""

    timeout: float = 5.0
    """HTTP timeout for regular operations (seconds)."""

    sse_read_timeout: float = 300.0
    """Timeout for SSE read operations (seconds)."""


class StreamableHttpServerConfig(BaseModel):
    """Configuration for Streamable HTTP MCP servers (HTTP POST with optional SSE responses)."""

    url: str
    """The endpoint URL."""

    headers: dict[str, str] | None = None
    """Optional HTTP headers."""

    timeout: float = 30.0
    """HTTP timeout (seconds)."""

    sse_read_timeout: float = 300.0
    """SSE read timeout (seconds)."""

    terminate_on_close: bool = True
    """Close session when transport closes."""


class WebSocketServerConfig(BaseModel):
    """Configuration for WebSocket-based MCP servers."""

    url: str
    """The WebSocket URL (must start with ws:// or wss://)."""


# Type alias for the union of all server config types
MCPServerConfig = StdioServerConfig | SseServerConfig | StreamableHttpServerConfig | WebSocketServerConfig


def _infer_server_config(data: dict[str, Any]) -> MCPServerConfig:
    """Infer and instantiate the correct config class from raw data.

    Inference rules:
    - 'command' field present -> StdioServerConfig
    - 'url' starts with ws:// or wss:// -> WebSocketServerConfig
    - 'url' ends with /sse -> SseServerConfig
    - 'url' present (default) -> StreamableHttpServerConfig

    Args:
        data: Raw configuration dictionary.

    Returns:
        Appropriate server config instance.

    Raises:
        ValueError: If neither 'command' nor 'url' is provided.
    """
    if "command" in data:
        return StdioServerConfig(**data)
    if "url" in data:
        url = data["url"]
        if url.startswith(("ws://", "wss://")):
            return WebSocketServerConfig(**data)
        if url.endswith("/sse"):
            return SseServerConfig(**data)
        return StreamableHttpServerConfig(**data)
    raise ValueError("Config must have 'command' (stdio) or 'url' (SSE/HTTP/WebSocket)")


class MCPConfig(BaseModel):
    """Root configuration matching mcp.json format."""

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")
    """Map of server names to their configurations."""

    @model_validator(mode="before")
    @classmethod
    def _infer_transport_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert raw server configs to appropriate typed instances."""
        if "mcpServers" in data:
            data["mcpServers"] = {
                name: _infer_server_config(config) if isinstance(config, dict) else config
                for name, config in data["mcpServers"].items()
            }
        return data


# === Manager ===


class MCPToolProvider(ToolProvider):
    """MCP tool provider that manages connections to multiple MCP servers.

    MCPToolProvider connects to MCP servers and exposes each server's tools
    as individual Tool objects.

    Usage with Agent (preferred):
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[*DEFAULT_TOOLS, MCPToolProvider.from_config("mcp.json")],
        )

        async with agent.session(output_dir="./output") as session:
            await session.run("Use MCP tools")

    Standalone usage with connect() context manager:
        provider = MCPToolProvider.from_config(Path("mcp.json"))
        async with provider.connect() as provider:
            tools = provider.get_all_tools()
            # Use tools...
    """

    def __init__(
        self,
        config: MCPConfig,
        server_names: list[str] | None = None,
    ) -> None:
        """Initialize the MCP manager.

        Args:
            config: MCPConfig instance.
            server_names: Which servers to connect to. If None, connects to all servers in config.
        """
        self._config = config
        self._server_names = server_names
        self._servers: dict[str, ClientSession] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        self._exit_stack: AsyncExitStack | None = None

    @classmethod
    def from_config(cls, config_path: Path | str, server_names: list[str] | None = None) -> Self:
        """Create an MCPToolProvider from a config file.

        Args:
            config_path: Path to the MCP config file.
            server_names: Which servers to connect to. If None, connects to all servers in config.

        Returns:
            MCPToolProvider instance.
        """
        config = MCPConfig.model_validate_json(Path(config_path).read_text())

        return cls(config=config, server_names=server_names)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Self]:
        """Connect to MCP servers from config file.

        Yields:
            Self with active connections to specified servers.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            KeyError: If a specified server name doesn't exist in config.
        """
        config = self._config
        servers_to_connect = self._server_names or list(config.mcp_servers.keys())

        async with AsyncExitStack() as stack:
            for name in servers_to_connect:
                if name not in config.mcp_servers:
                    raise KeyError(f"Server '{name}' not found in config. Available: {list(config.mcp_servers.keys())}")

                server_config = config.mcp_servers[name]

                # Connect to server based on transport type
                match server_config:
                    case StdioServerConfig():
                        server_params = StdioServerParameters(
                            command=server_config.command,
                            args=server_config.args,
                            env=server_config.env,
                            cwd=server_config.cwd,
                            encoding=server_config.encoding,
                        )
                        read, write = await stack.enter_async_context(stdio_client(server_params))
                    case SseServerConfig():
                        read, write = await stack.enter_async_context(
                            sse_client(
                                url=server_config.url,
                                headers=server_config.headers,
                                timeout=server_config.timeout,
                                sse_read_timeout=server_config.sse_read_timeout,
                            )
                        )
                    case StreamableHttpServerConfig():
                        read, write, _ = await stack.enter_async_context(
                            streamablehttp_client(
                                url=server_config.url,
                                headers=server_config.headers,
                                timeout=server_config.timeout,
                                sse_read_timeout=server_config.sse_read_timeout,
                                terminate_on_close=server_config.terminate_on_close,
                            )
                        )
                    case WebSocketServerConfig():
                        if websocket_client is None:
                            raise ImportError(
                                f"WebSocket transport for server '{name}' requires the 'websockets' package. "
                                "Install with: pip install websockets"
                            )
                        read, write = await stack.enter_async_context(websocket_client(url=server_config.url))

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                # Cache session and available tools
                self._servers[name] = session
                response = await session.list_tools()
                self._tools[name] = [
                    {"name": t.name, "description": t.description, "schema": t.inputSchema} for t in response.tools
                ]

            try:
                yield self
            finally:
                self._servers.clear()
                self._tools.clear()

    @property
    def servers(self) -> list[str]:
        """List of connected server names."""
        return list(self._servers.keys())

    def get_tools(self, server: str) -> list[dict[str, Any]]:
        """Get available tools for a specific server.

        Args:
            server: Server name.

        Returns:
            List of tool info dicts with name, description, and schema.
        """
        return self._tools.get(server, [])

    @property
    def all_tools(self) -> dict[str, list[str]]:
        """Get all available tools grouped by server.

        Returns:
            Dict mapping server names to lists of tool names.
        """
        return {server: [t["name"] for t in tools] for server, tools in self._tools.items()}

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on a specific MCP server.

        Args:
            server: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool result as a string (text content extracted from response).

        Raises:
            ValueError: If server is not connected.
        """
        session = self._servers.get(server)
        if session is None:
            raise ValueError(f"Server '{server}' not connected. Available: {self.servers}")

        result = await session.call_tool(tool_name, arguments)

        # Extract text content from result
        text_parts = [str(content.text) for content in result.content if hasattr(content, "text")]
        return "\n".join(text_parts)

    def get_all_tools(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Get individual Tool objects for each tool from all connected MCP servers.

        Each MCP tool is exposed as a separate Tool with its own parameter schema,
        allowing the LLM to see and call each tool directly without routing through
        a unified proxy.

        Tool names are formatted as '{server}__{tool_name}' to ensure uniqueness
        across servers (e.g., 'supabase__query_table').

        Returns:
            List of Tool objects, one for each tool available across all connected servers.
        """
        tools: list[Tool[Any, ToolUseCountMetadata]] = []

        for server_name, server_tools in self._tools.items():
            for tool_info in server_tools:
                mcp_tool_name = tool_info["name"]
                # Create unique tool name with server prefix
                unique_name = f"{server_name}__{mcp_tool_name}"

                # Convert JSON schema to Pydantic model
                params_model = create_model(
                    tool_info.get("schema", {}),
                )

                # Create executor closure - capture server_name and mcp_tool_name
                # using default arguments to avoid late binding issues in the loop
                async def executor(
                    params: BaseModel,
                    _server: str = server_name,
                    _tool: str = mcp_tool_name,
                ) -> ToolResult[ToolUseCountMetadata]:
                    content = await self.call_tool(_server, _tool, params.model_dump())
                    xml_content = f"<mcp_result>\n{content}\n</mcp_result>"
                    return ToolResult(content=xml_content, metadata=ToolUseCountMetadata())

                tools.append(
                    Tool(
                        name=unique_name,
                        description=tool_info.get("description") or f"Tool '{mcp_tool_name}' from {server_name}",
                        parameters=params_model,
                        executor=executor,  # ty: ignore[invalid-argument-type]
                    )
                )

        return tools

    # Tool lifecycle protocol implementation
    async def __aenter__(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Enter async context: connect to MCP servers and return all tools.

        Returns:
            List of Tool objects, one for each tool available across all connected servers.
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        config = self._config
        servers_to_connect = self._server_names or list(config.mcp_servers.keys())

        for name in servers_to_connect:
            if name not in config.mcp_servers:
                raise KeyError(f"Server '{name}' not found in config. Available: {list(config.mcp_servers.keys())}")

            server_config = config.mcp_servers[name]

            # Connect to server based on transport type
            match server_config:
                case StdioServerConfig():
                    server_params = StdioServerParameters(
                        command=server_config.command,
                        args=server_config.args,
                        env=server_config.env,
                        cwd=server_config.cwd,
                        encoding=server_config.encoding,
                    )
                    read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
                case SseServerConfig():
                    read, write = await self._exit_stack.enter_async_context(
                        sse_client(
                            url=server_config.url,
                            headers=server_config.headers,
                            timeout=server_config.timeout,
                            sse_read_timeout=server_config.sse_read_timeout,
                        )
                    )
                case StreamableHttpServerConfig():
                    read, write, _ = await self._exit_stack.enter_async_context(
                        streamablehttp_client(
                            url=server_config.url,
                            headers=server_config.headers,
                            timeout=server_config.timeout,
                            sse_read_timeout=server_config.sse_read_timeout,
                            terminate_on_close=server_config.terminate_on_close,
                        )
                    )
                case WebSocketServerConfig():
                    if websocket_client is None:
                        raise ImportError(
                            f"WebSocket transport for server '{name}' requires the 'websockets' package. "
                            "Install with: pip install websockets"
                        )
                    read, write = await self._exit_stack.enter_async_context(websocket_client(url=server_config.url))

            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # Cache session and available tools
            self._servers[name] = session
            response = await session.list_tools()
            self._tools[name] = [
                {"name": t.name, "description": t.description, "schema": t.inputSchema} for t in response.tools
            ]

        return self.get_all_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: disconnect from MCP servers."""
        self._servers.clear()
        self._tools.clear()
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None

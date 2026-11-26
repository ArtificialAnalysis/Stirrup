"""Expose tools from MCP servers as native Stirrup tools.

The official MCP Python SDK owns protocol negotiation and transports. This
module owns Stirrup-specific configuration, tool namespacing, schema bridging,
and result conversion.

Requires the optional MCP dependencies:
    pip install stirrup[mcp]
"""

import json
import logging
import os
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stirrup.core.models import (
    AudioContentBlock,
    Content,
    ContentBlock,
    ImageContentBlock,
    Tool,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
)

try:
    import httpx2
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
    from mcp import Client, MCPError, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import (
        AudioContent as MCPAudioContent,
    )
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        EmbeddedResource,
        Implementation,
        ResourceLink,
        TextResourceContents,
    )
    from mcp.types import (
        ImageContent as MCPImageContent,
    )
    from mcp.types import (
        TextContent as MCPTextContent,
    )
    from mcp.types import (
        Tool as MCPTool,
    )
except ImportError as exc:
    raise ImportError(
        "MCP support requires the optional dependencies. Install them with `pip install stirrup[mcp]` "
        "or `uv add stirrup[mcp]`."
    ) from exc


__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "MCPToolProvider",
    "SseServerConfig",
    "StdioServerConfig",
    "StreamableHttpServerConfig",
]

logger = logging.getLogger(__name__)

_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CLIENT_INFO = Implementation(name="stirrup", title="Stirrup", version=version("stirrup"))


class _ServerConfig(BaseModel):
    """Options shared by every MCP transport."""

    read_timeout: float = Field(default=120.0, gt=0)
    """Maximum seconds to wait for an MCP response."""


class StdioServerConfig(_ServerConfig):
    """Configuration for a client-launched MCP subprocess."""

    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    encoding: str = "utf-8"


class SseServerConfig(_ServerConfig):
    """Configuration for the deprecated HTTP+SSE transport."""

    transport: Literal["sse"] = "sse"
    url: str
    headers: dict[str, str] | None = None
    timeout: float = Field(default=5.0, gt=0)
    sse_read_timeout: float = Field(default=300.0, gt=0)


class StreamableHttpServerConfig(_ServerConfig):
    """Configuration for the standard Streamable HTTP transport."""

    transport: Literal["streamable-http"] = "streamable-http"
    url: str
    headers: dict[str, str] | None = None
    timeout: float = Field(default=30.0, gt=0)
    sse_read_timeout: float = Field(default=300.0, gt=0)
    terminate_on_close: bool = True


MCPServerConfig = StdioServerConfig | SseServerConfig | StreamableHttpServerConfig


def _replace_environment_variables(value: object) -> object:
    """Resolve documented ``${NAME}`` placeholders without mutating config input."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Environment variable '{name}' is not set")
            return os.environ[name]

        return _ENVIRONMENT_VARIABLE.sub(replace, value)
    if isinstance(value, list):
        return [_replace_environment_variables(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_environment_variables(item) for key, item in value.items()}
    return value


def _infer_server_config(data: dict[str, Any]) -> MCPServerConfig:
    """Parse common MCP config shapes while preferring an explicit transport."""
    transport = data.get("transport") or data.get("type")
    if "command" in data:
        if transport not in (None, "stdio"):
            raise ValueError("A server with 'command' must use the stdio transport")
        return StdioServerConfig(**data)
    if "url" not in data:
        raise ValueError("MCP server config requires either 'command' or 'url'")

    url = data["url"]
    if url.startswith(("ws://", "wss://")) or transport in {"websocket", "ws"}:
        raise ValueError("WebSocket is not an MCP transport; use Streamable HTTP instead")
    if transport in {"streamable_http", "http"}:
        data = {**data, "transport": "streamable-http"}
        return StreamableHttpServerConfig(**data)
    if transport == "streamable-http":
        return StreamableHttpServerConfig(**data)
    if transport == "sse":
        return SseServerConfig(**data)
    if transport is not None:
        raise ValueError(f"Unsupported MCP transport: {transport!r}")

    path = urlsplit(url).path.rstrip("/")
    if path.endswith("/sse"):
        return SseServerConfig(**data)
    return StreamableHttpServerConfig(**data)


class MCPConfig(BaseModel):
    """Root configuration using the conventional ``mcpServers`` mapping."""

    mcp_servers: dict[str, MCPServerConfig] = Field(alias="mcpServers")

    @model_validator(mode="before")
    @classmethod
    def _parse_server_configs(cls, data: object) -> object:
        expanded = _replace_environment_variables(data)
        if not isinstance(expanded, dict) or "mcpServers" not in expanded:
            return expanded
        expanded_mapping = cast(dict[str, object], expanded)
        servers = expanded_mapping["mcpServers"]
        if not isinstance(servers, dict):
            return expanded
        server_mapping = cast(dict[str, object], servers)
        return {
            **expanded_mapping,
            "mcpServers": {
                name: _infer_server_config(cast(dict[str, Any], config)) if isinstance(config, dict) else config
                for name, config in server_mapping.items()
            },
        }


class _MCPToolParameters(BaseModel):
    """Preserve raw JSON arguments while the original MCP schema is exposed to the model."""

    model_config = ConfigDict(extra="allow")


class _SchemaValidator(Protocol):
    def iter_errors(self, instance: object) -> Iterator[JSONSchemaValidationError]: ...


@dataclass(frozen=True)
class _ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    input_validator: _SchemaValidator
    output_validator: _SchemaValidator | None


class MCPToolProvider(ToolProvider):
    """Connect to MCP servers and expose their tools through Stirrup.

    The SDK client negotiates the modern 2026 protocol and automatically falls
    back to legacy initialization-based servers. Tool catalogs are fully
    paginated and remain fixed for the lifetime of the Stirrup agent session.
    """

    def __init__(self, config: MCPConfig, server_names: list[str] | None = None) -> None:
        self._config = config
        self._server_names = list(server_names) if server_names is not None else None
        self._servers: dict[str, Client] = {}
        self._tools: dict[str, dict[str, _ToolDefinition]] = {}
        self._exit_stack: AsyncExitStack | None = None

    @classmethod
    def from_config(cls, config_path: Path | str, server_names: list[str] | None = None) -> Self:
        """Create a provider from a conventional MCP JSON config file."""
        return cls(MCPConfig.model_validate_json(Path(config_path).read_text()), server_names)

    def _selected_server_names(self) -> list[str]:
        names = list(self._config.mcp_servers) if self._server_names is None else self._server_names
        missing = [name for name in names if name not in self._config.mcp_servers]
        if missing:
            raise KeyError(f"Unknown MCP server(s) {missing}. Available: {list(self._config.mcp_servers)}")
        return names

    async def _open_client(
        self,
        stack: AsyncExitStack,
        config: MCPServerConfig,
    ) -> Client:
        match config:
            case StdioServerConfig():
                server = stdio_client(
                    StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env,
                        cwd=config.cwd,
                        encoding=config.encoding,
                    )
                )
            case SseServerConfig():
                server = sse_client(
                    url=config.url,
                    headers=config.headers,
                    timeout=config.timeout,
                    sse_read_timeout=config.sse_read_timeout,
                )
            case StreamableHttpServerConfig():
                http_client = await stack.enter_async_context(
                    httpx2.AsyncClient(
                        headers=config.headers,
                        timeout=httpx2.Timeout(config.timeout, read=config.sse_read_timeout),
                        follow_redirects=True,
                    )
                )
                server = streamable_http_client(
                    url=config.url,
                    http_client=http_client,
                    terminate_on_close=config.terminate_on_close,
                )
        return await stack.enter_async_context(
            Client(
                cast(Any, server),  # SDK documents transport context managers; its public annotation is narrower.
                client_info=_CLIENT_INFO,
                mode="auto",
                read_timeout_seconds=config.read_timeout,
            )
        )

    @staticmethod
    def _build_tool_definition(server_name: str, tool: MCPTool) -> _ToolDefinition:
        try:
            Draft202012Validator.check_schema(tool.input_schema)
            input_validator = Draft202012Validator(tool.input_schema)
            output_validator = None
            if tool.output_schema is not None:
                Draft202012Validator.check_schema(tool.output_schema)
                output_validator = Draft202012Validator(tool.output_schema)
        except SchemaError as exc:
            raise ValueError(
                f"MCP server '{server_name}' returned an invalid schema for tool '{tool.name}': {exc}"
            ) from exc

        return _ToolDefinition(
            name=tool.name,
            description=tool.description or tool.title or f"Tool '{tool.name}' from {server_name}",
            input_schema=tool.input_schema,
            input_validator=input_validator,
            output_validator=output_validator,
        )

    async def _list_all_tools(self, server_name: str, client: Client) -> dict[str, _ToolDefinition]:
        capabilities = client.server_capabilities
        if capabilities is not None and capabilities.tools is None:
            return {}

        tools: dict[str, _ToolDefinition] = {}
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            response = await client.list_tools(cursor=cursor)
            for tool in response.tools:
                if tool.name in tools:
                    raise ValueError(f"MCP server '{server_name}' returned duplicate tool '{tool.name}'")
                tools[tool.name] = self._build_tool_definition(server_name, tool)

            cursor = response.next_cursor
            if cursor is None:
                return tools
            if cursor in seen_cursors:
                raise ValueError(f"MCP server '{server_name}' repeated pagination cursor {cursor!r}")
            seen_cursors.add(cursor)

    async def _connect_all(self, stack: AsyncExitStack) -> None:
        if self._servers:
            raise RuntimeError("MCPToolProvider is already connected")
        for name in self._selected_server_names():
            client = await self._open_client(stack, self._config.mcp_servers[name])
            self._servers[name] = client
            self._tools[name] = await self._list_all_tools(name, client)

    def _clear(self) -> None:
        self._servers.clear()
        self._tools.clear()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Self]:
        """Connect for direct programmatic access to servers and tool metadata."""
        if self._servers or self._exit_stack is not None:
            raise RuntimeError("MCPToolProvider is already connected")
        async with AsyncExitStack() as stack:
            try:
                await self._connect_all(stack)
                yield self
            finally:
                self._clear()

    @property
    def servers(self) -> list[str]:
        """Connected server names."""
        return list(self._servers)

    @property
    def protocol_versions(self) -> dict[str, str]:
        """Negotiated protocol version for each connected server."""
        return {
            name: client.protocol_version
            for name, client in self._servers.items()
            if client.protocol_version is not None
        }

    def get_tools(self, server: str) -> list[dict[str, Any]]:
        """Return model-facing metadata for a connected server's tools."""
        return [
            {"name": tool.name, "description": tool.description, "schema": tool.input_schema}
            for tool in self._tools.get(server, {}).values()
        ]

    @property
    def all_tools(self) -> dict[str, list[str]]:
        """Tool names grouped by connected server."""
        return {server: list(tools) for server, tools in self._tools.items()}

    @staticmethod
    def _json_content(label: str, value: object) -> str:
        return f"<{label}>\n{json.dumps(value, ensure_ascii=False, indent=2)}\n</{label}>"

    def _convert_mcp_content(self, content_blocks: list[Any]) -> Content:
        """Convert every core MCP tool content type into Stirrup content."""
        content: list[ContentBlock] = []
        for block in content_blocks:
            if isinstance(block, MCPTextContent):
                content.append(block.text)
            elif isinstance(block, MCPImageContent):
                content.append(ImageContentBlock(data=block.data))
            elif isinstance(block, MCPAudioContent):
                content.append(AudioContentBlock(data=block.data))
            elif isinstance(block, ResourceLink):
                content.append(self._json_content("resource_link", block.model_dump(by_alias=True, exclude_none=True)))
            elif isinstance(block, EmbeddedResource):
                resource = block.resource
                if isinstance(resource, TextResourceContents):
                    content.append(
                        self._json_content(
                            "embedded_resource",
                            resource.model_dump(by_alias=True, exclude_none=True),
                        )
                    )
                elif isinstance(resource, BlobResourceContents):
                    metadata = resource.model_dump(by_alias=True, exclude={"blob"}, exclude_none=True)
                    content.append(self._json_content("embedded_resource", metadata))
                    try:
                        if (resource.mime_type or "").startswith("image/"):
                            content.append(ImageContentBlock(data=resource.blob))
                        elif (resource.mime_type or "").startswith("audio/"):
                            content.append(AudioContentBlock(data=resource.blob))
                        else:
                            content.append(self._json_content("embedded_resource_blob", resource.blob))
                    except ValueError:
                        content.append(self._json_content("embedded_resource_blob", resource.blob))
            else:
                raise TypeError(f"Unsupported MCP content block: {type(block).__name__}")

        if not content:
            return ""
        if len(content) == 1 and isinstance(content[0], str):
            return content[0]
        return content

    def _convert_tool_result(
        self,
        result: CallToolResult,
        output_validator: _SchemaValidator | None,
    ) -> ToolResult[ToolUseCountMetadata]:
        structured_content_present = "structured_content" in result.model_fields_set
        if output_validator is not None:
            if not structured_content_present:
                return ToolResult(
                    content="MCP server omitted structuredContent required by the tool's outputSchema",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
            error = next(output_validator.iter_errors(result.structured_content), None)
            if error is not None:
                return ToolResult(
                    content=f"MCP server returned invalid structuredContent: {error.message}",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )

        content = self._convert_mcp_content(result.content)
        if structured_content_present:
            structured = self._json_content("structured_content", result.structured_content)
            content = [*content, structured] if isinstance(content, list) else [content, structured]
            content = [block for block in content if block != ""]

        if result.result_type != "complete":
            return ToolResult(
                content="MCP tool requires client input that Stirrup could not provide",
                success=False,
                metadata=ToolUseCountMetadata(),
            )
        return ToolResult(
            content=content,
            success=not result.is_error,
            metadata=ToolUseCountMetadata(),
        )

    @staticmethod
    async def _log_progress(progress: float, total: float | None, message: str | None) -> None:
        logger.debug("MCP tool progress: progress=%s total=%s message=%s", progress, total, message)

    async def call_tool_result(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult[ToolUseCountMetadata]:
        """Call one MCP tool while preserving validation and failure semantics."""
        client = self._servers.get(server)
        if client is None:
            raise ValueError(f"Server '{server}' not connected. Available: {self.servers}")
        definition = self._tools.get(server, {}).get(tool_name)
        if definition is None:
            raise ValueError(f"Tool '{tool_name}' is not available from server '{server}'")

        input_error = next(definition.input_validator.iter_errors(arguments), None)
        if input_error is not None:
            return ToolResult(
                content=f"Tool arguments are not valid: {input_error.message}",
                success=False,
                metadata=ToolUseCountMetadata(),
            )

        try:
            result = await client.call_tool(
                tool_name,
                arguments,
                progress_callback=self._log_progress,
            )
        except MCPError as exc:
            return ToolResult(content=f"MCP protocol error: {exc}", success=False, metadata=ToolUseCountMetadata())
        except TimeoutError:
            return ToolResult(content="MCP tool call timed out", success=False, metadata=ToolUseCountMetadata())
        return self._convert_tool_result(result, definition.output_validator)

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Content:
        """Compatibility helper returning only tool content.

        Prefer :meth:`call_tool_result` when the caller needs success metadata.
        """
        return (await self.call_tool_result(server, tool_name, arguments)).content

    def get_all_tools(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Create namespaced Stirrup tools for every connected MCP tool."""
        tools: list[Tool[Any, ToolUseCountMetadata]] = []
        for server_name, server_tools in self._tools.items():
            for definition in server_tools.values():

                async def executor(
                    params: _MCPToolParameters,
                    _server: str = server_name,
                    _tool: str = definition.name,
                ) -> ToolResult[ToolUseCountMetadata]:
                    return await self.call_tool_result(_server, _tool, params.model_dump())

                tools.append(
                    Tool(
                        name=f"{server_name}__{definition.name}",
                        description=definition.description,
                        parameters=_MCPToolParameters,
                        parameter_schema=definition.input_schema,
                        executor=executor,
                    )
                )
        return tools

    async def __aenter__(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        """Connect and return all namespaced tools for ``Agent.session()``."""
        if self._exit_stack is not None or self._servers:
            raise RuntimeError("MCPToolProvider is already entered")
        stack = AsyncExitStack()
        await stack.__aenter__()
        self._exit_stack = stack
        try:
            await self._connect_all(stack)
        except BaseException:
            self._clear()
            self._exit_stack = None
            await stack.aclose()
            raise
        return self.get_all_tools()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Disconnect all MCP clients."""
        stack = self._exit_stack
        self._exit_stack = None
        try:
            if stack is not None:
                await stack.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._clear()

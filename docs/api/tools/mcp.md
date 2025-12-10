# MCP Tool Provider

The `stirrup.tools.mcp` module provides MCP (Model Context Protocol) integration.

!!! note
    Requires `pip install stirrup[mcp]` (or: `uv add stirrup[mcp]`)

## MCPToolProvider

::: stirrup.tools.mcp.MCPToolProvider
    options:
      show_source: true
      members:
        - from_config
        - __init__
        - __aenter__
        - __aexit__

## Configuration Models

### MCPConfig

::: stirrup.tools.mcp.MCPConfig

### Server Configurations

#### StdioServerConfig

For local command-based MCP servers:

::: stirrup.tools.mcp.StdioServerConfig

#### SseServerConfig

For SSE (Server-Sent Events) based MCP servers:

::: stirrup.tools.mcp.SseServerConfig

#### StreamableHttpServerConfig

For streamable HTTP MCP servers:

::: stirrup.tools.mcp.StreamableHttpServerConfig

#### WebSocketServerConfig

For WebSocket-based MCP servers:

::: stirrup.tools.mcp.WebSocketServerConfig

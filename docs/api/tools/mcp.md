# MCP Tool Provider

The `stirrup.tools.mcp` module provides Model Context Protocol tool integration.

!!! note
    Requires `pip install stirrup[mcp]` (or: `uv add stirrup[mcp]`)

## MCPToolProvider

::: stirrup.tools.mcp.MCPToolProvider
    options:
      show_source: true
      members:
        - from_config
        - __init__
        - connect
        - call_tool
        - call_tool_result
        - protocol_versions
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

For legacy HTTP+SSE MCP servers:

::: stirrup.tools.mcp.SseServerConfig

#### StreamableHttpServerConfig

For standard Streamable HTTP MCP servers:

::: stirrup.tools.mcp.StreamableHttpServerConfig

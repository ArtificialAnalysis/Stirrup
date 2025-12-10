# MCP Integration

Stirrup supports the Model Context Protocol (MCP) for connecting to external tool servers.

!!! note
    Requires `pip install stirrup[mcp]` (or: `uv add stirrup[mcp]`)

## Quick Start

```python
--8<-- "examples/mcp_example.py:main"
```

## Configuration

Create an `mcp.json` file (e.g., `.mcp/mcp.json`) with your server configurations. The path is passed to `MCPToolProvider.from_config()`—use an absolute path or a path relative to where you run your script.

### SSE Server (HTTP)

For remote MCP servers that use Server-Sent Events:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/sse"
    }
  }
}
```

### Stdio Server (Local Process)

For local MCP servers that run as command-line processes:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropics/mcp-server-filesystem", "/path/to/files"]
    }
  }
}
```

## Tool Naming

MCP tools are prefixed with the server name:

- Server `deepwiki` with tool `search` → `deepwiki__search`
- Server `filesystem` with tool `read_file` → `filesystem__read_file`

## Environment Variables

Use `${VAR_NAME}` syntax for secrets:

```json
{
  "mcpServers": {
    "api_server": {
      "url": "https://api.example.com/sse",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}
```

## Next Steps

- [Creating Tools](tools.md) - Build your own tools
- [Tool Providers](tool-providers.md) - Provider pattern

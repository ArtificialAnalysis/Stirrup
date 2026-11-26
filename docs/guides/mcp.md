# MCP Integration

Stirrup connects to MCP servers and exposes their tools as native Stirrup tools.

!!! note
    Requires `pip install stirrup[mcp]` (or: `uv add stirrup[mcp]`)

Stirrup uses the MCP Python SDK 2.x. It negotiates the current `2026-07-28`
protocol by default and automatically falls back to initialization-based legacy
servers.

## Quick Start

```python
--8<-- "examples/mcp_example.py:main"
```

## Configuration

Create an `mcp.json` file (e.g., `.mcp/mcp.json`) with your server configurations. The path is passed to `MCPToolProvider.from_config()`—use an absolute path or a path relative to where you run your script.

### Streamable HTTP

Streamable HTTP is the standard remote transport:

```json
{
  "mcpServers": {
    "deepwiki": {
      "transport": "streamable-http",
      "url": "https://mcp.deepwiki.com/mcp",
      "read_timeout": 120
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

### Legacy SSE

HTTP+SSE is deprecated in MCP but remains available for legacy servers:

```json
{
  "mcpServers": {
    "legacy": {
      "transport": "sse",
      "url": "https://api.example.com/sse"
    }
  }
}
```

The `transport` field is optional. A config with `command` uses stdio, a URL
whose path ends in `/sse` uses legacy SSE, and every other HTTP URL uses
Streamable HTTP.

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
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}
```

Loading fails with a clear error if a referenced variable is missing. Stirrup
supports static headers; it does not perform an interactive MCP OAuth flow.

## Tool Behavior

- Every page from `tools/list` is loaded.
- The server's original JSON Schema is passed to the model unchanged and used
  to validate arguments.
- MCP `isError`, `structuredContent`, `outputSchema`, resource links, embedded
  resources, text, images, and audio are preserved in the Stirrup result.
- MCP progress is logged at debug level.
- `read_timeout` bounds each request. Cancelling the surrounding task is
  propagated through the SDK to the MCP server.
- The tool catalog is fixed for one `Agent.session()`. Start a new session to
  reload changed tools.

## Scope and Security

This integration is intentionally an **MCP tools host**, not a complete
general-purpose MCP client. It does not expose MCP resources or prompts as
host UI, provide roots/sampling/elicitation callbacks, render MCP Apps, or
implement task and OAuth extensions.

An MCP tool can perform arbitrary external actions. Stirrup does not insert an
approval prompt before each model-selected call, so only configure servers you
trust and scope their credentials to the minimum permissions required.

## Next Steps

- [Creating Tools](tools.md) - Build your own tools
- [Tool Providers](tool-providers.md) - Provider pattern

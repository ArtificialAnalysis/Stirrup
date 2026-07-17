# Tool Providers

Tool Providers manage resources and return tools via async context manager. Use them when your tools need lifecycle management.

## When to Use ToolProviders

Use a `ToolProvider` when your tool requires:

- **Connections** - HTTP clients, database connections, websockets
- **Temporary resources** - Temp directories, sandboxes, processes
- **Cleanup logic** - Releasing resources, closing connections
- **Shared state** - State shared across multiple tool calls

## ToolProvider Protocol

All providers must implement the [`ToolProvider`][stirrup.core.models.ToolProvider] protocol:

| Member | Description |
|--------|-------------|
| `has_lifecycle = True` | Required marker to identify as a provider |
| `__aenter__()` | Setup resources and return `Tool` or `list[Tool]` |
| `__aexit__()` | Cleanup resources (close connections, delete temp files) |

## Basic Example

An HTTP client provider that shares a connection pool:

```python
import httpx
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata


class FetchParams(BaseModel):
    url: str = Field(description="URL to fetch")


class HTTPToolProvider:
    """Provides an HTTP fetch tool with shared client."""

    has_lifecycle = True  # Required marker

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Tool:
        # Setup: create HTTP client
        self._client = httpx.AsyncClient(timeout=self.timeout)

        async def fetch(params: FetchParams) -> ToolResult[ToolUseCountMetadata]:
            response = await self._client.get(params.url)
            return ToolResult(
                content=response.text[:5000],  # Truncate for context
                metadata=ToolUseCountMetadata(),
            )

        return Tool(
            name="http_fetch",
            description="Fetch content from a URL",
            parameters=FetchParams,
            executor=fetch,
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cleanup: close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None
```

## Returning Multiple Tools

A provider can return multiple related tools:

```python
class DatabaseToolProvider:
    """Provides query and insert tools for a database."""

    has_lifecycle = True

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._conn = None

    async def __aenter__(self) -> list[Tool]:
        self._conn = await connect(self.connection_string)

        return [
            self._create_query_tool(),
            self._create_insert_tool(),
        ]

    def _create_query_tool(self) -> Tool:
        async def query(params: QueryParams) -> ToolResult[ToolUseCountMetadata]:
            results = await self._conn.execute(params.sql)
            return ToolResult(content=str(results), metadata=ToolUseCountMetadata())

        return Tool(
            name="db_query",
            description="Execute a SQL query",
            parameters=QueryParams,
            executor=query,
        )

    def _create_insert_tool(self) -> Tool:
        # Similar implementation...
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn:
            await self._conn.close()
```

## Using with Agents

The agent's `session()` automatically manages ToolProvider lifecycle:

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient

# Provider is set up when session starts, cleaned up when it ends
client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="my_agent",
    tools=[
        HTTPToolProvider(timeout=60),
        DatabaseToolProvider("postgresql://..."),
    ],
)

async with agent.session() as session:
    # Tools are available here
    await session.run("Fetch data from the API and store in database")
# Providers are cleaned up automatically
```

A custom provider instance owns its lifecycle directly. It can be reused by
sequential sessions, including sessions created by different `Agent` objects.
Overlapping sessions cannot use the same configured custom provider instance;
Stirrup rejects the second session before entering any resources. Create a
separate provider instance for each concurrent owner. The same rule applies to
subclasses of built-in providers.

Exact built-in providers are reconstructed privately for each session, so one
`Agent` can run overlapping sessions without sharing their clients, sandboxes,
temporary directories, or exit stacks. When a custom `ViewImageToolProvider`
subclass explicitly references a configured built-in code backend, the dependent
provider/backend pair follows the custom-provider rule instead: it is reusable
sequentially, and overlap on either configured object is rejected before entry.
Such a pair cannot be used by a `share_parent_exec_env=True` sub-agent because
sharing substitutes the parent's backend; disable sharing or use the exact
built-in `ViewImageToolProvider`.

## Built-in ToolProviders

Stirrup includes several ToolProviders:

| Provider | Tools Provided | Description |
|----------|----------------|-------------|
| `LocalCodeExecToolProvider` | `code_exec` | Local temp directory execution |
| `DockerCodeExecToolProvider` | `code_exec` | Docker container execution |
| `E2BCodeExecToolProvider` | `code_exec` | E2B cloud sandbox |
| `WebToolProvider` | `web_fetch`, `web_search` | Web tools with shared client |
| `ViewImageToolProvider` | `view_image` | View images from exec env |
| `MCPToolProvider` | varies | MCP server tools |

## Mixing Tools and Providers

You can mix regular tools and providers:

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL, WebToolProvider

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="mixed_agent",
    tools=[
        CALCULATOR_TOOL,           # Regular tool (no lifecycle)
        HTTPToolProvider(),        # Provider (has lifecycle)
        WebToolProvider(),         # Built-in provider
    ],
)
```

## Error Handling

Session teardown attempts every provider and logger cleanup and shields that
work from surrounding AnyIO cancellation. Custom-resource ownership is released
only after cleanup finishes or raises. If setup or the session body already
failed, cleanup failures are attached to that primary exception rather than
replacing it. A cleanup failure is raised directly only when there was no
earlier failure.

Handle setup/cleanup errors gracefully:

```python
class RobustProvider:
    has_lifecycle = True

    async def __aenter__(self) -> Tool:
        try:
            self._resource = await acquire_resource()
        except ConnectionError as e:
            # Log and re-raise with context
            raise RuntimeError(f"Failed to connect: {e}") from e

        return self._create_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._resource:
            try:
                await self._resource.close()
            except Exception:
                # Log but don't raise during cleanup
                pass
```

## Next Steps

- [Code Execution](code-execution.md) - Execution backend providers
- [MCP Integration](mcp.md) - MCP server provider
- [Extending Backends](../extending/code_backends.md) - Custom execution backends

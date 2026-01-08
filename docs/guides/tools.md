# Creating Tools

This guide covers how to create custom tools for your agents.

## Tool Anatomy

A `Tool` consists of four parts:

| Component | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier for the tool |
| `description` | `str` | What the tool does (shown to the LLM) |
| `parameters` | `type[BaseModel]` | Pydantic model defining input schema (defaults to `EmptyParams`) |
| `executor` | `Callable` | Function that executes the tool |

## Basic Example

```python
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata


class GreetParams(BaseModel):
    """Parameters for the greet tool."""
    name: str = Field(description="Name of the person to greet")
    formal: bool = Field(default=False, description="Use formal greeting")


def greet(params: GreetParams) -> ToolResult[ToolUseCountMetadata]:
    if params.formal:
        greeting = f"Good day, {params.name}."
    else:
        greeting = f"Hey {params.name}!"

    return ToolResult(
        content=greeting,
        metadata=ToolUseCountMetadata(),
    )


GREET_TOOL = Tool(
    name="greet",
    description="Greet someone by name",
    parameters=GreetParams,
    executor=greet,
)
```

## Parameter Schemas

Use Pydantic models with `Field` descriptions to define tool parameters. The descriptions are included in the tool schema sent to the LLM.

### Required vs Optional Parameters

```python
class SearchParams(BaseModel):
    query: str = Field(description="Search query")  # Required
    max_results: int = Field(default=10, description="Max results")  # Optional
    include_images: bool = Field(default=False, description="Include images")
```

### Complex Types

```python
from typing import Literal

class AnalyzeParams(BaseModel):
    text: str = Field(description="Text to analyze")
    language: Literal["en", "es", "fr"] = Field(description="Language code")
    options: list[str] = Field(default_factory=list, description="Analysis options")
```

### Annotated Types

```python
from typing import Annotated

class CalculateParams(BaseModel):
    expression: Annotated[str, Field(description="Mathematical expression")]
    precision: Annotated[int, Field(default=2, ge=0, le=10, description="Decimal places")]
```

### Parameterless Tools

For tools that don't require any parameters, use `EmptyParams`:

```python
from stirrup import Tool, ToolResult, ToolUseCountMetadata, EmptyParams

TIME_TOOL = Tool[EmptyParams, ToolUseCountMetadata](
    name="get_time",
    description="Get the current time",
    executor=lambda _: ToolResult(
        content=datetime.now().isoformat(),
        metadata=ToolUseCountMetadata(),
    ),
)
```

Since `parameters` defaults to `EmptyParams`, you can also omit it:

```python
TIME_TOOL = Tool(
    name="get_time",
    description="Get the current time",
    executor=lambda _: ToolResult(content=datetime.now().isoformat()),
)
```

## Sync vs Async Executors

Tools can use either synchronous or asynchronous executors:

### Synchronous

```python
def my_tool(params: MyParams) -> ToolResult[ToolUseCountMetadata]:
    result = do_something(params)
    return ToolResult(content=result, metadata=ToolUseCountMetadata())
```

### Asynchronous

```python
async def my_async_tool(params: MyParams) -> ToolResult[ToolUseCountMetadata]:
    result = await do_something_async(params)
    return ToolResult(content=result, metadata=ToolUseCountMetadata())
```

By default, synchronous executors run in a separate thread (`run_sync_in_thread=True`).

## Tool Results

Tools return `ToolResult[M]` where `M` is the metadata type:

```python
from stirrup import ToolResult, ToolUseCountMetadata

# Simple text result
return ToolResult(
    content="Operation completed successfully",
    metadata=ToolUseCountMetadata(),
)

# Result with structured content
return ToolResult(
    content=f"Found {len(results)} items:\n" + "\n".join(results),
    metadata=ToolUseCountMetadata(),
)
```

### Returning Images

```python
from stirrup import ImageContentBlock

async def screenshot_tool(params: ScreenshotParams) -> ToolResult[ToolUseCountMetadata]:
    image_bytes = await take_screenshot()

    return ToolResult(
        content=[
            "Here's the screenshot:",
            ImageContentBlock(data=image_bytes),
        ],
        metadata=ToolUseCountMetadata(),
    )
```

## Tool Metadata

Metadata is aggregated across tool calls in a run. Use it to track usage statistics.

### Built-in Metadata

```python
from stirrup import ToolUseCountMetadata

# Tracks number of times tool was called
return ToolResult(content="done", metadata=ToolUseCountMetadata())
```

### Custom Metadata

Create custom metadata by implementing the `Addable` protocol:

```python
from stirrup import Addable
from pydantic import BaseModel


class APICallMetadata(BaseModel, Addable):
    """Track API calls and costs."""
    calls: int = 1
    tokens_used: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "APICallMetadata") -> "APICallMetadata":
        return APICallMetadata(
            calls=self.calls + other.calls,
            tokens_used=self.tokens_used + other.tokens_used,
            cost_usd=self.cost_usd + other.cost_usd,
        )


async def api_tool(params: APIParams) -> ToolResult[APICallMetadata]:
    response = await call_api(params)

    return ToolResult(
        content=response.text,
        metadata=APICallMetadata(
            tokens_used=response.tokens,
            cost_usd=response.cost,
        ),
    )
```

### Accessing Aggregated Metadata

```python
from stirrup import aggregate_metadata

finish_params, history, metadata = await session.run("task")

# metadata is dict[str, list[Any]] - tool_name -> list of metadata objects
aggregated = aggregate_metadata(metadata)

# Access aggregated values
print(f"API calls: {aggregated['api_tool'].calls}")
print(f"Total cost: ${aggregated['api_tool'].cost_usd:.2f}")
```

## Using Tools with Agents

### Adding to Default Tools

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="my_agent",
    tools=[*DEFAULT_TOOLS, GREET_TOOL, MY_OTHER_TOOL],
)
```

### Replacing Default Tools

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="custom_agent",
    tools=[GREET_TOOL, CALCULATOR_TOOL],  # Only these tools available
)
```

## Next Steps

- [Tool Providers](tool-providers.md) - Tools with lifecycle management
- [Code Execution](code-execution.md) - Execution backends
- [Extending Tools](../extending/tools.md) - Advanced tool patterns

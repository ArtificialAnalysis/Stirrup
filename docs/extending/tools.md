# Custom Tools (Advanced)

This guide covers advanced patterns for creating sophisticated tools.

## Tools with External APIs

```python
import httpx
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata


class WeatherParams(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(default="celsius", description="Temperature units")


class WeatherToolProvider:
    """Weather tool with shared HTTP client."""

    has_lifecycle = True

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Tool:
        self._client = httpx.AsyncClient(
            base_url="https://api.weather.com",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        async def get_weather(params: WeatherParams) -> ToolResult[ToolUseCountMetadata]:
            response = await self._client.get(
                "/current",
                params={"city": params.city, "units": params.units},
            )
            data = response.json()
            return ToolResult(
                content=f"Weather in {params.city}: {data['temp']}° {data['condition']}",
                metadata=ToolUseCountMetadata(),
            )

        return Tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters=WeatherParams,
            executor=get_weather,
        )

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
```

## Tools with State

```python
class ConversationMemoryProvider:
    """Tool that remembers context across calls."""

    has_lifecycle = True

    def __init__(self):
        self._memories: list[str] = []

    async def __aenter__(self) -> list[Tool]:
        return [self._create_remember_tool(), self._create_recall_tool()]

    def _create_remember_tool(self) -> Tool:
        async def remember(params: RememberParams) -> ToolResult[ToolUseCountMetadata]:
            self._memories.append(params.fact)
            return ToolResult(
                content=f"Remembered: {params.fact}",
                metadata=ToolUseCountMetadata(),
            )

        return Tool(
            name="remember",
            description="Store a fact for later recall",
            parameters=RememberParams,
            executor=remember,
        )

    def _create_recall_tool(self) -> Tool:
        async def recall(params: RecallParams) -> ToolResult[ToolUseCountMetadata]:
            relevant = [m for m in self._memories if params.query.lower() in m.lower()]
            if relevant:
                return ToolResult(content="\n".join(relevant), metadata=ToolUseCountMetadata())
            return ToolResult(content="No relevant memories found", metadata=ToolUseCountMetadata())

        return Tool(
            name="recall",
            description="Recall previously stored facts",
            parameters=RecallParams,
            executor=recall,
        )

    async def __aexit__(self, *args):
        self._memories.clear()
```

## Custom Metadata Types

```python
from pydantic import BaseModel
from stirrup import Addable


class APICallMetadata(BaseModel, Addable):
    """Track API call statistics."""

    calls: int = 1
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def __add__(self, other: "APICallMetadata") -> "APICallMetadata":
        return APICallMetadata(
            calls=self.calls + other.calls,
            tokens_used=self.tokens_used + other.tokens_used,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
        )


async def my_tool(params: MyParams) -> ToolResult[APICallMetadata]:
    start = time.time()
    response = await call_api(params)
    latency = (time.time() - start) * 1000

    return ToolResult(
        content=response.text,
        metadata=APICallMetadata(
            tokens_used=response.tokens,
            cost_usd=response.cost,
            latency_ms=latency,
        ),
    )
```

## Tools Returning Images

```python
from stirrup import ImageContentBlock, ToolResult


async def chart_tool(params: ChartParams) -> ToolResult[ToolUseCountMetadata]:
    # Generate chart with matplotlib
    import matplotlib.pyplot as plt
    import io

    plt.figure()
    plt.plot(params.x_data, params.y_data)
    plt.title(params.title)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_bytes = buf.read()
    plt.close()

    return ToolResult(
        content=[
            f"Generated chart: {params.title}",
            ImageContentBlock(data=image_bytes),
        ],
        metadata=ToolUseCountMetadata(),
    )
```

## Testing Tools

```python
import pytest
from stirrup import ToolResult


@pytest.mark.asyncio
async def test_weather_tool():
    provider = WeatherToolProvider(api_key="test")

    async with provider as tool:
        # Test the tool directly
        result = await tool.executor(WeatherParams(city="London"))

        assert isinstance(result, ToolResult)
        assert "London" in result.content
        assert result.metadata is not None


@pytest.mark.asyncio
async def test_tool_error_handling():
    async def failing_executor(params):
        raise ConnectionError("Network error")

    # Test that errors are handled gracefully
    # ...
```

## Next Steps

- [Tool Providers](../guides/tool-providers.md) - Provider pattern basics
- [Custom Loggers](loggers.md) - Logging customization
- [Custom Backends](code_backends.md) - Code execution backends

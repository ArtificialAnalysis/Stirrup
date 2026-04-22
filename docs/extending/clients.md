# Custom Clients

This guide covers implementing custom LLM clients for Stirrup.

## Built-in Clients

Stirrup provides three built-in clients:

| Client | API | Best For |
|--------|-----|----------|
| [`ChatCompletionsClient`](../api/clients/chat_completions.md) | OpenAI Chat Completions | OpenAI, OpenRouter, vLLM, Ollama, and other OpenAI-compatible APIs |
| [`OpenResponsesClient`](../api/clients/open_responses.md) | OpenAI Responses API | Providers implementing the newer Responses API format |
| [`LiteLLMClient`](../api/clients/litellm.md) | LiteLLM | Multi-provider support (Anthropic, Google, Azure, etc.) |

## LLMClient Protocol

All LLM clients must implement the [`LLMClient`][stirrup.core.models.LLMClient] protocol:

| Member | Type | Description |
|--------|------|-------------|
| `generate()` | `async method` | Generate next message with optional tool calls |
| `model_slug` | `property` | Model identifier string (e.g., `"openai/gpt-4o"`) |
| `max_tokens` | `property` | Maximum context window size |
| `assistant_metadata_type` | `class attribute` | Concrete metadata model used for assistant messages at runtime |

## Basic Implementation

```python
from stirrup import (
    AssistantMessage,
    ChatMessage,
    EmptyMetadata,
    LLMClient,
    Tool,
    TokenUsage,
)


class MyCustomClient(LLMClient[EmptyMetadata]):
    """Custom LLM client implementation."""

    assistant_metadata_type: type[EmptyMetadata] = EmptyMetadata

    def __init__(
        self,
        model: str,
        max_tokens: int = 64_000,
        api_key: str | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key

    @property
    def model_slug(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(
        self,
        messages: list[ChatMessage[EmptyMetadata]],
        tools: dict[str, Tool],
    ) -> AssistantMessage[EmptyMetadata]:
        # Convert messages to your API format
        api_messages = self._convert_messages(messages)

        # Convert tools to your API format
        api_tools = self._convert_tools(tools)

        # Call your LLM API
        response = await self._call_api(api_messages, api_tools)

        # Convert response to AssistantMessage
        return self._parse_response(response)
```

## Typed Assistant Metadata

If your client needs provider-specific response fields, define a Pydantic
metadata model and return `AssistantMessage[MyMetadata]`. Messages without
metadata use `EmptyMetadata()` by default. If you use typed metadata, set
`assistant_metadata_type` to the same model:

```python
from pydantic import BaseModel

from stirrup import AssistantMessage, ChatMessage, LLMClient, Tool, TokenUsage


class ResponseMetadata(BaseModel):
    request_id: str


class MyTypedClient(LLMClient[ResponseMetadata]):
    assistant_metadata_type: type[ResponseMetadata] = ResponseMetadata

    async def generate(
        self,
        messages: list[ChatMessage[ResponseMetadata]],
        tools: dict[str, Tool],
    ) -> AssistantMessage[ResponseMetadata]:
        return AssistantMessage(
            content="Hello",
            token_usage=TokenUsage(answer=1),
            metadata=ResponseMetadata(request_id="resp_123"),
        )
```

## Using with Agent

```python
from stirrup import Agent

client = MyCustomClient(
    model="my-model-id",
    max_tokens=100_000,
    api_key="...",
)

# Pass custom client directly to Agent
agent = Agent(
    client=client,
    name="custom_agent",
)
```

## OpenAI API Example

Stirrup message types use OpenAI-compatible field names (`role`, `content`, `tool_call_id`), so conversion is straightforward. The main difference is the `tool_calls` structure—OpenAI nests them under `function`.

```python
import openai
from stirrup import (
    AssistantMessage,
    ChatMessage,
    EmptyMetadata,
    LLMClient,
    Tool,
    ToolCall,
    ToolMessage,
    TokenUsage,
)


class OpenAIClient(LLMClient[EmptyMetadata]):
    """Direct OpenAI API client."""

    assistant_metadata_type: type[EmptyMetadata] = EmptyMetadata

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 128_000):
        self._model = model
        self._max_tokens = max_tokens
        self._client = openai.AsyncOpenAI()

    @property
    def model_slug(self) -> str:
        return f"openai/{self._model}"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _convert_message(self, msg: ChatMessage[EmptyMetadata]) -> dict:
        """Convert a message to OpenAI format."""
        # SystemMessage, UserMessage, ToolMessage have compatible structure
        if isinstance(msg, AssistantMessage):
            result = {"role": "assistant", "content": str(msg.content)}
            if msg.tool_calls:
                result["tool_calls"] = [
                    {"id": tc.tool_call_id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in msg.tool_calls
                ]
            return result
        elif isinstance(msg, ToolMessage):
            return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": str(msg.content)}
        else:
            # SystemMessage and UserMessage: just use role and content
            return {"role": msg.role, "content": str(msg.content)}

    async def generate(
        self,
        messages: list[ChatMessage[EmptyMetadata]],
        tools: dict[str, Tool],
    ) -> AssistantMessage[EmptyMetadata]:
        api_messages = [self._convert_message(m) for m in messages]
        api_tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters.model_json_schema()}}
            for t in tools.values()
        ] or None

        response = await self._client.chat.completions.create(model=self._model, messages=api_messages, tools=api_tools)
        message = response.choices[0].message

        return AssistantMessage[EmptyMetadata](
            content=message.content or "",
            tool_calls=[ToolCall(name=tc.function.name, arguments=tc.function.arguments, tool_call_id=tc.id) for tc in (message.tool_calls or [])],
            token_usage=TokenUsage(input=response.usage.prompt_tokens, answer=response.usage.completion_tokens),
        )
```

You can optionally populate `request_start_time` and `request_end_time` on `AssistantMessage`
to track generation speed. The derived `e2e_otps` property computes output tokens per second.

## Testing with Mock Client

```python
class MockClient:
    """Mock client for testing."""

    assistant_metadata_type: type[EmptyMetadata] = EmptyMetadata

    def __init__(self, responses: list[AssistantMessage[EmptyMetadata]]):
        self._responses = responses
        self._call_count = 0

    @property
    def model_slug(self) -> str:
        return "mock/test-model"

    @property
    def max_tokens(self) -> int:
        return 10_000

    async def generate(self, messages, tools) -> AssistantMessage[EmptyMetadata]:
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


# Use in tests
mock = MockClient([
    AssistantMessage[EmptyMetadata](content="Hello!", tool_calls=[], token_usage=TokenUsage()),
])

agent = Agent(client=mock, name="test")
```

## Next Steps

- [Custom Tools](tools.md) - Advanced tool patterns
- [Custom Loggers](loggers.md) - Logging customization

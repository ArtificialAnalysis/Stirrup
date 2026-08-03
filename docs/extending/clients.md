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
| `model_slug` | `property` | Model identifier string (e.g., `"openai/gpt-5.6-luna"`) |
| `context_window_tokens` | `property` | Model context capacity the agent summarizes history against |

`context_window_tokens` must return a positive int; `Agent` reads it once at
construction and summarizes conversation history as usage approaches that
capacity. There is no fallback: a client that does not expose it fails at
`Agent` construction. An output cap (`max_tokens`) is a client-internal request
parameter — the built-in clients take one, but the protocol does not require it
and the framework never reads it.

Custom clients that adapt provider stop reasons should raise
`OutputTokenLimitError` when the provider exhausts the response budget. Reserve
`ContextOverflowError` for request input that does not fit the model context.
The agent may summarize and retry the latter; it surfaces an output-limit failure
without summarization or retry with the unchanged limit.

## Basic Implementation

```python
from stirrup import (
    AssistantMessage,
    ChatMessage,
    Tool,
    TokenUsage,
)


class MyCustomClient:
    """Custom LLM client implementation."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 8_192,
        *,
        context_window_tokens: int,
        api_key: str | None = None,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._context_window_tokens = context_window_tokens
        self._api_key = api_key

    @property
    def model_slug(self) -> str:
        return self._model

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        # Convert messages to your API format
        api_messages = self._convert_messages(messages)

        # Convert tools to your API format
        api_tools = self._convert_tools(tools)

        # Call your LLM API
        response = await self._call_api(api_messages, api_tools)

        # Convert response to AssistantMessage
        return self._parse_response(response)
```

## Using with Agent

```python
from stirrup import Agent

client = MyCustomClient(
    model="my-model-id",
    max_tokens=8_192,
    context_window_tokens=100_000,
    api_key="...",
)

# Pass custom client directly to Agent
agent = Agent(
    client=client,
    name="custom_agent",
)
```

## OpenAI API Example

Stirrup's user, system, and tool message types use OpenAI-compatible field names (`role`, `content`, `tool_call_id`), so conversion is straightforward; assistant messages are block-based, and the `joined_text` / `tool_call_blocks` helpers project them into channel shape. The main difference is the `tool_calls` structure—OpenAI nests them under `function`.

```python
import openai
from stirrup import AssistantMessage, ChatMessage, TextBlock, Tool, ToolCall, ToolMessage, TokenUsage, joined_text, tool_call_blocks


class OpenAIClient:
    """Direct OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        max_tokens: int = 8_192,
        *,
        context_window_tokens: int,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._context_window_tokens = context_window_tokens
        self._client = openai.AsyncOpenAI()

    @property
    def model_slug(self) -> str:
        return f"openai/{self._model}"

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    def _convert_message(self, msg: ChatMessage) -> dict:
        """Convert a message to OpenAI format."""
        # SystemMessage, UserMessage, ToolMessage have compatible structure
        if isinstance(msg, AssistantMessage):
            result = {"role": "assistant", "content": joined_text(msg.blocks) or ""}
            if tool_calls := tool_call_blocks(msg.blocks):
                result["tool_calls"] = [
                    {"id": tc.tool_call_id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in tool_calls
                ]
            return result
        elif isinstance(msg, ToolMessage):
            return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": str(msg.content)}
        else:
            # SystemMessage and UserMessage: just use role and content
            return {"role": msg.role, "content": str(msg.content)}

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        api_messages = [self._convert_message(m) for m in messages]
        api_tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters.model_json_schema()}}
            for t in tools.values()
        ] or None

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,
            tools=api_tools,
            max_completion_tokens=self._max_tokens,
        )
        message = response.choices[0].message

        blocks = []
        if message.content:
            blocks.append(TextBlock(text=message.content))
        blocks.extend(
            ToolCall(name=tc.function.name, arguments=tc.function.arguments, tool_call_id=tc.id)
            for tc in (message.tool_calls or [])
        )

        return AssistantMessage(
            blocks=blocks,
            token_usage=TokenUsage(input=response.usage.prompt_tokens, answer=response.usage.completion_tokens),
        )
```

You can optionally populate `request_start_time` and `request_end_time` on `AssistantMessage`
to track generation speed. The derived `e2e_otps` property computes output tokens per second.

Construct `AssistantMessage` from blocks and replay from `msg.blocks` when converting
history back to your API format. This preserves ordered provider output, including
interleaved thinking, text, and tool calls. Channel-shaped v0.1 data is supported only
when reading persisted histories.

## Integration contract

Stirrup guarantees the following to client and integration authors:

- **Stable identity.** `AssistantMessage.id` is assigned once at construction and survives
  `model_dump`/`model_validate` round trips.
- **Same object in history.** The exact object returned by `generate` is appended to
  history and passed back on subsequent `generate` calls — the framework never copies or
  rebuilds history messages outside summarization and context-overflow unwinding.
- **Subclasses are preserved.** `generate` may return an `AssistantMessage` subclass; use
  this as a typed in-memory carrier for integration state (mark extra fields
  `exclude=True` to keep them out of serialized histories).
- **`metadata` is opaque.** The framework never reads, writes, drops, or transmits message
  metadata. Namespace your keys (e.g. `"myco/..."`); the un-namespaced space belongs to users.
- **Summaries carry lineage.** When history is summarized, the `SummaryMessage` records the
  ids of the assistant messages it replaced (`replaced_ids`, chained transitively across
  summarization rounds), so dumped histories reconstruct which turns a summary stands for.

## Testing with Mock Client

```python
class MockClient:
    """Mock client for testing."""

    def __init__(self, responses: list[AssistantMessage]):
        self._responses = responses
        self._call_count = 0

    @property
    def model_slug(self) -> str:
        return "mock/test-model"

    @property
    def context_window_tokens(self) -> int:
        return 100_000

    async def generate(self, messages, tools) -> AssistantMessage:
        response = self._responses[self._call_count]
        self._call_count += 1
        return response


# Use in tests
mock = MockClient([
    AssistantMessage(blocks=[TextBlock(text="Hello!")], token_usage=TokenUsage()),
])

agent = Agent(client=mock, name="test")
```

## Next Steps

- [Custom Tools](tools.md) - Advanced tool patterns
- [Custom Loggers](loggers.md) - Logging customization

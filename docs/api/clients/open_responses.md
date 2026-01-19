# OpenResponses Client

The `OpenResponsesClient` uses OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) (`POST /v1/responses`) instead of the Chat Completions API. This client is useful for providers that implement the newer Responses API format.

## Key Differences from ChatCompletionsClient

| Feature | ChatCompletionsClient | OpenResponsesClient |
|---------|----------------------|---------------------|
| API endpoint | `chat.completions.create()` | `responses.create()` |
| System messages | Included in `messages` array | Passed as `instructions` parameter |
| Message format | `{"role": "user", "content": [...]}` | `{"role": "user", "content": [{"type": "input_text", ...}]}` |
| Tool call IDs | `tool_call_id` | `call_id` |
| Reasoning config | `reasoning_effort` param | `reasoning: {"effort": ...}` object |

## Usage

For models that support extended thinking (like o1/o3), you can configure the reasoning effort:

```python
--8<-- "examples/open_responses_example.py:example"
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier (e.g., `"gpt-4o"`, `"o1"`) |
| `max_tokens` | `int` | `64_000` | Maximum output tokens |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `api_key` | `str \| None` | `None` | API key (falls back to `OPENAI_API_KEY` env var) |
| `reasoning_effort` | `str \| None` | `None` | Reasoning effort for o1/o3 models: `"low"`, `"medium"`, `"high"` |
| `timeout` | `float \| None` | `None` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Number of retries for transient errors |
| `instructions` | `str \| None` | `None` | Default system instructions |
| `kwargs` | `dict \| None` | `None` | Additional arguments passed to `responses.create()` |

## API Reference

::: stirrup.clients.open_responses_client

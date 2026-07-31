# Core Concepts

This page explains the fundamental concepts in Stirrup.

## Agent

The `Agent` class is the main entry point. It manages the agent loop: generating LLM responses, executing tools, and accumulating messages until a task is complete.

### Configuration Options

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient

client = ChatCompletionsClient(...)

agent = Agent(
    client=client,                                        # (required) LLM client for generating responses
    name="my_agent",                                      # (required) Agent name for logging
    max_turns=30,                                         # (default: 30) Max iterations before stopping
    system_prompt="You are an agent specializing in ...", # (default: None) Instructions prepended to runs
    tools=None,                                           # (default: default_tools()) Available tools
    finish_tool=None,                                     # (default: SIMPLE_FINISH_TOOL) Completion signal(s)
    context_summarization_cutoff=0.7,                     # (default: 0.7) Context % before summarization
    run_sync_in_thread=True,                              # (default: True) Run sync tools in thread
    text_only_tool_responses=True,                        # (default: True) Extract images from responses
    block_successive_assistant_messages=True,               # (default: True) Inject continue prompt between assistant messages
    recover_from_context_overflow=True,                   # (default: True) Retry overflows by unwinding recent progress
    logger=None,                                          # (default: None) Custom logger instance
)
```

??? info "Full Parameter Reference"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `client` | `LLMClient` | required | LLM client (use factory methods or create directly) |
    | `name` | `str` | required | Agent name for logging |
    | `max_turns` | `int` | `30` | Maximum turns before stopping |
    | `system_prompt` | `str \| None` | `None` | System prompt prepended to runs |
    | `tools` | `list[Tool \| ToolProvider] \| None` | `default_tools()` | Available tools |
    | `finish_tool` | `Tool \| list[Tool]` | `SIMPLE_FINISH_TOOL` | Tool(s) to signal completion |
    | `context_summarization_cutoff` | `float` | `0.7` | Context % before summarization |
    | `run_sync_in_thread` | `bool` | `True` | Run sync tools in separate thread |
    | `text_only_tool_responses` | `bool` | `True` | Extract images to user messages |
    | `block_successive_assistant_messages` | `bool` | `True` | Inject continue prompt to prevent back-to-back assistant messages |
    | `recover_from_context_overflow` | `bool` | `True` | Retry context overflows by unwinding recent completed turns |
    | `logger` | `AgentLoggerBase \| None` | `None` | Custom logger instance |

### Understanding Agent Output

The `run()` method returns a tuple of three values:

```python
finish_params, history, metadata = await session.run("Your task")
```

#### `finish_params`

Contains the agent's final response when it calls the finish tool:

- `reason`: Explanation of what was accomplished
- `paths`: List of files created/modified in the execution environment

```python
finish_params = {
    "reason": "Successfully found Australia's population for 2022-2024 and created a chart.",
    "paths": ["australia_population_chart.png"]
}
```

#### `history`

A list of message groups representing the conversation history. Each group contains:

- `SystemMessage`: System prompts
- `UserMessage`: User inputs and file contents
- `AssistantMessage`: LLM responses with tool calls
- `ToolMessage`: Results from tool executions

```python
history = [
    SystemMessage(role='system', content="You are an AI agent..."),
    UserMessage(role='user', content="What is the population of Australia..."),
    AssistantMessage(
        role='assistant',
        content="I'll search for Australia's population data...",
        tool_calls=[ToolCall(name='web_search', arguments='{"query": "..."}', tool_call_id='...')],
        token_usage=TokenUsage(input=1523, answer=156, reasoning=0)
    ),
    ToolMessage(role='tool', content="<results>...ABS data...</results>", name='web_search', ...),
    # ... additional turns ...
    AssistantMessage(
        role='assistant',
        content="All files are ready. Let me finish the task.",
        tool_calls=[ToolCall(name='finish', arguments='{"reason": "...", "paths": [...]}', ...)],
        token_usage=TokenUsage(input=25102, answer=285, reasoning=0)
    ),
    ToolMessage(role='tool', content="Successfully completed...", name='finish', ...),
]
```

#### `metadata`

A dictionary containing metadata from tool executions:

- `token_usage`: Total token counts (input, output, reasoning)
- Per-tool metadata (e.g., `code_exec`, `web_search`, `web_fetch`)

```python
metadata = {
    "web_search": [WebSearchMetadata(num_uses=1, pages_returned=5)],
    "fetch_web_page": [WebFetchMetadata(num_uses=1, pages_fetched=['https://...'])],
    "code_exec": [ToolUseCountMetadata(num_uses=3)],
    "finish": [ToolUseCountMetadata(num_uses=1)],
    "token_usage": [TokenUsage(input=239283, answer=4189, reasoning=0)]
}
```

Use `aggregate_metadata` to combine metadata across tool calls:

```python
from stirrup import aggregate_metadata

aggregated = aggregate_metadata(metadata)
print(f"Total tokens: {aggregated['token_usage'].total}")
```

Speed metrics are available directly on each `AssistantMessage` via `request_start_time`, `request_end_time`, and the derived `e2e_otps` property. Similarly, `ToolMessage` has `tool_start_time`, `tool_end_time`, and a `tool_duration` property.

## Session

The `session()` method returns the agent configured as an async context manager. Sessions handle:

- Tool lifecycle (setup and teardown of ToolProviders)
- File uploads to execution environment
- Skills loading and system prompt addition
- Output file saving
- Logging

```python
async with agent.session(
    output_dir="./output",           # Where to save output files
    input_files=["data.csv"],        # Files to upload
    skills_dir="skills",             # Directory containing skills
) as session:
    result = await session.run("Your task")
```

### Passing Input Files to the Agent

Provide files to the agent's execution environment via `input_files`:

```python
async with agent.session(
    input_files=["data.csv", "config.json"],
    output_dir="./output",
) as session:
    await session.run("Analyze the data in data.csv")
```

Supported formats:

| Format | Example | Description |
|--------|---------|-------------|
| Single file | `"data.csv"` | Upload one file |
| Multiple files | `["file1.txt", "file2.txt"]` | Upload a list of files |
| Directory | `"./data/"` | Upload directory contents recursively |
| Glob pattern | `"data/*.csv"`, `"**/*.py"` | Upload files matching pattern |

### Receiving Output Files from the Agent

When the agent creates files, save them to a local directory via `output_dir`:

```python
async with agent.session(output_dir="./results") as session:
    finish_params, _, _ = await session.run(
        "Create a Python script that prints hello world"
    )
    # Files listed in finish_params.paths are saved to ./results/
```

The agent signals which files to save by including their paths in `finish_params.paths` when calling the finish tool.

Saving is best-effort: a declared path may fail (it doesn't exist, isn't a regular file, escapes the execution root, or collides with another output). After the session, `session.last_output_files_result` reports what happened — `.saved` lists the files written and `.failed` maps each rejected source path to the reason. Files are copied, so the originals stay in the execution environment.

### Loading Skills

Skills are modular packages that extend agent capabilities with domain-specific instructions and scripts. Pass a skills directory to make them available:

```python
async with agent.session(
    skills_dir="skills",
    output_dir="./output",
) as session:
    await session.run("Analyze the data using the data_analysis skill")
```

The agent receives a list of available skills in its system prompt and can read the full instructions via `cat skills/<skill_name>/SKILL.md`.

→ See [Skills Guide](guides/skills.md) for full documentation.

## Client

Stirrup supports multiple ways to connect to LLM providers.

`max_tokens` and `context_window_tokens` are separate budgets: the first caps a single response,
the second is the model context capacity the agent summarizes history against. Built-in clients
require `context_window_tokens` at construction and reject `max_tokens` values that exceed it —
note the default `max_tokens` is `64_000`, so small context windows need an explicit `max_tokens`
too. The examples in this repository pair an explicit `max_tokens=8_192` response budget with an
explicit context window. Exceeding a response budget raises `OutputTokenLimitError` and aborts
the run rather than retrying, so increase `max_tokens` if your task needs long responses.

### ChatCompletionsClient

Use `ChatCompletionsClient` for OpenAI or OpenAI-compatible APIs:

```python
--8<-- "examples/deepseek_example.py:client"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier (e.g., `"gpt-5"`, `"deepseek-chat"`) |
| `max_tokens` | `int` | `64_000` | Maximum provider response/output tokens |
| `context_window_tokens` | `int` | required | Context capacity for summarization |
| `base_url` | `str \| None` | `None` | Custom API URL (for Deepseek, vLLM, etc.) |
| `api_key` | `str \| None` | `None` | API key (defaults to `OPENROUTER_API_KEY` env var) |
| `timeout` | `float \| None` | `None` | Request timeout in seconds |
| `max_retries` | `int` | `2` | Number of retries for transient errors |

### LiteLLMClient

Use `LiteLLMClient` for Anthropic, Google, and other providers via [LiteLLM](https://docs.litellm.ai/docs/providers):

```python
--8<-- "examples/litellm_example.py:client"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_slug` | `str` | required | Provider/model string (e.g., `"anthropic/claude-sonnet-4-5"`) |
| `max_tokens` | `int` | `64_000` | Maximum provider response/output tokens |
| `context_window_tokens` | `int` | required | Context capacity for summarization |
| `reasoning_effort` | `str \| None` | `None` | For reasoning models (o1/o3) |
| `kwargs` | `dict \| None` | `None` | Additional provider-specific arguments |

!!! note "LiteLLM Installation"
    Requires `pip install stirrup[litellm]` (or: `uv add stirrup[litellm]`)

### Creating Your Own Client

Implement the `LLMClient` protocol to create a custom client:

```python
from stirrup.core.models import LLMClient, AssistantMessage, ChatMessage, Tool

class MyCustomClient(LLMClient):
    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        # Make API call and return AssistantMessage
        ...

    @property
    def model_slug(self) -> str:
        return "my-model"

    @property
    def max_tokens(self) -> int:
        return 8_192

    @property
    def context_window_tokens(self) -> int:
        return 128_000
```

`Agent` reads `context_window_tokens` at construction and raises a `ValueError`
unless it is a positive int.

→ See [Custom Clients](extending/clients.md) for full documentation.

## Tools

### default_tools()

When you create an Agent without specifying tools, it uses `default_tools()`:

```python
from stirrup.tools import default_tools

# default_tools() returns a fresh list containing:
# - LocalCodeExecToolProvider() → provides "code_exec" tool
# - WebToolProvider() → provides "web_fetch" and "web_search" tools
```

| Tool Provider | Tools Provided | Description |
|--------------|----------------|-------------|
| `LocalCodeExecToolProvider` | `code_exec` | Execute shell commands in an isolated temp directory |
| `WebToolProvider` | `web_fetch`, `web_search` | Fetch web pages and search (search requires `BRAVE_API_KEY`) |

Each call returns fresh provider instances. Provider instances hold per-session state (a temp
directory, an HTTP client), so concurrent sessions must not share them.

**Breaking change:** the `DEFAULT_TOOLS` list was removed because every caller shared the same two
provider instances. Migrate `tools=DEFAULT_TOOLS` to `tools=default_tools()`, and
`tools=[*DEFAULT_TOOLS, extra_tool]` to `tools=[*default_tools(), extra_tool]`.

#### Extending vs Replacing

```python
--8<-- "examples/web_calculator.py:setup"
```

### Tool

A `Tool` has the following attributes:

- **name**: Unique identifier
- **description**: What the tool does (shown to the LLM)
- **parameters**: Pydantic model defining the input schema
- **executor**: Function that executes the tool

```python
--8<-- "examples/custom_tool_example.py:tool"
```

→ See [Creating Tools](guides/tools.md) for full documentation.

### Sub-agents

Convert any agent into a tool using `agent.to_tool()`. This enables hierarchical agent patterns where a supervisor delegates to specialized workers:

```python
--8<-- "examples/sub_agent_example.py:subagent"
```

The supervisor can then use sub-agents as tools:

```python
supervisor_agent = Agent(
    client=client,
    name="supervisor",
    tools=[research_subagent_tool, writer_subagent_tool],
)
```

→ See [Sub-Agents Guide](guides/sub-agents.md) for full documentation.

### Tool Provider

A `ToolProvider` is a class that manages resources and returns tools via async context manager. Use for tools requiring:

- Connections (HTTP clients, databases)
- Temporary directories
- Cleanup logic

```python
from stirrup import ToolProvider, Tool

class MyToolProvider(ToolProvider):
    async def __aenter__(self) -> Tool | list[Tool]:
        # Setup resources
        self.client = await create_client()
        return self._create_tool()

    async def __aexit__(self, *args):
        # Cleanup
        await self.client.close()
```

The agent's `session()` automatically calls `__aenter__` and `__aexit__` for all ToolProviders.

→ See [Tool Providers](guides/tool-providers.md) for full documentation.

### Finish Tools

A finish tool signals task completion. By default, agents use `SIMPLE_FINISH_TOOL`:

```python
from stirrup.tools.finish import FinishParams, SIMPLE_FINISH_TOOL

# Default FinishParams has:
# - reason: str - Explanation of what was accomplished
# - paths: list[str] - Files created/modified
```

Create custom finish tools for structured output:

```python
from pydantic import BaseModel, Field
from stirrup import Tool, ToolResult, ToolUseCountMetadata

class AnalysisResult(BaseModel):
    summary: str = Field(description="Analysis summary")
    confidence: float = Field(description="Confidence score 0-1")
    paths: list[str] = Field(default_factory=list)

custom_finish = Tool(
    name="finish",
    description="Complete the analysis task",
    parameters=AnalysisResult,
    executor=lambda p: ToolResult(
        content=p.summary,
        metadata=ToolUseCountMetadata()
    ),
)

agent = Agent(client=client, name="analyst", finish_tool=custom_finish)
```

You can also provide multiple finish tools. A successful call to any of them ends the agent loop:

```python
agent = Agent(
    client=client,
    name="analyst",
    finish_tool=[submit_files_tool, finish_without_files_tool],
)
```

### Tool Metadata

Tools return `ToolResult[M]` where `M` is the metadata type:

```python
from stirrup import ToolResult, ToolUseCountMetadata

def my_tool(params: MyParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(
        content="Result text",
        metadata=ToolUseCountMetadata(),  # Tracks number of uses
    )
```

Metadata aggregates across tool calls during a run. Built-in metadata types:

| Type | Description |
|------|-------------|
| `ToolUseCountMetadata` | Counts number of tool invocations |
| `TokenUsage` | Tracks input/output/reasoning tokens |
| `SubAgentMetadata` | Captures sub-agent message history |

Access aggregated metadata:

```python
from stirrup import aggregate_metadata

_, _, metadata = await session.run("task")
aggregated = aggregate_metadata(metadata)
print(f"Total tokens: {aggregated['token_usage'].total}")
```

### Context Overflow Recovery

By default, Stirrup retries context overflow errors by shortening the conversation and trying again.
An output-limit stop is different: `OutputTokenLimitError` surfaces without
summarization or retry with the same `max_tokens` limit.

The built-in OpenAI-family clients detect overflow from OpenAI's `context_length_exceeded`
error code. OpenAI-compatible endpoints that report a different code (such as OpenRouter,
whose error codes depend on the upstream provider) surface a plain `BadRequestError` instead,
so this recovery does not trigger for them — proactive summarization at
`context_summarization_cutoff` remains the primary protection.

When overflow happens, the agent removes the latest completed assistant turn. It will not remove the original prompt, existing summaries, or the only completed turn after either boundary; this ensures the surviving trajectory still has forward progress.

This also applies when eager summarization overflows. Any removed turn is also removed from final metadata and does not count against `max_turns`.

To fail immediately instead:

```python
agent = Agent(
    client=client,
    name="my_agent",
    recover_from_context_overflow=False,
)
```

Recovery only covers `ContextOverflowError`. Two other client failures deliberately abort the run:

- Summarization is itself a model call, so `summarize_messages` can raise `OutputTokenLimitError`
  out of `session.run()` even though you never call it directly. Retrying it with the same
  `max_tokens` cannot succeed, so it is not recovered.
- `OpenResponsesClient` raises `IncompleteResponseError` for incomplete responses that are not
  output-limit stops, such as `incomplete_details.reason == "content_filter"`. Before token
  budgets were split these unwound a turn and retried; only context overflow does that now.

## Logging

The agent uses `AgentLogger` by default, which provides rich console output with:

- Progress spinners showing steps, tool calls, and token usage
- Visual hierarchy for sub-agents
- Syntax-highlighted tool results

```python
from stirrup.utils.logging import AgentLogger
import logging

# Custom log level
logger = AgentLogger(level=logging.DEBUG)
agent = Agent(client=client, name="assistant", logger=logger)
```

### Custom Loggers

Implement `AgentLoggerBase` for custom logging:

```python
from stirrup.utils.logging import AgentLoggerBase

class MyLogger(AgentLoggerBase):
    def __enter__(self):
        # Setup logging
        return self

    def __exit__(self, *args):
        # Cleanup
        pass

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0):
        # Called after each step
        print(f"Step {step}: {tool_calls} tool calls")

    # Implement other required methods...
```

→ See [Custom Loggers](extending/loggers.md) for full documentation.

## Next Steps

- [Examples](examples.md) - Working examples for common patterns
- [Creating Tools](guides/tools.md) - Build your own tools
- [Code Execution](guides/code-execution.md) - Execution backends
- [Sub-Agents](guides/sub-agents.md) - Hierarchical agent patterns

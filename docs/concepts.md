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
    tools=None,                                           # (default: DEFAULT_TOOLS) Available tools
    finish_tool=None,                                     # (default: SIMPLE_FINISH_TOOL) Completion signal
    context_summarization_cutoff=0.7,                     # (default: 0.7) Context % before summarization
    run_sync_in_thread=True,                              # (default: True) Run sync tools in thread
    text_only_tool_responses=True,                        # (default: True) Extract images from responses
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
    | `tools` | `list[Tool \| ToolProvider] \| None` | `DEFAULT_TOOLS` | Available tools |
    | `finish_tool` | `Tool` | `SIMPLE_FINISH_TOOL` | Tool to signal completion |
    | `context_summarization_cutoff` | `float` | `0.7` | Context % before summarization |
    | `run_sync_in_thread` | `bool` | `True` | Run sync tools in separate thread |
    | `text_only_tool_responses` | `bool` | `True` | Extract images to user messages |
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
        token_usage=TokenUsage(input=1523, output=156, reasoning=0)
    ),
    ToolMessage(role='tool', content="<results>...ABS data...</results>", name='web_search', ...),
    # ... additional turns ...
    AssistantMessage(
        role='assistant',
        content="All files are ready. Let me finish the task.",
        tool_calls=[ToolCall(name='finish', arguments='{"reason": "...", "paths": [...]}', ...)],
        token_usage=TokenUsage(input=25102, output=285, reasoning=0)
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
    "token_usage": [TokenUsage(input=239283, output=4189, reasoning=0)]
}
```

Use `aggregate_metadata` to combine metadata across tool calls:

```python
from stirrup import aggregate_metadata

aggregated = aggregate_metadata(metadata)
print(f"Total tokens: {aggregated['token_usage'].total}")
```

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

### ChatCompletionsClient

Use `ChatCompletionsClient` for OpenAI or OpenAI-compatible APIs:

```python
--8<-- "examples/deepseek_example.py:client"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier (e.g., `"gpt-5"`, `"deepseek-chat"`) |
| `max_tokens` | `int` | `64_000` | Context window size |
| `base_url` | `str \| None` | `None` | Custom API URL (for Deepseek, vLLM, etc.) |
| `api_key` | `str \| None` | `None` | API key (defaults to `OPENROUTER_API_KEY` env var) |
| `supports_audio_input` | `bool` | `False` | Whether model supports audio |
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
| `max_tokens` | `int` | required | Context window size |
| `supports_audio_input` | `bool` | `False` | Whether model supports audio |
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
        return 128_000
```

→ See [Custom Clients](extending/clients.md) for full documentation.

## Tools

### DEFAULT_TOOLS

When you create an Agent without specifying tools, it uses `DEFAULT_TOOLS`:

```python
from stirrup.tools import DEFAULT_TOOLS

# DEFAULT_TOOLS contains:
# - LocalCodeExecToolProvider() → provides "code_exec" tool
# - WebToolProvider() → provides "web_fetch" and "web_search" tools
```

| Tool Provider | Tools Provided | Description |
|--------------|----------------|-------------|
| `LocalCodeExecToolProvider` | `code_exec` | Execute shell commands in an isolated temp directory |
| `WebToolProvider` | `web_fetch`, `web_search` | Fetch web pages and search (search requires `BRAVE_API_KEY`) |

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

The finish tool signals task completion. By default, agents use `SIMPLE_FINISH_TOOL`:

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
    executor=lambda p: ToolResult(content=p.summary, metadata=ToolUseCountMetadata()),
)

agent = Agent(client=client, name="analyst", finish_tool=custom_finish)
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

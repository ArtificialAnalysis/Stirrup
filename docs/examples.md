# Examples

Working examples demonstrating common Stirrup patterns. Full source code is in the `examples/` directory.

## Web Search Agent

A simple agent using default tools to search the web.

```python
--8<-- "examples/getting_started.py"
```

!!! note
    Web search requires `BRAVE_API_KEY` environment variable.

## Web Calculator

An agent with calculator added to default tools.

```python
--8<-- "examples/web_calculator.py"
```

## Code Execution

Execute code in isolated environments with multiple backend options.

```python
--8<-- "examples/code_executor/code_executor.py"
```

!!! note "Backend Options"
    - **Local**: `LocalCodeExecToolProvider()` - runs in temp directory
    - **Docker**: `DockerCodeExecToolProvider.from_image("python:3.12-slim")` - requires Docker
    - **E2B**: `E2BCodeExecToolProvider()` - requires `E2B_API_KEY`

## MCP Integration

Connect to MCP servers for additional tools.

```python
--8<-- "examples/mcp_example.py"
```

Example `.mcp/mcp.json`:

```json
{
  "mcpServers": {
    "deepwiki": {
      "url": "https://mcp.deepwiki.com/sse"
    }
  }
}
```

!!! note
    Requires `pip install stirrup[mcp]` (or: `uv add stirrup[mcp]`).

## Image Processing

An agent that can download and view images.

```python
--8<-- "examples/view_image_example.py"
```

## Sub-Agent Pattern

Use one agent as a tool for another. This example shows a supervisor agent coordinating specialized sub-agents for research and report writing.

```python
--8<-- "examples/sub_agent_example.py"
```

!!! warning "File Transfer Requirement"
    If a sub-agent has a code execution environment and produces files, the parent agent **must** also have a `CodeExecToolProvider` to receive those files.

## OpenAI-Compatible APIs

Connect to any OpenAI-compatible API by specifying a custom `base_url`. Ensure you have set the correct environment variables required for the specific provider.

```python
--8<-- "examples/deepseek_example.py"
```

!!! note
    Requires `DEEPSEEK_API_KEY` environment variable (or the appropriate key for your provider).

## LiteLLM Multi-Provider Support

Use LiteLLM to connect to non-OpenAI providers like Anthropic Claude, Google Gemini, and many others.

```python
--8<-- "examples/litellm_example.py"
```

!!! note
    Requires `pip install stirrup[litellm]` (or: `uv add stirrup[litellm]`) and the appropriate API key for your chosen provider (e.g., `ANTHROPIC_API_KEY` for Claude).

## Custom Finish Tool

Define structured output with a custom finish tool:

```python
import asyncio

from pydantic import BaseModel, Field

from stirrup import Agent, Tool, ToolResult, ToolUseCountMetadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient


class AnalysisResult(BaseModel):
    """Structured analysis output."""
    summary: str = Field(description="Brief summary of findings")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="URLs of sources used")


async def main():
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    # Create custom finish tool
    finish_tool = Tool(
        name="finish",
        description="Complete the analysis with structured results",
        parameters=AnalysisResult,
        executor=lambda p: ToolResult(
            content="Analysis complete",
            metadata=ToolUseCountMetadata()
        ),
    )

    agent = Agent(
        client=client,
        name="analyst",
        finish_tool=finish_tool,
    )

    async with agent.session() as session:
        finish_params, _, _ = await session.run(
            "Analyze the current state of renewable energy adoption globally."
        )

        # finish_params is now typed as AnalysisResult
        print(f"Summary: {finish_params.summary}")
        print(f"Confidence: {finish_params.confidence}")
        print(f"Sources: {finish_params.sources}")


asyncio.run(main())
```
<div align="center">
  <a href="https://stirrup.artificialanalysis.ai">
    <picture>
        <img alt="Stirrup" src="https://github.com/ArtificialAnalysis/Stirrup/blob/048653717d8662b0b81d152a037995af1c926afc/assets/stirrup-banner.png?raw=true" width="700">
    </picture>
  </a>
  <br></br>
  <h1>The lightweight foundation for building agents</h1>
<br>
</div>


<p align="center">
  <a href="https://pypi.python.org/pypi/stirrup"><img src="https://img.shields.io/pypi/v/stirrup" alt="PyPI version" /></a>&nbsp;<!--
  --><a href="https://github.com/ArtificialAnalysis/Stirrup/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ArtificialAnalysis/Stirrup" alt="License" /></a>&nbsp;<!--
  --><a href="https://stirrup.artificialanalysis.ai"><img src="https://img.shields.io/badge/MkDocs-4F46E5?logo=materialformkdocs&logoColor=fff" alt="MkDocs" /></a>
</p>


Stirrup is a lightweight framework, or starting point template, for building agents. It differs from other agent frameworks by:

- **Working with the model, not against it:** Stirrup gets out of the way and lets the model choose its own approach to completing tasks (similar to Claude Code). Many frameworks impose rigid workflows that can degrade results.
- **Best practices and tools built-in:** We analyzed the leading agents (Claude Code, Codex, and others) to understand and incorporate best practices relating to topics like context management and foundational tools (e.g., code execution).
- **Fully customizable:** Use Stirrup as a package or as a starting template to build your own fully customized agents.

## Features

- **Essential tools built-in:**
  - Online search / web browsing
  - Code execution (local, Docker container, E2B sandbox)
  - MCP client
  - Document input and output
- **Skills system:** Extend agent capabilities with modular, domain-specific instruction packages
- **Flexible tool execution:** A generic `Tool` class allows easy tool definition and extension
- **Context management:** Automatically summarizes conversation history when approaching context limits
- **Flexible provider support:** Pre-built support for OpenAI-compatible APIs and LiteLLM, or bring your own client
- **Multimodal support:** Process images, video, and audio with automatic format conversion

## Installation

```bash
# Core framework
pip install stirrup      # or: uv add stirrup

# With all optional components
pip install 'stirrup[all]'  # or: uv add 'stirrup[all]'

# Individual extras
pip install 'stirrup[litellm]'  # or: uv add 'stirrup[litellm]'
pip install 'stirrup[docker]'   # or: uv add 'stirrup[docker]'
pip install 'stirrup[e2b]'      # or: uv add 'stirrup[e2b]'
pip install 'stirrup[mcp]'      # or: uv add 'stirrup[mcp]'
```

## Quick Start

```python
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient


async def main() -> None:
    """Run an agent that searches the web and creates a chart."""

    # Create client using ChatCompletionsClient
    # Automatically uses OPENROUTER_API_KEY environment variable
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    # As no tools are provided, the agent will use the default tools, which consist of:
    # - Web tools (web search and web fetching, note web search requires BRAVE_API_KEY)
    # - Local code execution tool (to execute shell commands)
    agent = Agent(client=client, name="agent", max_turns=15)

    # Run with session context - handles tool lifecycle, logging and file outputs
    async with agent.session(output_dir="./output/getting_started_example") as session:
        finish_params, history, metadata = await session.run(
            """
            What is the population of Australia over the last 3 years? Search the web to find out and create a
            simple chart using matplotlib showing the current population per year."""
        )

        print("Finish params: ", finish_params)
        print("History: ", history)
        print("Metadata: ", metadata)


if __name__ == "__main__":
    asyncio.run(main())
```

> **Note:** This example uses OpenRouter. Set `OPENROUTER_API_KEY` in your environment before running. Web search requires a `BRAVE_API_KEY`. The agent will still work without it, but web search will be unavailable.

## Full Customization

For using Stirrup as a foundation for your own fully customized agent, you can clone and import Stirrup locally:

```bash
# Clone the repository
git clone https://github.com/ArtificialAnalysis/Stirrup.git
cd stirrup

# Install in editable mode
pip install -e .      # or: uv venv && uv pip install -e .

# Or with all optional dependencies
pip install -e '.[all]'  # or: uv venv && uv pip install -e '.[all]'
```

See the [Full Customization guide](https://stirrup.artificialanalysis.ai/extending/full-customization/) for more details.

## How It Works

- **`Agent`** - Configures and runs the agent loop until a finish tool is called or max turns reached
- **`session()`** - Context manager that sets up tools, manages files, and handles cleanup
- **`Tool`** - Define tools with Pydantic parameters
- **`ToolProvider`** - Manage tools that require lifecycle (connections, temp directories, etc.)
- **`DEFAULT_TOOLS`** - Standard tools included by default: code execution and web tools

## Using Other LLM Providers

For non-OpenAI providers, change the base URL of the `ChatCompletionsClient`, use the `LiteLLMClient` (requires installation of optional `stirrup[litellm]` dependencies), or create your own client.

### OpenAI-Compatible APIs

```python
# Create client using Deepseek's OpenAI-compatible endpoint
client = ChatCompletionsClient(
    base_url="https://api.deepseek.com",
    model="deepseek-chat",  # or "deepseek-reasoner" for R1
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

agent = Agent(client=client, name="deepseek_agent")
```

### LiteLLM (Anthropic, Google, etc.)

```python
# Ensure LiteLLM is added with: pip install 'stirrup[litellm]'  # or: uv add 'stirrup[litellm]'
# Create LiteLLM client for Anthropic Claude
# See https://docs.litellm.ai/docs/providers for all supported providers
client = LiteLLMClient(
    model_slug="anthropic/claude-sonnet-4-5",
    max_tokens=200_000,
)

# Pass client to Agent - model info comes from client.model_slug
agent = Agent(
    client=client,
    name="claude_agent",
)
```

See [LiteLLM Example](https://stirrup.artificialanalysis.ai/examples/#litellm-multi-provider-support) or [Deepseek Example](https://stirrup.artificialanalysis.ai/examples/#openai-compatible-apis-deepseek-vllm-ollama) for complete examples.

## Default Tools

When you create an `Agent` without specifying tools, it uses `DEFAULT_TOOLS`:

| Tool Provider               | Tools Provided            | Description                                                  |
| --------------------------- | ------------------------- | ------------------------------------------------------------ |
| `LocalCodeExecToolProvider` | `code_exec`               | Execute shell commands in an isolated temp directory         |
| `WebToolProvider`           | `web_fetch`, `web_search` | Fetch web pages and search (search requires `BRAVE_API_KEY`) |

## Extending with Pre-Built Tools

```python
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL, DEFAULT_TOOLS

# Create client for OpenRouter
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# Create agent with default tools + calculator tool
agent = Agent(
    client=client,
    name="web_calculator_agent",
    tools=[*DEFAULT_TOOLS, CALCULATOR_TOOL],
)
```

## Defining Custom Tools

```python
from pydantic import BaseModel, Field

from stirrup import Agent, Tool, ToolResult, ToolUseCountMetadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS


class GreetParams(BaseModel):
    """Parameters for the greet tool."""

    name: str = Field(description="Name of the person to greet")
    formal: bool = Field(default=False, description="Use formal greeting")


def greet(params: GreetParams) -> ToolResult[ToolUseCountMetadata]:
    greeting = f"Good day, {params.name}." if params.formal else f"Hey {params.name}!"

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

# Create client for OpenRouter
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# Add custom tool to default tools
agent = Agent(
    client=client,
    name="greeting_agent",
    tools=[*DEFAULT_TOOLS, GREET_TOOL],
)
```

## Next Steps

- [Getting Started](https://stirrup.artificialanalysis.ai/getting-started/) - Installation and first agent tutorial
- [Core Concepts](https://stirrup.artificialanalysis.ai/concepts/) - Understand Agent, Tools, and Sessions
- [Examples](https://stirrup.artificialanalysis.ai/examples/) - Working examples for common patterns
- [Creating Tools](https://stirrup.artificialanalysis.ai/guides/tools/) - Build your own tools

## Documentation

Full documentation: [artificialanalysis.github.io/Stirrup](https://stirrup.artificialanalysis.ai)

Build and serve locally:

```bash
uv run mkdocs serve
```

## Development

```bash
# Format and lint code
uv run ruff format
uv run ruff check

# Type check
uv run ty check

# Run tests
uv run pytest tests
```

## License

Licensed under the [MIT LICENSE](LICENSE).

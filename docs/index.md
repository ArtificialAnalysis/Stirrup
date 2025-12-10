# Stirrup

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
--8<-- "examples/getting_started.py:simple"
```

!!! note "Environment Variables"
    This example uses OpenRouter. Set `OPENROUTER_API_KEY` in your environment before running.

    Web search requires a `BRAVE_API_KEY`. The agent will still work without it, but web search will be unavailable.

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
--8<-- "examples/deepseek_example.py:client"
```

### LiteLLM (Anthropic, Google, etc.)

```python
--8<-- "examples/litellm_example.py:client"
```

See [LiteLLM Example](examples.md#litellm-multi-provider-support) or [Deepseek Example](examples.md#openai-compatible-apis) for complete examples.

## Default Tools

When you create an `Agent` without specifying tools, it uses `DEFAULT_TOOLS`:

| Tool Provider | Tools Provided | Description |
|--------------|----------------|-------------|
| `LocalCodeExecToolProvider` | `code_exec` | Execute shell commands in an isolated temp directory |
| `WebToolProvider` | `web_fetch`, `web_search` | Fetch web pages and search (search requires `BRAVE_API_KEY`) |

## Extending with Pre-Built Tools

```python
--8<-- "examples/web_calculator.py:setup"
```

## Defining Custom Tools

```python
--8<-- "examples/custom_tool_example.py:tool"
```

## Full Customization

For deep customization of the framework internals, you can clone and import Stirrup locally:

```bash
# Clone the repository
git clone https://github.com/ArtificialAnalysis/Stirrup.git
cd stirrup

# Install in editable mode
pip install -e .      # or: uv venv && uv pip install -e .

# Or with all optional dependencies
pip install -e '.[all]'  # or: uv venv && uv pip install -e '.[all]'
```

See the [Full Customization guide](extending/full-customization.md) for more details.

## Next Steps

- [Getting Started](getting-started.md) - Installation and first agent tutorial
- [Core Concepts](concepts.md) - Understand Agent, Tools, and Sessions
- [Examples](examples.md) - Working examples for common patterns
- [Creating Tools](guides/tools.md) - Build your own tools

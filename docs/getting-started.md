# Getting Started

This guide walks you through installing Stirrup and creating your first agent.

## Prerequisites

- Python 3.12 or higher
- An API key from your LLM provider (OpenRouter, Anthropic, OpenAI, etc.)

## Installation

Install the core framework:

```bash
pip install stirrup      # or: uv add stirrup
```

Or with all optional components:

```bash
pip install 'stirrup[all]'  # or: uv add 'stirrup[all]'
```

### Optional Extras

| Extra | Description |
|-------|-------------|
| `litellm` | Multi-provider support via LiteLLM (Anthropic, Google, etc.) |
| `docker` | Docker-based code execution |
| `e2b` | E2B cloud sandboxes |
| `mcp` | MCP server support |
| `all` | All optional components |

## Your First Agent

Create a simple agent that can search the web and execute code:

```python
--8<-- "examples/getting_started.py:simple"
```

!!! note "Environment Variables"
    This example uses OpenRouter. Set `OPENROUTER_API_KEY` in your environment before running.

    Web search requires a `BRAVE_API_KEY`. The agent will still work without it, but web search will be unavailable.

## Tools

By default, agents include code execution and web tools:

| Tool | Description |
|------|-------------|
| `code_exec` | Execute shell commands in an isolated temp directory |
| `web_fetch` | Fetch and parse web pages |
| `web_search` | Search the web (requires `BRAVE_API_KEY`) |

Extend with additional tools:

```python
--8<-- "examples/web_calculator.py:setup"
```

→ See [Tools](concepts.md#tools) for full documentation on DEFAULT_TOOLS, custom tools, sub-agents, and tool providers.

## Choosing a Client

`stirrup` ships with support for OpenAI-Compatible APIs and `LiteLLM`, with the open to build your own client.

### OpenAI-Compatible APIs

Use `ChatCompletionsClient` to use OpenAI models or OpenAI-compatible APIs:

```python
--8<-- "examples/deepseek_example.py:client"
```

→ See [Client](concepts.md#client) for parameter tables and creating custom clients.

## Understanding the Output

The `run()` method returns three values:

```python
finish_params, history, metadata = await session.run("Your task")
```

- **`finish_params`**: Agent's final response (`reason`, `paths`)
- **`history`**: Conversation message history
- **`metadata`**: Aggregated tool metadata and token usage

→ See [Understanding Agent Output](concepts.md#understanding-agent-output) for details.

## Uploading Input Files

Provide files to the agent's execution environment:

```python
async with agent.session(
    input_files=["data.csv", "config.json"],
    output_dir="./output",
) as session:
    await session.run("Analyze the data")
```

Supports single files, lists, directories, and glob patterns (`"data/*.csv"`).

→ See [Passing Input Files](concepts.md#passing-input-files-to-the-agent) for details.

## Saving Output Files

Save files created by the agent by providing an output directory through the session `output_dir` argument:

```python
async with agent.session(output_dir="./results") as session:
    finish_params, _, _ = await session.run("Create a chart")
    # Files in finish_params.paths are saved to ./results/
```

→ See [Receiving Output Files](concepts.md#receiving-output-files-from-the-agent) for details.

## Next Steps

- [Core Concepts](concepts.md) - Deep dive into Agent, Session, Client, Tools, and Logging
- [Examples](examples.md) - Working examples for common patterns
- [Creating Tools](guides/tools.md) - Build your own tools
- [Code Execution](guides/code-execution.md) - Different execution backends

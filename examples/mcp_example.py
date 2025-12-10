"""Example: MCP-powered agent with simplified session API.

This example demonstrates how to create an agent with access to MCP server tools
using the new session-based API.

Prerequisites:
- Create mcp.json in .mcp/ directory with server configuration

Example .mcp/mcp.json:
    {
      "mcpServers": {
        "deepwiki": {
          "url": "https://mcp.deepwiki.com/sse"
        }
      }
    }
"""

import asyncio
from pathlib import Path

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.mcp import MCPToolProvider


# --8<-- [start:main]
async def main() -> None:
    """Run an agent with MCP tools."""
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
        max_tokens=50_000,
    )

    # Create agent with default tools + MCP tools
    agent = Agent(
        client=client,
        name="mcp_example_agent",
        tools=[*DEFAULT_TOOLS, MCPToolProvider.from_config(".mcp/mcp.json")],
        max_turns=20,
    )

    # Run with session context - handles tool lifecycle, logging, and file saving
    async with agent.session(output_dir=Path("./output/mcp_example")) as session:
        task = """You have access to MCP server tools and a code execution environment.
            Using the same implementation as TheAlgorithms/Python (you can use DeepWiki MCP
            to research), write a Python file quicksort.py that implements quicksort and
            another that tests (and times) it.
            When done, call the finish tool including your findings."""

        _finish_params, _history, _metadata = await session.run(task)


# --8<-- [end:main]

if __name__ == "__main__":
    asyncio.run(main())

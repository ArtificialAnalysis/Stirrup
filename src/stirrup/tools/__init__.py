"""Tool implementations and providers.

This module provides tools and tool providers for the Agent.

## Tool vs ToolProvider

- **Tool**: A simple, stateless callable with a name, description, parameters, and executor.
  Use for tools that don't require setup/teardown.

- **ToolProvider**: A class that manages resources and returns Tool(s) via async context manager.
  Use for tools requiring lifecycle management (connections, temp directories, etc.).

## DEFAULT_TOOLS

DEFAULT_TOOLS provides a standard set of tool providers:
- LocalCodeExecToolProvider: Code execution in isolated temp directory
- WebToolProvider: Web fetch and search (search requires BRAVE_API_KEY)

Example usage:
    from stirrup import Agent, DEFAULT_TOOLS
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools.mcp import MCPToolProvider

    # Create a client for your LLM provider
    client = ChatCompletionsClient(model="gpt-5")

    # Use default tools
    agent = Agent(client=client, name="assistant")

    # Extend default tools
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*DEFAULT_TOOLS, MCPToolProvider.from_config("mcp.json")],
    )

    # Custom tools only (no defaults)
    agent = Agent(
        client=client,
        name="assistant",
        tools=[CALCULATOR_TOOL, my_custom_tool],
    )

## Optional Dependencies

Optional tool providers require explicit imports from their submodules:
- DockerCodeExecToolProvider: `from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider`
- E2BCodeExecToolProvider: `from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider`
- MCPToolProvider: `from stirrup.tools.mcp import MCPToolProvider`
"""

from typing import Any

from stirrup.core.models import Tool, ToolProvider
from stirrup.tools.calculator import CALCULATOR_TOOL
from stirrup.tools.code_backends import CodeExecToolProvider, LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL, FinishParams
from stirrup.tools.user_input import USER_INPUT_TOOL
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.tools.web import WebToolProvider

# DEFAULT_TOOLS provides a standard set of tool providers for the Agent.
# ToolProviders are automatically set up and torn down by Agent.session().
DEFAULT_TOOLS: list[Tool[Any, Any] | ToolProvider] = [
    LocalCodeExecToolProvider(),  # ToolProvider, returns code_exec tool
    WebToolProvider(),  # ToolProvider, returns web_fetch + web_search (if API key)
]

__all__ = [
    "CALCULATOR_TOOL",
    "DEFAULT_TOOLS",
    "SIMPLE_FINISH_TOOL",
    "USER_INPUT_TOOL",
    "CodeExecToolProvider",
    "FinishParams",
    "LocalCodeExecToolProvider",
    "ViewImageToolProvider",
    "WebToolProvider",
]

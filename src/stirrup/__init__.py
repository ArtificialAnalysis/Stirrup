"""Artificial Analysis' reference agent harness - originally built for running evaluations, simple to use and extend.

Example usage:
    from stirrup import Agent
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools import default_tools
    from stirrup.tools.mcp import MCPToolProvider

    # Create a client for your LLM provider
    client = ChatCompletionsClient(model="gpt-5")

    # Simple usage with default tools
    agent = Agent(
        client=client,
        name="assistant",
        system_prompt="You are a helpful assistant.",
    )

    async with agent.session(output_dir="./output") as session:
        finish_params, history, metadata = await session.run("Your task here")
        print(finish_params.reason)

    # Extend default tools with MCP
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*default_tools(), MCPToolProvider.from_config("mcp.json")],
    )
"""

from stirrup import tools
from stirrup.core.agent import Agent, SessionAgent
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.middleware import (
    DiskSpillMiddleware,
    ExecEnvSink,
    LocalDirSink,
    Sink,
    ToolMiddleware,
    ToolTruncatorMiddleware,
)
from stirrup.core.models import (
    Addable,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    EmptyParams,
    ImageContentBlock,
    LLMClient,
    SubAgentMetadata,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
    UserMessage,
    VideoContentBlock,
    aggregate_metadata,
)

__all__ = [
    "Addable",
    "Agent",
    "AssistantMessage",
    "AudioContentBlock",
    "ChatMessage",
    "ContextOverflowError",
    "DiskSpillMiddleware",
    "EmptyParams",
    "ExecEnvSink",
    "ImageContentBlock",
    "LLMClient",
    "LocalDirSink",
    "SessionAgent",
    "Sink",
    "SubAgentMetadata",
    "SummaryMessage",
    "SystemMessage",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "ToolMiddleware",
    "ToolProvider",
    "ToolResult",
    "ToolTruncatorMiddleware",
    "ToolUseCountMetadata",
    "UserMessage",
    "VideoContentBlock",
    "aggregate_metadata",
    "tools",
]

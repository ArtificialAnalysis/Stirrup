"""Artificial Analysis' reference agent harness - originally built for running evaluations, simple to use and extend.

Example usage:
    from stirrup import Agent
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools import default_tools
    from stirrup.tools.mcp import MCPToolProvider

    # Create a client for your LLM provider
    client = ChatCompletionsClient(model="gpt-5.6-luna", max_tokens=8_192, context_window_tokens=1_000_000)

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
from stirrup.core.exceptions import ContextOverflowError, IncompleteResponseError, OutputTokenLimitError
from stirrup.core.models import (
    Addable,
    AnyReasoningBlock,
    AssistantBlock,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    EmptyParams,
    EncryptedReasoningBlock,
    ImageContentBlock,
    LLMClient,
    OpaqueBlock,
    ReasoningBlock,
    ReasoningRefBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    SubAgentMetadata,
    SystemMessage,
    TextBlock,
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
    final_text,
    joined_text,
    reasoning_blocks,
    tool_call_blocks,
)
from stirrup.tools.middleware import (
    DiskSpillMiddleware,
    ExecEnvSink,
    Sink,
    ToolMiddleware,
    ToolTruncatorMiddleware,
)

__all__ = [
    "Addable",
    "Agent",
    "AnyReasoningBlock",
    "AssistantBlock",
    "AssistantMessage",
    "AudioContentBlock",
    "ChatMessage",
    "ContextOverflowError",
    "DiskSpillMiddleware",
    "EmptyParams",
    "EncryptedReasoningBlock",
    "ExecEnvSink",
    "ImageContentBlock",
    "IncompleteResponseError",
    "LLMClient",
    "OpaqueBlock",
    "OutputTokenLimitError",
    "ReasoningBlock",
    "ReasoningRefBlock",
    "RedactedReasoningBlock",
    "SessionAgent",
    "SignedReasoningBlock",
    "Sink",
    "SubAgentMetadata",
    "SystemMessage",
    "TextBlock",
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
    "final_text",
    "joined_text",
    "reasoning_blocks",
    "tool_call_blocks",
    "tools",
]

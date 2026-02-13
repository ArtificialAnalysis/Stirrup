"""Shared utilities for OpenAI-compatible message and tool conversion.

These helper functions convert Stirrup's internal message and tool formats
to the OpenAI API format. Since LiteLLM and the OpenAI SDK use identical
formats, these utilities are shared between both client implementations.
"""

from typing import Any

from stirrup.core.models import (
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    Content,
    EmptyParams,
    ImageContentBlock,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
    VideoContentBlock,
)

__all__ = [
    "content_to_openai",
    "to_openai_messages",
    "to_openai_tools",
]


def to_openai_tools(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    """Convert Tool objects to OpenAI function calling format.

    Args:
        tools: Dictionary mapping tool names to Tool objects.

    Returns:
        List of tool definitions in OpenAI's function calling format.

    Example:
        >>> tools = {"calculator": calculator_tool}
        >>> openai_tools = to_openai_tools(tools)
        >>> # Returns: [{"type": "function", "function": {"name": "calculator", ...}}]
    """
    out: list[dict[str, Any]] = []
    for t in tools.values():
        function: dict[str, Any] = {
            "name": t.name,
            "description": t.description,
        }
        if t.parameters is not EmptyParams:
            function["parameters"] = t.parameters.model_json_schema()
        tool_payload: dict[str, Any] = {
            "type": "function",
            "function": function,
        }
        out.append(tool_payload)
    return out


def content_to_openai(content: Content) -> list[dict[str, Any]] | str:
    """Convert Content blocks to OpenAI message content format.

    Handles text, images, audio, and video content blocks, converting them
    to the appropriate OpenAI API structure.

    Args:
        content: Either a string or list of content blocks.

    Returns:
        List of content dictionaries in OpenAI format, or the original string
        wrapped in a text content block.

    Raises:
        NotImplementedError: If an unsupported content block type is encountered.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            out.append({"type": "text", "text": block})
        elif isinstance(block, ImageContentBlock):
            out.append({"type": "image_url", "image_url": {"url": block.to_base64_url()}})
        elif isinstance(block, AudioContentBlock):
            out.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": block.to_base64_url().split(",")[1],
                        "format": block.extension,
                    },
                }
            )
        elif isinstance(block, VideoContentBlock):
            out.append({"type": "file", "file": {"file_data": block.to_base64_url()}})
        else:
            raise NotImplementedError(f"Unsupported content block: {type(block)}")
    return out


def to_openai_messages(msgs: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage list to OpenAI-compatible message dictionaries.

    Handles all message types: SystemMessage, UserMessage, AssistantMessage,
    and ToolMessage. Preserves reasoning content and tool calls for assistant
    messages.

    Args:
        msgs: List of ChatMessage objects (System, User, Assistant, or Tool messages).

    Returns:
        List of message dictionaries ready for the OpenAI API.

    Raises:
        NotImplementedError: If an unsupported message type is encountered.
    """
    out: list[dict[str, Any]] = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": content_to_openai(m.content)})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": content_to_openai(m.content)})
        elif isinstance(m, AssistantMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": content_to_openai(m.content)}

            if m.reasoning:
                if m.reasoning.content:
                    msg["reasoning_content"] = m.reasoning.content

                if m.reasoning.signature:
                    msg["thinking_blocks"] = [
                        {"type": "thinking", "signature": m.reasoning.signature, "thinking": m.reasoning.content}
                    ]

            if m.tool_calls:
                msg["tool_calls"] = []
                for tool in m.tool_calls:
                    tool_dict = tool.model_dump()
                    tool_dict["id"] = tool.tool_call_id
                    tool_dict["type"] = "function"
                    if tool.signature is not None:
                        tool_dict["provider_specific_fields"] = {
                            "thought_signature": tool.signature,
                        }
                    tool_dict["function"] = {
                        "name": tool.name,
                        "arguments": tool.arguments,
                    }
                    msg["tool_calls"].append(tool_dict)

            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "content": content_to_openai(m.content),
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                }
            )
        else:
            raise NotImplementedError(f"Unsupported message type: {type(m)}")

    return out

"""Shared utilities for OpenAI-compatible message and tool conversion.

These helper functions convert Stirrup's internal message and tool formats
to the OpenAI API format. Since LiteLLM and the OpenAI SDK use identical
formats, these utilities are shared between both client implementations.
"""

from collections.abc import Sequence
from typing import Any, assert_never

from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    Content,
    EmptyParams,
    EncryptedReasoningBlock,
    ImageContentBlock,
    OpaqueBlock,
    ReasoningBlock,
    ReasoningRefBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    SystemMessage,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    VideoContentBlock,
    joined_text,
    reasoning_blocks,
    tool_call_blocks,
)

__all__ = [
    "content_to_openai",
    "to_openai_messages",
    "to_openai_tools",
    "validate_token_budgets",
]


def validate_token_budgets(max_tokens: int, context_window_tokens: int) -> None:
    """Reject an invalid budget pair at client construction.

    Raises:
        ValueError: If ``context_window_tokens`` is not positive, or
            ``max_tokens`` exceeds it.
    """
    if context_window_tokens <= 0:
        raise ValueError(f"context_window_tokens must be positive, got {context_window_tokens!r}")
    if max_tokens > context_window_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) must not exceed context_window_tokens ({context_window_tokens})")


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
        List of content dictionaries in OpenAI format; a non-empty string is wrapped
        as a single text part. Empty string/list with no parts becomes ``""``

    Raises:
        NotImplementedError: If an unsupported content block type is encountered.
    """
    if isinstance(content, str):
        if content == "":
            return ""
        return [{"type": "text", "text": content}]

    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            if block == "":
                continue
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
    if not out:
        return ""
    return out


def _assistant_content(blocks: Sequence[AssistantBlock]) -> Content:
    """Project answer/media blocks without using the deprecated message accessor."""
    media_types = ImageContentBlock | VideoContentBlock | AudioContentBlock
    if any(isinstance(block, media_types) for block in blocks):
        return [
            block.text if isinstance(block, TextBlock) else block
            for block in blocks
            if isinstance(block, TextBlock | media_types)
        ]
    return joined_text(blocks) or ""


def _validate_chat_assistant_blocks(
    blocks: Sequence[AssistantBlock],
    *,
    allow_tool_call_signatures: bool,
) -> None:
    for block in blocks:
        match block:
            case TextBlock(signature=None) | ReasoningBlock() | SignedReasoningBlock() | RedactedReasoningBlock():
                pass
            case ToolCall(has_provider_tool_call_id=True, signature=None):
                pass
            case ToolCall(has_provider_tool_call_id=True) if allow_tool_call_signatures:
                pass
            case ImageContentBlock() | VideoContentBlock() | AudioContentBlock():
                pass
            case TextBlock():
                raise TypeError("OpenAI-compatible chat cannot pass back signed text blocks")
            case ToolCall(has_provider_tool_call_id=True):
                raise NotImplementedError("Direct OpenAI-compatible chat cannot pass back signed tool calls")
            case ToolCall():
                raise NotImplementedError(
                    "OpenAI-compatible chat cannot pass back a tool call without a provider-issued ID"
                )
            case ReasoningRefBlock() | EncryptedReasoningBlock() | OpaqueBlock():
                raise NotImplementedError(
                    f"OpenAI-compatible chat cannot pass back {type(block).__name__} assistant blocks"
                )
            case _:
                assert_never(block)


def to_openai_messages(
    msgs: list[ChatMessage],
    *,
    allow_tool_call_signatures: bool = False,
) -> list[dict[str, Any]]:
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
            _validate_chat_assistant_blocks(
                m.blocks,
                allow_tool_call_signatures=allow_tool_call_signatures,
            )
            # Note: message metadata is deliberately NOT sent on the wire — it is
            # integrator/user state, opaque to the framework.
            msg: dict[str, Any] = {"role": "assistant", "content": content_to_openai(_assistant_content(m.blocks))}

            rblocks = reasoning_blocks(m.blocks)
            reasoning_parts: list[str] = []
            for block in rblocks:
                if isinstance(block, RedactedReasoningBlock):
                    continue
                if block.content is not None:
                    reasoning_parts.append(block.content)
            reasoning_content = "".join(reasoning_parts)
            if reasoning_content:
                msg["reasoning_content"] = reasoning_content

            # Signed passback is per-block: one thinking entry per signed block,
            # redacted payloads re-emitted verbatim, in emission order.
            thinking_payload: list[dict[str, Any]] = []
            for block in rblocks:
                if isinstance(block, SignedReasoningBlock):
                    thinking_payload.append(
                        {"type": "thinking", "signature": block.signature, "thinking": block.content}
                    )
                elif isinstance(block, RedactedReasoningBlock):
                    thinking_payload.append({"type": "redacted_thinking", "data": block.data})
            if thinking_payload:
                msg["thinking_blocks"] = thinking_payload

            tool_calls = tool_call_blocks(m.blocks)
            if tool_calls:
                msg["tool_calls"] = []
                for tool in tool_calls:
                    tool_dict: dict[str, Any] = {
                        "id": tool.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": tool.arguments,
                        },
                    }
                    if tool.signature is not None:
                        tool_dict["provider_specific_fields"] = {
                            "thought_signature": tool.signature,
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

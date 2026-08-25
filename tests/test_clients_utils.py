"""Tests for OpenAI client utility helpers."""

from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.models import AssistantMessage, EmptyParams, TextBlock, TokenUsage, Tool, ToolResult


def test_assistant_message_generates_id() -> None:
    first = AssistantMessage(
        blocks=[
            TextBlock(text="Hello"),
        ],
        token_usage=TokenUsage(),
        metadata={},
    )
    second = AssistantMessage(
        blocks=[
            TextBlock(text="Hello again"),
        ],
        token_usage=TokenUsage(),
        metadata={},
    )

    assert first.id
    assert second.id
    assert first.id != second.id


def test_to_openai_messages_never_sends_metadata() -> None:
    """Message metadata is integrator/user state — opaque to the framework, never on the wire."""
    message = AssistantMessage(
        blocks=[
            TextBlock(text="Hello"),
        ],
        token_usage=TokenUsage(),
        metadata={"source": "cache", "attempt": 2},
    )

    result = to_openai_messages([message])

    assert result == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
        }
    ]


def test_to_openai_tools_prefers_an_explicit_parameter_schema() -> None:
    schema = {"type": "object", "unevaluatedProperties": False}
    tool = Tool[EmptyParams, None](
        name="exact_schema",
        description="Keep this schema unchanged",
        parameter_schema=schema,
        executor=lambda _: ToolResult(content="ok"),
    )

    result = to_openai_tools({tool.name: tool})

    assert result[0]["function"]["parameters"] == schema

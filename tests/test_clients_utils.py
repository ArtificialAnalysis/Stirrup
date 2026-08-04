"""Tests for OpenAI client utility helpers."""

from stirrup.clients.utils import to_openai_messages
from stirrup.core.models import AssistantMessage, TextBlock, TokenUsage


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

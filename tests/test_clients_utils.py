"""Tests for OpenAI client utility helpers."""

from stirrup.clients.utils import to_openai_messages
from stirrup.core.models import AssistantMessage, TokenUsage


def test_to_openai_messages_forwards_assistant_metadata() -> None:
    message = AssistantMessage(
        content="Hello",
        token_usage=TokenUsage(),
        metadata={"source": "cache", "attempt": 2},
    )

    result = to_openai_messages([message])

    assert result == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "metadata": {"source": "cache", "attempt": 2},
        }
    ]

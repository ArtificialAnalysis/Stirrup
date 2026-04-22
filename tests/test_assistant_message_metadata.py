from pydantic import BaseModel
import pytest

from stirrup.core.cache import deserialize_message, serialize_message
from stirrup.core.models import AssistantMessage, EmptyMetadata, TokenUsage


class ResponseMetadata(BaseModel):
    request_id: str


def test_assistant_message_supports_typed_metadata() -> None:
    # Create a typed assistant message
    msg: AssistantMessage[ResponseMetadata] = AssistantMessage[ResponseMetadata](
        content="hi",
        tool_calls=[],
        token_usage=TokenUsage(answer=1),
        metadata=ResponseMetadata(request_id="resp_123"),
    )

    # Assertions
    assert msg.metadata is not None
    assert msg.metadata.request_id == "resp_123"


def test_assistant_message_without_type_defaults_metadata_to_empty_metadata() -> None:
    # Create an untyped assistant message
    msg = AssistantMessage(
        content="hi",
        tool_calls=[],
        token_usage=TokenUsage(answer=1),
    )

    # Assertions
    assert isinstance(msg.metadata, EmptyMetadata)


def test_assistant_message_typed_metadata_cannot_be_none() -> None:
    # Verify typed metadata rejects None
    with pytest.raises(ValueError, match="valid dictionary or instance of ResponseMetadata"):
        AssistantMessage[ResponseMetadata](
            content="hi",
            tool_calls=[],
            token_usage=TokenUsage(answer=1),
            metadata=None,  # ty: ignore[invalid-argument-type]
        )


def test_assistant_message_typed_metadata_cannot_be_omitted() -> None:
    # Verify typed metadata cannot be omitted
    with pytest.raises(ValueError, match="metadata is required"):
        AssistantMessage[ResponseMetadata](
            content="hi",
            tool_calls=[],
            token_usage=TokenUsage(answer=1),
        )


def test_assistant_message_pydantic_metadata_serializes() -> None:
    # Create a typed assistant message
    msg = AssistantMessage[ResponseMetadata](
        content="hi",
        tool_calls=[],
        token_usage=TokenUsage(answer=1),
        metadata=ResponseMetadata(request_id="resp_123"),
    )

    # Serialize the message
    serialized = serialize_message(msg)

    # Assertions
    assert serialized["metadata"] == {"request_id": "resp_123"}


def test_assistant_message_metadata_round_trips_through_cache() -> None:
    # Create a typed assistant message
    msg = AssistantMessage[ResponseMetadata](
        content="hi",
        tool_calls=[],
        token_usage=TokenUsage(answer=1),
        metadata=ResponseMetadata(request_id="resp_123"),
    )

    # Round-trip through cache serialization
    restored = deserialize_message(serialize_message(msg), ResponseMetadata)

    # Assertions
    assert isinstance(restored, AssistantMessage)
    assert isinstance(restored.metadata, ResponseMetadata)
    assert restored.metadata.request_id == "resp_123"


def test_assistant_message_deserialize_none_metadata_as_empty_metadata() -> None:
    # Deserialize a message with null metadata
    restored = deserialize_message(
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [],
            "token_usage": {"input": 0, "answer": 1, "reasoning": 0},
            "metadata": None,
        }
    )

    # Assertions
    assert isinstance(restored, AssistantMessage)
    assert isinstance(restored.metadata, EmptyMetadata)


def test_assistant_message_deserialize_empty_dict_metadata_as_empty_metadata() -> None:
    # Deserialize a message with empty metadata
    restored = deserialize_message(
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [],
            "token_usage": {"input": 0, "answer": 1, "reasoning": 0},
            "metadata": {},
        }
    )

    # Assertions
    assert isinstance(restored, AssistantMessage)
    assert isinstance(restored.metadata, EmptyMetadata)

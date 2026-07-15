"""Tests for ChatCompletionsClient response parsing."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import AssistantMessage, ReasoningBlock, TextBlock, ToolCall, UserMessage


def _mock_response(
    *,
    content: str | None,
    reasoning_content: str | None = None,
    tool_calls: list[MagicMock] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    message_spec = ["content", "tool_calls"]
    if reasoning_content is not None:
        message_spec.append("reasoning_content")
    message = MagicMock(spec=message_spec)
    message.content = content
    message.tool_calls = tool_calls
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content

    usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "completion_tokens_details"])
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.completion_tokens_details = None

    response = MagicMock()
    response.choices = [MagicMock(message=message, finish_reason=finish_reason)]
    response.usage = usage
    return response


def _mock_tool_call(call_id: str | None, name: str, arguments: str) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _client_with_response(response: MagicMock) -> ChatCompletionsClient:
    client = ChatCompletionsClient(model="gpt-4o", api_key="test-key")
    client._client.chat.completions.create = AsyncMock(return_value=response)  # type: ignore[method-assign]  # noqa: SLF001
    return client


class TestChatCompletionsParsing:
    """Blocks are built in canonical channel order: reasoning → text → tool calls."""

    async def test_reasoning_text_and_tool_calls_in_channel_order(self) -> None:
        response = _mock_response(
            content="The answer",
            reasoning_content="Thinking...",
            tool_calls=[_mock_tool_call("call_1", "search", '{"q": "x"}')],
        )
        client = _client_with_response(response)

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        assert result.blocks == (
            ReasoningBlock(content="Thinking..."),
            TextBlock(text="The answer"),
            ToolCall(tool_call_id="call_1", name="search", arguments='{"q": "x"}'),
        )

    async def test_tool_calls_only_turn_has_no_text_block(self) -> None:
        response = _mock_response(content=None, tool_calls=[_mock_tool_call("call_2", "lookup", "{}")])
        client = _client_with_response(response)

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        assert result.blocks == (ToolCall(tool_call_id="call_2", name="lookup", arguments="{}"),)

    async def test_idless_tool_call_gets_stable_internal_id_without_provider_provenance(self) -> None:
        response = _mock_response(content=None, tool_calls=[_mock_tool_call(None, "lookup", "{}")])
        client = _client_with_response(response)

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        [tool_call] = result.blocks
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_call_id
        assert not tool_call.has_provider_tool_call_id

    async def test_message_without_reasoning_content_attribute(self) -> None:
        response = _mock_response(content="plain answer")
        client = _client_with_response(response)

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        assert result.blocks == (TextBlock(text="plain answer"),)

    async def test_signed_tool_call_history_fails_before_direct_request(self) -> None:
        client = _client_with_response(_mock_response(content="unused"))
        create_mock = AsyncMock()
        client._client.chat.completions.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001
        message = AssistantMessage(
            blocks=[ToolCall(tool_call_id="call_1", name="lookup", arguments="{}", signature="google-signature")]
        )

        with pytest.raises(NotImplementedError, match="cannot pass back signed tool calls"):
            await client.generate(messages=[message], tools={})

        create_mock.assert_not_awaited()

    async def test_length_finish_reason_raises_context_overflow(self) -> None:
        response = _mock_response(content="truncat", finish_reason="length")
        client = _client_with_response(response)

        with pytest.raises(ContextOverflowError):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})

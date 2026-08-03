"""Tests for OpenResponsesClient."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import Request, Response
from openai import APIStatusError, BadRequestError, NotFoundError

from stirrup.clients.open_responses_client import (
    OpenResponsesClient,
    _content_to_open_responses_input,
    _content_to_open_responses_output,
    _parse_response_output,
    _to_open_responses_input,
    _to_open_responses_request,
    _to_open_responses_tools,
)
from stirrup.core.exceptions import ContextOverflowError, IncompleteResponseError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    EncryptedReasoningBlock,
    ReasoningBlock,
    ReasoningRefBlock,
    SystemMessage,
    TextBlock,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UserMessage,
    joined_text,
    reasoning_blocks,
    tool_call_blocks,
)


def _status_error(
    error_type: type[APIStatusError],
    status_code: int,
    *,
    code: str,
    param: str | None,
) -> APIStatusError:
    request = Request("POST", "https://api.openai.com/v1/responses")
    return error_type(
        "Responses request failed",
        response=Response(status_code, request=request),
        body={"message": "request failed", "type": "invalid_request_error", "code": code, "param": param},
    )


def _previous_response_not_found_error() -> APIStatusError:
    return _status_error(
        BadRequestError,
        400,
        code="previous_response_not_found",
        param="previous_response_id",
    )


class TestContentConversion:
    """Tests for content conversion functions."""

    def test_string_content_to_input(self) -> None:
        """Test converting string content to input format."""
        result = _content_to_open_responses_input("Hello world")
        assert result == [{"type": "input_text", "text": "Hello world"}]

    def test_list_content_to_input(self) -> None:
        """Test converting list content to input format."""
        result = _content_to_open_responses_input(["Hello", "World"])
        assert result == [
            {"type": "input_text", "text": "Hello"},
            {"type": "input_text", "text": "World"},
        ]

    def test_string_content_to_output(self) -> None:
        """Test converting string content to output format."""
        result = _content_to_open_responses_output("Response text")
        assert result == [{"type": "output_text", "text": "Response text"}]


class TestMessageConversion:
    """Tests for message conversion to OpenResponses format."""

    def test_system_message_becomes_instructions(self) -> None:
        """Test that SystemMessage is extracted as instructions."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            UserMessage(content="Hello"),
        ]
        instructions, input_items = _to_open_responses_input(messages)

        assert instructions == "You are a helpful assistant"
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"

    def test_user_message_conversion(self) -> None:
        """Test UserMessage conversion to input format."""
        messages = [UserMessage(content="Hello")]
        instructions, input_items = _to_open_responses_input(messages)

        assert instructions is None
        assert len(input_items) == 1
        assert input_items[0] == {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }

    def test_unexpected_assistant_block_raises(self) -> None:
        unexpected = cast(AssistantBlock, object())
        message = AssistantMessage.model_construct(blocks=[unexpected])

        # The replay match is statically exhaustive; an out-of-union object trips assert_never.
        with pytest.raises(AssertionError, match="Expected code to be unreachable"):
            _to_open_responses_input([message])

    def test_assistant_message_conversion(self) -> None:
        """Test AssistantMessage conversion to input format."""
        messages = [
            AssistantMessage(
                blocks=[
                    TextBlock(text="I can help with that"),
                ],
                token_usage=TokenUsage(),
            ),
        ]
        instructions, input_items = _to_open_responses_input(messages)

        assert instructions is None
        assert len(input_items) == 1
        assert input_items[0] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I can help with that"}],
        }

    def test_assistant_message_with_tool_calls(self) -> None:
        """Test AssistantMessage with tool calls adds function_call items."""
        messages = [
            AssistantMessage(
                blocks=[
                    TextBlock(text="Let me search for that"),
                    ToolCall(
                        tool_call_id="call_123",
                        name="search",
                        arguments='{"query": "test"}',
                    ),
                ],
                token_usage=TokenUsage(),
            ),
        ]
        _instructions, input_items = _to_open_responses_input(messages)

        assert len(input_items) == 2
        # First item is the assistant message
        assert input_items[0]["role"] == "assistant"
        # Second item is the function call
        assert input_items[1] == {
            "type": "function_call",
            "call_id": "call_123",
            "name": "search",
            "arguments": '{"query": "test"}',
        }

    def test_tool_message_conversion(self) -> None:
        """Test ToolMessage conversion to function_call_output format."""
        messages = [
            ToolMessage(
                content="Search results here",
                tool_call_id="call_123",
                name="search",
            ),
        ]
        _instructions, input_items = _to_open_responses_input(messages)

        assert len(input_items) == 1
        assert input_items[0] == {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Search results here",
        }

    def test_non_string_tool_message_output_raises(self) -> None:
        message = ToolMessage(content=["part one", "part two"], tool_call_id="call_123")

        with pytest.raises(NotImplementedError, match="require string content"):
            _to_open_responses_input([message])

    def test_full_conversation_flow(self) -> None:
        """Test converting a complete conversation with tool use."""
        messages = [
            SystemMessage(content="You are a search assistant"),
            UserMessage(content="Find information about Python"),
            AssistantMessage(
                blocks=[
                    TextBlock(text="I'll search for that"),
                    ToolCall(tool_call_id="call_1", name="search", arguments='{"q": "Python"}'),
                ],
                token_usage=TokenUsage(),
            ),
            ToolMessage(content="Python is a programming language", tool_call_id="call_1", name="search"),
            AssistantMessage(
                blocks=[
                    TextBlock(text="Python is a programming language"),
                ],
                token_usage=TokenUsage(),
            ),
        ]
        instructions, input_items = _to_open_responses_input(messages)

        assert instructions == "You are a search assistant"
        assert len(input_items) == 5  # user, assistant, function_call, function_call_output, assistant

    def test_provider_response_id_slices_stored_history_but_keeps_instructions(self) -> None:
        messages = [
            SystemMessage(content="Keep this instruction"),
            UserMessage(content="old input"),
            AssistantMessage(
                provider_response_id="resp_123",
                blocks=[
                    ReasoningRefBlock(id="rs_123", content="summary"),
                    ToolCall(tool_call_id="call_123", name="lookup", arguments="{}"),
                ],
            ),
            ToolMessage(content="tool output", tool_call_id="call_123"),
            UserMessage(content="new input"),
        ]

        instructions, previous_response_id, input_items = _to_open_responses_request(
            messages,
            use_provider_response_id=True,
        )

        assert instructions == "Keep this instruction"
        assert previous_response_id == "resp_123"
        assert input_items == [
            {"type": "function_call_output", "call_id": "call_123", "output": "tool output"},
            {"role": "user", "content": [{"type": "input_text", "text": "new input"}]},
        ]

    def test_stored_continuation_rejects_provider_idless_tool_call(self) -> None:
        call = ToolCall.from_provider(provider_id=None, name="lookup", arguments="{}")
        messages = [
            AssistantMessage(provider_response_id="resp_123", blocks=[call]),
            ToolMessage(content="done", tool_call_id=call.tool_call_id, name="lookup"),
        ]

        with pytest.raises(NotImplementedError, match="cannot correlate a provider-idless tool call"):
            _to_open_responses_request(messages, use_provider_response_id=True)

    def test_stateless_request_replays_encrypted_reasoning_instead_of_using_response_id(self) -> None:
        messages = [
            AssistantMessage(
                provider_response_id="resp_123",
                blocks=[
                    EncryptedReasoningBlock(
                        id="rs_123",
                        encrypted_content="opaque",
                        content="private detail",
                        summary=["summary"],
                    )
                ],
            )
        ]

        _instructions, previous_response_id, input_items = _to_open_responses_request(
            messages,
            use_provider_response_id=False,
        )

        assert previous_response_id is None
        assert input_items == [
            {
                "type": "reasoning",
                "id": "rs_123",
                "content": [{"type": "reasoning_text", "text": "private detail"}],
                "summary": [{"type": "summary_text", "text": "summary"}],
                "encrypted_content": "opaque",
            }
        ]

    def test_stateless_request_rejects_reference_only_reasoning(self) -> None:
        message = AssistantMessage(blocks=[ReasoningRefBlock(id="rs_123", content="summary")])

        with pytest.raises(NotImplementedError, match="requires encrypted reasoning content"):
            _to_open_responses_request([message], use_provider_response_id=False)

    def test_stateful_request_without_continuation_rejects_reference_only_reasoning(self) -> None:
        message = AssistantMessage(blocks=[ReasoningRefBlock(id="rs_123", content="summary")])

        with pytest.raises(NotImplementedError, match="requires encrypted reasoning content"):
            _to_open_responses_request([message], use_provider_response_id=True)

    @pytest.mark.parametrize(
        "block",
        [
            TextBlock(text="signed", signature="google-signature"),
            ToolCall(
                tool_call_id="internal",
                has_provider_tool_call_id=False,
                name="lookup",
                arguments="{}",
            ),
            ReasoningBlock(content="unreferenced"),
        ],
    )
    def test_unrepresentable_assistant_blocks_fail_loudly(self, block: AssistantBlock) -> None:
        with pytest.raises((TypeError, NotImplementedError)):
            _to_open_responses_input([AssistantMessage(blocks=[block])])


class TestToolConversion:
    """Tests for tool conversion to OpenResponses format."""

    def test_tool_format_has_name_at_top_level(self) -> None:
        """Test that tools have name at top level, not nested under function."""
        from pydantic import BaseModel

        from stirrup.core.models import Tool, ToolResult

        class SearchParams(BaseModel):
            query: str

        search_tool = Tool[SearchParams, None](
            name="search",
            description="Search the web",
            parameters=SearchParams,
            executor=lambda _p: ToolResult(content="results"),
        )

        result = _to_open_responses_tools({"search": search_tool})

        assert len(result) == 1
        tool = result[0]
        # Key assertion: name is at top level, not nested
        assert tool["type"] == "function"
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web"
        assert "parameters" in tool
        assert "function" not in tool  # Should NOT have nested function key

    def test_tool_without_parameters(self) -> None:
        """Test tool with EmptyParams doesn't include parameters."""
        from stirrup.core.models import EmptyParams, Tool, ToolResult

        time_tool = Tool[EmptyParams, None](
            name="get_time",
            description="Get current time",
            executor=lambda _: ToolResult(content="12:00"),
        )

        result = _to_open_responses_tools({"get_time": time_tool})

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_time"
        assert "parameters" not in tool


class TestResponseParsing:
    """Tests for parsing OpenResponses output."""

    def test_parse_message_output(self) -> None:
        """Test parsing a simple message response."""
        output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="Hello there!")],
            )
        ]
        blocks = _parse_response_output(output)

        assert blocks == [TextBlock(text="Hello there!")]

    def test_parse_function_call_output(self) -> None:
        """Test parsing a function call response."""
        fn_call = MagicMock()
        fn_call.type = "function_call"
        fn_call.call_id = "call_abc"
        fn_call.name = "get_weather"
        fn_call.arguments = '{"city": "NYC"}'
        output = [fn_call]
        blocks = _parse_response_output(output)

        assert len(blocks) == 1
        tool_call = blocks[0]
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_call_id == "call_abc"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == '{"city": "NYC"}'

    def test_parse_idless_function_call_synthesizes_one_internal_id(self) -> None:
        fn_call = MagicMock()
        fn_call.type = "function_call"
        fn_call.call_id = None
        fn_call.name = "get_weather"
        fn_call.arguments = "{}"

        [tool_call] = _parse_response_output([fn_call])

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_call_id
        assert not tool_call.has_provider_tool_call_id

    def test_parse_reasoning_output(self) -> None:
        """Test parsing a response with reasoning (no id: in-band reasoning block)."""
        reasoning_item = MagicMock(spec=["type", "summary"])
        reasoning_item.type = "reasoning"
        reasoning_item.summary = "Let me think about this..."

        output = [
            reasoning_item,
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="The answer is 42")],
            ),
        ]
        blocks = _parse_response_output(output)

        assert blocks == [
            ReasoningBlock(content="Let me think about this..."),
            TextBlock(text="The answer is 42"),
        ]

    def test_parse_stored_reasoning_output_keeps_only_readable_content(self) -> None:
        """Response-level continuation state makes stored reasoning item IDs redundant."""
        content_part = MagicMock(spec=["text"])
        content_part.text = "Private detail."
        summary_part = MagicMock(spec=["text"])
        summary_part.text = "Thinking..."
        reasoning_item = MagicMock(spec=["type", "id", "content", "summary", "encrypted_content"])
        reasoning_item.type = "reasoning"
        reasoning_item.id = "rs_123"
        reasoning_item.content = [content_part]
        reasoning_item.summary = [summary_part]
        reasoning_item.encrypted_content = None

        empty_reasoning_item = MagicMock(spec=["type", "id", "content", "summary"])
        empty_reasoning_item.type = "reasoning"
        empty_reasoning_item.id = "rs_456"
        empty_reasoning_item.content = []
        empty_reasoning_item.summary = []

        blocks = _parse_response_output([reasoning_item, empty_reasoning_item])

        assert blocks == [ReasoningBlock(content="Private detail.\nThinking...")]

    def test_stateless_parse_rejects_reference_only_reasoning(self) -> None:
        reasoning_item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        reasoning_item.type = "reasoning"
        reasoning_item.id = "rs_123"
        reasoning_item.summary = []
        reasoning_item.encrypted_content = None

        with pytest.raises(NotImplementedError, match="reference-only reasoning"):
            _parse_response_output([reasoning_item], allow_reference_reasoning=False)

    def test_parse_reasoning_output_with_encrypted_content(self) -> None:
        """A reasoning item carrying encrypted_content becomes an EncryptedReasoningBlock,
        with summary parts kept split for verbatim re-emission."""
        part_1 = MagicMock(spec=["text"])
        part_1.text = "First thought."
        part_2 = MagicMock(spec=["text"])
        part_2.text = "Second thought."
        content_part = MagicMock(spec=["text"])
        content_part.text = "Private detail."
        encrypted_item = MagicMock(spec=["type", "id", "content", "summary", "encrypted_content"])
        encrypted_item.type = "reasoning"
        encrypted_item.id = "rs_789"
        encrypted_item.content = [content_part]
        encrypted_item.summary = [part_1, part_2]
        encrypted_item.encrypted_content = "opaque-zdr-payload"

        bare_encrypted_item = MagicMock(spec=["type", "id", "content", "summary", "encrypted_content"])
        bare_encrypted_item.type = "reasoning"
        bare_encrypted_item.id = "rs_790"
        bare_encrypted_item.content = []
        bare_encrypted_item.summary = []
        bare_encrypted_item.encrypted_content = "opaque-zdr-payload-2"

        blocks = _parse_response_output([encrypted_item, bare_encrypted_item])

        assert blocks == [
            EncryptedReasoningBlock(
                id="rs_789",
                encrypted_content="opaque-zdr-payload",
                content="Private detail.",
                summary=["First thought.", "Second thought."],
            ),
            EncryptedReasoningBlock(id="rs_790", encrypted_content="opaque-zdr-payload-2"),
        ]

    def test_parse_mixed_output_preserves_order(self) -> None:
        """Interleaved message/function_call items keep their emission order."""
        fn_call_1 = MagicMock()
        fn_call_1.type = "function_call"
        fn_call_1.call_id = "call_1"
        fn_call_1.name = "tool1"
        fn_call_1.arguments = "{}"

        fn_call_2 = MagicMock()
        fn_call_2.type = "function_call"
        fn_call_2.call_id = "call_2"
        fn_call_2.name = "tool2"
        fn_call_2.arguments = '{"x": 1}'

        output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="I'll help you with that")],
            ),
            fn_call_1,
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="And one more thing")],
            ),
            fn_call_2,
        ]
        blocks = _parse_response_output(output)

        assert [type(b) for b in blocks] == [TextBlock, ToolCall, TextBlock, ToolCall]
        assert blocks[0] == TextBlock(text="I'll help you with that")
        assert blocks[2] == TextBlock(text="And one more thing")
        tool_names = [b.name for b in blocks if isinstance(b, ToolCall)]
        assert tool_names == ["tool1", "tool2"]

    def test_unexpected_output_item_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Unsupported OpenAI Responses output item type"):
            _parse_response_output([MagicMock(type="computer_call")])

    def test_unexpected_message_content_raises(self) -> None:
        output = [MagicMock(type="message", content=[MagicMock(type="output_audio")])]

        with pytest.raises(NotImplementedError, match="Unsupported OpenAI Responses message content type"):
            _parse_response_output(output)

    def test_refusal_content_surfaces_as_text(self) -> None:
        """A refusal is a normal API response: its text becomes the assistant's answer."""
        output = [MagicMock(type="message", content=[MagicMock(type="refusal", refusal="I can't help with that.")])]

        assert _parse_response_output(output) == [TextBlock(text="I can't help with that.")]

    def test_encrypted_content_without_id_raises(self) -> None:
        """The item id is the passback handle; encrypted payloads without one cannot be re-emitted."""
        item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        item.type = "reasoning"
        item.id = None
        item.summary = []
        item.encrypted_content = "opaque-zdr-payload"

        with pytest.raises(ValueError, match="encrypted_content but no id"):
            _parse_response_output([item])

    def test_reasoning_summary_entry_without_text_raises(self) -> None:
        bad_part = MagicMock(spec=["type"])
        bad_part.type = "summary_image"
        item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        item.type = "reasoning"
        item.id = "rs_1"
        item.summary = [bad_part]
        item.encrypted_content = None

        with pytest.raises(ValueError, match="reasoning summary entry has no text"):
            _parse_response_output([item])


class TestOpenResponsesClient:
    """Tests for OpenResponsesClient class."""

    def test_client_properties(self) -> None:
        """Test client property accessors."""
        client = OpenResponsesClient(
            model="gpt-4o",
            max_tokens=50_000,
            context_window_tokens=120_000,
            api_key="test-key",
        )

        assert client.model_slug == "gpt-4o"
        assert client.max_tokens == 50_000
        assert client.context_window_tokens == 120_000

    async def test_generate_basic(self) -> None:
        """Test basic generation with mocked response."""
        client = OpenResponsesClient(
            model="gpt-4o",
            context_window_tokens=64_000,
            api_key="test-key",
        )

        # Mock the responses.create method
        mock_response = MagicMock()
        mock_response.id = "resp_basic"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="Hello!")],
            )
        ]
        mock_response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            output_tokens_details=None,
        )

        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        result = await client.generate(
            messages=[UserMessage(content="Hi")],
            tools={},
        )

        assert isinstance(result, AssistantMessage)
        assert result.provider_response_id == "resp_basic"
        assert joined_text(result.blocks) == "Hello!"
        assert result.token_usage.input == 10
        assert result.token_usage.answer == 5

    async def test_generate_encrypted_reasoning_sends_stateless_params(self) -> None:
        """encrypted_reasoning=True sends store=false + the encrypted-content include."""
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
            context_window_tokens=64_000,
            encrypted_reasoning=True,
        )

        mock_response = MagicMock()
        mock_response.id = "resp_stateless"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="Hello!")],
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5, output_tokens_details=None)

        create_mock = AsyncMock(return_value=mock_response)
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        request_kwargs = create_mock.call_args.kwargs
        assert request_kwargs["store"] is False
        assert request_kwargs["include"] == ["reasoning.encrypted_content"]
        assert result.provider_response_id is None

    async def test_stateful_generate_requires_response_id(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        mock_response = MagicMock(
            id=None,
            status="completed",
            output=[MagicMock(type="message", content=[MagicMock(type="output_text", text="Hello!")])],
            usage=MagicMock(input_tokens=10, output_tokens=5, output_tokens_details=None),
        )
        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(ValueError, match="require a non-empty response ID"):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})

    async def test_generate_uses_previous_response_id(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        mock_response = MagicMock(
            id="resp_next",
            status="completed",
            output=[MagicMock(type="message", content=[MagicMock(type="output_text", text="next")])],
            usage=MagicMock(input_tokens=3, output_tokens=2, output_tokens_details=None),
        )
        create_mock = AsyncMock(return_value=mock_response)
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        result = await client.generate(
            messages=[
                SystemMessage(content="Always be concise"),
                UserMessage(content="old"),
                AssistantMessage(provider_response_id="resp_previous", blocks=[TextBlock(text="old answer")]),
                UserMessage(content="new"),
            ],
            tools={},
        )

        request_kwargs = create_mock.call_args.kwargs
        assert request_kwargs["previous_response_id"] == "resp_previous"
        assert request_kwargs["instructions"] == "Always be concise"
        assert request_kwargs["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "new"}]}]
        assert result.provider_response_id == "resp_next"

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("background", True),
            ("conversation", "conv_123"),
            ("model", "other-model"),
            ("input", []),
            ("instructions", "hidden override"),
            ("max_output_tokens", 1),
            ("previous_response_id", "resp_configured"),
            ("store", False),
            ("stream", True),
        ],
    )
    def test_client_owned_kwargs_raise_at_init(self, key: str, value: object) -> None:
        with pytest.raises(ValueError, match="owns request keys"):
            OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000, kwargs={key: value})

    @pytest.mark.parametrize("key", ["tools", "tool_choice"])
    async def test_tool_kwargs_conflict_with_dedicated_tools(self, key: str) -> None:
        from stirrup.core.models import EmptyParams, Tool, ToolResult

        client = OpenResponsesClient(
            model="gpt-4o", api_key="test-key", context_window_tokens=64_000, kwargs={key: object()}
        )
        tool = Tool[EmptyParams, None](
            name="get_time",
            description="Get current time",
            executor=lambda _: ToolResult(content="12:00"),
        )

        with pytest.raises(ValueError, match=rf"{key}.*conflict"):
            await client.generate(messages=[UserMessage(content="Hi")], tools={tool.name: tool})

    async def test_reasoning_kwarg_conflicts_with_reasoning_effort(self) -> None:
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
            context_window_tokens=64_000,
            reasoning_effort="medium",
            kwargs={"reasoning": {"effort": "low"}},
        )

        with pytest.raises(ValueError, match=r"reasoning.*conflict"):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})

    async def test_conditional_kwargs_without_dedicated_configuration_are_forwarded(self) -> None:
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
            context_window_tokens=64_000,
            kwargs={
                "tools": [{"type": "web_search_preview"}],
                "tool_choice": "auto",
                "reasoning": {"effort": "low"},
            },
        )
        response = MagicMock(
            id="resp_next",
            status="completed",
            output=[MagicMock(type="message", content=[MagicMock(type="output_text", text="ok")])],
            usage=MagicMock(input_tokens=1, output_tokens=1, output_tokens_details=None),
        )
        create_mock = AsyncMock(return_value=response)
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        await client.generate(messages=[UserMessage(content="Hi")], tools={})

        assert create_mock.call_args.kwargs["tools"] == [{"type": "web_search_preview"}]
        assert create_mock.call_args.kwargs["tool_choice"] == "auto"
        assert create_mock.call_args.kwargs["reasoning"] == {"effort": "low"}

    @pytest.mark.parametrize(
        ("error_type", "status_code"),
        [(BadRequestError, 400), (NotFoundError, 404)],
    )
    async def test_missing_stored_response_retries_once_with_full_history(
        self,
        error_type: type[APIStatusError],
        status_code: int,
    ) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        mock_response = MagicMock(
            id="resp_next",
            status="completed",
            output=[MagicMock(type="message", content=[MagicMock(type="output_text", text="ok")])],
            usage=MagicMock(input_tokens=1, output_tokens=1, output_tokens_details=None),
        )
        missing_error = _status_error(
            error_type,
            status_code,
            code="previous_response_not_found",
            param="previous_response_id",
        )
        create_mock = AsyncMock(side_effect=[missing_error, mock_response])
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        result = await client.generate(
            messages=[
                SystemMessage(content="Be concise"),
                UserMessage(content="old question"),
                AssistantMessage(provider_response_id="resp_history", blocks=[TextBlock(text="old answer")]),
                UserMessage(content="new"),
            ],
            tools={},
        )

        first_request, replay_request = [call.kwargs for call in create_mock.call_args_list]
        assert first_request["previous_response_id"] == "resp_history"
        assert first_request["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "new"}]}]
        assert "previous_response_id" not in replay_request
        assert replay_request["instructions"] == "Be concise"
        assert replay_request["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "old question"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "old answer"}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "new"}]},
        ]
        assert result.provider_response_id == "resp_next"

    async def test_missing_stored_response_with_reference_reasoning_is_unrecoverable(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        create_mock = AsyncMock(side_effect=_previous_response_not_found_error())
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(RuntimeError, match="cannot be replayed exactly"):
            await client.generate(
                messages=[
                    AssistantMessage(
                        provider_response_id="resp_history",
                        blocks=[ReasoningRefBlock(id="rs_1", content="summary")],
                    ),
                    UserMessage(content="new"),
                ],
                tools={},
            )

        assert create_mock.await_count == 1

    @pytest.mark.parametrize(
        "error",
        [
            _status_error(BadRequestError, 400, code="invalid_model", param="model"),
            _status_error(NotFoundError, 404, code="not_found", param=None),
        ],
    )
    async def test_unrelated_status_error_does_not_trigger_history_replay(self, error: APIStatusError) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        create_mock = AsyncMock(side_effect=error)
        client._client.responses.create = create_mock  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(type(error)):
            await client.generate(
                messages=[
                    AssistantMessage(provider_response_id="resp_history", blocks=[TextBlock(text="old")]),
                    UserMessage(content="new"),
                ],
                tools={},
            )

        assert create_mock.await_count == 1

    @pytest.mark.parametrize("status", ["queued", "in_progress", "failed", "cancelled"])
    async def test_non_completed_response_status_raises(self, status: str) -> None:
        client = OpenResponsesClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
        response = MagicMock(status=status, error={"message": "provider failure"})
        client._client.responses.create = AsyncMock(return_value=response)  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(RuntimeError, match=repr(status)):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})

    async def test_generate_with_tools(self) -> None:
        """Test generation with tool calls."""
        from stirrup.core.models import EmptyParams, Tool, ToolResult

        client = OpenResponsesClient(
            model="gpt-4o",
            context_window_tokens=64_000,
            api_key="test-key",
        )

        # Mock response with function call
        fn_call = MagicMock()
        fn_call.type = "function_call"
        fn_call.call_id = "call_xyz"
        fn_call.name = "get_time"
        fn_call.arguments = "{}"

        mock_response = MagicMock()
        mock_response.id = "resp_tools"
        mock_response.status = "completed"
        mock_response.output = [fn_call]
        mock_response.usage = MagicMock(
            input_tokens=15,
            output_tokens=8,
            output_tokens_details=None,
        )

        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        test_tool = Tool[EmptyParams, None](
            name="get_time",
            description="Get current time",
            executor=lambda _: ToolResult(content="12:00"),
        )

        result = await client.generate(
            messages=[UserMessage(content="What time is it?")],
            tools={"get_time": test_tool},
        )

        tool_calls = tool_call_blocks(result.blocks)
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_time"
        assert tool_calls[0].tool_call_id == "call_xyz"

    async def test_generate_with_reasoning_tokens(self) -> None:
        """Test that reasoning tokens are properly extracted."""
        client = OpenResponsesClient(
            model="o1-preview",
            context_window_tokens=64_000,
            api_key="test-key",
            reasoning_effort="medium",
        )

        reasoning_item = MagicMock(spec=["type", "summary"])
        reasoning_item.type = "reasoning"
        reasoning_item.summary = "Thinking step by step..."

        mock_response = MagicMock()
        mock_response.id = "resp_reasoning"
        mock_response.status = "completed"
        mock_response.output = [
            reasoning_item,
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="The answer")],
            ),
        ]
        mock_response.usage = MagicMock(
            input_tokens=20,
            output_tokens=100,  # Total including reasoning
            output_tokens_details=MagicMock(reasoning_tokens=80),
        )

        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        result = await client.generate(
            messages=[UserMessage(content="Solve this")],
            tools={},
        )

        assert reasoning_blocks(result.blocks) == [ReasoningBlock(content="Thinking step by step...")]
        assert result.token_usage.reasoning == 80
        assert result.token_usage.answer == 20  # 100 - 80

    async def test_output_limit_is_forwarded_and_reported_without_retry(self) -> None:
        client = OpenResponsesClient(
            model="gpt-4o",
            max_tokens=456,
            context_window_tokens=120_000,
            api_key="test-key",
        )
        mock_response = MagicMock(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[],
            usage=MagicMock(input_tokens=100, output_tokens=0),
        )
        provider_call = AsyncMock(return_value=mock_response)
        client._client.responses.create = provider_call  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(OutputTokenLimitError) as exc_info:
            await client.generate(
                messages=[UserMessage(content="Very long request")],
                tools={},
            )

        assert exc_info.value.model_slug == "gpt-4o"
        assert exc_info.value.max_tokens == 456
        assert exc_info.value.provider_reason == "max_output_tokens"
        provider_call.assert_awaited_once()
        provider_request = provider_call.await_args
        assert provider_request is not None
        assert provider_request.kwargs["max_output_tokens"] == 456

    async def test_context_length_rejection_surfaces_as_context_overflow(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", context_window_tokens=64_000, api_key="test-key")
        response = httpx.Response(
            400,
            json={"error": {"message": "boom", "type": "invalid_request_error", "code": "context_length_exceeded"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )
        client._client.responses.create = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            side_effect=BadRequestError("boom", response=response, body=response.json()["error"])
        )

        with pytest.raises(ContextOverflowError):
            await client.generate(messages=[UserMessage(content="Very long request")], tools={})

    async def test_unrelated_bad_request_is_not_mapped_to_context_overflow(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", context_window_tokens=64_000, api_key="test-key")
        response = httpx.Response(
            400,
            json={"error": {"message": "boom", "type": "invalid_request_error", "code": "invalid_value"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )
        client._client.responses.create = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            side_effect=BadRequestError("boom", response=response, body=response.json()["error"])
        )

        with pytest.raises(BadRequestError):
            await client.generate(messages=[UserMessage(content="hello")], tools={})

    async def test_non_output_limit_incomplete_raises_stirrup_error(self) -> None:
        client = OpenResponsesClient(model="gpt-4o", context_window_tokens=64_000, api_key="test-key")
        mock_response = MagicMock(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="content_filter"),
            output=[],
            usage=None,
        )
        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(IncompleteResponseError, match="content_filter"):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})

    async def test_instructions_from_system_message(self) -> None:
        """Test that SystemMessage is passed as instructions parameter."""
        client = OpenResponsesClient(
            model="gpt-4o",
            context_window_tokens=64_000,
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.id = "resp_instructions"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="OK")],
            )
        ]
        mock_response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            output_tokens_details=None,
        )

        mock_create = AsyncMock(return_value=mock_response)
        client._client.responses.create = mock_create  # type: ignore[method-assign]  # noqa: SLF001

        await client.generate(
            messages=[
                SystemMessage(content="You are a helpful assistant"),
                UserMessage(content="Hello"),
            ],
            tools={},
        )

        # Verify instructions was passed
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["instructions"] == "You are a helpful assistant"
        # Verify input doesn't contain the system message
        assert all(item.get("role") != "system" for item in call_kwargs["input"])

    async def test_default_instructions_fallback(self) -> None:
        """Test that default instructions are used when no SystemMessage provided."""
        client = OpenResponsesClient(
            model="gpt-4o",
            context_window_tokens=64_000,
            api_key="test-key",
            instructions="Default instructions",
        )

        mock_response = MagicMock()
        mock_response.id = "resp_default_instructions"
        mock_response.status = "completed"
        mock_response.output = [
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", text="OK")],
            )
        ]
        mock_response.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            output_tokens_details=None,
        )

        mock_create = AsyncMock(return_value=mock_response)
        client._client.responses.create = mock_create  # type: ignore[method-assign]  # noqa: SLF001

        await client.generate(
            messages=[UserMessage(content="Hello")],
            tools={},
        )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["instructions"] == "Default instructions"

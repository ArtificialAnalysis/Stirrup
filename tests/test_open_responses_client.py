"""Tests for OpenResponsesClient."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from stirrup.clients.open_responses_client import (
    OpenResponsesClient,
    _content_to_open_responses_input,
    _content_to_open_responses_output,
    _parse_response_output,
    _to_open_responses_input,
    _to_open_responses_tools,
)
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

        with pytest.raises(
            NotImplementedError,
            match="Unsupported assistant block type for OpenAI Responses replay: object",
        ):
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

    def test_parse_reasoning_output_with_id(self) -> None:
        """A reasoning item with an id becomes a ReasoningRefBlock — the id is the passback handle."""
        summary_part = MagicMock(spec=["text"])
        summary_part.text = "Thinking..."
        reasoning_item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        reasoning_item.type = "reasoning"
        reasoning_item.id = "rs_123"
        reasoning_item.summary = [summary_part]
        reasoning_item.encrypted_content = None

        empty_reasoning_item = MagicMock(spec=["type", "id", "summary"])
        empty_reasoning_item.type = "reasoning"
        empty_reasoning_item.id = "rs_456"
        empty_reasoning_item.summary = []

        blocks = _parse_response_output([reasoning_item, empty_reasoning_item])

        # id retained even when the summary is empty
        assert blocks == [
            ReasoningRefBlock(id="rs_123", content="Thinking..."),
            ReasoningRefBlock(id="rs_456", content=""),
        ]

    def test_parse_reasoning_output_with_encrypted_content(self) -> None:
        """A reasoning item carrying encrypted_content becomes an EncryptedReasoningBlock,
        with summary parts kept split for verbatim re-emission."""
        part_1 = MagicMock(spec=["text"])
        part_1.text = "First thought."
        part_2 = MagicMock(spec=["text"])
        part_2.text = "Second thought."
        encrypted_item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        encrypted_item.type = "reasoning"
        encrypted_item.id = "rs_789"
        encrypted_item.summary = [part_1, part_2]
        encrypted_item.encrypted_content = "opaque-zdr-payload"

        bare_encrypted_item = MagicMock(spec=["type", "id", "summary", "encrypted_content"])
        bare_encrypted_item.type = "reasoning"
        bare_encrypted_item.id = "rs_790"
        bare_encrypted_item.summary = []
        bare_encrypted_item.encrypted_content = "opaque-zdr-payload-2"

        blocks = _parse_response_output([encrypted_item, bare_encrypted_item])

        assert blocks == [
            EncryptedReasoningBlock(
                id="rs_789", encrypted_content="opaque-zdr-payload", summary=("First thought.", "Second thought.")
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


class TestOpenResponsesClient:
    """Tests for OpenResponsesClient class."""

    def test_client_properties(self) -> None:
        """Test client property accessors."""
        client = OpenResponsesClient(
            model="gpt-4o",
            max_tokens=50000,
            api_key="test-key",
        )
        assert client.model_slug == "gpt-4o"
        assert client.max_tokens == 50000

    @pytest.mark.asyncio
    async def test_generate_basic(self) -> None:
        """Test basic generation with mocked response."""
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
        )

        # Mock the responses.create method
        mock_response = MagicMock()
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
        assert joined_text(result.blocks) == "Hello!"
        assert result.token_usage.input == 10
        assert result.token_usage.answer == 5

    @pytest.mark.asyncio
    async def test_generate_encrypted_reasoning_sends_stateless_params(self) -> None:
        """encrypted_reasoning=True sends store=false + the encrypted-content include."""
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
            encrypted_reasoning=True,
        )

        mock_response = MagicMock()
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

        await client.generate(messages=[UserMessage(content="Hi")], tools={})

        request_kwargs = create_mock.call_args.kwargs
        assert request_kwargs["store"] is False
        assert request_kwargs["include"] == ["reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_generate_with_tools(self) -> None:
        """Test generation with tool calls."""
        from stirrup.core.models import EmptyParams, Tool, ToolResult

        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
        )

        # Mock response with function call
        fn_call = MagicMock()
        fn_call.type = "function_call"
        fn_call.call_id = "call_xyz"
        fn_call.name = "get_time"
        fn_call.arguments = "{}"

        mock_response = MagicMock()
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

    @pytest.mark.asyncio
    async def test_generate_with_reasoning_tokens(self) -> None:
        """Test that reasoning tokens are properly extracted."""
        client = OpenResponsesClient(
            model="o1-preview",
            api_key="test-key",
            reasoning_effort="medium",
        )

        reasoning_item = MagicMock(spec=["type", "summary"])
        reasoning_item.type = "reasoning"
        reasoning_item.summary = "Thinking step by step..."

        mock_response = MagicMock()
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

    @pytest.mark.asyncio
    async def test_generate_incomplete_raises_error(self) -> None:
        """Test that incomplete response raises ContextOverflowError."""
        from stirrup.core.exceptions import ContextOverflowError

        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status = "incomplete"
        mock_response.incomplete_details = "max_output_tokens reached"
        mock_response.output = []
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=0)

        client._client.responses.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]  # noqa: SLF001

        with pytest.raises(ContextOverflowError, match="incomplete"):
            await client.generate(
                messages=[UserMessage(content="Very long request")],
                tools={},
            )

    @pytest.mark.asyncio
    async def test_instructions_from_system_message(self) -> None:
        """Test that SystemMessage is passed as instructions parameter."""
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
        )

        mock_response = MagicMock()
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

    @pytest.mark.asyncio
    async def test_default_instructions_fallback(self) -> None:
        """Test that default instructions are used when no SystemMessage provided."""
        client = OpenResponsesClient(
            model="gpt-4o",
            api_key="test-key",
            instructions="Default instructions",
        )

        mock_response = MagicMock()
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

"""Tests for OpenResponsesClient."""

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
    AssistantMessage,
    SystemMessage,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UserMessage,
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

    def test_assistant_message_conversion(self) -> None:
        """Test AssistantMessage conversion to input format."""
        messages = [
            AssistantMessage(content="I can help with that", tool_calls=[], token_usage=TokenUsage()),
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
                content="Let me search for that",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_123",
                        name="search",
                        arguments='{"query": "test"}',
                    )
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
                content="I'll search for that",
                tool_calls=[ToolCall(tool_call_id="call_1", name="search", arguments='{"q": "Python"}')],
                token_usage=TokenUsage(),
            ),
            ToolMessage(content="Python is a programming language", tool_call_id="call_1", name="search"),
            AssistantMessage(content="Python is a programming language", tool_calls=[], token_usage=TokenUsage()),
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
        content, tool_calls, reasoning = _parse_response_output(output)

        assert content == "Hello there!"
        assert tool_calls == []
        assert reasoning is None

    def test_parse_function_call_output(self) -> None:
        """Test parsing a function call response."""
        fn_call = MagicMock()
        fn_call.type = "function_call"
        fn_call.call_id = "call_abc"
        fn_call.name = "get_weather"
        fn_call.arguments = '{"city": "NYC"}'
        output = [fn_call]
        content, tool_calls, _reasoning = _parse_response_output(output)

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_call_id == "call_abc"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == '{"city": "NYC"}'

    def test_parse_reasoning_output(self) -> None:
        """Test parsing a response with reasoning."""
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
        content, _tool_calls, reasoning = _parse_response_output(output)

        assert content == "The answer is 42"
        assert reasoning is not None
        assert reasoning.content == "Let me think about this..."

    def test_parse_mixed_output(self) -> None:
        """Test parsing response with message and function calls."""
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
            fn_call_2,
        ]
        content, tool_calls, _reasoning = _parse_response_output(output)

        assert content == "I'll help you with that"
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "tool1"
        assert tool_calls[1].name == "tool2"


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
        assert result.content == "Hello!"
        assert result.token_usage.input == 10
        assert result.token_usage.output == 5

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

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_time"
        assert result.tool_calls[0].tool_call_id == "call_xyz"

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

        assert result.reasoning is not None
        assert result.reasoning.content == "Thinking step by step..."
        assert result.token_usage.reasoning == 80
        assert result.token_usage.output == 20  # 100 - 80

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

"""Tests for ChatCompletionsClient response parsing and token-budget behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import BadRequestError
from tenacity import wait_none

from stirrup import ContextOverflowError, EmptyResponseError, OutputTokenLimitError
from stirrup.clients import chat_completions_client as chat_completions_module
from stirrup.clients.chat_completions_client import ChatCompletionsClient
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
    client = ChatCompletionsClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)
    client._client.chat.completions.create = AsyncMock(return_value=response)  # type: ignore[method-assign]  # noqa: SLF001
    return client


def _bad_request_error(code: str) -> BadRequestError:
    response = httpx.Response(
        400,
        json={"error": {"message": "boom", "type": "invalid_request_error", "code": code}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return BadRequestError("boom", response=response, body=response.json()["error"])


def _client_with_provider_call(monkeypatch: pytest.MonkeyPatch, provider_call: AsyncMock) -> ChatCompletionsClient:
    provider_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider_call)))
    monkeypatch.setattr(chat_completions_module, "AsyncOpenAI", lambda **_kwargs: provider_client)
    return ChatCompletionsClient(model="gpt-4o", api_key="test-key", context_window_tokens=64_000)


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

        assert result.blocks == [
            ReasoningBlock(content="Thinking..."),
            TextBlock(text="The answer"),
            ToolCall(tool_call_id="call_1", name="search", arguments='{"q": "x"}'),
        ]

    async def test_tool_calls_only_turn_has_no_text_block(self) -> None:
        response = _mock_response(content=None, tool_calls=[_mock_tool_call("call_2", "lookup", "{}")])
        client = _client_with_response(response)

        result = await client.generate(messages=[UserMessage(content="Hi")], tools={})

        assert result.blocks == [ToolCall(tool_call_id="call_2", name="lookup", arguments="{}")]

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

        assert result.blocks == [TextBlock(text="plain answer")]

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

    async def test_length_finish_reason_raises_output_token_limit(self) -> None:
        response = _mock_response(content="truncat", finish_reason="length")
        client = _client_with_response(response)

        with pytest.raises(OutputTokenLimitError):
            await client.generate(messages=[UserMessage(content="Hi")], tools={})


@pytest.mark.parametrize("context_window_tokens", [0, -1])
def test_context_window_must_be_positive(context_window_tokens: int) -> None:
    with pytest.raises(ValueError, match="context_window_tokens must be positive"):
        ChatCompletionsClient(
            model="gpt-4o",
            api_key="test-key",
            context_window_tokens=context_window_tokens,
        )


def test_max_tokens_must_fit_context_window() -> None:
    with pytest.raises(ValueError, match="must not exceed context_window_tokens"):
        ChatCompletionsClient(
            model="gpt-4o",
            api_key="test-key",
            max_tokens=128_000,
            context_window_tokens=8_192,
        )


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_output_limit_is_forwarded_and_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    provider_call = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason)],
            usage=None,
        )
    )
    provider_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider_call)))
    monkeypatch.setattr(chat_completions_module, "AsyncOpenAI", lambda **_kwargs: provider_client)
    client = ChatCompletionsClient(
        model="gpt-4o",
        api_key="test-key",
        max_tokens=321,
        context_window_tokens=64_000,
    )

    with pytest.raises(OutputTokenLimitError) as exc_info:
        await client.generate([UserMessage(content="hello")], {})

    assert exc_info.value.model_slug == "gpt-4o"
    assert exc_info.value.max_tokens == 321
    assert exc_info.value.provider_reason == finish_reason
    provider_call.assert_awaited_once()
    provider_request = provider_call.await_args
    assert provider_request is not None
    assert provider_request.kwargs["max_completion_tokens"] == 321


async def test_context_length_rejection_surfaces_as_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_provider_call(
        monkeypatch, AsyncMock(side_effect=_bad_request_error("context_length_exceeded"))
    )

    with pytest.raises(ContextOverflowError):
        await client.generate([UserMessage(content="hello")], {})


async def test_unrelated_bad_request_is_not_mapped_to_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_provider_call(monkeypatch, AsyncMock(side_effect=_bad_request_error("invalid_value")))

    with pytest.raises(BadRequestError):
        await client.generate([UserMessage(content="hello")], {})


def _reply_without_choices(choices: list[object] | None = None, *, error: object = None) -> SimpleNamespace:
    error = error if error is not None else {"message": "Provider returned error", "code": 502}
    return SimpleNamespace(choices=choices, usage=None, model_extra={"error": error})


@pytest.fixture
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_completions_module, "_EMPTY_RESPONSE_WAIT", wait_none())


def _client_with_retries(
    monkeypatch: pytest.MonkeyPatch, provider_call: AsyncMock, max_retries: int
) -> ChatCompletionsClient:
    provider_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=provider_call)))
    monkeypatch.setattr(chat_completions_module, "AsyncOpenAI", lambda **_kwargs: provider_client)
    return ChatCompletionsClient(
        model="gpt-4o", api_key="test-key", context_window_tokens=64_000, max_retries=max_retries
    )


@pytest.mark.usefixtures("no_retry_wait")
class TestRepliesWithoutChoices:
    """Gateways such as OpenRouter can answer HTTP 200 with an error and no choices."""

    @pytest.mark.parametrize("choices", [None, []])
    async def test_reply_without_choices_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, choices: list[object] | None
    ) -> None:
        provider_call = AsyncMock(side_effect=[_reply_without_choices(choices), _mock_response(content="recovered")])
        client = _client_with_provider_call(monkeypatch, provider_call)

        result = await client.generate([UserMessage(content="hello")], {})

        assert result.blocks == [TextBlock(text="recovered")]
        assert result.token_usage.input == 10
        assert provider_call.await_count == 2

    @pytest.mark.parametrize("choices", [None, []])
    async def test_every_reply_without_choices_raises_with_provider_error(
        self, monkeypatch: pytest.MonkeyPatch, choices: list[object] | None
    ) -> None:
        provider_call = AsyncMock(return_value=_reply_without_choices(choices))
        client = _client_with_retries(monkeypatch, provider_call, max_retries=3)

        with pytest.raises(EmptyResponseError, match="Provider returned error") as exc_info:
            await client.generate([UserMessage(content="hello")], {})

        assert provider_call.await_count == 4
        assert exc_info.value.retryable is True

    @pytest.mark.parametrize("code", [400, 401, 404])
    async def test_client_error_code_is_not_retried(self, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
        provider_call = AsyncMock(return_value=_reply_without_choices(error={"message": "Bad input", "code": code}))
        client = _client_with_retries(monkeypatch, provider_call, max_retries=3)

        with pytest.raises(EmptyResponseError, match="Bad input") as exc_info:
            await client.generate([UserMessage(content="hello")], {})

        assert not exc_info.value.retryable
        provider_call.assert_awaited_once()

    @pytest.mark.parametrize("code", [408, 429, 502, "server_error"])
    async def test_transient_error_code_is_retried(self, monkeypatch: pytest.MonkeyPatch, code: int | str) -> None:
        reply = _reply_without_choices(error={"message": "Try again", "code": code})
        provider_call = AsyncMock(side_effect=[reply, _mock_response(content="recovered")])
        client = _client_with_retries(monkeypatch, provider_call, max_retries=3)

        result = await client.generate([UserMessage(content="hello")], {})

        assert result.blocks == [TextBlock(text="recovered")]
        assert provider_call.await_count == 2

    async def test_long_provider_error_is_truncated_in_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = _reply_without_choices(error={"message": "x" * 5_000, "code": 502})
        client = _client_with_retries(monkeypatch, AsyncMock(return_value=reply), max_retries=0)

        with pytest.raises(EmptyResponseError) as exc_info:
            await client.generate([UserMessage(content="hello")], {})

        message = str(exc_info.value)
        assert message.endswith("... [truncated]")
        assert len(message) < 600

    async def test_zero_max_retries_makes_one_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider_call = AsyncMock(return_value=_reply_without_choices())
        client = _client_with_retries(monkeypatch, provider_call, max_retries=0)

        with pytest.raises(EmptyResponseError):
            await client.generate([UserMessage(content="hello")], {})

        provider_call.assert_awaited_once()

    async def test_reply_without_choices_or_error_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = SimpleNamespace(choices=None, usage=None, model_extra={})
        client = _client_with_retries(monkeypatch, AsyncMock(return_value=reply), max_retries=0)

        with pytest.raises(EmptyResponseError, match="returned no choices"):
            await client.generate([UserMessage(content="hello")], {})

    async def test_reply_with_choices_makes_one_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider_call = AsyncMock(return_value=_mock_response(content="first time"))
        client = _client_with_provider_call(monkeypatch, provider_call)

        result = await client.generate([UserMessage(content="hello")], {})

        assert result.blocks == [TextBlock(text="first time")]
        provider_call.assert_awaited_once()

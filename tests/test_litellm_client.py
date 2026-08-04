"""Tests for LiteLLM client parsing and token-budget behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("litellm")

from litellm.exceptions import ContextWindowExceededError

from stirrup.clients import litellm_client as litellm_module
from stirrup.clients.litellm_client import LiteLLMClient, _parse_thinking_blocks
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantMessage,
    ReasoningBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    ToolCall,
    UserMessage,
)


class TestParseThinkingBlocks:
    """Tests for _parse_thinking_blocks: LiteLLM thinking_blocks → reasoning blocks."""

    def test_none_and_empty_input(self) -> None:
        assert _parse_thinking_blocks(None) == []
        assert _parse_thinking_blocks([]) == []

    def test_signed_thinking_block(self) -> None:
        blocks = _parse_thinking_blocks([{"type": "thinking", "thinking": "Let me think.", "signature": "sig-1"}])

        assert blocks == [SignedReasoningBlock(signature="sig-1", content="Let me think.")]

    def test_signature_via_thinking_signature_key(self) -> None:
        blocks = _parse_thinking_blocks([{"thinking": "hmm", "thinking_signature": "sig-2"}])

        assert blocks == [SignedReasoningBlock(signature="sig-2", content="hmm")]

    def test_signature_key_takes_precedence(self) -> None:
        blocks = _parse_thinking_blocks([{"thinking": "hmm", "signature": "primary", "thinking_signature": "other"}])

        assert blocks == [SignedReasoningBlock(signature="primary", content="hmm")]

    def test_signature_without_content(self) -> None:
        blocks = _parse_thinking_blocks([{"signature": "sig-3"}])

        assert blocks == [SignedReasoningBlock(signature="sig-3", content="")]

    def test_redacted_thinking_block(self) -> None:
        blocks = _parse_thinking_blocks([{"type": "redacted_thinking", "data": "opaque-blob"}])

        assert blocks == [RedactedReasoningBlock(data="opaque-blob")]

    def test_unsigned_thinking_block(self) -> None:
        blocks = _parse_thinking_blocks([{"type": "thinking", "thinking": "plain reasoning"}])

        assert blocks == [ReasoningBlock(content="plain reasoning")]

    def test_multiple_blocks_preserved_in_order(self) -> None:
        blocks = _parse_thinking_blocks(
            [
                {"type": "thinking", "thinking": "first", "signature": "sig-a"},
                {"type": "redacted_thinking", "data": "blob"},
                {"type": "thinking", "thinking": "second", "signature": "sig-b"},
            ]
        )

        assert blocks == [
            SignedReasoningBlock(signature="sig-a", content="first"),
            RedactedReasoningBlock(data="blob"),
            SignedReasoningBlock(signature="sig-b", content="second"),
        ]

    def test_entry_without_signature_or_content_raises(self) -> None:
        with pytest.raises(ValueError, match="Signature and content not found"):
            _parse_thinking_blocks([{"type": "thinking"}])


async def test_idless_litellm_tool_call_gets_internal_id(monkeypatch: pytest.MonkeyPatch) -> None:
    message = {
        "content": None,
        "tool_calls": [{"id": None, "function": {"name": "lookup", "arguments": "{}"}}],
    }
    choice = MagicMock(finish_reason="stop")
    choice.__getitem__.side_effect = {"message": message}.__getitem__
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    completion = AsyncMock(return_value={"choices": [choice], "usage": usage})
    monkeypatch.setattr("stirrup.clients.litellm_client.acompletion", completion)

    result = await LiteLLMClient(model="test", context_window_tokens=76_543).generate(
        [UserMessage(content="prompt")], {}
    )

    [tool_call] = result.blocks
    assert isinstance(tool_call, ToolCall)
    assert tool_call.tool_call_id
    assert not tool_call.has_provider_tool_call_id


async def test_litellm_passes_back_tool_call_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    message = {"content": None, "tool_calls": []}
    choice = MagicMock(finish_reason="stop")
    choice.__getitem__.side_effect = {"message": message}.__getitem__
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    completion = AsyncMock(return_value={"choices": [choice], "usage": usage})
    monkeypatch.setattr("stirrup.clients.litellm_client.acompletion", completion)
    history = AssistantMessage(
        blocks=[ToolCall(tool_call_id="call_1", name="lookup", arguments="{}", signature="google-signature")]
    )

    await LiteLLMClient(model="test", context_window_tokens=76_543).generate([history], {})

    [wire_call] = completion.call_args.kwargs["messages"][0]["tool_calls"]
    assert wire_call["provider_specific_fields"] == {"thought_signature": "google-signature"}


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_output_limit_is_forwarded_and_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    provider_call = AsyncMock(return_value={"choices": [SimpleNamespace(finish_reason=finish_reason)]})
    monkeypatch.setattr(litellm_module, "acompletion", provider_call)
    client = LiteLLMClient(
        model="test/provider-model",
        max_tokens=789,
        context_window_tokens=76_543,
    )

    with pytest.raises(OutputTokenLimitError) as exc_info:
        await client.generate([], {})

    assert exc_info.value.model_slug == "test/provider-model"
    assert exc_info.value.max_tokens == 789
    assert exc_info.value.provider_reason == finish_reason
    provider_call.assert_awaited_once()
    provider_request = provider_call.await_args
    assert provider_request is not None
    assert provider_request.kwargs["max_tokens"] == 789


async def test_context_window_rejection_surfaces_as_context_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_call = AsyncMock(
        side_effect=ContextWindowExceededError(
            message="boom",
            model="test/provider-model",
            llm_provider="test",
        )
    )
    monkeypatch.setattr(litellm_module, "acompletion", provider_call)
    client = LiteLLMClient(model="test/provider-model", max_tokens=8_192, context_window_tokens=76_543)

    with pytest.raises(ContextOverflowError):
        await client.generate([], {})

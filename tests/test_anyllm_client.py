"""Tests for any-llm client parsing and token-budget behavior."""

import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("any_llm")

from anthropic import DefaultAsyncHttpxClient
from any_llm.exceptions import ContextLengthExceededError

from stirrup.clients import anyllm_client as anyllm_module
from stirrup.clients.anyllm_client import AnyLLMClient, _parse_thinking_blocks
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantMessage,
    ReasoningBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class TestParseThinkingBlocks:
    """Tests for _parse_thinking_blocks: any-llm thinking_blocks → reasoning blocks."""

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


async def test_idless_anyllm_tool_call_gets_internal_id(monkeypatch: pytest.MonkeyPatch) -> None:
    message = {
        "content": None,
        "tool_calls": [
            SimpleNamespace(type="function", id=None, function=SimpleNamespace(name="lookup", arguments="{}"))
        ],
    }
    choice = MagicMock(finish_reason="stop")
    choice.message = SimpleNamespace(**message, extra_content=None, reasoning=None)
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    completion = AsyncMock(return_value=SimpleNamespace(choices=[choice], usage=usage))
    monkeypatch.setattr(
        anyllm_module.AnyLLM, "create", lambda *_args, **_kwargs: SimpleNamespace(acompletion=completion)
    )

    result = await AnyLLMClient(model="test", context_window_tokens=76_543).generate(
        [UserMessage(content="prompt")], {}
    )

    [tool_call] = result.blocks
    assert isinstance(tool_call, ToolCall)
    assert tool_call.tool_call_id
    assert not tool_call.has_provider_tool_call_id


async def test_anyllm_passes_back_tool_call_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    message = {"content": None, "tool_calls": []}
    choice = MagicMock(finish_reason="stop")
    choice.message = SimpleNamespace(**message, extra_content=None, reasoning=None)
    usage = MagicMock(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    completion = AsyncMock(return_value=SimpleNamespace(choices=[choice], usage=usage))
    monkeypatch.setattr(
        anyllm_module.AnyLLM, "create", lambda *_args, **_kwargs: SimpleNamespace(acompletion=completion)
    )
    history = AssistantMessage(
        blocks=[ToolCall(tool_call_id="call_1", name="lookup", arguments="{}", signature="google-signature")]
    )

    await AnyLLMClient(model="test", context_window_tokens=76_543).generate([history], {})

    [wire_call] = completion.call_args.kwargs["messages"][0]["tool_calls"]
    assert wire_call["extra_content"] == {"google": {"thought_signature": "google-signature"}}


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_output_limit_is_forwarded_and_reported_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    provider_call = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason)]))
    monkeypatch.setattr(
        anyllm_module.AnyLLM, "create", lambda *_args, **_kwargs: SimpleNamespace(acompletion=provider_call)
    )
    client = AnyLLMClient(
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
        side_effect=ContextLengthExceededError(
            message="boom",
            provider_name="test",
        )
    )
    monkeypatch.setattr(
        anyllm_module.AnyLLM, "create", lambda *_args, **_kwargs: SimpleNamespace(acompletion=provider_call)
    )
    client = AnyLLMClient(model="test/provider-model", max_tokens=8_192, context_window_tokens=76_543)

    with pytest.raises(ContextOverflowError):
        await client.generate([], {})


async def test_anthropic_thinking_blocks_survive_tool_round_trip() -> None:
    transport = httpx if issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient) else import_module("httpx2")
    blocks = [
        {"type": "thinking", "thinking": "first", "signature": "sig-a"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "thinking": "second", "signature": "sig-b"},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
    ]
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return transport.Response(
            200,
            json={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": blocks,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    async with transport.AsyncClient(transport=transport.MockTransport(respond)) as http_client:
        client = AnyLLMClient(
            model="anthropic:claude-sonnet-4-6",
            api_key="test-key",
            context_window_tokens=128_000,
            kwargs={"client_args": {"http_client": http_client}},
        )
        prompt = UserMessage(content="Look it up")
        result = await client.generate([prompt], {})
        assert result.blocks == [
            SignedReasoningBlock(content="first", signature="sig-a"),
            RedactedReasoningBlock(data="opaque"),
            SignedReasoningBlock(content="second", signature="sig-b"),
            ToolCall(tool_call_id="toolu_1", name="lookup", arguments="{}"),
        ]
        await client.generate([prompt, result, ToolMessage(tool_call_id="toolu_1", content="42")], {})

    assert requests[1]["messages"][1]["content"] == blocks
    assert requests[1]["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"


async def test_gemini_tool_signature_survives_tool_round_trip() -> None:
    pytest.importorskip("google.genai")
    signature = "Z2VtaW5pLXNpZw=="
    function_call = {"name": "lookup", "args": {}}
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": function_call, "thoughtSignature": signature},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = AnyLLMClient(
            model="gemini:gemini-3-flash-preview",
            api_key="test-key",
            context_window_tokens=128_000,
            kwargs={"client_args": {"http_options": {"httpx_async_client": http_client}}},
        )
        prompt = UserMessage(content="Look it up")
        result = await client.generate([prompt], {})
        [tool_call] = result.blocks
        assert isinstance(tool_call, ToolCall)
        assert (tool_call.name, tool_call.arguments, tool_call.signature) == ("lookup", "{}", signature)
        await client.generate(
            [
                prompt,
                result,
                ToolMessage(tool_call_id=tool_call.tool_call_id, name="lookup", content="42"),
            ],
            {},
        )

    [part] = requests[1]["contents"][1]["parts"]
    assert part["functionCall"] == function_call
    assert part["thoughtSignature"] == signature
    assert requests[1]["contents"][2]["parts"][0]["functionResponse"]["name"] == "lookup"

"""Tests for block-based assistant messages (SP-001).

Covers: block accessors, channel projections, channel-order synthesis, the
mixing guard, serialization round trips (v0.1 goldens and v0.2 blocks),
wire-format goldens for the OpenAI-compatible clients, and the §8 integration
contract (id stability, metadata opacity, subclass preservation).
"""

from collections.abc import Sequence
from io import BytesIO
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from stirrup.clients.open_responses_client import _to_open_responses_input
from stirrup.clients.utils import to_openai_messages
from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    ChatMessage,
    EncryptedReasoningBlock,
    ImageContentBlock,
    LLMClient,
    OpaqueBlock,
    Reasoning,
    ReasoningBlock,
    ReasoningRefBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    SubAgentMetadata,
    SummaryMessage,
    SystemMessage,
    TextBlock,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    TurnWarningMessage,
    UserMessage,
    final_text,
    joined_text,
    reasoning_blocks,
    tool_call_blocks,
)
from stirrup.tools.finish import SIMPLE_FINISH_TOOL


def _png_block() -> ImageContentBlock:
    img = Image.new("RGB", (1, 1), color=(0, 128, 255))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return ImageContentBlock(data=buffer.getvalue())


INTERLEAVED_BLOCKS: list[AssistantBlock] = [
    SignedReasoningBlock(signature="sig-1", content="first thought"),
    TextBlock(text="Let me look."),
    SignedReasoningBlock(signature="sig-2", content="second thought"),
    ToolCall(name="grep", arguments='{"pattern": "x"}', tool_call_id="call-1"),
    TextBlock(text="Found it."),
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_joined_text_concatenates_exact_text_deltas() -> None:
    assert joined_text(INTERLEAVED_BLOCKS) == "Let me look.Found it."
    assert joined_text([ToolCall(name="t", arguments="{}")]) is None
    assert joined_text([]) is None


def test_final_text_returns_last_text_block() -> None:
    assert final_text(INTERLEAVED_BLOCKS) == "Found it."
    assert final_text([SignedReasoningBlock(signature="s")]) is None


def test_tool_call_and_reasoning_accessors_preserve_order() -> None:
    assert [tc.tool_call_id for tc in tool_call_blocks(INTERLEAVED_BLOCKS)] == ["call-1"]
    assert [type(b) for b in reasoning_blocks(INTERLEAVED_BLOCKS)] == [SignedReasoningBlock, SignedReasoningBlock]


# ---------------------------------------------------------------------------
# Channel projections (deprecated, read-only, derived from blocks)
# ---------------------------------------------------------------------------


def test_content_projection_joins_text_blocks() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    with pytest.warns(DeprecationWarning, match="AssistantMessage.content is deprecated"):
        assert msg.content == "Let me look.Found it."


def test_content_projection_empty_string_when_no_text_blocks() -> None:
    msg = AssistantMessage(blocks=[ToolCall(name="t", arguments="{}")])
    with pytest.warns(DeprecationWarning, match="AssistantMessage.content is deprecated"):
        assert msg.content == ""


def test_content_projection_media_list_in_block_order() -> None:
    image = _png_block()
    msg = AssistantMessage(blocks=[TextBlock(text="see:"), image, TextBlock(text="above")])
    with pytest.warns(DeprecationWarning, match="AssistantMessage.content is deprecated"):
        assert msg.content == ["see:", image, "above"]


def test_reasoning_projection_rejects_multiple_signed_blocks() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    with (
        pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"),
        pytest.raises(NotImplementedError, match="cannot represent multiple signed blocks"),
    ):
        _ = msg.reasoning


def test_reasoning_projection_rejects_signed_and_unsigned_mix() -> None:
    msg = AssistantMessage(
        blocks=[SignedReasoningBlock(signature="sig", content="signed"), ReasoningBlock(content="unsigned")]
    )
    with (
        pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"),
        pytest.raises(NotImplementedError, match="signed reasoning mixed"),
    ):
        _ = msg.reasoning


def test_reasoning_projection_concatenates_unsigned_blocks() -> None:
    msg = AssistantMessage(blocks=[ReasoningBlock(content="one"), ReasoningBlock(content="two")])
    with pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"):
        assert msg.reasoning == Reasoning(content="onetwo")


def test_reasoning_projection_none_without_reasoning_blocks() -> None:
    with pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"):
        assert AssistantMessage(blocks=[TextBlock(text="hi")]).reasoning is None


def test_reasoning_projection_redacted_blocks_contribute_nothing() -> None:
    msg = AssistantMessage(
        blocks=[
            RedactedReasoningBlock(data="opaque-payload"),
            SignedReasoningBlock(signature="sig", content="visible"),
        ]
    )
    with pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"):
        reasoning = msg.reasoning
    assert reasoning is not None
    assert reasoning.content == "visible"
    assert reasoning.signature == "sig"

    with pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"):
        assert AssistantMessage(blocks=[RedactedReasoningBlock(data="opaque")]).reasoning is None


def test_tool_calls_projection() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    with pytest.warns(DeprecationWarning, match="AssistantMessage.tool_calls is deprecated"):
        assert [tc.name for tc in msg.tool_calls] == ["grep"]
    with pytest.warns(DeprecationWarning, match="AssistantMessage.tool_calls is deprecated"):
        assert AssistantMessage(blocks=[TextBlock(text="x")]).tool_calls == []


# ---------------------------------------------------------------------------
# Legacy v0.1 reader conversion
# ---------------------------------------------------------------------------


def test_legacy_channel_payload_synthesizes_blocks_in_channel_order() -> None:
    msg = AssistantMessage.model_validate(
        {
            "content": "answer",
            "reasoning": {"content": "thinking"},
            "tool_calls": [{"name": "t", "arguments": "{}", "tool_call_id": "c1"}],
        }
    )
    assert [b.kind for b in msg.blocks] == ["reasoning", "text", "tool_call"]


def test_flat_reasoning_splits_by_signature_presence() -> None:
    signed = AssistantMessage.model_validate({"content": "", "reasoning": {"signature": "sig", "content": "deep"}})
    assert signed.blocks == [SignedReasoningBlock(signature="sig", content="deep")]

    in_band = AssistantMessage.model_validate({"content": "", "reasoning": {"content": "deep"}})
    assert in_band.blocks == [ReasoningBlock(content="deep")]


def test_media_content_list_passes_through_in_place() -> None:
    image = _png_block()
    msg = AssistantMessage.model_validate({"content": ["before", image, "after"]})
    assert msg.blocks == [TextBlock(text="before"), image, TextBlock(text="after")]


def test_empty_content_synthesizes_no_text_block() -> None:
    msg = AssistantMessage.model_validate(
        {"content": "", "tool_calls": [{"name": "t", "arguments": "{}", "tool_call_id": None}]}
    )
    assert [b.kind for b in msg.blocks] == ["tool_call"]
    with pytest.warns(DeprecationWarning, match="AssistantMessage.content is deprecated"):
        assert msg.content == ""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_mixing_blocks_with_channel_fields_raises() -> None:
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage.model_validate({"blocks": [{"kind": "text", "text": "x"}], "content": "y"})
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage.model_validate({"blocks": [], "tool_calls": [{"name": "t", "arguments": "{}"}]})
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage.model_validate({"blocks": [], "reasoning": {"content": "r"}})
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage.model_validate({"blocks": [], "content": {}})


def test_mixing_guard_ignores_empty_channel_values() -> None:
    # Empty values ("", [], {}) drop nothing — they don't conflict.
    msg = AssistantMessage.model_validate({"blocks": [{"kind": "text", "text": "x"}], "content": "", "tool_calls": []})
    assert msg.blocks == [TextBlock(text="x")]
    msg = AssistantMessage.model_validate({"blocks": [{"kind": "text", "text": "x"}], "content": [], "reasoning": {}})
    assert msg.blocks == [TextBlock(text="x")]


def test_falsy_channel_reasoning_synthesizes_no_block() -> None:
    # v0.1 required Reasoning.content, so an empty dict was never a valid payload;
    # it must not turn into a spurious empty ReasoningBlock.
    msg = AssistantMessage.model_validate({"content": "hi", "reasoning": {}})
    assert msg.blocks == [TextBlock(text="hi")]


def test_malformed_channel_content_fails_validation() -> None:
    # A non-str, non-list v0.1 content payload must fail loudly, not vanish.
    with pytest.raises(ValidationError, match="must be a string or list"):
        AssistantMessage.model_validate({"role": "assistant", "content": {"oops": 1}})


# ---------------------------------------------------------------------------
# Opaque provider blocks
# ---------------------------------------------------------------------------


OPAQUE_PAYLOAD = '{"type": "provider_marker", "detail": {"reason": "switch"}}'


def test_opaque_block_round_trips_and_stays_out_of_projections() -> None:
    msg = AssistantMessage(
        blocks=[
            SignedReasoningBlock(signature="sig-1", content="thinking"),
            OpaqueBlock(data=OPAQUE_PAYLOAD),
            TextBlock(text="answer"),
        ]
    )
    reloaded = AssistantMessage.model_validate(msg.model_dump())
    assert reloaded.blocks == msg.blocks
    # Not reasoning: excluded from the reasoning family and the channel projections.
    assert reasoning_blocks(msg.blocks) == [SignedReasoningBlock(signature="sig-1", content="thinking")]
    assert joined_text(msg.blocks) == "answer"
    assert tool_call_blocks(msg.blocks) == []


def test_opaque_block_rejected_on_openai_replay() -> None:
    msg = AssistantMessage(
        blocks=[
            OpaqueBlock(data=OPAQUE_PAYLOAD),
            TextBlock(text="answer"),
        ]
    )
    with pytest.raises(NotImplementedError, match="OpaqueBlock"):
        to_openai_messages([msg])


def test_opaque_block_rejected_on_responses_replay() -> None:
    msg = AssistantMessage(
        blocks=[
            OpaqueBlock(data=OPAQUE_PAYLOAD),
            TextBlock(text="answer"),
        ]
    )
    with pytest.raises(NotImplementedError, match="OpaqueBlock"):
        _to_open_responses_input([msg])


# ---------------------------------------------------------------------------
# Explicit block replacement
# ---------------------------------------------------------------------------


def test_text_block_signature_round_trips() -> None:
    block = TextBlock(text="provider-bound", signature="sig-1")
    message = AssistantMessage.model_validate(AssistantMessage(blocks=[block]).model_dump(mode="json"))

    assert message.blocks == [block]


def test_block_models_are_frozen_but_containers_are_lists() -> None:
    encrypted = EncryptedReasoningBlock(id="rs_1", encrypted_content="opaque", summary=["one"])
    message = AssistantMessage(blocks=[encrypted])

    assert isinstance(message.blocks, list)
    assert encrypted.model_config.get("frozen") is True
    assert encrypted.summary == ["one"]


def test_with_blocks_replaces_block_list() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    stripped = msg.with_blocks([b for b in msg.blocks if b.kind != "tool_call"])
    assert tool_call_blocks(stripped.blocks) == []
    assert stripped.id == msg.id


def test_mutators_clear_provider_response_id() -> None:
    """The continuation handle is bound to the exact emitted content; an edited copy must replay in full."""
    msg = AssistantMessage(provider_response_id="resp_1", blocks=[TextBlock(text="original")])

    assert msg.with_blocks([TextBlock(text="other")]).provider_response_id is None
    assert msg.provider_response_id == "resp_1"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

V01_DUMP: dict[str, Any] = {
    "id": "abc123",
    "role": "assistant",
    "reasoning": {"signature": "sig", "content": "thought"},
    "content": "hello",
    "tool_calls": [{"signature": None, "name": "t", "arguments": "{}", "tool_call_id": "c1"}],
    "token_usage": {"input": 10, "answer": 5, "reasoning": 2},
    "metadata": {"user_key": 1},
    "request_start_time": 1.0,
    "request_end_time": 2.0,
}


def test_v01_golden_dump_upgrades_to_blocks() -> None:
    msg = AssistantMessage.model_validate(V01_DUMP)
    assert msg.id == "abc123"
    assert msg.blocks == [
        SignedReasoningBlock(signature="sig", content="thought"),
        TextBlock(text="hello"),
        ToolCall(name="t", arguments="{}", tool_call_id="c1"),
    ]
    # The block helpers expose the same information without the deprecated projections.
    assert joined_text(msg.blocks) == "hello"
    assert reasoning_blocks(msg.blocks) == [SignedReasoningBlock(signature="sig", content="thought")]
    assert msg.metadata == {"user_key": 1}


def test_v01_dump_nested_in_subagent_metadata_upgrades() -> None:
    nested = SubAgentMetadata.model_validate(
        {"message_history": [[V01_DUMP]], "run_metadata": {}},
    )
    inner = nested.message_history[0][0]
    assert isinstance(inner, AssistantMessage)
    assert [b.kind for b in inner.blocks] == ["signed_reasoning", "text", "tool_call"]


def test_v02_dump_emits_blocks_only() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    dump = msg.model_dump(mode="json")
    assert "blocks" in dump
    assert "content" not in dump
    assert "reasoning" not in dump
    assert "tool_calls" not in dump
    # every block carries its kind discriminator
    assert [b["kind"] for b in dump["blocks"]] == [
        "signed_reasoning",
        "text",
        "signed_reasoning",
        "tool_call",
        "text",
    ]


def test_tool_call_provider_id_provenance_round_trips() -> None:
    call = ToolCall(
        name="t",
        arguments="{}",
        tool_call_id="internal-call-id",
        has_provider_tool_call_id=False,
    )
    restored = AssistantMessage.model_validate_json(AssistantMessage(blocks=[call]).model_dump_json())

    assert restored.blocks == [call]


def test_tool_call_normalizes_one_internal_id_and_preserves_provenance() -> None:
    idless = ToolCall.from_provider(provider_id=None, name="t", arguments="{}")
    native = ToolCall.from_provider(provider_id="provider-call", name="t", arguments="{}")

    assert idless.tool_call_id
    assert not idless.has_provider_tool_call_id
    assert native.tool_call_id == "provider-call"
    assert native.has_provider_tool_call_id
    assert ToolCall.model_validate_json(idless.model_dump_json()) == idless


def test_legacy_none_tool_call_id_normalizes_but_tool_result_requires_correlation() -> None:
    call = ToolCall.model_validate({"name": "t", "arguments": "{}", "tool_call_id": None})

    assert call.tool_call_id
    assert not call.has_provider_tool_call_id
    with pytest.raises(ValidationError):
        ToolMessage.model_validate({"content": "result"})
    with pytest.raises(ValidationError):
        ToolMessage(content="result", tool_call_id="")


@pytest.mark.parametrize(
    "block",
    [
        ReasoningBlock(content="in-band"),
        SignedReasoningBlock(signature="sig", content="signed"),
        RedactedReasoningBlock(data="opaque"),
        ReasoningRefBlock(id="rs_1", content="summary"),
        ReasoningRefBlock(id="rs_2"),
        EncryptedReasoningBlock(id="rs_3", encrypted_content="zdr-payload", summary=["part one", "part two"]),
    ],
)
def test_each_reasoning_kind_round_trips(block: ReasoningBlock) -> None:
    msg = AssistantMessage(blocks=[block, TextBlock(text="t")])
    restored = AssistantMessage.model_validate(msg.model_dump(mode="json"))
    assert restored.blocks == msg.blocks


def test_interleaved_round_trip_is_lossless_and_id_stable() -> None:
    msg = AssistantMessage(
        provider_response_id="resp_123",
        blocks=INTERLEAVED_BLOCKS,
        token_usage=TokenUsage(input=1, answer=2, reasoning=3),
    )
    restored = AssistantMessage.model_validate(msg.model_dump(mode="json"))
    assert restored.blocks == msg.blocks
    assert restored.id == msg.id
    assert restored.provider_response_id == "resp_123"
    assert restored.token_usage == msg.token_usage


def test_unknown_block_kind_fails_loudly() -> None:
    with pytest.raises(ValidationError):
        AssistantMessage.model_validate({"role": "assistant", "blocks": [{"kind": "hologram", "text": "?"}]})


def test_summary_message_replaced_ids_round_trips() -> None:
    summary = SummaryMessage(content="bridge", replaced_ids=["a", "b"])
    restored = SummaryMessage.model_validate(summary.model_dump(mode="json"))
    assert restored.replaced_ids == ["a", "b"]
    # v0.1 summary dumps (no replaced_ids) still validate
    legacy = SummaryMessage.model_validate({"role": "user", "kind": "summary", "content": "bridge"})
    assert legacy.replaced_ids == []


def test_agent_injected_user_messages_round_trip_through_chat_message_union() -> None:
    """SummaryMessage/TurnWarningMessage rehydrate as their own types (not base
    UserMessage) through the ChatMessage union — dumped histories keep summary
    lineage via replaced_ids."""
    meta = SubAgentMetadata(
        message_history=[
            [
                UserMessage(content="task"),
                AssistantMessage(id="turn-1", blocks=[TextBlock(text="working")]),
                SummaryMessage(content="bridge", replaced_ids=["turn-1"]),
                TurnWarningMessage(content="2 turns remaining"),
            ]
        ],
        run_metadata={},
    )
    reloaded = SubAgentMetadata.model_validate_json(meta.model_dump_json())
    _user, _assistant, summary, warning = reloaded.message_history[0]
    assert isinstance(summary, SummaryMessage)
    assert summary.replaced_ids == ["turn-1"]
    assert isinstance(warning, TurnWarningMessage)


# ---------------------------------------------------------------------------
# Wire-format goldens: chat-completions-shaped replay
# ---------------------------------------------------------------------------


def test_chat_wire_from_legacy_payload_uses_strict_api_shape() -> None:
    """A v0.1-shaped payload still produces a valid OpenAI-compatible request."""
    msg = AssistantMessage.model_validate(
        {
            "content": "hi",
            "reasoning": {"content": "think"},
            "tool_calls": [{"name": "t", "arguments": "{}", "tool_call_id": "c1"}],
        }
    )
    assert to_openai_messages([msg]) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        }
    ]


def test_chat_wire_single_signed_turn_matches_v01_shape() -> None:
    msg = AssistantMessage.model_validate({"content": "hi", "reasoning": {"signature": "sig", "content": "think"}})
    assert to_openai_messages([msg]) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "reasoning_content": "think",
            "thinking_blocks": [{"type": "thinking", "signature": "sig", "thinking": "think"}],
        }
    ]


def test_chat_wire_emits_one_thinking_entry_per_signed_block() -> None:
    msg = AssistantMessage(
        blocks=[
            SignedReasoningBlock(signature="s1", content="a"),
            RedactedReasoningBlock(data="opaque"),
            SignedReasoningBlock(signature="s2", content="b"),
            TextBlock(text="answer"),
        ]
    )
    (wire,) = to_openai_messages([msg])
    assert wire["thinking_blocks"] == [
        {"type": "thinking", "signature": "s1", "thinking": "a"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "signature": "s2", "thinking": "b"},
    ]
    assert wire["reasoning_content"] == "ab"


def test_chat_wire_tool_call_has_only_supported_fields() -> None:
    msg = AssistantMessage(
        blocks=[ToolCall(name="t", arguments="{}", tool_call_id="c1", signature="thought-signature")]
    )
    with pytest.raises(NotImplementedError, match="cannot pass back signed tool calls"):
        to_openai_messages([msg])

    (wire,) = to_openai_messages([msg], allow_tool_call_signatures=True)
    assert wire["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "t", "arguments": "{}"},
            "provider_specific_fields": {"thought_signature": "thought-signature"},
        }
    ]


def test_chat_wire_rejects_synthetic_tool_ids() -> None:
    call = ToolCall.from_provider(provider_id=None, name="t", arguments="{}")

    with pytest.raises(NotImplementedError, match="without a provider-issued ID"):
        to_openai_messages(
            [
                AssistantMessage(blocks=[call]),
                ToolMessage(content="done", tool_call_id=call.tool_call_id, name="t"),
            ]
        )


# ---------------------------------------------------------------------------
# Wire-format goldens: Responses replay
# ---------------------------------------------------------------------------


def test_responses_replay_preserves_interleaved_order() -> None:
    msg = AssistantMessage(
        blocks=[
            ReasoningRefBlock(id="rs_1", content="thinking"),
            TextBlock(text="first"),
            ToolCall(name="grep", arguments="{}", tool_call_id="call-1"),
            TextBlock(text="second"),
        ]
    )
    _instructions, items = _to_open_responses_input([msg])
    assert [item["type"] for item in items] == ["reasoning", "message", "function_call", "message"]
    assert items[0] == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "thinking"}],
    }
    assert items[1]["content"] == [{"type": "output_text", "text": "first"}]
    assert items[2]["call_id"] == "call-1"
    assert items[3]["content"] == [{"type": "output_text", "text": "second"}]


def test_encrypted_reasoning_feeds_channel_projections() -> None:
    """The reasoning projection and CC replay see encrypted summaries as readable content."""
    msg = AssistantMessage(
        blocks=[EncryptedReasoningBlock(id="rs_9", encrypted_content="zdr", summary=["alpha", "beta"])]
    )
    with pytest.warns(DeprecationWarning, match="AssistantMessage.reasoning is deprecated"):
        reasoning = msg.reasoning
    assert reasoning is not None
    assert reasoning.content == "alpha\nbeta"
    assert reasoning.signature is None


def test_responses_replay_encrypted_reasoning_echoes_item_verbatim() -> None:
    msg = AssistantMessage(blocks=[EncryptedReasoningBlock(id="rs_9", encrypted_content="zdr", summary=["a", "b"])])
    _instructions, items = _to_open_responses_input([msg])
    assert items == [
        {
            "type": "reasoning",
            "id": "rs_9",
            "summary": [{"type": "summary_text", "text": "a"}, {"type": "summary_text", "text": "b"}],
            "encrypted_content": "zdr",
        }
    ]


def test_responses_replay_of_legacy_channel_payload_keeps_v01_order() -> None:
    """A legacy flat payload replays message-then-calls, as v0.1.11 emitted."""
    msg = AssistantMessage.model_validate(
        {
            "content": "hi",
            "tool_calls": [
                {"name": "a", "arguments": "{}", "tool_call_id": "c1"},
                {"name": "b", "arguments": "{}", "tool_call_id": "c2"},
            ],
        }
    )
    _instructions, items = _to_open_responses_input([msg])
    assert [item["type"] for item in items] == ["message", "function_call", "function_call"]


def test_responses_replay_rejects_inband_and_signed_reasoning() -> None:
    msg = AssistantMessage(
        blocks=[
            ReasoningBlock(content="in-band"),
            SignedReasoningBlock(signature="sig", content="signed"),
            TextBlock(text="hi"),
        ]
    )
    with pytest.raises(NotImplementedError, match="without an item ID"):
        _to_open_responses_input([msg])


# ---------------------------------------------------------------------------
# §8 contract: identity, metadata opacity, subclass preservation
# ---------------------------------------------------------------------------


class _SubclassedAssistantMessage(AssistantMessage):
    """Stand-in for an integrator's typed carrier (an out-of-band side-store reference)."""

    side_ref: str | None = None


class _ScriptedClient(LLMClient):
    def __init__(self, responses: Sequence[AssistantMessage | Exception], max_tokens: int = 100_000) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self._max_tokens = max_tokens

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
        response = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


def _finish_message(*, token_usage: TokenUsage | None = None, message_id: str | None = None) -> AssistantMessage:
    kwargs: dict[str, Any] = {}
    if message_id is not None:
        kwargs["id"] = message_id
    return AssistantMessage(
        blocks=[
            TextBlock(text="Done"),
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments='{"reason": "Completed", "paths": []}',
                tool_call_id="call-finish",
            ),
        ],
        token_usage=token_usage or TokenUsage(input=10, answer=5),
        **kwargs,
    )


async def test_agent_preserves_subclass_and_metadata_and_id() -> None:
    """The exact object returned by generate is appended to history: same identity,
    subclass intact, metadata untouched."""
    response = _SubclassedAssistantMessage(
        blocks=[
            TextBlock(text="Done"),
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments='{"reason": "Completed", "paths": []}',
                tool_call_id="call-finish",
            ),
        ],
        token_usage=TokenUsage(input=10, answer=5),
        metadata={"integrator/ref": "entry-1"},
        side_ref="side-store-key",
    )
    agent = Agent(
        client=_ScriptedClient([response]),
        name="test-agent",
        max_turns=3,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )
    async with agent.session(cache_on_interrupt=False) as session:
        _params, history, _meta = await session.run([UserMessage(content="Task")])

    stored = [m for group in history for m in group if isinstance(m, AssistantMessage)]
    assert len(stored) == 1
    assert stored[0] is response  # same object, not a copy
    assert isinstance(stored[0], _SubclassedAssistantMessage)
    assert stored[0].side_ref == "side-store-key"
    assert stored[0].metadata == {"integrator/ref": "entry-1"}


async def test_chained_summarization_carries_replaced_ids_transitively() -> None:
    """A second summarization removes the first summary bridge; the new summary's
    replaced_ids must still cover the turns the first summary stood for, so dumped
    histories reconstruct lineage across rounds."""
    heavy = TokenUsage(input=250, answer=100)  # total=350 >= 0.3*1000
    responses = [
        AssistantMessage(id="turn-1", blocks=[TextBlock(text="Working on it")], token_usage=heavy),
        AssistantMessage(blocks=[TextBlock(text="Summary one.")], token_usage=TokenUsage(input=10, answer=5)),
        AssistantMessage(id="turn-2", blocks=[TextBlock(text="Still working")], token_usage=heavy),
        AssistantMessage(blocks=[TextBlock(text="Summary two.")], token_usage=TokenUsage(input=10, answer=5)),
        _finish_message(),
    ]
    agent = Agent(
        client=_ScriptedClient(responses, max_tokens=1000),
        name="test-agent",
        max_turns=10,
        turns_remaining_warning_threshold=2,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        context_summarization_cutoff=0.3,
    )
    async with agent.session(cache_on_interrupt=False) as session:
        _params, history, _meta = await session.run([SystemMessage(content="System"), UserMessage(content="Task")])

    bridges = [m for group in history for m in group if isinstance(m, SummaryMessage)]
    assert len(bridges) == 2
    assert bridges[0].replaced_ids == ["turn-1"]
    assert bridges[-1].replaced_ids == ["turn-1", "turn-2"]

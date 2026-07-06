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
    AssistantMessage,
    ChatMessage,
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


INTERLEAVED_BLOCKS = [
    SignedReasoningBlock(signature="sig-1", content="first thought"),
    TextBlock(text="Let me look."),
    SignedReasoningBlock(signature="sig-2", content="second thought"),
    ToolCall(name="grep", arguments='{"pattern": "x"}', tool_call_id="call-1"),
    TextBlock(text="Found it."),
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def test_joined_text_joins_with_newline() -> None:
    assert joined_text(INTERLEAVED_BLOCKS) == "Let me look.\nFound it."
    assert joined_text([ToolCall(name="t", arguments="{}")]) is None
    assert joined_text([]) is None


def test_final_text_returns_last_text_block() -> None:
    assert final_text(INTERLEAVED_BLOCKS) == "Found it."
    assert final_text([SignedReasoningBlock(signature="s")]) is None


def test_tool_call_and_reasoning_accessors_preserve_order() -> None:
    assert [tc.tool_call_id for tc in tool_call_blocks(INTERLEAVED_BLOCKS)] == ["call-1"]
    assert [type(b) for b in reasoning_blocks(INTERLEAVED_BLOCKS)] == [SignedReasoningBlock, SignedReasoningBlock]


# ---------------------------------------------------------------------------
# Channel projections (read-only, derived from blocks)
# ---------------------------------------------------------------------------


def test_content_projection_joins_text_blocks() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    assert msg.content == "Let me look.\nFound it."


def test_content_projection_empty_string_when_no_text_blocks() -> None:
    msg = AssistantMessage(blocks=[ToolCall(name="t", arguments="{}")])
    assert msg.content == ""


def test_content_projection_media_list_in_block_order() -> None:
    image = _png_block()
    msg = AssistantMessage(blocks=[TextBlock(text="see:"), image, TextBlock(text="above")])
    assert msg.content == ["see:", image, "above"]


def test_reasoning_projection_concatenates_without_separator_and_takes_first_signature() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    assert msg.reasoning is not None
    assert msg.reasoning.signature == "sig-1"
    assert msg.reasoning.content == "first thoughtsecond thought"


def test_reasoning_projection_none_without_reasoning_blocks() -> None:
    assert AssistantMessage(blocks=[TextBlock(text="hi")]).reasoning is None


def test_reasoning_projection_redacted_blocks_contribute_nothing() -> None:
    msg = AssistantMessage(
        blocks=[
            RedactedReasoningBlock(data="opaque-payload"),
            SignedReasoningBlock(signature="sig", content="visible"),
        ]
    )
    assert msg.reasoning is not None
    assert msg.reasoning.content == "visible"
    assert msg.reasoning.signature == "sig"


def test_tool_calls_projection() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    assert [tc.name for tc in msg.tool_calls] == ["grep"]
    assert AssistantMessage(blocks=[TextBlock(text="x")]).tool_calls == []


# ---------------------------------------------------------------------------
# Channel-order synthesis (permanent legacy-upgrade path)
# ---------------------------------------------------------------------------


def test_channel_construction_synthesizes_blocks_in_channel_order() -> None:
    msg = AssistantMessage(
        content="answer",
        reasoning=Reasoning(content="thinking"),
        tool_calls=[ToolCall(name="t", arguments="{}", tool_call_id="c1")],
    )
    assert [b.kind for b in msg.blocks] == ["reasoning", "text", "tool_call"]


def test_flat_reasoning_splits_by_signature_presence() -> None:
    signed = AssistantMessage(content="", reasoning=Reasoning(signature="sig", content="deep"))
    assert signed.blocks == [SignedReasoningBlock(signature="sig", content="deep")]

    in_band = AssistantMessage(content="", reasoning=Reasoning(content="deep"))
    assert in_band.blocks == [ReasoningBlock(content="deep")]


def test_media_content_list_passes_through_in_place() -> None:
    image = _png_block()
    msg = AssistantMessage(content=["before", image, "after"])
    assert msg.blocks == [TextBlock(text="before"), image, TextBlock(text="after")]


def test_empty_content_synthesizes_no_text_block() -> None:
    msg = AssistantMessage(content="", tool_calls=[ToolCall(name="t", arguments="{}")])
    assert [b.kind for b in msg.blocks] == ["tool_call"]
    assert msg.content == ""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_mixing_blocks_with_channel_fields_raises() -> None:
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage(blocks=[TextBlock(text="x")], content="y")  # type: ignore
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage(blocks=[], tool_calls=[ToolCall(name="t", arguments="{}")])  # type: ignore
    with pytest.raises(ValidationError, match="cannot mix"):
        AssistantMessage(blocks=[], reasoning=Reasoning(content="r"))  # type: ignore


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


def test_channel_assignment_raises() -> None:
    msg = AssistantMessage(blocks=[TextBlock(text="x")])
    with pytest.raises(AttributeError, match="migration guide"):
        msg.content = "y"  # type: ignore
    with pytest.raises(AttributeError, match="migration guide"):
        msg.tool_calls = []  # type: ignore
    with pytest.raises(AttributeError, match="migration guide"):
        msg.reasoning = Reasoning(content="r")  # type: ignore


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
    assert msg.reasoning == Reasoning(signature="sig-1", content="thinking")
    assert msg.content == "answer"
    assert msg.tool_calls == []


def test_opaque_block_skipped_on_openai_replay() -> None:
    msg = AssistantMessage(
        blocks=[
            OpaqueBlock(data=OPAQUE_PAYLOAD),
            TextBlock(text="answer"),
        ]
    )
    [wire] = to_openai_messages([msg])
    assert wire == {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}


def test_opaque_block_skipped_on_responses_replay() -> None:
    msg = AssistantMessage(
        blocks=[
            OpaqueBlock(data=OPAQUE_PAYLOAD),
            TextBlock(text="answer"),
        ]
    )
    _, items = _to_open_responses_input([msg])
    assert [item["type"] for item in items] == ["message"]


# ---------------------------------------------------------------------------
# Explicit mutators
# ---------------------------------------------------------------------------


def test_with_text_replaces_text_blocks_before_tool_calls() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    replaced = msg.with_text("redacted")
    assert [b.kind for b in replaced.blocks] == ["signed_reasoning", "signed_reasoning", "text", "tool_call"]
    assert joined_text(replaced.blocks) == "redacted"
    # original untouched
    assert joined_text(msg.blocks) == "Let me look.\nFound it."


def test_with_blocks_replaces_block_list() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS)
    stripped = msg.with_blocks([b for b in msg.blocks if b.kind != "tool_call"])
    assert stripped.tool_calls == []
    assert stripped.id == msg.id


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
    # projections round-trip the channel view
    assert msg.content == "hello"
    assert msg.reasoning == Reasoning(signature="sig", content="thought")
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


@pytest.mark.parametrize(
    "block",
    [
        ReasoningBlock(content="in-band"),
        SignedReasoningBlock(signature="sig", content="signed"),
        RedactedReasoningBlock(data="opaque"),
        ReasoningRefBlock(id="rs_1", content="summary", encrypted_content="zdr-payload"),
        ReasoningRefBlock(id="rs_2"),
    ],
)
def test_each_reasoning_kind_round_trips(block: ReasoningBlock) -> None:
    msg = AssistantMessage(blocks=[block, TextBlock(text="t")])
    restored = AssistantMessage.model_validate(msg.model_dump(mode="json"))
    assert restored.blocks == msg.blocks


def test_interleaved_round_trip_is_lossless_and_id_stable() -> None:
    msg = AssistantMessage(blocks=INTERLEAVED_BLOCKS, token_usage=TokenUsage(input=1, answer=2, reasoning=3))
    restored = AssistantMessage.model_validate(msg.model_dump(mode="json"))
    assert restored.blocks == msg.blocks
    assert restored.id == msg.id
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
                AssistantMessage(id="turn-1", content="working"),
                SummaryMessage(content="bridge", replaced_ids=["turn-1"]),
                TurnWarningMessage(content="2 turns remaining"),
            ]
        ]
    )
    reloaded = SubAgentMetadata.model_validate_json(meta.model_dump_json())
    _user, _assistant, summary, warning = reloaded.message_history[0]
    assert isinstance(summary, SummaryMessage)
    assert summary.replaced_ids == ["turn-1"]
    assert isinstance(warning, TurnWarningMessage)


# ---------------------------------------------------------------------------
# Wire-format goldens: chat-completions-shaped replay
# ---------------------------------------------------------------------------


def test_chat_wire_byte_identical_for_non_signed_turns() -> None:
    """Channel-constructed (v0.1-shaped) messages produce the exact v0.1.11 payload."""
    msg = AssistantMessage(
        content="hi",
        reasoning=Reasoning(content="think"),
        tool_calls=[ToolCall(name="t", arguments="{}", tool_call_id="c1")],
    )
    assert to_openai_messages([msg]) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "signature": None,
                    "name": "t",
                    "arguments": "{}",
                    "tool_call_id": "c1",
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                }
            ],
        }
    ]


def test_chat_wire_single_signed_turn_matches_v01_shape() -> None:
    msg = AssistantMessage(content="hi", reasoning=Reasoning(signature="sig", content="think"))
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


def test_chat_wire_tool_call_dicts_carry_no_kind_key() -> None:
    msg = AssistantMessage(blocks=[ToolCall(name="t", arguments="{}", tool_call_id="c1")])
    (wire,) = to_openai_messages([msg])
    assert "kind" not in wire["tool_calls"][0]


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


def test_responses_replay_reasoning_ref_with_encrypted_content() -> None:
    msg = AssistantMessage(blocks=[ReasoningRefBlock(id="rs_9", encrypted_content="zdr")])
    _instructions, items = _to_open_responses_input([msg])
    assert items == [{"type": "reasoning", "id": "rs_9", "summary": [], "encrypted_content": "zdr"}]


def test_responses_replay_of_channel_constructed_message_keeps_v01_order() -> None:
    """Legacy flat construction replays message-then-calls, as v0.1.11 emitted."""
    msg = AssistantMessage(
        content="hi",
        tool_calls=[
            ToolCall(name="a", arguments="{}", tool_call_id="c1"),
            ToolCall(name="b", arguments="{}", tool_call_id="c2"),
        ],
    )
    _instructions, items = _to_open_responses_input([msg])
    assert [item["type"] for item in items] == ["message", "function_call", "function_call"]


def test_responses_replay_skips_inband_and_signed_reasoning() -> None:
    msg = AssistantMessage(
        blocks=[
            ReasoningBlock(content="in-band"),
            SignedReasoningBlock(signature="sig", content="signed"),
            TextBlock(text="hi"),
        ]
    )
    _instructions, items = _to_open_responses_input([msg])
    assert [item["type"] for item in items] == ["message"]


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
        content="Done",
        tool_calls=[
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments='{"reason": "Completed", "paths": []}',
                tool_call_id="call-finish",
            )
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
        AssistantMessage(id="turn-1", content="Working on it", token_usage=heavy),
        AssistantMessage(content="Summary one.", token_usage=TokenUsage(input=10, answer=5)),
        AssistantMessage(id="turn-2", content="Still working", token_usage=heavy),
        AssistantMessage(content="Summary two.", token_usage=TokenUsage(input=10, answer=5)),
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

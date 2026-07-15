"""Tests for LiteLLM client parsing."""

import pytest

from stirrup.clients.litellm_client import _parse_thinking_blocks
from stirrup.core.models import ReasoningBlock, RedactedReasoningBlock, SignedReasoningBlock


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

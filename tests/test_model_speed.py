"""Tests for AssistantMessage.e2e_otps property."""

import warnings

import pytest

from stirrup.core.models import AssistantMessage, TokenUsage


def test_e2e_otps_valid() -> None:
    """Valid timestamps and output tokens produce correct tok/s."""
    msg = AssistantMessage(
        content="hi",
        token_usage=TokenUsage(input=10, answer=80, reasoning=20),
        request_start_time=100.0,
        request_end_time=102.5,
    )
    # output = answer + reasoning = 100, duration = 2.5 → 40.0 tok/s
    assert msg.e2e_otps == pytest.approx(40.0)


def test_e2e_otps_none_timestamps() -> None:
    """Returns None when timestamps are missing."""
    msg = AssistantMessage(
        content="hi",
        token_usage=TokenUsage(input=10, answer=80, reasoning=20),
    )
    assert msg.e2e_otps is None

    msg2 = AssistantMessage(
        content="hi",
        token_usage=TokenUsage(input=10, answer=80, reasoning=20),
        request_start_time=100.0,
    )
    assert msg2.e2e_otps is None


def test_e2e_otps_zero_duration() -> None:
    """Returns None when duration is zero."""
    msg = AssistantMessage(
        content="hi",
        token_usage=TokenUsage(input=10, answer=80, reasoning=20),
        request_start_time=100.0,
        request_end_time=100.0,
    )
    assert msg.e2e_otps is None


def test_e2e_otps_zero_output_tokens() -> None:
    """Returns None when output tokens are zero."""
    msg = AssistantMessage(
        content="hi",
        token_usage=TokenUsage(input=10, answer=0, reasoning=0),
        request_start_time=100.0,
        request_end_time=102.5,
    )
    assert msg.e2e_otps is None


def test_token_usage_output_deprecation_warning() -> None:
    """TokenUsage(output=...) should emit DeprecationWarning and map to answer."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        usage = TokenUsage(output=50)  # type: ignore[unknown-argument]
    assert usage.answer == 50
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "output" in str(w[0].message).lower()

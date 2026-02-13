"""Tests for model speed measurement and metadata aggregation."""

import warnings

import pytest

from stirrup.clients.utils import compute_model_speed
from stirrup.core.models import ModelSpeed, SubAgentMetadata, TokenUsage, aggregate_metadata


def test_compute_model_speed() -> None:
    """Compute model speed from LLM call wall time."""
    throughput = compute_model_speed(
        model_slug="gpt-4o",
        output_tokens=100,
        reasoning_tokens=20,
        llm_call_duration_seconds=2.5,
    )
    assert throughput is not None
    assert throughput.model_slug == "gpt-4o"
    assert throughput.num_calls == 1
    assert throughput.output_tokens == 100
    assert throughput.reasoning_tokens == 20
    assert throughput.answer_tokens == 80
    assert throughput.llm_call_duration_seconds == 2.5
    assert throughput.e2e_otps == pytest.approx(40.0)


def test_compute_model_speed_invalid_duration() -> None:
    """Return None when output duration is invalid."""
    throughput = compute_model_speed(
        model_slug="gpt-4o",
        output_tokens=100,
        reasoning_tokens=0,
        llm_call_duration_seconds=0.0,
    )
    assert throughput is None


def test_aggregate_metadata_rolls_up_model_speed_same_model() -> None:
    """Roll up root and subagent model speed for the same model."""
    root_throughput = ModelSpeed(
        model_slug="gpt-4o",
        num_calls=1,
        output_tokens=120,
        reasoning_tokens=30,
        llm_call_duration_seconds=1.5,
    )
    sub_throughput = ModelSpeed(
        model_slug="gpt-4o",
        num_calls=1,
        output_tokens=150,
        reasoning_tokens=50,
        llm_call_duration_seconds=1.5,
    )

    metadata = {
        "model_speed": [root_throughput],
        "sub_agent": [
            SubAgentMetadata(
                message_history=[],
                run_metadata={"model_speed": [sub_throughput]},
            )
        ],
    }

    aggregated = aggregate_metadata(metadata, return_json_serializable=False)
    assert len(aggregated["model_speed"]) == 1
    total = aggregated["model_speed"][0]
    assert total.model_slug == "gpt-4o"
    assert total.num_calls == 2
    assert total.output_tokens == 270
    assert total.reasoning_tokens == 80
    assert total.answer_tokens == 190
    assert total.llm_call_duration_seconds == 3.0


def test_aggregate_metadata_groups_model_speed_by_model() -> None:
    """Separate models get separate speed entries after aggregation."""
    root_throughput = ModelSpeed(
        model_slug="gpt-4o",
        num_calls=1,
        output_tokens=120,
        reasoning_tokens=0,
        llm_call_duration_seconds=1.5,
    )
    sub_throughput = ModelSpeed(
        model_slug="claude-sonnet",
        num_calls=1,
        output_tokens=150,
        reasoning_tokens=50,
        llm_call_duration_seconds=1.5,
    )

    metadata = {
        "model_speed": [root_throughput],
        "sub_agent": [
            SubAgentMetadata(
                message_history=[],
                run_metadata={"model_speed": [sub_throughput]},
            )
        ],
    }

    aggregated = aggregate_metadata(metadata, return_json_serializable=False)
    throughput_list = aggregated["model_speed"]
    assert len(throughput_list) == 2
    by_model = {t.model_slug: t for t in throughput_list}
    assert by_model["gpt-4o"].num_calls == 1
    assert by_model["gpt-4o"].output_tokens == 120
    assert by_model["claude-sonnet"].num_calls == 1
    assert by_model["claude-sonnet"].output_tokens == 150


def test_model_speed_add_different_models_raises() -> None:
    """Adding speed from different models should raise ValueError."""
    a = ModelSpeed(model_slug="gpt-4o", output_tokens=100, llm_call_duration_seconds=1.0)
    b = ModelSpeed(model_slug="claude-sonnet", output_tokens=100, llm_call_duration_seconds=1.0)
    with pytest.raises(ValueError, match="Cannot combine speed from different models"):
        a + b


def test_token_usage_output_deprecation_warning() -> None:
    """TokenUsage(output=...) should emit DeprecationWarning and map to answer."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        usage = TokenUsage(output=50)  # type: ignore[unknown-argument]
    assert usage.answer == 50
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "output" in str(w[0].message).lower()

"""Tests for effective throughput measurement and metadata aggregation."""

from stirrup.clients.utils import compute_effective_throughput
from stirrup.core.models import EffectiveThroughputUsage, SubAgentMetadata, aggregate_metadata


def test_compute_effective_throughput() -> None:
    """Compute effective throughput from LLM call wall time."""
    throughput = compute_effective_throughput(
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
    assert throughput.sum_output_tokens_per_second == 40.0


def test_compute_effective_throughput_invalid_duration() -> None:
    """Return None when output duration is invalid."""
    throughput = compute_effective_throughput(
        model_slug="gpt-4o",
        output_tokens=100,
        reasoning_tokens=0,
        llm_call_duration_seconds=0.0,
    )
    assert throughput is None


def test_aggregate_metadata_rolls_up_effective_throughput_same_model() -> None:
    """Roll up root and subagent effective throughput for the same model."""
    root_throughput = EffectiveThroughputUsage(
        model_slug="gpt-4o",
        num_calls=1,
        sum_output_tokens_per_second=80.0,
        output_tokens=120,
        reasoning_tokens=30,
        llm_call_duration_seconds=1.5,
    )
    sub_throughput = EffectiveThroughputUsage(
        model_slug="gpt-4o",
        num_calls=1,
        sum_output_tokens_per_second=100.0,
        output_tokens=150,
        reasoning_tokens=50,
        llm_call_duration_seconds=1.5,
    )

    metadata = {
        "effective_throughput": [root_throughput],
        "sub_agent": [
            SubAgentMetadata(
                message_history=[],
                run_metadata={"effective_throughput": [sub_throughput]},
            )
        ],
    }

    aggregated = aggregate_metadata(metadata, return_json_serializable=False)
    assert len(aggregated["effective_throughput"]) == 1
    total = aggregated["effective_throughput"][0]
    assert total.model_slug == "gpt-4o"
    assert total.num_calls == 2
    assert total.output_tokens == 270
    assert total.reasoning_tokens == 80
    assert total.answer_tokens == 190
    assert total.llm_call_duration_seconds == 3.0


def test_aggregate_metadata_groups_effective_throughput_by_model() -> None:
    """Separate models get separate throughput entries after aggregation."""
    root_throughput = EffectiveThroughputUsage(
        model_slug="gpt-4o",
        num_calls=1,
        sum_output_tokens_per_second=80.0,
        output_tokens=120,
        reasoning_tokens=0,
        llm_call_duration_seconds=1.5,
    )
    sub_throughput = EffectiveThroughputUsage(
        model_slug="claude-sonnet",
        num_calls=1,
        sum_output_tokens_per_second=100.0,
        output_tokens=150,
        reasoning_tokens=50,
        llm_call_duration_seconds=1.5,
    )

    metadata = {
        "effective_throughput": [root_throughput],
        "sub_agent": [
            SubAgentMetadata(
                message_history=[],
                run_metadata={"effective_throughput": [sub_throughput]},
            )
        ],
    }

    aggregated = aggregate_metadata(metadata, return_json_serializable=False)
    throughput_list = aggregated["effective_throughput"]
    assert len(throughput_list) == 2
    by_model = {t.model_slug: t for t in throughput_list}
    assert by_model["gpt-4o"].num_calls == 1
    assert by_model["gpt-4o"].output_tokens == 120
    assert by_model["claude-sonnet"].num_calls == 1
    assert by_model["claude-sonnet"].output_tokens == 150

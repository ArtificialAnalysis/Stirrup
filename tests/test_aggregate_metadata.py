"""Tests for aggregating metadata that mixes cached JSON dicts with live models."""

from typing import Any

from stirrup.core.models import ToolUseCountMetadata, aggregate_metadata


def _aggregate(metadata_list: list[Any]) -> dict[str, Any]:
    result = aggregate_metadata({"code_exec": metadata_list}, return_json_serializable=False)
    assert isinstance(result, dict)
    return result


def test_aggregate_cached_dict_then_live_model_does_not_raise() -> None:
    """A resumed run lists cached turns (dicts) before new turns (models)."""
    result = _aggregate([{"num_uses": 1}, ToolUseCountMetadata(num_uses=2)])

    assert result["code_exec"] == ToolUseCountMetadata(num_uses=2)


def test_aggregate_live_model_then_cached_dict_does_not_raise() -> None:
    result = _aggregate([ToolUseCountMetadata(num_uses=2), {"num_uses": 1}])

    assert result["code_exec"] == ToolUseCountMetadata(num_uses=2)


def test_aggregate_mixed_list_sums_the_live_models() -> None:
    result = _aggregate([{"num_uses": 1}, {"num_uses": 1}, ToolUseCountMetadata(num_uses=2), ToolUseCountMetadata()])

    assert result["code_exec"] == ToolUseCountMetadata(num_uses=3)


def test_aggregate_only_cached_dicts_still_merge() -> None:
    result = _aggregate([{"num_uses": 1}, {"num_uses": 2}])

    assert result["code_exec"] == {"num_uses": 3}

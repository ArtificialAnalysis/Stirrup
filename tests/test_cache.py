"""Durable cache identity, canonical state, generations, and restore behavior."""

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel

from stirrup.core import cache as cache_module
from stirrup.core.cache import (
    CACHE_IDENTITY_VERSION,
    CACHE_LAYOUT_VERSION,
    CacheFileIdentity,
    CacheManager,
    CacheState,
    build_tool_registry,
    compute_task_hash,
    decode_cached_message_sequence,
)
from stirrup.core.models import (
    AssistantMessage,
    ImageContentBlock,
    SubAgentMetadata,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    ToolUseCountMetadata,
    UserMessage,
    aggregate_metadata,
)
from stirrup.tools.finish import FinishParams


class HealthMetadata(BaseModel):
    healthy: bool

    def __add__(self, other: "HealthMetadata") -> "HealthMetadata":
        return type(self)(healthy=self.healthy and other.healthy)


class CountParams(BaseModel):
    count: int


class OtherParams(BaseModel):
    value: str


def _tool(name: str, parameters: type[BaseModel] = CountParams, *, description: str | None = None) -> Tool:
    return Tool(
        name=name,
        description=description or name,
        parameters=parameters,
        executor=lambda _params: ToolResult(content="ok"),
    )


def _state(task_hash: str, content: str = "state") -> CacheState:
    return CacheState(msgs=[UserMessage(content=content)], full_msg_history=[], task_hash=task_hash)


def _selected_generation(cache_root: Path, task_hash: str) -> Path:
    pointer = json.loads((cache_root / task_hash / "current.json").read_text())
    return cache_root / task_hash / "generations" / pointer["generation"]


def test_identity_covers_configuration_tool_definitions_and_uploaded_content() -> None:
    work = _tool("work", description="Do work")
    finish = _tool("finish", FinishParams, description="Finish work")
    _, _, tool_definitions = build_tool_registry([work], [finish])
    _, _, changed_description = build_tool_registry(
        [_tool("work", description="Do different work")],
        [finish],
    )
    _, _, changed_schema = build_tool_registry([_tool("work", OtherParams)], [finish])
    _, _, changed_category = build_tool_registry([], [work, finish])
    input_file = CacheFileIdentity.from_content("input/data.txt", b"input")
    skill_file = CacheFileIdentity.from_content("review/SKILL.md", b"skill")

    baseline = compute_task_hash(
        [UserMessage(content="task")],
        agent_name="agent",
        model_slug="model",
        system_prompt="system",
        tool_definitions=tool_definitions,
        input_files=[input_file],
        skill_files=[skill_file],
    )
    variants = [
        compute_task_hash([UserMessage(content="other")], agent_name="agent"),
        compute_task_hash([UserMessage(content="task")], agent_name="other"),
        compute_task_hash([UserMessage(content="task")], agent_name="agent", model_slug="other"),
        compute_task_hash([UserMessage(content="task")], agent_name="agent", system_prompt="other"),
        compute_task_hash([UserMessage(content="task")], agent_name="agent", tool_definitions=changed_description),
        compute_task_hash([UserMessage(content="task")], agent_name="agent", tool_definitions=changed_schema),
        compute_task_hash([UserMessage(content="task")], agent_name="agent", tool_definitions=changed_category),
        compute_task_hash(
            [UserMessage(content="task")],
            agent_name="agent",
            input_files=[CacheFileIdentity.from_content("input/data.txt", b"changed")],
        ),
        compute_task_hash(
            [UserMessage(content="task")],
            agent_name="agent",
            skill_files=[CacheFileIdentity.from_content("review/SKILL.md", b"changed")],
        ),
    ]

    assert len(baseline) == 64
    assert all(variant != baseline for variant in variants)
    assert baseline == compute_task_hash(
        [UserMessage(content="task")],
        agent_name="agent",
        model_slug="model",
        system_prompt="system",
        tool_definitions=tool_definitions,
        input_files=[input_file],
        skill_files=[skill_file],
    )


def test_file_identity_rejects_unsafe_names() -> None:
    for name in ("/absolute", "../escape", "safe/../../escape"):
        with pytest.raises(ValueError, match="safe relative path"):
            CacheFileIdentity.from_content(name, b"content")


def test_state_round_trip_preserves_flat_typed_metadata() -> None:
    child = SubAgentMetadata(
        message_history=[[UserMessage(content="child")]],
        run_metadata={
            "usage": [ToolUseCountMetadata(num_uses=2)],
            "health": [HealthMetadata(healthy=True)],
            "token_usage": [TokenUsage(input=3, answer=4)],
        },
    )
    state = CacheState(
        msgs=[UserMessage(content="parent")],
        full_msg_history=[],
        run_metadata={"worker": [child], "health": [HealthMetadata(healthy=True), HealthMetadata(healthy=False)]},
        task_hash="typed",
    )

    serialized = state.to_dict()
    restored = CacheState.from_dict(serialized)

    assert "run_metadata_by_turn" not in serialized
    assert isinstance(restored.run_metadata["worker"][0], SubAgentMetadata)
    # Only Stirrup-owned metadata types are restored; application models come back as plain dicts.
    assert restored.run_metadata["health"] == [{"healthy": True}, {"healthy": False}]
    assert aggregate_metadata({"worker": restored.run_metadata["worker"]}) == aggregate_metadata({"worker": [child]})


def test_plain_metadata_envelope_collisions_round_trip_as_mappings() -> None:
    collisions = [
        {"__stirrup_metadata_type__": "TokenUsage.v1", "value": {"input": 9, "answer": 4}},
        {
            "__stirrup_metadata_type__": "PydanticBaseModel.v1",
            "value": {"module": "application", "qualname": "Record", "payload": {"nested": True}},
        },
        {"__stirrup_metadata_type__": "Mapping.v1", "value": {"arbitrary": "application data"}},
    ]
    state = CacheState(
        msgs=[UserMessage(content="metadata")],
        full_msg_history=[],
        task_hash="collisions",
        run_metadata={"plain": [{"nested": collisions}, *collisions], "typed": [TokenUsage(input=1, answer=2)]},
    )

    restored = CacheState.from_dict(state.to_dict())

    assert restored.run_metadata["plain"] == [{"nested": collisions}, *collisions]
    assert restored.run_metadata["typed"] == [TokenUsage(input=1, answer=2)]


def test_decoder_derives_partial_complete_skipped_and_terminal_calls() -> None:
    finish_tool = _tool("finish", FinishParams)
    assistant = AssistantMessage(
        content="ordered",
        tool_calls=[
            ToolCall(name="work", arguments='{"count": 1}', tool_call_id="one"),
            ToolCall(name="finish", arguments='{"reason": "done", "paths": []}', tool_call_id="finish"),
            ToolCall(name="work", arguments='{"count": 2}', tool_call_id="skipped"),
        ],
        token_usage=TokenUsage(),
    )
    partial = decode_cached_message_sequence(
        [assistant, ToolMessage(content="one", name="work", tool_call_id="one", success=True)]
    )
    assert [call.tool_call_id for call in partial.completed_calls] == ["one"]
    assert [call.tool_call_id for call in partial.pending_calls] == ["finish", "skipped"]

    complete_messages = [
        assistant,
        ToolMessage(content="one", name="work", tool_call_id="one", success=True),
        ToolMessage(content="done", name="finish", tool_call_id="finish", success=True),
        ToolMessage(content="Skipped", name="work", tool_call_id="skipped", success=False),
    ]
    complete = decode_cached_message_sequence(complete_messages, {"finish": finish_tool})
    assert complete.pending_calls == ()
    assert complete.finish_tool_name == "finish"
    assert isinstance(complete.finish_params, FinishParams)
    assert complete.finish_params.reason == "done"


def test_decoder_keeps_atomic_text_only_image_with_pending_call(tmp_path: Path) -> None:
    assistant = AssistantMessage(
        content="image then work",
        tool_calls=[
            ToolCall(name="image", arguments="{}", tool_call_id="image"),
            ToolCall(name="work", arguments="{}", tool_call_id="work"),
        ],
        token_usage=TokenUsage(),
    )
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), color="red").save(image_path)
    image_user = UserMessage(
        content=["Here is the image for tool call image", ImageContentBlock(data=image_path.read_bytes())]
    )
    progress = decode_cached_message_sequence(
        [
            assistant,
            ToolMessage(
                content=["Done! The User will provide the image for tool call image"],
                name="image",
                tool_call_id="image",
                success=True,
            ),
            image_user,
        ]
    )

    assert [call.name for call in progress.pending_calls] == ["work"]
    assert progress.user_messages == (image_user,)


def test_decoder_matches_earlier_image_result_before_later_finish(tmp_path: Path) -> None:
    finish_tool = _tool("finish", FinishParams)
    assistant = AssistantMessage(
        content="image then finish",
        tool_calls=[
            ToolCall(name="image", arguments="{}", tool_call_id="image"),
            ToolCall(name="finish", arguments='{"reason": "done", "paths": []}', tool_call_id="finish"),
        ],
        token_usage=TokenUsage(),
    )
    image_path = tmp_path / "terminal.png"
    Image.new("RGB", (1, 1), color="blue").save(image_path)
    image_user = UserMessage(
        content=["Here is the image for tool call image", ImageContentBlock(data=image_path.read_bytes())]
    )

    progress = decode_cached_message_sequence(
        [
            assistant,
            ToolMessage(
                content=["Done! The User will provide the image for tool call image"],
                name="image",
                tool_call_id="image",
                success=True,
            ),
            ToolMessage(content="done", name="finish", tool_call_id="finish", success=True),
            image_user,
        ],
        {"finish": finish_tool},
    )

    assert progress.finish_tool_name == "finish"
    assert progress.finish_params == FinishParams(reason="done", paths=[])
    assert progress.user_messages == (image_user,)


def test_decoder_rejects_image_without_its_completed_call_placeholder(tmp_path: Path) -> None:
    assistant = AssistantMessage(
        content="image",
        tool_calls=[ToolCall(name="image", arguments="{}", tool_call_id="image")],
        token_usage=TokenUsage(),
    )
    image_path = tmp_path / "unmatched.png"
    Image.new("RGB", (1, 1)).save(image_path)
    image_user = UserMessage(
        content=["Here is the image for tool call image", ImageContentBlock(data=image_path.read_bytes())]
    )

    with pytest.raises(ValueError, match="placeholder"):
        decode_cached_message_sequence(
            [assistant, ToolMessage(content="not an image placeholder", name="image", tool_call_id="image"), image_user]
        )


@pytest.mark.parametrize(
    "messages",
    [
        [ToolMessage(content="orphan", name="work", tool_call_id="one")],
        [
            AssistantMessage(
                content="bad",
                tool_calls=[ToolCall(name="work", arguments="{}", tool_call_id="one")],
                token_usage=TokenUsage(),
            ),
            ToolMessage(content="wrong", name="work", tool_call_id="other"),
        ],
        [
            AssistantMessage(
                content="bad",
                tool_calls=[
                    ToolCall(name="work", arguments="{}", tool_call_id="same"),
                    ToolCall(name="work", arguments="{}", tool_call_id="same"),
                ],
                token_usage=TokenUsage(),
            ),
        ],
        [
            AssistantMessage(
                content="pending",
                tool_calls=[ToolCall(name="work", arguments="{}", tool_call_id="one")],
                token_usage=TokenUsage(),
            ),
            UserMessage(content="advanced ambiguously"),
        ],
    ],
)
def test_decoder_rejects_ambiguous_or_malformed_pairing(messages: list[Any]) -> None:
    with pytest.raises(ValueError):
        decode_cached_message_sequence(messages)


def _terminal_finish_messages() -> list[Any]:
    return [
        AssistantMessage(
            content="finish",
            tool_calls=[ToolCall(name="finish", arguments='{"reason": "done", "paths": []}', tool_call_id="finish")],
            token_usage=TokenUsage(),
        ),
        ToolMessage(content="done", name="finish", tool_call_id="finish", success=True),
    ]


def test_historical_finish_is_terminal_for_the_whole_cache_state(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    finish_tool = _tool("finish", FinishParams)
    terminal = CacheState(
        msgs=[],
        full_msg_history=[_terminal_finish_messages()],
        task_hash="terminal-history",
    )
    manager.save_state("terminal-history", terminal)

    restored = manager.load_state("terminal-history", finish_tools={"finish": finish_tool})

    assert restored is not None
    assert restored.message_progress.finish_tool_name == "finish"
    assert restored.message_progress.finish_params == FinishParams(reason="done", paths=[])


def test_active_execution_after_historical_finish_is_unavailable(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    finish_tool = _tool("finish", FinishParams)
    malformed = CacheState(
        msgs=[AssistantMessage(content="continued", tool_calls=[], token_usage=TokenUsage())],
        full_msg_history=[_terminal_finish_messages()],
        task_hash="continued-after-finish",
    )
    manager.save_state("continued-after-finish", malformed)

    assert manager.load_state("continued-after-finish", finish_tools={"finish": finish_tool}) is None


def test_manager_rejects_task_hash_mismatch_and_incompatible_versions(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        manager.save_state("requested", _state("different"))

    manager.save_state("task", _state("task"))
    generation = _selected_generation(tmp_path, "task")
    state_path = generation / "state.json"
    data = json.loads(state_path.read_text())
    data["identity_version"] = CACHE_IDENTITY_VERSION - 1
    state_path.write_text(json.dumps(data))
    assert manager.load_state("task") is None

    manager.save_state("verified", _state("verified"))
    verified_state_path = _selected_generation(tmp_path, "verified") / "state.json"
    verified_data = json.loads(verified_state_path.read_text())
    verified_data["task_hash"] = "different"
    verified_state_path.write_text(json.dumps(verified_data))
    assert manager.load_state("verified") is None

    manager.save_state("layout", _state("layout"))
    pointer_path = tmp_path / "layout" / "current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["cache_layout_version"] = CACHE_LAYOUT_VERSION - 1
    pointer_path.write_text(json.dumps(pointer))
    assert manager.load_state("layout") is None


@pytest.mark.parametrize("failure_phase", ["state", "files", "pointer"])
def test_failed_generation_keeps_prior_state_and_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    manager = CacheManager(cache_base_dir=tmp_path / "cache")
    old_files = tmp_path / "old"
    new_files = tmp_path / "new"
    old_files.mkdir()
    new_files.mkdir()
    (old_files / "marker").write_text("old")
    (new_files / "marker").write_text("new")
    manager.save_state("task", _state("task", "old"), old_files)

    original_write = manager._write_json_file  # noqa: SLF001
    original_copy = cache_module.shutil.copytree

    def write(path: Path, data: dict[str, Any]) -> None:
        if failure_phase == "state" and path.name == "state.json":
            raise OSError("state failed")
        original_write(path, data)

    def copy(source: Path, destination: Path, symlinks: bool = False) -> Path:
        if failure_phase == "files" and source == new_files:
            raise OSError("files failed")
        return original_copy(source, destination, symlinks=symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(manager, "_write_json_file", write)
        patch.setattr(cache_module.shutil, "copytree", copy)
        if failure_phase == "pointer":
            patch.setattr(manager, "_replace_pointer", lambda *_args: (_ for _ in ()).throw(OSError("pointer failed")))
        with pytest.raises(OSError):
            manager.save_state("task", _state("task", "new"), new_files)

    restored_root = tmp_path / "restored"
    restored = manager.load_state("task", restore_files_to=restored_root)
    assert restored is not None
    assert restored.msgs[0].content == "old"
    assert (restored_root / "marker").read_text() == "old"
    assert len(list((tmp_path / "cache" / "task" / "generations").iterdir())) == 1


def test_concurrent_saves_never_mix_state_and_files_and_use_one_root_lock(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    snapshots: list[Path] = []
    for index in range(12):
        snapshot = tmp_path / f"snapshot-{index}"
        snapshot.mkdir()
        (snapshot / "marker").write_text(str(index))
        snapshots.append(snapshot)

    def save(index: int) -> None:
        CacheManager(cache_base_dir=cache_root).save_state("task", _state("task", str(index)), snapshots[index])

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(save, range(len(snapshots))))

    restored_root = tmp_path / "restored"
    restored = CacheManager(cache_base_dir=cache_root).load_state("task", restore_files_to=restored_root)
    assert restored is not None
    assert (restored_root / "marker").read_text() == restored.msgs[0].content
    assert {path.relative_to(cache_root).as_posix() for path in cache_root.rglob("*.lock")} == {".cache.lock"}


def test_restore_exactly_replaces_only_the_managed_root(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path / "cache")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "kept").write_text("cached")
    (snapshot / "empty").mkdir()
    (snapshot / "link").symlink_to("kept")
    manager.save_state("task", _state("task"), snapshot)

    provider_root = tmp_path / "provider"
    managed_root = provider_root / "managed"
    managed_root.mkdir(parents=True)
    (provider_root / "infrastructure").write_text("untouched")
    (managed_root / "fresh-upload").write_text("remove")
    (managed_root / "stale").mkdir()

    restored = manager.load_state("task", restore_files_to=managed_root)
    assert restored is not None
    assert (provider_root / "infrastructure").read_text() == "untouched"
    assert (managed_root / "kept").read_text() == "cached"
    assert (managed_root / "empty").is_dir()
    assert (managed_root / "link").is_symlink()
    assert not (managed_root / "fresh-upload").exists()
    assert not (managed_root / "stale").exists()


def test_cache_symlinks_never_remove_external_victims(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("safe")

    linked_base = tmp_path / "linked-cache"
    linked_base.symlink_to(victim, target_is_directory=True)
    with pytest.raises(ValueError, match="Cache base directory must not be a symlink"):
        CacheManager(cache_base_dir=linked_base).save_state("task", _state("task"))

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "task").symlink_to(victim, target_is_directory=True)
    manager = CacheManager(cache_base_dir=cache_root)
    assert manager.load_state("task") is None
    with pytest.raises(ValueError, match="Task cache directory must not be a symlink"):
        manager.clear_cache("task")

    assert marker.read_text() == "safe"


def test_symlinked_generations_and_generation_are_unavailable_and_not_cleared(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("safe")
    cache_root = tmp_path / "cache"
    manager = CacheManager(cache_base_dir=cache_root)

    task_dir = cache_root / "generations-link"
    task_dir.mkdir(parents=True)
    (task_dir / "generations").symlink_to(victim, target_is_directory=True)
    assert manager.load_state("generations-link") is None
    with pytest.raises(ValueError, match="Cache generations directory must not be a symlink"):
        manager.clear_cache("generations-link")

    manager.save_state("generation-link", _state("generation-link"))
    generation = _selected_generation(cache_root, "generation-link")
    cache_module.shutil.rmtree(generation)
    generation.symlink_to(victim, target_is_directory=True)
    assert manager.load_state("generation-link") is None
    with pytest.raises(ValueError, match="generation directory must not be a symlink"):
        manager.clear_cache("generation-link")

    assert marker.read_text() == "safe"


def test_cache_paths_are_private_under_permissive_umask(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not available")
    cache_root = tmp_path / "cache"
    old_umask = os.umask(0)
    try:
        CacheManager(cache_base_dir=cache_root).save_state("private", _state("private"))
    finally:
        os.umask(old_umask)

    generation = _selected_generation(cache_root, "private")
    for directory in (cache_root, cache_root / "private", cache_root / "private" / "generations", generation):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for file_path in (cache_root / ".cache.lock", cache_root / "private" / "current.json", generation / "state.json"):
        assert stat.S_IMODE(file_path.stat().st_mode) == 0o600


def test_restore_preserves_execution_file_modes(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not available")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    executable = snapshot / "run.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o751)
    manager = CacheManager(cache_base_dir=tmp_path / "cache")
    manager.save_state("modes", _state("modes"), snapshot)

    restored = tmp_path / "restored"
    assert manager.load_state("modes", restore_files_to=restored) is not None
    assert stat.S_IMODE((restored / "run.sh").stat().st_mode) == 0o751


def test_get_cache_info_summarizes_the_selected_generation(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    manager.save_state("info", _state("info"))
    data = json.loads((_selected_generation(tmp_path, "info") / "state.json").read_text())

    assert manager.get_cache_info("info") == {
        "task_hash": "info",
        "turn": 0,
        "timestamp": data["timestamp"],
        "agent_name": "",
    }


def test_malformed_pointer_and_state_are_unavailable_but_permission_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    manager.save_state("pointer", _state("pointer"))
    (tmp_path / "pointer" / "current.json").write_text("not json")
    assert manager.load_state("pointer") is None
    assert manager.has_cache("pointer")

    manager.save_state("state", _state("state"))
    (_selected_generation(tmp_path, "state") / "state.json").write_text('{"identity_version": 2}')
    assert manager.load_state("state") is None

    manager.save_state("permission", _state("permission"))
    selected_state = _selected_generation(tmp_path, "permission") / "state.json"
    original_open = Path.open

    def denied(path: Path, *args: object, **kwargs: object) -> object:
        if path == selected_state:
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)  # ty: ignore[no-matching-overload]

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError, match="denied"):
        manager.load_state("permission")


def test_load_cleans_unselected_generations(tmp_path: Path) -> None:
    manager = CacheManager(cache_base_dir=tmp_path)
    manager.save_state("task", _state("task"))
    generations = tmp_path / "task" / "generations"
    (generations / "unselected-after-power-loss").mkdir()

    assert manager.load_state("task") is not None
    assert len(list(generations.iterdir())) == 1

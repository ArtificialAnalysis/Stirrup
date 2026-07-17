"""Cache module for persisting and resuming agent state.

Provides functionality to cache agent state (messages, run metadata, execution environment files)
on non-success exits and restore that state for resumption in new runs.
"""

import base64
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, NotRequired, TypedDict, cast
from weakref import WeakValueDictionary

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from pydantic import BaseModel, TypeAdapter

from stirrup.core.models import (
    Addable,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    ImageContentBlock,
    SubAgentMetadata,
    SummaryMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolUseCountMetadata,
    TurnWarningMessage,
    UserMessage,
    VideoContentBlock,
    _UnresolvedMetadata,
)

logger = logging.getLogger(__name__)

# Default cache directory relative to the project root
DEFAULT_CACHE_DIR = Path("~/.cache/stirrup/").expanduser()
CACHE_IDENTITY_VERSION = 2
CACHE_LAYOUT_VERSION = 2

_METADATA_TYPE_KEY = "__stirrup_metadata_type__"
_METADATA_VALUE_KEY = "value"
_SUBAGENT_METADATA_TYPE = "SubAgentMetadata.v1"
_TOKEN_USAGE_METADATA_TYPE = "TokenUsage.v1"
_TOOL_USE_COUNT_METADATA_TYPE = "ToolUseCountMetadata.v1"
_APPLICATION_METADATA_TYPE = "PydanticBaseModel.v1"
_MAPPING_METADATA_TYPE = "Mapping.v1"


class _CacheRootLock:
    """Process-local half of a cache-root-scoped lock."""

    def __init__(self) -> None:
        self.thread_lock = threading.RLock()


_ROOT_LOCKS: WeakValueDictionary[Path, _CacheRootLock] = WeakValueDictionary()
_ROOT_LOCKS_GUARD = threading.Lock()


def _get_root_lock(cache_root: Path) -> _CacheRootLock:
    with _ROOT_LOCKS_GUARD:
        root_lock = _ROOT_LOCKS.get(cache_root)
        if root_lock is None:
            root_lock = _CacheRootLock()
            _ROOT_LOCKS[cache_root] = root_lock
        return root_lock


def _lock_file(file: BinaryIO) -> None:
    if sys.platform == "win32":
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def _unlock_file(file: BinaryIO) -> None:
    if sys.platform == "win32":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


# TypeAdapter for deserializing ChatMessage discriminated union
ChatMessageAdapter: TypeAdapter[ChatMessage] = TypeAdapter(ChatMessage)


class _SerializedCacheState(TypedDict):
    msgs: list[dict[str, Any]]
    full_msg_history: list[list[dict[str, Any]]]
    task_hash: str
    run_metadata: dict[str, list[Any]]
    identity_version: int
    timestamp: NotRequired[str]
    agent_name: NotRequired[str]


_CacheStateAdapter: TypeAdapter[_SerializedCacheState] = TypeAdapter(_SerializedCacheState)


@dataclass(frozen=True)
class CacheFileIdentity:
    """Immutable identity of one file in an uploaded execution-environment snapshot."""

    relative_name: str
    sha256: str

    def __post_init__(self) -> None:
        normalized_name = PurePosixPath(self.relative_name.replace("\\", "/"))
        if normalized_name.is_absolute() or not normalized_name.parts or ".." in normalized_name.parts:
            raise ValueError(f"Cache file name must be a safe relative path: {self.relative_name}")
        object.__setattr__(self, "relative_name", normalized_name.as_posix())

    @classmethod
    def from_content(cls, relative_name: str, content: bytes) -> "CacheFileIdentity":
        """Fingerprint bytes already present in the execution environment."""
        return cls(relative_name=relative_name, sha256=hashlib.sha256(content).hexdigest())


def build_tool_registry(
    ordinary_tools: Iterable[Tool[Any, Any]],
    finish_tools: Iterable[Tool[Any, Any]],
) -> tuple[dict[str, Tool[Any, Any]], dict[str, Tool[Any, Any]], tuple[dict[str, Any], ...]]:
    """Validate model-visible tool names and build their registry and cache identity together."""
    registry: dict[str, Tool[Any, Any]] = {}
    categories: dict[str, str] = {}
    finish_registry: dict[str, Tool[Any, Any]] = {}
    definitions: list[dict[str, Any]] = []

    for category, tools in (("ordinary", ordinary_tools), ("finish", finish_tools)):
        for tool in tools:
            if tool.name in registry:
                if categories[tool.name] != category:
                    raise ValueError(
                        f"Tool name {tool.name!r} in ordinary tools collides with a finish tool; "
                        "names must be unique across ordinary and finish tools"
                    )
                raise ValueError(f"Tool name {tool.name!r} is duplicated across {category} tools")
            registry[tool.name] = tool
            categories[tool.name] = category
            if category == "finish":
                finish_registry[tool.name] = tool
            definitions.append(
                {
                    "category": category,
                    "name": tool.name,
                    "description": tool.description,
                    "parameter_schema": tool.parameters.model_json_schema(),
                }
            )

    return registry, finish_registry, tuple(definitions)


def compute_task_hash(
    init_msgs: str | list[ChatMessage],
    *,
    agent_name: str = "",
    model_slug: str = "",
    system_prompt: str | None = None,
    tool_definitions: Iterable[Mapping[str, Any]] = (),
    input_files: Iterable[CacheFileIdentity] = (),
    skill_files: Iterable[CacheFileIdentity] = (),
) -> str:
    """Compute the deterministic identity for a resumable agent run.

    Args:
        init_msgs: Either a string prompt or list of ChatMessage objects.
        agent_name: Name of the agent executing the run.
        model_slug: Model identifier exposed by the LLM client.
        system_prompt: Complete system prompt used by the run.
        tool_definitions: Complete category, name, description, and parameter schema
            definitions exposed to the model.
        input_files: Immutable fingerprints captured from the uploaded execution snapshot.
        skill_files: Immutable fingerprints captured from the uploaded skill snapshot.

    Returns:
        SHA256 hash of the versioned canonical identity payload.
    """
    serialized_init_msgs: dict[str, Any]
    if isinstance(init_msgs, str):
        serialized_init_msgs = {"kind": "prompt", "value": init_msgs}
    else:
        serialized_init_msgs = {
            "kind": "messages",
            "value": [serialize_message(msg) for msg in init_msgs],
        }

    identity = {
        "identity_version": CACHE_IDENTITY_VERSION,
        "init_msgs": serialized_init_msgs,
        "agent_name": agent_name,
        "model_slug": model_slug,
        "system_prompt": system_prompt,
        "tools": sorted(
            (dict(definition) for definition in tool_definitions),
            key=lambda definition: (definition.get("category", ""), definition.get("name", "")),
        ),
        "input_files": sorted(
            ({"relative_name": input_file.relative_name, "sha256": input_file.sha256} for input_file in input_files),
            key=lambda entry: (entry["relative_name"], entry["sha256"]),
        ),
        "skill_files": sorted(
            ({"relative_name": skill_file.relative_name, "sha256": skill_file.sha256} for skill_file in skill_files),
            key=lambda entry: (entry["relative_name"], entry["sha256"]),
        ),
    }
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()


def _serialize_content_block(block: Any) -> dict | str:  # noqa: ANN401
    """Serialize a content block, encoding binary data as base64.

    Args:
        block: A content block (string, ImageContentBlock, VideoContentBlock, AudioContentBlock).

    Returns:
        JSON-serializable representation with base64-encoded binary data.
    """
    if isinstance(block, str):
        return block
    elif isinstance(block, ImageContentBlock):
        return {
            "kind": "image_content_block",
            "data": base64.b64encode(block.data).decode("ascii"),
        }
    elif isinstance(block, VideoContentBlock):
        return {
            "kind": "video_content_block",
            "data": base64.b64encode(block.data).decode("ascii"),
        }
    elif isinstance(block, AudioContentBlock):
        return {
            "kind": "audio_content_block",
            "data": base64.b64encode(block.data).decode("ascii"),
        }
    elif isinstance(block, dict):
        # Handle dict from model_dump that might contain unencoded bytes
        # This can happen when Pydantic fails to base64-encode bytes in mode="json"
        if "data" in block and isinstance(block["data"], bytes):
            return {
                **block,
                "data": base64.b64encode(block["data"]).decode("ascii"),
            }
        return block
    else:
        raise ValueError(f"Unknown content block type: {type(block)}")


def _deserialize_content_block(data: dict | str) -> Any:  # noqa: ANN401
    """Deserialize a content block, decoding base64 binary data.

    Args:
        data: JSON-serialized content block.

    Returns:
        Restored content block with decoded binary data.
    """
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return data

    kind = data.get("kind")
    if kind == "image_content_block":
        return ImageContentBlock(data=base64.b64decode(data["data"]))
    elif kind == "video_content_block":
        return VideoContentBlock(data=base64.b64decode(data["data"]))
    elif kind == "audio_content_block":
        return AudioContentBlock(data=base64.b64decode(data["data"]))
    else:
        # Unknown or already-processed block
        return data


def serialize_message(msg: ChatMessage) -> dict:
    """Serialize a ChatMessage to JSON-compatible format.

    Handles binary content blocks (images, video, audio) by base64 encoding.

    Args:
        msg: A ChatMessage (SystemMessage, UserMessage, AssistantMessage, ToolMessage).

    Returns:
        JSON-serializable dictionary.
    """
    # Use Pydantic's model_dump for base serialization
    data = msg.model_dump(mode="json")

    # Handle content field which may contain binary blocks
    content = data.get("content")
    if isinstance(content, list):
        data["content"] = [_serialize_content_block(block) for block in content]
    elif content is not None and not isinstance(content, str):
        data["content"] = _serialize_content_block(content)

    return data


def deserialize_message(data: dict) -> ChatMessage:
    """Deserialize a ChatMessage from JSON format.

    Handles base64-encoded binary content blocks.

    Args:
        data: JSON dictionary representing a ChatMessage.

    Returns:
        Restored ChatMessage object.
    """
    # Handle content field which may contain base64-encoded binary blocks
    content = data.get("content")
    if isinstance(content, list):
        data["content"] = [_deserialize_content_block(block) for block in content]
    elif content is not None and not isinstance(content, str):
        data["content"] = _deserialize_content_block(content)

    if data.get("role") == "user":
        if data.get("kind") == "summary":
            return SummaryMessage.model_validate(data)
        if data.get("kind") == "turn_warning":
            return TurnWarningMessage.model_validate(data)

    # Use TypeAdapter for discriminated union deserialization
    return ChatMessageAdapter.validate_python(data)


def serialize_messages(msgs: list[ChatMessage]) -> list[dict]:
    """Serialize a list of ChatMessages to JSON-compatible format.

    Args:
        msgs: List of ChatMessage objects.

    Returns:
        List of JSON-serializable dictionaries.
    """
    return [serialize_message(msg) for msg in msgs]


def _serialize_metadata_item(item: Any) -> Any:  # noqa: ANN401
    """Serialize a single metadata item to JSON-compatible format.

    Handles Pydantic models by calling model_dump(mode='json').
    Handles bytes by base64 encoding them.
    """
    if isinstance(item, _UnresolvedMetadata):
        return {
            _METADATA_TYPE_KEY: _APPLICATION_METADATA_TYPE,
            _METADATA_VALUE_KEY: {
                "module": item.module,
                "qualname": item.qualname,
                "payload": item.payload,
            },
        }
    elif type(item) is SubAgentMetadata:
        return {
            _METADATA_TYPE_KEY: _SUBAGENT_METADATA_TYPE,
            _METADATA_VALUE_KEY: {
                "message_history": [serialize_messages(group) for group in item.message_history],
                "run_metadata": _serialize_metadata_item(item.run_metadata),
            },
        }
    elif type(item) is TokenUsage:
        return {
            _METADATA_TYPE_KEY: _TOKEN_USAGE_METADATA_TYPE,
            _METADATA_VALUE_KEY: item.model_dump(mode="json"),
        }
    elif type(item) is ToolUseCountMetadata:
        return {
            _METADATA_TYPE_KEY: _TOOL_USE_COUNT_METADATA_TYPE,
            _METADATA_VALUE_KEY: item.model_dump(mode="json"),
        }
    elif isinstance(item, BaseModel):
        model_type = type(item)
        return {
            _METADATA_TYPE_KEY: _APPLICATION_METADATA_TYPE,
            _METADATA_VALUE_KEY: {
                "module": model_type.__module__,
                "qualname": model_type.__qualname__,
                "payload": item.model_dump(mode="json"),
            },
        }
    elif isinstance(item, bytes):
        # Base64 encode raw bytes to make them JSON-serializable
        return base64.b64encode(item).decode("ascii")
    elif isinstance(item, dict):
        serialized = {k: _serialize_metadata_item(v) for k, v in item.items()}
        if set(item) == {_METADATA_TYPE_KEY, _METADATA_VALUE_KEY}:
            return {
                _METADATA_TYPE_KEY: _MAPPING_METADATA_TYPE,
                _METADATA_VALUE_KEY: serialized,
            }
        return serialized
    elif isinstance(item, list):
        return [_serialize_metadata_item(i) for i in item]
    else:
        return item


def _validate_serialized_message(message: object) -> None:
    """Reject malformed message objects before cache metadata is deserialized."""
    if not isinstance(message, dict):
        raise ValueError("Cached SubAgentMetadata message must be a dictionary")
    message_data = cast(dict[str, object], message)
    if message_data.get("role") == "user" and message_data.get("kind") == "summary":
        SummaryMessage.model_validate(message_data, strict=True)
    elif message_data.get("role") == "user" and message_data.get("kind") == "turn_warning":
        TurnWarningMessage.model_validate(message_data, strict=True)
    else:
        ChatMessageAdapter.validate_python(message_data, strict=True)


def _validate_serialized_metadata_item(item: object) -> None:
    """Validate recognized nested metadata envelopes without resolving application types."""
    if isinstance(item, list):
        for value in item:
            _validate_serialized_metadata_item(value)
        return
    if not isinstance(item, dict):
        return

    item_data = cast(dict[str, object], item)
    if (
        set(item_data) == {_METADATA_TYPE_KEY, _METADATA_VALUE_KEY}
        and item_data[_METADATA_TYPE_KEY] == _SUBAGENT_METADATA_TYPE
    ):
        value = item_data[_METADATA_VALUE_KEY]
        if not isinstance(value, dict) or set(value) != {"message_history", "run_metadata"}:
            raise ValueError("Cached SubAgentMetadata payload is malformed")

        value_data = cast(dict[str, object], value)
        message_history = value_data["message_history"]
        run_metadata = value_data["run_metadata"]
        if not isinstance(message_history, list) or not isinstance(run_metadata, dict):
            raise ValueError("Cached SubAgentMetadata payload is malformed")
        for group in message_history:
            if not isinstance(group, list):
                raise ValueError("Cached SubAgentMetadata message history must contain lists")
            for message in group:
                _validate_serialized_message(message)
        for name, metadata_value in cast(dict[object, object], run_metadata).items():
            if not isinstance(name, str):
                raise ValueError("Cached SubAgentMetadata run metadata names must be strings")
            _validate_serialized_metadata_item(metadata_value)
        return

    if set(item_data) == {_METADATA_TYPE_KEY, _METADATA_VALUE_KEY}:
        metadata_type = item_data[_METADATA_TYPE_KEY]
        if metadata_type == _MAPPING_METADATA_TYPE:
            value = item_data[_METADATA_VALUE_KEY]
            if not isinstance(value, dict):
                raise ValueError("Cached mapping metadata payload must be a dictionary")
            for nested_value in value.values():
                _validate_serialized_metadata_item(nested_value)
            return
        if metadata_type in {
            _TOKEN_USAGE_METADATA_TYPE,
            _TOOL_USE_COUNT_METADATA_TYPE,
            _APPLICATION_METADATA_TYPE,
        }:
            return

    for value in item_data.values():
        _validate_serialized_metadata_item(value)


def _deserialize_metadata_item(item: Any) -> Any:  # noqa: ANN401
    """Restore Stirrup-owned metadata types while leaving application dictionaries untouched."""
    if isinstance(item, list):
        return [_deserialize_metadata_item(value) for value in item]
    if not isinstance(item, dict):
        return item

    if set(item) == {_METADATA_TYPE_KEY, _METADATA_VALUE_KEY}:
        metadata_type = item[_METADATA_TYPE_KEY]
        value = item[_METADATA_VALUE_KEY]
        if metadata_type == _TOKEN_USAGE_METADATA_TYPE:
            return TokenUsage.model_validate(value)
        if metadata_type == _TOOL_USE_COUNT_METADATA_TYPE:
            return ToolUseCountMetadata.model_validate(value)
        if metadata_type == _SUBAGENT_METADATA_TYPE:
            if not isinstance(value, dict):
                raise ValueError("Cached SubAgentMetadata payload must be a dictionary")
            message_history = value.get("message_history")
            run_metadata = value.get("run_metadata")
            if not isinstance(message_history, list) or not isinstance(run_metadata, dict):
                raise ValueError("Cached SubAgentMetadata payload is malformed")
            return SubAgentMetadata(
                message_history=[deserialize_messages(group) for group in message_history],
                run_metadata=_deserialize_metadata_item(run_metadata),
            )
        if metadata_type == _APPLICATION_METADATA_TYPE:
            return _deserialize_application_metadata(value)
        if metadata_type == _MAPPING_METADATA_TYPE:
            if not isinstance(value, dict):
                raise ValueError("Cached mapping metadata payload must be a dictionary")
            return {key: _deserialize_metadata_item(nested) for key, nested in value.items()}

    return {key: _deserialize_metadata_item(value) for key, value in item.items()}


def _deserialize_application_metadata(value: object) -> BaseModel | _UnresolvedMetadata:
    """Restore an application model only when its already-loaded class is safe to use."""
    if not isinstance(value, dict):
        return _UnresolvedMetadata("<invalid>", "<invalid>", {"value": value}, "malformed envelope")

    value_dict = cast(dict[str, object], value)
    module_name = value_dict.get("module")
    qualname = value_dict.get("qualname")
    payload = value_dict.get("payload")
    safe_payload = cast(dict[str, Any], payload) if isinstance(payload, dict) else {"value": payload}
    if not isinstance(module_name, str) or not isinstance(qualname, str) or not isinstance(payload, dict):
        return _UnresolvedMetadata(
            str(module_name),
            str(qualname),
            safe_payload,
            "malformed type identity or payload",
        )
    payload_dict = cast(dict[str, Any], payload)

    module_parts = module_name.split(".")
    qualname_parts = qualname.split(".")
    if (
        not module_parts
        or not all(part.isidentifier() for part in module_parts)
        or not qualname_parts
        or "<locals>" in qualname_parts
        or not all(part.isidentifier() for part in qualname_parts)
    ):
        return _UnresolvedMetadata(module_name, qualname, payload_dict, "unsafe type identity")

    target: object = sys.modules.get(module_name)
    if target is None:
        return _UnresolvedMetadata(module_name, qualname, payload_dict, "module is not loaded")
    for part in qualname_parts:
        try:
            namespace = vars(target)
        except TypeError:
            return _UnresolvedMetadata(module_name, qualname, payload_dict, "type path is not a namespace")
        if part not in namespace:
            return _UnresolvedMetadata(module_name, qualname, payload_dict, "type is not loaded")
        target = namespace[part]

    if not isinstance(target, type) or not issubclass(target, BaseModel):
        return _UnresolvedMetadata(module_name, qualname, payload_dict, "target is not a BaseModel class")
    try:
        restored = target.model_validate(payload_dict)
    except Exception as error:
        return _UnresolvedMetadata(
            module_name,
            qualname,
            payload_dict,
            f"payload validation raised {type(error).__name__}",
        )
    if not isinstance(restored, Addable):
        return _UnresolvedMetadata(module_name, qualname, payload_dict, "model is not Addable")
    return restored


def _serialize_run_metadata(run_metadata: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Serialize run_metadata dict containing Pydantic models to JSON-compatible format.

    Args:
        run_metadata: Dict mapping tool names to lists of metadata (may contain Pydantic models).

    Returns:
        JSON-serializable dictionary.
    """
    return {
        tool_name: [_serialize_metadata_item(item) for item in metadata_list]
        for tool_name, metadata_list in run_metadata.items()
    }


def _deserialize_run_metadata(run_metadata: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Restore typed metadata from the flat append-only accepted-results log."""
    for metadata_items in run_metadata.values():
        for item in metadata_items:
            _validate_serialized_metadata_item(item)
    return {
        tool_name: [_deserialize_metadata_item(item) for item in metadata_items]
        for tool_name, metadata_items in run_metadata.items()
    }


def deserialize_messages(data: list[dict]) -> list[ChatMessage]:
    """Deserialize a list of ChatMessages from JSON format.

    Args:
        data: List of JSON dictionaries representing ChatMessages.

    Returns:
        List of restored ChatMessage objects.
    """
    return [deserialize_message(msg_data) for msg_data in data]


@dataclass(frozen=True)
class CachedMessageProgress:
    """Execution boundary derived exclusively from one validated message sequence."""

    assistant: AssistantMessage | None = None
    assistant_index: int | None = None
    tool_messages: tuple[ToolMessage, ...] = ()
    user_messages: tuple[UserMessage, ...] = ()
    completed_calls: tuple[ToolCall, ...] = ()
    pending_calls: tuple[ToolCall, ...] = ()
    finish_tool_name: str | None = None
    finish_params: BaseModel | None = None


def decode_cached_message_sequence(
    messages: list[ChatMessage],
    finish_tools: Mapping[str, Tool[Any, Any]] | None = None,
) -> CachedMessageProgress:
    """Validate one message group and derive its latest resumable boundary."""
    latest = CachedMessageProgress()
    active_assistant: AssistantMessage | None = None
    active_index: int | None = None
    tool_messages: list[ToolMessage] = []
    user_messages: list[UserMessage] = []
    consumed_image_placeholders: dict[str | None, int] = {}

    def progress(*, require_terminal_complete: bool = False) -> CachedMessageProgress:
        if active_assistant is None:
            return CachedMessageProgress()
        calls = active_assistant.tool_calls
        if len(tool_messages) > len(calls):
            raise ValueError("Cached assistant turn has more tool results than tool calls")
        seen_call_ids: set[str] = set()
        for call in calls:
            if call.tool_call_id is not None and call.tool_call_id in seen_call_ids:
                raise ValueError("Cached assistant turn has ambiguous duplicate tool-call ids")
            if call.tool_call_id is not None:
                seen_call_ids.add(call.tool_call_id)
        for call, result in zip(calls, tool_messages, strict=False):
            if result.tool_call_id != call.tool_call_id or result.name != call.name:
                raise ValueError("Cached tool results do not pair with assistant calls in order")

        finish_name: str | None = None
        finish_params: BaseModel | None = None
        if finish_tools:
            successful_finishes = [
                (call, result)
                for call, result in zip(calls, tool_messages, strict=False)
                if result.success and call.name in finish_tools
            ]
            if len(successful_finishes) > 1:
                raise ValueError("Cached assistant turn contains multiple successful finish results")
            if successful_finishes:
                finish_call, finish_result = successful_finishes[0]
                finish_name = finish_call.name
                finish_params = finish_tools[finish_name].parameters.model_validate_json(
                    finish_call.arguments if finish_call.arguments.strip() else "{}"
                )
                if require_terminal_complete and len(tool_messages) != len(calls):
                    raise ValueError("Cached successful finish turn is missing skipped tool results")
                finish_index = tool_messages.index(finish_result)
                if any(result.success for result in tool_messages[finish_index + 1 :]):
                    raise ValueError("Cached successful finish is followed by another successful tool result")

        return CachedMessageProgress(
            assistant=active_assistant,
            assistant_index=active_index,
            tool_messages=tuple(tool_messages),
            user_messages=tuple(user_messages),
            completed_calls=tuple(calls[: len(tool_messages)]),
            pending_calls=tuple(calls[len(tool_messages) :]),
            finish_tool_name=finish_name,
            finish_params=finish_params,
        )

    def consume_image_placeholder(message: ChatMessage) -> bool:
        if (
            not isinstance(message, UserMessage)
            or not isinstance(message.content, list)
            or len(message.content) != 2
            or not isinstance(message.content[0], str)
            or not message.content[0].startswith("Here is the image for tool call ")
            or not isinstance(message.content[1], ImageContentBlock)
        ):
            return False

        matching_results = [
            result
            for result in tool_messages
            if message.content[0] == f"Here is the image for tool call {result.tool_call_id}"
        ]
        if len(matching_results) != 1:
            raise ValueError("Cached image message does not identify one completed tool call")
        result = matching_results[0]
        placeholder = f"Done! The User will provide the image for tool call {result.tool_call_id}"
        blocks = result.content if isinstance(result.content, list) else [result.content]
        placeholder_count = sum(block == placeholder for block in blocks)
        consumed_count = consumed_image_placeholders.get(result.tool_call_id, 0)
        if consumed_count >= placeholder_count:
            raise ValueError("Cached image message has no corresponding completed-call placeholder")
        consumed_image_placeholders[result.tool_call_id] = consumed_count + 1
        return True

    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            if active_assistant is not None and len(tool_messages) != len(active_assistant.tool_calls):
                raise ValueError("Cached message sequence advances past an incomplete assistant tool batch")
            latest = progress(require_terminal_complete=True)
            if latest.finish_params is not None:
                raise ValueError("Cached message sequence advances past a successful finish")
            active_assistant = message
            active_index = index
            tool_messages = []
            user_messages = []
            consumed_image_placeholders = {}
        elif isinstance(message, ToolMessage):
            if active_assistant is None or user_messages:
                raise ValueError("Cached tool result is outside its assistant tool batch")
            tool_messages.append(message)
            progress()
        elif active_assistant is not None:
            current = progress()
            is_image_message = consume_image_placeholder(message)
            if current.pending_calls and not is_image_message:
                raise ValueError("Cached message sequence advances past an incomplete assistant tool batch")
            if current.finish_params is not None and not is_image_message:
                raise ValueError("Cached message sequence advances past a successful finish")
            if isinstance(message, UserMessage):
                user_messages.append(message)
            else:
                latest = progress(require_terminal_complete=True)
                active_assistant = None
                active_index = None
                tool_messages = []
                user_messages = []
                consumed_image_placeholders = {}

    return progress(require_terminal_complete=True) if active_assistant is not None else latest


@dataclass
class CacheState:
    """Canonical message and accepted-metadata state for resuming an agent run."""

    msgs: list[ChatMessage]
    full_msg_history: list[list[ChatMessage]]
    task_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent_name: str = ""
    run_metadata: dict[str, list[Any]] = field(default_factory=dict)
    identity_version: int = CACHE_IDENTITY_VERSION
    message_progress: CachedMessageProgress = field(default_factory=CachedMessageProgress, init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "msgs": serialize_messages(self.msgs),
            "full_msg_history": [serialize_messages(group) for group in self.full_msg_history],
            "run_metadata": _serialize_run_metadata(self.run_metadata),
            "task_hash": self.task_hash,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "identity_version": self.identity_version,
        }

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        finish_tools: Mapping[str, Tool[Any, Any]] | None = None,
    ) -> "CacheState":
        state_data = _CacheStateAdapter.validate_python(data, strict=True)
        identity_version = state_data["identity_version"]
        if identity_version != CACHE_IDENTITY_VERSION:
            raise ValueError(
                f"Unsupported cache identity version {identity_version!r}; expected {CACHE_IDENTITY_VERSION}"
            )
        state = cls(
            msgs=deserialize_messages(state_data["msgs"]),
            full_msg_history=[deserialize_messages(group) for group in state_data["full_msg_history"]],
            task_hash=state_data["task_hash"],
            timestamp=state_data.get("timestamp", ""),
            agent_name=state_data.get("agent_name", ""),
            run_metadata=_deserialize_run_metadata(state_data["run_metadata"]),
            identity_version=identity_version,
        )
        terminal_progress: CachedMessageProgress | None = None
        for group in state.full_msg_history:
            if terminal_progress is not None and group:
                raise ValueError("Cached message history continues after a successful finish")
            progress = decode_cached_message_sequence(group, finish_tools)
            if progress.pending_calls:
                raise ValueError("Cached historical message group contains a pending tool batch")
            if progress.finish_params is not None:
                terminal_progress = progress

        active_progress = decode_cached_message_sequence(state.msgs, finish_tools)
        if terminal_progress is not None:
            if state.msgs:
                raise ValueError("Cached active execution follows a successful historical finish")
            state.message_progress = terminal_progress
        else:
            state.message_progress = active_progress
        return state


class CacheManager:
    """Persist state and files as one immutable, atomically selected generation."""

    def __init__(self, cache_base_dir: Path | None = None) -> None:
        """Initialize a manager rooted at a private, non-symlinked cache directory."""
        self._cache_base_dir = cache_base_dir or DEFAULT_CACHE_DIR
        self._root_lock = _get_root_lock(self._cache_base_dir.absolute())

    def _get_cache_dir(self, task_hash: str) -> Path:
        """Get cache directory path for a task hash."""
        if not task_hash or Path(task_hash).name != task_hash or task_hash in {".", ".."}:
            raise ValueError(f"Cache task hash must be a safe path component: {task_hash!r}")
        return self._cache_base_dir / task_hash

    @staticmethod
    def _require_real_directory(path: Path, description: str, *, allow_missing: bool = False) -> None:
        if path.is_symlink():
            raise ValueError(f"{description} must not be a symlink: {path}")
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"{description} must be a directory: {path}")
        elif not allow_missing:
            raise ValueError(f"{description} does not exist: {path}")

    @classmethod
    def _ensure_private_directory(cls, path: Path, description: str, *, parents: bool = False) -> None:
        cls._require_real_directory(path, description, allow_missing=True)
        path.mkdir(mode=0o700, parents=parents, exist_ok=True)
        cls._require_real_directory(path, description)
        path.chmod(0o700)

    @staticmethod
    def _open_private_binary_file(path: Path, flags: int) -> BinaryIO:
        if path.is_symlink():
            raise ValueError(f"Cache file must not be a symlink: {path}")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags | nofollow, 0o600)
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b")

    @contextmanager
    def _cache_lock(self) -> Iterator[None]:
        """Serialize all transactions for one cache root across threads and processes."""
        with self._root_lock.thread_lock:
            self._ensure_private_directory(self._cache_base_dir, "Cache base directory", parents=True)
            lock_path = self._cache_base_dir / ".cache.lock"
            with self._open_private_binary_file(lock_path, os.O_RDWR | os.O_CREAT | os.O_APPEND) as lock_file:
                _lock_file(lock_file)
                try:
                    yield
                finally:
                    _unlock_file(lock_file)

    @staticmethod
    def _write_json_file(path: Path, data: dict[str, Any]) -> None:
        if path.is_symlink():
            raise ValueError(f"Cache file must not be a symlink: {path}")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _replace_pointer(temp_pointer: Path, current_pointer: Path) -> None:
        if current_pointer.is_symlink():
            raise ValueError(f"Cache pointer must not be a symlink: {current_pointer}")
        temp_pointer.replace(current_pointer)

    @classmethod
    def _remove_path(cls, path: Path, managed_root: Path) -> None:
        """Remove one managed child without following it or symlinked ancestors."""
        cls._require_real_directory(managed_root, "Managed removal root")
        try:
            relative_path = path.absolute().relative_to(managed_root.absolute())
        except ValueError as error:
            raise ValueError(f"Refusing to remove a path outside its managed root: {path}") from error
        if not relative_path.parts:
            raise ValueError(f"Refusing to remove the managed root itself: {managed_root}")

        ancestor = managed_root
        for part in relative_path.parts[:-1]:
            ancestor /= part
            cls._require_real_directory(ancestor, "Managed removal ancestor")

        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
            return
        if path.exists():
            try:
                path.resolve().relative_to(managed_root.resolve())
            except ValueError as error:
                raise ValueError(f"Refusing to recursively remove a path outside its managed root: {path}") from error
            shutil.rmtree(path)

    @classmethod
    def _replace_directory_contents(cls, source: Path, destination: Path) -> None:
        """Make the managed destination root exactly match a generation snapshot."""
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError(f"Cache restore destination must be a real directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        for existing in destination.iterdir():
            cls._remove_path(existing, destination)
        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)

    @classmethod
    def _read_current_generation(cls, cache_dir: Path) -> str | None:
        cls._require_real_directory(cache_dir, "Task cache directory")
        generations_dir = cache_dir / "generations"
        cls._require_real_directory(generations_dir, "Cache generations directory")
        current_file = cache_dir / "current.json"
        if current_file.is_symlink() or (current_file.exists() and not current_file.is_file()):
            return None
        try:
            with current_file.open(encoding="utf-8") as file:
                manifest = json.load(file)
        except FileNotFoundError:
            return None
        except (UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict) or manifest.get("cache_layout_version") != CACHE_LAYOUT_VERSION:
            return None
        generation = manifest.get("generation")
        if not isinstance(generation, str):
            return None
        try:
            if uuid.UUID(generation).hex != generation:
                return None
        except ValueError:
            return None
        generation_dir = generations_dir / generation
        if generation_dir.is_symlink() or not generation_dir.is_dir():
            return None
        return generation

    @classmethod
    def _cleanup_generations(cls, cache_dir: Path, current_generation: str) -> None:
        generations_dir = cache_dir / "generations"
        cls._require_real_directory(generations_dir, "Cache generations directory")
        for generation_dir in generations_dir.iterdir():
            if generation_dir.name == current_generation:
                continue
            if generation_dir.is_symlink():
                raise ValueError(f"Cache generation directory must not be a symlink: {generation_dir}")
            cls._remove_path(generation_dir, cache_dir)

    def save_state(
        self,
        task_hash: str,
        state: CacheState,
        exec_env_dir: Path | None = None,
    ) -> None:
        """Save state and files, then atomically select their complete generation.

        Args:
            task_hash: Unique identifier for this task/cache.
            state: CacheState to persist.
            exec_env_dir: Optional path to execution environment temp directory.
                         If provided, all files will be copied to cache.
        """
        if state.task_hash != task_hash:
            raise ValueError(
                f"Cache state task hash {state.task_hash!r} does not match requested task hash {task_hash!r}"
            )

        state_data = state.to_dict()
        if exec_env_dir is not None and not exec_env_dir.is_dir():
            raise FileNotFoundError(f"Execution environment snapshot does not exist: {exec_env_dir}")

        cache_dir = self._get_cache_dir(task_hash)
        generation = uuid.uuid4().hex
        generations_dir = cache_dir / "generations"
        generation_dir = generations_dir / generation
        pointer_temp = cache_dir / f".current-{generation}.tmp"

        with self._cache_lock():
            self._ensure_private_directory(cache_dir, "Task cache directory")
            self._ensure_private_directory(generations_dir, "Cache generations directory")
            try:
                self._ensure_private_directory(generation_dir, "Cache generation directory")
                self._write_json_file(generation_dir / "state.json", state_data)
                files_dir = generation_dir / "files"
                if exec_env_dir is None:
                    self._ensure_private_directory(files_dir, "Cached files directory")
                else:
                    shutil.copytree(exec_env_dir, files_dir, symlinks=True)

                self._write_json_file(
                    pointer_temp,
                    {
                        "cache_layout_version": CACHE_LAYOUT_VERSION,
                        "generation": generation,
                    },
                )
                self._replace_pointer(pointer_temp, cache_dir / "current.json")
                self._cleanup_generations(cache_dir, generation)
                logger.info("Saved cache generation %s for task %s", generation, task_hash)
            except BaseException:
                self._remove_path(pointer_temp, cache_dir)
                if self._read_current_generation(cache_dir) != generation:
                    self._remove_path(generation_dir, cache_dir)
                raise

    def load_state(
        self,
        task_hash: str,
        *,
        restore_files_to: Path | None = None,
        finish_tools: Mapping[str, Tool[Any, Any]] | None = None,
    ) -> CacheState | None:
        """Load cached state for a task hash.

        Args:
            task_hash: Unique identifier for the task/cache.

        Returns:
            CacheState if cache exists, None otherwise.
        """
        cache_dir = self._get_cache_dir(task_hash)
        with self._cache_lock():
            generations_dir = cache_dir / "generations"
            if (
                cache_dir.is_symlink()
                or not cache_dir.is_dir()
                or generations_dir.is_symlink()
                or not generations_dir.is_dir()
            ):
                logger.debug("No compatible cache generation found for task %s", task_hash)
                return None
            generation = self._read_current_generation(cache_dir)
            if generation is None:
                logger.debug("No compatible cache generation found for task %s", task_hash)
                return None
            if any(path.is_symlink() for path in generations_dir.iterdir()):
                logger.warning("Ignoring cache with a symlinked generation directory for task %s", task_hash)
                return None

            generation_dir = generations_dir / generation
            state_file = generation_dir / "state.json"
            files_dir = generation_dir / "files"
            if (
                generation_dir.is_symlink()
                or not generation_dir.is_dir()
                or state_file.is_symlink()
                or not state_file.is_file()
                or files_dir.is_symlink()
                or not files_dir.is_dir()
            ):
                logger.warning("Ignoring incomplete cache generation %s for task %s", generation, task_hash)
                return None
            try:
                with state_file.open(encoding="utf-8") as file:
                    data = json.load(file)
                state = CacheState.from_dict(data, finish_tools=finish_tools)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                logger.warning("Failed to load cache for task %s: %s", task_hash, error)
                return None

            if state.task_hash != task_hash:
                logger.warning(
                    "Ignoring cache for task %s because its stored task hash is %s",
                    task_hash,
                    state.task_hash,
                )
                return None
            if restore_files_to is not None:
                self._replace_directory_contents(files_dir, restore_files_to)
            self._cleanup_generations(cache_dir, generation)
            logger.info("Loaded cache generation %s for task %s", generation, task_hash)
            return state

    def restore_files(self, task_hash: str, dest_dir: Path) -> bool:
        """Restore cached files to the destination directory.

        Args:
            task_hash: Unique identifier for the task/cache.
            dest_dir: Destination directory (typically the new exec env temp dir).

        Returns:
            True if files were restored, False if no files cache exists.
        """
        return self.load_state(task_hash, restore_files_to=dest_dir) is not None

    def clear_cache(self, task_hash: str) -> None:
        """Remove cache for a specific task.

        Called after successful completion to clean up.

        Args:
            task_hash: Unique identifier for the task/cache.
        """
        cache_dir = self._get_cache_dir(task_hash)
        with self._cache_lock():
            if not cache_dir.exists() and not cache_dir.is_symlink():
                return
            self._require_real_directory(cache_dir, "Task cache directory")
            generations_dir = cache_dir / "generations"
            if generations_dir.exists() or generations_dir.is_symlink():
                self._require_real_directory(generations_dir, "Cache generations directory")
                if any(path.is_symlink() for path in generations_dir.iterdir()):
                    raise ValueError(f"Cache generation directory must not be a symlink: {generations_dir}")
            self._remove_path(cache_dir, self._cache_base_dir)
            logger.info("Cleared cache for task %s", task_hash)

    def list_caches(self) -> list[str]:
        """List all available cache hashes.

        Returns:
            List of task hashes with existing caches.
        """
        if not self._cache_base_dir.exists() and not self._cache_base_dir.is_symlink():
            return []

        with self._cache_lock():
            caches: list[str] = []
            for directory in self._cache_base_dir.iterdir():
                generations_dir = directory / "generations"
                if (
                    directory.is_symlink()
                    or not directory.is_dir()
                    or generations_dir.is_symlink()
                    or not generations_dir.is_dir()
                ):
                    continue
                if self._read_current_generation(directory) is not None:
                    caches.append(directory.name)
            return caches

    def get_cache_info(self, task_hash: str) -> dict | None:
        """Get metadata about a cache without fully loading it.

        Args:
            task_hash: Unique identifier for the task/cache.

        Returns:
            Dictionary with cache info (turn, timestamp, agent_name) or None.
        """
        cache_dir = self._get_cache_dir(task_hash)
        with self._cache_lock():
            generations_dir = cache_dir / "generations"
            if (
                cache_dir.is_symlink()
                or not cache_dir.is_dir()
                or generations_dir.is_symlink()
                or not generations_dir.is_dir()
            ):
                return None
            generation = self._read_current_generation(cache_dir)
            if generation is None or any(path.is_symlink() for path in generations_dir.iterdir()):
                return None
            generation_dir = generations_dir / generation
            state_file = generation_dir / "state.json"
            files_dir = generation_dir / "files"
            if (
                generation_dir.is_symlink()
                or not generation_dir.is_dir()
                or state_file.is_symlink()
                or not state_file.is_file()
                or files_dir.is_symlink()
                or not files_dir.is_dir()
            ):
                return None
            try:
                with state_file.open(encoding="utf-8") as file:
                    data = json.load(file)
            except (FileNotFoundError, UnicodeError, json.JSONDecodeError):
                return None

            if not isinstance(data, dict):
                return None
            if data.get("identity_version") != CACHE_IDENTITY_VERSION or data.get("task_hash") != task_hash:
                return None
            timestamp = data.get("timestamp", "")
            agent_name = data.get("agent_name", "")
            messages = data.get("msgs")
            historical_groups = data.get("full_msg_history")
            if (
                not isinstance(timestamp, str)
                or not isinstance(agent_name, str)
                or not isinstance(messages, list)
                or not isinstance(historical_groups, list)
                or any(not isinstance(group, list) for group in historical_groups)
            ):
                return None

            raw_groups = [*cast(list[list[object]], historical_groups), cast(list[object], messages)]
            if any(not isinstance(message, dict) for group in raw_groups for message in group):
                return None
            turn = sum(
                cast(dict[str, object], message).get("role") == "assistant" for group in raw_groups for message in group
            )
            return {
                "task_hash": task_hash,
                "turn": turn,
                "timestamp": timestamp,
                "agent_name": agent_name,
                "has_files": True,
            }

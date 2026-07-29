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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from stirrup.core.exceptions import CacheUnusableError
from stirrup.core.models import (
    AudioContentBlock,
    ChatMessage,
    ImageContentBlock,
    SummaryMessage,
    TurnWarningMessage,
    VideoContentBlock,
)

logger = logging.getLogger(__name__)

# Default cache directory relative to the project root
DEFAULT_CACHE_DIR = Path("~/.cache/stirrup/").expanduser()

# Version of the identity payload below. Bump it to invalidate every existing cache in one
# edit, with no migration: caches written by an older version stop matching and are ignored.
# Two changes require a bump - altering the identity payload's shape or meaning, and editing
# src/stirrup/prompts/base_system_prompt.txt, whose text is the one model-visible input the
# payload does not cover.
CACHE_IDENTITY_VERSION = 1

# TypeAdapter for deserializing ChatMessage discriminated union
ChatMessageAdapter: TypeAdapter[ChatMessage] = TypeAdapter(ChatMessage)


def compute_task_hash(
    init_msgs: str | list[ChatMessage],
    *,
    agent_name: str = "",
    model_slug: str = "",
    system_prompt: str | None = None,
    tool_definitions: Iterable[Mapping[str, Any]] = (),
    input_files: Iterable[tuple[str, str]] = (),
    skill_files: Iterable[tuple[str, str]] = (),
    skills: Iterable[tuple[str, str]] = (),
) -> str:
    """Compute the identity of a resumable run from everything that shapes its transcript.

    A cached run is resumed only when this value matches exactly, so every input that could
    make a restored transcript wrong must appear here.

    Deliberately excluded:
        - max_turns: raising it and resuming is the reason a max-turns cache exists at all,
          so including it would refuse that resume every time. It reaches the model only
          through the base system prompt, which is replayed verbatim from the cache.
        - The assembled system prompt: its identity-bearing parts (input file list, skills
          section, tool guidance) are covered by the entries below, and the base template
          text is covered by CACHE_IDENTITY_VERSION.
        - Sampling parameters, API endpoint and credentials: never shown to the model, and a
          credential must not reach a value used as a directory name.
        - Run-control knobs (context summarization cutoff, turn warnings, output_dir): the
          messages they inject are already recorded verbatim in the cached transcript.

    Args:
        init_msgs: Either a string prompt or list of ChatMessage objects.
        agent_name: Name of the agent executing the run.
        model_slug: Model identifier exposed by the LLM client.
        system_prompt: User-supplied system prompt, before assembly.
        tool_definitions: Name, description, parameter schema and finish flag for every tool
            the model can see.
        input_files: (destination path, sha256) for each file uploaded to the environment.
        skill_files: (path, sha256) for each file in the uploaded skills tree.
        skills: (name, description) for each loaded skill.

    Returns:
        SHA256 hash (full hex digest) of the canonical identity payload.
    """
    serialized_init_msgs: dict[str, Any]
    if isinstance(init_msgs, str):
        serialized_init_msgs = {"kind": "prompt", "value": init_msgs}
    else:
        serialized_init_msgs = {"kind": "messages", "value": [serialize_message(msg) for msg in init_msgs]}

    identity = {
        "identity_version": CACHE_IDENTITY_VERSION,
        "init_msgs": serialized_init_msgs,
        "agent_name": agent_name,
        "model_slug": model_slug,
        "system_prompt": system_prompt,
        "tools": [dict(definition) for definition in tool_definitions],
        "input_files": sorted(list(entry) for entry in input_files),
        "skill_files": sorted(list(entry) for entry in skill_files),
        "skills": sorted(list(entry) for entry in skills),
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
    from pydantic import BaseModel

    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    elif isinstance(item, bytes):
        # Base64 encode raw bytes to make them JSON-serializable
        return base64.b64encode(item).decode("ascii")
    elif isinstance(item, dict):
        return {k: _serialize_metadata_item(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_serialize_metadata_item(i) for i in item]
    else:
        return item


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


def _serialize_run_metadata_by_turn(
    run_metadata_by_turn: dict[str, dict[str, list[Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Serialize per-turn metadata dicts to JSON-compatible format."""
    return {turn_id: _serialize_run_metadata(turn_metadata) for turn_id, turn_metadata in run_metadata_by_turn.items()}


def deserialize_messages(data: list[dict]) -> list[ChatMessage]:
    """Deserialize a list of ChatMessages from JSON format.

    Args:
        data: List of JSON dictionaries representing ChatMessages.

    Returns:
        List of restored ChatMessage objects.
    """
    return [deserialize_message(msg_data) for msg_data in data]


@dataclass
class CacheState:
    """Serializable state for resuming an agent run.

    Captures all necessary state to resume execution from a specific turn.
    """

    msgs: list[ChatMessage]
    """Current conversation messages in the active run loop."""

    full_msg_history: list[list[ChatMessage]]
    """Groups of messages (separated when context summarization occurs)."""

    task_hash: str
    """Hash of the original init_msgs for verification on resume."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    """ISO timestamp when cache was created."""

    agent_name: str = ""
    """Name of the agent that created this cache."""

    run_metadata_by_turn: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    """Accumulated tool metadata keyed by assistant message id."""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "msgs": serialize_messages(self.msgs),
            "full_msg_history": [serialize_messages(group) for group in self.full_msg_history],
            "run_metadata_by_turn": _serialize_run_metadata_by_turn(self.run_metadata_by_turn),
            "task_hash": self.task_hash,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheState":
        """Create CacheState from JSON dictionary."""
        return cls(
            msgs=deserialize_messages(data["msgs"]),
            full_msg_history=[deserialize_messages(group) for group in data["full_msg_history"]],
            task_hash=data["task_hash"],
            timestamp=data.get("timestamp", ""),
            agent_name=data.get("agent_name", ""),
            # Rebuild rather than adopt: a malformed value must fail here, where the caller
            # treats it as a refusal, not later when the run tries to merge it.
            run_metadata_by_turn={
                str(turn_id): dict(metadata) for turn_id, metadata in data["run_metadata_by_turn"].items()
            },
        )


class CacheManager:
    """Manages cache operations for agent sessions.

    Handles saving/loading cache state and execution environment files.
    """

    def __init__(
        self,
        cache_base_dir: Path | None = None,
        clear_on_success: bool = True,
    ) -> None:
        """Initialize CacheManager.

        Args:
            cache_base_dir: Base directory for cache storage.
                           Defaults to ~/.cache/stirrup/
            clear_on_success: If True (default), automatically clear the cache when
                             the agent completes successfully. Set to False to preserve
                             caches for inspection or manual management.
        """
        self._cache_base_dir = cache_base_dir or DEFAULT_CACHE_DIR
        self.clear_on_success = clear_on_success

    def _get_cache_dir(self, task_hash: str) -> Path:
        """Get cache directory path for a task hash."""
        return self._cache_base_dir / task_hash

    def _get_state_file(self, task_hash: str) -> Path:
        """Get state.json file path for a task hash."""
        return self._get_cache_dir(task_hash) / "state.json"

    def _get_files_dir(self, task_hash: str) -> Path:
        """Get files directory path for a task hash."""
        return self._get_cache_dir(task_hash) / "files"

    def save_state(
        self,
        task_hash: str,
        state: CacheState,
        exec_env_dir: Path | None = None,
    ) -> None:
        """Save cache state and optionally archive execution environment files.

        state.json is the commit point: it is removed first, the files are refreshed, and it
        is written back last via an atomic rename. A save interrupted partway therefore
        leaves an uncommitted cache that load_state refuses, never a state file paired with
        half-copied files.

        Args:
            task_hash: Unique identifier for this task/cache.
            state: CacheState to persist.
            exec_env_dir: Optional path to execution environment temp directory.
                         If provided, all files will be copied to cache.
        """
        cache_dir = self._get_cache_dir(task_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)

        state_file = self._get_state_file(task_hash)
        state_file.unlink(missing_ok=True)

        has_files = False
        if exec_env_dir is not None and exec_env_dir.exists():
            has_files = True
            files_dir = self._get_files_dir(task_hash)
            try:
                if files_dir.exists():
                    shutil.rmtree(files_dir)  # Clear existing files
                shutil.copytree(exec_env_dir, files_dir, dirs_exist_ok=True)
            except Exception as e:
                # Leave the cache uncommitted rather than fail the run that was being cached.
                logger.warning("Failed to cache execution environment files for task %s: %s", task_hash, e)
                return
            logger.info("Saved execution environment files to %s", files_dir)

        state_data = state.to_dict()
        state_data["identity_version"] = CACHE_IDENTITY_VERSION
        state_data["has_files"] = has_files
        logger.debug("Serialized cache state: msgs=%d", len(state.msgs))

        temp_file = state_file.with_suffix(".json.tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        temp_file.replace(state_file)  # Atomic rename (on POSIX systems) - commits the cache
        logger.info("Saved cache state to %s", state_file)

    def has_cached_files(self, task_hash: str) -> bool:
        """Whether a cache holds archived execution environment files."""
        return self._get_files_dir(task_hash).exists()

    def load_state(self, task_hash: str) -> CacheState | None:
        """Load cached state for a task hash.

        Absence and refusal are distinct: a caller that finds no cache should start fresh
        quietly, while a caller whose cache was rejected must say so, because tool calls the
        cached run already made will be made again.

        Args:
            task_hash: Unique identifier for the task/cache.

        Returns:
            CacheState, or None when no cache exists for this task hash.

        Raises:
            CacheUnusableError: A cache exists but cannot be trusted to resume from.
        """
        if not self._get_cache_dir(task_hash).exists():
            logger.debug("No cache found for task %s", task_hash)
            return None

        state_file = self._get_state_file(task_hash)
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise CacheUnusableError(f"state file is unreadable: {e}") from e

        identity_version = data.get("identity_version")
        if identity_version != CACHE_IDENTITY_VERSION:
            raise CacheUnusableError(f"identity version {identity_version} does not match {CACHE_IDENTITY_VERSION}")
        if data.get("task_hash") != task_hash:
            raise CacheUnusableError("the cached task identity does not match this run")
        if data.get("has_files") and not self.has_cached_files(task_hash):
            raise CacheUnusableError("cached execution environment files are missing")

        try:
            state = CacheState.from_dict(data)
        except Exception as e:
            # Anything that cannot be reconstructed is a refusal, never a run failure - the
            # caller starts fresh. A narrower catch misses shape errors that surface deep in
            # deserialization (a "msgs" that is a string raises AttributeError, not ValueError).
            raise CacheUnusableError(f"state file is malformed: {e}") from e

        logger.info("Loaded cache state from %s", state_file)
        return state

    def restore_files(self, task_hash: str, dest_dir: Path) -> bool:
        """Restore cached files to the destination directory.

        Args:
            task_hash: Unique identifier for the task/cache.
            dest_dir: Destination directory (typically the new exec env temp dir).

        Returns:
            True if files were restored, False if no files cache exists.
        """
        files_dir = self._get_files_dir(task_hash)
        if not files_dir.exists():
            logger.debug("No cached files for task %s", task_hash)
            return False

        # Copy all files from cache to destination
        for item in files_dir.iterdir():
            dest_item = dest_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest_item)
            else:
                shutil.copytree(item, dest_item, dirs_exist_ok=True)

        logger.info("Restored cached files from %s to %s", files_dir, dest_dir)
        return True

    def clear_cache(self, task_hash: str) -> None:
        """Remove cache for a specific task.

        Called after successful completion to clean up.

        Args:
            task_hash: Unique identifier for the task/cache.
        """
        cache_dir = self._get_cache_dir(task_hash)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("Cleared cache for task %s", task_hash)

    def list_caches(self) -> list[str]:
        """List all available cache hashes.

        Returns:
            List of task hashes with existing caches.
        """
        if not self._cache_base_dir.exists():
            return []

        return [d.name for d in self._cache_base_dir.iterdir() if d.is_dir() and (d / "state.json").exists()]

    def get_cache_info(self, task_hash: str) -> dict | None:
        """Get metadata about a cache without fully loading it.

        Args:
            task_hash: Unique identifier for the task/cache.

        Returns:
            Dictionary with cache info (turn, timestamp, agent_name) or None.
        """
        state_file = self._get_state_file(task_hash)
        if not state_file.exists():
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "task_hash": task_hash,
                "turn": data.get("turn", 0),
                "timestamp": data.get("timestamp", ""),
                "agent_name": data.get("agent_name", ""),
                "has_files": self._get_files_dir(task_hash).exists(),
            }
        except (json.JSONDecodeError, KeyError):
            return None

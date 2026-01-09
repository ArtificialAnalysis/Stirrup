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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from stirrup.core.models import (
    AudioContentBlock,
    ChatMessage,
    ImageContentBlock,
    VideoContentBlock,
)

logger = logging.getLogger(__name__)

# Default cache directory relative to the project root
DEFAULT_CACHE_DIR = Path("~/.cache/stirrup/").expanduser()

# TypeAdapter for deserializing ChatMessage discriminated union
ChatMessageAdapter: TypeAdapter[ChatMessage] = TypeAdapter(ChatMessage)


def compute_task_hash(init_msgs: str | list[ChatMessage]) -> str:
    """Compute deterministic hash from initial messages for cache identification.

    Args:
        init_msgs: Either a string prompt or list of ChatMessage objects.

    Returns:
        First 12 characters of SHA256 hash (hex) for readability.
    """
    if isinstance(init_msgs, str):
        content = init_msgs
    else:
        # Serialize messages to JSON for hashing
        content = json.dumps(
            [serialize_message(msg) for msg in init_msgs],
            sort_keys=True,
            ensure_ascii=True,
        )

    hash_bytes = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hash_bytes[:12]


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

    turn: int
    """Current turn number (0-indexed) - resume will start from this turn."""

    run_metadata: dict[str, list[Any]]
    """Accumulated tool metadata from the run."""

    task_hash: str
    """Hash of the original init_msgs for verification on resume."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    """ISO timestamp when cache was created."""

    agent_name: str = ""
    """Name of the agent that created this cache."""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "msgs": serialize_messages(self.msgs),
            "full_msg_history": [serialize_messages(group) for group in self.full_msg_history],
            "turn": self.turn,
            "run_metadata": _serialize_run_metadata(self.run_metadata),
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
            turn=data["turn"],
            run_metadata=data["run_metadata"],
            task_hash=data["task_hash"],
            timestamp=data.get("timestamp", ""),
            agent_name=data.get("agent_name", ""),
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

        Uses atomic writes to prevent corrupted cache files if interrupted mid-write.

        Args:
            task_hash: Unique identifier for this task/cache.
            state: CacheState to persist.
            exec_env_dir: Optional path to execution environment temp directory.
                         If provided, all files will be copied to cache.
        """
        cache_dir = self._get_cache_dir(task_hash)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save state JSON using atomic write (write to temp file, then rename)
        state_file = self._get_state_file(task_hash)
        temp_file = state_file.with_suffix(".json.tmp")

        try:
            state_data = state.to_dict()
            logger.debug("Serialized cache state: turn=%d, msgs=%d", state.turn, len(state.msgs))

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            logger.debug("Wrote temp file: %s", temp_file)

            # Atomic rename (on POSIX systems)
            temp_file.replace(state_file)
            logger.info("Saved cache state to %s (turn %d)", state_file, state.turn)
        except Exception as e:
            logger.exception("Failed to save cache state: %s", e)
            # Try direct write as fallback
            try:
                logger.warning("Attempting direct write as fallback")
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                logger.info("Fallback write succeeded to %s", state_file)
            except Exception as e2:
                logger.exception("Fallback write also failed: %s", e2)
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            raise

        # Copy execution environment files if provided
        if exec_env_dir and exec_env_dir.exists():
            files_dir = self._get_files_dir(task_hash)
            if files_dir.exists():
                shutil.rmtree(files_dir)  # Clear existing files
            shutil.copytree(exec_env_dir, files_dir, dirs_exist_ok=True)
            logger.info("Saved execution environment files to %s", files_dir)

    def load_state(self, task_hash: str) -> CacheState | None:
        """Load cached state for a task hash.

        Args:
            task_hash: Unique identifier for the task/cache.

        Returns:
            CacheState if cache exists, None otherwise.
        """
        state_file = self._get_state_file(task_hash)
        if not state_file.exists():
            logger.debug("No cache found for task %s", task_hash)
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            state = CacheState.from_dict(data)
            logger.info("Loaded cache state from %s (turn %d)", state_file, state.turn)
            return state
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load cache for task %s: %s", task_hash, e)
            return None

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

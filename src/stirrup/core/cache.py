"""Cache module for persisting and resuming agent state.

Provides functionality to cache agent state (messages, run metadata, execution environment files)
on non-success exits and restore that state for resumption in new runs.
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from stirrup.core.models import ChatMessage

logger = logging.getLogger(__name__)

# Default cache directory relative to the project root
DEFAULT_CACHE_DIR = Path("~/.cache/stirrup/").expanduser()


def compute_task_hash[GenerationMetadataT: BaseModel | None](
    init_msgs: str | list[ChatMessage[GenerationMetadataT]],
) -> str:
    """Compute deterministic hash from initial messages for cache identification.

    Args:
        init_msgs: Either a string prompt or list of ChatMessage objects.

    Returns:
        First 12 characters of SHA256 hash (hex) for readability.
    """
    if isinstance(init_msgs, str):
        content = init_msgs
    else:
        content = json.dumps(
            [msg.model_dump(mode="json") for msg in init_msgs],
            sort_keys=True,
            ensure_ascii=True,
        )

    hash_bytes = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hash_bytes[:12]


def serialize_message[GenerationMetadataT: BaseModel | None](msg: ChatMessage[GenerationMetadataT]) -> dict:
    """Serialize a ChatMessage to JSON-compatible format."""
    return msg.model_dump(mode="json")

def deserialize_message(
    data: dict,
    generation_metadata_type: type[BaseModel] | None = None,
) -> ChatMessage[BaseModel | None]:
    """Deserialize a ChatMessage from JSON format."""
    if generation_metadata_type is None and data.get("metadata") in (None, {}):
        data["metadata"] = None

    if generation_metadata_type is None:
        adapter: TypeAdapter[Any] = TypeAdapter(ChatMessage[None])
    else:
        adapter = TypeAdapter(ChatMessage[generation_metadata_type])  # ty: ignore[invalid-type-form]
    return adapter.validate_python(data)

class CacheState[GenerationMetadataT: BaseModel | None](BaseModel):
    """Serializable state for resuming an agent run.

    Captures all necessary state to resume execution from a specific turn.
    """

    msgs: list[ChatMessage[GenerationMetadataT]]
    """Current conversation messages in the active run loop."""

    full_msg_history: list[list[ChatMessage[GenerationMetadataT]]]
    """Groups of messages (separated when context summarization occurs)."""

    turn: int
    """Current turn number (0-indexed) - resume will start from this turn."""

    run_metadata: dict[str, list[Any]]
    """Accumulated tool metadata from the run."""

    task_hash: str
    """Hash of the original init_msgs for verification on resume."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    """ISO timestamp when cache was created."""

    agent_name: str = ""
    """Name of the agent that created this cache."""

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

    def save_state[GenerationMetadataT: BaseModel | None](
        self,
        task_hash: str,
        state: CacheState[GenerationMetadataT],
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
            logger.debug("Serialized cache state: turn=%d, msgs=%d", state.turn, len(state.msgs))

            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))
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
                    f.write(state.model_dump_json(indent=2))
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

    def load_state(
        self,
        task_hash: str,
        generation_metadata_type: type[BaseModel] | None = None,
    ) -> CacheState[BaseModel | None] | None:
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
                raw_json = f.read()
            if generation_metadata_type is None:
                state_model: type[BaseModel] = CacheState[None]
            else:
                state_model = CacheState[generation_metadata_type]  # ty: ignore[invalid-type-form]
            state = state_model.model_validate_json(raw_json)
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

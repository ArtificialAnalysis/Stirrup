import contextvars
import copy
import glob as glob_module
import inspect
import logging
import re
import signal
import threading
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from itertools import chain, takewhile
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import TracebackType
from typing import Annotated, Any, cast, overload

import anyio
from pydantic import BaseModel, Field, ValidationError

from stirrup.constants import (
    AGENT_MAX_TURNS,
    CONTEXT_SUMMARIZATION_CUTOFF,
    TURNS_REMAINING_WARNING_THRESHOLD,
)
from stirrup.core.cache import (
    CacheFileIdentity,
    CacheManager,
    CacheState,
    build_tool_registry,
    compute_task_hash,
)
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    SubAgentMetadata,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
    TurnWarningMessage,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER, MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE
from stirrup.skills import SkillMetadata, format_skills_section, load_skills_metadata
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.code_backends.base import CodeExecToolProvider, UploadedFile
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.tools.finish import FinishParams as SimpleFinishParams
from stirrup.utils.logging import AgentLogger, AgentLoggerBase

_ACTIVE_SESSION_RESERVATIONS: dict[tuple[str, int | str], object] = {}
_SESSION_RESERVATIONS_LOCK = threading.Lock()
_ACTIVE_SIGINT_SESSION_COUNT = 0
_ORIGINAL_SIGINT_HANDLER: Any = None

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-session state for resource lifecycle management.

    Kept minimal - only contains resources that need async lifecycle management
    (exit_stack, exec_env) and session-specific configuration (output_dir).

    Tool availability is managed via Agent._active_tools (instance-scoped),
    and run results are stored on the agent instance temporarily.

    For subagent file transfer:
    - parent_exec_env: Reference to the parent's exec env (for cross-env transfers)
    - depth: Agent depth (0 = root, >0 = subagent)
    - output_dir: For root agent, this is a local filesystem path. For subagents,
      this is a path within the parent's exec env.
    - exec_env_owned: Whether this session owns the exec_env and should clean it up.
      When share_parent_exec_env=True, the subagent borrows the parent's exec_env
      and exec_env_owned=False to prevent cleanup on subagent exit.
    """

    exit_stack: AsyncExitStack
    owner: "SessionAgent[Any, Any] | None" = None
    exec_env: CodeExecToolProvider | None = None
    output_dir: str | None = None  # String path (contextual: local for root, in parent env for subagent)
    parent_exec_env: CodeExecToolProvider | None = None
    depth: int = 0
    exec_env_owned: bool = True  # Whether this session owns (and should cleanup) the exec_env
    uploaded_file_paths: list[str] = field(default_factory=list)  # Paths of files uploaded to exec_env
    cache_input_files: list[CacheFileIdentity] = field(default_factory=list)
    cache_skill_files: list[CacheFileIdentity] = field(default_factory=list)
    skills_metadata: list[SkillMetadata] = field(default_factory=list)  # Loaded skills metadata
    logger: AgentLoggerBase | None = None  # Logger for pause/resume during user input


_SESSION_STATE: contextvars.ContextVar[SessionState | None] = contextvars.ContextVar("session_state", default=None)

__all__ = [
    "Agent",
    "SessionAgent",
    "SubAgentParams",
]

LOGGER = logging.getLogger(__name__)


def _handle_interrupt_signal(_signum: int, _frame: object) -> None:
    """Turn SIGINT into an exception so active sessions can cache on exit."""
    raise KeyboardInterrupt("Agent interrupted - state will be cached")


def _install_interrupt_handler() -> bool:
    """Install one process handler for main-thread root sessions."""
    global _ACTIVE_SIGINT_SESSION_COUNT, _ORIGINAL_SIGINT_HANDLER

    if threading.current_thread() is not threading.main_thread():
        return False
    if _ACTIVE_SIGINT_SESSION_COUNT == 0:
        _ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handle_interrupt_signal)
    _ACTIVE_SIGINT_SESSION_COUNT += 1
    return True


def _release_interrupt_handler() -> None:
    """Restore the process handler after the final root session exits."""
    global _ACTIVE_SIGINT_SESSION_COUNT, _ORIGINAL_SIGINT_HANDLER

    _ACTIVE_SIGINT_SESSION_COUNT -= 1
    if _ACTIVE_SIGINT_SESSION_COUNT == 0:
        signal.signal(signal.SIGINT, _ORIGINAL_SIGINT_HANDLER)
        _ORIGINAL_SIGINT_HANDLER = None


def _reserve_custom_session_resources(resources: list[object]) -> None:
    """Claim custom lifecycle objects before any session resource is entered."""
    with _SESSION_RESERVATIONS_LOCK:
        for resource in resources:
            if ("custom", id(resource)) in _ACTIVE_SESSION_RESERVATIONS:
                raise RuntimeError(
                    f"Overlapping sessions cannot share configured {type(resource).__name__}. "
                    "Custom providers, custom loggers, and subclasses of built-ins support sequential sessions only."
                )
        for resource in resources:
            _ACTIVE_SESSION_RESERVATIONS[("custom", id(resource))] = resource


def _release_custom_session_resources(resources: list[object]) -> None:
    with _SESSION_RESERVATIONS_LOCK:
        for resource in resources:
            _ACTIVE_SESSION_RESERVATIONS.pop(("custom", id(resource)), None)


class _TaskCacheOwnership:
    """Reserve task caches and guard every cache operation by ownership."""

    def __init__(self) -> None:
        self._owner = object()
        self._task_hashes: set[str] = set()

    def reserve(self, task_hash: str) -> None:
        reservation = ("cache", task_hash)
        with _SESSION_RESERVATIONS_LOCK:
            active_owner = _ACTIVE_SESSION_RESERVATIONS.get(reservation)
            if active_owner is self._owner:
                return
            if active_owner is not None:
                raise RuntimeError(
                    "Concurrent root runs cannot use the same task cache. "
                    "Wait for the active cache-enabled or resuming run to finish."
                )
            _ACTIVE_SESSION_RESERVATIONS[reservation] = self._owner
            self._task_hashes.add(task_hash)

    def owns(self, task_hash: str) -> bool:
        with _SESSION_RESERVATIONS_LOCK:
            return (
                task_hash in self._task_hashes and _ACTIVE_SESSION_RESERVATIONS.get(("cache", task_hash)) is self._owner
            )

    def _require(self, task_hash: str) -> None:
        if not self.owns(task_hash):
            raise RuntimeError(f"Task cache operation requires reservation ownership for {task_hash}")

    def save_state(
        self,
        cache_manager: CacheManager,
        task_hash: str,
        state: CacheState,
        exec_env_dir: Path | None,
    ) -> None:
        self._require(task_hash)
        cache_manager.save_state(task_hash, state, exec_env_dir)

    def load_state(
        self,
        cache_manager: CacheManager,
        task_hash: str,
        *,
        restore_files_to: Path | None,
        finish_tools: Mapping[str, Tool[Any, Any]],
    ) -> CacheState | None:
        self._require(task_hash)
        return cache_manager.load_state(
            task_hash,
            restore_files_to=restore_files_to,
            finish_tools=finish_tools,
        )

    def clear_cache_if_owned(self, cache_manager: CacheManager, task_hash: str) -> bool:
        if not self.owns(task_hash):
            return False
        cache_manager.clear_cache(task_hash)
        return True

    def release(self) -> None:
        with _SESSION_RESERVATIONS_LOCK:
            for task_hash in self._task_hashes:
                reservation = ("cache", task_hash)
                if _ACTIVE_SESSION_RESERVATIONS.get(reservation) is self._owner:
                    _ACTIVE_SESSION_RESERVATIONS.pop(reservation)
            self._task_hashes.clear()


def _record_session_cleanup_failure(
    primary_exception: BaseException | None,
    phase: str,
    cleanup_failure: BaseException,
) -> BaseException:
    """Keep the first failure and record later cleanup failures on it."""
    if primary_exception is None:
        return cleanup_failure
    primary_exception.add_note(
        f"{phase} failed during session cleanup: {type(cleanup_failure).__name__}: {cleanup_failure}"
    )
    return primary_exception


def _detach_builtin_provider(provider: ToolProvider, replacements: dict[int, ToolProvider]) -> ToolProvider | None:
    """Return a fresh runtime owner for an exact built-in provider."""
    provider_type = type(provider)

    from stirrup.tools.view_image import ViewImageToolProvider
    from stirrup.tools.web import WebToolProvider

    if isinstance(provider, LocalCodeExecToolProvider) and provider_type is LocalCodeExecToolProvider:
        detached = copy.copy(provider)
        detached._allowed_commands = (  # noqa: SLF001
            list(provider._allowed_commands) if provider._allowed_commands is not None else None  # noqa: SLF001
        )
        detached._compiled_allowed = (  # noqa: SLF001
            list(provider._compiled_allowed) if provider._compiled_allowed is not None else None  # noqa: SLF001
        )
        detached._temp_dir = None  # noqa: SLF001
        return detached
    if isinstance(provider, WebToolProvider) and provider_type is WebToolProvider:
        detached = copy.copy(provider)
        detached._client = None  # noqa: SLF001
        return detached
    if isinstance(provider, ViewImageToolProvider) and provider_type is ViewImageToolProvider:
        detached = copy.copy(provider)
        configured_exec_env = provider._exec_env  # noqa: SLF001
        if configured_exec_env is not None:
            replacement = replacements.get(id(configured_exec_env))
            detached._exec_env = (  # noqa: SLF001
                replacement if isinstance(replacement, CodeExecToolProvider) else configured_exec_env
            )
        return detached

    if provider_type.__module__ == "stirrup.tools.code_backends.docker":
        from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

        if isinstance(provider, DockerCodeExecToolProvider) and provider_type is DockerCodeExecToolProvider:
            detached = copy.copy(provider)
            detached._allowed_commands = (  # noqa: SLF001
                list(provider._allowed_commands) if provider._allowed_commands is not None else None  # noqa: SLF001
            )
            detached._compiled_allowed = (  # noqa: SLF001
                list(provider._compiled_allowed) if provider._compiled_allowed is not None else None  # noqa: SLF001
            )
            detached._env_vars = (  # noqa: SLF001
                list(provider._env_vars) if provider._env_vars is not None else None  # noqa: SLF001
            )
            detached._temp_dir = None  # noqa: SLF001
            detached._client = None  # noqa: SLF001
            detached._container = None  # noqa: SLF001
            return detached
    elif provider_type.__module__ == "stirrup.tools.code_backends.e2b":
        from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

        if isinstance(provider, E2BCodeExecToolProvider) and provider_type is E2BCodeExecToolProvider:
            detached = copy.copy(provider)
            detached._allowed_commands = (  # noqa: SLF001
                list(provider._allowed_commands) if provider._allowed_commands is not None else None  # noqa: SLF001
            )
            detached._compiled_allowed = (  # noqa: SLF001
                list(provider._compiled_allowed) if provider._compiled_allowed is not None else None  # noqa: SLF001
            )
            detached._sandbox_kwargs = dict(provider._sandbox_kwargs)  # noqa: SLF001
            for key in ("envs", "metadata"):
                value = detached._sandbox_kwargs.get(key)  # noqa: SLF001
                if isinstance(value, dict):
                    detached._sandbox_kwargs[key] = dict(value)  # noqa: SLF001
            detached._sbx = None  # noqa: SLF001
            return detached
    elif provider_type.__module__ == "stirrup.tools.mcp":
        from stirrup.tools.mcp import MCPToolProvider

        if isinstance(provider, MCPToolProvider) and provider_type is MCPToolProvider:
            detached = copy.copy(provider)
            detached._config = provider._config.model_copy(deep=True)  # noqa: SLF001
            detached._server_names = (  # noqa: SLF001
                list(provider._server_names) if provider._server_names is not None else None  # noqa: SLF001
            )
            detached._servers = {}  # noqa: SLF001
            detached._tools = {}  # noqa: SLF001
            detached._exit_stack = None  # noqa: SLF001
            return detached
    elif provider_type.__module__ == "stirrup.tools.browser_use":
        from stirrup.tools.browser_use import BrowserUseToolProvider

        if isinstance(provider, BrowserUseToolProvider) and provider_type is BrowserUseToolProvider:
            detached = copy.copy(provider)
            detached._extra_args = (  # noqa: SLF001
                list(provider._extra_args) if provider._extra_args is not None else None  # noqa: SLF001
            )
            detached._session = None  # noqa: SLF001
            return detached

    return None


def _reset_logger_state(session_logger: AgentLoggerBase) -> None:
    session_logger.name = "agent"
    session_logger.model = None
    session_logger.max_turns = None
    session_logger.depth = 0
    session_logger.finish_params = None
    session_logger.run_metadata = None
    session_logger.output_dir = None


def _detach_logger(configured_logger: AgentLoggerBase) -> tuple[AgentLoggerBase, bool]:
    """Detach exact built-in logger state; identify custom sequential owners."""
    logger_type = type(configured_logger)
    if isinstance(configured_logger, AgentLogger) and logger_type is AgentLogger:
        detached = copy.copy(configured_logger)
        _reset_logger_state(detached)
        detached._current_step = 0  # noqa: SLF001
        detached._tool_calls = 0  # noqa: SLF001
        detached._input_tokens = 0  # noqa: SLF001
        detached._output_tokens = 0  # noqa: SLF001
        detached._live = None  # noqa: SLF001
        return detached, False

    if logger_type.__module__ == "stirrup.integrations.slack.slack":
        from stirrup.integrations.slack.slack import SlackLogger

        if isinstance(configured_logger, SlackLogger) and logger_type is SlackLogger:
            detached = copy.copy(configured_logger)
            _reset_logger_state(detached)
            return detached, False

    return configured_logger, True


def _num_turns_remaining_msg(number_of_turns_remaining: int) -> TurnWarningMessage:
    """Create a user message warning the agent about remaining turns before max_turns is reached."""
    if number_of_turns_remaining == 1:
        return TurnWarningMessage(content="This is the last turn. Please finish the task by calling a finish tool.")
    return TurnWarningMessage(
        content=f"You have {number_of_turns_remaining} turns remaining to complete the task. Please continue. Remember you will need a separate turn to call a finish tool.",
    )


def _as_posix_path(path: str | Path) -> PurePosixPath:
    return PurePosixPath(str(path).replace("\\", "/"))


async def _fingerprint_uploaded_files(
    exec_env: CodeExecToolProvider,
    uploaded_files: list[UploadedFile],
) -> list[CacheFileIdentity]:
    """Fingerprint bytes in the execution environment, never mutable source files.

    Identity follows the destination layout a backend chose. A backend that relocates the same
    logical inputs therefore changes identity, which is a code change and should invalidate caches.
    """
    identities: list[CacheFileIdentity] = []
    for uploaded_file in uploaded_files:
        dest_path = _as_posix_path(uploaded_file.dest_path)
        # Container backends report absolute destinations, but an identity name must be relative.
        relative_name = PurePosixPath(*dest_path.parts[1:]) if dest_path.is_absolute() else dest_path
        content = await exec_env.read_file_bytes(uploaded_file.dest_path)
        identities.append(CacheFileIdentity.from_content(relative_name.as_posix(), content))
    return identities


async def _fingerprint_shared_inputs(
    exec_env: CodeExecToolProvider,
    input_paths: list[str | Path],
) -> list[CacheFileIdentity]:
    """Fingerprint subagent inputs already present in a shared environment."""
    identities: list[CacheFileIdentity] = []
    for input_path_value in input_paths:
        input_path = _as_posix_path(input_path_value)
        contextual_path = PurePosixPath(*input_path.parts[1:]) if input_path.is_absolute() else input_path
        if await exec_env.is_directory(input_path.as_posix()):
            for relative_file in await exec_env.list_files(input_path.as_posix()):
                relative_file_path = _as_posix_path(relative_file)
                content = await exec_env.read_file_bytes((input_path / relative_file_path).as_posix())
                identities.append(
                    CacheFileIdentity.from_content((contextual_path / relative_file_path).as_posix(), content)
                )
        else:
            content = await exec_env.read_file_bytes(input_path.as_posix())
            identities.append(CacheFileIdentity.from_content(contextual_path.as_posix(), content))
    return identities


def _handle_text_only_tool_responses(tool_messages: list[ToolMessage]) -> tuple[list[ToolMessage], list[UserMessage]]:
    """Return an idempotent text-only representation without mutating inputs."""
    transformed_tool_messages: list[ToolMessage] = []
    user_messages: list[UserMessage] = []
    for tool_message in tool_messages:
        transformed_content = tool_message.content
        if isinstance(tool_message.content, list):
            content_blocks: list[str | ImageContentBlock] = []
            found_image = False
            for block in tool_message.content:
                if isinstance(block, ImageContentBlock):
                    found_image = True
                    user_messages.append(
                        UserMessage(content=[f"Here is the image for tool call {tool_message.tool_call_id}", block])
                    )
                    content_blocks.append(
                        f"Done! The User will provide the image for tool call {tool_message.tool_call_id}"
                    )
                elif isinstance(block, str):
                    content_blocks.append(block)
                else:
                    raise NotImplementedError(f"Unsupported content block: {type(block)}")
            if found_image:
                transformed_content = content_blocks
        transformed_tool_messages.append(
            tool_message
            if transformed_content is tool_message.content
            else tool_message.model_copy(update={"content": transformed_content})
        )
    return transformed_tool_messages, user_messages


def _normalize_finish_tools[FinishParams: BaseModel, FinishMeta](
    finish_tool: Tool[FinishParams, FinishMeta] | list[Tool] | None,
) -> list[Tool[Any, Any]]:
    """Normalize the finish-tool option before centralized runtime validation."""
    if finish_tool is None:
        return [SIMPLE_FINISH_TOOL]
    if isinstance(finish_tool, list):
        if not finish_tool:
            raise ValueError("finish_tool list cannot be empty")
        return finish_tool
    return [finish_tool]


def _get_total_token_usage(messages: list[list[ChatMessage]]) -> list[TokenUsage]:
    """
    Returns a list of TokenUsage objects aggregated from all AssistantMessage
    instances across the provided grouped message history.

    Args:
        messages: A list where each item is a list of ChatMessage objects representing a segment
                  or turn group of the conversation history.

    Returns:
        List of TokenUsage corresponding to each AssistantMessage in the flattened conversation history.
    """
    return [msg.token_usage for msg in chain.from_iterable(messages) if isinstance(msg, AssistantMessage)]


def _get_tool_durations(messages: list[list[ChatMessage]]) -> dict[str, list[float]]:
    """Collect tool execution durations grouped by tool name from message history."""
    durations: dict[str, list[float]] = {}
    for msg in chain.from_iterable(messages):
        if isinstance(msg, ToolMessage) and msg.name and msg.tool_duration is not None:
            durations.setdefault(msg.name, []).append(msg.tool_duration)
    return durations


def _get_turn_count(full_msg_history: list[list[ChatMessage]], messages: list[ChatMessage]) -> int:
    """Count accepted assistant turns still present in history."""
    all_messages = chain(chain.from_iterable(full_msg_history), messages)
    return sum(1 for msg in all_messages if isinstance(msg, AssistantMessage))


def _get_model_speed_stats(messages: list[list[ChatMessage]], model_slug: str) -> dict[str, float | int | str]:
    """Compute speed stats for this agent's model from AssistantMessages.

    Returns a flat dict with model_slug, num_calls, output_tokens, duration, e2e_otps.
    Returns empty dict if no timed messages found.
    """
    num_calls = 0
    output_tokens = 0
    duration = 0.0
    for msg in chain.from_iterable(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        if msg.request_start_time is None or msg.request_end_time is None:
            continue
        msg_duration = msg.request_end_time - msg.request_start_time
        if msg_duration <= 0:
            continue
        num_calls += 1
        output_tokens += msg.token_usage.output
        duration += msg_duration
    if num_calls == 0:
        return {}
    return {
        "model_slug": model_slug,
        "num_calls": num_calls,
        "output_tokens": output_tokens,
        "duration": duration,
        "e2e_otps": output_tokens / duration if duration > 0 else 0.0,
    }


class SubAgentParams(BaseModel):
    """Parameters for sub-agent tool invocation."""

    task: Annotated[str, Field(description="The task/prompt for the sub-agent to complete")]
    input_files: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of file paths to upload to the sub-agent's execution environment. "
            "Use paths from output_dir (e.g., files saved by previous sub-agents).",
        ),
    ]


DEFAULT_SUB_AGENT_DESCRIPTION = "A sub agent that can be used to handle a contained, specific task."

# Agent name validation pattern: alphanumeric, underscores, hyphens, 1-128 chars
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class Agent[FinishParams: BaseModel, FinishMeta]:
    """Agent that executes tool-using loops with automatic context management.

    Runs up to max_turns iterations of: LLM generation → tool execution → message accumulation.
    When conversation history exceeds context window limits, older messages are automatically
    condensed into a summary to preserve working memory.

    The Agent can be used as an async context manager via .session() for automatic tool
    lifecycle management, logging, and file saving:

        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        # Create client and agent
        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(client=client, name="assistant")

        async with agent.session(output_dir="./output") as session:
            finish_params, history, metadata = await session.run("Your task here")
    """

    _parent_session_state: SessionState | None
    _session_state: SessionState | None
    _session_state_token: contextvars.Token[SessionState | None] | None
    _interrupt_handler_installed: bool
    _custom_session_resources: list[object]
    _custom_resources_reserved: bool
    _task_cache_ownership: _TaskCacheOwnership

    @overload
    def __init__(
        self: "Agent[SimpleFinishParams, ToolUseCountMetadata]",
        client: LLMClient,
        name: str,
        *,
        max_turns: int = ...,
        system_prompt: str | None = ...,
        tools: list[Tool | ToolProvider] | None = ...,
        finish_tool: None = None,
        context_summarization_cutoff: float = ...,
        turns_remaining_warning_threshold: int = ...,
        run_sync_in_thread: bool = ...,
        text_only_tool_responses: bool = ...,
        block_successive_assistant_messages: bool = ...,
        recover_from_context_overflow: bool = ...,
        share_parent_exec_env: bool = ...,
        logger: AgentLoggerBase | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "Agent[FinishParams, FinishMeta]",
        client: LLMClient,
        name: str,
        *,
        max_turns: int = ...,
        system_prompt: str | None = ...,
        tools: list[Tool | ToolProvider] | None = ...,
        finish_tool: Tool[FinishParams, FinishMeta],
        context_summarization_cutoff: float = ...,
        turns_remaining_warning_threshold: int = ...,
        run_sync_in_thread: bool = ...,
        text_only_tool_responses: bool = ...,
        block_successive_assistant_messages: bool = ...,
        recover_from_context_overflow: bool = ...,
        share_parent_exec_env: bool = ...,
        logger: AgentLoggerBase | None = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "Agent[BaseModel, Any]",
        client: LLMClient,
        name: str,
        *,
        max_turns: int = ...,
        system_prompt: str | None = ...,
        tools: list[Tool | ToolProvider] | None = ...,
        finish_tool: list[Tool],
        context_summarization_cutoff: float = ...,
        turns_remaining_warning_threshold: int = ...,
        run_sync_in_thread: bool = ...,
        text_only_tool_responses: bool = ...,
        block_successive_assistant_messages: bool = ...,
        recover_from_context_overflow: bool = ...,
        share_parent_exec_env: bool = ...,
        logger: AgentLoggerBase | None = ...,
    ) -> None: ...

    def __init__(
        self,
        client: LLMClient,
        name: str,
        *,
        max_turns: int = AGENT_MAX_TURNS,
        system_prompt: str | None = None,
        tools: list[Tool | ToolProvider] | None = None,
        finish_tool: Tool[FinishParams, FinishMeta] | list[Tool] | None = None,
        # Agent options
        context_summarization_cutoff: float = CONTEXT_SUMMARIZATION_CUTOFF,
        turns_remaining_warning_threshold: int = TURNS_REMAINING_WARNING_THRESHOLD,
        run_sync_in_thread: bool = True,
        text_only_tool_responses: bool = True,
        block_successive_assistant_messages: bool = True,
        recover_from_context_overflow: bool = True,
        # Subagent options
        share_parent_exec_env: bool = False,
        # Logging
        logger: AgentLoggerBase | None = None,
    ) -> None:
        """Initialize the agent with an LLM client and configuration.

        Args:
            client: LLM client for generating responses. Use ChatCompletionsClient for
                    OpenAI/OpenAI-compatible APIs, or LiteLLMClient for other providers.
            name: Name of the agent (used for logging purposes)
            max_turns: Maximum number of turns before stopping
            system_prompt: System prompt to prepend to all runs (when using string prompts)
            tools: List of Tools and/or ToolProviders available to the agent.
                   If None, uses DEFAULT_TOOLS. ToolProviders are automatically
                   set up and torn down by Agent.session().
                   Use [*DEFAULT_TOOLS, extra_tool] to extend defaults.
            finish_tool: Tool or list of Tools used to signal task completion.
                         Defaults to SIMPLE_FINISH_TOOL. If a list is provided,
                         a successful call to any listed tool ends the run.
            context_summarization_cutoff: Fraction of context window (0-1) at which to trigger summarization
            run_sync_in_thread: Execute synchronous tool executors in a separate thread
            text_only_tool_responses: Extract images from tool responses as separate user messages
            block_successive_assistant_messages: If True (default), automatically inject a continue
                                               message when assistant responds without tool calls to
                                               prevent successive assistant messages.
            recover_from_context_overflow: If True (default), summarize the largest fitting older
                                           complete-turn prefix and retry when the model reports a
                                           context overflow. If no safe prefix fits, the context
                                           error is raised.
            share_parent_exec_env: When True and used as a subagent, share the parent's code
                                   execution environment instead of creating a new one. This
                                   provides better performance (no file copying) and allows
                                   the subagent to see all files in the parent's environment.
                                   Only effective when the agent is used as a subagent via to_tool().
                                   Custom ViewImageToolProvider/backend pairs are unsupported.
            logger: Optional logger instance. If None, creates AgentLogger() internally.

        """
        # Validate agent name
        if not AGENT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid agent name '{name}'. "
                "Agent names must match pattern '^[a-zA-Z0-9_-]{1,128}$' "
                "(alphanumeric, underscores, hyphens only, 1-128 characters)."
            )

        self._client: LLMClient = client
        self._name = name
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._tools = tools if tools is not None else DEFAULT_TOOLS
        finish_tools = _normalize_finish_tools(finish_tool)
        self._context_summarization_cutoff = context_summarization_cutoff
        self._turns_remaining_warning_threshold = turns_remaining_warning_threshold
        self._run_sync_in_thread = run_sync_in_thread
        self._text_only_tool_responses = text_only_tool_responses
        self._block_successive_assistant_messages = block_successive_assistant_messages
        self._recover_from_context_overflow = recover_from_context_overflow
        self._share_parent_exec_env = share_parent_exec_env

        # Logger (can be passed in or created here)
        self._logger: AgentLoggerBase = logger if logger is not None else AgentLogger()

        # Session configuration (set during session(), used in __aenter__)
        self._pending_output_dir: Path | None = None
        self._pending_input_files: str | Path | list[str | Path] | None = None
        self._pending_skills_dir: Path | None = None
        self._resume: bool = False
        self._clear_cache_on_success: bool = True
        self._cache_on_interrupt: bool = True

        # Eagerly register static and finish tools; provider tools are resolved on session entry.
        static_tools = [tool for tool in self._tools if isinstance(tool, Tool)]
        self._active_tools, self._finish_tools, self._tool_definitions = build_tool_registry(
            static_tools,
            finish_tools,
        )
        self._has_tool_providers = any(isinstance(t, ToolProvider) for t in self._tools)

        self._last_finish_params: Any = None  # FinishParams type parameter
        self._last_run_metadata: dict[str, list[Any]] = {}
        self._transferred_paths: list[str] = []  # Paths transferred to parent (for subagents)

        # Cache state for resumption (set during run(), used in __aexit__ for caching on interrupt)
        self._current_task_hash: str | None = None
        self._current_run_state: CacheState | None = None

    @property
    def name(self) -> str:
        """The name of this agent."""
        return self._name

    @property
    def client(self) -> LLMClient:
        """The LLM client used by this agent."""
        return self._client

    @property
    def tools(self) -> dict[str, Tool]:
        """Currently active tools (available after entering session context)."""
        return self._active_tools

    @property
    def finish_tool(self) -> Tool[FinishParams, FinishMeta]:
        """The single finish tool used to signal task completion.

        Raises ValueError if multiple finish tools are configured; use ``finish_tools`` instead.
        """
        if len(self._finish_tools) != 1:
            raise ValueError(
                "Agent has multiple finish tools configured. Use finish_tools to access all finish tools by name."
            )
        return next(iter(self._finish_tools.values()))

    @property
    def finish_tools(self) -> dict[str, Tool]:
        """All finish tools keyed by name."""
        return self._finish_tools

    @property
    def logger(self) -> AgentLoggerBase:
        """The logger instance used by this agent."""
        return self._logger

    def session(
        self,
        output_dir: Path | str | None = None,
        input_files: str | Path | list[str | Path] | None = None,
        skills_dir: Path | str | None = None,
        resume: bool = False,
        clear_cache_on_success: bool = True,
        cache_on_interrupt: bool = True,
    ) -> "SessionAgent[FinishParams, FinishMeta]":
        """Create a detached runtime for use as an async context manager.

        Args:
            output_dir: Directory to save output files from finish_params.paths
            input_files: Files to upload to the execution environment at session start.
                        Accepts a single path or list of paths. Supports:
                        - File paths (str or Path)
                        - Directory paths (uploaded recursively)
                        - Glob patterns (e.g., "data/*.csv", "**/*.py")
                        Raises ValueError if no CodeExecToolProvider is configured
                        or if a glob pattern matches no files.
            skills_dir: Directory containing skill definitions to load and make available
                       to the agent. Skills are uploaded to the execution environment
                       and their metadata is included in the system prompt.
            resume: If True, attempt to resume from cached state if available.
                   The cache is identified by hashing the init_msgs passed to run().
                   Cached state includes message history, current turn, and execution
                   environment files from a previous interrupted run.
            clear_cache_on_success: If True (default), automatically clear an owned
                                   cache when the agent completes successfully. Set to
                                   False to preserve caches for inspection or debugging.
            cache_on_interrupt: If True (default), set up a SIGINT handler to cache
                               state on Ctrl+C. Set to False when running agents in
                               threads or subprocesses where signal handlers cannot
                               be registered from non-main threads. Concurrent root runs
                               in one process must use different task identities while
                               caching or resumption is enabled.

        Returns:
            A fresh SessionAgent for use with `async with agent.session(...) as session:`

        Example:
            async with agent.session(output_dir="./output", input_files="data/*.csv") as session:
                result = await session.run("Analyze the CSV files")

        Note:
            Exact built-in providers and loggers support overlapping sessions.
            Custom lifecycle objects support sequential sessions only.

        """
        return SessionAgent.from_agent(
            self,
            output_dir=output_dir,
            input_files=input_files,
            skills_dir=skills_dir,
            resume=resume,
            clear_cache_on_success=clear_cache_on_success,
            cache_on_interrupt=cache_on_interrupt,
        )

    def _resolve_input_files(self, input_files: str | Path | list[str | Path]) -> list[Path]:
        """Resolve input file paths, expanding globs and normalizing to Path objects.

        Args:
            input_files: Single path or list of paths (strings, Paths, or glob patterns)

        Returns:
            List of resolved Path objects

        Raises:
            ValueError: If a glob pattern matches no files

        """
        # Normalize to list
        paths = [input_files] if isinstance(input_files, str | Path) else list(input_files)

        resolved: list[Path] = []
        for path in paths:
            path_str = str(path)

            # Check if it looks like a glob pattern
            if any(c in path_str for c in ("*", "?", "[")):
                # Expand glob pattern
                matches = glob_module.glob(path_str, recursive=True)
                if not matches:
                    raise ValueError(f"Glob pattern '{path_str}' matched no files")
                resolved.extend(Path(m) for m in matches)
            else:
                # Regular path - add as-is (upload_files will handle non-existent)
                resolved.append(Path(path))

        return resolved

    def _collect_all_tools(self) -> list[Tool | ToolProvider]:
        """Collect all tools from this agent and any sub-agents recursively."""
        all_tools: list[Tool | ToolProvider] = list(self._tools)

        for tool in self._tools:
            # Check if this tool wraps a sub-agent (created via to_tool())
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                # Check if the executor is a closure that captured an Agent
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                # Recursively collect from sub-agent
                                all_tools.extend(cell_contents._collect_all_tools())  # noqa: SLF001
                        except ValueError:
                            # cell_contents can raise ValueError if empty
                            pass

        return all_tools

    def _collect_warnings(self) -> list[str]:
        """Collect warnings about agent configuration."""
        warnings = []

        # Collect all tools including from sub-agents
        all_tools = self._collect_all_tools()

        # Check for LocalCodeExecToolProvider (security risk) - only in top-level agent
        for tool in self._tools:
            if isinstance(tool, LocalCodeExecToolProvider):
                warnings.append(
                    "LocalCodeExecToolProvider can access your local filesystem. "
                    "Consider using DockerCodeExecToolProvider or E2BCodeExecToolProvider for sandboxed execution.",
                )
                break

        # Check for missing default tools (across entire agent tree)
        for default_tool in DEFAULT_TOOLS:
            default_type = type(default_tool)

            # Special case: For code exec providers, check if ANY CodeExecToolProvider is present
            if isinstance(default_tool, CodeExecToolProvider):
                found = any(isinstance(t, CodeExecToolProvider) for t in all_tools)
            else:
                found = any(isinstance(t, default_type) for t in all_tools)

            if not found:
                warnings.append(f"Missing default tool: {default_type.__name__}")

        # Check for code execution tool per-agent (including sub-agents)
        agents_without_code_exec = self._collect_agents_without_code_exec()
        warnings.extend(
            f"Agent '{agent_name}' has no code execution tool. It will not be able to save files to the output directory."
            for agent_name in agents_without_code_exec
        )

        # Check for code execution without output directory
        state = _SESSION_STATE.get(None)
        if state and state.exec_env and not state.output_dir:
            warnings.append(
                "Code execution environment is configured but no output_dir is set. "
                "Files created by the agent will be lost when the session ends.",
            )

        return warnings

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt: base + input files + user instructions.

        Returns:
            Complete system prompt string combining base prompt, input file listing,
            and user's custom system_prompt (if provided).
        """
        from stirrup.prompts import BASE_SYSTEM_PROMPT_TEMPLATE

        parts: list[str] = []

        # Base prompt with max_turns
        parts.append(BASE_SYSTEM_PROMPT_TEMPLATE.format(max_turns=self._max_turns))

        # User interaction guidance based on whether user_input tool is available
        if "user_input" in self._active_tools:
            parts.append(
                " You have access to the user_input tool which allows you to ask the user "
                "questions when you need clarification or are uncertain about something."
            )
        else:
            parts.append(" You are not able to interact with the user during the task.")

        # Input files section (if any were uploaded)
        state = _SESSION_STATE.get(None)
        if state and state.uploaded_file_paths:
            files_section = "\n\nThe following input files have been provided for this task:"
            for file_path in state.uploaded_file_paths:
                files_section += f"\n- {file_path}"
            parts.append(files_section)

        # Skills section (if skills were loaded)
        if state and state.skills_metadata:
            skills_section = format_skills_section(state.skills_metadata)
            if skills_section:
                parts.append(f"\n\n{skills_section}")

        # User's custom system prompt (if provided)
        if self._system_prompt:
            parts.append(f"\n\nFollow these instructions from the User:\n{self._system_prompt}")

        return "".join(parts)

    def _collect_agents_without_code_exec(self) -> list[str]:
        """Collect names of agents (including self and sub-agents) that lack a code execution tool."""
        agents_missing: list[str] = []

        # Check if this agent has a code execution tool
        has_code_exec = any(isinstance(t, CodeExecToolProvider) for t in self._tools)
        if not has_code_exec:
            agents_missing.append(self._name)

        # Recursively check sub-agents
        for tool in self._tools:
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                agents_missing.extend(cell_contents._collect_agents_without_code_exec())  # noqa: SLF001
                        except ValueError:
                            pass

        return agents_missing

    def _validate_subagent_code_exec_requirements(self) -> None:
        """Validate that if any subagent has code exec, the parent must also have code exec.

        This validation ensures proper file transfer chain - subagent files transfer to
        parent's exec env, so parent must have one to receive them.

        Raises:
            ValueError: If a subagent has code exec but this parent doesn't.

        """
        parent_has_code_exec = any(isinstance(t, CodeExecToolProvider) for t in self._tools)

        for tool in self._tools:
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                subagent = cell_contents
                                subagent_has_code_exec = any(
                                    isinstance(t, CodeExecToolProvider)
                                    for t in subagent._tools  # noqa: SLF001
                                )

                                if subagent_has_code_exec and not parent_has_code_exec:
                                    raise ValueError(
                                        f"Subagent '{subagent._name}' has a code execution tool, "  # noqa: SLF001
                                        f"but parent agent '{self._name}' does not. "
                                        f"Parent must have a code execution tool to receive files from subagent."
                                    )

                                # Recursively validate nested subagents
                                subagent._validate_subagent_code_exec_requirements()  # noqa: SLF001
                        except ValueError as e:
                            if "code execution tool" in str(e):
                                raise
                            # cell_contents can raise ValueError if empty - ignore

    async def __aenter__(self) -> "SessionAgent[FinishParams, FinishMeta]":
        """Enter this detached session and initialize its resources."""
        if not isinstance(self, SessionAgent):
            raise RuntimeError("Use `async with agent.session(...)` to enter an Agent session")
        if self._session_state is not None:
            raise RuntimeError("Agent session is already active")

        _reserve_custom_session_resources(self._custom_session_resources)
        self._custom_resources_reserved = True
        exit_stack = AsyncExitStack()
        try:
            await exit_stack.__aenter__()
        except BaseException:
            _release_custom_session_resources(self._custom_session_resources)
            self._custom_resources_reserved = False
            raise

        parent_state = self._parent_session_state
        current_depth = parent_state.depth + 1 if parent_state is not None else 0
        state = SessionState(
            exit_stack=exit_stack,
            owner=self,
            output_dir=str(self._pending_output_dir) if self._pending_output_dir else None,
            parent_exec_env=parent_state.exec_env if parent_state else None,
            depth=current_depth,
            logger=self._logger,
        )
        self._session_state = state
        self._session_state_token = _SESSION_STATE.set(state)
        logger_entered = False

        try:
            # === TWO-PASS TOOL INITIALIZATION ===
            # First pass initializes CodeExecToolProvider so that dependent tools
            # (like ViewImageToolProvider) can access state.exec_env in second pass.
            active_tools: list[Tool] = []
            code_exec_providers = [tool for tool in self._tools if isinstance(tool, CodeExecToolProvider)]
            if len(code_exec_providers) > 1:
                raise ValueError(
                    f"Agent can only have one CodeExecToolProvider, found {len(code_exec_providers)}: "
                    f"{[type(provider).__name__ for provider in code_exec_providers]}"
                )

            # Check if we should share parent's exec_env (subagent with share_parent_exec_env=True)
            should_share_exec_env = (
                self._share_parent_exec_env
                and current_depth > 0
                and parent_state is not None
                and parent_state.exec_env is not None
            )

            if should_share_exec_env:
                # SHARED EXEC ENV: Use parent's exec_env directly, don't create new one
                state.exec_env = parent_state.exec_env  # type: ignore[union-attr]
                state.exec_env_owned = False
                logger.debug(
                    "[%s __aenter__] Sharing parent's exec_env: %s (temp_dir=%s)",
                    self._name,
                    type(state.exec_env).__name__,
                    getattr(state.exec_env, "_temp_dir", "N/A"),
                )
                # Skip CodeExecToolProvider initialization but still need to add code exec tool
                # Create the tool from the shared exec_env using get_code_exec_tool()
                # (the exec_env is already entered by parent, so we just create the tool wrapper)
                if state.exec_env is None:
                    raise RuntimeError("Expected shared exec_env to be set, but it is None")
                code_exec_tool = state.exec_env.get_code_exec_tool()
                active_tools.append(code_exec_tool)
            else:
                # OWNED EXEC ENV: Initialize our own CodeExecToolProvider.
                if code_exec_providers:
                    provider = code_exec_providers[0]
                    result = await exit_stack.enter_async_context(provider)
                    if isinstance(result, list):
                        active_tools.extend(result)
                    else:
                        active_tools.append(result)
                    state.exec_env = provider
                    state.exec_env_owned = True

            # Second pass: Initialize remaining ToolProviders and static Tools
            for tool in self._tools:
                if isinstance(tool, CodeExecToolProvider):
                    continue  # Already processed in first pass

                if isinstance(tool, ToolProvider):
                    # ToolProvider: enter context and get returned tool(s)
                    result = await exit_stack.enter_async_context(tool)
                    # Handle both single Tool and list[Tool] returns (e.g., MCPToolProvider)
                    if isinstance(result, list):
                        active_tools.extend(result)
                    else:
                        active_tools.append(result)
                else:
                    # Static Tool, use directly
                    active_tools.append(tool)

            self._active_tools, self._finish_tools, self._tool_definitions = build_tool_registry(
                active_tools,
                self._finish_tools.values(),
            )

            # Validate subagent code exec requirements (only at root level)
            if current_depth == 0:
                self._validate_subagent_code_exec_requirements()

            # Upload input files to exec_env if specified
            if self._pending_input_files:
                if not state.exec_env:
                    raise ValueError("input_files specified but no CodeExecToolProvider configured")
                pending_input_paths = (
                    [self._pending_input_files]
                    if isinstance(self._pending_input_files, (str, Path))
                    else list(self._pending_input_files)
                )

                logger.debug(
                    "[%s __aenter__] Uploading input files: %s, depth=%d, parent_exec_env=%s, parent_exec_env._temp_dir=%s, exec_env_owned=%s",
                    self._name,
                    self._pending_input_files,
                    state.depth,
                    type(state.parent_exec_env).__name__ if state.parent_exec_env else None,
                    getattr(state.parent_exec_env, "_temp_dir", "N/A") if state.parent_exec_env else None,
                    state.exec_env_owned,
                )

                if state.depth > 0 and state.parent_exec_env:
                    if not state.exec_env_owned:
                        # SHARED EXEC ENV: Files already accessible - no transfer needed
                        state.uploaded_file_paths = sorted(str(path) for path in pending_input_paths)
                        state.cache_input_files = await _fingerprint_shared_inputs(state.exec_env, pending_input_paths)
                        logger.debug(
                            "[%s __aenter__] Shared exec_env - files already accessible: %s",
                            self._name,
                            state.uploaded_file_paths,
                        )
                    else:
                        # SEPARATE EXEC ENV: Read files from parent's exec env, write to subagent's exec env
                        # input_files are paths within the parent's environment
                        result = await state.exec_env.upload_files(
                            *pending_input_paths,
                            source_env=state.parent_exec_env,
                        )
                        logger.debug(
                            "[%s __aenter__] Upload result: uploaded=%s, failed=%s",
                            self._name,
                            result.uploaded,
                            result.failed,
                        )
                        state.uploaded_file_paths = sorted(uploaded.dest_path for uploaded in result.uploaded)
                        if result.failed:
                            raise RuntimeError(f"Failed to upload files: {result.failed}")
                        state.cache_input_files = await _fingerprint_uploaded_files(
                            state.exec_env,
                            result.uploaded,
                        )
                else:
                    # ROOT AGENT: Read files from local filesystem
                    resolved = self._resolve_input_files(self._pending_input_files)
                    result = await state.exec_env.upload_files(*resolved)
                    logger.debug(
                        "[%s __aenter__] Upload result: uploaded=%s, failed=%s",
                        self._name,
                        result.uploaded,
                        result.failed,
                    )
                    state.uploaded_file_paths = sorted(uploaded.dest_path for uploaded in result.uploaded)
                    if result.failed:
                        raise RuntimeError(f"Failed to upload files: {result.failed}")
                    state.cache_input_files = await _fingerprint_uploaded_files(
                        state.exec_env,
                        result.uploaded,
                    )
            self._pending_input_files = None  # Clear pending state

            # Upload skills directory if it exists and load metadata
            if self._pending_skills_dir:
                skills_path = self._pending_skills_dir
                if skills_path.exists() and skills_path.is_dir():
                    if state.exec_env:
                        logger.debug("[%s __aenter__] Uploading skills directory: %s", self._name, skills_path)
                        result = await state.exec_env.upload_files(skills_path, dest_dir="skills")
                        if result.failed:
                            raise RuntimeError(f"Failed to upload skills: {result.failed}")
                        state.cache_skill_files = await _fingerprint_uploaded_files(
                            state.exec_env,
                            result.uploaded,
                        )
                    # Load skills metadata (even if no exec_env, for system prompt)
                    state.skills_metadata = load_skills_metadata(skills_path)
                    logger.debug("[%s __aenter__] Loaded %d skills", self._name, len(state.skills_metadata))
                self._pending_skills_dir = None  # Clear pending state
            elif parent_state is not None and parent_state.skills_metadata:
                # Sub-agent: inherit skills from its explicit parent
                state.skills_metadata = parent_state.skills_metadata
                logger.debug("[%s __aenter__] Inherited %d skills from parent", self._name, len(state.skills_metadata))
                # Transfer skills directory from parent's exec_env to sub-agent's exec_env
                # (only if we have a separate exec_env)
                if state.exec_env and parent_state.exec_env and state.exec_env_owned:
                    result = await state.exec_env.upload_files("skills", source_env=parent_state.exec_env)
                    if result.failed:
                        raise RuntimeError(f"Failed to inherit skills: {result.failed}")
                    state.cache_skill_files = await _fingerprint_uploaded_files(
                        state.exec_env,
                        result.uploaded,
                    )
                elif not state.exec_env_owned:
                    state.cache_skill_files = list(parent_state.cache_skill_files)

            # Configure and enter logger context
            self._logger.name = self._name
            self._logger.model = self._client.model_slug
            self._logger.max_turns = self._max_turns
            self._logger.depth = current_depth
            self._logger.finish_params = None
            self._logger.run_metadata = None
            self._logger.output_dir = None
            self._logger.__enter__()
            logger_entered = True

            if current_depth == 0 and self._cache_on_interrupt:
                self._interrupt_handler_installed = _install_interrupt_handler()

            return self

        except BaseException as exc:
            with anyio.CancelScope(shield=True):
                if logger_entered:
                    try:
                        self._logger.__exit__(type(exc), exc, exc.__traceback__)
                    except BaseException as cleanup_failure:
                        _record_session_cleanup_failure(exc, "Logger", cleanup_failure)
                try:
                    await exit_stack.__aexit__(type(exc), exc, exc.__traceback__)
                except BaseException as cleanup_failure:
                    _record_session_cleanup_failure(exc, "Tool provider", cleanup_failure)
                try:
                    if self._session_state_token is not None:
                        _SESSION_STATE.reset(self._session_state_token)
                        self._session_state_token = None
                except BaseException as cleanup_failure:
                    _record_session_cleanup_failure(exc, "Ambient session context", cleanup_failure)
                self._session_state = None
                try:
                    if self._custom_resources_reserved:
                        _release_custom_session_resources(self._custom_session_resources)
                        self._custom_resources_reserved = False
                except BaseException as cleanup_failure:
                    _record_session_cleanup_failure(exc, "Custom resource reservation", cleanup_failure)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit session context: save files, cleanup resources.

        File handling is depth-aware:
        - Root agent (depth=0): Saves files to local filesystem output_dir
        - Subagent (depth>0): Transfers files to parent's exec env at output_dir path
        """
        state = self._session_state
        if state is None:
            raise RuntimeError("Agent session is not active")

        with anyio.CancelScope(shield=True):
            await self._exit_session(state, exc_type, exc_val)

    async def _exit_session(
        self,
        state: SessionState,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
    ) -> None:
        """Finalize and release a session inside its shielded teardown scope."""
        primary_exception = exc_val
        try:
            # Persist the latest accepted boundary before any fallible finalization.
            # A terminal generation is selected here and cleared only after teardown.
            should_cache = (
                state.depth == 0
                and self._cache_on_interrupt
                and self._current_task_hash is not None
                and self._current_run_state is not None
            )

            logger.debug(
                "[%s __aexit__] Cache decision: should_cache=%s, depth=%d, exc_type=%s, "
                "finish_params=%s, task_hash=%s, run_state=%s",
                self._name,
                should_cache,
                state.depth,
                exc_type,
                self._last_finish_params is not None,
                self._current_task_hash,
                self._current_run_state is not None,
            )

            if should_cache:
                # A checkpoint is a best-effort recovery aid, so a failed save must not skip the
                # output export below - that would leave a successful run with no deliverables.
                try:
                    cache_manager = CacheManager()

                    exec_env_dir = state.exec_env.temp_dir if state.exec_env else None

                    # Explicit checks to keep type checker happy - should_cache condition guarantees these
                    if self._current_task_hash is None or self._current_run_state is None:
                        raise ValueError("Cache state is unexpectedly None after should_cache check")

                    if threading.current_thread() is threading.main_thread():
                        # Prevent a second SIGINT from interrupting the cache write.
                        original_handler = signal.getsignal(signal.SIGINT)
                        signal.signal(signal.SIGINT, signal.SIG_IGN)
                        try:
                            self._task_cache_ownership.save_state(
                                cache_manager,
                                self._current_task_hash,
                                self._current_run_state,
                                exec_env_dir,
                            )
                        finally:
                            signal.signal(signal.SIGINT, original_handler)
                    else:
                        self._task_cache_ownership.save_state(
                            cache_manager,
                            self._current_task_hash,
                            self._current_run_state,
                            exec_env_dir,
                        )
                    self._logger.info(f"Cached state for task {self._current_task_hash}")
                except BaseException as cleanup_failure:
                    primary_exception = _record_session_cleanup_failure(
                        primary_exception, "Cache checkpoint", cleanup_failure
                    )
            # Save files from finish_params.paths based on depth
            if state.output_dir and self._last_finish_params and state.exec_env:
                paths = getattr(self._last_finish_params, "paths", None)
                if paths:
                    if state.depth == 0:
                        # ROOT AGENT: Save to local filesystem
                        output_path = Path(state.output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        logger.debug(
                            "[%s] ROOT AGENT (depth=0): Saving %d file(s) to local filesystem: %s -> %s",
                            self._name,
                            len(paths),
                            paths,
                            output_path,
                        )
                        result = await state.exec_env.save_output_files(paths, output_path, dest_env=None)
                        logger.debug(
                            "[%s] ROOT AGENT: Saved %d file(s), failed %d",
                            self._name,
                            len(result.saved),
                            len(result.failed),
                        )
                        for failed_source, reason in result.failed.items():
                            logger.warning(
                                "[%s] ROOT AGENT: Output file not saved: %s (%s)",
                                self._name,
                                failed_source,
                                reason,
                            )
                    else:
                        # SUBAGENT: Handle file transfer based on exec_env ownership
                        if not state.exec_env_owned:
                            # SHARED EXEC ENV: Files already in parent's env - no transfer needed
                            # Just record the paths for reporting to parent
                            self._transferred_paths = list(paths)
                            logger.debug(
                                "[%s] SUBAGENT (depth=%d, shared_exec_env): Files already in parent env: %s",
                                self._name,
                                state.depth,
                                self._transferred_paths,
                            )
                        elif state.parent_exec_env:
                            # SEPARATE EXEC ENV: Transfer to parent's exec env
                            logger.debug(
                                "[%s] SUBAGENT (depth=%d): Transferring %d file(s) to parent exec env: %s -> %s",
                                self._name,
                                state.depth,
                                len(paths),
                                paths,
                                state.output_dir,
                            )
                            result = await state.exec_env.save_output_files(
                                paths, state.output_dir, dest_env=state.parent_exec_env
                            )
                            # Store transferred paths for returning to parent
                            self._transferred_paths = [str(sf.output_path) for sf in result.saved]
                            logger.debug(
                                "[%s] SUBAGENT: Transferred %d file(s) to parent, failed %d. Paths: %s",
                                self._name,
                                len(result.saved),
                                len(result.failed),
                                self._transferred_paths,
                            )
                            if result.failed:
                                logger.warning("Failed to transfer some files to parent env: %s", result.failed)
                        else:
                            logger.warning(
                                "Subagent at depth %d has exec_env but no parent_exec_env. "
                                "Files will not be transferred.",
                                state.depth,
                            )
        except BaseException as finalization_failure:
            primary_exception = _record_session_cleanup_failure(
                primary_exception, "Session finalization", finalization_failure
            )

        try:
            if self._interrupt_handler_installed:
                _release_interrupt_handler()
                self._interrupt_handler_installed = False
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(primary_exception, "Interrupt handler", cleanup_failure)

        try:
            self._logger.finish_params = self._last_finish_params
            self._logger.run_metadata = self._last_run_metadata
            self._logger.output_dir = str(state.output_dir) if state.output_dir else None
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(
                primary_exception, "Logger final state", cleanup_failure
            )
        try:
            current_exc_type = type(primary_exception) if primary_exception is not None else None
            current_exc_tb = primary_exception.__traceback__ if primary_exception is not None else None
            self._logger.__exit__(current_exc_type, primary_exception, current_exc_tb)
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(primary_exception, "Logger", cleanup_failure)

        try:
            self._active_tools, self._finish_tools, self._tool_definitions = build_tool_registry(
                (tool for tool in self._tools if isinstance(tool, Tool)),
                self._finish_tools.values(),
            )
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(primary_exception, "Active tool reset", cleanup_failure)

        try:
            current_exc_type = type(primary_exception) if primary_exception is not None else None
            current_exc_tb = primary_exception.__traceback__ if primary_exception is not None else None
            await state.exit_stack.__aexit__(current_exc_type, primary_exception, current_exc_tb)
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(primary_exception, "Tool provider", cleanup_failure)

        try:
            if self._session_state_token is not None:
                _SESSION_STATE.reset(self._session_state_token)
                self._session_state_token = None
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(
                primary_exception, "Ambient session context", cleanup_failure
            )
        self._session_state = None

        try:
            if self._custom_resources_reserved:
                _release_custom_session_resources(self._custom_session_resources)
                self._custom_resources_reserved = False
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(
                primary_exception, "Custom resource reservation", cleanup_failure
            )

        if (
            primary_exception is None
            and self._last_finish_params is not None
            and self._clear_cache_on_success
            and self._current_task_hash is not None
        ):
            try:
                if self._task_cache_ownership.clear_cache_if_owned(CacheManager(), self._current_task_hash):
                    self._current_task_hash = None
                    self._current_run_state = None
            except BaseException as cleanup_failure:
                primary_exception = _record_session_cleanup_failure(
                    primary_exception, "Success cache cleanup", cleanup_failure
                )

        try:
            self._task_cache_ownership.release()
        except BaseException as cleanup_failure:
            primary_exception = _record_session_cleanup_failure(
                primary_exception, "Task cache reservation", cleanup_failure
            )

        if exc_val is None and primary_exception is not None:
            raise primary_exception

    def _require_own_active_session_context(self) -> None:
        """Reject use of a detached session outside its ambient ownership context."""
        if isinstance(self, SessionAgent):
            session_state = self._session_state
            if session_state is None or _SESSION_STATE.get() is not session_state:
                raise RuntimeError("SessionAgent requires its own active session context")

    async def run_tool(self, tool_call: ToolCall, run_metadata: dict[str, list[Any]]) -> ToolMessage:
        """Execute a single tool call with error handling for invalid JSON/arguments.

        Returns a ToolMessage containing either the tool output or an error description.
        Metadata from the tool result is stored in the provided run_metadata dict.
        """
        self._require_own_active_session_context()
        tool = self._active_tools.get(tool_call.name)
        result: ToolResult
        args_valid = True

        # Ensure tool is tracked in metadata dict (even if no metadata returned)
        if tool_call.name not in run_metadata:
            run_metadata[tool_call.name] = []

        tool_start_time = perf_counter()

        if tool:
            try:
                # Normalize empty arguments to valid empty JSON object
                args = tool_call.arguments if tool_call.arguments and tool_call.arguments.strip() else "{}"
                params = tool.parameters.model_validate_json(args)

                if inspect.iscoroutinefunction(tool.executor):
                    result = await tool.executor(params)  # ty: ignore[invalid-await]
                elif self._run_sync_in_thread:
                    # ty: ignore - type checker doesn't understand iscoroutinefunction narrowing
                    result = await anyio.to_thread.run_sync(tool.executor, params)  # ty: ignore[unresolved-attribute]
                else:
                    # ty: ignore - iscoroutinefunction check above ensures this is sync
                    result = tool.executor(params)  # ty: ignore[invalid-assignment]

                # Store metadata if present
                if result.metadata is not None:
                    run_metadata[tool_call.name].append(result.metadata)
            except ValidationError:
                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )
                result = ToolResult(content="Tool arguments are not valid", success=False)
                args_valid = False
        else:
            LOGGER.debug(f"LLMClient tried to use the tool {tool_call.name} which is not in the tools list")
            result = ToolResult(content=f"{tool_call.name} is not a valid tool", success=False)

        tool_end_time = perf_counter()

        return ToolMessage(
            content=result.content,
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            args_was_valid=args_valid,
            success=result.success,
            tool_start_time=tool_start_time,
            tool_end_time=tool_end_time,
        )

    def _validate_context_overflow_recovery(self, messages: list[ChatMessage]) -> None:
        """Require accepted work after the active context boundary before compression."""
        if not self._recover_from_context_overflow:
            raise ContextOverflowError("Context overflow recovery is disabled")
        for message in reversed(messages):
            if isinstance(message, SummaryMessage):
                break
            if isinstance(message, AssistantMessage):
                return
        raise self._context_boundary_error(messages)

    @staticmethod
    def _context_boundary_error(messages: list[ChatMessage]) -> ContextOverflowError:
        boundary = (
            "summarized context" if any(isinstance(msg, SummaryMessage) for msg in messages) else "original prompt"
        )
        return ContextOverflowError(f"Context overflow reached the {boundary}")

    def _checkpoint_run_state(
        self,
        messages: list[ChatMessage],
        full_message_history: list[list[ChatMessage]],
        run_metadata: dict[str, list[Any]],
        task_hash: str,
    ) -> None:
        """Snapshot canonical accepted messages and flat metadata for recovery.

        The snapshot stays in memory until the root session exits, so only endings that unwind
        through ``__aexit__`` are recoverable. An abruptly terminated process (SIGKILL, SIGTERM,
        OOM kill, ``docker stop``) loses every accepted result. Persisting each checkpoint would
        earn its write cost only if abrupt termination is observed in practice.
        """
        self._current_run_state = CacheState(
            msgs=list(messages),
            full_msg_history=[list(group) for group in full_message_history],
            run_metadata={name: list(items) for name, items in run_metadata.items()},
            task_hash=task_hash,
            agent_name=self._name,
        )

    async def _execute_assistant_tool_calls(
        self,
        assistant_message: AssistantMessage,
        tool_calls: list[ToolCall],
        run_metadata: dict[str, list[Any]],
        *,
        initial_tool_messages: list[ToolMessage] | None = None,
        initial_user_messages: list[UserMessage] | None = None,
        adapt_text_only: bool,
        on_tool_result: Callable[[AssistantMessage, list[ToolMessage], list[UserMessage]], None] | None = None,
    ) -> tuple[list[ToolMessage], list[UserMessage], FinishParams | None]:
        """Execute ordered calls and expose each complete accepted result before callbacks."""
        finish_params: FinishParams | None = None
        tool_messages = list(initial_tool_messages or [])
        user_messages = list(initial_user_messages or [])
        finish_call_names = [call.name for call in assistant_message.tool_calls if call.name in self._finish_tools]
        reject_all_finish_calls = len(finish_call_names) > 1

        for index, tool_call in enumerate(tool_calls):
            skipped_messages: list[ToolMessage] = []
            if reject_all_finish_calls and tool_call.name in self._finish_tools:
                now = perf_counter()
                tool_message = ToolMessage(
                    content=(
                        f"Cannot call finish tool '{tool_call.name}': multiple finish tools "
                        f"({sorted(set(finish_call_names))}) were called in the same turn. "
                        "Only one finish tool may be called per turn — retry with a single finish tool call."
                    ),
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    args_was_valid=True,
                    success=False,
                    tool_start_time=now,
                    tool_end_time=now,
                )
            else:
                tool_message = await self.run_tool(tool_call, run_metadata)

            new_user_messages: list[UserMessage] = []
            if adapt_text_only:
                transformed, new_user_messages = _handle_text_only_tool_responses([tool_message])
                tool_message = transformed[0]
            tool_messages.append(tool_message)
            user_messages.extend(new_user_messages)

            if tool_message.success and tool_message.name in self._finish_tools:
                finish_tool = self._finish_tools[tool_message.name]
                finish_params = finish_tool.parameters.model_validate_json(
                    tool_call.arguments if tool_call.arguments.strip() else "{}"
                )
                skipped_messages.extend(
                    ToolMessage(
                        content=(
                            f"Skipped tool '{skipped_call.name}' because finish tool "
                            f"'{tool_message.name}' completed successfully earlier in the same turn."
                        ),
                        tool_call_id=skipped_call.tool_call_id,
                        name=skipped_call.name,
                        success=False,
                    )
                    for skipped_call in tool_calls[index + 1 :]
                )
                tool_messages.extend(skipped_messages)

            if on_tool_result is not None:
                on_tool_result(assistant_message, list(tool_messages), list(user_messages))

            self._logger.tool_result(tool_message)
            for skipped_message in skipped_messages:
                self._logger.tool_result(skipped_message)
            for user_message in new_user_messages:
                self._logger.user_message(user_message)
            if finish_params is not None:
                break

        return tool_messages, user_messages, finish_params

    async def _step_with_semantic_results(
        self,
        messages: list[ChatMessage],
        run_metadata: dict[str, list[Any]],
        turn: int,
        max_turns: int,
        on_tool_result: Callable[[AssistantMessage, list[ToolMessage], list[UserMessage]], None],
    ) -> tuple[AssistantMessage, list[ToolMessage], list[UserMessage], FinishParams | None]:
        assistant_message = await self._client.generate(messages, self._active_tools)
        self._logger.assistant_message(turn, max_turns, assistant_message)
        tool_messages, user_messages, finish_params = await self._execute_assistant_tool_calls(
            assistant_message,
            assistant_message.tool_calls,
            run_metadata,
            adapt_text_only=self._text_only_tool_responses,
            on_tool_result=on_tool_result,
        )
        return assistant_message, tool_messages, user_messages, finish_params

    async def step(
        self,
        messages: list[ChatMessage],
        run_metadata: dict[str, list[Any]],
        turn: int = 0,
        max_turns: int = 0,
    ) -> tuple[AssistantMessage, list[ToolMessage], FinishParams | None]:
        """Execute one public agent step without text-only message adaptation."""
        assistant_message = await self._client.generate(messages, self._active_tools)
        if turn > 0:
            self._logger.assistant_message(turn, max_turns, assistant_message)
        tool_messages, _user_messages, finish_params = await self._execute_assistant_tool_calls(
            assistant_message,
            assistant_message.tool_calls,
            run_metadata,
            adapt_text_only=False,
        )
        return assistant_message, tool_messages, finish_params

    async def summarize_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Summarize accepted messages without discarding their metadata."""
        return await self._summarize_message_prefix(messages, len(messages))

    async def _summarize_older_context(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Summarize the largest fitting older prefix while retaining the latest turn."""
        latest_summary_index = max(
            (index for index, message in enumerate(messages) if isinstance(message, SummaryMessage)),
            default=-1,
        )
        assistant_turns: list[tuple[int, AssistantMessage]] = []
        for index, message in enumerate(messages):
            if index <= latest_summary_index or not isinstance(message, AssistantMessage):
                continue
            turn_start = index - 1 if index > 0 and isinstance(messages[index - 1], TurnWarningMessage) else index
            assistant_turns.append((turn_start, message))
        if len(assistant_turns) < 2:
            raise self._context_boundary_error(messages)

        last_overflow: ContextOverflowError | None = None
        for prefix_end, retained_assistant in reversed(assistant_turns[1:]):
            if retained_assistant.token_usage.input >= self._client.max_tokens:
                continue
            try:
                return await self._summarize_message_prefix(messages, prefix_end)
            except ContextOverflowError as error:
                last_overflow = error
        if last_overflow is not None:
            raise last_overflow
        raise self._context_boundary_error(messages)

    async def _summarize_message_prefix(
        self,
        messages: list[ChatMessage],
        prefix_end: int,
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """Summarize one complete prefix and retain its remainder unchanged."""
        task_context = list(takewhile(lambda msg: not isinstance(msg, (AssistantMessage, SummaryMessage)), messages))
        messages_to_summarize = messages[:prefix_end]
        summary = await self._client.generate(
            [*messages_to_summarize, UserMessage(content=MESSAGE_SUMMARIZER)],
            self._active_tools,
        )
        summary_bridge_prompt = MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary.content)
        summary_bridge = SummaryMessage(content=summary_bridge_prompt)
        acknowledgement = UserMessage(content="Got it, thanks!")
        summary_content = summary.content if isinstance(summary.content, str) else str(summary.content)
        self._logger.context_summarization_complete(summary_content, summary_bridge_prompt)
        return messages_to_summarize, [*task_context, summary_bridge, acknowledgement, *messages[prefix_end:]]

    async def run(
        self,
        init_msgs: str | list[ChatMessage],
        *,
        depth: int | None = None,
    ) -> tuple[FinishParams | None, list[list[ChatMessage]], dict[str, Any]]:
        """Execute the agent loop until finish tool is called or max_turns reached.

        A base system prompt is automatically prepended to all runs, including:
        - Agent purpose and max_turns info
        - List of input files (if provided via session())
        - User's custom system_prompt (if configured in __init__)

        Args:
            init_msgs: Either a string prompt (converted to UserMessage) or a list of
                      ChatMessage to extend the conversation after the system prompt.
            depth: Logging depth for sub-agent runs. If provided, updates logger.depth for this run.

        Returns:
            Tuple of (finish params, message history, run metadata).
            finish params is None if max_turns reached.
            run metadata maps tool/agent names to lists of metadata returned by each call.

        Example:
            # Simple string prompt
            await agent.run("Analyze this data and create a report")

            # Multiple messages
            await agent.run([
                UserMessage(content="First, read the data"),
                AssistantMessage(content="I've read the data file..."),
                UserMessage(content="Now analyze it"),
            ])

        """

        self._require_own_active_session_context()
        if not isinstance(self, SessionAgent) and self._has_tool_providers:
            provider_names = [type(tool).__name__ for tool in self._tools if isinstance(tool, ToolProvider)]
            raise RuntimeError(
                f"Agent.run() called without a session, but the agent has ToolProviders "
                f"that require session initialization: {provider_names}. "
                f"Use `async with agent.session(...) as session: await session.run(...)` instead."
            )

        if isinstance(self, SessionAgent):
            return await self._run(init_msgs, depth=depth)

        ambient_session_token = _SESSION_STATE.set(None)
        task_cache_ownership = _TaskCacheOwnership()
        full_system_prompt = self._build_system_prompt()
        task_hash = compute_task_hash(
            init_msgs,
            agent_name=self._name,
            model_slug=self._client.model_slug,
            system_prompt=full_system_prompt,
            tool_definitions=self._tool_definitions,
        )
        primary_exception: BaseException | None = None
        try:
            if self._cache_on_interrupt:
                task_cache_ownership.reserve(task_hash)
            result = await self._run(init_msgs, depth=depth, task_hash=task_hash)
            if result[0] is not None and self._clear_cache_on_success:
                task_cache_ownership.clear_cache_if_owned(CacheManager(), task_hash)
            return result
        except BaseException as exc:
            primary_exception = exc
            raise
        finally:
            try:
                task_cache_ownership.release()
            except BaseException as cleanup_failure:
                if primary_exception is None:
                    raise
                primary_exception.add_note(
                    "Task cache reservation release failed after Agent.run(): "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
            finally:
                _SESSION_STATE.reset(ambient_session_token)

    async def _run(
        self,
        init_msgs: str | list[ChatMessage],
        *,
        depth: int | None = None,
        task_hash: str | None = None,
    ) -> tuple[FinishParams | None, list[list[ChatMessage]], dict[str, Any]]:
        """Execute a run after its session ownership and ambient context are established."""
        session_state = _SESSION_STATE.get()
        full_system_prompt = self._build_system_prompt()
        task_hash = task_hash or compute_task_hash(
            init_msgs,
            agent_name=self._name,
            model_slug=self._client.model_slug,
            system_prompt=full_system_prompt,
            tool_definitions=self._tool_definitions,
            input_files=session_state.cache_input_files if session_state else (),
            skill_files=session_state.cache_skill_files if session_state else (),
        )
        if (
            isinstance(self, SessionAgent)
            and (self._cache_on_interrupt or self._resume)
            and session_state is not None
            and session_state.depth == 0
        ):
            self._task_cache_ownership.reserve(task_hash)
        self._current_task_hash = task_hash
        self._last_finish_params = None
        self._last_run_metadata = {}

        cache_manager = CacheManager()
        resumed = False
        restored_progress = None
        run_metadata: dict[str, list[Any]] = {}

        if self._resume:
            if session_state is None:
                raise RuntimeError("Cannot resume an Agent run without an active session")
            restore_dir = session_state.exec_env.temp_dir if session_state.exec_env else None
            if session_state.exec_env and restore_dir is None:
                cached = None
                self._logger.warning(
                    f"Resume caching is unavailable for {type(session_state.exec_env).__name__}; starting fresh"
                )
            else:
                cached = self._task_cache_ownership.load_state(
                    cache_manager,
                    task_hash,
                    restore_files_to=restore_dir,
                    finish_tools=self._finish_tools,
                )
                if cached is None:
                    if cache_manager.has_cache(task_hash):
                        # Starting fresh from a cache that exists replays every accepted tool call.
                        self._logger.warning(f"Ignoring an unusable cache for task {task_hash}; starting fresh")
                    else:
                        self._logger.info(f"No cache found for task {task_hash}, starting fresh")
            if cached is not None:
                msgs = cached.msgs
                full_msg_history = cached.full_msg_history
                run_metadata.update(cached.run_metadata)
                restored_progress = cached.message_progress
                self._current_run_state = cached
                resumed = True
                self._logger.info(f"Resuming from cached state at turn {_get_turn_count(full_msg_history, msgs)}")

        if not resumed:
            msgs: list[ChatMessage] = [SystemMessage(content=full_system_prompt)]
            if isinstance(init_msgs, str):
                msgs.append(UserMessage(content=init_msgs))
            else:
                msgs.extend(init_msgs)
            full_msg_history: list[list[ChatMessage]] = []

        if depth is not None:
            self._logger.depth = depth
        if not resumed:
            self._logger.task_message(msgs[-1].content)
        if self._logger.depth == 0 and isinstance(self._logger, AgentLogger):
            run_warnings = self._collect_warnings()
            if run_warnings:
                self._logger.warnings_message(run_warnings)

        finish_params: FinishParams | None = None
        if restored_progress is not None:
            finish_params = cast(FinishParams | None, restored_progress.finish_params)

        all_accepted_messages = [*chain.from_iterable(full_msg_history), *msgs]
        total_tool_calls = sum(isinstance(message, ToolMessage) for message in all_accepted_messages)
        accepted_assistants = [message for message in all_accepted_messages if isinstance(message, AssistantMessage)]
        total_input_tokens = sum(message.token_usage.input for message in accepted_assistants)
        total_output_tokens = sum(message.token_usage.output for message in accepted_assistants)
        step_callback = self._logger.on_step

        if finish_params is None and restored_progress is not None and restored_progress.pending_calls:
            pending_assistant = restored_progress.assistant
            pending_assistant_index = restored_progress.assistant_index
            if pending_assistant is None or pending_assistant_index is None:
                raise ValueError("Cached pending calls have no assistant message")
            pending_prefix = msgs[:pending_assistant_index]

            def checkpoint_pending_result(
                assistant: AssistantMessage,
                partial_tool_messages: list[ToolMessage],
                partial_user_messages: list[UserMessage],
            ) -> None:
                self._checkpoint_run_state(
                    [*pending_prefix, assistant, *partial_tool_messages, *partial_user_messages],
                    full_msg_history,
                    run_metadata,
                    task_hash,
                )

            resumed_tool_messages, resumed_user_messages, finish_params = await self._execute_assistant_tool_calls(
                pending_assistant,
                list(restored_progress.pending_calls),
                run_metadata,
                initial_tool_messages=list(restored_progress.tool_messages),
                initial_user_messages=list(restored_progress.user_messages),
                adapt_text_only=self._text_only_tool_responses,
                on_tool_result=checkpoint_pending_result,
            )
            msgs[:] = [
                *pending_prefix,
                pending_assistant,
                *resumed_tool_messages,
                *resumed_user_messages,
            ]
            self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)
            total_tool_calls += len(resumed_tool_messages) - len(restored_progress.tool_messages)
            step_callback(
                _get_turn_count(full_msg_history, msgs),
                total_tool_calls,
                total_input_tokens,
                total_output_tokens,
            )

        while finish_params is None and _get_turn_count(full_msg_history, msgs) < self._max_turns:
            turn = _get_turn_count(full_msg_history, msgs)
            self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)
            if self._max_turns - turn <= self._turns_remaining_warning_threshold and turn != 0:
                warning_message = _num_turns_remaining_msg(self._max_turns - turn)
                msgs.append(warning_message)
                self._logger.user_message(warning_message)

            while True:
                checkpoint_prefix = list(msgs)

                def checkpoint_tool_result(
                    assistant: AssistantMessage,
                    partial_tool_messages: list[ToolMessage],
                    partial_user_messages: list[UserMessage],
                    accepted_prefix: list[ChatMessage] = checkpoint_prefix,
                ) -> None:
                    self._checkpoint_run_state(
                        [*accepted_prefix, assistant, *partial_tool_messages, *partial_user_messages],
                        full_msg_history,
                        run_metadata,
                        task_hash,
                    )

                try:
                    (
                        assistant_message,
                        tool_messages,
                        user_messages,
                        finish_params,
                    ) = await self._step_with_semantic_results(
                        msgs,
                        run_metadata,
                        turn=turn + 1,
                        max_turns=self._max_turns,
                        on_tool_result=checkpoint_tool_result,
                    )
                    break
                except ContextOverflowError:
                    self._validate_context_overflow_recovery(msgs)
                    try:
                        messages_to_summarize, compressed_messages = await self._summarize_older_context(msgs)
                    except ContextOverflowError as error:
                        self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)
                        raise self._context_boundary_error(msgs) from error
                    full_msg_history.append(messages_to_summarize)
                    msgs = compressed_messages
                    self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)
                    turn = _get_turn_count(full_msg_history, msgs)

            accepted_messages = [*msgs, assistant_message, *tool_messages, *user_messages]
            self._checkpoint_run_state(accepted_messages, full_msg_history, run_metadata, task_hash)
            accepted_turn = _get_turn_count(full_msg_history, msgs) + 1
            total_tool_calls += len(tool_messages)
            total_input_tokens += assistant_message.token_usage.input
            total_output_tokens += assistant_message.token_usage.output
            step_callback(accepted_turn, total_tool_calls, total_input_tokens, total_output_tokens)

            msgs[:] = accepted_messages
            if finish_params is not None:
                break

            pct_context_used = assistant_message.token_usage.total / self._client.max_tokens
            if pct_context_used >= self._context_summarization_cutoff and accepted_turn != self._max_turns:
                self._logger.context_summarization_start(pct_context_used, self._context_summarization_cutoff)
                try:
                    messages_to_summarize, msgs = await self.summarize_messages(msgs)
                except ContextOverflowError:
                    self._validate_context_overflow_recovery(msgs)
                    try:
                        messages_to_summarize, msgs = await self._summarize_older_context(msgs)
                    except ContextOverflowError as error:
                        self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)
                        raise self._context_boundary_error(msgs) from error
                full_msg_history.append(messages_to_summarize)
                self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)

            next_turn_will_show_warning = self._max_turns - accepted_turn <= self._turns_remaining_warning_threshold
            if (
                self._block_successive_assistant_messages
                and not tool_messages
                and not user_messages
                and not next_turn_will_show_warning
            ):
                msgs.append(UserMessage(content="Please continue the task"))
                self._checkpoint_run_state(msgs, full_msg_history, run_metadata, task_hash)

        if finish_params is None:
            LOGGER.error(
                f"Maximum number of turns reached: {self._max_turns}. The agent was not able to finish the task. Consider increasing the max_turns parameter."
            )

        full_msg_history.append(msgs)
        public_run_metadata = {name: list(items) for name, items in run_metadata.items()}
        public_run_metadata["token_usage"] = _get_total_token_usage(full_msg_history)
        public_run_metadata["_tool_durations"] = _get_tool_durations(full_msg_history)  # type: ignore[assignment]
        public_run_metadata["_model_speed"] = _get_model_speed_stats(  # type: ignore[assignment]
            full_msg_history,
            self._client.model_slug,
        )
        self._last_finish_params = finish_params
        self._last_run_metadata = public_run_metadata
        return finish_params, full_msg_history, public_run_metadata

    def to_tool(
        self,
        *,
        description: str = DEFAULT_SUB_AGENT_DESCRIPTION,
        system_prompt: str | None = None,
    ) -> Tool[SubAgentParams, SubAgentMetadata]:
        """Convert this Agent to a Tool for use as a sub-agent.

        Args:
            description: Tool description shown to the parent agent
            system_prompt: Optional system prompt to prepend when running

        Returns:
            Tool that executes this agent when called, returning SubAgentMetadata
            containing token usage, message history, and any metadata from tools
            the sub-agent used.

        """
        agent = self  # Capture self for closure
        sub_agent_tool: Tool[SubAgentParams, SubAgentMetadata]

        async def sub_agent_executor(params: SubAgentParams) -> ToolResult[SubAgentMetadata]:
            """Execute this agent as a child of the active session."""
            parent_session_state = _SESSION_STATE.get(None)
            if (
                parent_session_state is None
                or parent_session_state.owner is None
                or parent_session_state.owner.tools.get(agent.name) is not sub_agent_tool
            ):
                return ToolResult(
                    content=(
                        f"<sub_agent_result>\n<error>Sub-agent tool '{agent.name}' requires an active parent "
                        "Agent session. Use `async with parent.session() as session:` before running the parent."
                        "</error>\n</sub_agent_result>"
                    ),
                    success=False,
                    metadata=SubAgentMetadata(message_history=[], run_metadata={}),
                )
            try:
                init_msgs: list[ChatMessage] = []
                if system_prompt:
                    init_msgs.append(SystemMessage(content=system_prompt))
                init_msgs.append(UserMessage(content=params.task))

                agent_session = SessionAgent.from_agent(
                    agent,
                    output_dir=".",
                    input_files=list(params.input_files) if params.input_files else None,
                    parent_session_state=parent_session_state,
                )
                async with agent_session:
                    finish_params, msg_history, run_metadata = await agent_session.run(init_msgs)

                last_assistant_msg: AssistantMessage | None = None
                for msg_group in reversed(msg_history):
                    for msg in reversed(msg_group):
                        if isinstance(msg, AssistantMessage) and msg.content:
                            last_assistant_msg = msg
                            break
                    if last_assistant_msg:
                        break

                content_parts: list[str] = []
                if last_assistant_msg and last_assistant_msg.content:
                    content = last_assistant_msg.content
                    if isinstance(content, list):
                        content = "\n".join(str(block) for block in content)
                    content_parts.append(content)

                if finish_params is not None:
                    finish_dict = finish_params.model_dump()
                    if finish_dict:
                        content_parts.append(f"Finish params: {finish_dict}")

                if agent_session._transferred_paths:  # noqa: SLF001
                    content_parts.append(
                        f"Files available in your environment: {agent_session._transferred_paths}"  # noqa: SLF001
                    )

                if not content_parts:
                    result_content = "<sub_agent_result>\n<error>No assistant message or finish params found</error>\n</sub_agent_result>"
                else:
                    content = "\n".join(content_parts)
                    result_content = (
                        f"<sub_agent_result>"
                        f"\n<response>{content}</response>"
                        f"\n<finished>{finish_params is not None}</finished>"
                        f"\n</sub_agent_result>"
                    )

                return ToolResult(
                    content=result_content,
                    metadata=SubAgentMetadata(message_history=msg_history, run_metadata=run_metadata),
                )
            except Exception as exc:
                return ToolResult(
                    content=f"<sub_agent_result>\n<error>{exc!s}</error>\n</sub_agent_result>",
                    success=False,
                    metadata=SubAgentMetadata(message_history=[], run_metadata={}),
                )

        sub_agent_tool = Tool[SubAgentParams, SubAgentMetadata](
            name=self._name,
            description=description,
            parameters=SubAgentParams,
            executor=sub_agent_executor,  # ty: ignore[invalid-argument-type]
        )
        return sub_agent_tool


class SessionAgent[FinishParams: BaseModel, FinishMeta](Agent[FinishParams, FinishMeta]):
    """Detached runtime returned by ``Agent.session()``.

    Agent configuration, the LLM client, and static tools are shared. Active
    tools, provider and logger lifecycle, run results, cache state, skills, and
    output state belong to this session.
    """

    @classmethod
    def from_agent(
        cls,
        agent: Agent[FinishParams, FinishMeta],
        *,
        output_dir: Path | str | None = None,
        input_files: str | Path | list[str | Path] | None = None,
        skills_dir: Path | str | None = None,
        resume: bool = False,
        clear_cache_on_success: bool = True,
        cache_on_interrupt: bool = True,
        parent_session_state: SessionState | None = None,
    ) -> "SessionAgent[FinishParams, FinishMeta]":
        """Shallow-copy configuration and replace all mutable runtime owners."""
        from stirrup.tools.view_image import ViewImageToolProvider

        configured_tools = agent._tools  # noqa: SLF001
        configured_providers = {id(tool): tool for tool in configured_tools if isinstance(tool, ToolProvider)}
        sequential_provider_ids = {
            id(tool._exec_env)  # noqa: SLF001
            for tool in configured_providers.values()
            if isinstance(tool, ViewImageToolProvider)
            and type(tool) is not ViewImageToolProvider
            and tool._exec_env is not None  # noqa: SLF001
            and id(tool._exec_env) in configured_providers  # noqa: SLF001
        }

        replacements: dict[int, ToolProvider] = {}
        should_share_exec_env = (
            agent._share_parent_exec_env  # noqa: SLF001
            and parent_session_state is not None
            and parent_session_state.exec_env is not None
        )
        if should_share_exec_env and sequential_provider_ids:
            raise ValueError(
                f"Subagent '{agent._name}' cannot use share_parent_exec_env=True with a custom "  # noqa: SLF001
                "ViewImageToolProvider configured for its own code backend. Disable environment sharing "
                "or use the exact built-in ViewImageToolProvider."
            )

        for configured_tool in agent._tools:  # noqa: SLF001
            if not isinstance(configured_tool, ToolProvider):
                continue
            if type(configured_tool) is ViewImageToolProvider:
                continue
            if id(configured_tool) in sequential_provider_ids:
                replacements[id(configured_tool)] = configured_tool
                continue
            if should_share_exec_env and isinstance(configured_tool, CodeExecToolProvider):
                replacements[id(configured_tool)] = parent_session_state.exec_env  # type: ignore[assignment]
                continue
            replacements[id(configured_tool)] = (
                _detach_builtin_provider(configured_tool, replacements) or configured_tool
            )

        session_tools: list[Tool | ToolProvider] = []
        custom_resources: list[object] = []
        for configured_tool in agent._tools:  # noqa: SLF001
            if not isinstance(configured_tool, ToolProvider):
                session_tools.append(configured_tool)
                continue
            session_tool = replacements.get(id(configured_tool))
            if session_tool is None:
                session_tool = _detach_builtin_provider(configured_tool, replacements) or configured_tool
                replacements[id(configured_tool)] = session_tool
            session_tools.append(session_tool)
            if session_tool is configured_tool:
                custom_resources.append(configured_tool)

        session_logger, is_custom_logger = _detach_logger(agent._logger)  # noqa: SLF001
        if is_custom_logger:
            custom_resources.append(session_logger)
        custom_resources = list({id(resource): resource for resource in custom_resources}.values())

        session: SessionAgent[FinishParams, FinishMeta] = object.__new__(cls)
        session.__dict__ = agent.__dict__.copy()
        session._tools = session_tools
        session._active_tools, session._finish_tools, session._tool_definitions = build_tool_registry(
            (tool for tool in session_tools if isinstance(tool, Tool)),
            agent._finish_tools.values(),  # noqa: SLF001
        )
        session._has_tool_providers = any(isinstance(tool, ToolProvider) for tool in session_tools)
        session._logger = session_logger

        session._pending_output_dir = Path(output_dir) if output_dir else None
        session._pending_input_files = list(input_files) if isinstance(input_files, list) else input_files
        session._pending_skills_dir = Path(skills_dir) if skills_dir else None
        session._resume = resume
        session._clear_cache_on_success = clear_cache_on_success
        session._cache_on_interrupt = cache_on_interrupt

        session._last_finish_params = None
        session._last_run_metadata = {}
        session._transferred_paths = []
        session._current_task_hash = None
        session._current_run_state = None
        session._parent_session_state = parent_session_state
        session._session_state = None
        session._session_state_token = None
        session._interrupt_handler_installed = False
        session._custom_session_resources = custom_resources
        session._custom_resources_reserved = False
        session._task_cache_ownership = _TaskCacheOwnership()
        return session

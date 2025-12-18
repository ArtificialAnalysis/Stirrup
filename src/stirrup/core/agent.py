# Context var for passing parent depth to sub-agent executors
import contextvars
import glob as glob_module
import inspect
import json
import logging
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from itertools import chain, takewhile
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Self

import anyio
from pydantic import BaseModel, Field, ValidationError

from stirrup.constants import (
    AGENT_MAX_TURNS,
    CONTEXT_SUMMARIZATION_CUTOFF,
    FINISH_TOOL_NAME,
)
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    SubAgentMetadata,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER, MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE
from stirrup.skills import SkillMetadata, format_skills_section, load_skills_metadata
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.code_backends.base import CodeExecToolProvider
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.utils.logging import AgentLogger, AgentLoggerBase

_PARENT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("parent_depth", default=0)

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
    """

    exit_stack: AsyncExitStack
    exec_env: CodeExecToolProvider | None = None
    output_dir: str | None = None  # String path (contextual: local for root, in parent env for subagent)
    parent_exec_env: CodeExecToolProvider | None = None
    depth: int = 0
    uploaded_file_paths: list[str] = field(default_factory=list)  # Paths of files uploaded to exec_env
    skills_metadata: list[SkillMetadata] = field(default_factory=list)  # Loaded skills metadata


_SESSION_STATE: contextvars.ContextVar[SessionState] = contextvars.ContextVar("session_state")

__all__ = [
    "Agent",
    "SubAgentParams",
]

LOGGER = logging.getLogger(__name__)


def _num_turns_remaining_msg(number_of_turns_remaining: int) -> UserMessage:
    """Create a user message warning the agent about remaining turns before max_turns is reached."""
    if number_of_turns_remaining == 1:
        return UserMessage(content="This is the last turn. Please finish the task by calling the finish tool.")
    return UserMessage(
        content=f"You have {number_of_turns_remaining} turns remaining to complete the task. Please continue. Remember you will need a separate turn to finish the task.",
    )


def _handle_text_only_tool_responses(tool_messages: list[ToolMessage]) -> tuple[list[ToolMessage], list[UserMessage]]:
    """Extract image blocks from tool messages and convert them to user messages for text-only models."""
    user_messages: list[UserMessage] = []
    for tm in tool_messages:
        if isinstance(tm.content, list):
            for idx, block in enumerate(tm.content):
                if isinstance(block, ImageContentBlock):
                    user_messages.append(
                        UserMessage(content=[f"Here is the image for tool call {tm.tool_call_id}", block]),
                    )
                    tm.content[idx] = f"Done! The User will provide the image for tool call {tm.tool_call_id}"
                elif isinstance(block, str):
                    continue
                else:
                    raise NotImplementedError(f"Unsupported content block: {type(block)}")

    return tool_messages, user_messages


def _get_total_token_usage(messages: list[list[ChatMessage]]) -> TokenUsage:
    """Aggregate token usage across all assistant messages in grouped conversation history.

    Args:
        messages: List of message groups, where each group represents a segment of conversation.

    """
    return sum(
        [msg.token_usage for msg in chain.from_iterable(messages) if isinstance(msg, AssistantMessage)],
        start=TokenUsage(),
    )


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

    def __init__(
        self,
        client: LLMClient,
        name: str,
        *,
        max_turns: int = AGENT_MAX_TURNS,
        system_prompt: str | None = None,
        tools: list[Tool | ToolProvider] | None = None,
        finish_tool: Tool[FinishParams, FinishMeta] | None = None,
        # Agent options
        context_summarization_cutoff: float = CONTEXT_SUMMARIZATION_CUTOFF,
        run_sync_in_thread: bool = True,
        text_only_tool_responses: bool = True,
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
            finish_tool: Tool used to signal task completion. Defaults to SIMPLE_FINISH_TOOL.
            context_summarization_cutoff: Fraction of context window (0-1) at which to trigger summarization
            run_sync_in_thread: Execute synchronous tool executors in a separate thread
            text_only_tool_responses: Extract images from tool responses as separate user messages
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
        self._finish_tool: Tool = finish_tool if finish_tool is not None else SIMPLE_FINISH_TOOL
        self._context_summarization_cutoff = context_summarization_cutoff
        self._run_sync_in_thread = run_sync_in_thread
        self._text_only_tool_responses = text_only_tool_responses

        # Logger (can be passed in or created here)
        self._logger: AgentLoggerBase = logger if logger is not None else AgentLogger()

        # Session configuration (set during session(), used in __aenter__)
        self._pending_output_dir: Path | None = None
        self._pending_input_files: str | Path | list[str | Path] | None = None
        self._pending_skills_dir: Path | None = None

        # Instance-scoped state (populated during __aenter__, isolated per agent instance)
        self._active_tools: dict[str, Tool] = {}
        self._last_finish_params: Any = None  # FinishParams type parameter
        self._last_run_metadata: dict[str, list[Any]] = {}
        self._transferred_paths: list[str] = []  # Paths transferred to parent (for subagents)

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
    def finish_tool(self) -> Tool:
        """The finish tool used to signal task completion."""
        return self._finish_tool

    @property
    def logger(self) -> AgentLoggerBase:
        """The logger instance used by this agent."""
        return self._logger

    def session(
        self,
        output_dir: Path | str | None = None,
        input_files: str | Path | list[str | Path] | None = None,
        skills_dir: Path | str | None = None,
    ) -> Self:
        """Configure a session and return self for use as async context manager.

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

        Returns:
            Self, for use with `async with agent.session(...) as session:`

        Example:
            async with agent.session(output_dir="./output", input_files="data/*.csv") as session:
                result = await session.run("Analyze the CSV files")

        Note:
            Multiple concurrent sessions from the same Agent instance are supported.
            Each session maintains isolated state via ContextVar.

        """
        self._pending_output_dir = Path(output_dir) if output_dir else None
        self._pending_input_files = input_files
        self._pending_skills_dir = Path(skills_dir) if skills_dir else None
        return self

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

    async def __aenter__(self) -> Self:
        """Enter session context: set up tools, logging, and resources.

        Creates a new SessionState and stores it in the _SESSION_STATE ContextVar,
        allowing concurrent sessions from the same Agent instance.
        """
        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()

        # Get parent state if exists (for subagent file transfer)
        parent_state = _SESSION_STATE.get(None)

        current_depth = _PARENT_DEPTH.get()

        # Create session state and store in ContextVar
        state = SessionState(
            exit_stack=exit_stack,
            output_dir=str(self._pending_output_dir) if self._pending_output_dir else None,
            parent_exec_env=parent_state.exec_env if parent_state else None,
            depth=current_depth,
        )
        _SESSION_STATE.set(state)

        try:
            # === TWO-PASS TOOL INITIALIZATION ===
            # First pass initializes CodeExecToolProvider so that dependent tools
            # (like ViewImageToolProvider) can access state.exec_env in second pass.
            active_tools: list[Tool] = []

            # First pass: Initialize CodeExecToolProvider (at most one allowed)
            code_exec_providers = [t for t in self._tools if isinstance(t, CodeExecToolProvider)]
            if len(code_exec_providers) > 1:
                raise ValueError(
                    f"Agent can only have one CodeExecToolProvider, found {len(code_exec_providers)}: "
                    f"{[type(p).__name__ for p in code_exec_providers]}"
                )

            if code_exec_providers:
                provider = code_exec_providers[0]
                result = await exit_stack.enter_async_context(provider)
                if isinstance(result, list):
                    active_tools.extend(result)
                else:
                    active_tools.append(result)
                state.exec_env = provider

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

            # Build active tools dict with finish tool (stored on instance, not session)
            self._active_tools = {FINISH_TOOL_NAME: self._finish_tool}
            self._active_tools.update({t.name: t for t in active_tools})

            # Validate subagent code exec requirements (only at root level)
            if current_depth == 0:
                self._validate_subagent_code_exec_requirements()

            # Upload input files to exec_env if specified
            if self._pending_input_files:
                if not state.exec_env:
                    raise ValueError("input_files specified but no CodeExecToolProvider configured")

                logger.debug(
                    "[%s __aenter__] Uploading input files: %s, depth=%d, parent_exec_env=%s, parent_exec_env._temp_dir=%s",
                    self._name,
                    self._pending_input_files,
                    state.depth,
                    type(state.parent_exec_env).__name__ if state.parent_exec_env else None,
                    getattr(state.parent_exec_env, "_temp_dir", "N/A") if state.parent_exec_env else None,
                )

                if state.depth > 0 and state.parent_exec_env:
                    # SUBAGENT: Read files from parent's exec env, write to subagent's exec env
                    # input_files are paths within the parent's environment
                    result = await state.exec_env.upload_files(
                        *self._pending_input_files,
                        source_env=state.parent_exec_env,
                    )
                else:
                    # ROOT AGENT: Read files from local filesystem
                    resolved = self._resolve_input_files(self._pending_input_files)
                    result = await state.exec_env.upload_files(*resolved)

                logger.debug(
                    "[%s __aenter__] Upload result: uploaded=%s, failed=%s", self._name, result.uploaded, result.failed
                )

                # Store uploaded paths for system prompt
                state.uploaded_file_paths = [uf.dest_path for uf in result.uploaded]

                if result.failed:
                    raise RuntimeError(f"Failed to upload files: {result.failed}")
            self._pending_input_files = None  # Clear pending state

            # Upload skills directory if it exists and load metadata
            if self._pending_skills_dir:
                skills_path = self._pending_skills_dir
                if skills_path.exists() and skills_path.is_dir():
                    if state.exec_env:
                        logger.debug("[%s __aenter__] Uploading skills directory: %s", self._name, skills_path)
                        await state.exec_env.upload_files(skills_path, dest_dir="skills")
                    # Load skills metadata (even if no exec_env, for system prompt)
                    state.skills_metadata = load_skills_metadata(skills_path)
                    logger.debug("[%s __aenter__] Loaded %d skills", self._name, len(state.skills_metadata))
                self._pending_skills_dir = None  # Clear pending state

            # Configure and enter logger context
            self._logger.name = self._name
            self._logger.model = self._client.model_slug
            self._logger.max_turns = self._max_turns
            # depth is already set (0 for main agent, passed in for sub-agents)
            self._logger.__enter__()

            return self

        except Exception:
            await exit_stack.__aexit__(None, None, None)
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
        state = _SESSION_STATE.get()

        try:
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
                    else:
                        # SUBAGENT: Transfer to parent's exec env
                        if state.parent_exec_env:
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
        finally:
            # Exit logger context
            self._logger.finish_params = self._last_finish_params
            self._logger.run_metadata = self._last_run_metadata
            self._logger.output_dir = str(state.output_dir) if state.output_dir else None
            self._logger.__exit__(exc_type, exc_val, exc_tb)

            # Cleanup all async resources
            await state.exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def run_tool(self, tool_call: ToolCall, run_metadata: dict[str, list[Any]]) -> ToolMessage:
        """Execute a single tool call with error handling for invalid JSON/arguments.

        Returns a ToolMessage containing either the tool output or an error description.
        Metadata from the tool result is stored in the provided run_metadata dict.
        """
        tool = self._active_tools.get(tool_call.name)
        result: ToolResult
        args_valid = True

        # Ensure tool is tracked in metadata dict (even if no metadata returned)
        if tool_call.name not in run_metadata:
            run_metadata[tool_call.name] = []

        if tool:
            try:
                # Parse parameters if the tool has them, otherwise use None
                params = (
                    tool.parameters.model_validate_json(tool_call.arguments) if tool.parameters is not None else None
                )

                # Set parent depth for sub-agent tools to read
                prev_depth = _PARENT_DEPTH.set(self._logger.depth)
                try:
                    if inspect.iscoroutinefunction(tool.executor):
                        result = await tool.executor(params)  # ty: ignore[invalid-await]
                    elif self._run_sync_in_thread:
                        # ty: ignore - type checker doesn't understand iscoroutinefunction narrowing
                        result = await anyio.to_thread.run_sync(tool.executor, params)  # ty: ignore[unresolved-attribute]
                    else:
                        # ty: ignore - iscoroutinefunction check above ensures this is sync
                        result = tool.executor(params)  # ty: ignore[invalid-assignment]
                finally:
                    _PARENT_DEPTH.reset(prev_depth)

                # Store metadata if present
                if result.metadata is not None:
                    run_metadata[tool_call.name].append(result.metadata)
            except ValidationError:
                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )
                result = ToolResult(content="Tool arguments are not valid")
                args_valid = False
        else:
            LOGGER.debug(f"LLMClient tried to use the tool {tool_call.name} which is not in the tools list")
            result = ToolResult(content=f"{tool_call.name} is not a valid tool")

        return ToolMessage(
            content=result.content,
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            args_was_valid=args_valid,
        )

    async def step(
        self,
        messages: list[ChatMessage],
        run_metadata: dict[str, list[Any]],
        turn: int = 0,
        max_turns: int = 0,
    ) -> tuple[AssistantMessage, list[ToolMessage], ToolCall | None]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            messages: Current conversation messages
            run_metadata: Metadata storage for tool results
            turn: Current turn number (1-indexed) for logging
            max_turns: Maximum turns for logging

        Returns the assistant message, tool execution results, and finish tool call (if present).

        """
        assistant_message = await self._client.generate(messages, self._active_tools)

        # Log assistant message immediately
        if turn > 0:
            self._logger.assistant_message(turn, max_turns, assistant_message)

        tool_messages: list[ToolMessage] = []
        finish_call: ToolCall | None = None

        if assistant_message.tool_calls:
            finish_call = next(
                (tc for tc in assistant_message.tool_calls if tc.name == FINISH_TOOL_NAME),
                None,
            )

            tool_messages = []
            for tool_call in assistant_message.tool_calls:
                tool_message = await self.run_tool(tool_call, run_metadata)
                tool_messages.append(tool_message)

                # Log tool result immediately
                self._logger.tool_result(tool_message)

        return assistant_message, tool_messages, finish_call

    async def summarize_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Condense message history using LLM to stay within context window."""
        task_context: list[ChatMessage] = list(takewhile(lambda m: not isinstance(m, AssistantMessage), messages))

        summary_prompt = [*messages, UserMessage(content=MESSAGE_SUMMARIZER)]

        # We need to pass the tools to the client so that it has context of tools used in the conversation
        summary = await self._client.generate(summary_prompt, self._active_tools)

        summary_bridge_prompt = MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary.content)
        summary_bridge = UserMessage(content=summary_bridge_prompt)
        acknowledgement_msg = UserMessage(content="Got it, thanks!")

        # Log the completed summary
        summary_content = summary.content if isinstance(summary.content, str) else str(summary.content)
        self._logger.context_summarization_complete(summary_content, summary_bridge_prompt)

        return [*task_context, summary_bridge, acknowledgement_msg]

    async def run(
        self,
        init_msgs: str | list[ChatMessage],
        *,
        depth: int | None = None,
    ) -> tuple[FinishParams | None, list[list[ChatMessage]], dict[str, list[Any]]]:
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
        msgs: list[ChatMessage] = []

        # Build the complete system prompt (base + input files + user instructions)
        full_system_prompt = self._build_system_prompt()
        msgs.append(SystemMessage(content=full_system_prompt))

        if isinstance(init_msgs, str):
            msgs.append(UserMessage(content=init_msgs))
        else:
            msgs.extend(init_msgs)

        # Set logger depth if provided (for sub-agent runs)
        if depth is not None:
            self._logger.depth = depth

        # Log the task at run start
        self._logger.task_message(msgs[-1].content)

        # Show warnings (top-level only, if logger supports it)
        if self._logger.depth == 0 and isinstance(self._logger, AgentLogger):
            run_warnings = self._collect_warnings()
            if run_warnings:
                self._logger.warnings_message(run_warnings)

        # Use logger callback if available and not overridden
        step_callback = self._logger.on_step

        # Local metadata storage - isolated per run() invocation for thread safety
        run_metadata: dict[str, list[Any]] = {}

        full_msg_history: list[list[ChatMessage]] = []
        finish_params: FinishParams | None = None

        # Cumulative stats for spinner
        total_tool_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i in range(self._max_turns):
            if self._max_turns - i <= 30 and i != 0:
                num_turns_remaining_msg = _num_turns_remaining_msg(self._max_turns - i)
                msgs.append(num_turns_remaining_msg)
                self._logger.user_message(num_turns_remaining_msg)

            # Pass turn info to step() for real-time logging
            assistant_message, tool_messages, finish_call = await self.step(
                msgs,
                run_metadata,
                turn=i + 1,
                max_turns=self._max_turns,
            )

            # Update cumulative stats
            total_tool_calls += len(tool_messages)
            total_input_tokens += assistant_message.token_usage.input
            total_output_tokens += assistant_message.token_usage.output

            # Call progress callback after step completes
            if step_callback:
                step_callback(i + 1, total_tool_calls, total_input_tokens, total_output_tokens)

            user_messages: list[UserMessage] = []
            if self._text_only_tool_responses:
                tool_messages, user_messages = _handle_text_only_tool_responses(tool_messages)

            # Log user messages (e.g., image content extracted from tool responses)
            for user_msg in user_messages:
                self._logger.user_message(user_msg)

            msgs.extend([assistant_message, *tool_messages, *user_messages])

            if finish_call:
                try:
                    finish_arguments = json.loads(finish_call.arguments)
                    if self._finish_tool.parameters is not None:
                        finish_params = self._finish_tool.parameters.model_validate(finish_arguments)
                    break
                except (json.JSONDecodeError, ValidationError, TypeError):
                    LOGGER.debug(
                        "Agent tried to use the finish tool but the tool call is not valid: %r",
                        finish_call.arguments,
                    )
                    # continue until the finish tool call is valid

            pct_context_used = assistant_message.token_usage.total / self._client.max_tokens
            if pct_context_used >= self._context_summarization_cutoff and i + 1 != self._max_turns:
                self._logger.context_summarization_start(pct_context_used, self._context_summarization_cutoff)
                full_msg_history.append(msgs)
                msgs = await self.summarize_messages(msgs)
        else:
            LOGGER.error(
                f"Maximum number of turns reached: {self._max_turns}. The agent was not able to finish the task. Consider increasing the max_turns parameter.",
            )

        full_msg_history.append(msgs)

        # Add agent's own token usage to run_metadata under "token_usage" key
        agent_token_usage = _get_total_token_usage(full_msg_history)
        if "token_usage" not in run_metadata:
            run_metadata["token_usage"] = []
        run_metadata["token_usage"].append(agent_token_usage)

        # Store for __aexit__ to access (on instance for this agent)
        self._last_finish_params = finish_params
        self._last_run_metadata = run_metadata

        return finish_params, full_msg_history, run_metadata

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

        async def sub_agent_executor(params: SubAgentParams) -> ToolResult[SubAgentMetadata]:
            """Execute the sub-agent with the given task.

            Sub-agents enter their own full session to ensure:
            1. Tool isolation - each agent only sees its own tools (fixes recursive sub-agent bug)
            2. Proper ToolProvider lifecycle - sub-agent's ToolProviders are initialized
            3. Correct logging - logger context is entered for proper output formatting
            """
            # Get parent's depth and calculate subagent depth
            parent_depth = _PARENT_DEPTH.get()
            sub_agent_depth = parent_depth + 1

            # Save parent's session state so we can restore it after subagent completes
            # This ensures sibling subagents see the parent's state, not a previous sibling's stale state
            parent_session_state = _SESSION_STATE.get(None)
            logger.debug(
                "[%s] PRE-SESSION: _SESSION_STATE=%s, exec_env=%s, exec_env._temp_dir=%s",
                agent.name,
                id(parent_session_state) if parent_session_state else None,
                type(parent_session_state.exec_env).__name__
                if parent_session_state and parent_session_state.exec_env
                else None,
                getattr(parent_session_state.exec_env, "_temp_dir", "N/A")
                if parent_session_state and parent_session_state.exec_env
                else None,
            )

            # Set _PARENT_DEPTH to subagent's depth BEFORE entering session
            # so that __aenter__ reads the correct depth for SessionState.depth
            prev_depth = _PARENT_DEPTH.set(sub_agent_depth)
            try:
                init_msgs: list[ChatMessage] = []
                if system_prompt:
                    init_msgs.append(SystemMessage(content=system_prompt))
                init_msgs.append(UserMessage(content=params.task))

                # Sub-agent enters its own full session for tool isolation and proper lifecycle
                # output_dir is a path within the parent's exec env (not local filesystem)
                # Files are transferred to parent's env at __aexit__ via save_output_files(dest_env=parent)
                async with agent.session(
                    output_dir=".",  # Path in parent's exec env
                    input_files=list(params.input_files) if params.input_files else None,  # ty: ignore[invalid-argument-type]
                ) as agent_session:
                    # Override logger depth for proper indentation in console output
                    agent_session._logger.depth = sub_agent_depth  # noqa: SLF001

                    finish_params, msg_history, run_metadata = await agent_session.run(init_msgs)

                    # Extract the last assistant message with actual content (not just tool calls)
                    last_assistant_msg: AssistantMessage | None = None
                    for msg_group in reversed(msg_history):
                        for msg in reversed(msg_group):
                            if isinstance(msg, AssistantMessage) and msg.content:
                                last_assistant_msg = msg
                                break
                        if last_assistant_msg:
                            break

                    # Build content from the assistant message and/or finish params
                    content_parts: list[str] = []

                    if last_assistant_msg and last_assistant_msg.content:
                        content = last_assistant_msg.content
                        if isinstance(content, list):
                            content = "\n".join(str(block) for block in content)
                        content_parts.append(content)

                    # Include finish params if available (they often contain the actual result)
                    if finish_params is not None:
                        finish_dict = finish_params.model_dump()
                        if finish_dict:
                            content_parts.append(f"Finish params: {finish_dict}")

                    # Report files transferred to parent's exec env (set in __aexit__)
                    transferred_paths = agent_session._transferred_paths  # noqa: SLF001
                    if transferred_paths:
                        content_parts.append(f"Files available in your environment: {transferred_paths}")

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

                    # Create subagent metadata with token usage, message history, and run metadata
                    sub_metadata = SubAgentMetadata(
                        message_history=msg_history,
                        run_metadata=run_metadata,
                    )

                    return ToolResult(content=result_content, metadata=sub_metadata)

            except Exception as e:
                # On error, return empty metadata
                error_metadata = SubAgentMetadata(
                    message_history=[],
                    run_metadata={},
                )
                return ToolResult(
                    content=f"<sub_agent_result>\n<error>{e!s}</error>\n</sub_agent_result>",
                    metadata=error_metadata,
                )
            finally:
                # DEBUG: Log SESSION_STATE after subagent session
                post_session_state = _SESSION_STATE.get(None)
                logger.debug(
                    "[%s] POST-SESSION: _SESSION_STATE=%s, exec_env=%s, exec_env._temp_dir=%s",
                    agent.name,
                    id(post_session_state) if post_session_state else None,
                    type(post_session_state.exec_env).__name__
                    if post_session_state and post_session_state.exec_env
                    else None,
                    getattr(post_session_state.exec_env, "_temp_dir", "N/A")
                    if post_session_state and post_session_state.exec_env
                    else None,
                )

                # Restore parent's depth
                _PARENT_DEPTH.reset(prev_depth)
                # Restore parent's session state so next sibling subagent sees it
                if parent_session_state is not None:
                    _SESSION_STATE.set(parent_session_state)

        return Tool[SubAgentParams, SubAgentMetadata](
            name=self._name,
            description=description,
            parameters=SubAgentParams,
            executor=sub_agent_executor,  # ty: ignore[invalid-argument-type]
        )
